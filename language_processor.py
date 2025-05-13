import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.grammar import CFG, Production
from nltk.parse.generate import generate
import logging
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from difflib import get_close_matches
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote_plus

logging.basicConfig(level=logging.INFO)

class LanguageProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)

        # Initialize sentiment analyzer
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Track conversation history for learning
        self.conversation_memory = {}
        self.learned_patterns = set()

        # Response variation settings
        self.variation_level = 0.7  # Controls how much variation to add
        self.learned_phrases = set()  # Store unique phrases learned from conversations

        # Initialize tokenizer for cleaning text
        self.tokenizer = RegexpTokenizer(r'\w+')

        # Conversation context tracking
        self.context = {
            'topics': [],
            'sentiment': 'neutral',
            'user_preferences': {},
            'conversation_history': []
        }

        # Enhanced grammar rules for more natural responses
        self.grammar = CFG.fromstring("""
            S -> NP VP | NP VP PP | INTJ NP VP
            NP -> DET N | DET ADJ N | PRO | NAME
            VP -> V | V NP | AUX V | AUX V NP | ADV V
            PP -> P NP
            INTJ -> 'wow' | 'oh' | 'hey' | 'hmm'
            DET -> 'the' | 'a' | 'an' | 'this' | 'that' | 'my' | 'your'
            N -> 'situation' | 'topic' | 'idea' | 'thought' | 'concept' | 'point'
            V -> 'understand' | 'think' | 'believe' | 'see' | 'feel' | 'appreciate'
            ADJ -> 'interesting' | 'important' | 'complex' | 'clear' | 'fascinating'
            PRO -> 'I' | 'you' | 'we' | 'they' | 'it'
            AUX -> 'can' | 'will' | 'should' | 'might' | 'must'
            P -> 'about' | 'with' | 'in' | 'through' | 'during'
            ADV -> 'really' | 'definitely' | 'certainly' | 'probably'
            NAME -> 'AI' | 'Assistant'
        """)

        # Response templates for different conversation scenarios
        self.response_templates = {
            'agreement': [
                "I completely agree with your perspective on {topic}.",
                "That's exactly what I was thinking about {topic}.",
                "You make an excellent point about {topic}."
            ],
            'clarification': [
                "Could you elaborate more on {topic}?",
                "I'm interested in understanding more about your thoughts on {topic}.",
                "What specific aspects of {topic} would you like to explore?"
            ],
            'empathy': [
                "I understand how you feel about {topic}.",
                "Your perspective on {topic} is very insightful.",
                "I can see why you would feel that way about {topic}."
            ],
            'transition': [
                "Speaking of {topic}, I'd love to hear your thoughts on {new_topic}.",
                "That reminds me of something interesting about {new_topic}.",
                "Your point about {topic} connects well with {new_topic}."
            ]
        }

    def analyze_sentiment(self, text):
        """Analyze the sentiment of input text"""
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        return 'neutral'

    def update_context(self, user_input):
        """Update conversation context based on user input"""
        # Extract topics
        tokens = word_tokenize(user_input.lower())
        tags = pos_tag(tokens)
        nouns = [word for word, tag in tags if tag.startswith('NN')]

        # Update topic history
        if nouns:
            self.context['topics'].extend(nouns)
            self.context['topics'] = self.context['topics'][-5:]  # Keep last 5 topics

        # Update sentiment
        self.context['sentiment'] = self.analyze_sentiment(user_input)

        # Add to conversation history
        self.context['conversation_history'].append(user_input)
        if len(self.context['conversation_history']) > 10:
            self.context['conversation_history'].pop(0)

    def generate_contextual_response(self, user_input):
        """Generate a context-aware response"""
        self.update_context(user_input)

        # Select response type based on context
        if '?' in user_input:
            template_type = 'clarification'
        elif self.context['sentiment'] == 'positive':
            template_type = 'agreement'
        elif self.context['sentiment'] == 'negative':
            template_type = 'empathy'
        else:
            template_type = 'transition'

        # Get current topic
        current_topic = self.context['topics'][-1] if self.context['topics'] else "this topic"

        # Generate response
        templates = self.response_templates[template_type]
        response = random.choice(templates).format(
            topic=current_topic,
            new_topic=random.choice(self.context['topics'][:-1]) if len(self.context['topics']) > 1 else "related topics"
        )

        return self.enhance_naturalness(response)

    def enhance_naturalness(self, response):
        """Add natural language variations and personality"""
        if not response:
            return response

        # Natural conversation starters
        starters = [
            "You know what, ",
            "I think ",
            "Well, ",
            "Actually, ",
            "Honestly, ",
            "From what I understand, "
        ]

        # Natural conversation endings - only add if no question mark exists
        endings = [
            " What are your thoughts on that?",
            " Does that make sense?",
            " Let me know if you want to explore this further!",
            " I'd love to hear your perspective.",
            " What do you think?"
        ]

        # Add starter only if response doesn't already have one
        if not any(s.lower() in response.lower() for s in starters):
            if random.random() < 0.3:
                response = random.choice(starters) + response.lower()

        # Add ending only if response doesn't already have a question
        if not any(e.lower() in response.lower() for e in endings) and '?' not in response:
            if random.random() < 0.2:
                response += random.choice(endings)

        return response

    def find_synonyms(self, word):
        """Find synonyms for a given word using WordNet"""
        synonyms = []
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    if lemma.name() != word:
                        synonyms.append(lemma.name())
            return list(set(synonyms))  # Remove duplicates
        except:
            return []

    def web_search(self, query):
        """Enhanced web search with continuous learning and multiple sources"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # Expanded list of knowledge sources
            search_sites = [
                f"https://html.duckduckgo.com/html/?q=site:wikipedia.org {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q=site:reddit.com {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q=site:stackoverflow.com {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q=site:quora.com {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q=site:medium.com {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q=site:arxiv.org {quote_plus(query)}",
                f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
            ]

            # Generate related queries for background research
            related_terms = query.lower().split()
            related_queries = []
            for term in related_terms:
                if len(term) > 3:  # Only meaningful terms
                    related_queries.extend([
                        f"what is {term}",
                        f"how does {term} work",
                        f"latest developments in {term}",
                        f"examples of {term}"
                    ])

            all_results = []
            learned_patterns = set()

            for search_url in search_sites:
                try:
                    response = requests.get(search_url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Extract search results
                        for result in soup.select('.result__body'):
                            title = result.select_one('.result__title')
                            snippet = result.select_one('.result__snippet')
                            if title and snippet:
                                text = snippet.get_text(strip=True)
                                all_results.append(text)

                                # Learn from patterns in responses
                                sentences = text.split('.')
                                for sentence in sentences:
                                    words = sentence.strip().split()
                                    if len(words) > 3:  # Only learn from meaningful phrases
                                        learned_patterns.add(' '.join(words[:3]))  # Learn beginning patterns

                except requests.exceptions.RequestException:
                    continue

            # Add learned patterns to phrases
            if learned_patterns:
                self.learned_phrases.update(learned_patterns)

            if all_results:
                # Process and synthesize results
                all_sentences = []
                for result in all_results:
                    sentences = [s.strip() for s in result.split('.') if len(s.strip()) > 20]
                    all_sentences.extend(sentences)

                # Group related information
                topic_groups = {}
                for sentence in all_sentences:
                    words = set(sentence.lower().split())
                    for key_term in query.lower().split():
                        if key_term in words:
                            if key_term not in topic_groups:
                                topic_groups[key_term] = []
                            topic_groups[key_term].append(sentence)

                # Synthesize comprehensive response
                response_parts = []
                for term, sentences in topic_groups.items():
                    if sentences:
                        response_parts.append(f"Regarding {term}: {sentences[0]}")

                if response_parts:
                    return ' '.join(response_parts)

                # Background learning from related queries
                for related_query in related_queries:
                    try:
                        self.learn_from_conversation(related_query, self.web_search(related_query))
                    except:
                        continue

            return "I couldn't find specific information about that."

        except Exception as e:
            error_msg = f"Search error: {e}"
            logging.error(error_msg)
            return f"I encountered an error while searching: {str(e)}. Please try again or rephrase your question."

    def learn_from_conversation(self, user_input, response):
        """Learn from conversation interactions"""
        # Extract key phrases
        words = word_tokenize(user_input.lower())
        if len(words) > 2:
            for i in range(len(words)-2):
                phrase = " ".join(words[i:i+3])
                self.learned_phrases.add(phrase)

        # Update conversation memory
        if user_input not in self.conversation_memory:
            self.conversation_memory[user_input] = set()
        self.conversation_memory[user_input].add(response)

        # Learn patterns
        if len(user_input.split()) > 2 and len(response.split()) > 2:
            pattern = (user_input.split()[0], response.split()[0])
            self.learned_patterns.add(pattern)

    def add_variation(self, response):
        """Add natural variation to responses"""
        if random.random() > self.variation_level:
            return response

        variations = {
            "I think": ["In my view", "From my perspective", "It seems to me", "I believe"],
            "Yes": ["Indeed", "Absolutely", "Definitely", "Certainly"],
            "No": ["Not quite", "I don't think so", "That's not correct"],
            "Maybe": ["Perhaps", "Possibly", "It's possible", "It could be"],
            "Interesting": ["Fascinating", "Intriguing", "That's noteworthy", "How remarkable"]
        }

        words = response.split()
        for i, word in enumerate(words):
            key = word.lower()
            if key in variations and random.random() < self.variation_level:
                words[i] = random.choice(variations[key])

        # Sometimes add learned phrases
        if self.learned_phrases and random.random() < 0.3:
            words.append(random.choice(list(self.learned_phrases)))

        return " ".join(words)

    def get_word_variations(self, word):
        """Get variations of a word including synonyms and related forms"""
        variations = set()

        # Add synonyms
        synonyms = self.find_synonyms(word)
        if synonyms:
            variations.update(synonyms)

        # Add basic word forms (simple plurals/singulars)
        if word.endswith('s'):
            variations.add(word[:-1])  # Remove 's'
        else:
            variations.add(word + 's')  # Add 's'

        # Add capitalized version
        variations.add(word.capitalize())

        return list(variations)

    def correct_spelling(self, text):
        """Basic spelling correction"""
        # Common word dictionary
        common_words = set(["hello", "hi", "hey", "how", "what", "where", "when", "why", "who",
                          "is", "are", "was", "were", "will", "would", "could", "should", "can",
                          "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"])

        words = text.lower().split()
        corrected_words = []

        for word in words:
            # Skip short words and numbers
            if len(word) <= 2 or word.isdigit() or word in common_words:
                corrected_words.append(word)
                continue

            # Try to find close matches
            matches = get_close_matches(word, common_words, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0])
            else:
                corrected_words.append(word)

        return " ".join(corrected_words)

    def generate_response(self, query):
        """Generate a natural, context-aware response"""
        # Correct spelling in query
        corrected_query = self.correct_spelling(query)
        query_lower = corrected_query.lower().strip()

        # Handle casual greetings more naturally
        greeting_responses = {
            "hi": ["Hey there! What's up?", "Hi! How's it going?", "Hello! Nice to chat with you!"],
            "hello": ["Hey! How are you doing today?", "Hello! What's new?", "Hi there! How's your day going?"],
            "hey": ["Hey! What's on your mind?", "Hi! What can I help you with?", "Hey there! How's everything?"],
            "how are you": ["I'm doing great, thanks for asking! How about you?", "Pretty good! What's new with you?", "I'm good! How's your day going?"]
        }

        # Check for greetings first
        for greeting, responses in greeting_responses.items():
            if greeting in query_lower:
                return random.choice(responses)

        # Try to get information from web search for questions
        if any(q in query_lower for q in ['what', 'how', 'why', 'when', 'where', 'who']) or '?' in query:
            try:
                search_result = self.web_search(query_lower)
                if search_result and search_result != "I couldn't find specific information about that.":
                    # Format the search result into a natural response
                    intro_phrases = [
                        "Based on what I found, ",
                        "According to my research, ",
                        "Here's what I learned: ",
                        "Let me share what I found: ",
                        "From what I understand, "
                    ]
                    return random.choice(intro_phrases) + search_result
            except Exception as e:
                print(f"Search error: {e}")

        # Extract key terms for better context
        words = [w for w in query_lower.split() if len(w) > 3]
        key_terms = ' '.join(words[:3]) if words else query_lower

        # Try another web search with key terms
        try:
            search_result = self.web_search(key_terms)
            if search_result and search_result != "I couldn't find specific information about that.":
                return f"Let me share some relevant information about {key_terms}: {search_result}"
        except Exception as e:
            print(f"Search error: {e}")

        # Fallback responses if web search fails
        context_responses = [
            f"I understand you're interested in {key_terms}. Could you be more specific about what you'd like to know?",
            f"That's an interesting topic about {key_terms}. What specific aspects would you like to explore?",
            f"I'd be happy to discuss {key_terms} in more detail. What would you like to focus on?",
            f"Let's explore {key_terms} together. What particular aspects interest you most?"
        ]

        return random.choice(context_responses)
