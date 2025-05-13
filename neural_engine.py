import numpy as np
import os
import pickle
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import nltk
nltk.download('averaged_perceptron_tagger', quiet=True)

class NeuralEngine:
    def __init__(self, memory_file="brain_model.pkl", vocab_size=1000, max_sequence_length=20):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.memory_file = memory_file
        self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
        self.label_encoder = LabelEncoder()
        self.trained = False
        self.model = None
        self.conversation_data = {"queries": [], "responses": [], "contexts": []}
        self.experience_buffer = []
        self.self_reflection_log = []

        # Load or create model
        if os.path.exists(memory_file):
            self.load_model()
        else:
            self.create_model()

        # Personality system
        self.personality_weights = {
            "friendly": {
                "warmth": 0.8,
                "enthusiasm": 0.7,
                "formality": 0.3
            },
            "formal": {
                "warmth": 0.3,
                "enthusiasm": 0.4,
                "formality": 0.9
            },
            "casual": {
                "warmth": 0.6,
                "enthusiasm": 0.8,
                "formality": 0.2
            },
            "technical": {
                "warmth": 0.4,
                "enthusiasm": 0.5,
                "formality": 0.7
            }
        }

    def add_personality(self, response, emotional_state=None):
        """Add natural personality and conversational elements"""
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

        # Natural conversation endings
        endings = [
            " What are your thoughts on that?",
            " Does that make sense?",
            " Let me know if you want to explore this further!",
            " I'd love to hear your perspective.",
            " What do you think?"
        ]

        # Add natural elements
        if random.random() < 0.3:
            response = random.choice(starters) + response.lower()
        if random.random() < 0.2 and not response.endswith('?'):
            response += random.choice(endings)

        # Make contractions more natural
        contractions = {
            "I am": "I'm",
            "you are": "you're",
            "that is": "that's",
            "what is": "what's",
            "how is": "how's",
            "could not": "couldn't",
            "would not": "wouldn't",
            "should not": "shouldn't",
            "cannot": "can't",
            "will not": "won't"
        }
        
        for full, contraction in contractions.items():
            response = response.replace(full, contraction)

        return response

        # Adjust based on emotional context
        warmth = (warmth + emotional_state.get('friendly', 0)) / 2
        enthusiasm = (enthusiasm + emotional_state.get('enthusiastic', 0)) / 2

        # Add personality markers
        if warmth > 0.6:
            response = random.choice(["You know what? ", "I think ", "I feel like "]) + response

        if enthusiasm > 0.6:
            response = response.replace(".", "!")
            response += random.choice([" What do you think?", " Isn't that interesting?", " I'd love to hear your thoughts!"])

        if formality < 0.4:
            response = response.replace("Hello", "Hey").replace("Greetings", "Hi")
            response = response.replace("would you", "wanna").replace("going to", "gonna")

        return response

        # Initialize or load model
        if os.path.exists(memory_file):
            self.load_model()
        else:
            self.create_model()

        # Memory for experiences
        self.experience_buffer = []
        self.self_reflection_log = []

    def create_model(self):
        """Create a new neural network model"""
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            max_iter=200,
            random_state=42
        )

        # Empty initial dataset
        self.conversation_data = {"queries": [], "responses": [], "contexts": []}
        self.trained = False

    def load_model(self):
        """Load a previously saved model and data"""
        try:
            with open(self.memory_file, 'rb') as f:
                saved_data = pickle.load(f)

            self.model = saved_data.get('model')
            self.conversation_data = saved_data.get('conversation_data', {"queries": [], "responses": [], "contexts": []})
            self.vectorizer = saved_data.get('vectorizer', TfidfVectorizer(max_features=self.vocab_size))
            self.label_encoder = saved_data.get('label_encoder', LabelEncoder())
            self.trained = saved_data.get('trained', False)
            self.self_reflection_log = saved_data.get('self_reflection_log', [])
        except (FileNotFoundError, EOFError, pickle.PickleError) as e:
            print(f"Error loading model: {e}")
            self.create_model()

    def save_model(self):
        """Save the current model and data"""
        save_data = {
            'model': self.model,
            'conversation_data': self.conversation_data,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder,
            'trained': self.trained,
            'self_reflection_log': self.self_reflection_log
        }

        with open(self.memory_file, 'wb') as f:
            pickle.dump(save_data, f)

    def add_training_example(self, query, response, context=None):
        """Add a new example to the training data"""
        self.conversation_data["queries"].append(query)
        self.conversation_data["responses"].append(response)
        self.conversation_data["contexts"].append(context or "")

        # Add to experience buffer
        self.experience_buffer.append({
            "query": query,
            "response": response,
            "context": context,
            "time": time.time()
        })

        # Periodically train if we have enough examples
        if len(self.conversation_data["queries"]) % 10 == 0:
            self.train_model()
            self.self_reflection()

    def vectorize_text(self, texts):
        """Convert text to numeric vectors"""
        if not self.trained or len(texts) == 0:
            return np.zeros((len(texts), self.vocab_size))

        try:
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit(texts)
            return self.vectorizer.transform(texts).toarray()
        except Exception as e:
            print(f"Vectorization error: {e}")
            return np.zeros((len(texts), self.vocab_size))

    def train_model(self):
        """Train the neural network on accumulated data"""
        if len(self.conversation_data["queries"]) < 5:
            return False

        try:
            # Prepare data
            X = self.vectorize_text(self.conversation_data["queries"])

            # Encode responses as classes
            unique_responses = list(set(self.conversation_data["responses"]))
            self.label_encoder.fit(unique_responses)
            y = self.label_encoder.transform(self.conversation_data["responses"])

            # Train the model
            self.model.fit(X, y)
            self.trained = True
            self.save_model()
            return True
        except Exception as e:
            print(f"Training error: {e}")
            return False

    def generate_response(self, query, recent_conversations=None):
        """Generate a natural, context-aware response"""
        query_lower = query.lower().strip()
        
        # Handle casual greetings more naturally
        greeting_responses = {
            "hi": ["Hey there! What's up?", "Hi! How's it going?", "Hello! Nice to chat with you!"],
            "hello": ["Hey! How are you doing today?", "Hello! What's new?", "Hi there! How's your day going?"],
            "hey": ["Hey! What's on your mind?", "Hi! What can I help you with?", "Hey there! How's everything?"],
            "how are you": ["I'm doing great, thanks for asking! How about you?", "Pretty good! What's new with you?", "I'm good! How's your day going?"]
        }

        # Check for casual greetings
        for greeting, responses in greeting_responses.items():
            if greeting in query_lower:
                return random.choice(responses)

        # Build context from recent conversations
        context_responses = []
        if recent_conversations:
            context_responses = [conv[1] for conv in recent_conversations[-5:]]

        # Extract key concepts from query
        key_concepts = [word for word in query_lower.split() 
                       if len(word) > 3 and word not in ['what', 'when', 'where', 'how', 'why', 'the', 'and', 'but']]

        # Generate response based on query type and context
        is_question = '?' in query
        
        if is_question:
            question_starters = [
                "Based on what I know, ",
                "From my understanding, ",
                "Let me share my thoughts on this. ",
                "Here's what I think: ",
                "I've given this some thought - "
            ]
            
            if any(word in query_lower for word in ['can', 'could', 'would']):
                responses = [
                    "I believe I can help you with that.",
                    "Yes, I'd be happy to assist with this.",
                    "Let me help you with that request.",
                    "I can definitely work on that for you.",
                ]
            elif any(word in query_lower for word in ['what', 'how']):
                responses = [
                    f"The key aspect of {' '.join(key_concepts)} is...",
                    f"When it comes to {' '.join(key_concepts)}, there are several important points.",
                    f"Let me explain about {' '.join(key_concepts)} in detail.",
                    f"Here's what you need to know about {' '.join(key_concepts)}.",
                ]
            elif any(word in query_lower for word in ['why']):
                responses = [
                    f"The main reason involves {' '.join(key_concepts)}.",
                    f"There are several factors related to {' '.join(key_concepts)} that explain this.",
                    f"This happens because of how {' '.join(key_concepts)} work together.",
                    f"The relationship between {' '.join(key_concepts)} explains this.",
                ]
            else:
                responses = [
                    "That's an interesting question. Here's my perspective:",
                    "I've analyzed this, and here's what I found:",
                    "Let me share my insights on this:",
                    "Based on my analysis, I can tell you that:",
                ]
                
            response = random.choice(question_starters) + random.choice(responses)
            
        else:
            # Handle statements
            statement_starters = [
                "I see what you mean about ",
                "That's interesting regarding ",
                "Your point about ",
                "I understand the connection with ",
                "That makes me think about "
            ]
            
            if len(key_concepts) > 0:
                concept_responses = [
                    f"This relates to how {' '.join(key_concepts)} work in practice.",
                    f"The impact of {' '.join(key_concepts)} is quite significant.",
                    f"There's a lot to explore about {' '.join(key_concepts)}.",
                    f"Your perspective on {' '.join(key_concepts)} is intriguing.",
                ]
                response = random.choice(statement_starters) + ' '.join(key_concepts) + ". " + random.choice(concept_responses)
            else:
                response = "That's interesting. Could you tell me more about your thoughts on this?"

        # Add personality and context
        return self.add_personality(response)

        # Handle common interactions naturally
        if any(word in query_lower for word in ['hi', 'hello', 'hey']):
            responses = [
                "Hey! Nice to see you here. How's your day going?",
                "Hi there! I was just thinking about something interesting. What's on your mind?",
                "Hello! I've been having quite a day. What brings you here?",
                "Hey! Always good to chat with you. What's new?"
            ]
            return random.choice(responses)

        # Handle basic questions
        if query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who')):
            try:
                # Use existing memory if available
                for q, responses in self.conversation_data.items():
                    if q.lower() == query_lower and responses:
                        return random.choice(responses)

                # Fallback to generic response
                return "I understand you're asking about " + query_lower[query_lower.find(' ')+1:] + ". Could you please be more specific?"
            except:
                return "Could you please rephrase your question?"

        # Handle statements and commands
        try:
            # Use neural network if trained
            if self.trained and len(self.conversation_data["queries"]) >= 5:
                query_vec = self.vectorize_text([query])
                predicted_class = self.model.predict(query_vec)[0]
                try:
                    return self.label_encoder.inverse_transform([predicted_class])[0]
                except:
                    pass

            # Fallback to context-based response
            return "I understand what you're saying about " + query_lower + ". Would you like to discuss this further?"
        except Exception as e:
            return "I understand. Please tell me more about your thoughts on this."

    def self_reflection(self):
        """Perform self-reflection on recent conversations"""
        if len(self.experience_buffer) < 5:
            return

        # Sample recent experiences
        recent_experiences = self.experience_buffer[-10:]

        # Look for patterns in queries and responses
        query_patterns = {}
        response_patterns = {}

        for exp in recent_experiences:
            query = exp["query"].lower()
            response = exp["response"]

            # Count word frequencies
            for word in query.split():
                if word not in query_patterns:
                    query_patterns[word] = 0
                query_patterns[word] += 1

            # Count response patterns
            if response not in response_patterns:
                response_patterns[response] = 0
            response_patterns[response] += 1

        # Find most common words and responses
        common_words = sorted(query_patterns.items(), key=lambda x: x[1], reverse=True)[:5]
        common_responses = sorted(response_patterns.items(), key=lambda x: x[1], reverse=True)[:3]

        reflection = {
            "timestamp": time.time(),
            "common_query_words": common_words,
            "common_responses": common_responses,
            "unique_responses_ratio": len(response_patterns) / len(recent_experiences) if recent_experiences else 0
        }

        self.self_reflection_log.append(reflection)
        self.save_model()
