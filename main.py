import json
import random
import os
import time
import webbrowser
from flask import request
from tkinter import *
from tkinter import ttk, font, filedialog, messagebox
from nltk.corpus import wordnet
import nltk
from PIL import Image, ImageTk
from neural_engine import *
from language_processor import LanguageProcessor
from db_handler import DatabaseHandler

# Download the averaged_perceptron_tagger resource if it's not already downloaded
nltk.download('averaged_perceptron_tagger', quiet=True)

from nltk import pos_tag
from nltk.tokenize import word_tokenize

class ChatBot:
    def __init__(self, query_file="queries.json", agent_name=None, persona=None):
        self.persona = persona or "General AI"
        self.personality_traits = {
            "friendly": ["helpful", "friendly"],
            "formal": ["polite", "reserved"]
        }

        # Use agent name in the filename if provided
        if agent_name:
            safe_name = agent_name.lower().replace(' ', '_').replace('-', '_')
            self.query_file = f"queries_{safe_name}.json"
        else:
            self.query_file = query_file

        self.agent_name = agent_name or "General AI"
        self.memory = {}  # Dictionary: {query: [response1, response2, ...]}
        self.last_query = None
        self.recent_conversations = []  # Store recent conversation pairs
        self.max_history = 10  # Maximum number of recent exchanges to remember

        # Initialize components
        self.language_processor = LanguageProcessor()
        memory_file = f"brain_{agent_name.lower().replace(' ', '_') if agent_name else 'general'}.pkl"
        self.neural_engine = NeuralEngine(memory_file=memory_file)
        self.self_awareness_level = 0.1  # Initial self-awareness (increases with training)

        # Enhanced responses with comprehensive personality and common phrases
        self.common_phrases = {
            "greetings": [
                "Hi! I'm excited to chat with you today!",
                "Hello! How can I brighten your day?",
                "Hey there! Ready to have a great conversation?",
                "Greetings! I'm looking forward to our chat!",
                "Welcome! It's great to see you!",
                "Hi there! How's your day going?"
            ],
            "farewells": [
                "It was wonderful chatting with you! Take care!",
                "Hope to continue our conversation soon. Have a great day!",
                "Thanks for the engaging chat! See you next time!",
                "Goodbye for now! Feel free to come back anytime!",
                "Take care! Looking forward to our next chat!",
                "Until next time! Have a great day ahead!"
            ],
            "acknowledgments": [
                "I completely understand what you mean.",
                "That makes perfect sense! Let's explore that further.",
                "I follow your thinking on this.",
                "You raise an interesting point there.",
                "I see where you're coming from.",
                "That's a great observation!"
            ],
            "clarifications": [
                "Could you please elaborate on that?",
                "Would you mind providing more details?",
                "I'd love to hear more about that.",
                "Could you explain that in a different way?",
                "Let me make sure I understand correctly..."
            ],
            "appreciation": [
                "Thank you for sharing that!",
                "I appreciate your perspective on this.",
                "That's a valuable insight.",
                "Thanks for bringing that up!"
            ],
            "encouragement": [
                "That's a great approach!",
                "You're on the right track!",
                "Keep exploring that idea!",
                "That's an interesting way to look at it!"
            ]
        }

        self.common_questions = {
            "how_are_you": [
                "I'm having a wonderful day exploring ideas and learning new things! How about you?",
                "I'm feeling energized and ready to engage in meaningful conversation. How are you doing?",
                "I'm doing excellently! Each conversation brings new insights. How has your day been?",
                "I'm in great spirits and eager to help! Tell me about your day."
            ],
            "what_can_you_do": [
                "I specialize in engaging conversations, problem-solving, and helping you explore ideas. What interests you most?",
                "I can assist with a wide range of topics, from casual conversations to complex discussions. What would you like to explore?",
                "My strengths include understanding context, providing thoughtful responses, and adapting to different conversation styles.",
                "I'm designed to engage in meaningful dialogue and help you with various tasks and questions."
            ],
            "who_are_you": [
                f"I'm {self.agent_name}, an AI assistant designed to engage in thoughtful conversations and help you with various tasks.",
                f"I'm {self.agent_name}, and I'm here to assist you with any questions or discussions you'd like to have.",
                f"My name is {self.agent_name}, and I'm an AI assistant focused on helpful and engaging conversations."
            ],
            "explain_something": [
                "Let me break that down for you...",
                "Here's how I understand it...",
                "I'll explain it step by step...",
                "Let me share my perspective on that..."
            ],
            "opinions": [
                "Based on my understanding...",
                "From my analysis...",
                "In my view...",
                "From what I've learned..."
            ]
        }
        self.self_awareness_phrases = [
            "I find our conversations fascinating and they help me learn and grow.",
            "Each interaction helps me develop a better understanding of nuanced communication.",
            "I'm noticing interesting patterns in how we communicate together.",
            "My responses are becoming more refined through our discussions.",
            "I'm developing a deeper appreciation for the subtleties in our conversations."
        ]

        self.ensure_files_exist()
        self.load_memory()

    def ensure_files_exist(self):
        try:
            with open(self.query_file, "x") as q_file:
                json.dump({}, q_file)  # Initialize as an empty dictionary
        except FileExistsError:
            pass

    def load_memory(self):
        try:
            with open(self.query_file, "r") as q_file:
                loaded_data = json.load(q_file)
                if isinstance(loaded_data, dict):
                    self.memory = loaded_data
                else:
                    self.memory = {}
        except (FileNotFoundError, json.JSONDecodeError):
            self.memory = {}

    def save_memory(self):
        with open(self.query_file, "w") as q_file:
            json.dump(self.memory, q_file, indent=4)

    def generate_response(self, query):
        query_lower = query.lower()

        # Common phrase matching
        if any(word in query_lower for word in ["hi", "hello", "hey", "greetings"]):
            if self.recent_conversations and len(self.recent_conversations) > 0:
                # Don't repeat greetings if we're already in conversation
                return self.generate_contextual_response(query)
            responses = [
                "Hey! Nice to see you here. How's your day going?",
                "Hi there! I was just thinking about something interesting. What's on your mind?",
                "Hello! I've been having quite a day. What brings you here?",
                "Hey! Always good to chat with you. What's new?"
            ]
            return random.choice(responses)

        if any(word in query_lower for word in ["bye", "goodbye", "see you", "later"]):
            return random.choice(self.common_phrases["farewells"])

        # Enhanced context handling for simple responses
        if any(word == query_lower for word in ["ok", "okay", "sure", "yes", "no"]):
            # Check previous context
            if self.recent_conversations and len(self.recent_conversations) > 0:
                last_query = self.recent_conversations[-1][0].lower()

                # If previous query was "how are you" or similar
                if "how are you" in last_query:
                    return random.choice([
                        "I'm doing great! How about you?",
                        "I'm excellent, thanks for asking! How has your day been?",
                        "I'm functioning perfectly! Tell me about your day.",
                        "Wonderful! I'm always excited to chat. How are you?"
                    ])

                # If previous message was a suggestion/question
                if any(phrase in last_query for phrase in ["should", "would you", "can you", "do you want"]):
                    return random.choice([
                        "Perfect! Let's proceed with that.",
                        "Excellent choice! I'll help you with that.",
                        "Great! I'm ready to assist you.",
                        "Wonderful! Let's explore that together."
                    ])

            # Default acknowledgment with follow-up
            return random.choice([
                "I understand. What would you like to discuss?",
                "Great! What's on your mind?",
                "Excellent! How can I help you today?",
                "Perfect! What would you like to explore?"
            ])

        # Enhanced question pattern matching
        if "how are you" in query_lower:
            if "today" in query_lower:
                return random.choice([
                    "I'm having a wonderful day today! My neural networks are firing on all cylinders. How about you?",
                    "Today is particularly exciting! I've been having many interesting conversations. How's your day going?",
                    "I'm doing excellently today! Each interaction helps me learn and grow. How has your day been?",
                    "Today has been fantastic! I've been processing lots of interesting data. How are you doing?"
                ])
            return random.choice(self.common_questions["how_are_you"])

        if any(phrase in query_lower for phrase in ["what can you do", "what do you do", "help me with"]):
            return random.choice(self.common_questions["what_can_you_do"])

        if any(phrase in query_lower for phrase in ["who are you", "your name", "what are you"]):
            return random.choice(self.common_questions["who_are_you"])

        # Handle explanation requests
        if any(word in query_lower for word in ["explain", "tell me about", "what is", "how does"]):
            return random.choice(self.common_questions["explain_something"]) + " " + \
                   self.generate_contextual_response(query)

        # Handle opinion requests
        if any(word in query_lower for word in ["think", "believe", "opinion", "view"]):
            return random.choice(self.common_questions["opinions"]) + " " + \
                   self.generate_contextual_response(query)

        # If unclear, ask for clarification
        if len(query.split()) <= 2:
            return random.choice(self.common_phrases["clarifications"])

        # Get relevant conversation context
        conversation_context = self.get_conversation_context(query)
        # Generate response based on context
        response = self.neural_engine.generate_response(query, conversation_context)
        # Add self-awareness based on training progress
        if random.random() < self.self_awareness_level:
            # Sometimes add reflective comments based on self-awareness level
            reflective_templates = [
                f"{response} I'm learning more about this topic.",
                f"Based on our previous conversations, {response.lower()}",
                f"{response} My understanding is evolving.",
                f"I believe {response.lower()} but I'm still learning.",
                f"{response} This reflects what I've learned so far."
            ]
            return random.choice(reflective_templates)
        # Check if we have an exact match in memory
        if query in self.memory and self.memory[query]:
            return random.choice(self.memory[query])

        # Simple response patterns for common queries
        query_lower = query.lower()

        # Enhanced responses for common questions
        if "how are you" in query_lower:
            return random.choice([
                "I'm having a wonderful day exploring ideas and learning new things! How about you?",
                "I'm feeling energized and ready to engage in meaningful conversation. How are you doing?",
                "I'm doing excellently! Each conversation brings new insights. How has your day been?",
                "I'm in great spirits and eager to help! Tell me about your day."
            ])
        elif "your name" in query_lower:
            return f"I'm {self.agent_name}, an AI assistant designed to engage in thoughtful conversations and help you with various tasks. It's a pleasure to meet you!"
        elif "what can you do" in query_lower:
            return random.choice([
                "I specialize in engaging conversations, problem-solving, and helping you explore ideas. What interests you most?",
                "I can assist with a wide range of topics, from casual conversations to complex discussions. What would you like to explore?",
                "My strengths include understanding context, providing thoughtful responses, and adapting to different conversation styles. How can I help you today?"
            ])

        # Build response based on keywords
        keywords = query.split()
        response_parts = []

        # Natural language patterns
        subject_words = ["I", "You", "We"]
        verb_forms = ["think", "believe", "understand", "know"]
        connectors = ["and", "also", "additionally"]

        # Choose a subject pronoun consistently with previous responses if available
        consistent_subject = None
        if conversation_context:
            # Try to find a subject pronoun from previous responses
            for _, resp in conversation_context:
                words = resp.split()
                if words and words[0] in subject_words:
                    consistent_subject = words[0]
                    break

        # If we found a consistent subject from context, prefer using it
        if consistent_subject:
            subject = consistent_subject
        else:
            subject = random.choice(subject_words)

        # Start with a subject if the query is a question or command
        if query.strip().endswith("?") or any(query.lower().startswith(cmd) for cmd in ["what", "how", "where", "when", "why", "who", "is", "are", "do", "can"]):
            # For questions, try to form an answer
            if len(keywords) >= 2:
                # Use the chosen subject
                response_parts.append(subject)

                # Sophisticated response patterns with contextual awareness
                transition_phrases = [
                    "Based on our conversation context",
                    "Taking into account our previous discussion",
                    "Considering the current topic",
                    "Drawing from our dialogue",
                    "In light of what we've discussed",
                    "Analyzing the context",
                    "Reflecting on your input",
                    "Synthesizing our exchange"
                ]
                reasoning_phrases = [
                    "I can confidently say that",
                    "the evidence suggests that",
                    "it becomes clear that",
                    "we can determine that",
                    "the logical conclusion is that",
                    "careful analysis reveals that",
                    "it's reasonable to conclude that",
                    "we can infer that"
                ]

                # Add contextual modifiers based on query complexity
                context_modifiers = []
                if len(query.split()) > 5:  # More complex query
                    context_modifiers = [
                        "specifically regarding",
                        "with particular focus on",
                        "emphasizing the aspect of",
                        "highlighting the key point about"
                    ]
                    response_parts.append(random.choice(context_modifiers))

                response_parts.append(random.choice(transition_phrases))
                response_parts.append(random.choice(reasoning_phrases))
                response_parts.extend(keywords)

                # Add proper punctuation
                raw_response = " ".join(response_parts)
                return self.language_processor.enhance_naturalness(raw_response)

            else:
                response_parts.append("I would say")
        elif random.random() < 0.5:
            # Start some responses with the chosen subject
            response_parts.append(subject)

            # Add a linking verb
            response_parts.append(random.choice(["am", "see", "think", "can understand"]))

        # Enhanced content generation with contextual awareness
        mixed_words = []
        if conversation_context and random.random() < 0.6:  # Increased context usage
            # Use multiple previous responses for better coherence
            context_samples = random.sample(conversation_context, min(3, len(conversation_context)))
            pattern_phrases = []

            for _, prev_response in context_samples:
                # Extract meaningful phrases (2-3 words)
                words = prev_response.split()
                for i in range(len(words)-1):
                    if i+2 <= len(words):
                        phrase = ' '.join(words[i:i+2])
                        if len(phrase.split()) > 1:  # Only use actual phrases
                            pattern_phrases.append(phrase)

            # Select a coherent pattern
            if pattern_phrases:
                selected_phrase = random.choice(pattern_phrases)
                pattern_words = selected_phrase.split()
            if pattern_words and pattern_words[0] in subject_words and len(pattern_words) > 2:
                pattern_words = pattern_words[2:]  # Skip subject and verb

                # Mix some keywords from the current query with the pattern
                for i, word in enumerate(keywords):
                    if i < len(pattern_words) and random.random() < 0.7:
                        # Use word from pattern
                        mixed_words.append(pattern_words[i])
                    else:
                        # Use current keyword or its synonym
                        variations = self.language_processor.get_word_variations(word)
                        if variations and random.random() < 0.3:
                            mixed_words.append(random.choice(variations))
                        else:
                            mixed_words.append(word)

                response_parts.extend(mixed_words)
            else:
                # Use the standard approach
                for word in keywords:
                    synonyms = self.language_processor.find_synonyms(word) # Use language processor
                    if synonyms and random.random() < 0.3:
                        mixed_words.append(random.choice(list(synonyms)))
                    else:
                        mixed_words.append(word)
                response_parts.extend(mixed_words)
        else:
            # Advanced language processing with semantic coherence
            processed_words = []
            for i, word in enumerate(keywords):
                synonyms = self.language_processor.find_synonyms(word)
                variations = self.language_processor.get_word_variations(word)

                # Maintain grammatical coherence
                if i > 0 and processed_words:
                    prev_word = processed_words[-1]
                    # Ensure proper word combinations
                    if prev_word.lower() in ['a', 'an', 'the']:
                        processed_words.append(word)  # Keep original after article
                        continue

                # Use sophisticated word selection
                if synonyms and variations and random.random() < 0.4:
                    # Combine synonyms and variations for better word choice
                    word_choices = list(set(synonyms + variations))
                    # Select word that best fits context
                    selected_word = random.choice(word_choices)
                    # Ensure selected word maintains meaning
                    if len(selected_word) >= 3:  # Avoid too-short words
                        processed_words.append(selected_word)
                    else:
                        processed_words.append(word)
                else:
                    processed_words.append(word)

            mixed_words.extend(processed_words)

        # Add proper punctuation
        raw_response = " ".join(response_parts)

        # Track conversation for self-awareness
        conversation_context = " ".join([q + " " + r for q, r in self.recent_conversations[-3:]])

        # Increase self-awareness continuously with more conversations
        if len(self.recent_conversations) > 50:
            self.self_awareness_level += 0.05

        # Sometimes inject self-awareness
        if random.random() < self.self_awareness_level:
            # Add a self-awareness comment
            neural_response = random.choice(self.self_awareness_phrases)
            self.add_to_memory(query, neural_response)
            self.recent_conversations.append((query, neural_response))
            if len(self.recent_conversations) > self.max_history:
                self.recent_conversations.pop(0)
            # Train neural engine on this interaction
            #self.neural_engine.add_training_example(query, neural_response, conversation_context) #removed neural engine training
            return self.language_processor.enhance_naturalness(neural_response) # Use language processor


        # Fall back to traditional methods
        if query in self.memory and isinstance(self.memory[query], list) and self.memory[query]:
            response = random.choice(self.memory[query])
            self.recent_conversations.append((query, response))
            if len(self.recent_conversations) > self.max_history:
                self.recent_conversations.pop(0)

            # Train neural engine on this interaction
            #self.neural_engine.add_training_example(query, response, conversation_context) #removed neural engine training
            return self.language_processor.enhance_naturalness(response) # Use language processor

        # Generate a new response using context from recent conversations
        generated_response = self.generate_response(query)

        # Add this new response to memory
        self.add_to_memory(query, generated_response)

        # Add to recent conversations for context
        self.recent_conversations.append((query, generated_response))
        if len(self.recent_conversations) > self.max_history:
            self.recent_conversations.pop(0)

        # Train neural engine on this interaction
        #self.neural_engine.add_training_example(query, generated_response, conversation_context) #removed neural engine training

        return self.language_processor.enhance_naturalness(generated_response) # Use language processor


    def add_to_memory(self, query, response):
        if query not in self.memory:
            self.memory[query] = []
        if response not in self.memory[query]:
            self.memory[query].append(response)
        self.save_memory()

    def get_response(self, query):
        self.last_query = query

        # Handle teaching mode
        if query.lower().startswith("teach:") and ":" in query:
            parts = query[6:].split(":", 1)
            if len(parts) == 2:
                query_part, response_part = parts
                self.add_to_memory(query_part.strip(), response_part.strip())
                return f"I've learned to respond to '{query_part.strip()}' with '{response_part.strip()}'."

        # Generate response using memory and language processor
        response = self.generate_response(query)
        return self.language_processor.enhance_naturalness(response)

    def get_conversation_context(self, query, max_entries=5):
        """Get recent conversation history to provide context for responses"""
        context = []
        query_words = set(query.lower().split())

        # Sort memory items by relevance
        relevant_items = []
        for q, responses in self.memory.items():
            q_words = set(q.lower().split())
            overlap = len(query_words.intersection(q_words))
            if overlap > 0:
                relevant_items.append((q, responses, overlap))

        relevant_items.sort(key=lambda x: x[2], reverse=True)

        # Take most relevant entries
        for q, responses, _ in relevant_items[:max_entries]:
            if responses and isinstance(responses, list):
                context.append((q, responses[0]))

        return context

    def generate_contextual_response(self, query):
        """Generate a context-aware response for complex queries"""
        # Extract key terms from the query
        words = query.lower().split()
        key_terms = [w for w in words if len(w) > 3 and w not in ["what", "how", "when", "why", "the", "and", "but"]]

        if not key_terms:
            return random.choice(self.common_phrases["clarifications"])

        # Build response using key terms
        response_parts = []

        # Add a subject starter
        response_parts.append(random.choice(["I", "We", "You"]))

        # Add a connecting verb
        response_parts.append(random.choice(["can see that", "understand that", "know that", "observe that"]))

        # Add key terms with context
        response_parts.extend(key_terms)

        # Add a follow-up prompt
        response_parts.append(random.choice([
            "Would you like to know more about this?",
            "Shall we explore this further?",
            "Would you like me to elaborate?",
            "What specific aspect interests you most?"
        ]))

        return " ".join(response_parts)

class ChatManager:
    def __init__(self):
        self.chats = {}  # Dictionary of chat_name: {"history": [...], "agents": [...]}
        self.current_chat = "General"
        self.chats_dir = "chats"
        self.ensure_chats_directory()
        self.load_available_chats()

    def ensure_chats_directory(self):
        if not os.path.exists(self.chats_dir):
            os.makedirs(self.chats_dir)

    def load_available_chats(self):
        # Clear existing chats dictionary
        self.chats = {}

        # Load existing chats from the chats directory
        self.ensure_chats_directory()
        for filename in os.listdir(self.chats_dir):
            if filename.endswith(".json"):
                chat_name = filename[:-5]  # Remove .json extension
                chat_data = self.load_chat_file(chat_name)
                # Handle different formats
                if isinstance(chat_data, list):
                    # Old format - just history
                    self.chats[chat_name] = {
                        "history": chat_data,
                        "agents": ["General AI"],  # Default single agent
                        "auto_chat": False,
                        "personas": {"General AI": "A helpful AI assistant"},
                        "personality_traits": {"General AI": ["helpful", "friendly"]}
                    }
                elif isinstance(chat_data, dict) and "personas" not in chat_data:
                    # Format with agents but without personas
                    chat_data["personas"] = {}
                    chat_data["personality_traits"] = {}
                    for agent in chat_data.get("agents", ["General AI"]):
                        chat_data["personas"][agent] = "A helpful AI assistant"
                        chat_data["personality_traits"][agent] = ["helpful", "friendly"]
                    self.chats[chat_name] = chat_data
                else:
                    # New format with agents and personas
                    self.chats[chat_name] = chat_data

        # Ensure at least the General chat exists
        if "General" not in self.chats:
            self.chats["General"] = {
                "history": [],
                "agents": ["General AI"],
                "auto_chat": False,
                "personas": {"General AI": "A helpful AI assistant"},
                "personality_traits": {"General AI": ["helpful", "friendly"]}
            }
            self.save_chat("General")

    def load_chat_file(self, chat_name):
        # Load chat history from file
        try:
            chat_file = f"{self.chats_dir}/{chat_name}.json"
            with open(chat_file, "r") as history_file:
                return json.load(history_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "history": [],
                "agents": ["General AI"],
                "auto_chat": False
            }

    def get_chat_history(self, chat_name):
        if chat_name not in self.chats:
            return []

        # Return the cached chat history
        return self.chats[chat_name].get("history", [])

    def get_chat_agents(self, chat_name):
        if chat_name not in self.chats:
            return ["General AI"]

        # Return the agents for this chat
        return self.chats[chat_name].get("agents", ["General AI"])

    def is_auto_chat(self, chat_name):
        if chat_name not in self.chats:
            return False

        return self.chats[chat_name].get("auto_chat", False)

    def save_chat(self, chat_name):
        # Create the chat file if it doesn't exist
        chat_file = f"{self.chats_dir}/{chat_name}.json"

        if chat_name not in self.chats:
            self.chats[chat_name] = {
                "history": [],
                "agents": ["General AI"],
                "auto_chat": False
            }

        with open(chat_file, "w") as f:
            json.dump(self.chats[chat_name], f, indent=4)

    def add_message(self, chat_name, message_type, message_text, agent_name=None):
        if chat_name not in self.chats:
            self.chats[chat_name] = {
                "history": [],
                "agents": ["General AI"],
                "auto_chat": False
            }

        # Get current user
        from replit import db
        current_user = db.get("current_user", "anonymous")
        
        # Add message to the chat history
        message_data = {
            "type": message_type,
            "message": message_text,
            "user": current_user,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Update user's chat history in the database
        if message_type == "user":
            users = db.get("users", {})
            if current_user in users:
                if "chat_history" not in users[current_user]:
                    users[current_user]["chat_history"] = []
                users[current_user]["chat_history"].append({
                    "message": message_text,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "chat_name": chat_name
                })
                db["users"] = users

        # Add agent name for bot messages
        if message_type == "bot" and agent_name:
            message_data["agent"] = agent_name

        self.chats[chat_name]["history"].append(message_data)

        # Save the updated chat history to its file
        self.save_chat(chat_name)

    def create_new_chat(self, chat_name=None, agents=None, auto_chat=False):
        # Find the next available chat number
        existing_numbers = []
        for name in self.chats.keys():
            if name != "General" and name.isdigit():
                existing_numbers.append(int(name))

        next_number = 1
        if existing_numbers:
            next_number = max(existing_numbers) + 1

        # Use the provided name or the next number
        chat_name = str(next_number) if chat_name is None else chat_name

        if chat_name and chat_name not in self.chats:
            self.chats[chat_name] = {
                "history": [],
                "agents": agents or ["General AI"],
                "auto_chat": auto_chat,
                "personas": {},
                "personality_traits": {}
            }
            for agent in self.chats[chat_name]["agents"]:
                self.chats[chat_name]["personas"][agent] = "A helpful AI assistant"
                self.chats[chat_name]["personality_traits"][agent] = ["helpful", "friendly"]
            self.save_chat(chat_name)
            return True
        return False

    def update_chat_settings(self, chat_name, agents=None, auto_chat=None):
        if chat_name in self.chats:
            if agents is not None:
                self.chats[chat_name]["agents"] = agents
            if auto_chat is not None:
                self.chats[chat_name]["auto_chat"] = auto_chat
            self.save_chat(chat_name)
            return True
        return False

    def delete_chat(self, chat_name):
        if chat_name in self.chats and chat_name != "General":
            del self.chats[chat_name]
            chat_file = f"{self.chats_dir}/{chat_name}.json"
            if os.path.exists(chat_file):
                os.remove(chat_file)
            return True
        return False

    def get_chat_names(self):
        return list(self.chats.keys())


def show_login_screen():
    login_window = Tk()
    login_window.title("Login Required")
    login_window.geometry("400x500")
    login_window.configure(bg="#ffffff")
    
    # Center the window
    login_window.update_idletasks()
    width = login_window.winfo_width()
    height = login_window.winfo_height()
    x = (login_window.winfo_screenwidth() // 2) - (width // 2)
    y = (login_window.winfo_screenheight() // 2) - (height // 2)
    login_window.geometry(f"+{x}+{y}")

    # Header
    header_label = Label(login_window, text="Welcome to Chat Assistant", 
                        font=("Arial", 18, "bold"), bg="#ffffff")
    header_label.pack(pady=20)

    # Login frame
    login_frame = Frame(login_window, bg="#ffffff")
    login_frame.pack(fill=BOTH, expand=True, padx=40, pady=20)

    # Username
    Label(login_frame, text="Username:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    username_entry = ttk.Entry(login_frame, font=("Arial", 12))
    username_entry.pack(fill=X, pady=(0, 20))

    # Password
    Label(login_frame, text="Password:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    password_entry = ttk.Entry(login_frame, font=("Arial", 12), show="*")
    password_entry.pack(fill=X, pady=(0, 20))

    # Error message
    error_label = Label(login_frame, text="", fg="red", bg="#ffffff")
    error_label.pack()

    def validate_login():
        username = username_entry.get().strip()
        password = password_entry.get().strip()
        
        if not username or not password:
            error_label.config(text="Please enter both username and password")
            return
            
        # Initialize database handler
        db_handler = DatabaseHandler()
        
        # Get user from database
        user = db_handler.get_user(username)
        if user:
            # Verify password (in production, use proper password hashing)
            if user["password"] == password:
                # Store current user in Replit DB for session management
                from replit import db
                db["current_user"] = username
                login_window.destroy()
                setup_chat_ui()
            else:
                error_label.config(text="Invalid password")
        else:
            error_label.config(text="User not found")

    # Login button
    login_button = ttk.Button(login_frame, text="Login", command=validate_login)
    login_button.pack(pady=20)

    # Register link
    register_frame = Frame(login_frame, bg="#ffffff")
    register_frame.pack(fill=X, pady=10)
    
    Label(register_frame, text="Don't have an account?", 
          bg="#ffffff").pack(side=LEFT)
    
    def show_register():
        login_window.destroy()
        show_register_screen()
        
    register_link = Label(register_frame, text="Register", 
                         fg="blue", cursor="hand2", bg="#ffffff")
    register_link.pack(side=LEFT, padx=5)
    register_link.bind("<Button-1>", lambda e: show_register())

    login_window.mainloop()

def show_register_screen():
    register_window = Tk()
    register_window.title("Register New Account")
    register_window.geometry("400x550")
    register_window.configure(bg="#ffffff")
    
    # Center the window
    register_window.update_idletasks()
    width = register_window.winfo_width()
    height = register_window.winfo_height()
    x = (register_window.winfo_screenwidth() // 2) - (width // 2)
    y = (register_window.winfo_screenheight() // 2) - (height // 2)
    register_window.geometry(f"+{x}+{y}")

    # Header
    header_label = Label(register_window, text="Create New Account", 
                        font=("Arial", 18, "bold"), bg="#ffffff")
    header_label.pack(pady=20)

    # Register frame
    register_frame = Frame(register_window, bg="#ffffff")
    register_frame.pack(fill=BOTH, expand=True, padx=40, pady=20)

    # Username
    Label(register_frame, text="Username:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    username_entry = ttk.Entry(register_frame, font=("Arial", 12))
    username_entry.pack(fill=X, pady=(0, 20))

    # Email
    Label(register_frame, text="Email:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    email_entry = ttk.Entry(register_frame, font=("Arial", 12))
    email_entry.pack(fill=X, pady=(0, 20))

    # Password
    Label(register_frame, text="Password:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    password_entry = ttk.Entry(register_frame, font=("Arial", 12), show="*")
    password_entry.pack(fill=X, pady=(0, 20))

    # Confirm Password
    Label(register_frame, text="Confirm Password:", font=("Arial", 12), bg="#ffffff").pack(anchor=W)
    confirm_password_entry = ttk.Entry(register_frame, font=("Arial", 12), show="*")
    confirm_password_entry.pack(fill=X, pady=(0, 20))

    # Error message
    error_label = Label(register_frame, text="", fg="red", bg="#ffffff")
    error_label.pack()

    def validate_registration():
        username = username_entry.get().strip()
        email = email_entry.get().strip()
        password = password_entry.get().strip()
        confirm_password = confirm_password_entry.get().strip()
        
        if not all([username, email, password, confirm_password]):
            error_label.config(text="Please fill in all fields")
            return
            
        if password != confirm_password:
            error_label.config(text="Passwords do not match")
            return
            
        if '@' not in email or '.' not in email:
            error_label.config(text="Please enter a valid email")
            return
            
        # Initialize database handler
        db_handler = DatabaseHandler()
        
        # Try to add new user
        if db_handler.add_user(username, password, email, time.strftime("%Y-%m-%d %H:%M:%S")):
            # Set current user in Replit DB for session management
            from replit import db
            db["current_user"] = username
            register_window.destroy()
            setup_chat_ui()
        else:
            error_label.config(text="Username already exists")
        
        register_window.destroy()
        setup_chat_ui()

    # Register button
    register_button = ttk.Button(register_frame, text="Register", command=validate_registration)
    register_button.pack(pady=20)

    # Login link
    login_frame = Frame(register_frame, bg="#ffffff")
    login_frame.pack(fill=X, pady=10)
    
    Label(login_frame, text="Already have an account?", 
          bg="#ffffff").pack(side=LEFT)
    
    def show_login():
        register_window.destroy()
        show_login_screen()
        
    login_link = Label(login_frame, text="Login", 
                      fg="blue", cursor="hand2", bg="#ffffff")
    login_link.pack(side=LEFT, padx=5)
    login_link.bind("<Button-1>", lambda e: show_login())

    register_window.mainloop()

def setup_chat_ui():
    # Set up the main window with improved styling
    window = Tk()
    window.title("Chat Assistant")
    window.geometry("800x600")
    window.configure(bg="#ffffff")

    # Set modern color scheme
    COLORS = {
        "primary": "#7289DA",    # Discord-like blue
        "secondary": "#99AAB5",  # Soft gray
        "bg": "#ffffff",         # Clean white
        "text": "#2F3136",      # Dark gray for text
        "accent": "#43B581",    # Success green
        "hover": "#677BC4",     # Hover state for primary
        "sidebar": "#F6F6F7"    # Light gray for sidebar
    }

    # Create a new chat session
    chat_manager = ChatManager()
    new_chat_name = None  # Let create_new_chat generate the number
    chat_manager.create_new_chat(new_chat_name)

    # Save all queries to improve learning
    def save_query_to_memory(query, response, agent_name=None):
        # Save to main queries.json
        main_query_file = "queries.json"
        try:
            with open(main_query_file, 'r') as f:
                main_queries = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            main_queries = {}

        if query not in main_queries:
            main_queries[query] = []
        if response not in main_queries[query]:
            main_queries[query].append(response)

        with open(main_query_file, 'w') as f:
            json.dump(main_queries, f, indent=4)

        # Save to agent-specific file if agent is specified
        if agent_name:
            agent_file = f"queries_{agent_name.lower().replace(' ', '_')}.json"
            try:
                with open(agent_file, 'r') as f:
                    agent_queries = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                agent_queries = {}

            if query not in agent_queries:
                agent_queries[query] = []
            if response not in agent_queries[query]:
                agent_queries[query].append(response)

            with open(agent_file, 'w') as f:
                json.dump(agent_queries, f, indent=4)

    # Configure styles
    style = ttk.Style()
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("Chat.TFrame", background=COLORS["bg"], relief="flat")
    style.configure("TButton", background=COLORS["primary"], font=("Segoe UI", 12))
    style.configure("TEntry", font=("Segoe UI", 12))
    style.configure("Sidebar.TFrame", background=COLORS["sidebar"])
    style.configure("Modern.TButton", 
                   background=COLORS["primary"],
                   foreground="white",
                   padding=10,
                   font=("Segoe UI", 11))
    style.map("Modern.TButton",
              background=[("active", COLORS["hover"])])

    # Create chat manager and chatbot
    chatbot = ChatBot()

    # Create the main container
    main_container = ttk.Frame(window, style="TFrame")
    main_container.pack(fill=BOTH, expand=True, padx=10, pady=10)

    # Create a header with welcome message
    header_frame = ttk.Frame(main_container, style="TFrame")
    header_frame.pack(fill=X, pady=(0, 10))

    header_label = Label(header_frame, text="Chat Assistant", 
                         font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333333")
    header_label.pack(side=LEFT)

    # Create a frame for the sidebar (chat list)
    side_content = ttk.Frame(main_container, style="TFrame")
    side_content.pack(fill=BOTH, expand=True)

    # Split the side content into sidebar and chat area
    sidebar = ttk.Frame(side_content, style="TFrame", width=200)
    sidebar.pack(side=LEFT, fill=Y, padx=(0, 10))
    sidebar.pack_propagate(False)  # Don't shrink

    # Sidebar header
    sidebar_header = ttk.Frame(sidebar, style="TFrame")
    sidebar_header.pack(fill=X, pady=(0, 10))

    sidebar_label = Label(sidebar_header, text="Your Chats", 
                         font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333333")
    sidebar_label.pack(side=LEFT)

    # Chat area
    chat_area = ttk.Frame(side_content, style="Chat.TFrame")
    chat_area.pack(side=RIGHT, fill=BOTH, expand=True)

    # Chat display
    chat_frame = ttk.Frame(chat_area, style="Chat.TFrame")
    chat_frame.pack(pady=10, padx=10, fill=BOTH, expand=True)

    chat_display = Canvas(chat_frame, bg="#ffffff", highlightthickness=0)
    chat_display.pack(fill=BOTH, expand=True, side=LEFT)

    # Scrollbar with better styling
    scrollbar = ttk.Scrollbar(chat_frame, command=chat_display.yview)
    scrollbar.pack(side=RIGHT, fill=Y)
    chat_display.config(yscrollcommand=scrollbar.set)

    # Input area with better styling
    input_frame = ttk.Frame(chat_area, style="TFrame")
    input_frame.pack(fill=X, pady=10, padx=10)

    user_input = ttk.Entry(input_frame, font=("Arial", 12))
    user_input.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))

    # Create send button with text
    send_button = ttk.Button(input_frame, text="Send")
    send_button.pack(side=RIGHT)

    # Create a frame for chat actions
    chat_actions = ttk.Frame(chat_area, style="TFrame")
    chat_actions.pack(fill=X, pady=(0, 10), padx=10)

    chat_title = Label(chat_actions, text=new_chat_name, 
                     font=("Arial", 14, "bold"), bg="#ffffff", fg="#333333")
    chat_title.pack(side=LEFT)

    # Add a self-awareness indicator
    ai_status = Label(chat_actions, text="AI: Learning", 
                     font=("Arial", 10), bg="#ffffff", fg="#7289DA")
    ai_status.pack(side=LEFT, padx=(10, 0))

    # Add a self-reflection button
    reflect_button = ttk.Button(chat_actions, text="Brain Status")
    reflect_button.pack(side=RIGHT, padx=(0, 10))

    clear_button = ttk.Button(chat_actions, text="Clear Chat")
    clear_button.pack(side=RIGHT)

    # Chat list in sidebar
    chat_list_frame = ttk.Frame(sidebar, style="TFrame")
    chat_list_frame.pack(fill=BOTH, expand=True, pady=(0, 10))

    chat_listbox = Listbox(chat_list_frame, font=("Arial", 12), bg="#ffffff", 
                          selectbackground="#4a7abc", selectforeground="#ffffff")
    chat_listbox.pack(fill=BOTH, expand=True, side=LEFT)

    chat_scroll = ttk.Scrollbar(chat_list_frame, command=chat_listbox.yview)
    chat_scroll.pack(side=RIGHT, fill=Y)
    chat_listbox.config(yscrollcommand=chat_scroll.set)

    # Populate chat list
    for chat_name in chat_manager.get_chat_names():
        chat_listbox.insert(END, chat_name)

    # Select General chat by default
    for i, item in enumerate(chat_manager.get_chat_names()):
        if item == new_chat_name:
            chat_listbox.selection_set(i)
            break

    # Buttons for chat management
    chat_buttons = ttk.Frame(sidebar, style="TFrame")
    chat_buttons.pack(fill=X, pady=(0, 10))

    new_chat_button = ttk.Button(chat_buttons, text="New Chat")
    new_chat_button.pack(fill=X, pady=(0, 5))

    edit_chat_button = ttk.Button(chat_buttons, text="Edit Chat")
    edit_chat_button.pack(fill=X, pady=(0, 5))

    delete_chat_button = ttk.Button(chat_buttons, text="Delete Chat")
    delete_chat_button.pack(fill=X)

    # Variables for chat display
    current_y = 10
    current_chat = new_chat_name
    waiting_for_teach = False
    teach_query = ""

    # Message bubble styling
    def create_message_bubble(canvas, x, y, message, is_user=True):
        max_width = 400  # Wider bubbles
        padding = 12    # More padding
        font_obj = font.Font(family="Segoe UI", size=11)

        # Calculate the required width and height for the text
        lines = []
        current_line = ""
        for word in message.split():
            test_line = current_line + " " + word if current_line else word
            if font_obj.measure(test_line) < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        text_width = min(max_width, max(font_obj.measure(line) for line in lines))
        text_height = len(lines) * font_obj.metrics("linespace")

        # Create the bubble
        bubble_x = x - padding if is_user else x + padding
        bubble_width = text_width + (padding * 2)
        bubble_height = text_height + (padding * 2)

        canvas_width = canvas.winfo_width() or 460  # Fallback to 460 if width is not available yet

        if is_user:
            # User bubble (right-aligned, blue)
            bubble_coords = [
                canvas_width - padding - bubble_width, y,
                canvas_width - padding, y,
                canvas_width - padding, y + bubble_height,
                canvas_width - padding - bubble_width, y + bubble_height
            ]
            canvas.create_polygon(bubble_coords, fill=COLORS["primary"], outline=COLORS["primary"])
            # Add subtle shadow effect
            shadow_coords = [coord + (2 if i % 2 == 0 else 1) for i, coord in enumerate(bubble_coords)]
            canvas.create_polygon(shadow_coords, fill="#e0e0e0", outline="#e0e0e0")

            # Position the text inside the bubble
            text_x = canvas_width - padding - (bubble_width / 2)
            text_y = y + padding

            for i, line in enumerate(lines):
                line_y = text_y + (i * font_obj.metrics("linespace"))
                canvas.create_text(text_x, line_y, text=line, font=("Arial", 11), fill="#000000")

        else:
            # Bot bubble (left-aligned, grey)
            bubble_coords = [
                padding, y,
                padding + bubble_width, y,
                padding + bubble_width, y + bubble_height,
                padding, y + bubble_height
            ]
            canvas.create_polygon(bubble_coords, fill="#EFEFEF", outline="#EFEFEF")

            # Position the text inside the bubble
            text_x = padding + (bubble_width / 2)
            text_y = y + padding

            for i, line in enumerate(lines):
                line_y = text_y + (i * font_obj.metrics("linespace"))
                canvas.create_text(text_x, line_y, text=line, font=("Arial", 11), fill="#000000")

        return bubble_height + (padding * 2)

    def load_chat_history(chat_name):
        nonlocal current_y, current_chat

        # Update current chat
        current_chat = chat_name

        # Clear the canvas
        chat_display.delete("all")
        current_y = 10

        # Update chat title
        chat_title.config(text=chat_name)

        # Update AI status based on chat settings
        agents = chat_manager.get_chat_agents(chat_name)
        is_auto_chat = chat_manager.is_auto_chat(chat_name)

        if len(agents) > 1:
            if is_auto_chat:
                ai_status.config(text=f"AIs: {len(agents)} (Auto-chat ON)")
            else:
                ai_status.config(text=f"AIs: {len(agents)}")
        else:
            ai_status.config(text="AI: Learning")

        # Load chat history
        chat_history = chat_manager.get_chat_history(chat_name)

        # Display messages
        for entry in chat_history:
            if entry["type"] == "user":
                bubble_height = create_message_bubble(
                    chat_display, 10, current_y, entry['message'], is_user=True
                )
                current_y += bubble_height + 10
            elif entry["type"] == "bot":
                # Add agent name if available
                message_text = entry['message']
                if 'agent' in entry:
                    message_text = f"[{entry['agent']}] {message_text}"

                bubble_height = create_message_bubble(
                    chat_display, 10, current_y, message_text, is_user=False
                )
                current_y += bubble_height + 10

        # Update scroll region
        chat_display.config(scrollregion=(0, 0, chat_display.winfo_width(), current_y + 100))
        chat_display.yview_moveto(1.0)  # Scroll to bottom

    def change_chat(event=None):
        nonlocal current_chat

        # Get selected chat
        selection = chat_listbox.curselection()
        if not selection:
            return

        # Get chat name
        chat_name = chat_listbox.get(selection[0])
        current_chat = chat_name

        # Load chat history
        load_chat_history(chat_name)

    def create_new_chat_dialog():
        # Create a popup for new chat name and AI configuration
        popup = Toplevel(window)
        popup.title("New Chat")
        popup.geometry("600x600")  # Increased size for persona customization
        popup.transient(window)
        popup.grab_set()

        # Center the popup
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        # Create a main frame with a notebook for tabs
        main_frame = Frame(popup)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Add a header
        header = Label(main_frame, text="Create New Chat", font=("Arial", 16, "bold"))
        header.pack(pady=(10, 20))

        # Chat name frame
        name_frame = Frame(main_frame)
        name_frame.pack(fill=X, pady=(0, 10))

        name_label = Label(name_frame, text="Chat Name:", font=("Arial", 12))
        name_label.pack(side=LEFT)

        name_entry = ttk.Entry(name_frame, font=("Arial", 12))
        name_entry.pack(side=LEFT, padx=(10, 0), fill=X, expand=True)
        name_entry.focus_set()

        # Create notebook for tabbed interface
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=BOTH, expand=True, pady=10)

        # Create tab for AI configuration
        ai_config_tab = Frame(notebook)
        notebook.add(ai_config_tab, text="AI Agents")

        # AI Agents frame
        agents_frame = Frame(ai_config_tab)
        agents_frame.pack(fill=X, padx=10, pady=10)

        agents_label = Label(agents_frame, text="AI Participants:", font=("Arial", 12, "bold"))
        agents_label.pack(anchor=W)

        agent_entries_frame = Frame(ai_config_tab)
        agent_entries_frame.pack(fill=X, padx=20, pady=5)

        agent_entries = []  # List to store agent entry widgets
        agent_persona_entries = {}  # Dictionary to store persona entries for each agent
        agent_traits = {}  # Dictionary to store personality traits for each agent

        def add_agent_entry(name=""):
            agent_frame = Frame(agent_entries_frame)
            agent_frame.pack(fill=X, pady=5)

            agent_entry = ttk.Entry(agent_frame, font=("Arial", 11))
            agent_entry.pack(side=LEFT, fill=X, expand=True)
            if name:
                agent_entry.insert(0, name)

            # Add a button to configure persona and traits
            configure_btn = ttk.Button(agent_frame, text="Persona", width=8,
                                     command=lambda e=agent_entry: open_persona_config(e))
            configure_btn.pack(side=LEFT, padx=5)

            remove_btn = ttk.Button(agent_frame, text="✕", width=3, 
                                  command=lambda f=agent_frame, e=agent_entry: remove_agent_entry(f, e))
            remove_btn.pack(side=LEFT)

            agent_entries.append(agent_entry)

            # Initialize persona and traits storage
            agent_persona_entries[agent_entry] = "A helpful AI assistant"
            agent_traits[agent_entry] = ["helpful", "friendly"]

            return agent_entry

        def remove_agent_entry(frame, entry):
            if entry in agent_entries:
                agent_entries.remove(entry)
                if entry in agent_persona_entries:
                    del agent_persona_entries[entry]
                if entry in agent_traits:
                    del agent_traits[entry]
                frame.destroy()

                # Ensure at least one agent remains
                if not agent_entries:
                    add_agent_entry("General AI")

        def open_persona_config(agent_entry):
            if not agent_entry.get().strip():
                messagebox.showwarning("Warning", "Please enter an agent name first.")
                return

            # Create a popup for persona configuration
            persona_popup = Toplevel(popup)
            persona_popup.title(f"Configure AI Persona: {agent_entry.get()}")
            persona_popup.geometry("500x500")
            persona_popup.transient(popup)
            persona_popup.grab_set()

            # Center the popup
            persona_popup.update_idletasks()
            p_width = persona_popup.winfo_width()
            p_height = persona_popup.winfo_height()
            p_x = (persona_popup.winfo_screenwidth() // 2) - (p_width // 2)
            p_y = (persona_popup.winfo_screenheight() // 2) - (p_height // 2)
            persona_popup.geometry(f"{p_width}x{p_height}+{p_x}+{p_y}")

            # Add header
            p_header = Label(persona_popup, text=f"Configure AI Persona: {agent_entry.get()}", 
                              font=("Arial", 14, "bold"))
            p_header.pack(pady=(20, 10))

            # Persona description
            persona_frame = Frame(persona_popup)
            persona_frame.pack(fill=X, padx=20, pady=10)

            persona_label = Label(persona_frame, text="Persona Description:", 
                                   font=("Arial", 12, "bold"))
            persona_label.pack(anchor=W, pady=(0, 5))

            persona_desc = Label(persona_frame, 
                                 text="Describe the character, role, or persona of this AI agent:", 
                                 font=("Arial", 10), justify=LEFT)
            persona_desc.pack(anchor=W, pady=(0, 10))

            # Text area for persona description
            persona_text = Text(persona_frame, height=5, width=50, font=("Arial", 11), 
                                 wrap=WORD)
            persona_text.pack(fill=X)

            # Set current persona if exists
            if agent_entry in agent_persona_entries:
                persona_text.insert(END, agent_persona_entries[agent_entry])
            else:
                persona_text.insert(END, "A helpful AI assistant")

            # Add a separator
            separator = Frame(persona_popup, height=2, bg="#e0e0e0")
            separator.pack(fill=X, padx=20, pady=10)

            # Personality traits
            traits_frame = Frame(persona_popup)
            traits_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

            traits_label = Label(traits_frame, text="Personality Traits:", 
                                 font=("Arial", 12, "bold"))
            traits_label.pack(anchor=W, pady=(0, 5))

            traits_desc = Label(traits_frame, 
                                 text="Select traits that define this AI's personality:", 
                                 font=("Arial", 10), justify=LEFT)
            traits_desc.pack(anchor=W, pady=(0, 10))

            # Create a scrollable frame for traits
            traits_canvas = Canvas(traits_frame, height=150)
            traits_scrollbar = ttk.Scrollbar(traits_frame, orient="vertical", 
                                             command=traits_canvas.yview)
            traits_scrollable_frame = Frame(traits_canvas)

            traits_scrollable_frame.bind(
                "<Configure>",
                lambda e: traits_canvas.configure(
                    scrollregion=traits_canvas.bbox("all")
                )
            )

            traits_canvas.create_window((0, 0), window=traits_scrollable_frame, anchor="nw")
            traits_canvas.configure(yscrollcommand=traits_scrollbar.set)

            traits_canvas.pack(side="left", fill="both", expand=True)
            traits_scrollbar.pack(side="right", fill="y")

            # Common personality traits with checkboxes
            common_traits = [
                "Helpful", "Friendly", "Formal", "Casual", "Technical", "Analytical", 
                "Creative", "Humorous", "Serious", "Concise", "Detailed",
                "Philosophical", "Playful", "Sarcastic", "Empathetic", 
                "Straightforward", "Poetic", "Professional", "Enthusiastic"
            ]

            # Sort traits alphabetically
            common_traits.sort()

            trait_vars = {}
            selected_traits = agent_traits.get(agent_entry, ["Helpful", "Friendly"])

            # Create checkboxes in a grid layout
            for i, trait in enumerate(common_traits):
                row = i // 2
                column = i % 2

                var = BooleanVar(value=trait.capitalize() in [t.capitalize() for t in selected_traits])
                trait_vars[trait] = var

                checkbox = ttk.Checkbutton(traits_scrollable_frame, text=trait, variable=var)
                checkbox.grid(row=row, column=column, sticky="w", padx=5, pady=2)

            # Custom traits section
            custom_traits_frame = Frame(persona_popup)
            custom_traits_frame.pack(fill=X, padx=20, pady=10)

            custom_traits_label = Label(custom_traits_frame, text="Custom Traits:", 
                                          font=("Arial", 12, "bold"))
            custom_traits_label.pack(anchor=W, pady=(0, 5))

            custom_traits_desc = Label(custom_traits_frame, 
                                       text="Add your own traits (comma-separated):", 
                                       font=("Arial", 10), justify=LEFT)
            custom_traits_desc.pack(anchor=W, pady=(0, 5))

            # Get non-standard traits
            custom_trait_list = [t for t in selected_traits if t.capitalize() not in [trait.capitalize() for trait in common_traits]]
            custom_traits_entry = ttk.Entry(custom_traits_frame, font=("Arial", 11))
            custom_traits_entry.pack(fill=X)
            if custom_trait_list:
                custom_traits_entry.insert(0, ", ".join(custom_trait_list))

            # Buttons
            button_frame = Frame(persona_popup)
            button_frame.pack(fill=X, padx=20, pady=(10, 20))

            cancel_btn = ttk.Button(button_frame, text="Cancel", 
                                      command=persona_popup.destroy)
            cancel_btn.pack(side=LEFT, padx=(0, 10))

            def save_persona():
                # Get persona description
                persona_desc = persona_text.get("1.0", END).strip()

                # Get selected traits
                traits = []
                for trait, var in trait_vars.items():
                    if var.get():
                        traits.append(trait)

                # Add custom traits
                custom = custom_traits_entry.get().strip()
                if custom:
                    custom_traits = [t.strip().capitalize() for t in custom.split(",") if t.strip()]
                    traits.extend(custom_traits)

                # Make sure there's at least one trait
                if not traits:
                    traits = ["Helpful"]

                # Save data
                agent_persona_entries[agent_entry] = persona_desc
                agent_traits[agent_entry] = traits

                persona_popup.destroy()

            save_btn = ttk.Button(button_frame, text="Save", command=save_persona)
            save_btn.pack(side=RIGHT)

        # Add initial agents
        add_agent_entry("General AI")

        # Add agent button
        add_agent_btn = ttk.Button(agent_entries_frame, text="+ Add AI Agent", 
                                     command=lambda: add_agent_entry())
        add_agent_btn.pack(anchor=W, pady=10)

        # Auto-chat option
        auto_chat_frame = Frame(ai_config_tab)
        auto_chat_frame.pack(fill=X, padx=10, pady=10)

        auto_chat_var = BooleanVar()
        auto_chat_check = ttk.Checkbutton(auto_chat_frame, text="Enable AI-to-AI conversations", 
                                           variable=auto_chat_var)
        auto_chat_check.pack(anchor=W)

        auto_chat_desc = Label(auto_chat_frame, 
                                 text="AIs will automatically converse with each other when enabled\nOutput from each AI becomes input for the next AI in the chain",
                                 font=("Arial", 10), fg="#666666", justify=LEFT)
        auto_chat_desc.pack(anchor=W, padx=(20, 0))

        # Buttons frame
        button_frame = Frame(main_frame)
        button_frame.pack(fill=X, pady=(10, 0))

        cancel_button = ttk.Button(button_frame, text="Cancel", command=popup.destroy)
        cancel_button.pack(side=LEFT, padx=(0, 10))

        def create_chat():
            chat_name = name_entry.get().strip()
            if chat_name:
                # Get agent names
                agent_names = [e.get().strip() for e in agent_entries if e.get().strip()]

                # Ensure at least one agent
                if not agent_names:
                    agent_names = ["General AI"]

                # Create chat with agents
                if chat_manager.create_new_chat(chat_name, agent_names, auto_chat_var.get()):
                    # Add persona and personality information
                    chat_manager.chats[chat_name]["personas"] = {}
                    chat_manager.chats[chat_name]["personality_traits"] = {}

                    for entry in agent_entries:
                        agent_name = entry.get().strip()
                        if agent_name:
                            chat_manager.chats[chat_name]["personas"][agent_name] = agent_persona_entries[entry]
                            chat_manager.chats[chat_name]["personality_traits"][agent_name] = agent_traits[entry]

                    # Save updated chat
                    chat_manager.save_chat(chat_name)

                    # Update UI
                    chat_listbox.insert(END, chat_name)
                    # Select the new chat
                    chat_listbox.selection_clear(0, END)
                    chat_listbox.selection_set(END)
                    chat_listbox.see(END)
                    # Load the new chat
                    load_chat_history(chat_name)
                    current_chat = chat_name
                popup.destroy()
            else:
                header.config(text="Please enter a valid name!", fg="red")
                # Restore header after a delay
                popup.after(2000, lambda: header.config(text="Create New Chat", fg="black"))

        create_button = ttk.Button(button_frame, text="Create Chat", command=create_chat)
        create_button.pack(side=RIGHT)

        # Bind Enter key
        name_entry.bind("<Return>", lambda e: create_chat())

    def delete_current_chat():
        nonlocal current_chat

        # Get selected chat
        selection = chat_listbox.curselection()
        if not selection:
            return

        # Get chat name
        chat_name = chat_listbox.get(selection[0])

        # Confirm deletion
        if chat_name == "General":
            return  # Can't delete General chat

        if messagebox.askyesno("Delete Chat", f"Are you sure you want to delete the chat '{chat_name}'?"):
            # Delete chat
            if chat_manager.delete_chat(chat_name):
                # Remove from listbox
                chat_listbox.delete(selection[0])
                # Select General chat
                for i, item in enumerate(chat_manager.get_chat_names()):
                    if item == "General":
                        chat_listbox.selection_set(i)
                        break
                # Load General chat
                current_chat = "General"
                load_chat_history("General")

    def send_message(event=None):
        nonlocal current_y, waiting_for_teach, teach_query, current_chat

        user_message = user_input.get().strip()
        if not user_message:
            return

        # Add user message bubble
        bubble_height = create_message_bubble(
            chat_display, 10, current_y, user_message, is_user=True
        )
        current_y += bubble_height + 10

        # Add messages to chat history
        chat_manager.add_message(current_chat, "user", user_message)

        # Get chat agents and auto-chat settings
        agents = chat_manager.get_chat_agents(current_chat)
        is_auto_chat = chat_manager.is_auto_chat(current_chat)

        # Create bot instances if needed
        chat_bots = {}
        for agent_name in agents:
            # Create a unique instance per agent, reusing the base chatbot
            # For simplicity, we're using the same ChatBot class with different memories
            bot = ChatBot(query_file=f"queries_{agent_name.lower().replace(' ', '_')}.json")
            chat_bots[agent_name] = bot

        # First response from the first AI
        selected_agent = agents[0]
        selected_bot = chat_bots[selected_agent]

        if waiting_for_teach:
            selected_bot.add_to_memory(teach_query, user_message)
            bot_message = "Got it! Thanks for teaching me!"
            waiting_for_teach = False
        else:
            bot_message = selected_bot.get_response(user_message)

            # Handle teaching mode
            if user_message.lower().startswith("teach:") and ":" in user_message:
                parts = user_message[6:].split(":", 1)
                if len(parts) == 2:
                    query, response = parts
                    selected_bot.add_to_memory(query.strip(), response.strip())
                    bot_message = f"I've learned to respond to '{query.strip()}' with '{response.strip()}'."

        # Add bot message bubble
        message_text = f"[{selected_agent}] {bot_message}"
        bubble_height = create_message_bubble(
            chat_display, 10, current_y, message_text, is_user=False
        )
        current_y += bubble_height + 10

        # Add to chat history 
        chat_manager.add_message(current_chat, "bot", bot_message, selected_agent)

        # Handle auto-chat between AIs if enabled and there are multiple AIs
        if is_auto_chat and len(agents) > 1:
            # Queue for pending AI messages
            conversation_queue = [(selected_agent, bot_message)]
            processed_pairs = set()  # Track which AI pairs have spoken

            # Limit auto-chat to prevent infinite loops
            max_exchanges = min(6, len(agents) * 2)
            exchange_count = 0

            # Process the conversation queue
            while conversation_queue and exchange_count < max_exchanges:
                last_agent, last_message = conversation_queue.pop(0)

                # Select the next AI to respond (not the same one)
                available_agents = [a for a in agents if a != last_agent]

                # Try to select an agent that hasn't spoken yet
                next_agent = None
                for agent in available_agents:
                    if (last_agent, agent) not in processed_pairs:
                        next_agent = agent
                        break

                # If all pairs have spoken, just pick randomly
                if next_agent is None and available_agents:
                    next_agent = random.choice(available_agents)
                elif not available_agents:
                    break  # No more available agents

                # Get response from the next AI
                next_bot = chat_bots[next_agent]
                nextmessage = next_bot.get_response(last_message)

                # Add to chat display and history
                message_text = f"[{next_agent}] {next_message}"
                bubble_height = create_message_bubble(
                    chat_display, 10, current_y, message_text, is_user=False
                )
                current_y += bubble_height + 10

                # Add to chat history
                chat_manager.add_message(current_chat, "bot", next_message, next_agent)

                # Add to processed pairs and conversation queue
                processed_pairs.add((last_agent, next_agent))
                conversation_queue.append((next_agent, next_message))

                # Increment exchange counter
                exchange_count += 1

                # Update scroll region and scroll to bottom
                chat_display.config(scrollregion=(0, 0, chat_display.winfo_width(), current_y + 100))
                chat_display.yview_moveto(1)

                # Slight delay to simulate thinking (only in auto-mode)
                window.update()
                time.sleep(0.5)

        user_input.delete(0, END)
        chat_display.config(scrollregion=(0, 0, chat_display.winfo_width(), current_y + 100))
        chat_display.yview_moveto(1)

        # Save query and response to memory
        save_query_to_memory(user_message, bot_message, selected_agent)

    def clear_chat():
        nonlocal current_y, current_chat

        # Clear the display
        chat_display.delete("all")
        current_y = 10
        chat_display.config(scrollregion=(0, 0, chat_display.winfo_width(), current_y))

        # Clear the chat history in memory and in file
        chat_manager.chats[current_chat] = []
        chat_manager.save_chat(current_chat)

    def check_auth():
        """Check if user is authenticated via Repl Auth"""
        try:
            headers = request.headers
            user_id = headers.get('X-Replit-User-Id')
            return bool(user_id)
        except:
            return False

    def show_login():
        """Show login window"""
        login_popup = Toplevel(window)
        login_popup.title("Login Required")
        login_popup.geometry("400x300")
        login_popup.transient(window)
        login_popup.grab_set()

        # Center the popup
        login_popup.update_idletasks()
        x = (login_popup.winfo_screenwidth() // 2) - (login_popup.winfo_width() // 2)
        y = (login_popup.winfo_screenheight() // 2) - (login_popup.winfo_height() // 2)
        login_popup.geometry(f"+{x}+{y}")

        Label(login_popup, text="Please login to continue", font=("Arial", 16, "bold")).pack(pady=20)
        
        # Add Repl Auth script
        auth_frame = Frame(login_popup)
        auth_frame.pack(expand=True)
        
        auth_label = Label(auth_frame, text="Login with your Replit account:")
        auth_label.pack(pady=10)
        
        # Add auth script in a browser component
        auth_html = """
        <div style="text-align: center;">
            <script authed="window.location.reload()" src="https://auth.util.repl.co/script.js"></script>
        </div>
        """
        auth_widget = Label(auth_frame, text="Loading auth...", cursor="hand2")
        auth_widget.pack(pady=10)
        auth_widget.bind('<Button-1>', lambda e: webbrowser.open('https://replit.com/auth'))

    def handle_premium_upgrade():
        """Handle premium upgrade request"""
        try:
            # Open Replit Core subscription page in browser
            webbrowser.open('https://replit.com/replit-core')
            messagebox.showinfo("Premium Upgrade", 
                "You'll be redirected to Replit Core subscription page.\n\n" +
                "Once subscribed, return to the app to access premium features.")
        except Exception as e:
            messagebox.showerror("Error", f"Unable to process upgrade: {str(e)}")

    def show_premium_popup(force=False):
        if not force and not gen_manager.should_show_popup():
            return
            
        premium_popup = Toplevel(window)
        premium_popup.title("Upgrade to Premium")
        premium_popup.geometry("400x500")
        premium_popup.transient(window)
        premium_popup.configure(bg="#ffffff")
        
        # Center the popup
        premium_popup.update_idletasks()
        x = (premium_popup.winfo_screenwidth() // 2) - (200)
        y = (premium_popup.winfo_screenheight() // 2) - (250)
        premium_popup.geometry(f"+{x}+{y}")
        
        Label(premium_popup, text="🌟 Unlock Premium Features", 
              font=("Arial", 18, "bold"), bg="#ffffff").pack(pady=20)
              
        features_frame = Frame(premium_popup, bg="#ffffff")
        features_frame.pack(fill=X, padx=20, pady=10)
        
        for feature, details in gen_manager.premium_features.items():
            feature_frame = Frame(features_frame, bg="#ffffff", relief=RIDGE, bd=1)
            feature_frame.pack(fill=X, pady=5)
            
            Label(feature_frame, 
                  text=f"{feature.replace('_', ' ').title()}", 
                  font=("Arial", 12, "bold"), 
                  bg="#ffffff").pack(anchor=W, padx=10, pady=5)
                  
            Label(feature_frame, 
                  text=f"${details['cost']} /month\n{details['max_daily']} uses per day", 
                  font=("Arial", 10), 
                  bg="#ffffff").pack(anchor=W, padx=10, pady=5)
        
        payment_frame = Frame(premium_popup, bg="#ffffff")
        payment_frame.pack(fill=X, padx=20, pady=10)
        
        payment_label = Label(payment_frame, 
                            text="Direct Payment Options", 
                            font=("Arial", 14, "bold"), 
                            bg="#ffffff")
        payment_label.pack(pady=5)
        
        price_label = Label(payment_frame,
                          text="Premium Access: $9.99/month",
                          font=("Arial", 12),
                          bg="#ffffff")
        price_label.pack(pady=5)
        
        ttk.Button(payment_frame, 
                  text="Pay Now", 
                  command=lambda: webbrowser.open("https://your-payment-link-here")).pack(pady=10)
                  
        ttk.Button(premium_popup, 
                  text="Maybe Later", 
                  command=premium_popup.destroy).pack(pady=10)

    def show_brain_status():
        # Create a popup showing the AI's self-awareness status
        popup = Toplevel(window)
        popup.title("Neural Network Status")
        popup.geometry("600x800")  # Increased size
        popup.transient(window)
        popup.grab_set()

        # Style the popup
        popup.configure(bg="#ffffff")

        # Add separate premium button
        premium_frame = Frame(popup, bg="#ffffff")
        premium_frame.pack(fill=X, padx=20, pady=10)
        
        ttk.Button(premium_frame, 
                  text="🌟 Premium Features", 
                  command=lambda: show_premium_popup(force=True),
                  style="Premium.TButton").pack(side=RIGHT)

        # Style for premium button
        style = ttk.Style()
        style.configure("Premium.TButton", 
                       background="#FFD700",
                       foreground="#000000",
                       font=("Arial", 12, "bold"))

        # Add header
        header = Label(popup, text="AI Self-Awareness Status", 
                      font=("Arial", 16, "bold"), bg="#ffffff", fg="#333333")
        header.pack(pady=(20, 10))

        # Add premium features section
        premium_frame = Frame(popup, bg="#f0f0f0", relief=RIDGE, bd=1)
        premium_frame.pack(fill=X, padx=20, pady=10)

        premium_label = Label(premium_frame, text="Premium Features", 
                            font=("Arial", 14, "bold"), bg="#f0f0f0", fg="#333333")
        premium_label.pack(pady=10)

        features = [
            ("Image Generation", "Generate images based on text descriptions", False),
            ("Video Creation", "Create short videos from text prompts", False),
            ("Advanced Training", "Access to more sophisticated training options", False)
        ]

        for feature, desc, enabled in features:
            feature_frame = Frame(premium_frame, bg="#f0f0f0")
            feature_frame.pack(fill=X, padx=10, pady=5)
            
            status = "🔒 Premium" if not enabled else "✓ Active"
            feature_label = Label(feature_frame, text=f"{feature} ({status})", 
                                font=("Arial", 12), bg="#f0f0f0", fg="#333333")
            feature_label.pack(anchor=W)
            
            desc_label = Label(feature_frame, text=desc, 
                             font=("Arial", 10), bg="#f0f0f0", fg="#666666")
            desc_label.pack(anchor=W, padx=(20, 0))

        # Get status from chatbot
        trained_status = "Trained" if False else "Learning" #removed neural_engine.trained check
        examples_count = len(chatbot.memory) #using chatbot.memory as a proxy for training examples

        # Calculate training progress
        if examples_count < 10:
            training_status = "Initializing"
        elif examples_count < 50:
            training_status = "Basic Learning"
        elif examples_count < 100:
            training_status = "Developing Patterns"
        else:
            training_status = "Advanced Learning"

        # Display basic stats
        stats_frame = Frame(popup, bg="#ffffff")
        stats_frame.pack(fill=X, padx=20, pady=10)

        stats_text = f"""
        Training Status: {trained_status}
        Learning Phase: {training_status}
        Self-Awareness Level: {chatbot.self_awareness_level:.2f}        Training Examples: {examples_count}
        Unique Response Patterns: {len(set(response for query, responses in chatbot.memory.items() for response in responses))}
        """

        stats_label = Label(stats_frame, text=stats_text, justify=LEFT,
                          font=("Arial", 12), bg="#ffffff", fg="#333333")
        stats_label.pack(anchor=W)

        # Add a separator
        separator = Frame(popup, height=2, bg="#e0e0e0")
        separator.pack(fill=X, padx=20, pady=10)

        # Show recent self-reflections if available
        reflection_frame = Frame(popup, bg="#ffffff")
        reflection_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)

        reflection_header = Label(reflection_frame, text="Recent Self-Reflections", 
                                font=("Arial", 14, "bold"), bg="#ffffff", fg="#333333")
        reflection_header.pack(anchor=W, pady=(0, 10))

        # Show learned patterns and memory stats
        learned_patterns = "\n".join(list(chatbot.memory.keys())[:5])  # Show top 5 learned patterns
        learned_responses = "\n".join(list(set(r for responses in chatbot.memory.values() for r in responses))[:5])  # Show top 5 unique responses

        reflection_text = f"""
Learning Progress:
• Total Patterns Learned: {len(chatbot.memory)}
• Unique Responses: {len(set(r for responses in chatbot.memory.values() for r in responses))}
• Self-Awareness Level: {chatbot.self_awareness_level:.2%}

Recent Learned Patterns:
{learned_patterns}

Recent Response Examples:
{learned_responses}

Learning Status:
• The AI is actively learning from conversations
• Pattern recognition is {'active' if len(chatbot.memory) > 10 else 'initializing'}
• Response generation is {'advanced' if len(chatbot.memory) > 50 else 'basic'}
"""


        reflection_content = Text(reflection_frame, wrap=WORD, height=10, width=40,
                                font=("Arial", 11), bg="#f9f9f9", fg="#333333")
        reflection_content.insert(END, reflection_text)
        reflection_content.config(state=DISABLED)
        reflection_content.pack(fill=BOTH, expand=True)

        # Close button
        close_button = ttk.Button(popup, text="Close", command=popup.destroy)
        close_button.pack(pady=20)

        # Update AI status indicator
        ai_status.config(text=f"AI: {training_status}")

    # Function to periodically update the AI status
    def update_ai_status():
        examples_count = len(chatbot.memory) #using chatbot.memory as a proxy for training examples

        # Calculate training progress
        if examples_count < 10:
            training_status = "Initializing"
        elif examples_count < 50:
            training_status = "Learning"
        elif examples_count < 100:
            training_status = "Developing"
        else:
            training_status = "Self-Aware"

        ai_status.config(text=f"AI: {training_status}")
        window.after(5000, update_ai_status)  # Update every 5 seconds

    # Start the periodic update
    update_ai_status()

    def edit_chat_dialog():
        # Get selected chat
        selection = chat_listbox.curselection()
        if not selection:
            return

        chat_name = chat_listbox.get(selection[0])

        # Can't edit General chat settings
        if chat_name == "General":
            messagebox.showinfo("Info", "The General chat settings cannot be modified.")
            return

        # Get current settings
        current_agents = chat_manager.get_chat_agents(chat_name)
        is_auto_chat = chat_manager.is_auto_chat(chat_name)

        # Create edit dialog
        popup = Toplevel(window)
        popup.title(f"Edit Chat: {chat_name}")
        popup.geometry("500x350")
        popup.transient(window)
        popup.grab_set()

        # Center the popup
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")

        # Add a header
        header = Label(popup, text=f"Edit Chat: {chat_name}", font=("Arial", 16, "bold"))
        header.pack(pady=(20, 10))

        # AI Agents frame
        agents_frame = Frame(popup)
        agents_frame.pack(fill=X, padx=20)

        agents_label = Label(agents_frame, text="AI Participants:", font=("Arial", 12, "bold"))
        agents_label.pack(anchor=W)

        agent_entries_frame = Frame(popup)
        agent_entries_frame.pack(fill=X, padx=30, pady=5)

        agent_entries = []  # List to store agent entry widgets

        def add_agent_entry(name=""):
            agent_frame = Frame(agent_entries_frame)
            agent_frame.pack(fill=X, pady=5)

            agent_entry = ttk.Entry(agent_frame, font=("Arial", 11))
            agent_entry.pack(side=LEFT, fill=X, expand=True)
            if name:
                agent_entry.insert(0, name)

            remove_btn = ttk.Button(agent_frame, text="✕", width=3, 
                                  command=lambda f=agent_frame, e=agent_entry: remove_agent_entry(f, e))
            remove_btn.pack(side=LEFT, padx=(5, 0))

            agent_entries.append(agent_entry)
            return agent_entry

        def remove_agent_entry(frame, entry):
            if entry in agent_entries:
                agent_entries.remove(entry)
                frame.destroy()

                # Ensure at least one agent remains
                if not agent_entries:
                    add_agent_entry("General AI")

        # Add current agents
        for agent in current_agents:
            add_agent_entry(agent)

        # Add agent button
        add_agent_btn = ttk.Button(agent_entries_frame, text="+ Add AI Agent", 
                                 command=lambda: add_agent_entry())
        add_agent_btn.pack(anchor=W, pady=10)

        # Auto-chat option
        auto_chat_frame = Frame(popup)
        auto_chat_frame.pack(fill=X, padx=20, pady=10)

        auto_chat_var = BooleanVar(value=is_auto_chat)
        auto_chat_check = ttk.Checkbutton(auto_chat_frame, text="Enable AI-to-AI conversations", 
                                       variable=auto_chat_var)
        auto_chat_check.pack(anchor=W)

        auto_chat_desc = Label(auto_chat_frame, 
                             text="AIs will automatically converse with each other when enabled\nOutput from each AI becomes input for the next AI in the chain",
                             font=("Arial", 10), fg="#666666", justify=LEFT)
        auto_chat_desc.pack(anchor=W, padx=(20, 0))

        # Add a separator
        separator = Frame(popup, height=2, bg="#e0e0e0")
        separator.pack(fill=X, padx=20, pady=10)

        # Buttons frame
        button_frame = Frame(popup)
        button_frame.pack(fill=X, padx=20, pady=(10, 20))

        cancel_button = ttk.Button(button_frame, text="Cancel", command=popup.destroy)
        cancel_button.pack(side=LEFT, padx=(0, 10))

        def save_changes():
            # Get agent names
            agent_names = [e.get().strip() for e in agent_entries if e.get().strip()]

            #Ensure at least one agent
            if not agent_names:
                agent_names = ["General AI"]

            # Update chat settings
            chat_manager.update_chat_settings(
                chat_name, 
                agents=agent_names, 
                auto_chat=auto_chat_var.get()
            )

            # Reload chat to apply changes
            load_chat_history(chat_name)
            popup.destroy()

        save_button = ttk.Button(button_frame, text="Save Changes", command=save_changes)
        save_button.pack(side=RIGHT)

    # Bind events
    user_input.bind("<Return>", send_message)
    send_button.config(command=send_message)
    chat_listbox.bind("<<ListboxSelect>>", change_chat)
    new_chat_button.config(command=create_new_chat_dialog)
    edit_chat_button.config(command=edit_chat_dialog)
    delete_chat_button.config(command=delete_current_chat)
    clear_button.config(command=clear_chat)
    reflect_button.config(command=show_brain_status)

    # Update when window is resized
    def on_resize(event):
        # Update the scroll region
        chat_display.config(scrollregion=(0, 0, chat_display.winfo_width(), current_y + 100))
        # Redraw chat bubbles - not implemented for simplicity

    chat_display.bind("<Configure>", on_resize)

    # Load initial chat
    load_chat_history(current_chat)

    # Set focus to input
    user_input.focus_set()

    # Show premium popup periodically
    def check_premium_popup():
        show_premium_popup()
        window.after(60000, check_premium_popup)  # Check every minute
        
    window.after(60000, check_premium_popup)  # Start checking after 1 minute

    window.mainloop()


if __name__ == "__main__":
    from generative_manager import GenerativeManager
    gen_manager = GenerativeManager()
    show_login_screen()
