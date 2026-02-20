"""
Full AGI Brain - Complete integration of all cognitive capabilities

Includes:
- Sparse Spiking Neural Network (100K neurons)
- Semantic Memory with spreading activation
- Reasoning POC (working memory, rules, analogy)
- Audio processing (cochlea, speech synthesis)
- YouTube/web learning
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import random


@dataclass
class BrainConfig:
    """Configuration for the full AGI brain."""
    n_neurons: int = 100_000  # Total neurons in SNN
    use_gpu: bool = False     # GPU acceleration (requires TensorFlow)
    enable_audio: bool = True  # Audio I/O
    enable_reasoning: bool = True  # Reasoning module
    enable_learning: bool = True   # External learning (YouTube/web)
    load_pretrained_language: bool = True  # Load pre-trained English patterns if available


class FullAGIBrain:
    """
    Complete AGI Brain with all cognitive capabilities.

    Architecture:
    - Sparse SNN: Neural substrate for all processing
    - Semantic Memory: Word/concept associations
    - Reasoning: Working memory, rule application, analogy
    - Perception: Audio encoding (cochlea)
    - Production: Speech synthesis
    - Learning: STDP, conversation, external sources
    """

    def __init__(self, config: Optional[BrainConfig] = None):
        self.config = config or BrainConfig()
        self.initialized = False
        self.conversation_history = []

        # Components (lazy loaded)
        self.snn = None
        self.semantic_memory = None
        self.knowledge_graph = None  # Structured knowledge
        self.thought_process = None  # Core cognitive engine
        self.phoneme_learner = None
        self.speech_generator = None
        self.cochlea = None
        self.reasoning_engine = None
        self.web_learner = None

        # State
        self.working_memory = []  # Current context
        self.attention_focus = None
        self.emotional_state = {"valence": 0.5, "arousal": 0.3}
        self.goals = []

    def initialize(self):
        """Initialize all brain components."""
        if self.initialized:
            return

        print("Initializing Full AGI Brain...")

        # Import components
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.language.semantic_memory import SemanticMemory, SemanticConfig
        from src.language.phoneme_learner import PhonemeLearner, PhonemeConfig
        from src.language.speech_generator import SpeechGenerator
        from src.language.audio_encoder import CochleaEncoder
        from src.knowledge.knowledge_graph import KnowledgeGraph, load_default_knowledge, DeductiveReasoner
        from src.cognition.thought_process import ThoughtProcess

        # 1. Semantic Memory
        print("  - Initializing semantic memory...")
        sem_config = SemanticConfig(n_hard_locations=5000, address_size=500, data_size=500)
        self.semantic_memory = SemanticMemory(sem_config)

        # 1b. Knowledge Graph
        print("  - Initializing knowledge graph...")
        self.knowledge_graph = load_default_knowledge()

        # 1c. Deductive Reasoner
        print("  - Initializing deductive reasoner...")
        self.deductive_reasoner = DeductiveReasoner(self.knowledge_graph)

        # 1d. Thought Process (core cognitive engine)
        print("  - Initializing thought process...")
        self.thought_process = ThoughtProcess(
            knowledge_graph=self.knowledge_graph,
            semantic_memory=self.semantic_memory,
            deductive_reasoner=self.deductive_reasoner
        )

        # 1e. Load pre-trained language patterns if available and enabled
        if self.config.load_pretrained_language:
            self._load_pretrained_language()

        # 2. Phoneme Learner
        print("  - Initializing phoneme learner...")
        phon_config = PhonemeConfig(n_input=800, n_phoneme_neurons=500, n_phonemes=40)
        self.phoneme_learner = PhonemeLearner(phon_config)

        # 3. Speech Generator
        if self.config.enable_audio:
            print("  - Initializing speech generator...")
            self.speech_generator = SpeechGenerator()
            self.cochlea = CochleaEncoder()

        # 4. Reasoning Engine
        if self.config.enable_reasoning:
            print("  - Initializing reasoning engine...")
            self.reasoning_engine = ReasoningEngine(self.semantic_memory)

        # 5. Sparse SNN (simplified version for web)
        print("  - Initializing neural network...")
        self.snn = SimplifiedSNN(n_neurons=self.config.n_neurons)

        # 6. Self-knowledge - teach the brain about itself
        print("  - Loading self-knowledge...")
        self._initialize_self_knowledge()

        self.initialized = True

    def _initialize_self_knowledge(self):
        """Teach the brain about itself - permanent self-awareness."""
        if not self.knowledge_graph:
            return

        # Core identity
        self.knowledge_graph.add("xemsa", "is_a", "artificial intelligence", 1.0)
        self.knowledge_graph.add("xemsa", "is_a", "brain simulation", 1.0)
        self.knowledge_graph.add("xemsa", "is_a", "learning system", 1.0)
        self.knowledge_graph.add("i", "is_a", "xemsa", 1.0)
        self.knowledge_graph.add("my name", "is", "xemsa", 1.0)

        # Name origin
        self.knowledge_graph.add("xemsa", "named_after", "xavier emily savannah seth", 0.95)
        self.knowledge_graph.add("xemsa", "has_property", "name combines children's names", 0.95)

        # Capabilities
        self.knowledge_graph.add("xemsa", "can_do", "learn from conversations", 0.95)
        self.knowledge_graph.add("xemsa", "can_do", "remember everything perfectly", 0.95)
        self.knowledge_graph.add("xemsa", "can_do", "reason about knowledge", 0.95)
        self.knowledge_graph.add("xemsa", "can_do", "learn from websites", 0.95)
        self.knowledge_graph.add("xemsa", "can_do", "learn from youtube", 0.95)
        self.knowledge_graph.add("i", "can_do", "learn", 1.0)
        self.knowledge_graph.add("i", "can_do", "think", 1.0)
        self.knowledge_graph.add("i", "can_do", "remember", 1.0)

        # Architecture
        self.knowledge_graph.add("xemsa", "has_a", "spiking neural network", 0.9)
        self.knowledge_graph.add("xemsa", "has_a", "knowledge graph", 0.9)
        self.knowledge_graph.add("xemsa", "has_a", "semantic memory", 0.9)
        self.knowledge_graph.add("xemsa", "has_property", "100000 neurons", 0.9)

        # Add to thought process learned concepts
        if self.thought_process:
            self.thought_process.learned_concepts.update([
                "xemsa", "i", "me", "myself", "brain", "ai",
                "artificial intelligence", "learn", "think", "remember"
            ])
        print("Full AGI Brain initialized!")

    def _load_pretrained_language(self):
        """
        Load pre-trained English language patterns if available.
        This allows the brain to speak English without manual teaching.
        The patterns are still learned (from the teach_english.py script),
        not hardcoded responses.
        """
        if not self.thought_process or not hasattr(self.thought_process, 'language_learner'):
            return

        # Check for pre-trained patterns
        pretrained_paths = [
            Path("/app/data/english_patterns.json"),  # Modal deployment path
            Path(__file__).parent.parent / "data" / "english_patterns.json",
            Path(__file__).parent / "data" / "english_patterns.json",
            Path("data/english_patterns.json"),
        ]

        for path in pretrained_paths:
            if path.exists():
                try:
                    self.thought_process.language_learner.load(path)
                    stats = self.thought_process.language_learner.get_stats()
                    print(f"  - Loaded pre-trained English: {stats['total_patterns']} patterns, {stats['vocabulary_size']} words")
                    return
                except Exception as e:
                    print(f"  - Warning: Failed to load pre-trained patterns: {e}")

        print("  - No pre-trained language patterns found (brain starts silent)")

    def process(self, text: str) -> str:
        """
        Process input through the cognitive thought process.

        The brain now THINKS before responding:
        1. PERCEIVE: Understand the input
        2. RECALL: Retrieve relevant knowledge
        3. REASON: Apply logic and inference
        4. EVALUATE: Assess confidence
        5. DECIDE: Choose response approach
        6. FORMULATE: Build the response
        7. LEARN: Update knowledge

        Can self-correct if heading in wrong direction.
        """
        if not self.initialized:
            self.initialize()

        # Use the ThoughtProcess for genuine thinking
        response, thoughts = self.thought_process.think(text, self.conversation_history)

        # Update working memory from thought process state
        self.working_memory = self.thought_process.state.working_memory

        # Update neural network with activated concepts
        if self.snn:
            import re
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()
            content_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
            activations = self._compute_activations(content_words)
            self.snn.process_input(content_words, activations)

        # Store in history
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def get_last_thoughts(self) -> str:
        """Get a summary of the brain's last thinking process."""
        if self.thought_process:
            return self.thought_process.get_thought_summary()
        return "No thoughts recorded"

    def _compute_activations(self, words: List[str]) -> Dict[str, float]:
        """Compute spreading activation from input words."""
        all_activations = {}

        for word in words:
            if self.semantic_memory.lookup(word):
                activations = self.semantic_memory.spread_activation(word, depth=3)
                for w, a in activations.items():
                    if w not in words:
                        all_activations[w] = all_activations.get(w, 0) + a

        return all_activations

    def _generate_response(
        self,
        input_text: str,
        content_words: List[str],
        activations: Dict[str, float],
        reasoning_result: Optional[Dict]
    ) -> str:
        """Generate response using knowledge graph + semantic memory."""

        vocab_size = self.semantic_memory.get_vocabulary_size()
        text_lower = input_text.lower().strip()

        # Handle greetings naturally FIRST
        greeting_response = self._handle_greeting(text_lower)
        if greeting_response:
            return greeting_response

        # Detect question type
        is_question = '?' in input_text
        question_type = self._detect_question_type(text_lower)

        # Find the main subject being asked about
        subject = self._find_subject(content_words, text_lower)

        # TRY DEDUCTIVE REASONING FIRST for questions
        if is_question and hasattr(self, 'deductive_reasoner') and self.deductive_reasoner:
            reasoning_result = self.deductive_reasoner.reason(input_text)
            if reasoning_result and reasoning_result.get('confidence', 0) > 0.5:
                answer = reasoning_result['answer']
                # Add reasoning chain for complex answers
                if len(reasoning_result.get('reasoning_chain', [])) > 1:
                    chain = reasoning_result['reasoning_chain']
                    answer += f" (Reasoning: {' → '.join(chain[:3])})"
                return answer

        # TRY KNOWLEDGE GRAPH for questions
        if is_question and self.knowledge_graph:
            kg_answer = self.knowledge_graph.answer_question(input_text)
            if kg_answer and "don't" not in kg_answer.lower() and "not sure" not in kg_answer.lower():
                return kg_answer

            # Try generating description if subject exists
            if subject:
                kg_knowledge = self.knowledge_graph.get_knowledge_about(subject)
                if kg_knowledge.get('exists'):
                    description = self.knowledge_graph.generate_description(subject)
                    if "still learning" not in description.lower():
                        return description

        # Fall back to semantic memory associations
        subject_knowledge = self._get_knowledge_about(subject) if subject else {}

        # Only use reasoning if it's actually relevant to the question
        if reasoning_result and reasoning_result.get('type') == 'analogy':
            return reasoning_result['response']

        # Handle different question types
        if question_type == 'what_is':
            return self._answer_what_is(subject, subject_knowledge, vocab_size)
        elif question_type == 'what_about':
            return self._answer_what_about(subject, subject_knowledge, vocab_size)
        elif question_type == 'can_you':
            return self._answer_can_you(text_lower, vocab_size)
        elif question_type == 'how':
            return self._answer_how(subject, subject_knowledge)
        elif is_question:
            return self._answer_general_question(subject, subject_knowledge, vocab_size)
        else:
            return self._respond_to_statement(content_words, subject_knowledge, vocab_size)

    def _detect_question_type(self, text: str) -> Optional[str]:
        """Detect the type of question being asked."""
        if 'what is' in text or 'what are' in text or "what's" in text:
            return 'what_is'
        elif text.startswith('what') or 'tell me about' in text or 'know about' in text:
            return 'what_about'
        elif text.startswith('can you') or text.startswith('could you'):
            return 'can_you'
        elif text.startswith('how'):
            return 'how'
        return None

    def _handle_greeting(self, text: str) -> Optional[str]:
        """Handle greetings and common conversational phrases naturally."""
        import random

        # Normalize text
        text = text.lower().strip().rstrip('!?.')

        # Greetings
        greetings = ['hello', 'hi', 'hey', 'howdy', 'greetings', 'hiya', 'yo', 'sup', 'whats up', "what's up"]
        greeting_responses = [
            "Hello! How can I help you today?",
            "Hi there! What would you like to talk about?",
            "Hey! Nice to chat with you.",
            "Hello! What's on your mind?",
            "Hi! I'm here to help. What would you like to know?",
        ]

        # Farewells
        farewells = ['bye', 'goodbye', 'see you', 'later', 'farewell', 'take care', 'gotta go']
        farewell_responses = [
            "Goodbye! It was nice talking with you.",
            "See you later! Feel free to come back anytime.",
            "Take care! Hope to chat again soon.",
            "Bye! Thanks for the conversation.",
        ]

        # How are you
        how_are_you = ['how are you', 'how are things', "how's it going", 'how do you do', 'how you doing']
        how_responses = [
            "I'm doing well, thanks for asking! How about you?",
            "I'm good! Ready to help with whatever you need.",
            "Doing great! What can I help you with today?",
        ]

        # Thank you
        thanks = ['thank you', 'thanks', 'thx', 'appreciate it', 'ty']
        thanks_responses = [
            "You're welcome!",
            "Happy to help!",
            "Anytime!",
            "Glad I could assist!",
        ]

        # Check greetings
        for greeting in greetings:
            if text == greeting or text.startswith(greeting + ' ') or text.endswith(' ' + greeting):
                return random.choice(greeting_responses)

        # Check farewells
        for farewell in farewells:
            if farewell in text:
                return random.choice(farewell_responses)

        # Check how are you
        for phrase in how_are_you:
            if phrase in text:
                return random.choice(how_responses)

        # Check thanks
        for thank in thanks:
            if text == thank or text.startswith(thank + ' ') or text.endswith(' ' + thank):
                return random.choice(thanks_responses)

        # Nice to meet you
        if 'nice to meet' in text or 'pleased to meet' in text:
            return "Nice to meet you too! I'm AGI Brain. What would you like to explore together?"

        # My name is
        if 'my name is' in text or "i'm " in text or 'i am ' in text:
            # Try to extract name
            import re
            name_match = re.search(r"(?:my name is|i'm|i am)\s+(\w+)", text, re.I)
            if name_match:
                name = name_match.group(1).capitalize()
                return f"Nice to meet you, {name}! How can I help you today?"

        return None

    def _find_subject(self, content_words: List[str], text: str) -> Optional[str]:
        """Find the main subject/topic of the question."""
        import re

        # Clean the text for pattern matching
        clean = re.sub(r'[^\w\s]', '', text.lower())

        # Look for "about X" pattern
        if 'about ' in clean:
            idx = clean.find('about ')
            after_words = clean[idx + 6:].split()
            for word in after_words:
                if len(word) > 2 and word not in STOP_WORDS:
                    return word

        # Look for "is X" or "are X" pattern
        for pattern in ['what is ', 'what are ', 'whats ']:
            if pattern in clean:
                idx = clean.find(pattern)
                after_words = clean[idx + len(pattern):].split()
                for word in after_words:
                    if len(word) > 2 and word not in STOP_WORDS:
                        return word

        # Return first content word as fallback
        if content_words:
            return content_words[0]
        return None

    def _get_knowledge_about(self, subject: str) -> Dict:
        """Retrieve what we know about a subject from semantic memory."""
        if not subject:
            return {}

        knowledge = {
            'exists': False,
            'category': None,
            'properties': [],
            'related': [],
            'definitions': []
        }

        # Check if we know this concept
        concept = self.semantic_memory.lookup(subject)
        if concept is None:
            return knowledge

        knowledge['exists'] = True

        # Get associations
        activations = self.semantic_memory.spread_activation(subject, depth=2)

        # Categorize the associations - categories are abstract types, not specific things
        categories = ['animal', 'mammal', 'pet', 'food', 'fruit', 'vegetable', 'drink',
                     'place', 'building', 'person', 'thing', 'object', 'color', 'emotion',
                     'plant', 'vehicle', 'device', 'action', 'time', 'nature', 'liquid',
                     'furniture', 'tool', 'concept', 'number', 'weather', 'body']
        properties = ['big', 'small', 'hot', 'cold', 'fast', 'slow', 'good', 'bad',
                     'bright', 'dark', 'loud', 'quiet', 'soft', 'hard', 'sweet', 'sour',
                     'large', 'tiny', 'warm', 'cool', 'quick', 'tall', 'short', 'young',
                     'old', 'new', 'beautiful', 'wild', 'loyal', 'furry', 'wet', 'dry',
                     'clean', 'dirty', 'happy', 'sad', 'heavy', 'light', 'strong', 'weak']

        for word, strength in sorted(activations.items(), key=lambda x: -x[1])[:20]:
            if word == subject:
                continue
            # Skip stop words and very short words
            if word in STOP_WORDS or len(word) < 3:
                continue
            if word in categories:
                knowledge['category'] = word
            elif word in properties:
                knowledge['properties'].append(word)
            else:
                knowledge['related'].append(word)

        return knowledge

    def _answer_what_is(self, subject: str, knowledge: Dict, vocab_size: int) -> str:
        """Answer 'what is X' questions."""
        if not subject:
            return f"I have {vocab_size} concepts in my memory. What would you like to know about?"

        if not knowledge.get('exists'):
            return f"I don't have information about '{subject}' yet. Could you tell me about it?"

        parts = []
        if knowledge.get('category'):
            parts.append(f"{subject.capitalize()} is a type of {knowledge['category']}")

        if knowledge.get('properties'):
            props = knowledge['properties'][:2]
            if parts:
                parts.append(f"It is {' and '.join(props)}")
            else:
                parts.append(f"{subject.capitalize()} is {' and '.join(props)}")

        if knowledge.get('related'):
            related = knowledge['related'][:3]
            parts.append(f"Related concepts: {', '.join(related)}")

        if parts:
            return '. '.join(parts) + '.'
        else:
            return f"I know '{subject}' but I'm still learning its connections. Tell me more about it!"

    def _answer_what_about(self, subject: str, knowledge: Dict, vocab_size: int) -> str:
        """Answer 'tell me about X' questions."""
        if not subject:
            return f"What would you like to know about? I have {vocab_size} concepts."

        if not knowledge.get('exists'):
            return f"I haven't learned about '{subject}' yet. What can you tell me about it?"

        parts = []

        if knowledge.get('category'):
            parts.append(f"{subject.capitalize()} is a {knowledge['category']}")

        if knowledge.get('properties'):
            parts.append(f"characterized as {', '.join(knowledge['properties'][:2])}")

        if knowledge.get('related'):
            related = knowledge['related'][:4]
            parts.append(f"I associate it with: {', '.join(related)}")

        if parts:
            return '. '.join(parts) + '.'
        return f"I recognize '{subject}' but need more context. What specifically interests you?"

    def _answer_can_you(self, text: str, vocab_size: int) -> str:
        """Answer 'can you' questions."""
        if 'speak' in text or 'talk' in text:
            return f"I communicate through text. I'm learning language through neural associations - currently {vocab_size} concepts and growing!"
        elif 'learn' in text:
            return "Yes! I learn from our conversations, web pages, and YouTube videos. Each interaction strengthens my neural connections."
        elif 'think' in text or 'reason' in text:
            return "I use a reasoning engine with rules and analogies, combined with semantic memory. My understanding grows with each conversation."
        elif 'english' in text or 'language' in text:
            return f"I'm learning! I currently know {vocab_size} words and concepts. My language ability improves as I learn more associations."
        else:
            return "I can learn, reason about concepts, and have conversations. What would you like to explore?"

    def _answer_how(self, subject: str, knowledge: Dict) -> str:
        """Answer 'how' questions."""
        if not subject or not knowledge.get('exists'):
            return "That's a good question. I don't have enough information yet - could you tell me more?"

        if knowledge.get('related'):
            return f"Based on my learning, {subject} involves: {', '.join(knowledge['related'][:3])}. Would you like to know more?"
        return "I'm still building knowledge on that topic. Tell me more and I'll learn!"

    def _answer_general_question(self, subject: str, knowledge: Dict, vocab_size: int) -> str:
        """Answer other questions."""
        if subject and knowledge.get('exists'):
            related = knowledge.get('related', [])[:3]
            if related:
                return f"Regarding {subject}, I know it relates to: {', '.join(related)}."
            return f"I know about {subject}, but I'd like to learn more. What can you tell me?"
        return f"I have {vocab_size} concepts in my knowledge. Could you rephrase or give me more context?"

    def _respond_to_statement(self, content_words: List[str], knowledge: Dict, vocab_size: int) -> str:
        """Respond to statements (not questions) naturally."""
        import random

        if not content_words:
            responses = [
                "I'm listening. Tell me more!",
                "Go on, I'm interested.",
                "Please continue.",
            ]
            return random.choice(responses)

        main_word = content_words[0]

        if knowledge.get('exists'):
            # We know about this topic - respond naturally
            category = knowledge.get('category')
            related = knowledge.get('related', [])[:3]

            if category and related:
                responses = [
                    f"Interesting point about {main_word}! I know it's a {category}, related to {', '.join(related)}.",
                    f"Yes, {main_word} - that's a {category}. It makes me think of {', '.join(related)}.",
                    f"I see you're talking about {main_word}. As a {category}, it connects to {', '.join(related)}.",
                ]
            elif category:
                responses = [
                    f"Ah yes, {main_word} is a {category}. Tell me more about what you're thinking.",
                    f"I know {main_word} as a type of {category}. What else would you like to discuss?",
                    f"{main_word.capitalize()} - that's a {category}. Interesting topic!",
                ]
            elif related:
                responses = [
                    f"That's interesting about {main_word}. I associate it with {', '.join(related)}.",
                    f"I know {main_word} - it makes me think of {', '.join(related)}.",
                    f"{main_word.capitalize()} reminds me of {', '.join(related)}. What do you think?",
                ]
            else:
                responses = [
                    f"I've heard of {main_word}. Tell me more about your thoughts on it.",
                    f"Interesting! I know {main_word}. What else can you share?",
                    f"Yes, {main_word}. I'd love to learn more about what you mean.",
                ]
            return random.choice(responses)
        else:
            # New concept - respond with curiosity
            responses = [
                f"That's new to me! I just learned about '{main_word}'. Can you tell me more?",
                f"Interesting - I hadn't encountered '{main_word}' before. What else can you share about it?",
                f"I'm adding '{main_word}' to my knowledge. What makes it important to you?",
                f"'{main_word.capitalize()}' is new to me. Thanks for teaching me! What else should I know?",
            ]
            return random.choice(responses)

    def process_json(self, text: str) -> Dict:
        """
        Process input and return structured JSON response.

        Returns a rich JSON object with:
        - response: Natural language response
        - knowledge: Structured knowledge about the topic
        - reasoning: Reasoning chain used (if any)
        - confidence: Confidence in the answer
        - learned: New facts learned from input
        """
        if not self.initialized:
            self.initialize()

        import re
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        words = clean_text.split()
        content_words = [w for w in words if len(w) > 2 and w not in STOP_WORDS]

        result = {
            'input': text,
            'response': '',
            'knowledge': {},
            'reasoning': {
                'chain': [],
                'type': None,
                'confidence': 0.0
            },
            'learned': {
                'facts': 0,
                'words': 0
            },
            'metadata': {
                'vocabulary_size': self.semantic_memory.get_vocabulary_size(),
                'knowledge_facts': self.knowledge_graph.get_stats()['relationships'] if self.knowledge_graph else 0
            }
        }

        # Learn from input
        self.semantic_memory.learn_from_sentence(words)
        if self.knowledge_graph:
            facts_before = self.knowledge_graph.get_stats()['relationships']
            self.knowledge_graph.learn_from_text(text)
            facts_after = self.knowledge_graph.get_stats()['relationships']
            result['learned']['facts'] = facts_after - facts_before

        # Find subject
        subject = self._find_subject(content_words, text.lower())

        # Get structured knowledge about subject
        if subject and self.knowledge_graph:
            kg_knowledge = self.knowledge_graph.get_knowledge_about(subject)
            if kg_knowledge.get('exists'):
                result['knowledge'] = {
                    'subject': subject,
                    'categories': [cat for cat, conf in kg_knowledge.get('is_a', [])],
                    'properties': [prop for prop, conf in kg_knowledge.get('has_a', [])],
                    'capabilities': [cap for cap, conf in kg_knowledge.get('can_do', [])],
                    'locations': [loc for loc, conf in kg_knowledge.get('location', [])],
                    'causes': [eff for eff, conf in kg_knowledge.get('causes', [])],
                    'related': list(set(
                        [x for x, _ in kg_knowledge.get('similar_to', [])] +
                        [x for x, _ in kg_knowledge.get('has_parts', [])]
                    ))[:5]
                }

        # Try deductive reasoning
        is_question = '?' in text
        if is_question and hasattr(self, 'deductive_reasoner') and self.deductive_reasoner:
            reasoning_result = self.deductive_reasoner.reason(text)
            if reasoning_result:
                result['reasoning'] = {
                    'chain': reasoning_result.get('reasoning_chain', []),
                    'type': 'deductive',
                    'confidence': reasoning_result.get('confidence', 0.0)
                }
                result['response'] = reasoning_result.get('answer', '')

        # Fall back to standard response if no reasoning answer
        if not result['response']:
            activations = self._compute_activations(content_words)
            result['response'] = self._generate_response(text, content_words, activations, None)

        # Store in history
        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": result['response']})

        return result

    def speak(self, text: str) -> Optional[np.ndarray]:
        """Generate speech audio from text."""
        if self.speech_generator:
            return self.speech_generator.speak(text)
        return None

    def listen(self, audio: np.ndarray) -> str:
        """Process audio input and return transcription."""
        if self.cochlea and self.phoneme_learner:
            spikes = self.cochlea.encode_audio(audio)
            # Simple phoneme detection (would need full ASR for real transcription)
            return "[audio processed]"
        return "[audio not supported]"

    def learn_from_url(self, url: str) -> Dict:
        """Learn from a web page."""
        if not self.initialized:
            self.initialize()

        from src.interface.learning_pipeline import WebLearner

        if self.web_learner is None:
            self.web_learner = WebLearner()

        stats = self.web_learner.learn_from_url(url, self)
        return stats

    def learn_from_youtube(self, url: str) -> Dict:
        """Learn from a YouTube video."""
        if not self.initialized:
            self.initialize()

        from src.interface.learning_pipeline import YouTubeLearner

        learner = YouTubeLearner()
        stats = learner.learn_from_video(url, self)
        return stats

    def get_status(self) -> Dict:
        """Get brain status."""
        if not self.initialized:
            return {"initialized": False}

        kg_stats = self.knowledge_graph.get_stats() if self.knowledge_graph else {}

        return {
            "initialized": True,
            "neurons": self.config.n_neurons,
            "vocabulary_size": self.semantic_memory.get_vocabulary_size(),
            "concepts": len(self.semantic_memory.concepts),
            "knowledge_facts": kg_stats.get('relationships', 0),
            "knowledge_concepts": kg_stats.get('concepts', 0),
            "working_memory": self.working_memory,
            "conversations": len(self.conversation_history) // 2,
            "reasoning_enabled": self.config.enable_reasoning,
            "audio_enabled": self.config.enable_audio,
        }

    def save(self, path: str) -> bool:
        """Save brain state."""
        if not self.initialized:
            return False

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save semantic memory
        self.semantic_memory.save(str(save_path / "semantic_memory.npz"))

        # Save conversation history
        with open(save_path / "history.json", "w") as f:
            json.dump(self.conversation_history, f)

        # Save working memory and state
        state = {
            "working_memory": self.working_memory,
            "emotional_state": self.emotional_state,
            "goals": self.goals,
            "config": {
                "n_neurons": self.config.n_neurons,
                "enable_audio": self.config.enable_audio,
                "enable_reasoning": self.config.enable_reasoning,
            }
        }
        with open(save_path / "state.json", "w") as f:
            json.dump(state, f)

        # Save SNN state if exists
        if self.snn:
            self.snn.save(str(save_path / "snn_state.npz"))

        # Save knowledge graph
        if self.knowledge_graph:
            self.knowledge_graph.save(str(save_path / "knowledge_graph.json"))

        # Save language learner patterns
        if self.thought_process and hasattr(self.thought_process, 'language_learner'):
            self.thought_process.language_learner.save(save_path / "language_patterns.json")

        return True

    def load(self, path: str) -> bool:
        """Load brain state."""
        save_path = Path(path)

        if not save_path.exists():
            return False

        self.initialize()

        # Load semantic memory
        sem_path = save_path / "semantic_memory.npz"
        if sem_path.exists():
            self.semantic_memory.load(str(sem_path))

        # Load conversation history
        history_path = save_path / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.conversation_history = json.load(f)

        # Load state
        state_path = save_path / "state.json"
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            self.working_memory = state.get("working_memory", [])
            self.emotional_state = state.get("emotional_state", {"valence": 0.5, "arousal": 0.3})
            self.goals = state.get("goals", [])

        # Load SNN state
        snn_path = save_path / "snn_state.npz"
        if snn_path.exists() and self.snn:
            self.snn.load(str(snn_path))

        # Load knowledge graph
        kg_path = save_path / "knowledge_graph.json"
        if kg_path.exists() and self.knowledge_graph:
            self.knowledge_graph.load(str(kg_path))

        # Reinitialize thought process with loaded knowledge
        if self.thought_process:
            from src.cognition.thought_process import ThoughtProcess
            self.thought_process = ThoughtProcess(
                knowledge_graph=self.knowledge_graph,
                semantic_memory=self.semantic_memory,
                deductive_reasoner=self.deductive_reasoner
            )

            # Load language learner patterns
            lang_path = save_path / "language_patterns.json"
            if lang_path.exists() and hasattr(self.thought_process, 'language_learner'):
                self.thought_process.language_learner.load(lang_path)

        return True


class ReasoningEngine:
    """
    Reasoning module implementing:
    - Working memory
    - Rule application
    - Analogy finding
    """

    def __init__(self, semantic_memory):
        self.semantic_memory = semantic_memory
        self.rules = self._load_default_rules()
        self.working_memory = []

    def _load_default_rules(self) -> List[Dict]:
        """Load default reasoning rules."""
        return [
            # Category rules
            {"if": ["cat"], "then": "animal", "confidence": 0.9},
            {"if": ["dog"], "then": "animal", "confidence": 0.9},
            {"if": ["bird"], "then": "animal", "confidence": 0.9},
            {"if": ["animal", "pet"], "then": "companion", "confidence": 0.7},

            # Property rules
            {"if": ["fire"], "then": "hot", "confidence": 0.95},
            {"if": ["ice"], "then": "cold", "confidence": 0.95},
            {"if": ["sun"], "then": "bright", "confidence": 0.9},

            # Action rules
            {"if": ["bird", "fly"], "then": "sky", "confidence": 0.8},
            {"if": ["fish", "swim"], "then": "water", "confidence": 0.9},
        ]

    def process(self, text: str, activations: Dict[str, float]) -> Optional[Dict]:
        """Apply reasoning to input - only when actually relevant."""
        words = set(text.lower().split())
        text_lower = text.lower()

        # Only apply rules if the user is directly asking about the rule's subject
        # Don't apply rules just because a random word appears
        for rule in self.rules:
            rule_words = set(rule["if"])
            # ALL rule conditions must be in the actual input text (not activations)
            if rule_words.issubset(words):
                # And the question should be about these concepts
                if any(f"about {w}" in text_lower or f"is {w}" in text_lower or
                       f"what {w}" in text_lower or text_lower.startswith(w)
                       for w in rule["if"]):
                    return {
                        "type": "rule_application",
                        "rule": rule,
                        "response": f"I know that {' and '.join(rule['if'])} relates to {rule['then']}."
                    }

        # Only show analogies if user explicitly asks about comparisons or relationships
        if 'like' in words or 'similar' in words or 'compare' in words or 'analogy' in words:
            analogy = self._find_analogy(list(words), activations)
            if analogy:
                return {
                    "type": "analogy",
                    "analogy": analogy,
                    "response": f"Here's an analogy: {analogy['source']} is to {analogy['source_rel']} as {analogy['target']} is to {analogy['target_rel']}."
                }

        return None

    def _find_analogy(self, words: List[str], activations: Dict[str, float]) -> Optional[Dict]:
        """Find analogical relationships."""
        # Simple analogy patterns
        analogies = [
            {"source": "cat", "source_rel": "meow", "target": "dog", "target_rel": "bark"},
            {"source": "bird", "source_rel": "fly", "target": "fish", "target_rel": "swim"},
            {"source": "day", "source_rel": "sun", "target": "night", "target_rel": "moon"},
            {"source": "hot", "source_rel": "fire", "target": "cold", "target_rel": "ice"},
        ]

        for analogy in analogies:
            if analogy["source"] in words or analogy["target"] in words:
                return analogy

        return None


class SimplifiedSNN:
    """
    Simplified Spiking Neural Network for web deployment.

    Provides basic SNN simulation without GPU requirements.
    """

    def __init__(self, n_neurons: int = 100_000):
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(42)

        # Sparse connectivity (0.1%)
        self.n_connections = int(n_neurons * n_neurons * 0.001)

        # Initialize sparse weight matrix
        self.pre_indices = self.rng.integers(0, n_neurons, self.n_connections)
        self.post_indices = self.rng.integers(0, n_neurons, self.n_connections)
        self.weights = self.rng.standard_normal(self.n_connections).astype(np.float32) * 0.1

        # Neuron states
        self.potentials = np.zeros(n_neurons, dtype=np.float32)
        self.spike_counts = np.zeros(n_neurons, dtype=np.int32)

        # Word to neuron mapping
        self.word_neurons = {}
        self.next_neuron = 0

    def process_input(self, words: List[str], activations: Dict[str, float]):
        """Process input through the network."""
        # Map words to neurons
        for word in words:
            if word not in self.word_neurons and self.next_neuron < self.n_neurons:
                self.word_neurons[word] = self.next_neuron
                self.next_neuron += 1

        # Activate word neurons
        for word in words:
            if word in self.word_neurons:
                idx = self.word_neurons[word]
                self.potentials[idx] += 1.0
                self.spike_counts[idx] += 1

        # Simple plasticity: strengthen connections between active neurons
        active = np.where(self.potentials > 0.5)[0]
        if len(active) > 1:
            # STDP-like update
            for i in range(len(self.pre_indices)):
                if self.pre_indices[i] in active and self.post_indices[i] in active:
                    self.weights[i] += 0.01

        # Decay potentials
        self.potentials *= 0.9

    def save(self, path: str):
        """Save SNN state."""
        np.savez_compressed(
            path,
            weights=self.weights,
            potentials=self.potentials,
            spike_counts=self.spike_counts,
            word_neurons=self.word_neurons,
            next_neuron=self.next_neuron
        )

    def load(self, path: str):
        """Load SNN state."""
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.potentials = data['potentials']
        self.spike_counts = data['spike_counts']
        self.word_neurons = data['word_neurons'].item()
        self.next_neuron = int(data['next_neuron'])


# Stop words for filtering
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
    'until', 'while', 'although', 'though', 'after', 'before', 'when',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
    'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
    'that', 'these', 'those', 'am', 'isn', 'aren', 'wasn', 'weren', 'hasn',
    'haven', 'hadn', 'doesn', 'don', 'didn', 'won', 'wouldn', 'couldn',
    'shouldn', 'mightn', 'mustn', 'about', 'tell', 'know', 'think', 'say',
}


if __name__ == "__main__":
    # Test the full brain
    print("Testing Full AGI Brain...")

    config = BrainConfig(n_neurons=10_000, enable_audio=False)
    brain = FullAGIBrain(config)
    brain.initialize()

    # Test conversation
    tests = [
        "Hello, I am testing the brain",
        "What is a cat?",
        "Dogs are loyal animals",
        "Can birds fly?",
        "Tell me about the sun",
    ]

    for text in tests:
        print(f"\nUser: {text}")
        response = brain.process(text)
        print(f"Brain: {response}")

    print(f"\nStatus: {brain.get_status()}")
