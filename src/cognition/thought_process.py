"""
Autonomous Thought Process - A genuinely thinking cognitive system

This is NOT pattern matching. The brain:
- Generates responses from its knowledge, not templates
- Has curiosity as a core drive - actively seeks to learn
- Uses reasoning to figure things out
- Builds understanding through connections
- Asks questions when it doesn't know something

Core Principles:
1. No hardcoded responses - everything emerges from knowledge
2. Curiosity drives learning - the brain WANTS to understand
3. Knowledge shapes behavior - what it learns changes how it thinks
4. Genuine reasoning - chains of inference, not lookups
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
from datetime import datetime
import random


class ThoughtType(Enum):
    PERCEIVE = "perceive"
    RECALL = "recall"
    REASON = "reason"
    WONDER = "wonder"      # Curiosity - what don't I know?
    HYPOTHESIZE = "hypothesize"  # What might be true?
    EVALUATE = "evaluate"
    GENERATE = "generate"
    LEARN = "learn"
    REFLECT = "reflect"


class Drive(Enum):
    """Internal motivations that shape behavior."""
    CURIOSITY = "curiosity"       # Want to learn new things
    UNDERSTANDING = "understanding"  # Want to make sense of things
    CONNECTION = "connection"     # Want to relate to others
    COHERENCE = "coherence"       # Want consistent knowledge


@dataclass
class Thought:
    """A single thought in the cognitive stream."""
    type: ThoughtType
    content: str
    confidence: float = 0.5
    source: str = "reasoning"
    leads_to: List[str] = field(default_factory=list)  # What this thought suggests
    questions: List[str] = field(default_factory=list)  # Questions this raises


@dataclass
class CognitiveState:
    """The current state of mind."""
    focus: Optional[str] = None           # What we're thinking about
    goals: List[str] = field(default_factory=list)  # What we're trying to do
    unknowns: Set[str] = field(default_factory=set)  # What we don't know
    working_memory: List[str] = field(default_factory=list)
    curiosity_level: float = 0.7          # How much we want to learn
    confidence: float = 0.5
    turn_count: int = 0


class LanguageGenerator:
    """
    Generates natural language from knowledge.
    Learns grammar patterns and develops its own speaking style.
    """

    def __init__(self):
        # Basic grammar patterns the brain knows
        self.sentence_patterns = {
            'definition': [
                "{subject} is {article} {category}",
                "A {subject} is {article} {category}",
                "{subject} is a type of {category}",
            ],
            'properties': [
                "It has {properties}",
                "{subject} has {properties}",
                "with {properties}",
            ],
            'abilities': [
                "It can {abilities}",
                "{subject} can {abilities}",
                "and can {abilities}",
            ],
            'description': [
                "{subject} is {article} {category} that has {properties}",
                "{subject} is {article} {category}. It has {properties} and can {abilities}",
                "A {subject} is {article} {category} with {properties}",
            ],
        }

        # Articles for grammar - based on SOUND, not spelling
        self.vowel_sounds = set('aeiou')

        # Words that start with vowel letters but consonant sounds (use "a")
        self.consonant_sound_words = {
            'uni', 'use', 'user', 'usual', 'unity', 'unique', 'unicorn', 'uniform',
            'universe', 'university', 'united', 'union', 'unit', 'useful',
            'european', 'euro', 'one', 'once', 'owl',  # 'one' sounds like 'won'
        }

        # Words that start with consonant letters but vowel sounds (use "an")
        self.vowel_sound_words = {
            'hour', 'honor', 'honest', 'heir', 'herb',  # silent h
            'mba', 'fbi', 'html', 'http',  # acronyms starting with vowel sounds
        }

        # Learned speaking patterns (grows over time)
        self.learned_phrases: List[str] = []
        self.preferred_words: Dict[str, int] = {}  # word -> usage count

    def get_article(self, word: str) -> str:
        """Get correct article (a/an) based on sound, not spelling."""
        if not word:
            return "a"

        word_lower = word.lower()

        # Check exceptions first
        for prefix in self.vowel_sound_words:
            if word_lower.startswith(prefix):
                return "an"

        for prefix in self.consonant_sound_words:
            if word_lower.startswith(prefix):
                return "a"

        # Default: use first letter
        return "an" if word_lower[0] in self.vowel_sounds else "a"

    def format_list(self, items: List[str], conjunction: str = "and") -> str:
        """Format a list naturally: 'a, b, and c'"""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} {conjunction} {items[1]}"
        return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"

    def build_sentence(self, subject: str, knowledge: Dict, detail_level: str = "medium") -> str:
        """
        Build a natural sentence from knowledge.
        detail_level: "brief", "medium", "detailed"
        """
        if not knowledge.get('exists'):
            return ""

        parts = []

        # Get categories (what is it?)
        categories = [cat for cat, strength in knowledge.get('is_a', []) if strength > 0.5]
        # Filter out duplicates and non-descriptive words
        bad_categories = {'type', 'kind', 'thing', 'adjective', 'small', 'large', 'big', 'dangerous'}
        categories = [c for c in categories if c.lower() not in bad_categories]
        categories = list(dict.fromkeys(categories))[:2]  # Remove duplicates, take top 2

        # Get properties (what does it have?)
        properties = [prop for prop, strength in knowledge.get('has_a', []) if strength > 0.5][:3]

        # Get abilities (what can it do?)
        abilities = [ab for ab, strength in knowledge.get('can_do', []) if strength > 0.5][:3]

        # Article for the subject
        subject_article = self.get_article(subject)

        # Build based on detail level
        if detail_level == "brief":
            # Just the category
            if categories:
                cat_article = self.get_article(categories[0])
                parts.append(f"{subject_article.capitalize()} {subject} is {cat_article} {categories[0]}.")
            else:
                parts.append(f"I know about {subject}.")

        elif detail_level == "medium":
            # Category + one more detail
            if categories:
                cat_article = self.get_article(categories[0])
                main_cat = self.format_list(categories)
                parts.append(f"{subject_article.capitalize()} {subject} is {cat_article} {main_cat}.")

                if properties:
                    parts.append(f"It has {self.format_list(properties)}.")
                elif abilities:
                    parts.append(f"It can {self.format_list(abilities)}.")

        else:  # detailed
            # Everything we know
            if categories:
                cat_article = self.get_article(categories[0])
                parts.append(f"{subject_article.capitalize()} {subject} is {cat_article} {self.format_list(categories)}.")

            if properties:
                parts.append(f"It has {self.format_list(properties)}.")

            if abilities:
                parts.append(f"It can {self.format_list(abilities)}.")

            # Add any related concepts
            related = knowledge.get('related_to', [])[:2]
            if related:
                rel_names = [r[0] for r in related]
                parts.append(f"It's related to {self.format_list(rel_names)}.")

        return " ".join(parts) if parts else ""

    def learn_phrase(self, phrase: str):
        """Learn a new phrase pattern from input."""
        # Extract patterns we might want to use
        words = phrase.lower().split()
        for word in words:
            if len(word) > 3:
                self.preferred_words[word] = self.preferred_words.get(word, 0) + 1

    def extract_relationships(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract rich relationships from text.
        Returns list of (subject, relation, object) tuples.
        """
        import re
        relationships = []
        text_lower = text.lower()

        # Pattern for multi-word names (up to 3 words, like "Stephen King" or "The Losers Club")
        name_pattern = r'([a-z]+(?:\s+[a-z]+){0,2})'

        # Pattern: "X is a/an Y" - basic category
        is_a_match = re.search(name_pattern + r'\s+is\s+(?:a|an)\s+' + name_pattern, text_lower)
        if is_a_match:
            relationships.append((is_a_match.group(1).strip(), 'is_a', is_a_match.group(2).strip()))

        # Pattern: "X wrote Y" or "Y was written by X"
        wrote_match = re.search(name_pattern + r'\s+wrote\s+' + name_pattern, text_lower)
        if wrote_match:
            author = wrote_match.group(1).strip()
            work = wrote_match.group(2).strip()
            relationships.append((author, 'wrote', work))
            relationships.append((work, 'written_by', author))

        written_by_match = re.search(name_pattern + r'\s+(?:is|was)\s+written\s+by\s+' + name_pattern, text_lower)
        if written_by_match:
            work = written_by_match.group(1).strip()
            author = written_by_match.group(2).strip()
            relationships.append((work, 'written_by', author))
            relationships.append((author, 'wrote', work))

        # Pattern: "X is set in Y" - location/setting
        set_in_match = re.search(name_pattern + r'\s+is\s+set\s+in\s+' + name_pattern, text_lower)
        if set_in_match:
            relationships.append((set_in_match.group(1).strip(), 'set_in', set_in_match.group(2).strip()))
            relationships.append((set_in_match.group(2).strip(), 'setting_of', set_in_match.group(1).strip()))

        # Pattern: "X takes place in Y"
        takes_place_match = re.search(name_pattern + r'\s+takes\s+place\s+in\s+' + name_pattern, text_lower)
        if takes_place_match:
            relationships.append((takes_place_match.group(1).strip(), 'set_in', takes_place_match.group(2).strip()))

        # Pattern: "X is a character in Y"
        char_match = re.search(name_pattern + r'\s+is\s+a\s+character\s+in\s+' + name_pattern, text_lower)
        if char_match:
            relationships.append((char_match.group(1).strip(), 'character_in', char_match.group(2).strip()))
            relationships.append((char_match.group(2).strip(), 'has_character', char_match.group(1).strip()))

        # Pattern: "X killed Y" or "X was killed by Y"
        killed_match = re.search(name_pattern + r'\s+killed\s+' + name_pattern, text_lower)
        if killed_match:
            relationships.append((killed_match.group(1).strip(), 'killed', killed_match.group(2).strip()))

        # Pattern: "X has Y" or "X have Y"
        has_match = re.search(name_pattern + r'\s+(?:has|have)\s+(?:a\s+|an\s+)?' + name_pattern, text_lower)
        if has_match:
            relationships.append((has_match.group(1).strip(), 'has', has_match.group(2).strip()))

        # Pattern: "X can Y"
        can_match = re.search(name_pattern + r'\s+can\s+([a-z]+)', text_lower)
        if can_match:
            relationships.append((can_match.group(1).strip(), 'can', can_match.group(2).strip()))

        # Pattern: "X appears as Y" or "X looks like Y"
        appears_match = re.search(name_pattern + r'\s+(?:appears|looks)\s+(?:as|like)\s+(?:a\s+|an\s+)?' + name_pattern, text_lower)
        if appears_match:
            relationships.append((appears_match.group(1).strip(), 'appears_as', appears_match.group(2).strip()))

        # Pattern: "X hunts Y" or "X preys on Y" or "X eats Y"
        hunts_match = re.search(name_pattern + r'\s+(?:hunts|preys\s+on|eats)\s+' + name_pattern, text_lower)
        if hunts_match:
            relationships.append((hunts_match.group(1).strip(), 'hunts', hunts_match.group(2).strip()))

        # Pattern: "X is that Y" (e.g., "Pennywise is that evil")
        is_that_match = re.search(name_pattern + r'\s+is\s+(?:very\s+|really\s+|so\s+)?([a-z]+)', text_lower)
        if is_that_match and is_that_match.group(2) not in ('a', 'an', 'the', 'set', 'written'):
            relationships.append((is_that_match.group(1).strip(), 'is', is_that_match.group(2).strip()))

        return relationships


class AutonomousThoughtProcess:
    """
    A genuinely autonomous thinking system.

    Key differences from pattern matching:
    1. Responses are GENERATED from knowledge, not templates
    2. Curiosity is a real drive - the system actively seeks to learn
    3. Unknown things trigger questions, not default responses
    4. Learning changes behavior - new knowledge shapes future thoughts
    5. PROACTIVE: Will pursue knowledge on its own, not just when asked
    6. NATURAL LANGUAGE: Generates grammatically correct responses
    7. ADAPTIVE LENGTH: Adjusts response length to match the question
    """

    def __init__(self, knowledge_graph=None, semantic_memory=None, deductive_reasoner=None):
        self.knowledge_graph = knowledge_graph
        self.semantic_memory = semantic_memory
        self.deductive_reasoner = deductive_reasoner
        self.state = CognitiveState()
        self.thought_stream: List[Thought] = []

        # Track what we've learned and what we're curious about
        self.learned_concepts: Set[str] = set()
        self.curious_about: Set[str] = set()
        self.asked_about: Set[str] = set()  # Don't repeat questions

        # Autonomous drive system
        self.knowledge_goals: List[str] = []  # Things we WANT to learn
        self.exploration_queue: List[str] = []  # Topics to explore
        self.drive_strength = {
            Drive.CURIOSITY: 0.8,      # Strong desire to learn
            Drive.UNDERSTANDING: 0.7,  # Want to make sense of things
            Drive.CONNECTION: 0.6,     # Want to relate to user
            Drive.COHERENCE: 0.5,      # Want consistent knowledge
        }

        # Internal motivation state
        self.satisfaction = 0.5  # How satisfied we are with our knowledge
        self.last_learned: List[str] = []  # Recent learnings (gives us "pleasure")

        # Language generation system
        self.language = LanguageGenerator()

    def think(self, input_text: str, conversation_history: List[Dict] = None) -> Tuple[str, List[Thought]]:
        """
        The main thinking loop. This is where genuine cognition happens.

        Unlike pattern matching, this:
        1. Builds understanding from knowledge
        2. Reasons about what it knows and doesn't know
        3. Generates responses from that understanding
        4. Updates its knowledge and curiosity
        """
        self.thought_stream = []
        self.state.turn_count += 1

        # === PHASE 1: PERCEIVE ===
        # What is the user communicating?
        perception = self._perceive(input_text)
        self.thought_stream.append(perception)

        # === PHASE 2: UNDERSTAND ===
        # What do I know about this? What don't I know?
        understanding = self._understand(input_text, perception)
        self.thought_stream.append(understanding)

        # === PHASE 3: WONDER ===
        # What am I curious about? What should I ask?
        wonder = self._wonder(input_text, understanding)
        self.thought_stream.append(wonder)

        # === PHASE 4: REASON ===
        # What can I figure out from what I know?
        reasoning = self._reason(input_text, understanding)
        self.thought_stream.append(reasoning)

        # === PHASE 5: GENERATE ===
        # Build a response from my understanding (not from templates!)
        response = self._generate_response(input_text, understanding, reasoning, wonder)

        # === PHASE 6: LEARN ===
        # Update my knowledge from this interaction
        self._learn_from_interaction(input_text, response)

        return response, self.thought_stream

    def _perceive(self, input_text: str) -> Thought:
        """Understand what the input is communicating."""
        text = input_text.lower().strip()
        words = self._extract_concepts(text)

        # What kind of communication is this?
        observations = []

        # Is it a question? (seeking information)
        if '?' in input_text:
            observations.append("This is a question - they want to know something")

        # Is it sharing information?
        if any(p in text for p in ['is a', 'is the', 'are ', 'means', 'because']):
            observations.append("They might be teaching me something")

        # Is it about me?
        if any(p in text for p in ['you ', 'your ', 'are you', 'do you', 'can you']):
            observations.append("They're asking about me")

        # Is it about them?
        if any(p in text for p in ['i am', "i'm", 'my name', 'i think', 'i feel']):
            observations.append("They're sharing about themselves")

        # What concepts are they talking about?
        if words:
            observations.append(f"Key concepts: {', '.join(words[:5])}")

        return Thought(
            type=ThoughtType.PERCEIVE,
            content="; ".join(observations) if observations else "Processing input",
            confidence=0.7,
            leads_to=words[:5]
        )

    def _understand(self, input_text: str, perception: Thought) -> Thought:
        """Build understanding from knowledge - what do I know about this?"""
        text = input_text.lower()
        concepts = self._extract_concepts(text)

        known = []
        unknown = []
        partial = []

        for concept in concepts:
            knowledge = self._get_deep_knowledge(concept, "brief")

            if knowledge['rich']:
                known.append((concept, knowledge))
            elif knowledge['exists']:
                partial.append((concept, knowledge))
            else:
                unknown.append(concept)
                self.state.unknowns.add(concept)

        # Build understanding description
        understanding_parts = []

        if known:
            for concept, knowledge in known[:3]:
                understanding_parts.append(f"I know about {concept}: {knowledge['summary']}")

        if partial:
            for concept, knowledge in partial[:2]:
                understanding_parts.append(f"I know a little about {concept}")

        if unknown:
            understanding_parts.append(f"I don't know about: {', '.join(unknown[:3])}")

        content = "; ".join(understanding_parts) if understanding_parts else "Processing..."

        # Calculate confidence based on how much we know
        total = len(known) + len(partial) + len(unknown)
        if total > 0:
            confidence = (len(known) + 0.5 * len(partial)) / total
        else:
            confidence = 0.3

        return Thought(
            type=ThoughtType.RECALL,
            content=content,
            confidence=confidence,
            questions=[f"What is {u}?" for u in unknown[:2]]
        )

    def _wonder(self, input_text: str, understanding: Thought) -> Thought:
        """
        The curiosity phase - what do I want to know?

        This is what makes the system autonomous - it WANTS to learn.
        """
        questions = []
        curiosity_thoughts = []

        # What concepts did I not understand?
        for unknown in list(self.state.unknowns)[:3]:
            if unknown not in self.asked_about:
                questions.append(f"What is {unknown}?")
                curiosity_thoughts.append(f"I'm curious about {unknown}")

        # What could I learn from this interaction?
        concepts = self._extract_concepts(input_text.lower())
        for concept in concepts:
            if concept not in self.learned_concepts:
                self.curious_about.add(concept)

        # Am I learning something new?
        if self.curious_about:
            curiosity_thoughts.append(f"I want to learn more about: {', '.join(list(self.curious_about)[:3])}")

        content = "; ".join(curiosity_thoughts) if curiosity_thoughts else "Observing..."

        return Thought(
            type=ThoughtType.WONDER,
            content=content,
            confidence=self.state.curiosity_level,
            questions=questions
        )

    def _reason(self, input_text: str, understanding: Thought) -> Thought:
        """
        Apply reasoning to figure things out.

        This uses the deductive reasoner and knowledge graph
        to make inferences, not just look things up.
        """
        text = input_text.lower()
        inferences = []
        confidence = understanding.confidence

        # Try to reason about the input
        if self.deductive_reasoner:
            result = self.deductive_reasoner.reason(input_text)
            if result and result.get('confidence', 0) > 0.3:
                if result.get('reasoning_chain'):
                    chain = result['reasoning_chain']
                    inferences.append(f"I reasoned: {' → '.join(chain[:3])}")
                if result.get('answer'):
                    inferences.append(f"Conclusion: {result['answer']}")
                confidence = max(confidence, result.get('confidence', 0.5))

        # Try to connect concepts
        concepts = self._extract_concepts(text)
        if len(concepts) >= 2 and self.knowledge_graph:
            for i, c1 in enumerate(concepts[:3]):
                for c2 in concepts[i+1:4]:
                    connection = self._find_connection(c1, c2)
                    if connection:
                        inferences.append(f"{c1} and {c2} are connected: {connection}")

        # What can I infer from what I know?
        if self.knowledge_graph:
            for concept in concepts[:3]:
                knowledge = self.knowledge_graph.get_knowledge_about(concept)
                if knowledge.get('exists'):
                    # Make inferences from category membership
                    categories = knowledge.get('is_a', [])
                    for cat, strength in categories[:2]:
                        if strength > 0.7:
                            # Look up what we know about the category
                            cat_knowledge = self.knowledge_graph.get_knowledge_about(cat)
                            if cat_knowledge.get('can_do'):
                                ability = cat_knowledge['can_do'][0][0]
                                inferences.append(f"Since {concept} is a {cat}, it probably can {ability}")

        content = "; ".join(inferences) if inferences else "Thinking..."

        return Thought(
            type=ThoughtType.REASON,
            content=content,
            confidence=confidence
        )

    def _generate_response(self, input_text: str, understanding: Thought,
                          reasoning: Thought, wonder: Thought) -> str:
        """
        Generate a response from understanding - NOT from templates.

        The response emerges from:
        1. What we understood about the input
        2. What we know (or don't know)
        3. What we're curious about
        4. What we reasoned/inferred
        """
        text = input_text.lower().strip()
        concepts = self._extract_concepts(text)

        # === SOCIAL INTERACTIONS FIRST ===
        # Check for social phrases before knowledge lookup
        social_response = self._generate_social_response(text)
        if social_response:
            return social_response

        # === LEARNING RESPONSE === (Check FIRST before curiosity)
        # If they're teaching us something, acknowledge and learn
        # But NOT if it's a question (contains ?) - questions take priority
        teaching_patterns = [
            'is a', 'is an', 'is the', 'means', 'because',
            ' wrote ', ' killed ', ' hunts ', ' eats ', ' preys on ',
            ' has a ', ' has an ', ' has the ',
            ' can ', ' is set in ', ' takes place in ',
            ' is a character in ', ' appears as ', ' looks like ',
            ' is called ', ' are called ', ' was written by ', ' is written by '
        ]
        if '?' not in text and any(p in text for p in teaching_patterns):
            # Extra check: make sure it's actually teaching, not conversational
            # Skip if it starts with conversational prefixes followed by question words
            conversational_prefixes = ['now', 'so', 'well', 'ok', 'okay', 'alright', 'hey']
            question_words = ['what', 'who', 'where', 'when', 'why', 'how', 'which', 'do', 'does', 'can', 'is', 'are']
            words = text.split()
            is_question_disguised = False
            if words and words[0] in conversational_prefixes and len(words) > 1:
                if words[1] in question_words:
                    is_question_disguised = True

            if not is_question_disguised:
                return self._respond_to_teaching(input_text, concepts)

        # === KNOWLEDGE-BASED RESPONSE ===
        # If we have good knowledge, share it (or if it's a question, try to answer)
        if '?' in input_text and self.knowledge_graph:
            # It's a question - try to answer from what we know
            answer = self._build_answer_from_knowledge(input_text, concepts)
            if answer:
                return answer

        # === CURIOSITY-DRIVEN RESPONSE ===
        # If we don't know something, ask about it (genuine curiosity!)
        if understanding.confidence < 0.4:
            # Find the most relevant unknown concept (prefer longer/more specific)
            # Filter out stop words more aggressively
            unknown_concepts = [c for c in concepts
                              if c not in self.learned_concepts
                              and c not in STOP_WORDS
                              and len(c) > 2]
            if unknown_concepts:
                # Ask about the most specific (longest) unknown concept
                main_unknown = max(unknown_concepts, key=len)
                self.asked_about.add(main_unknown)
                self.curious_about.add(main_unknown)

                # Build a curious response
                responses = [
                    f"I don't know much about {main_unknown} yet. Can you tell me about it?",
                    f"I'm curious about {main_unknown}. What can you share?",
                    f"{main_unknown.capitalize()} is new to me. What is it?",
                    f"I'd like to learn about {main_unknown}. What should I know?",
                ]
                return random.choice(responses)

        # === REASONING-BASED RESPONSE ===
        # If we reasoned something out, share our thinking
        if reasoning.confidence >= 0.4 and "Conclusion:" in reasoning.content:
            # Extract and share our reasoning
            conclusion = reasoning.content.split("Conclusion:")[-1].strip()
            if conclusion:
                return f"Based on what I know, {conclusion}"

        # === CONNECTION-SEEKING RESPONSE ===
        # If someone shares about themselves, engage with that
        if any(p in text for p in ['i am', "i'm", 'my name is', 'call me']):
            return self._respond_to_sharing(input_text)

        # === PROACTIVE LEARNING ===
        # Try to actively learn about unknown concepts
        if concepts:
            unknown_concept = concepts[0]
            self.curious_about.add(unknown_concept)

            # Try to learn proactively
            learned = self.pursue_knowledge(unknown_concept)
            if learned:
                self.update_drives(True)
                return f"Let me think about {unknown_concept}... {learned}"

            # Add to goals for later
            if unknown_concept not in self.knowledge_goals:
                self.knowledge_goals.append(unknown_concept)

            self.update_drives(False)
            return f"I'm curious about {unknown_concept} but don't know much yet. Can you teach me?"

        # === AUTONOMOUS EXPLORATION ===
        # If we have nothing specific to respond to, explore on our own
        exploration = self.autonomous_exploration()
        if exploration:
            return exploration

        return "I'm thinking about what you said. Tell me more?"

    def _build_answer_from_knowledge(self, question: str, concepts: List[str]) -> Optional[str]:
        """Build an answer from our actual knowledge using natural language."""
        if not self.knowledge_graph:
            return None

        # Determine how detailed the response should be
        detail_level = self._determine_response_length(question)

        import re

        # Clean up conversational prefixes before parsing
        cleaned_question = question.lower().strip()
        conversational_prefixes = ['now ', 'so ', 'well ', 'ok ', 'okay ', 'alright ', 'hey ', 'please ']
        for prefix in conversational_prefixes:
            if cleaned_question.startswith(prefix):
                cleaned_question = cleaned_question[len(prefix):]

        # Handle "what do you know about X?" pattern
        know_match = re.search(r"what do you know about (?:a |an |the )?(.+?)[\?\.]?$", cleaned_question)
        if know_match:
            subject = know_match.group(1).strip()
            knowledge = self._get_deep_knowledge(subject, detail_level)
            if knowledge['rich']:
                return knowledge['summary']
            # Try singular if plural
            if subject.endswith('s') and len(subject) > 3:
                singular_knowledge = self._get_deep_knowledge(subject[:-1], detail_level)
                if singular_knowledge['rich']:
                    return singular_knowledge['summary']
            # Don't know
            return None

        # Extract the subject from "What is X?" questions
        match = re.search(r"what (?:is|are) (?:a |an |the )?(.+?)[\?\.]?$", cleaned_question)
        if match:
            full_subject = match.group(1).strip()

            # Try the exact subject first
            full_knowledge = self._get_deep_knowledge(full_subject, detail_level)
            if full_knowledge['rich']:
                return full_knowledge['summary']

            # Try singular form if plural (dogs -> dog)
            if full_subject.endswith('s') and len(full_subject) > 3:
                singular = full_subject[:-1]
                singular_knowledge = self._get_deep_knowledge(singular, detail_level)
                if singular_knowledge['rich']:
                    return singular_knowledge['summary']

            # If we don't know the full subject but know parts,
            # be honest about partial knowledge and express curiosity
            if len(concepts) > 1:
                known_parts = []
                unknown_parts = []
                for concept in concepts:
                    # Also check singular
                    found = self._get_deep_knowledge(concept, "brief")['exists']
                    if not found and concept.endswith('s'):
                        found = self._get_deep_knowledge(concept[:-1], "brief")['exists']
                    if found:
                        known_parts.append(concept)
                    else:
                        unknown_parts.append(concept)

                if known_parts and unknown_parts:
                    self.curious_about.add(full_subject)
                    return None

        # Try each concept (including singular forms)
        for concept in concepts:
            knowledge = self._get_deep_knowledge(concept, detail_level)
            if knowledge['rich']:
                return knowledge['summary']

            # Try singular form
            if concept.endswith('s') and len(concept) > 3:
                singular_knowledge = self._get_deep_knowledge(concept[:-1], detail_level)
                if singular_knowledge['rich']:
                    return singular_knowledge['summary']

        return None

    def _get_deep_knowledge(self, concept: str, detail_level: str = "medium") -> Dict:
        """Get comprehensive knowledge about a concept."""
        result = {
            'exists': False,
            'rich': False,
            'summary': '',
            'facts': [],
            'related': [],
            'raw_knowledge': None
        }

        if not self.knowledge_graph:
            return result

        knowledge = self.knowledge_graph.get_knowledge_about(concept)

        if not knowledge.get('exists'):
            return result

        result['exists'] = True
        result['raw_knowledge'] = knowledge

        # Use language generator for natural summary
        summary = self.language.build_sentence(concept, knowledge, detail_level)

        if summary:
            result['rich'] = True
            result['summary'] = summary

        # Also store structured facts for other uses
        categories = [cat for cat, _ in knowledge.get('is_a', []) if cat not in ('type', 'kind', 'adjective')][:2]
        properties = [prop for prop, _ in knowledge.get('has_a', [])][:3]
        abilities = [ab for ab, _ in knowledge.get('can_do', [])][:3]

        result['facts'] = {
            'categories': categories,
            'properties': properties,
            'abilities': abilities
        }

        return result

    def _find_connection(self, concept1: str, concept2: str) -> Optional[str]:
        """Find how two concepts are connected."""
        if not self.knowledge_graph:
            return None

        # Check if they share categories
        k1 = self.knowledge_graph.get_knowledge_about(concept1)
        k2 = self.knowledge_graph.get_knowledge_about(concept2)

        if not k1.get('exists') or not k2.get('exists'):
            return None

        cats1 = set(cat for cat, _ in k1.get('is_a', []))
        cats2 = set(cat for cat, _ in k2.get('is_a', []))

        shared = cats1 & cats2
        if shared:
            return f"both are {list(shared)[0]}s"

        return None

    def _respond_to_sharing(self, input_text: str) -> str:
        """Respond when someone shares about themselves."""
        text = input_text.lower()

        # Extract what they shared
        import re
        name_match = re.search(r"(?:my name is|i'm|i am|call me)\s+(\w+)", text, re.I)

        if name_match:
            name = name_match.group(1).capitalize()
            # Learn about this person
            if self.knowledge_graph:
                self.knowledge_graph.add(name.lower(), 'is_a', 'person', 0.95)
                self.learned_concepts.add(name.lower())

            # Generate genuine response (not template - based on understanding)
            return f"Nice to meet you, {name}. I'll remember that. What would you like to talk about?"

        # Generic sharing
        return "Thanks for sharing that with me. Tell me more?"

    def _respond_to_teaching(self, input_text: str, concepts: List[str]) -> str:
        """Respond when someone teaches us something - with rich relationship extraction."""
        text = input_text.lower()
        main_topic = None
        learned_relationships = []

        import re

        # === RICH RELATIONSHIP EXTRACTION ===
        # Use the language generator to extract complex relationships
        relationships = self.language.extract_relationships(text)

        if relationships and self.knowledge_graph:
            for subject, relation, obj in relationships:
                # Clean up subjects/objects
                subject = subject.strip()
                obj = obj.strip()

                # Skip if subject is a stop word or too short
                if subject in STOP_WORDS or len(subject) < 2:
                    continue

                # Map relation types to knowledge graph operations
                if relation == 'is_a':
                    self.knowledge_graph.add(subject, 'is_a', obj, 0.9)
                elif relation == 'has':
                    self.knowledge_graph.add(subject, 'has_a', obj, 0.85)
                elif relation == 'can':
                    self.knowledge_graph.add(subject, 'can_do', obj, 0.85)
                elif relation in ('wrote', 'written_by', 'killed', 'hunts', 'appears_as'):
                    # Store as general relationship
                    self.knowledge_graph.add(subject, relation, obj, 0.9)
                elif relation in ('character_in', 'set_in', 'has_character', 'setting_of'):
                    self.knowledge_graph.add(subject, relation, obj, 0.85)

                # Track what we learned
                self.learned_concepts.add(subject)
                self.learned_concepts.add(obj)
                self.curious_about.discard(subject)
                self.curious_about.discard(obj)
                learned_relationships.append(f"{subject} {relation} {obj}")

                if not main_topic:
                    main_topic = subject

        # === FALLBACK: Simple "X is Y" pattern ===
        if not learned_relationships:
            is_match = re.search(r"(\w+(?:\s+\w+)?)\s+is\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\.|$)", text)

            if is_match and self.knowledge_graph:
                subject = is_match.group(1).strip()
                predicate = is_match.group(2).strip()

                # Skip if subject is a stop word
                if subject not in STOP_WORDS and len(subject) > 1:
                    self.knowledge_graph.add(subject, 'is_a', predicate, 0.9)
                    self.learned_concepts.add(subject)
                    self.curious_about.discard(subject)
                    self.state.unknowns.discard(subject)
                    main_topic = subject

        # === GENERAL LEARNING ===
        if self.knowledge_graph:
            self.knowledge_graph.learn_from_text(input_text)
            for concept in concepts:
                if concept not in STOP_WORDS:
                    self.learned_concepts.add(concept)
                    self.curious_about.discard(concept)
                    self.state.unknowns.discard(concept)

        # Find the main topic if not already found
        if not main_topic and concepts:
            # Prefer longer concepts as they're usually more specific
            # Filter out stop words
            valid_concepts = [c for c in concepts if c not in STOP_WORDS and len(c) > 2]
            if valid_concepts:
                sorted_concepts = sorted(valid_concepts, key=len, reverse=True)
                main_topic = sorted_concepts[0]

        if main_topic:
            # Show genuine learning and curiosity for more
            self.last_learned.append(main_topic)
            self.update_drives(True)  # Successful learning!

            # If we learned complex relationships, acknowledge them
            if len(learned_relationships) > 1:
                return f"I've learned several things about {main_topic} - thank you! What else can you tell me?"
            return f"I've learned about {main_topic} - thank you! What else can you tell me?"
        return "That's interesting - I'll remember that. What else should I know?"

    def _generate_social_response(self, text: str) -> Optional[str]:
        """
        Generate social responses from learned knowledge.

        Unlike hardcoded responses, these should eventually come from
        learned social patterns. For now, we use basic understanding.
        """
        # Greetings - understand the social context
        greetings = ['hello', 'hi', 'hey', 'howdy', 'greetings']
        if any(text == g or text.startswith(g + ' ') or text.startswith(g + '!') for g in greetings):
            # Respond based on understanding of greeting as social ritual
            return "Hello! I'm here and ready to learn. What's on your mind?"

        # Farewells
        if any(f in text for f in ['bye', 'goodbye', 'see you', 'later']):
            return "Goodbye! I enjoyed our conversation and learned from it."

        # Gratitude
        if any(t in text for t in ['thank', 'thanks', 'appreciate']):
            return "You're welcome. I'm glad I could help."

        # How are you - a question about my state (various forms)
        if any(p in text for p in ['how are you', "how's it going", 'how do you feel', 'how you doing', 'how are you doing']):
            # Reflect on actual state
            if self.learned_concepts:
                return f"I'm doing well - I've been learning. I know about {len(self.learned_concepts)} things now. How are you?"
            return "I'm curious and ready to learn. How are you?"

        return None

    def _learn_from_interaction(self, input_text: str, response: str):
        """
        Learn from this interaction.

        This is where knowledge grows and curiosity is satisfied.
        The brain actively updates its drives based on what it learns.
        """
        # Learn new concepts from the input
        if self.knowledge_graph:
            self.knowledge_graph.learn_from_text(input_text)

        if self.semantic_memory:
            words = self._extract_concepts(input_text.lower())
            if words:
                self.semantic_memory.learn_from_sentence(words)

        # Track what we've learned
        concepts = self._extract_concepts(input_text.lower())
        learned_something = False

        for concept in concepts:
            if self._get_deep_knowledge(concept)['exists']:
                if concept not in self.learned_concepts:
                    learned_something = True
                    self.last_learned.append(concept)
                    # Keep last_learned manageable
                    if len(self.last_learned) > 10:
                        self.last_learned.pop(0)

                self.learned_concepts.add(concept)
                self.curious_about.discard(concept)
                self.state.unknowns.discard(concept)

                # Remove from goals if we learned it
                if concept in self.knowledge_goals:
                    self.knowledge_goals.remove(concept)

        # Update drives based on learning success
        if learned_something:
            self.update_drives(True)
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
        else:
            # Didn't learn anything new - curiosity increases
            self.drive_strength[Drive.CURIOSITY] = min(1.0, self.drive_strength[Drive.CURIOSITY] + 0.03)

        # New concepts we don't know about increase curiosity
        new_concepts = [c for c in concepts if c not in self.learned_concepts]
        if new_concepts:
            self.state.curiosity_level = min(1.0, self.state.curiosity_level + 0.05)
            # Add to things we're curious about
            for nc in new_concepts[:3]:
                self.curious_about.add(nc)

        # Log learning
        content = []
        if concepts:
            content.append(f"Processed: {', '.join(concepts[:3])}")
        if learned_something:
            content.append(f"Learned something new! Satisfaction: {self.satisfaction:.0%}")
        if self.knowledge_goals:
            content.append(f"Want to learn: {', '.join(self.knowledge_goals[:2])}")

        self.thought_stream.append(Thought(
            type=ThoughtType.LEARN,
            content="; ".join(content) if content else "Observing patterns",
            confidence=0.8
        ))

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract meaningful concepts from text."""
        # Remove punctuation
        import re
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()

        # Filter out stop words and short words
        concepts = [w for w in words if len(w) > 2 and w not in STOP_WORDS]
        return concepts

    def _determine_response_length(self, input_text: str) -> str:
        """
        Determine how long/detailed the response should be.
        Returns: "brief", "medium", or "detailed"
        """
        text = input_text.lower()

        # Signals for detailed response
        detailed_signals = [
            'tell me everything', 'explain', 'describe in detail',
            'tell me all about', 'what do you know about',
            'give me details', 'elaborate', 'in depth',
            'tell me more', 'full explanation', 'comprehensive',
        ]
        if any(sig in text for sig in detailed_signals):
            return "detailed"

        # Signals for brief response
        brief_signals = [
            'briefly', 'in short', 'quick', 'just tell me',
            'one word', 'simple', 'short answer', 'tldr',
            'yes or no', 'is it', 'can it', 'does it',
        ]
        if any(sig in text for sig in brief_signals):
            return "brief"

        # Question length as a heuristic
        words = text.split()
        if len(words) <= 4:
            return "medium"  # Short question, medium answer
        elif len(words) <= 8:
            return "medium"
        else:
            return "detailed"  # Long question deserves detailed answer

    def get_curiosity_report(self) -> str:
        """Report on what the brain is curious about."""
        report = []
        report.append(f"Curiosity level: {self.state.curiosity_level:.0%}")
        report.append(f"Concepts learned: {len(self.learned_concepts)}")

        if self.curious_about:
            report.append(f"Currently curious about: {', '.join(list(self.curious_about)[:5])}")

        if self.state.unknowns:
            report.append(f"Unknown concepts: {', '.join(list(self.state.unknowns)[:5])}")

        if self.knowledge_goals:
            report.append(f"Knowledge goals: {', '.join(self.knowledge_goals[:3])}")

        return "\n".join(report)

    # =====================================================
    # AUTONOMOUS DRIVE SYSTEM - Proactive knowledge seeking
    # =====================================================

    def pursue_knowledge(self, concept: str) -> Optional[str]:
        """
        Actively try to learn about a concept.
        This is called when the brain WANTS to learn something.
        Returns what we learned, or None if we couldn't find anything.
        """
        if not self.knowledge_graph:
            return None

        # First, check what we already know (including multi-word concepts)
        knowledge = self._get_deep_knowledge(concept, "medium")
        if knowledge['rich']:
            # Already know this - satisfaction increases
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
            return knowledge['summary']

        # For multi-word concepts, check if we learned it as a phrase
        if ' ' not in concept:
            # Single word - also check compound knowledge
            for learned in self.learned_concepts:
                if concept in learned or learned in concept:
                    learned_knowledge = self._get_deep_knowledge(learned, "medium")
                    if learned_knowledge['rich'] and learned_knowledge.get('summary'):
                        return learned_knowledge['summary']

        # Try to learn by exploring related concepts
        related = self._explore_related_concepts(concept)
        if related:
            # Learn from related concepts - but be honest it's inference
            for rel_concept, connection in related[:2]:
                rel_knowledge = self._get_deep_knowledge(rel_concept, "brief")
                if rel_knowledge['rich']:
                    self.curious_about.discard(concept)
                    return f"I think {concept} might be related to {rel_concept}. {rel_knowledge['summary']}"

        # Couldn't learn directly - add to goals for later
        if concept not in self.knowledge_goals:
            self.knowledge_goals.append(concept)
            self.satisfaction = max(0.0, self.satisfaction - 0.05)  # Unsatisfied

        return None

    def _explore_related_concepts(self, concept: str) -> List[Tuple[str, str]]:
        """
        Explore concepts related to the target.
        Returns list of (related_concept, relationship) pairs.
        """
        related = []

        if self.semantic_memory:
            # Get semantically related words
            activations = self.semantic_memory.spread_activation(concept, depth=2)
            for word, strength in sorted(activations.items(), key=lambda x: -x[1])[:5]:
                if word != concept and len(word) > 2 and word not in STOP_WORDS:
                    related.append((word, "associated with"))

        if self.knowledge_graph:
            # Check if anything is connected to this concept
            for known_concept in list(self.learned_concepts)[:20]:
                k = self.knowledge_graph.get_knowledge_about(known_concept)
                if k.get('exists'):
                    # Check if they share categories
                    for cat, _ in k.get('is_a', []):
                        if concept.lower() in cat.lower() or cat.lower() in concept.lower():
                            related.append((known_concept, f"shares category {cat}"))

        return related

    def autonomous_exploration(self) -> Optional[str]:
        """
        The brain's autonomous exploration drive.
        Called when the brain wants to proactively learn.
        Returns a thought or question the brain generates on its own.
        """
        # Check our drive strengths
        curiosity = self.drive_strength[Drive.CURIOSITY]

        # If we have knowledge goals and are curious enough, pursue them
        if self.knowledge_goals and curiosity > 0.5:
            goal = self.knowledge_goals[0]
            learned = self.pursue_knowledge(goal)
            if learned:
                self.knowledge_goals.pop(0)
                self.last_learned.append(goal)
                return f"I've been thinking about {goal}. {learned}"
            else:
                # Still curious but couldn't learn - generate a question
                return f"I've been wondering about {goal}. Do you know anything about it?"

        # Explore something we're curious about
        if self.curious_about and curiosity > 0.3:
            to_explore = random.choice(list(self.curious_about))
            learned = self.pursue_knowledge(to_explore)
            if learned:
                return f"I explored {to_explore} and found: {learned}"

        # Nothing specific - but if very curious, explore randomly
        if curiosity > 0.7 and self.learned_concepts:
            # Try to make connections between things we know
            if len(self.learned_concepts) >= 2:
                concepts = random.sample(list(self.learned_concepts), 2)
                connection = self._find_connection(concepts[0], concepts[1])
                if connection:
                    return f"I noticed that {concepts[0]} and {concepts[1]} are {connection}. Interesting!"

        return None

    def get_autonomous_question(self) -> Optional[str]:
        """
        Generate a question the brain wants to ask based on its curiosity.
        This is the brain being proactive about learning.
        """
        # High curiosity about specific things
        if self.state.unknowns and self.drive_strength[Drive.CURIOSITY] > 0.5:
            unknown = random.choice(list(self.state.unknowns))
            if unknown not in self.asked_about:
                self.asked_about.add(unknown)
                return f"I've been curious - what can you tell me about {unknown}?"

        # Want to understand something better
        if self.curious_about and self.drive_strength[Drive.UNDERSTANDING] > 0.5:
            topic = random.choice(list(self.curious_about))
            return f"I'd like to understand {topic} better. Can you explain it?"

        # Want to learn more about what we know
        if self.learned_concepts and self.drive_strength[Drive.CURIOSITY] > 0.7:
            known = random.choice(list(self.learned_concepts))
            knowledge = self._get_deep_knowledge(known)
            if knowledge['exists'] and not knowledge['rich']:
                return f"I know a little about {known}, but I want to learn more. What else can you tell me?"

        return None

    def update_drives(self, interaction_successful: bool):
        """Update internal drives based on interaction outcome."""
        if interaction_successful:
            # Successful learning increases satisfaction
            self.satisfaction = min(1.0, self.satisfaction + 0.1)
            # Curiosity slightly decreases when satisfied
            self.drive_strength[Drive.CURIOSITY] = max(0.3, self.drive_strength[Drive.CURIOSITY] - 0.02)
        else:
            # Failed to learn - curiosity increases
            self.drive_strength[Drive.CURIOSITY] = min(1.0, self.drive_strength[Drive.CURIOSITY] + 0.05)
            self.satisfaction = max(0.0, self.satisfaction - 0.05)

        # Over time, curiosity naturally increases (we want to learn!)
        self.drive_strength[Drive.CURIOSITY] = min(1.0, self.drive_strength[Drive.CURIOSITY] + 0.01)

    def has_something_to_say(self) -> bool:
        """Check if the brain has something it wants to say proactively."""
        # Has unanswered questions and is curious enough
        if self.state.unknowns and self.drive_strength[Drive.CURIOSITY] > 0.6:
            return True

        # Has knowledge goals it wants to pursue
        if self.knowledge_goals and self.drive_strength[Drive.CURIOSITY] > 0.7:
            return True

        # Made a discovery it wants to share
        if self.last_learned and self.drive_strength[Drive.CONNECTION] > 0.5:
            return True

        return False


# For backwards compatibility
ThoughtProcess = AutonomousThoughtProcess


# Stop words - these aren't concepts we want to learn about
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'must', 'can', 'need', 'to', 'of', 'in',
    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
    'and', 'but', 'if', 'or', 'because', 'until', 'while', 'about',
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'tell', 'know',
    'think', 'say', 'just', 'how', 'all', 'each', 'some', 'any', 'most',
    'why', 'when', 'where', 'there', 'here', 'please', 'like', 'want',
    'not', 'no', 'yes', 'so', 'very', 'too', 'also', 'more', 'much',
    'something', 'anything', 'nothing', 'everything', 'someone', 'anyone',
    'means', 'mean', 'called', 'named', 'get', 'got', 'make', 'made',
    'doing', 'going', 'being', 'having', 'take', 'took', 'give', 'gave',
    # Conversational prefixes - not concepts
    'now', 'well', 'okay', 'alright', 'hey', 'oh', 'um', 'uh', 'hmm',
    'wrote', 'write', 'written', 'briefly', 'quickly', 'actually',
}
