"""
Language Learning System - Completely Language-Agnostic

This system learns language patterns from pure input with ZERO hardcoded words.
The brain discovers word categories (function words, content words) through
statistical frequency analysis, not through programmed lists.

Core Principle: If the brain hasn't learned it, it returns silence (empty string).

How it works:
1. STATISTICAL LEARNING: Track word frequency and position patterns
   - High-frequency words (appear often) → likely function words → keep literal
   - Low-frequency words (appear rarely) → likely content words → abstract to slots

2. PATTERN DISCOVERY: Find recurring sentence structures
   - "The dog is an animal" + "The cat is a pet"
   - → discovers "The [X] is a [Y]" pattern

3. RESPONSE GENERATION: Combine learned patterns + knowledge
   - Only generates responses using patterns it has learned
   - Returns empty string if it can't generate (true silence)
"""

import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum, auto


class PatternType(Enum):
    """Types of sentence patterns - learned, not assumed."""
    STATEMENT = auto()
    QUESTION = auto()
    EXCLAMATION = auto()
    UNKNOWN = auto()


@dataclass
class WordStats:
    """Statistics about a word - learned from observation."""
    word: str
    frequency: int = 0
    # Position tracking: where does this word appear?
    position_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # start, middle, end, after_frequent, before_frequent

    # Co-occurrence: what words appear near this one?
    left_neighbors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    right_neighbors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Feature-based co-occurrence: track patterns like "an" before vowel words
    # Maps first letter of following word -> count
    right_first_letter: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    # Maps first letter of preceding word -> count
    left_first_letter: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Learned roles
    is_likely_function_word: bool = False  # High frequency, appears in fixed positions
    is_likely_content_word: bool = False   # Lower frequency, variable positions

    def to_dict(self) -> dict:
        return {
            'word': self.word,
            'frequency': self.frequency,
            'position_counts': dict(self.position_counts),
            'left_neighbors': dict(self.left_neighbors),
            'right_neighbors': dict(self.right_neighbors),
            'right_first_letter': dict(self.right_first_letter),
            'left_first_letter': dict(self.left_first_letter),
            'is_likely_function_word': self.is_likely_function_word,
            'is_likely_content_word': self.is_likely_content_word,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'WordStats':
        ws = cls(word=data['word'])
        ws.frequency = data.get('frequency', 0)
        ws.position_counts = defaultdict(int, data.get('position_counts', {}))
        ws.left_neighbors = defaultdict(int, data.get('left_neighbors', {}))
        ws.right_neighbors = defaultdict(int, data.get('right_neighbors', {}))
        ws.right_first_letter = defaultdict(int, data.get('right_first_letter', {}))
        ws.left_first_letter = defaultdict(int, data.get('left_first_letter', {}))
        ws.is_likely_function_word = data.get('is_likely_function_word', False)
        ws.is_likely_content_word = data.get('is_likely_content_word', False)
        return ws


@dataclass
class LearnedPattern:
    """A pattern discovered from input."""
    template: str           # "the [SLOT_1] is a [SLOT_2]"
    slot_count: int         # Number of slots
    examples: List[str] = field(default_factory=list)
    slot_fillers: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    frequency: int = 1
    pattern_type: PatternType = PatternType.UNKNOWN

    def to_dict(self) -> dict:
        return {
            'template': self.template,
            'slot_count': self.slot_count,
            'examples': self.examples[-20:],  # Keep last 20
            'slot_fillers': {k: list(v)[-10:] for k, v in self.slot_fillers.items()},
            'frequency': self.frequency,
            'pattern_type': self.pattern_type.name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'LearnedPattern':
        lp = cls(
            template=data['template'],
            slot_count=data.get('slot_count', 0),
            examples=data.get('examples', []),
            frequency=data.get('frequency', 1),
            pattern_type=PatternType[data.get('pattern_type', 'UNKNOWN')],
        )
        lp.slot_fillers = defaultdict(list, {k: list(v) for k, v in data.get('slot_fillers', {}).items()})
        return lp


class LanguageLearner:
    """
    Language-agnostic learning system.

    NO hardcoded word lists. Everything is learned from input.
    Word categories are discovered through statistical frequency analysis.
    """

    # Frequency threshold - words appearing more than this % of total are likely function words
    FUNCTION_WORD_THRESHOLD = 0.02  # 2% of all word occurrences

    # Minimum observations before we trust word statistics
    MIN_OBSERVATIONS = 5

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path

        # Word statistics - learned from observation
        self.word_stats: Dict[str, WordStats] = {}

        # Total word count for frequency calculation
        self.total_word_count = 0

        # Learned patterns
        self.patterns: Dict[str, LearnedPattern] = {}

        # Sentence examples for pattern discovery
        self.sentence_buffer: List[List[str]] = []
        self.max_buffer_size = 100

        # Load if exists
        if storage_path:
            self.load()

    def learn_from_sentence(self, sentence: str) -> Dict:
        """
        Learn from a sentence. Updates word statistics and discovers patterns.

        Returns what was learned.
        """
        sentence = sentence.strip()
        if not sentence:
            return {'learned': False}

        # Tokenize
        words = self._tokenize(sentence)
        if not words:
            return {'learned': False}

        learned = {
            'learned': True,
            'words_observed': len(words),
            'patterns_discovered': [],
            'new_words': [],
        }

        # Update word statistics
        for i, word in enumerate(words):
            is_new = word not in self.word_stats
            self._update_word_stats(word, i, len(words), words)
            if is_new:
                learned['new_words'].append(word)

        self.total_word_count += len(words)

        # Add to sentence buffer for pattern discovery
        self.sentence_buffer.append(words)
        if len(self.sentence_buffer) > self.max_buffer_size:
            self.sentence_buffer.pop(0)

        # Reclassify words FIRST based on updated frequencies
        # This ensures function words are identified before pattern discovery
        self._reclassify_words()

        # Try to discover patterns - need enough data for stable word classification
        # Wait until we have enough sentences AND some words classified as function words
        function_word_count = sum(1 for s in self.word_stats.values() if s.is_likely_function_word)
        if len(self.sentence_buffer) >= 5 and function_word_count >= 2:
            new_patterns = self._discover_patterns(words)
            learned['patterns_discovered'] = new_patterns

        return learned

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - keeps punctuation as information."""
        # Normalize
        text = text.lower().strip()

        # Separate punctuation but keep it
        text = re.sub(r'([.!?,:;])', r' \1', text)

        # Split and filter
        words = [w.strip() for w in text.split() if w.strip()]
        return words

    def _update_word_stats(self, word: str, position: int, total_words: int, all_words: List[str]):
        """Update statistics for a word based on where we observed it."""
        if word not in self.word_stats:
            self.word_stats[word] = WordStats(word=word)

        stats = self.word_stats[word]
        stats.frequency += 1

        # Track position
        if position == 0:
            stats.position_counts['start'] += 1
        elif position == total_words - 1:
            stats.position_counts['end'] += 1
        else:
            stats.position_counts['middle'] += 1

        # Relative position (normalized)
        rel_pos = position / max(total_words - 1, 1)
        if rel_pos < 0.33:
            stats.position_counts['first_third'] += 1
        elif rel_pos < 0.66:
            stats.position_counts['middle_third'] += 1
        else:
            stats.position_counts['last_third'] += 1

        # Track neighbors
        if position > 0:
            left = all_words[position - 1]
            stats.left_neighbors[left] += 1
            # Track first letter pattern
            if left and left[0].isalpha():
                stats.left_first_letter[left[0]] += 1
        if position < total_words - 1:
            right = all_words[position + 1]
            stats.right_neighbors[right] += 1
            # Track first letter pattern - crucial for learning article patterns
            if right and right[0].isalpha():
                stats.right_first_letter[right[0]] += 1

    def _reclassify_words(self):
        """
        Reclassify words as function/content based on current statistics.

        Function words: High frequency, appear in fixed patterns
        Content words: Lower frequency, more variable
        """
        if self.total_word_count < 10:
            return  # Not enough data

        for word, stats in self.word_stats.items():
            relative_freq = stats.frequency / self.total_word_count

            # High frequency = likely function word
            if relative_freq > self.FUNCTION_WORD_THRESHOLD and stats.frequency >= self.MIN_OBSERVATIONS:
                stats.is_likely_function_word = True
                stats.is_likely_content_word = False
            elif stats.frequency >= self.MIN_OBSERVATIONS:
                stats.is_likely_function_word = False
                stats.is_likely_content_word = True

    def _discover_patterns(self, current_words: List[str]) -> List[str]:
        """
        Discover patterns by abstracting content words to slots.

        Uses learned word classifications to decide what to abstract.
        """
        discovered = []

        # Build abstract pattern from current sentence
        pattern = self._abstract_to_pattern(current_words)

        if pattern:
            template = pattern['template']

            if template in self.patterns:
                # Reinforce existing pattern
                self.patterns[template].frequency += 1
                if ' '.join(current_words) not in self.patterns[template].examples:
                    self.patterns[template].examples.append(' '.join(current_words))
                # Track slot fillers
                for slot, filler in pattern['fillers'].items():
                    if filler not in self.patterns[template].slot_fillers[slot]:
                        self.patterns[template].slot_fillers[slot].append(filler)
            else:
                # New pattern!
                self.patterns[template] = LearnedPattern(
                    template=template,
                    slot_count=len(pattern['fillers']),
                    examples=[' '.join(current_words)],
                    pattern_type=self._detect_pattern_type(' '.join(current_words)),
                )
                for slot, filler in pattern['fillers'].items():
                    self.patterns[template].slot_fillers[slot].append(filler)
                discovered.append(template)

        return discovered

    def _abstract_to_pattern(self, words: List[str]) -> Optional[Dict]:
        """
        Convert a sentence to an abstract pattern.

        High-frequency words stay literal.
        Low-frequency words become slots.
        """
        if not words:
            return None

        template_parts = []
        fillers = {}
        slot_num = 1

        i = 0
        while i < len(words):
            word = words[i]
            stats = self.word_stats.get(word)

            # Punctuation stays literal
            if word in '.!?,:;':
                template_parts.append(word)
                i += 1
                continue

            # Decide based on learned classification
            if stats and stats.is_likely_function_word:
                # High-frequency word - keep literal
                template_parts.append(word)
            elif stats and stats.is_likely_content_word:
                # Low-frequency word - abstract to slot
                slot_name = f"SLOT_{slot_num}"

                # Check if next words should be grouped (multi-word content)
                content_words = [word]
                j = i + 1
                while j < len(words):
                    next_word = words[j]
                    next_stats = self.word_stats.get(next_word)
                    if next_stats and next_stats.is_likely_function_word:
                        break
                    if next_word in '.!?,:;':
                        break
                    content_words.append(next_word)
                    j += 1

                template_parts.append(f"[{slot_name}]")
                fillers[slot_name] = ' '.join(content_words)
                slot_num += 1
                i = j
                continue
            else:
                # Not enough data yet - treat as potential content word
                slot_name = f"SLOT_{slot_num}"
                template_parts.append(f"[{slot_name}]")
                fillers[slot_name] = word
                slot_num += 1

            i += 1

        template = ' '.join(template_parts)

        # Don't create patterns that are all slots or all literal
        if slot_num <= 1:
            return None  # No slots - nothing abstracted
        if len([p for p in template_parts if not p.startswith('[')]) == 0:
            return None  # All slots - too abstract

        return {
            'template': template,
            'fillers': fillers,
        }

    def _detect_pattern_type(self, sentence: str) -> PatternType:
        """Detect pattern type from punctuation and structure."""
        if '?' in sentence:
            return PatternType.QUESTION
        if '!' in sentence:
            return PatternType.EXCLAMATION
        if '.' in sentence or sentence:
            return PatternType.STATEMENT
        return PatternType.UNKNOWN

    def generate_response(self,
                         knowledge_facts: List[Tuple[str, str, str]],
                         context: str = "general",
                         input_sentence: str = "") -> str:
        """
        Generate a response using ONLY learned patterns and provided knowledge.

        Returns empty string if cannot generate (true silence).
        The brain needs to have learned enough patterns to speak coherently.
        """
        # Must have learned actual patterns to speak
        # Minimum threshold: need patterns with enough frequency to be reliable
        if not self.can_speak():
            return ""

        # Need at least one statement pattern (not just questions)
        has_statement_pattern = any(
            p.pattern_type == PatternType.STATEMENT and p.slot_count >= 2
            for p in self.patterns.values()
        )
        if not has_statement_pattern:
            return ""

        if not knowledge_facts:
            return ""

        responses = []

        for subject, relation, obj in knowledge_facts:
            # Find a pattern that can express this
            response = self._generate_from_pattern(subject, relation, obj)
            if response:
                responses.append(response)

        if responses:
            return self._combine_responses(responses)

        # Cannot express this knowledge - silence
        return ""

    def _generate_from_pattern(self, subject: str, relation: str, obj: str) -> Optional[str]:
        """Generate text from a learned pattern."""
        if not self.patterns:
            return None

        # Score patterns by fitness - ONLY use statement patterns
        candidates = []
        for template, pattern in self.patterns.items():
            # Pattern must have at least 2 slots and be a STATEMENT (not question)
            if pattern.slot_count < 2:
                continue
            if pattern.pattern_type != PatternType.STATEMENT:
                continue

            # Check if this pattern type fits the relation
            if relation in ('is', 'is_a', 'has', 'can'):
                score = pattern.frequency
                candidates.append((score, template, pattern))

        if not candidates:
            # Try any statement pattern with 2+ slots
            for template, pattern in self.patterns.items():
                if pattern.slot_count >= 2 and pattern.pattern_type == PatternType.STATEMENT:
                    candidates.append((pattern.frequency, template, pattern))

        if not candidates:
            return None

        # Pick a pattern (weighted by frequency)
        candidates.sort(key=lambda x: -x[0])
        _, template, pattern = candidates[0]

        # Fill the pattern with grammar correction
        filled = self._fill_pattern_with_grammar(template, subject, obj)

        # Capitalize
        if filled:
            filled = filled[0].upper() + filled[1:]

        return filled if filled else None

    def _fill_pattern_with_grammar(self, template: str, subject: str, obj: str) -> str:
        """
        Fill a pattern template with words, applying learned grammar rules.

        This handles things like "a" vs "an" based on learned first-letter patterns.
        """
        # Split template into tokens
        tokens = template.split()
        result_tokens = []

        slot_values = {'SLOT_1': subject, 'SLOT_2': obj}
        i = 0

        while i < len(tokens):
            token = tokens[i]

            # Check if this is a slot
            slot_match = re.match(r'\[SLOT_(\d+)\]', token)
            if slot_match:
                slot_name = f"SLOT_{slot_match.group(1)}"
                filler = slot_values.get(slot_name, '')

                # Check if previous token is a function word that might need adjustment
                if result_tokens and filler:
                    prev_token = result_tokens[-1]
                    prev_stats = self.word_stats.get(prev_token.lower())

                    if prev_stats and prev_stats.is_likely_function_word:
                        # Find the best variant of this function word for the filler
                        best_variant = self._find_best_variant(prev_token.lower(), filler)
                        if best_variant != prev_token.lower():
                            # Replace the previous token with the better variant
                            result_tokens[-1] = best_variant

                result_tokens.append(filler)
            else:
                result_tokens.append(token)

            i += 1

        # Clean up unfilled slots and extra spaces
        result = ' '.join(t for t in result_tokens if t and not re.match(r'\[SLOT_\d+\]', t))
        return result

    def _combine_responses(self, responses: List[str]) -> str:
        """Combine multiple responses."""
        if not responses:
            return ""
        if len(responses) == 1:
            return responses[0]
        return '. '.join(responses)

    def can_speak(self) -> bool:
        """Check if the brain has learned enough to speak."""
        return len(self.patterns) > 0

    def _find_best_variant(self, word: str, following_word: str) -> str:
        """
        Find the best variant of a word based on the following word.

        This is learned, not hardcoded. The brain observes that certain words
        (like "a"/"an") have different first-letter distributions for
        following words, and picks the one that best matches.
        """
        if not following_word or not following_word[0].isalpha():
            return word

        first_letter = following_word[0].lower()
        stats = self.word_stats.get(word)

        if not stats:
            return word

        # Find potential variant words - not just function words
        # Look for words that:
        # 1. Are similar length (within 2 characters)
        # 2. Have complementary first-letter distributions
        # 3. Appear in similar positions
        candidates = []
        for other_word, other_stats in self.word_stats.items():
            if other_word == word:
                continue

            # Must have enough observations
            if other_stats.frequency < 10:
                continue

            # Similar length (variants like a/an, this/these tend to be similar length)
            if abs(len(other_word) - len(word)) > 2:
                continue

            # Check if they could be positional variants
            if self._are_positional_variants(stats, other_stats):
                candidates.append((other_word, other_stats))

        if not candidates:
            return word

        # Score each candidate (including original) by how well their
        # first-letter patterns match the following word
        best_word = word
        best_score = stats.right_first_letter.get(first_letter, 0)

        for cand_word, cand_stats in candidates:
            score = cand_stats.right_first_letter.get(first_letter, 0)
            if score > best_score:
                best_score = score
                best_word = cand_word

        return best_word

    def _are_positional_variants(self, stats1: WordStats, stats2: WordStats) -> bool:
        """
        Check if two words are likely variants (interchangeable in context).

        For example, "a" and "an" are variants because:
        1. They appear in similar positions (before nouns)
        2. They have COMPLEMENTARY first-letter distributions (mutually exclusive)
           - "a" before consonants, "an" before vowels

        Words like "is" and "a" are NOT variants because they serve different roles.
        """
        # Both should have enough observations
        if stats1.frequency < 10 or stats2.frequency < 10:
            return False

        # Key insight: true variants have COMPLEMENTARY first-letter patterns
        # (they appear before different starting letters)
        letters1 = set(stats1.right_first_letter.keys())
        letters2 = set(stats2.right_first_letter.keys())

        if not letters1 or not letters2:
            return False

        # Calculate overlap in first-letter distributions
        shared_letters = letters1 & letters2
        total_letters = letters1 | letters2

        if not total_letters:
            return False

        overlap_ratio = len(shared_letters) / len(total_letters)

        # True variants should have LOW overlap (complementary distributions)
        # "a" appears before consonants, "an" before vowels = low overlap
        # "is" and "a" both appear before many letters = high overlap
        if overlap_ratio > 0.5:
            return False  # Too much overlap - not true variants

        # Also check they appear in similar sentence positions
        # (both before content words, both early in sentence, etc.)
        pos1 = dict(stats1.position_counts)
        pos2 = dict(stats2.position_counts)

        # Normalize position counts
        total1 = sum(pos1.values()) or 1
        total2 = sum(pos2.values()) or 1

        # Check if relative position distributions are similar
        position_similarity = 0
        for pos in ['start', 'middle', 'first_third', 'middle_third']:
            ratio1 = pos1.get(pos, 0) / total1
            ratio2 = pos2.get(pos, 0) / total2
            if abs(ratio1 - ratio2) < 0.2:  # Within 20%
                position_similarity += 1

        # Need similar positions AND complementary first-letter patterns
        return position_similarity >= 2 and overlap_ratio < 0.5

    def get_stats(self) -> Dict:
        """Get learning statistics."""
        function_words = [w for w, s in self.word_stats.items() if s.is_likely_function_word]
        content_words = [w for w, s in self.word_stats.items() if s.is_likely_content_word]

        return {
            'total_patterns': len(self.patterns),
            'vocabulary_size': len(self.word_stats),
            'total_word_occurrences': self.total_word_count,
            'learned_function_words': len(function_words),
            'learned_content_words': len(content_words),
            'function_word_examples': function_words[:20],
            'patterns_by_frequency': sorted(
                [(p.template, p.frequency) for p in self.patterns.values()],
                key=lambda x: -x[1]
            )[:10],
            'sentences_in_buffer': len(self.sentence_buffer),
        }

    def save(self, path: Optional[Path] = None):
        """Save learned language to disk."""
        path = path or self.storage_path
        if not path:
            return

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'word_stats': {k: v.to_dict() for k, v in self.word_stats.items()},
            'total_word_count': self.total_word_count,
            'patterns': {k: v.to_dict() for k, v in self.patterns.items()},
            'sentence_buffer': self.sentence_buffer[-50:],  # Keep last 50
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Optional[Path] = None) -> bool:
        """Load learned language from disk."""
        path = path or self.storage_path
        if not path:
            return False

        path = Path(path)
        if not path.exists():
            return False

        try:
            with open(path) as f:
                data = json.load(f)

            self.word_stats = {k: WordStats.from_dict(v) for k, v in data.get('word_stats', {}).items()}
            self.total_word_count = data.get('total_word_count', 0)
            self.patterns = {k: LearnedPattern.from_dict(v) for k, v in data.get('patterns', {}).items()}
            self.sentence_buffer = data.get('sentence_buffer', [])

            return True
        except Exception as e:
            print(f"Error loading language data: {e}")
            return False


# Demo/test function
def demo():
    """Demonstrate the language-agnostic learner."""
    learner = LanguageLearner()

    # Teaching corpus - the brain learns language structure from these
    sentences = [
        "The dog is an animal.",
        "The cat is a pet.",
        "A bird can fly.",
        "The fish is in water.",
        "My name is Xemsa.",
        "The sky is blue.",
        "A tree is a plant.",
        "The car is a vehicle.",
        "Water is a liquid.",
        "Fire is hot.",
        "Ice is cold.",
        "The sun is bright.",
        "A book has pages.",
        "The house is big.",
        "The mouse is small.",
    ]

    print("Teaching the brain (language-agnostic)...\n")

    for sentence in sentences:
        result = learner.learn_from_sentence(sentence)
        print(f"  '{sentence}'")
        if result['patterns_discovered']:
            print(f"    -> Discovered patterns: {result['patterns_discovered']}")

    print(f"\n--- Statistics ---")
    stats = learner.get_stats()
    print(f"Vocabulary: {stats['vocabulary_size']} words")
    print(f"Function words learned: {stats['learned_function_words']}")
    print(f"Content words learned: {stats['learned_content_words']}")
    print(f"Patterns discovered: {stats['total_patterns']}")
    print(f"\nFunction words: {stats['function_word_examples']}")
    print(f"\nTop patterns:")
    for template, freq in stats['patterns_by_frequency'][:5]:
        print(f"  {template} (used {freq}x)")

    print(f"\n--- Generation Test ---")

    # Test generation
    facts = [("dog", "is", "animal")]
    response = learner.generate_response(facts)
    print(f"Knowledge: dog is animal -> '{response}'")

    facts = [("xemsa", "is", "intelligence")]
    response = learner.generate_response(facts)
    print(f"Knowledge: xemsa is intelligence -> '{response}'")

    # Test with no patterns (should be silent)
    empty_learner = LanguageLearner()
    response = empty_learner.generate_response([("test", "is", "test")])
    print(f"Untaught brain response: '{response}' (should be empty)")


if __name__ == "__main__":
    demo()
