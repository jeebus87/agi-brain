"""
Semantic Memory - Word-Meaning Associations

Learns associations between:
- Sound patterns (phoneme sequences)
- Visual representations
- Abstract concepts
- Emotional valence

Uses Sparse Distributed Memory (SDM) for efficient storage
of high-dimensional associations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SemanticConfig:
    """Configuration for semantic memory."""
    n_neurons: int = 1_000_000    # Total neurons for memory
    address_size: int = 1000      # Address vector dimensionality
    data_size: int = 1000         # Data vector dimensionality
    n_hard_locations: int = 10000  # Number of hard memory locations
    activation_radius: int = 451   # Hamming distance threshold
    learning_rate: float = 0.1


class SparseDistributedMemory:
    """
    Sparse Distributed Memory (Kanerva, 1988)

    A content-addressable memory that stores patterns in a distributed
    manner across many hard locations. Provides:
    - Graceful degradation with noise
    - Generalization to similar patterns
    - Efficient storage of high-dimensional data
    """

    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()
        c = self.config

        # Hard location addresses (random binary vectors)
        self.rng = np.random.default_rng(42)
        self.addresses = self.rng.integers(0, 2, (c.n_hard_locations, c.address_size)).astype(np.int8)

        # Counters for each hard location
        self.counters = np.zeros((c.n_hard_locations, c.data_size), dtype=np.float32)

        # Access statistics
        self.read_count = 0
        self.write_count = 0

    def _compute_distances(self, address: np.ndarray) -> np.ndarray:
        """Compute Hamming distances to all hard locations."""
        return np.sum(self.addresses != address, axis=1)

    def _get_activated_locations(self, address: np.ndarray) -> np.ndarray:
        """Get indices of activated hard locations."""
        distances = self._compute_distances(address)
        return np.where(distances <= self.config.activation_radius)[0]

    def write(self, address: np.ndarray, data: np.ndarray):
        """Write data to memory at given address."""
        # Convert to binary
        address_bin = (address > 0).astype(np.int8)
        data_bin = np.sign(data)  # -1, 0, or 1

        # Find activated locations
        activated = self._get_activated_locations(address_bin)

        # Update counters
        for loc in activated:
            self.counters[loc] += data_bin * self.config.learning_rate

        self.write_count += 1

    def read(self, address: np.ndarray) -> np.ndarray:
        """Read data from memory at given address."""
        address_bin = (address > 0).astype(np.int8)

        # Find activated locations
        activated = self._get_activated_locations(address_bin)

        if len(activated) == 0:
            return np.zeros(self.config.data_size, dtype=np.float32)

        # Sum counters from activated locations
        sum_counters = np.sum(self.counters[activated], axis=0)

        # Threshold to binary
        result = np.sign(sum_counters)

        self.read_count += 1
        return result

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)


class ConceptNode:
    """
    Represents a single concept in semantic memory.

    Links together:
    - Phonological form (how it sounds)
    - Orthographic form (how it's spelled)
    - Semantic features (what it means)
    - Associations (related concepts)
    """

    def __init__(
        self,
        concept_id: int,
        vector_size: int = 1000,
        seed: int = None
    ):
        self.concept_id = concept_id
        self.vector_size = vector_size
        self.rng = np.random.default_rng(seed or concept_id)

        # Distributed representation
        self.vector = self.rng.randn(vector_size).astype(np.float32)
        self.vector /= np.linalg.norm(self.vector)

        # Labels and forms
        self.phonological: Optional[str] = None  # "kæt"
        self.orthographic: Optional[str] = None  # "cat"
        self.definitions: List[str] = []

        # Associations
        self.associations: Dict[int, float] = {}  # concept_id -> strength

        # Learning statistics
        self.activation_count = 0
        self.last_activated = 0

    def activate(self, time_step: int):
        """Record activation."""
        self.activation_count += 1
        self.last_activated = time_step

    def associate(self, other_id: int, strength: float = 0.1):
        """Create or strengthen association."""
        current = self.associations.get(other_id, 0.0)
        self.associations[other_id] = min(1.0, current + strength)

    def decay_associations(self, factor: float = 0.999):
        """Decay association strengths over time."""
        for k in self.associations:
            self.associations[k] *= factor


class SemanticMemory:
    """
    Complete semantic memory system.

    Manages concepts and their relationships:
    - Word learning (sound -> meaning)
    - Concept formation (clustering similar experiences)
    - Associative retrieval (spreading activation)
    """

    def __init__(self, config: Optional[SemanticConfig] = None):
        self.config = config or SemanticConfig()

        # Concept storage
        self.concepts: Dict[int, ConceptNode] = {}
        self.next_concept_id = 0

        # SDM for content-addressable retrieval
        self.sdm = SparseDistributedMemory(self.config)

        # Indices for fast lookup
        self.word_to_concept: Dict[str, int] = {}  # "cat" -> concept_id
        self.phoneme_to_concept: Dict[str, List[int]] = defaultdict(list)

        # Time tracking
        self.time_step = 0

    def create_concept(
        self,
        word: Optional[str] = None,
        phonemes: Optional[str] = None,
        definition: Optional[str] = None
    ) -> ConceptNode:
        """Create a new concept."""
        concept_id = self.next_concept_id
        self.next_concept_id += 1

        concept = ConceptNode(concept_id, self.config.data_size)

        if word:
            concept.orthographic = word
            self.word_to_concept[word.lower()] = concept_id

        if phonemes:
            concept.phonological = phonemes
            self.phoneme_to_concept[phonemes].append(concept_id)

        if definition:
            concept.definitions.append(definition)

        self.concepts[concept_id] = concept

        # Store in SDM
        address = self._word_to_vector(word or phonemes or str(concept_id))
        self.sdm.write(address, concept.vector)

        return concept

    def _word_to_vector(self, word: str) -> np.ndarray:
        """Convert word to binary vector (simple hash-based encoding)."""
        rng = np.random.default_rng(hash(word) % 2**32)
        return rng.integers(0, 2, self.config.address_size).astype(np.float32)

    def lookup(self, word: str) -> Optional[ConceptNode]:
        """Look up concept by word."""
        concept_id = self.word_to_concept.get(word.lower())
        if concept_id is not None:
            concept = self.concepts.get(concept_id)
            if concept:
                concept.activate(self.time_step)
            return concept
        return None

    def retrieve_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[ConceptNode, float]]:
        """Retrieve concepts similar to query vector."""
        results = []

        for concept in self.concepts.values():
            sim = self.sdm.similarity(query_vector, concept.vector)
            results.append((concept, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def learn_association(
        self,
        word1: str,
        word2: str,
        strength: float = 0.1
    ):
        """Learn association between two words."""
        c1 = self.lookup(word1)
        c2 = self.lookup(word2)

        if c1 is None:
            c1 = self.create_concept(word=word1)
        if c2 is None:
            c2 = self.create_concept(word=word2)

        c1.associate(c2.concept_id, strength)
        c2.associate(c1.concept_id, strength * 0.5)  # Weaker backward association

    def spread_activation(
        self,
        start_word: str,
        depth: int = 2,
        decay: float = 0.5
    ) -> Dict[str, float]:
        """
        Spreading activation from a concept.

        Returns activation levels of related concepts.
        """
        activations = {}

        concept = self.lookup(start_word)
        if not concept:
            return activations

        activations[concept.orthographic or str(concept.concept_id)] = 1.0

        current_level = {concept.concept_id: 1.0}

        for d in range(depth):
            next_level = {}

            for cid, activation in current_level.items():
                c = self.concepts.get(cid)
                if not c:
                    continue

                for assoc_id, strength in c.associations.items():
                    new_activation = activation * strength * decay
                    if assoc_id not in next_level or next_level[assoc_id] < new_activation:
                        next_level[assoc_id] = new_activation

                        assoc_concept = self.concepts.get(assoc_id)
                        if assoc_concept:
                            name = assoc_concept.orthographic or str(assoc_id)
                            activations[name] = max(activations.get(name, 0), new_activation)

            current_level = next_level

        return activations

    def learn_from_sentence(self, words: List[str], window_size: int = 3):
        """
        Learn associations from co-occurrence in sentence.

        Uses skip-gram style window for association learning.
        """
        self.time_step += 1

        # Ensure all words exist
        for word in words:
            if not self.lookup(word):
                self.create_concept(word=word)

        # Learn associations within window
        for i, word in enumerate(words):
            for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
                if i != j:
                    distance = abs(i - j)
                    strength = 0.1 / distance  # Closer words = stronger association
                    self.learn_association(word, words[j], strength)

    def get_vocabulary_size(self) -> int:
        """Get number of known words."""
        return len(self.word_to_concept)

    def get_statistics(self) -> Dict:
        """Get memory statistics."""
        return {
            'n_concepts': len(self.concepts),
            'vocabulary_size': self.get_vocabulary_size(),
            'sdm_writes': self.sdm.write_count,
            'sdm_reads': self.sdm.read_count,
            'time_step': self.time_step
        }

    def save(self, path: str):
        """Save semantic memory to file."""
        data = {
            'concepts': {
                cid: {
                    'vector': c.vector,
                    'orthographic': c.orthographic,
                    'phonological': c.phonological,
                    'definitions': c.definitions,
                    'associations': c.associations,
                    'activation_count': c.activation_count
                }
                for cid, c in self.concepts.items()
            },
            'word_to_concept': self.word_to_concept,
            'next_concept_id': self.next_concept_id,
            'sdm_counters': self.sdm.counters,
            'time_step': self.time_step
        }
        np.savez_compressed(path, data=data)

    def load(self, path: str):
        """Load semantic memory from file."""
        loaded = np.load(path, allow_pickle=True)
        data = loaded['data'].item()

        self.word_to_concept = data['word_to_concept']
        self.next_concept_id = data['next_concept_id']
        self.sdm.counters = data['sdm_counters']
        self.time_step = data['time_step']

        self.concepts = {}
        for cid_str, cdata in data['concepts'].items():
            cid = int(cid_str)
            concept = ConceptNode(cid, len(cdata['vector']))
            concept.vector = cdata['vector']
            concept.orthographic = cdata['orthographic']
            concept.phonological = cdata['phonological']
            concept.definitions = cdata['definitions']
            concept.associations = cdata['associations']
            concept.activation_count = cdata['activation_count']
            self.concepts[cid] = concept


class WordLearner:
    """
    Learns word-meaning mappings from multimodal input.

    Combines:
    - Audio input (phoneme sequences)
    - Visual input (object features)
    - Context (surrounding words/concepts)
    """

    def __init__(self, semantic_memory: SemanticMemory):
        self.memory = semantic_memory
        self.pending_associations: List[Tuple[str, np.ndarray]] = []

    def hear_word(self, phonemes: str) -> Optional[ConceptNode]:
        """Process heard word, return concept if known."""
        concept_ids = self.memory.phoneme_to_concept.get(phonemes, [])
        if concept_ids:
            concept = self.memory.concepts.get(concept_ids[0])
            if concept:
                concept.activate(self.memory.time_step)
            return concept
        return None

    def see_object(self, visual_features: np.ndarray, label: Optional[str] = None):
        """Process visual input, optionally with label."""
        if label:
            self.pending_associations.append((label, visual_features))

    def learn_word_object_association(
        self,
        word: str,
        visual_features: np.ndarray,
        phonemes: Optional[str] = None
    ):
        """Learn association between word and visual features."""
        concept = self.memory.lookup(word)

        if concept is None:
            concept = self.memory.create_concept(word=word, phonemes=phonemes)

        # Update concept vector toward visual features
        alpha = 0.1
        concept.vector = (1 - alpha) * concept.vector + alpha * visual_features
        concept.vector /= np.linalg.norm(concept.vector)

        # Store updated representation
        address = self.memory._word_to_vector(word)
        self.memory.sdm.write(address, concept.vector)

    def process_pending(self):
        """Process pending word-object associations."""
        for word, features in self.pending_associations:
            self.learn_word_object_association(word, features)
        self.pending_associations.clear()


if __name__ == "__main__":
    print("Testing Semantic Memory")
    print("=" * 50)

    # Create memory
    config = SemanticConfig(n_hard_locations=1000, data_size=100)
    memory = SemanticMemory(config)

    # Add some concepts
    print("Creating concepts...")
    memory.create_concept(word="cat", definition="A small furry animal")
    memory.create_concept(word="dog", definition="A loyal pet animal")
    memory.create_concept(word="pet", definition="An animal kept for companionship")
    memory.create_concept(word="animal", definition="A living creature")
    memory.create_concept(word="furry", definition="Covered with fur")

    # Learn associations
    print("Learning associations...")
    memory.learn_association("cat", "pet")
    memory.learn_association("cat", "furry")
    memory.learn_association("cat", "animal")
    memory.learn_association("dog", "pet")
    memory.learn_association("dog", "animal")
    memory.learn_association("pet", "animal")

    # Test spreading activation
    print("\nSpreading activation from 'cat':")
    activations = memory.spread_activation("cat", depth=2)
    for word, activation in sorted(activations.items(), key=lambda x: -x[1]):
        print(f"  {word}: {activation:.3f}")

    # Learn from sentence
    print("\nLearning from sentences...")
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "ran", "in", "the", "park"],
        ["my", "pet", "cat", "is", "furry"],
    ]
    for sentence in sentences:
        memory.learn_from_sentence(sentence)

    print(f"\nVocabulary size: {memory.get_vocabulary_size()}")
    stats = memory.get_statistics()
    print(f"Total concepts: {stats['n_concepts']}")
