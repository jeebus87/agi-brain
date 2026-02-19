"""
Language Learning Module

Enables the AGI brain to learn language from scratch through:
- Audio encoding (cochlea simulation)
- Phoneme recognition (STDP learning)
- Semantic memory (word-meaning associations)
- Speech generation (neural vocoder)
"""

# Lazy imports to avoid circular dependencies
def get_sparse_network():
    from .sparse_network import SparseSNN, SparsePopulation, create_language_brain
    return SparseSNN, SparsePopulation, create_language_brain

def get_audio_encoder():
    from .audio_encoder import CochleaEncoder, AudioProcessor
    return CochleaEncoder, AudioProcessor

def get_phoneme_learner():
    from .phoneme_learner import PhonemeLearner, PhonemeDetector
    return PhonemeLearner, PhonemeDetector

def get_semantic_memory():
    from .semantic_memory import SemanticMemory, ConceptNode
    return SemanticMemory, ConceptNode

def get_speech_generator():
    from .speech_generator import SpeechGenerator, NeuralVocoder
    return SpeechGenerator, NeuralVocoder
