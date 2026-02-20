"""
Dictionary Loader - Bootstrap brain with vocabulary

Uses WordNet to load:
- Word definitions
- Synonyms
- Word relationships (hypernyms, hyponyms)
"""

import os
from typing import Dict, List, Optional, Set
from pathlib import Path


class DictionaryLoader:
    """Load vocabulary from WordNet or built-in word lists."""

    def __init__(self):
        self.wordnet_available = False
        self._check_wordnet()

    def _check_wordnet(self):
        """Check if WordNet is available."""
        try:
            import nltk
            from nltk.corpus import wordnet
            # Try to access wordnet
            wordnet.synsets('test')
            self.wordnet_available = True
        except LookupError:
            # WordNet data not downloaded
            try:
                import nltk
                print("Downloading WordNet data...")
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
                from nltk.corpus import wordnet
                wordnet.synsets('test')
                self.wordnet_available = True
            except:
                self.wordnet_available = False
        except ImportError:
            self.wordnet_available = False

    def get_word_info(self, word: str) -> Dict:
        """Get information about a word from WordNet."""
        if not self.wordnet_available:
            return {'word': word, 'definitions': [], 'synonyms': [], 'related': []}

        from nltk.corpus import wordnet

        info = {
            'word': word,
            'definitions': [],
            'synonyms': set(),
            'related': set(),
            'hypernyms': set(),  # Broader terms
            'hyponyms': set(),   # Narrower terms
        }

        synsets = wordnet.synsets(word)

        for syn in synsets[:3]:  # Limit to first 3 meanings
            # Definition
            info['definitions'].append(syn.definition())

            # Synonyms (lemmas in same synset)
            for lemma in syn.lemmas():
                name = lemma.name().replace('_', ' ')
                if name.lower() != word.lower():
                    info['synonyms'].add(name)

            # Hypernyms (broader categories)
            for hyper in syn.hypernyms():
                for lemma in hyper.lemmas():
                    info['hypernyms'].add(lemma.name().replace('_', ' '))

            # Hyponyms (more specific terms)
            for hypo in syn.hyponyms()[:5]:
                for lemma in hypo.lemmas():
                    info['hyponyms'].add(lemma.name().replace('_', ' '))

        # Convert sets to lists
        info['synonyms'] = list(info['synonyms'])[:10]
        info['related'] = list(info['hypernyms'] | info['hyponyms'])[:10]
        info['hypernyms'] = list(info['hypernyms'])[:5]
        info['hyponyms'] = list(info['hyponyms'])[:5]

        return info

    def load_basic_vocabulary(self) -> List[Dict]:
        """Load a curated basic vocabulary with relationships."""
        # Core vocabulary organized by category
        vocabulary = {
            # Animals
            'cat': ['animal', 'pet', 'mammal', 'furry', 'meow', 'kitten'],
            'dog': ['animal', 'pet', 'mammal', 'bark', 'puppy', 'loyal'],
            'bird': ['animal', 'fly', 'wings', 'feathers', 'nest', 'sing'],
            'fish': ['animal', 'swim', 'water', 'fins', 'ocean', 'aquarium'],
            'horse': ['animal', 'mammal', 'ride', 'fast', 'farm', 'gallop'],
            'cow': ['animal', 'mammal', 'farm', 'milk', 'beef', 'moo'],
            'pig': ['animal', 'mammal', 'farm', 'oink', 'mud', 'bacon'],
            'chicken': ['animal', 'bird', 'farm', 'eggs', 'feathers', 'cluck'],
            'lion': ['animal', 'mammal', 'wild', 'roar', 'Africa', 'king'],
            'elephant': ['animal', 'mammal', 'large', 'trunk', 'Africa', 'gray'],

            # Nature
            'tree': ['plant', 'nature', 'leaves', 'wood', 'forest', 'tall'],
            'flower': ['plant', 'nature', 'beautiful', 'garden', 'bloom', 'smell'],
            'grass': ['plant', 'nature', 'green', 'lawn', 'grow', 'ground'],
            'sun': ['star', 'hot', 'bright', 'sky', 'day', 'light', 'warm'],
            'moon': ['sky', 'night', 'bright', 'space', 'lunar', 'crater'],
            'star': ['sky', 'night', 'bright', 'space', 'twinkle', 'distant'],
            'water': ['liquid', 'drink', 'wet', 'ocean', 'river', 'rain'],
            'fire': ['hot', 'burn', 'flame', 'heat', 'light', 'danger'],
            'rain': ['water', 'weather', 'wet', 'cloud', 'storm', 'umbrella'],
            'snow': ['cold', 'winter', 'white', 'ice', 'freeze', 'weather'],

            # Food
            'apple': ['fruit', 'food', 'red', 'sweet', 'healthy', 'tree'],
            'banana': ['fruit', 'food', 'yellow', 'sweet', 'monkey', 'peel'],
            'bread': ['food', 'bake', 'wheat', 'toast', 'sandwich', 'carbs'],
            'milk': ['drink', 'white', 'dairy', 'cow', 'calcium', 'cold'],
            'egg': ['food', 'chicken', 'breakfast', 'protein', 'cook', 'shell'],
            'meat': ['food', 'protein', 'animal', 'cook', 'beef', 'chicken'],
            'rice': ['food', 'grain', 'white', 'cook', 'asian', 'carbs'],
            'pizza': ['food', 'italian', 'cheese', 'delicious', 'dinner', 'slice'],

            # People
            'person': ['human', 'people', 'individual', 'someone', 'body', 'mind'],
            'man': ['person', 'male', 'human', 'adult', 'boy', 'father'],
            'woman': ['person', 'female', 'human', 'adult', 'girl', 'mother'],
            'child': ['person', 'young', 'kid', 'small', 'grow', 'play'],
            'baby': ['child', 'young', 'small', 'infant', 'cute', 'cry'],
            'family': ['people', 'home', 'love', 'parents', 'children', 'together'],
            'friend': ['person', 'companion', 'trust', 'fun', 'help', 'loyal'],
            'doctor': ['person', 'medical', 'health', 'hospital', 'help', 'cure'],
            'teacher': ['person', 'school', 'learn', 'education', 'knowledge', 'help'],

            # Places
            'house': ['building', 'home', 'live', 'family', 'room', 'shelter'],
            'school': ['building', 'learn', 'education', 'student', 'teacher', 'class'],
            'hospital': ['building', 'medical', 'doctor', 'sick', 'health', 'help'],
            'city': ['place', 'urban', 'buildings', 'people', 'busy', 'town'],
            'country': ['place', 'nation', 'land', 'people', 'government', 'rural'],
            'ocean': ['water', 'large', 'sea', 'fish', 'wave', 'deep', 'blue'],
            'mountain': ['land', 'tall', 'rock', 'climb', 'snow', 'peak'],
            'forest': ['nature', 'trees', 'green', 'animals', 'wild', 'dense'],

            # Objects
            'car': ['vehicle', 'drive', 'wheels', 'fast', 'road', 'transport'],
            'phone': ['device', 'call', 'communication', 'screen', 'mobile', 'talk'],
            'computer': ['device', 'technology', 'screen', 'work', 'internet', 'digital'],
            'book': ['read', 'pages', 'knowledge', 'story', 'learn', 'paper'],
            'table': ['furniture', 'flat', 'surface', 'eat', 'work', 'wood'],
            'chair': ['furniture', 'sit', 'legs', 'comfort', 'wood', 'seat'],
            'bed': ['furniture', 'sleep', 'rest', 'comfortable', 'bedroom', 'soft'],
            'door': ['entrance', 'open', 'close', 'room', 'wood', 'lock'],
            'window': ['glass', 'see', 'light', 'open', 'house', 'view'],

            # Actions (as concepts)
            'run': ['move', 'fast', 'exercise', 'legs', 'speed', 'race'],
            'walk': ['move', 'slow', 'legs', 'step', 'exercise', 'path'],
            'eat': ['food', 'consume', 'hungry', 'mouth', 'taste', 'meal'],
            'drink': ['liquid', 'consume', 'thirsty', 'water', 'mouth', 'swallow'],
            'sleep': ['rest', 'tired', 'bed', 'night', 'dream', 'eyes'],
            'work': ['job', 'effort', 'money', 'busy', 'office', 'task'],
            'play': ['fun', 'game', 'enjoy', 'child', 'sport', 'happy'],
            'learn': ['knowledge', 'study', 'understand', 'school', 'brain', 'grow'],
            'think': ['mind', 'brain', 'idea', 'thought', 'consider', 'reason'],
            'love': ['emotion', 'heart', 'care', 'happy', 'family', 'friend'],

            # Adjectives (as concepts)
            'big': ['large', 'huge', 'size', 'giant', 'massive', 'great'],
            'small': ['little', 'tiny', 'size', 'mini', 'short', 'compact'],
            'hot': ['warm', 'temperature', 'heat', 'fire', 'summer', 'burn'],
            'cold': ['cool', 'temperature', 'freeze', 'ice', 'winter', 'chill'],
            'happy': ['emotion', 'joy', 'smile', 'glad', 'pleased', 'cheerful'],
            'sad': ['emotion', 'unhappy', 'cry', 'tears', 'sorrow', 'down'],
            'good': ['positive', 'nice', 'great', 'excellent', 'fine', 'well'],
            'bad': ['negative', 'poor', 'wrong', 'terrible', 'awful', 'evil'],
            'fast': ['quick', 'speed', 'rapid', 'swift', 'hurry', 'run'],
            'slow': ['gradual', 'speed', 'careful', 'steady', 'patient', 'walk'],

            # Time
            'day': ['time', 'light', 'sun', 'morning', 'afternoon', 'hours'],
            'night': ['time', 'dark', 'moon', 'stars', 'sleep', 'evening'],
            'morning': ['time', 'day', 'early', 'sunrise', 'breakfast', 'wake'],
            'evening': ['time', 'night', 'late', 'sunset', 'dinner', 'dark'],
            'today': ['time', 'now', 'present', 'current', 'day'],
            'tomorrow': ['time', 'future', 'next', 'day', 'later'],
            'yesterday': ['time', 'past', 'before', 'day', 'previous'],

            # Colors
            'red': ['color', 'bright', 'apple', 'blood', 'fire', 'stop'],
            'blue': ['color', 'sky', 'ocean', 'water', 'calm', 'cold'],
            'green': ['color', 'grass', 'nature', 'tree', 'fresh', 'go'],
            'yellow': ['color', 'sun', 'bright', 'banana', 'happy', 'warm'],
            'black': ['color', 'dark', 'night', 'shadow', 'absence'],
            'white': ['color', 'light', 'snow', 'clean', 'pure', 'bright'],

            # Numbers (as concepts)
            'one': ['number', 'single', 'first', 'alone', 'unity'],
            'two': ['number', 'pair', 'couple', 'double', 'second'],
            'three': ['number', 'triple', 'third', 'few'],
            'many': ['number', 'lots', 'multiple', 'several', 'much'],
            'few': ['number', 'some', 'little', 'small', 'scarce'],
        }

        result = []
        for word, related in vocabulary.items():
            result.append({
                'word': word,
                'related': related,
                'definitions': []
            })
        return result

    def load_into_brain(self, brain, use_wordnet: bool = True, progress_callback=None) -> Dict:
        """
        Load vocabulary into the brain's semantic memory.

        Args:
            brain: Brain instance with semantic_memory
            use_wordnet: Whether to enhance with WordNet data
            progress_callback: Optional callback(progress, message)

        Returns:
            Statistics about what was loaded
        """
        stats = {
            'words_loaded': 0,
            'associations_created': 0,
        }

        # Get basic vocabulary
        vocab = self.load_basic_vocabulary()
        total = len(vocab)

        for i, entry in enumerate(vocab):
            word = entry['word']
            related = entry['related']

            # Create concept if not exists
            if not brain.semantic_memory.lookup(word):
                brain.semantic_memory.create_concept(word=word)
                stats['words_loaded'] += 1

            # Create related concepts and associations
            for rel_word in related:
                if not brain.semantic_memory.lookup(rel_word):
                    brain.semantic_memory.create_concept(word=rel_word)
                    stats['words_loaded'] += 1

                brain.semantic_memory.learn_association(word, rel_word, strength=0.3)
                stats['associations_created'] += 1

            # Enhance with WordNet if available
            if use_wordnet and self.wordnet_available:
                wn_info = self.get_word_info(word)

                # Add synonyms
                for syn in wn_info.get('synonyms', [])[:3]:
                    if not brain.semantic_memory.lookup(syn):
                        brain.semantic_memory.create_concept(word=syn)
                        stats['words_loaded'] += 1
                    brain.semantic_memory.learn_association(word, syn, strength=0.4)
                    stats['associations_created'] += 1

                # Add hypernyms (broader terms)
                for hyper in wn_info.get('hypernyms', [])[:2]:
                    if not brain.semantic_memory.lookup(hyper):
                        brain.semantic_memory.create_concept(word=hyper)
                        stats['words_loaded'] += 1
                    brain.semantic_memory.learn_association(word, hyper, strength=0.3)
                    stats['associations_created'] += 1

            if progress_callback:
                progress = (i + 1) / total
                progress_callback(progress, f"Loading: {word}")

        return stats


# Standalone function for easy use
def bootstrap_brain(brain, use_wordnet: bool = True) -> Dict:
    """Bootstrap a brain with dictionary vocabulary."""
    loader = DictionaryLoader()
    print(f"WordNet available: {loader.wordnet_available}")
    print("Loading vocabulary into brain...")

    stats = loader.load_into_brain(brain, use_wordnet)

    print(f"Loaded {stats['words_loaded']} words")
    print(f"Created {stats['associations_created']} associations")

    return stats


if __name__ == "__main__":
    # Test
    loader = DictionaryLoader()
    print(f"WordNet available: {loader.wordnet_available}")

    if loader.wordnet_available:
        info = loader.get_word_info("cat")
        print(f"\nWord: cat")
        print(f"Definitions: {info['definitions'][:2]}")
        print(f"Synonyms: {info['synonyms']}")
        print(f"Related: {info['related']}")
