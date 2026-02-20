"""
Knowledge Graph - Structured semantic knowledge representation

Supports typed relationships between concepts:
- IS_A: Category membership (dog IS_A animal)
- HAS_A: Possession/attribute (dog HAS_A tail)
- CAN_DO: Capability (bird CAN_DO fly)
- PART_OF: Composition (wheel PART_OF car)
- CAUSES: Causation (fire CAUSES heat)
- LOCATED_IN: Location (Paris LOCATED_IN France)
- MADE_OF: Material (table MADE_OF wood)
- USED_FOR: Purpose (hammer USED_FOR building)
- OPPOSITE_OF: Antonyms (hot OPPOSITE_OF cold)
- SIMILAR_TO: Synonyms (big SIMILAR_TO large)
"""

import json
import re
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path


@dataclass
class Relationship:
    """A typed relationship between two concepts."""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    source: str = "learned"  # "dictionary", "learned", "inferred"

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


class KnowledgeGraph:
    """
    Graph-based knowledge representation with typed relationships.

    Supports:
    - Adding/querying typed relationships
    - Transitive inference (A IS_A B, B IS_A C => A IS_A C)
    - Relationship extraction from natural language
    - Confidence-weighted knowledge
    """

    # Relationship types
    RELATIONS = {
        'IS_A': ['is a', 'is an', 'are', 'is type of', 'is kind of', 'is a type of'],
        'HAS_A': ['has', 'have', 'has a', 'have a', 'contains', 'includes', 'possesses'],
        'CAN_DO': ['can', 'could', 'able to', 'capable of', 'may', 'might'],
        'PART_OF': ['part of', 'component of', 'belongs to', 'member of'],
        'CAUSES': ['causes', 'leads to', 'results in', 'produces', 'makes', 'creates'],
        'LOCATED_IN': ['in', 'at', 'located in', 'found in', 'lives in', 'situated in'],
        'MADE_OF': ['made of', 'made from', 'composed of', 'consists of', 'built from'],
        'USED_FOR': ['used for', 'used to', 'for', 'helps', 'serves to'],
        'OPPOSITE_OF': ['opposite of', 'antonym of', 'contrary to', 'versus'],
        'SIMILAR_TO': ['similar to', 'like', 'same as', 'synonym of', 'resembles'],
        'BEFORE': ['before', 'prior to', 'precedes'],
        'AFTER': ['after', 'following', 'follows'],
        'BIGGER_THAN': ['bigger than', 'larger than', 'greater than'],
        'SMALLER_THAN': ['smaller than', 'less than', 'shorter than'],
    }

    # Relations that support transitive inference (upward through hierarchy)
    TRANSITIVE_RELATIONS = {'IS_A', 'PART_OF', 'LOCATED_IN'}

    # Relations that inherit downward (if parent has it, child has it)
    INHERITABLE_RELATIONS = {'CAN_DO', 'HAS_A'}

    def __init__(self):
        # Primary storage: subject -> list of relationships
        self.by_subject: Dict[str, List[Relationship]] = defaultdict(list)
        # Index by object for reverse lookups
        self.by_object: Dict[str, List[Relationship]] = defaultdict(list)
        # Index by relation type
        self.by_relation: Dict[str, List[Relationship]] = defaultdict(list)
        # All unique concepts
        self.concepts: Set[str] = set()
        # Inferred relationships (cached)
        self._inferred_cache: Dict[str, List[Relationship]] = {}
        self._cache_valid = False

    def add(self, subject: str, relation: str, obj: str,
            confidence: float = 1.0, source: str = "learned") -> bool:
        """Add a relationship to the knowledge graph."""
        subject = subject.lower().strip()
        obj = obj.lower().strip()
        relation = relation.upper()

        if not subject or not obj or relation not in self.RELATIONS:
            return False

        rel = Relationship(subject, relation, obj, confidence, source)

        # Check for duplicates
        for existing in self.by_subject[subject]:
            if existing == rel:
                # Update confidence if higher
                if confidence > existing.confidence:
                    existing.confidence = confidence
                return False

        # Add to indices
        self.by_subject[subject].append(rel)
        self.by_object[obj].append(rel)
        self.by_relation[relation].append(rel)
        self.concepts.add(subject)
        self.concepts.add(obj)

        # Invalidate inference cache
        self._cache_valid = False

        return True

    def add_from_text(self, subject: str, relation_text: str, obj: str,
                      confidence: float = 0.8) -> bool:
        """Add relationship from natural language relation phrase."""
        relation_text = relation_text.lower().strip()

        # Find matching relation type
        for rel_type, patterns in self.RELATIONS.items():
            for pattern in patterns:
                if pattern in relation_text or relation_text in pattern:
                    return self.add(subject, rel_type, obj, confidence, "learned")

        # Default to generic relation if no match
        return False

    def query(self, subject: Optional[str] = None,
              relation: Optional[str] = None,
              obj: Optional[str] = None,
              include_inferred: bool = True) -> List[Relationship]:
        """Query the knowledge graph with optional filters."""
        results = []

        if subject:
            subject = subject.lower().strip()
            candidates = self.by_subject.get(subject, [])
            if include_inferred:
                candidates = candidates + self._get_inferred(subject)
        elif obj:
            obj = obj.lower().strip()
            candidates = self.by_object.get(obj, [])
        elif relation:
            candidates = self.by_relation.get(relation.upper(), [])
        else:
            # Return all
            candidates = []
            for rels in self.by_subject.values():
                candidates.extend(rels)

        # Apply filters
        for rel in candidates:
            if subject and rel.subject != subject:
                continue
            if relation and rel.relation != relation.upper():
                continue
            if obj and rel.object != obj.lower():
                continue
            results.append(rel)

        # Sort by confidence
        results.sort(key=lambda r: -r.confidence)
        return results

    def get_knowledge_about(self, concept: str) -> Dict:
        """Get all structured knowledge about a concept."""
        concept = concept.lower().strip()

        knowledge = {
            'concept': concept,
            'exists': concept in self.concepts,
            'is_a': [],      # Categories this belongs to
            'has_a': [],     # Things this has
            'can_do': [],    # Things this can do
            'part_of': [],   # Larger things this is part of
            'has_parts': [], # Things that are part of this
            'causes': [],    # Things this causes
            'caused_by': [], # Things that cause this
            'location': [],  # Where this is located
            'contains': [],  # What this contains (location-wise)
            'used_for': [],  # What this is used for
            'made_of': [],   # What this is made of
            'similar_to': [],# Similar concepts
            'opposite_of': [],# Opposite concepts
            'properties': [],# Properties/attributes
        }

        if not knowledge['exists']:
            return knowledge

        # Get direct relationships where concept is subject
        for rel in self.query(subject=concept, include_inferred=True):
            if rel.relation == 'IS_A':
                knowledge['is_a'].append((rel.object, rel.confidence))
            elif rel.relation == 'HAS_A':
                knowledge['has_a'].append((rel.object, rel.confidence))
            elif rel.relation == 'CAN_DO':
                knowledge['can_do'].append((rel.object, rel.confidence))
            elif rel.relation == 'PART_OF':
                knowledge['part_of'].append((rel.object, rel.confidence))
            elif rel.relation == 'CAUSES':
                knowledge['causes'].append((rel.object, rel.confidence))
            elif rel.relation == 'LOCATED_IN':
                knowledge['location'].append((rel.object, rel.confidence))
            elif rel.relation == 'USED_FOR':
                knowledge['used_for'].append((rel.object, rel.confidence))
            elif rel.relation == 'MADE_OF':
                knowledge['made_of'].append((rel.object, rel.confidence))
            elif rel.relation == 'SIMILAR_TO':
                knowledge['similar_to'].append((rel.object, rel.confidence))
            elif rel.relation == 'OPPOSITE_OF':
                knowledge['opposite_of'].append((rel.object, rel.confidence))

        # Get reverse relationships where concept is object
        for rel in self.query(obj=concept):
            if rel.relation == 'PART_OF':
                knowledge['has_parts'].append((rel.subject, rel.confidence))
            elif rel.relation == 'CAUSES':
                knowledge['caused_by'].append((rel.subject, rel.confidence))
            elif rel.relation == 'LOCATED_IN':
                knowledge['contains'].append((rel.subject, rel.confidence))
            elif rel.relation == 'IS_A':
                # Things that are this type
                knowledge['properties'].append((rel.subject, rel.confidence))

        return knowledge

    def _get_inferred(self, subject: str) -> List[Relationship]:
        """Get inferred relationships through transitive and inheritance reasoning."""
        if self._cache_valid and subject in self._inferred_cache:
            return self._inferred_cache[subject]

        inferred = []
        visited = set()

        def traverse_transitive(current: str, relation: str, depth: int = 0):
            """Traverse upward through hierarchy (IS_A, PART_OF, etc.)"""
            if depth > 3 or current in visited:
                return
            visited.add(current)

            for rel in self.by_subject.get(current, []):
                if rel.relation == relation and rel.object not in visited:
                    # Create inferred relationship
                    conf = rel.confidence * (0.9 ** depth)  # Decay confidence
                    if conf > 0.3:  # Minimum confidence threshold
                        inferred.append(Relationship(
                            subject, relation, rel.object,
                            conf, "inferred"
                        ))
                    traverse_transitive(rel.object, relation, depth + 1)

        # Apply transitive inference for supported relations
        for relation in self.TRANSITIVE_RELATIONS:
            visited.clear()
            traverse_transitive(subject, relation)

        # Apply inheritance: get capabilities/properties from parent categories
        # First, find all categories this subject belongs to (including inferred)
        categories = set()
        visited.clear()

        def collect_categories(current: str, depth: int = 0):
            if depth > 3 or current in visited:
                return
            visited.add(current)
            for rel in self.by_subject.get(current, []):
                if rel.relation == 'IS_A':
                    categories.add(rel.object)
                    collect_categories(rel.object, depth + 1)

        collect_categories(subject)

        # Now inherit CAN_DO and HAS_A from all categories
        for category in categories:
            for rel in self.by_subject.get(category, []):
                if rel.relation in self.INHERITABLE_RELATIONS:
                    # Check if subject already has this directly
                    already_has = any(
                        r.relation == rel.relation and r.object == rel.object
                        for r in self.by_subject.get(subject, [])
                    )
                    if not already_has:
                        conf = rel.confidence * 0.85  # Slight confidence reduction for inheritance
                        if conf > 0.3:
                            inferred.append(Relationship(
                                subject, rel.relation, rel.object,
                                conf, "inherited"
                            ))

        self._inferred_cache[subject] = inferred
        return inferred

    def _normalize_word(self, word: str) -> str:
        """Normalize a word (singularize, lowercase)."""
        word = word.lower().strip()
        # Simple pluralization rules
        if word.endswith('ies') and len(word) > 4:
            return word[:-3] + 'y'  # berries -> berry
        if word.endswith('es') and len(word) > 3:
            if word.endswith('sses') or word.endswith('shes') or word.endswith('ches'):
                return word[:-2]  # glasses -> glass
            return word[:-1]  # goes -> go (rough)
        if word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
            return word[:-1]  # cats -> cat
        return word

    def extract_relationships(self, text: str) -> List[Tuple[str, str, str, float]]:
        """Extract relationships from natural language text."""
        extracted = []
        text = text.lower()

        # Comprehensive pattern-based extraction
        patterns = [
            # IS_A patterns
            (r'(\w+)\s+(?:is|are)\s+(?:a|an)\s+(\w+)', 'IS_A', 0.9),
            (r'(\w+)\s+are\s+(\w+)', 'IS_A', 0.85),  # "Whales are mammals"
            (r'(\w+),?\s+(?:a|an)\s+(\w+)', 'IS_A', 0.7),  # "Dogs, a mammal"
            (r'the\s+(\w+)\s+is\s+(?:a|an)\s+(\w+)', 'IS_A', 0.9),  # "The cat is an animal"
            (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(?:type|kind|form|sort)\s+of\s+(\w+)', 'IS_A', 0.95),
            (r'(\w+)\s+(?:is|are)\s+(?:considered|classified|known)\s+(?:as|to\s+be)\s+(?:a|an)?\s*(\w+)', 'IS_A', 0.85),
            (r'(\w+)\s+belongs?\s+to\s+(?:the\s+)?(\w+)', 'IS_A', 0.8),

            # HAS_A patterns
            (r'(\w+)\s+(?:has|have)\s+(?:a|an)?\s*(\w+)', 'HAS_A', 0.8),
            (r'(\w+)\s+(?:contains?|includes?)\s+(?:a|an)?\s*(\w+)', 'HAS_A', 0.8),
            (r'(\w+)\s+(?:is|are)\s+(?:characterized|distinguished)\s+by\s+(?:their\s+)?(\w+)', 'HAS_A', 0.75),
            (r'(\w+)\s+(?:possess|possesses)\s+(\w+)', 'HAS_A', 0.85),
            (r'the\s+(\w+)\s+of\s+(?:a|an|the)\s+(\w+)', 'HAS_A', 0.6),  # "the tail of a dog" -> dog HAS tail

            # CAN_DO patterns
            (r'(\w+)\s+can\s+(\w+)', 'CAN_DO', 0.8),
            (r'(\w+)\s+(?:is|are)\s+able\s+to\s+(\w+)', 'CAN_DO', 0.85),
            (r'(\w+)\s+(?:is|are)\s+capable\s+of\s+(\w+)', 'CAN_DO', 0.85),
            (r'(\w+)\s+(?:may|might|could)\s+(\w+)', 'CAN_DO', 0.6),
            (r'(\w+)\s+(?:often|usually|typically|commonly)\s+(\w+)', 'CAN_DO', 0.7),

            # LOCATED_IN patterns
            (r'(\w+)\s+(?:is|are|lives?|live|located|found|exists?)\s+in\s+(\w+)', 'LOCATED_IN', 0.8),
            (r'(\w+)\s+(?:is|are)\s+native\s+to\s+(\w+)', 'LOCATED_IN', 0.85),
            (r'(\w+)\s+(?:inhabits?|occupies?)\s+(\w+)', 'LOCATED_IN', 0.8),
            (r'in\s+(\w+),?\s+(?:there\s+)?(?:is|are)\s+(\w+)', 'LOCATED_IN', 0.7),  # reversed

            # MADE_OF patterns
            (r'(\w+)\s+(?:is|are)\s+made\s+(?:of|from|out\s+of)\s+(\w+)', 'MADE_OF', 0.9),
            (r'(\w+)\s+(?:consists?|composed)\s+(?:of|mainly\s+of)\s+(\w+)', 'MADE_OF', 0.85),
            (r'(\w+)\s+(?:contains?|includes?)\s+(\w+)', 'MADE_OF', 0.6),

            # CAUSES patterns
            (r'(\w+)\s+causes?\s+(\w+)', 'CAUSES', 0.8),
            (r'(\w+)\s+(?:leads?|results?)\s+(?:to|in)\s+(\w+)', 'CAUSES', 0.8),
            (r'(\w+)\s+(?:produces?|creates?|generates?)\s+(\w+)', 'CAUSES', 0.75),
            (r'(\w+)\s+(?:is|are)\s+(?:the\s+)?(?:cause|source)\s+of\s+(\w+)', 'CAUSES', 0.85),

            # USED_FOR patterns
            (r'(\w+)\s+(?:is|are)\s+used\s+(?:for|to|in)\s+(\w+)', 'USED_FOR', 0.8),
            (r'(\w+)\s+(?:helps?|assists?)\s+(?:with|in)?\s*(\w+)', 'USED_FOR', 0.7),
            (r'(\w+)\s+(?:is|are)\s+(?:useful|helpful)\s+for\s+(\w+)', 'USED_FOR', 0.75),
            (r'use\s+(\w+)\s+(?:for|to)\s+(\w+)', 'USED_FOR', 0.8),

            # PART_OF patterns
            (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?part\s+of\s+(\w+)', 'PART_OF', 0.9),
            (r'(\w+)\s+(?:is|are)\s+(?:a\s+)?(?:component|element|member)\s+of\s+(\w+)', 'PART_OF', 0.85),
            (r'(\w+)\s+belongs?\s+to\s+(\w+)', 'PART_OF', 0.7),

            # SIMILAR_TO patterns
            (r'(\w+)\s+(?:is|are)\s+(?:similar|comparable)\s+to\s+(\w+)', 'SIMILAR_TO', 0.8),
            (r'(\w+)\s+(?:is|are)\s+like\s+(\w+)', 'SIMILAR_TO', 0.75),
            (r'(\w+)\s+(?:resembles?)\s+(\w+)', 'SIMILAR_TO', 0.8),

            # OPPOSITE_OF patterns
            (r'(\w+)\s+(?:is|are)\s+(?:the\s+)?opposite\s+of\s+(\w+)', 'OPPOSITE_OF', 0.9),
            (r'(\w+)\s+(?:is|are)\s+(?:the\s+)?(?:contrary|antithesis)\s+of\s+(\w+)', 'OPPOSITE_OF', 0.85),
        ]

        for pattern, relation, confidence in patterns:
            for match in re.finditer(pattern, text):
                subject = self._normalize_word(match.group(1))
                obj = self._normalize_word(match.group(2))
                # Filter out stop words and short words
                if len(subject) > 2 and len(obj) > 2:
                    extracted.append((subject, relation, obj, confidence))

        return extracted

    def learn_from_text(self, text: str) -> int:
        """Learn structured relationships from text."""
        relationships = self.extract_relationships(text)
        count = 0
        for subject, relation, obj, confidence in relationships:
            if self.add(subject, relation, obj, confidence, "learned"):
                count += 1
        return count

    def generate_description(self, concept: str) -> str:
        """Generate a natural language description of a concept."""
        knowledge = self.get_knowledge_about(concept)

        if not knowledge['exists']:
            return f"I don't have information about '{concept}'."

        parts = []

        # Category - deduplicate and use proper articles
        if knowledge['is_a']:
            categories = list(dict.fromkeys([cat for cat, conf in knowledge['is_a'] if conf > 0.5]))[:2]
            if categories:
                # Use proper article (a/an)
                cat_str = categories[0]
                article = 'an' if cat_str[0].lower() in 'aeiou' else 'a'
                if len(categories) > 1:
                    parts.append(f"{concept.capitalize()} is {article} {categories[0]} and {categories[1]}")
                else:
                    parts.append(f"{concept.capitalize()} is {article} {categories[0]}")

        # Properties/attributes - deduplicate and pluralize
        if knowledge['has_a']:
            attrs = list(dict.fromkeys([attr for attr, conf in knowledge['has_a'] if conf > 0.5]))[:3]
            if attrs:
                # Add articles for single items
                if len(attrs) == 1:
                    article = 'an' if attrs[0][0].lower() in 'aeiou' else 'a'
                    parts.append(f"It has {article} {attrs[0]}")
                else:
                    parts.append(f"It has {', '.join(attrs[:-1])} and {attrs[-1]}")

        # Capabilities - deduplicate
        if knowledge['can_do']:
            caps = list(dict.fromkeys([cap for cap, conf in knowledge['can_do'] if conf > 0.5]))[:3]
            if caps:
                if len(caps) == 1:
                    parts.append(f"It can {caps[0]}")
                else:
                    parts.append(f"It can {', '.join(caps[:-1])} and {caps[-1]}")

        # Location - deduplicate
        if knowledge['location']:
            locs = list(dict.fromkeys([loc for loc, conf in knowledge['location'] if conf > 0.5]))[:2]
            if locs:
                if len(locs) == 1:
                    parts.append(f"It is found in {locs[0]}")
                else:
                    parts.append(f"It is found in {' and '.join(locs)}")

        # Composition
        if knowledge['made_of']:
            mats = [mat for mat, conf in knowledge['made_of'] if conf > 0.5][:2]
            if mats:
                parts.append(f"Made of {', '.join(mats)}")

        # Purpose
        if knowledge['used_for']:
            uses = [use for use, conf in knowledge['used_for'] if conf > 0.5][:2]
            if uses:
                parts.append(f"Used for {', '.join(uses)}")

        # Causation
        if knowledge['causes']:
            effects = [eff for eff, conf in knowledge['causes'] if conf > 0.5][:2]
            if effects:
                parts.append(f"Causes {', '.join(effects)}")

        if parts:
            return '. '.join(parts) + '.'
        else:
            return f"I know '{concept}' exists but I'm still learning about it."

    def answer_question(self, question: str) -> Optional[str]:
        """Try to answer a question using the knowledge graph."""
        question = question.lower().strip()

        # "What is X?"
        match = re.search(r'what\s+(?:is|are)\s+(?:a|an|the)?\s*(\w+)', question)
        if match:
            concept = match.group(1)
            return self.generate_description(concept)

        # "Is X a Y?"
        match = re.search(r'is\s+(?:a|an|the)?\s*(\w+)\s+(?:a|an)\s+(\w+)', question)
        if match:
            subject, category = match.group(1), match.group(2)
            rels = self.query(subject=subject, relation='IS_A', obj=category)
            if rels:
                return f"Yes, {subject} is a {category} (confidence: {rels[0].confidence:.0%})."
            else:
                return f"I'm not sure if {subject} is a {category}."

        # "Can X Y?"
        match = re.search(r'can\s+(?:a|an|the)?\s*(\w+)\s+(\w+)', question)
        if match:
            subject, action = match.group(1), match.group(2)
            rels = self.query(subject=subject, relation='CAN_DO', obj=action)
            if rels:
                return f"Yes, {subject} can {action} (confidence: {rels[0].confidence:.0%})."
            else:
                return f"I don't know if {subject} can {action}."

        # "What can X do?"
        match = re.search(r'what\s+can\s+(?:a|an|the)?\s*(\w+)\s+do', question)
        if match:
            subject = match.group(1)
            rels = self.query(subject=subject, relation='CAN_DO')
            if rels:
                actions = [r.object for r in rels[:5]]
                return f"{subject.capitalize()} can: {', '.join(actions)}."
            else:
                return f"I don't know what {subject} can do."

        # "What does X have?"
        match = re.search(r'what\s+does\s+(?:a|an|the)?\s*(\w+)\s+have', question)
        if match:
            subject = match.group(1)
            rels = self.query(subject=subject, relation='HAS_A')
            if rels:
                attrs = [r.object for r in rels[:5]]
                return f"{subject.capitalize()} has: {', '.join(attrs)}."
            else:
                return f"I don't know what {subject} has."

        return None

    def save(self, path: str):
        """Save the knowledge graph to a file."""
        data = {
            'relationships': [
                {
                    'subject': r.subject,
                    'relation': r.relation,
                    'object': r.object,
                    'confidence': r.confidence,
                    'source': r.source
                }
                for rels in self.by_subject.values()
                for r in rels
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> bool:
        """Load the knowledge graph from a file."""
        path = Path(path)
        if not path.exists():
            return False

        with open(path) as f:
            data = json.load(f)

        for rel in data.get('relationships', []):
            self.add(
                rel['subject'],
                rel['relation'],
                rel['object'],
                rel.get('confidence', 1.0),
                rel.get('source', 'loaded')
            )

        return True

    def get_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        return {
            'concepts': len(self.concepts),
            'relationships': sum(len(rels) for rels in self.by_subject.values()),
            'by_type': {
                rel_type: len(rels)
                for rel_type, rels in self.by_relation.items()
            }
        }


def load_default_knowledge() -> KnowledgeGraph:
    """Create a knowledge graph with default world knowledge."""
    kg = KnowledgeGraph()

    # Animals
    animals = [
        ('cat', 'IS_A', 'animal'), ('cat', 'IS_A', 'mammal'), ('cat', 'IS_A', 'pet'),
        ('cat', 'HAS_A', 'fur'), ('cat', 'HAS_A', 'tail'), ('cat', 'HAS_A', 'whiskers'),
        ('cat', 'CAN_DO', 'meow'), ('cat', 'CAN_DO', 'purr'), ('cat', 'CAN_DO', 'climb'),

        ('dog', 'IS_A', 'animal'), ('dog', 'IS_A', 'mammal'), ('dog', 'IS_A', 'pet'),
        ('dog', 'HAS_A', 'fur'), ('dog', 'HAS_A', 'tail'), ('dog', 'HAS_A', 'paws'),
        ('dog', 'CAN_DO', 'bark'), ('dog', 'CAN_DO', 'run'), ('dog', 'CAN_DO', 'fetch'),

        ('bird', 'IS_A', 'animal'), ('bird', 'HAS_A', 'feathers'), ('bird', 'HAS_A', 'wings'),
        ('bird', 'HAS_A', 'beak'), ('bird', 'CAN_DO', 'fly'), ('bird', 'CAN_DO', 'sing'),

        ('fish', 'IS_A', 'animal'), ('fish', 'HAS_A', 'fins'), ('fish', 'HAS_A', 'scales'),
        ('fish', 'CAN_DO', 'swim'), ('fish', 'LOCATED_IN', 'water'),

        ('elephant', 'IS_A', 'animal'), ('elephant', 'IS_A', 'mammal'),
        ('elephant', 'HAS_A', 'trunk'), ('elephant', 'HAS_A', 'tusks'),

        ('lion', 'IS_A', 'animal'), ('lion', 'IS_A', 'mammal'),
        ('lion', 'CAN_DO', 'roar'), ('lion', 'LOCATED_IN', 'africa'),
    ]

    # Nature
    nature = [
        ('tree', 'IS_A', 'plant'), ('tree', 'HAS_A', 'leaves'), ('tree', 'HAS_A', 'trunk'),
        ('tree', 'HAS_A', 'roots'), ('tree', 'MADE_OF', 'wood'),

        ('flower', 'IS_A', 'plant'), ('flower', 'HAS_A', 'petals'),

        ('sun', 'IS_A', 'star'), ('sun', 'CAUSES', 'light'), ('sun', 'CAUSES', 'heat'),

        ('moon', 'LOCATED_IN', 'sky'), ('moon', 'CAUSES', 'tides'),

        ('water', 'IS_A', 'liquid'), ('water', 'MADE_OF', 'hydrogen'),
        ('water', 'MADE_OF', 'oxygen'),

        ('fire', 'CAUSES', 'heat'), ('fire', 'CAUSES', 'light'), ('fire', 'CAUSES', 'smoke'),

        ('ice', 'IS_A', 'solid'), ('ice', 'MADE_OF', 'water'),
    ]

    # Objects
    objects = [
        ('car', 'IS_A', 'vehicle'), ('car', 'HAS_A', 'wheels'), ('car', 'HAS_A', 'engine'),
        ('car', 'USED_FOR', 'transportation'), ('car', 'CAN_DO', 'drive'),

        ('computer', 'IS_A', 'device'), ('computer', 'HAS_A', 'screen'),
        ('computer', 'HAS_A', 'keyboard'), ('computer', 'USED_FOR', 'computing'),

        ('phone', 'IS_A', 'device'), ('phone', 'USED_FOR', 'communication'),

        ('book', 'HAS_A', 'pages'), ('book', 'USED_FOR', 'reading'),
        ('book', 'MADE_OF', 'paper'),

        ('table', 'IS_A', 'furniture'), ('table', 'MADE_OF', 'wood'),
        ('table', 'HAS_A', 'legs'), ('table', 'USED_FOR', 'eating'),

        ('chair', 'IS_A', 'furniture'), ('chair', 'USED_FOR', 'sitting'),
        ('chair', 'HAS_A', 'legs'),
    ]

    # Abstract concepts
    abstract = [
        ('science', 'IS_A', 'knowledge'), ('science', 'USED_FOR', 'understanding'),

        ('mathematics', 'IS_A', 'science'), ('mathematics', 'USED_FOR', 'calculation'),

        ('physics', 'IS_A', 'science'), ('physics', 'PART_OF', 'science'),

        ('biology', 'IS_A', 'science'), ('biology', 'PART_OF', 'science'),

        ('language', 'USED_FOR', 'communication'),
    ]

    # Properties
    properties = [
        ('hot', 'OPPOSITE_OF', 'cold'),
        ('big', 'OPPOSITE_OF', 'small'),
        ('fast', 'OPPOSITE_OF', 'slow'),
        ('light', 'OPPOSITE_OF', 'dark'),
        ('good', 'OPPOSITE_OF', 'bad'),
        ('happy', 'OPPOSITE_OF', 'sad'),
        ('wet', 'OPPOSITE_OF', 'dry'),
        ('old', 'OPPOSITE_OF', 'new'),
        ('large', 'SIMILAR_TO', 'big'),
        ('tiny', 'SIMILAR_TO', 'small'),
        ('quick', 'SIMILAR_TO', 'fast'),
    ]

    # American English Grammar - Parts of Speech
    grammar_parts = [
        # Nouns
        ('noun', 'IS_A', 'word'), ('noun', 'USED_FOR', 'naming'),
        ('noun', 'HAS_A', 'singular'), ('noun', 'HAS_A', 'plural'),
        ('pronoun', 'IS_A', 'word'), ('pronoun', 'USED_FOR', 'replacement'),

        # Verbs
        ('verb', 'IS_A', 'word'), ('verb', 'USED_FOR', 'action'),
        ('verb', 'HAS_A', 'tense'), ('verb', 'HAS_A', 'conjugation'),

        # Adjectives & Adverbs
        ('adjective', 'IS_A', 'word'), ('adjective', 'USED_FOR', 'description'),
        ('adverb', 'IS_A', 'word'), ('adverb', 'USED_FOR', 'modification'),

        # Other parts
        ('preposition', 'IS_A', 'word'), ('conjunction', 'IS_A', 'word'),
        ('article', 'IS_A', 'word'), ('interjection', 'IS_A', 'word'),
    ]

    # American English - Article Usage
    grammar_articles = [
        ('a', 'IS_A', 'article'), ('an', 'IS_A', 'article'), ('the', 'IS_A', 'article'),
        ('a', 'USED_FOR', 'consonant'), ('an', 'USED_FOR', 'vowel'),
        ('the', 'USED_FOR', 'specific'),
    ]

    # American English - Verb Tenses
    grammar_tenses = [
        ('present', 'IS_A', 'tense'), ('past', 'IS_A', 'tense'), ('future', 'IS_A', 'tense'),
        ('progressive', 'IS_A', 'aspect'), ('perfect', 'IS_A', 'aspect'),
        ('simple', 'IS_A', 'aspect'),
    ]

    # American English - Common Patterns
    grammar_patterns = [
        # American English prefers simple past over present perfect
        ('ate', 'IS_A', 'past'), ('eaten', 'IS_A', 'participle'),
        ('went', 'IS_A', 'past'), ('gone', 'IS_A', 'participle'),
        ('got', 'IS_A', 'past'), ('gotten', 'IS_A', 'participle'),
        ('learned', 'IS_A', 'past'), ('dreamed', 'IS_A', 'past'),
        ('burned', 'IS_A', 'past'), ('spelled', 'IS_A', 'past'),

        # American English modals
        ('will', 'IS_A', 'modal'), ('would', 'IS_A', 'modal'),
        ('can', 'IS_A', 'modal'), ('could', 'IS_A', 'modal'),
        ('should', 'IS_A', 'modal'), ('must', 'IS_A', 'modal'),
    ]

    # Sentence structure
    grammar_structure = [
        ('sentence', 'HAS_A', 'subject'), ('sentence', 'HAS_A', 'predicate'),
        ('sentence', 'HAS_A', 'verb'), ('clause', 'IS_A', 'phrase'),
        ('phrase', 'PART_OF', 'sentence'), ('subject', 'PART_OF', 'sentence'),
        ('predicate', 'PART_OF', 'sentence'), ('object', 'PART_OF', 'sentence'),
    ]

    # Common American English expressions and idioms
    idioms = [
        # Common expressions
        ('hello', 'IS_A', 'greeting'), ('goodbye', 'IS_A', 'farewell'),
        ('thanks', 'SIMILAR_TO', 'gratitude'), ('please', 'IS_A', 'politeness'),
        ('sorry', 'IS_A', 'apology'), ('welcome', 'IS_A', 'greeting'),

        # Question words
        ('what', 'IS_A', 'question'), ('where', 'IS_A', 'question'),
        ('when', 'IS_A', 'question'), ('why', 'IS_A', 'question'),
        ('how', 'IS_A', 'question'), ('who', 'IS_A', 'question'),

        # Time expressions
        ('today', 'IS_A', 'time'), ('tomorrow', 'IS_A', 'time'),
        ('yesterday', 'IS_A', 'time'), ('now', 'IS_A', 'time'),
        ('always', 'IS_A', 'frequency'), ('never', 'IS_A', 'frequency'),
        ('sometimes', 'IS_A', 'frequency'), ('often', 'IS_A', 'frequency'),

        # Affirmation and negation
        ('yes', 'IS_A', 'affirmation'), ('no', 'IS_A', 'negation'),
        ('maybe', 'IS_A', 'uncertainty'), ('perhaps', 'SIMILAR_TO', 'maybe'),

        # Connectors
        ('and', 'IS_A', 'conjunction'), ('but', 'IS_A', 'conjunction'),
        ('or', 'IS_A', 'conjunction'), ('because', 'IS_A', 'conjunction'),
        ('therefore', 'IS_A', 'conjunction'), ('however', 'IS_A', 'conjunction'),
    ]

    # Pronouns
    pronouns = [
        ('i', 'IS_A', 'pronoun'), ('you', 'IS_A', 'pronoun'),
        ('he', 'IS_A', 'pronoun'), ('she', 'IS_A', 'pronoun'),
        ('it', 'IS_A', 'pronoun'), ('we', 'IS_A', 'pronoun'),
        ('they', 'IS_A', 'pronoun'), ('me', 'IS_A', 'pronoun'),
        ('him', 'IS_A', 'pronoun'), ('her', 'IS_A', 'pronoun'),
        ('us', 'IS_A', 'pronoun'), ('them', 'IS_A', 'pronoun'),
    ]

    # Common verbs
    common_verbs = [
        ('be', 'IS_A', 'verb'), ('have', 'IS_A', 'verb'),
        ('do', 'IS_A', 'verb'), ('say', 'IS_A', 'verb'),
        ('go', 'IS_A', 'verb'), ('get', 'IS_A', 'verb'),
        ('make', 'IS_A', 'verb'), ('know', 'IS_A', 'verb'),
        ('think', 'IS_A', 'verb'), ('take', 'IS_A', 'verb'),
        ('see', 'IS_A', 'verb'), ('come', 'IS_A', 'verb'),
        ('want', 'IS_A', 'verb'), ('look', 'IS_A', 'verb'),
        ('use', 'IS_A', 'verb'), ('find', 'IS_A', 'verb'),
        ('give', 'IS_A', 'verb'), ('tell', 'IS_A', 'verb'),
        ('work', 'IS_A', 'verb'), ('call', 'IS_A', 'verb'),
    ]

    # Add all knowledge
    all_knowledge = (animals + nature + objects + abstract + properties +
                     grammar_parts + grammar_articles + grammar_tenses +
                     grammar_patterns + grammar_structure + idioms +
                     pronouns + common_verbs)
    for subject, relation, obj in all_knowledge:
        kg.add(subject, relation, obj, 0.95, "dictionary")

    return kg


class DeductiveReasoner:
    """
    Deductive reasoning engine that chains knowledge to answer complex questions.

    Supports:
    - Syllogistic reasoning (All A are B, X is A, therefore X is B)
    - Property inheritance
    - Causal chains
    - Counterfactual reasoning
    """

    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph

    def reason(self, question: str) -> Optional[Dict]:
        """
        Apply deductive reasoning to answer a question.

        Returns dict with:
        - answer: The answer text
        - confidence: Confidence score
        - reasoning_chain: Steps used to derive the answer
        """
        question = question.lower().strip()

        # Try different reasoning strategies
        result = self._try_category_reasoning(question)
        if result:
            return result

        result = self._try_capability_reasoning(question)
        if result:
            return result

        result = self._try_property_reasoning(question)
        if result:
            return result

        result = self._try_causal_reasoning(question)
        if result:
            return result

        result = self._try_comparison_reasoning(question)
        if result:
            return result

        result = self._try_counterfactual_reasoning(question)
        if result:
            return result

        return None

    def _try_counterfactual_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about hypothetical scenarios."""
        # "What if X had/was/could Y?"
        match = re.search(r'what\s+(?:if|would\s+happen\s+if)\s+(?:a|an|the)?\s*(\w+)\s+(?:had|was|were|could|did)\s+(?:not\s+)?(?:have\s+)?(?:a|an)?\s*(\w+)', question)
        if match:
            subject, condition = match.group(1), match.group(2)
            subject = self.kg._normalize_word(subject)
            condition = self.kg._normalize_word(condition)

            # Check if this is about removing a property/capability
            is_negation = 'not' in question.lower() or "didn't" in question.lower() or "couldn't" in question.lower()

            chain = []
            consequences = []

            if is_negation:
                # Reasoning about lacking something
                # Check what depends on this property
                knowledge = self.kg.get_knowledge_about(subject)

                # If subject loses a capability, what effects?
                if any(cap == condition for cap, _ in knowledge.get('can_do', [])):
                    # Find things that depend on this capability
                    chain.append(f"{subject} normally CAN_DO {condition}")
                    chain.append(f"If {subject} could not {condition}:")
                    consequences.append(f"it would lose its ability to {condition}")
                    # Check what this capability enables
                    effects = self.kg.query(subject=condition, relation='CAUSES')
                    for eff in effects[:2]:
                        consequences.append(f"it couldn't cause {eff.object}")
                        chain.append(f"{condition} CAUSES {eff.object}")

                elif any(prop == condition for prop, _ in knowledge.get('has_a', [])):
                    chain.append(f"{subject} normally HAS_A {condition}")
                    chain.append(f"If {subject} didn't have {condition}:")
                    consequences.append(f"it would lack {condition}")

            else:
                # Reasoning about gaining something
                knowledge = self.kg.get_knowledge_about(subject)

                # What if subject gained a new capability?
                if condition not in [cap for cap, _ in knowledge.get('can_do', [])]:
                    chain.append(f"{subject} normally cannot {condition}")
                    chain.append(f"If {subject} could {condition}:")
                    # Check what this capability causes
                    effects = self.kg.query(subject=condition, relation='CAUSES')
                    if effects:
                        for eff in effects[:2]:
                            consequences.append(f"it could cause {eff.object}")
                            chain.append(f"{condition} CAUSES {eff.object}")

                    # Check what typically can do this
                    others_who_can = self.kg.query(relation='CAN_DO', obj=condition)
                    if others_who_can:
                        others = list(set(r.subject for r in others_who_can[:3]))
                        consequences.append(f"it would be similar to {', '.join(others)} in this regard")

            if consequences:
                return {
                    'answer': f"If {subject} {'could not' if is_negation else 'could'} {condition}: {'; '.join(consequences)}.",
                    'confidence': 0.7,
                    'reasoning_chain': chain
                }

        # "What would happen if X?" / "What if X?"
        match = re.search(r'what\s+(?:would\s+happen\s+)?if\s+(?:there\s+(?:was|were)\s+)?(?:no\s+)?(\w+)', question)
        if match:
            concept = self.kg._normalize_word(match.group(1))
            is_absence = 'no ' in question.lower() or 'without' in question.lower()

            knowledge = self.kg.get_knowledge_about(concept)
            chain = []
            consequences = []

            if knowledge['exists']:
                # What does this concept cause?
                if knowledge.get('causes'):
                    for effect, conf in knowledge['causes'][:3]:
                        if is_absence:
                            consequences.append(f"no {effect}")
                        else:
                            consequences.append(effect)
                        chain.append(f"{concept} CAUSES {effect}")

                if consequences:
                    if is_absence:
                        answer = f"Without {concept}, there would be: {', '.join(consequences)}."
                    else:
                        answer = f"If there was {concept}, it would cause: {', '.join(consequences)}."

                    return {
                        'answer': answer,
                        'confidence': 0.75,
                        'reasoning_chain': chain
                    }

        return None

    def _try_category_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about category membership."""
        # "Is X a Y?" / "Are X Y?"
        match = re.search(r'(?:is|are)\s+(?:a|an|the)?\s*(\w+)\s+(?:a|an)\s+(\w+)', question)
        if match:
            subject, category = match.group(1), match.group(2)
            subject = self.kg._normalize_word(subject)
            category = self.kg._normalize_word(category)

            # Direct check
            direct = self.kg.query(subject=subject, relation='IS_A', obj=category)
            if direct:
                return {
                    'answer': f"Yes, {subject} is a {category}.",
                    'confidence': direct[0].confidence,
                    'reasoning_chain': [f"{subject} IS_A {category} (direct knowledge)"]
                }

            # Check through inference
            all_categories = self._get_all_categories(subject)
            if category in all_categories:
                chain = self._build_category_chain(subject, category)
                conf = 0.95 ** len(chain)  # Confidence decreases with chain length
                return {
                    'answer': f"Yes, {subject} is a {category}.",
                    'confidence': conf,
                    'reasoning_chain': chain
                }

            # Check if category is a subcategory of something the subject is
            return {
                'answer': f"I cannot confirm that {subject} is a {category}.",
                'confidence': 0.5,
                'reasoning_chain': ["No direct or inferred relationship found"]
            }

        return None

    def _try_capability_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about capabilities through inheritance."""
        # "Can X Y?"
        match = re.search(r'can\s+(?:a|an|the)?\s*(\w+)\s+(\w+)', question)
        if match:
            subject, action = match.group(1), match.group(2)
            subject = self.kg._normalize_word(subject)
            action = self.kg._normalize_word(action)

            # Direct check
            direct = self.kg.query(subject=subject, relation='CAN_DO', obj=action)
            if direct:
                return {
                    'answer': f"Yes, {subject} can {action}.",
                    'confidence': direct[0].confidence,
                    'reasoning_chain': [f"{subject} CAN_DO {action} (direct knowledge)"]
                }

            # Check inherited capabilities
            categories = self._get_all_categories(subject)
            for cat in categories:
                cat_caps = self.kg.query(subject=cat, relation='CAN_DO', obj=action)
                if cat_caps:
                    chain = self._build_category_chain(subject, cat)
                    chain.append(f"{cat} CAN_DO {action}")
                    chain.append(f"Therefore, {subject} CAN_DO {action} (inherited)")
                    conf = 0.9 ** len(chain)
                    return {
                        'answer': f"Yes, {subject} can {action} (because {subject} is a {cat}, and {cat}s can {action}).",
                        'confidence': conf,
                        'reasoning_chain': chain
                    }

            return {
                'answer': f"I don't know if {subject} can {action}.",
                'confidence': 0.3,
                'reasoning_chain': ["No direct or inherited capability found"]
            }

        return None

    def _try_property_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about properties through inheritance."""
        # "Does X have Y?"
        match = re.search(r'(?:does|do)\s+(?:a|an|the)?\s*(\w+)\s+have\s+(?:a|an)?\s*(\w+)', question)
        if match:
            subject, prop = match.group(1), match.group(2)
            subject = self.kg._normalize_word(subject)
            prop = self.kg._normalize_word(prop)

            # Direct check
            direct = self.kg.query(subject=subject, relation='HAS_A', obj=prop)
            if direct:
                return {
                    'answer': f"Yes, {subject} has {prop}.",
                    'confidence': direct[0].confidence,
                    'reasoning_chain': [f"{subject} HAS_A {prop} (direct knowledge)"]
                }

            # Check inherited properties
            categories = self._get_all_categories(subject)
            for cat in categories:
                cat_props = self.kg.query(subject=cat, relation='HAS_A', obj=prop)
                if cat_props:
                    chain = self._build_category_chain(subject, cat)
                    chain.append(f"{cat} HAS_A {prop}")
                    chain.append(f"Therefore, {subject} HAS_A {prop} (inherited)")
                    conf = 0.9 ** len(chain)
                    return {
                        'answer': f"Yes, {subject} has {prop} (because {subject} is a {cat}, and {cat}s have {prop}).",
                        'confidence': conf,
                        'reasoning_chain': chain
                    }

            return {
                'answer': f"I don't know if {subject} has {prop}.",
                'confidence': 0.3,
                'reasoning_chain': ["No direct or inherited property found"]
            }

        return None

    def _try_causal_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about cause and effect chains."""
        # "What causes X?" / "What does X cause?"
        match = re.search(r'what\s+(?:causes?|leads?\s+to)\s+(\w+)', question)
        if match:
            effect = self.kg._normalize_word(match.group(1))
            causes = self.kg.query(obj=effect, relation='CAUSES')
            if causes:
                cause_list = [c.subject for c in causes[:3]]
                return {
                    'answer': f"{', '.join(cause_list)} can cause {effect}.",
                    'confidence': causes[0].confidence,
                    'reasoning_chain': [f"{c.subject} CAUSES {effect}" for c in causes[:3]]
                }

        match = re.search(r'what\s+does\s+(\w+)\s+cause', question)
        if match:
            cause = self.kg._normalize_word(match.group(1))
            effects = self.kg.query(subject=cause, relation='CAUSES')
            if effects:
                effect_list = [e.object for e in effects[:3]]
                return {
                    'answer': f"{cause.capitalize()} causes {', '.join(effect_list)}.",
                    'confidence': effects[0].confidence,
                    'reasoning_chain': [f"{cause} CAUSES {e.object}" for e in effects[:3]]
                }

        return None

    def _try_comparison_reasoning(self, question: str) -> Optional[Dict]:
        """Reason about similarities and differences."""
        # "How is X similar to Y?" / "What do X and Y have in common?"
        match = re.search(r'(?:how\s+is|are)\s+(\w+)\s+(?:similar|like|comparable)\s+to\s+(\w+)', question)
        if not match:
            match = re.search(r'what\s+do\s+(\w+)\s+and\s+(\w+)\s+have\s+in\s+common', question)

        if match:
            a, b = self.kg._normalize_word(match.group(1)), self.kg._normalize_word(match.group(2))

            # Find common categories
            cats_a = self._get_all_categories(a)
            cats_b = self._get_all_categories(b)
            common_cats = cats_a.intersection(cats_b)

            # Find common capabilities
            caps_a = set(r.object for r in self.kg.query(subject=a, relation='CAN_DO', include_inferred=True))
            caps_b = set(r.object for r in self.kg.query(subject=b, relation='CAN_DO', include_inferred=True))
            common_caps = caps_a.intersection(caps_b)

            # Find common properties
            props_a = set(r.object for r in self.kg.query(subject=a, relation='HAS_A', include_inferred=True))
            props_b = set(r.object for r in self.kg.query(subject=b, relation='HAS_A', include_inferred=True))
            common_props = props_a.intersection(props_b)

            similarities = []
            chain = []

            if common_cats:
                similarities.append(f"both are {list(common_cats)[0]}s")
                chain.append(f"Both {a} and {b} ARE {list(common_cats)[0]}")

            if common_caps:
                similarities.append(f"both can {list(common_caps)[0]}")
                chain.append(f"Both CAN_DO {list(common_caps)[0]}")

            if common_props:
                similarities.append(f"both have {list(common_props)[0]}")
                chain.append(f"Both HAVE {list(common_props)[0]}")

            if similarities:
                return {
                    'answer': f"{a.capitalize()} and {b} are similar: {'; '.join(similarities)}.",
                    'confidence': 0.8,
                    'reasoning_chain': chain
                }

            return {
                'answer': f"I don't see obvious similarities between {a} and {b}.",
                'confidence': 0.4,
                'reasoning_chain': ["No common categories, capabilities, or properties found"]
            }

        return None

    def _get_all_categories(self, subject: str) -> Set[str]:
        """Get all categories the subject belongs to (direct and inferred)."""
        categories = set()
        visited = set()

        def traverse(current: str):
            if current in visited:
                return
            visited.add(current)

            for rel in self.kg.by_subject.get(current, []):
                if rel.relation == 'IS_A':
                    categories.add(rel.object)
                    traverse(rel.object)

        traverse(subject)
        return categories

    def _build_category_chain(self, subject: str, target_category: str) -> List[str]:
        """Build the reasoning chain from subject to target category."""
        chain = []
        visited = set()

        def find_path(current: str, path: List[str]) -> bool:
            if current == target_category:
                chain.extend(path)
                return True
            if current in visited:
                return False
            visited.add(current)

            for rel in self.kg.by_subject.get(current, []):
                if rel.relation == 'IS_A':
                    new_path = path + [f"{current} IS_A {rel.object}"]
                    if find_path(rel.object, new_path):
                        return True

            return False

        find_path(subject, [])
        return chain

    def explain_concept(self, concept: str) -> str:
        """Generate a detailed explanation of a concept using reasoning."""
        concept = self.kg._normalize_word(concept)
        knowledge = self.kg.get_knowledge_about(concept)

        if not knowledge['exists']:
            return f"I don't have information about '{concept}'."

        parts = []

        # Category with reasoning
        if knowledge['is_a']:
            categories = list(dict.fromkeys([cat for cat, _ in knowledge['is_a']]))[:2]
            parts.append(f"{concept.capitalize()} is a {' and '.join(categories)}")

        # Properties (including inherited)
        all_props = set()
        for prop, conf in knowledge['has_a']:
            all_props.add(prop)
        # Add inherited properties
        for rel in self.kg._get_inferred(concept):
            if rel.relation == 'HAS_A':
                all_props.add(rel.object)
        if all_props:
            parts.append(f"It has {', '.join(list(all_props)[:4])}")

        # Capabilities (including inherited)
        all_caps = set()
        for cap, conf in knowledge['can_do']:
            all_caps.add(cap)
        for rel in self.kg._get_inferred(concept):
            if rel.relation == 'CAN_DO':
                all_caps.add(rel.object)
        if all_caps:
            parts.append(f"It can {', '.join(list(all_caps)[:4])}")

        # Location
        if knowledge['location']:
            locs = list(dict.fromkeys([loc for loc, _ in knowledge['location']]))[:2]
            parts.append(f"Found in {', '.join(locs)}")

        # Causal relationships
        if knowledge['causes']:
            effects = list(dict.fromkeys([e for e, _ in knowledge['causes']]))[:2]
            parts.append(f"It causes {', '.join(effects)}")

        if parts:
            return '. '.join(parts) + '.'
        return f"I know '{concept}' exists but need more information about it."


if __name__ == "__main__":
    # Test the knowledge graph
    print("Testing Knowledge Graph")
    print("=" * 50)

    kg = load_default_knowledge()
    stats = kg.get_stats()
    print(f"Loaded: {stats['concepts']} concepts, {stats['relationships']} relationships")
    print(f"By type: {stats['by_type']}")

    print("\n--- Testing Queries ---")

    # Test description generation
    for concept in ['cat', 'dog', 'car', 'science']:
        print(f"\n{concept.upper()}:")
        print(kg.generate_description(concept))

    print("\n--- Testing Questions ---")

    questions = [
        "What is a cat?",
        "Is a dog an animal?",
        "Can a bird fly?",
        "What can a dog do?",
        "What does a car have?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = kg.answer_question(q)
        print(f"A: {answer}")

    print("\n--- Testing Inference ---")
    # Test transitive inference
    kg.add("poodle", "IS_A", "dog", 0.99)
    print("\nAdded: poodle IS_A dog")

    # Query with inference
    rels = kg.query(subject="poodle", include_inferred=True)
    print("Poodle relationships (including inferred):")
    for rel in rels:
        print(f"  {rel.subject} {rel.relation} {rel.object} ({rel.confidence:.0%}, {rel.source})")

    print("\n--- Testing Text Learning ---")
    text = "A whale is a mammal. Whales can swim. Whales have fins. Whales live in oceans."
    count = kg.learn_from_text(text)
    print(f"Learned {count} relationships from text")
    print(kg.generate_description("whale"))
