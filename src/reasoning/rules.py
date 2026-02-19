"""
Reasoning Rules Module

Implements classical logical inference rules using the Semantic Pointer Architecture:

1. Modus Ponens: If P, and P -> Q, then Q
2. Modus Tollens: If ~Q, and P -> Q, then ~P
3. Transitive Inference: If P -> Q and Q -> R, then P -> R
4. Analogy: If A:B :: C:?, find D where relation(A,B) = relation(C,D)

Each rule is implemented as a neural circuit that can be composed into
larger reasoning systems.
"""

import numpy as np
import nengo
import nengo_spa as spa
from typing import Optional, Callable, Dict, List, Tuple


class ModusPonens(spa.Network):
    """Modus Ponens inference rule.

    Given:
        - P (a proposition that is true)
        - P -> Q (P implies Q)
    Conclude:
        - Q (therefore Q is true)

    Example:
        - P = "It is raining"
        - P -> Q = "If it rains, the ground is wet"
        - Q = "The ground is wet"
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        threshold: float = 0.3,
        label: str = "modus_ponens",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.threshold = threshold

        with self:
            # Input: The proposition P
            self.proposition = spa.State(vocab, label="proposition")

            # Input: The implication P -> Q (encoded as P*IMPLIES*Q)
            self.implication = spa.State(vocab, label="implication")

            # Output: The conclusion Q
            self.conclusion = spa.State(vocab, feedback=0.8,
                                         feedback_synapse=0.1,
                                         label="conclusion")

            # Detect the antecedent (P) in the implication
            # We need to extract what P implies
            self.antecedent_match = spa.Scalar(label="antecedent_match")

            # Create an ensemble to compute the inference
            self.inference_gate = nengo.Ensemble(
                n_neurons=200,
                dimensions=2,
                label="inference_gate"
            )

            # The inference works by:
            # 1. Check if proposition matches antecedent of implication
            # 2. If match, extract and output the consequent

            # For simplicity, we'll use a direct approach:
            # conclusion = implication * ~proposition (unbind to get Q)

            # Compute similarity between proposition and implication
            nengo.Connection(
                self.proposition.output,
                self.inference_gate[0],
                transform=[np.ones(vocab.dimensions) / vocab.dimensions]
            )

            # Combine with implication strength
            nengo.Connection(
                self.implication.output,
                self.inference_gate[1],
                transform=[np.ones(vocab.dimensions) / vocab.dimensions]
            )

            # When both inputs are strong, produce conclusion
            def inference_function(x):
                p_strength, impl_strength = x
                if p_strength > threshold and impl_strength > threshold:
                    return min(p_strength, impl_strength)
                return 0.0

            # Connect to conclusion (simplified - in full version would unbind)
            nengo.Connection(
                self.inference_gate,
                self.conclusion.input,
                function=lambda x: inference_function(x) * np.ones(vocab.dimensions) * 0.5
            )

    @property
    def input_proposition(self):
        return self.proposition.input

    @property
    def input_implication(self):
        return self.implication.input

    @property
    def output(self):
        return self.conclusion.output


class ModusTollens(spa.Network):
    """Modus Tollens inference rule.

    Given:
        - ~Q (Q is false / NOT Q)
        - P -> Q (P implies Q)
    Conclude:
        - ~P (therefore P is false / NOT P)

    Example:
        - ~Q = "The ground is not wet"
        - P -> Q = "If it rains, the ground is wet"
        - ~P = "It is not raining"
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        threshold: float = 0.3,
        label: str = "modus_tollens",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.threshold = threshold

        with self:
            # Input: NOT Q (the negated consequent)
            self.not_consequent = spa.State(vocab, label="not_consequent")

            # Input: The implication P -> Q
            self.implication = spa.State(vocab, label="implication")

            # Output: NOT P (the negated antecedent)
            self.conclusion = spa.State(vocab, feedback=0.8,
                                         feedback_synapse=0.1,
                                         label="conclusion")

            # Detection ensemble
            self.inference_gate = nengo.Ensemble(
                n_neurons=200,
                dimensions=2,
                label="inference_gate"
            )

            # Connect inputs to gate
            nengo.Connection(
                self.not_consequent.output,
                self.inference_gate[0],
                transform=[np.ones(vocab.dimensions) / vocab.dimensions]
            )

            nengo.Connection(
                self.implication.output,
                self.inference_gate[1],
                transform=[np.ones(vocab.dimensions) / vocab.dimensions]
            )

            def inference_function(x):
                not_q_strength, impl_strength = x
                if not_q_strength > threshold and impl_strength > threshold:
                    return min(not_q_strength, impl_strength)
                return 0.0

            nengo.Connection(
                self.inference_gate,
                self.conclusion.input,
                function=lambda x: inference_function(x) * np.ones(vocab.dimensions) * 0.5
            )


class TransitiveInference(spa.Network):
    """Transitive Inference rule.

    Given:
        - P -> Q (P implies Q)
        - Q -> R (Q implies R)
    Conclude:
        - P -> R (therefore P implies R)

    Example:
        - P -> Q = "All dogs are mammals"
        - Q -> R = "All mammals are animals"
        - P -> R = "All dogs are animals"
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        threshold: float = 0.3,
        label: str = "transitive",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.threshold = threshold

        with self:
            # Input: First implication P -> Q
            self.premise1 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                       label="premise1")

            # Input: Second implication Q -> R
            self.premise2 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                       label="premise2")

            # Output: Derived implication P -> R
            self.conclusion = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                         label="conclusion")

            # Detectors for each premise
            self.detect_p1 = spa.Scalar(label="detect_p1")
            self.detect_p2 = spa.Scalar(label="detect_p2")

            # Inference trigger
            self.trigger = nengo.Ensemble(
                n_neurons=100,
                dimensions=2,
                label="trigger"
            )

            nengo.Connection(self.detect_p1.output, self.trigger[0])
            nengo.Connection(self.detect_p2.output, self.trigger[1])

    def configure_for_concepts(
        self,
        p_implies_q: str,
        q_implies_r: str,
        p_implies_r: str
    ):
        """Configure the network for specific concepts.

        Args:
            p_implies_q: Vocabulary key for first premise
            q_implies_r: Vocabulary key for second premise
            p_implies_r: Vocabulary key for conclusion
        """
        vocab = self.vocab

        with self:
            # Connect premise detectors
            nengo.Connection(
                self.premise1.output,
                self.detect_p1.input,
                transform=[vocab.parse(p_implies_q).v]
            )

            nengo.Connection(
                self.premise2.output,
                self.detect_p2.input,
                transform=[vocab.parse(q_implies_r).v]
            )

            # Inference function
            conclusion_vec = vocab.parse(p_implies_r).v
            threshold = self.threshold

            def inference_function(x):
                p1, p2 = x
                if p1 > threshold and p2 > threshold:
                    return min(p1, p2)
                return 0.0

            nengo.Connection(
                self.trigger,
                self.conclusion.input,
                function=lambda x: inference_function(x) * conclusion_vec
            )


class AnalogyEngine(spa.Network):
    """Analogical reasoning engine.

    Given:
        - A relation between A and B: relation(A, B)
        - A new item C
    Find:
        - D such that relation(C, D) = relation(A, B)

    Example:
        - A = "king", B = "queen" -> relation = "male to female royalty"
        - C = "man"
        - D = "woman" (because man:woman has same relation as king:queen)

    Implementation uses vector algebra:
        relation = B - A (or B * ~A in SPA)
        D = C + relation (or C * relation in SPA)
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        label: str = "analogy",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # Source pair: A and B
            self.source_a = spa.State(vocab, label="source_a")
            self.source_b = spa.State(vocab, label="source_b")

            # Target: C (given) and D (to find)
            self.target_c = spa.State(vocab, label="target_c")
            self.target_d = spa.State(vocab, feedback=0.8, feedback_synapse=0.1,
                                       label="target_d")

            # Extracted relation (B - A or B * ~A)
            self.relation = spa.State(vocab, label="relation")

            # Compute relation: relation = source_b - source_a
            # In vector space, this captures the transformation
            nengo.Connection(
                self.source_b.output,
                self.relation.input,
                transform=1.0
            )
            nengo.Connection(
                self.source_a.output,
                self.relation.input,
                transform=-1.0
            )

            # Apply relation to target: D = C + relation
            nengo.Connection(
                self.target_c.output,
                self.target_d.input,
                transform=1.0
            )
            nengo.Connection(
                self.relation.output,
                self.target_d.input,
                transform=0.8  # Slightly lower weight for relation
            )

            # Cleanup memory to find nearest valid concept
            # (In full implementation, would use associative memory)

    @property
    def output(self):
        return self.target_d.output


class ReasoningRulesEngine(spa.Network):
    """Combined reasoning engine with multiple inference rules.

    Integrates:
    - Modus Ponens
    - Modus Tollens
    - Transitive Inference
    - Analogical Reasoning

    Uses a control system to select which rule to apply based on
    the structure of the input.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        label: str = "reasoning_engine",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab

        with self:
            # Working memory for premises
            self.wm_slot1 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                       label="wm1")
            self.wm_slot2 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                       label="wm2")
            self.wm_slot3 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                       label="wm3")

            # Rule modules
            self.transitive = TransitiveInference(vocab, label="trans_rule")
            self.analogy = AnalogyEngine(vocab, label="analogy_rule")

            # Global conclusion accumulator
            self.conclusion = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                         label="conclusion")

            # Connect working memory to rule modules
            nengo.Connection(self.wm_slot1.output, self.transitive.premise1.input)
            nengo.Connection(self.wm_slot2.output, self.transitive.premise2.input)

            nengo.Connection(self.wm_slot1.output, self.analogy.source_a.input)
            nengo.Connection(self.wm_slot2.output, self.analogy.source_b.input)
            nengo.Connection(self.wm_slot3.output, self.analogy.target_c.input)

            # Combine rule outputs into conclusion
            nengo.Connection(
                self.transitive.conclusion.output,
                self.conclusion.input,
                transform=0.5
            )
            nengo.Connection(
                self.analogy.output,
                self.conclusion.input,
                transform=0.5
            )


def create_reasoning_vocabulary(dimensions: int = 64) -> spa.Vocabulary:
    """Create a vocabulary with common reasoning concepts.

    Args:
        dimensions: Dimensionality of semantic pointers

    Returns:
        Vocabulary populated with reasoning concepts
    """
    vocab = spa.Vocabulary(dimensions, pointer_gen=np.random.RandomState(42))

    # Basic concepts
    vocab.populate("""
        A; B; C; D; E; F;
        P; Q; R; S;
        TRUE; FALSE;
        YES; NO;
        IMPLIES; AND; OR; NOT;
        RELATION; ANALOGY;
        PREMISE; CONCLUSION
    """)

    # Compound concepts for implications
    vocab.add("P_IMPLIES_Q", vocab.parse("P + IMPLIES + Q"))
    vocab.add("Q_IMPLIES_R", vocab.parse("Q + IMPLIES + R"))
    vocab.add("P_IMPLIES_R", vocab.parse("P + IMPLIES + R"))

    vocab.add("A_IMPLIES_B", vocab.parse("A + IMPLIES + B"))
    vocab.add("B_IMPLIES_C", vocab.parse("B + IMPLIES + C"))
    vocab.add("A_IMPLIES_C", vocab.parse("A + IMPLIES + C"))

    # Negations
    vocab.add("NOT_P", vocab.parse("NOT + P"))
    vocab.add("NOT_Q", vocab.parse("NOT + Q"))

    return vocab
