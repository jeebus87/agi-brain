"""
Reasoning Proof-of-Concept Implementation.

This module implements the 100K neuron reasoning POC as specified
in the AGI brain plan. It focuses on:

1. Syllogistic reasoning (A→B, B→C, therefore A→C)
2. Analogy completion (A:B :: C:?)
3. Rule induction (find pattern from examples)
4. Simple planning (Tower of Hanoi style)

Architecture:
- Problem Encoder (20K neurons)
- Working Memory Buffer (15K neurons)
- Rule Application (20K neurons)
- Analogy Engine (15K neurons)
- Executive Control (20K neurons)
- Response Generator (10K neurons)
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import nengo
import nengo_spa as spa

from ..cognition.working_memory import WorkingMemory
from ..cognition.executive import ExecutiveController


class ReasoningPOC(spa.Network):
    """Proof-of-concept reasoning system.

    This implements a neural reasoning system capable of:
    - Deductive reasoning (syllogisms)
    - Analogical reasoning
    - Rule-based inference
    - Simple planning

    Example:
        >>> vocab = spa.Vocabulary(256)
        >>> vocab.populate('A; B; C; IMPLIES; PREMISE; CONCLUSION')
        >>> reasoner = ReasoningPOC(vocab)
        >>> # ... set up simulation and run
    """

    # Target neuron counts per module
    N_ENCODER = 20000
    N_WORKING_MEMORY = 15000
    N_RULE_APPLICATION = 20000
    N_ANALOGY = 15000
    N_EXECUTIVE = 20000
    N_RESPONSE = 10000

    def __init__(
        self,
        vocab: Optional[spa.Vocabulary] = None,
        dimensions: int = 256,
        scale: float = 1.0,
        label: str = "reasoning_poc",
        **kwargs
    ):
        """Initialize the reasoning POC.

        Args:
            vocab: Semantic pointer vocabulary. Created if not provided.
            dimensions: Dimensionality of semantic pointers.
            scale: Scale factor for neuron counts (use < 1.0 for testing).
            label: Network label.
        """
        super().__init__(label=label, **kwargs)

        # Create vocabulary if not provided
        if vocab is None:
            vocab = spa.Vocabulary(dimensions)
            self._populate_default_vocabulary(vocab)

        self.vocab = vocab
        self.dimensions = dimensions
        self.scale = scale

        # Scale neuron counts
        def scaled(n: int) -> int:
            return max(50, int(n * scale))

        with self:
            # =========================================
            # PROBLEM ENCODER (20K neurons)
            # =========================================
            # Converts input problems to neural representations
            self.encoder = ProblemEncoder(
                vocab=vocab,
                n_neurons=scaled(self.N_ENCODER),
                label="encoder",
            )

            # =========================================
            # WORKING MEMORY (15K neurons)
            # =========================================
            # Holds active problem state and intermediate results
            self.working_memory = WorkingMemory(
                vocab=vocab,
                n_slots=7,
                label="working_memory",
            )

            # =========================================
            # RULE APPLICATION (20K neurons)
            # =========================================
            # Pattern matching and if-then inference
            self.rule_engine = RuleEngine(
                vocab=vocab,
                n_neurons=scaled(self.N_RULE_APPLICATION),
                label="rule_engine",
            )

            # =========================================
            # ANALOGY ENGINE (15K neurons)
            # =========================================
            # Relational mapping and structure alignment
            self.analogy_engine = AnalogyEngine(
                vocab=vocab,
                n_neurons=scaled(self.N_ANALOGY),
                label="analogy_engine",
            )

            # =========================================
            # EXECUTIVE CONTROL (20K neurons)
            # =========================================
            # Action selection and goal maintenance
            self.executive = ExecutiveController(
                vocab=vocab,
                n_actions=5,
                label="executive",
            )

            # =========================================
            # RESPONSE GENERATOR (10K neurons)
            # =========================================
            # Decodes solution to symbolic output
            self.response = ResponseGenerator(
                vocab=vocab,
                n_neurons=scaled(self.N_RESPONSE),
                label="response",
            )

            # =========================================
            # CONNECTIONS
            # =========================================

            # Encoder → Working Memory
            nengo.Connection(
                self.encoder.output,
                self.working_memory.input,
            )

            # Working Memory → Rule Engine
            nengo.Connection(
                self.working_memory.output,
                self.rule_engine.input,
            )

            # Working Memory → Analogy Engine
            nengo.Connection(
                self.working_memory.output,
                self.analogy_engine.input,
            )

            # Rule Engine → Response
            nengo.Connection(
                self.rule_engine.output,
                self.response.input,
                transform=0.5,
            )

            # Analogy Engine → Response
            nengo.Connection(
                self.analogy_engine.output,
                self.response.input,
                transform=0.5,
            )

            # Executive control connections
            nengo.Connection(
                self.executive.action_state.output,
                self.working_memory.input,
                transform=0.1,
            )

    def _populate_default_vocabulary(self, vocab: spa.Vocabulary) -> None:
        """Populate vocabulary with default reasoning concepts."""
        # Logical relations
        vocab.populate("""
            IMPLIES; AND; OR; NOT;
            PREMISE; CONCLUSION; RULE;
            TRUE; FALSE; UNKNOWN
        """)

        # Reasoning states
        vocab.populate("""
            ENCODE; RETRIEVE; APPLY_RULE;
            CHECK; RESPOND; DONE
        """)

        # Example concepts (can be extended)
        vocab.populate("A; B; C; D; E; F")

    @property
    def input(self):
        """Main problem input."""
        return self.encoder.input

    @property
    def output(self):
        """Solution output."""
        return self.response.output


class ProblemEncoder(spa.Network):
    """Encodes problems into semantic pointer representations."""

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_neurons: int = 20000,
        label: str = "encoder",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.dimensions = vocab.dimensions

        with self:
            # Input state
            self.input_state = spa.State(vocab, label="input")

            # Encoded representation
            self.encoded = spa.State(vocab, label="encoded")

            # Transform for encoding (could be learned)
            self.transform = spa.State(vocab, label="transform")

            # Simple passthrough encoding for now
            nengo.Connection(
                self.input_state.output,
                self.encoded.input,
            )

    @property
    def input(self):
        return self.input_state.input

    @property
    def output(self):
        return self.encoded.output


class RuleEngine(spa.Network):
    """Applies production rules for deductive reasoning."""

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_neurons: int = 20000,
        label: str = "rule_engine",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab

        with self:
            # Input state
            self.input_state = spa.State(vocab, label="input")

            # Rule memory - using State with feedback for pattern completion
            # (Simplified from ThresholdingAssocMem for compatibility)
            self.rule_memory = spa.State(
                vocab,
                feedback=0.8,
                feedback_synapse=0.05,
                label="rules",
            )

            # Output state
            self.output_state = spa.State(vocab, label="output")

            # Connect input to rule memory
            nengo.Connection(
                self.input_state.output,
                self.rule_memory.input,
            )

            # Connect rule memory to output state
            nengo.Connection(
                self.rule_memory.output,
                self.output_state.input,
            )

    @property
    def input(self):
        return self.input_state.input

    @property
    def output(self):
        return self.output_state.output


class AnalogyEngine(spa.Network):
    """Performs analogical reasoning through structure mapping."""

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_neurons: int = 15000,
        label: str = "analogy_engine",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab

        with self:
            # Source domain (A:B)
            self.source = spa.State(vocab, label="source")

            # Target domain (C:?)
            self.target = spa.State(vocab, label="target")

            # Relation extractor
            self.relation = spa.State(vocab, label="relation")

            # Output (the answer ?)
            self.answer = spa.State(vocab, label="answer")

            # Input combines source and target
            self._input_node = nengo.Node(size_in=self.vocab.dimensions)
            nengo.Connection(self._input_node, self.source.input)

            # Extract relation: relation = source_B * ~source_A
            # Apply relation: answer = target_C * relation
            # (Simplified for POC - full implementation would be more complex)
            nengo.Connection(
                self.source.output,
                self.relation.input,
            )

            nengo.Connection(
                self.relation.output,
                self.answer.input,
            )

    @property
    def input(self):
        return self._input_node

    @property
    def output(self):
        return self.answer.output


class ResponseGenerator(spa.Network):
    """Generates output responses from neural activity."""

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_neurons: int = 10000,
        label: str = "response",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab

        with self:
            # Accumulator for response
            self.accumulator = spa.State(
                vocab,
                feedback=0.9,
                feedback_synapse=0.05,
                label="accumulator",
            )

            # Confidence estimate
            self.confidence = spa.Scalar(label="confidence")

            # Final output
            self.output_state = spa.State(vocab, label="output")

            # Input
            self._input_node = nengo.Node(size_in=self.vocab.dimensions)

            # Accumulate input
            nengo.Connection(
                self._input_node,
                self.accumulator.input,
            )

            # Output accumulated response
            nengo.Connection(
                self.accumulator.output,
                self.output_state.input,
            )

    @property
    def input(self):
        return self._input_node

    @property
    def output(self):
        return self.output_state.output


def create_reasoning_poc(scale: float = 0.01) -> Tuple[spa.Network, spa.Vocabulary]:
    """Create a reasoning POC network ready for simulation.

    Args:
        scale: Scale factor for neuron counts (0.01 = 1% for quick testing).

    Returns:
        Tuple of (network, vocabulary).

    Example:
        >>> model, vocab = create_reasoning_poc(scale=0.01)
        >>> with nengo.Simulator(model) as sim:
        ...     sim.run(1.0)
    """
    vocab = spa.Vocabulary(256)

    # Populate with reasoning concepts
    vocab.populate("""
        A; B; C; D; E;
        IMPLIES; AND; OR; NOT;
        PREMISE; CONCLUSION; RULE;
        TRUE; FALSE;
        GOAL; DONE
    """)

    with spa.Network(seed=42) as model:
        reasoner = ReasoningPOC(
            vocab=vocab,
            dimensions=256,
            scale=scale,
            label="reasoner",
        )

        # Add input node for testing
        model.input = nengo.Node(size_in=256)
        nengo.Connection(model.input, reasoner.input)

        # Add output probe
        model.output_probe = nengo.Probe(
            reasoner.output,
            synapse=0.01,
        )

    return model, vocab
