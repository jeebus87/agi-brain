"""
Working Memory implementation.

Working memory is a limited-capacity system for temporarily holding
and manipulating information. This implementation models:

1. Limited capacity (~7 items, following Miller's law)
2. Decay without rehearsal
3. Interference between similar items
4. Active maintenance through recurrent activity

Reference:
    Baddeley, A. D. (2003). Working memory: looking back and looking forward.
    Nature Reviews Neuroscience, 4(10), 829-839.
"""

from typing import Optional, List, Dict, Any
import numpy as np
import nengo
import nengo_spa as spa


class WorkingMemorySlot(spa.Network):
    """A single slot in working memory.

    Each slot can hold one semantic pointer representation
    and maintains it through attractor dynamics.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        feedback_strength: float = 1.0,
        decay_rate: float = 0.1,
        label: str = None,
        **kwargs
    ):
        """Initialize a working memory slot.

        Args:
            vocab: Semantic pointer vocabulary.
            feedback_strength: Strength of maintenance (1.0 = perfect).
            decay_rate: Rate of passive decay.
            label: Optional label.
        """
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.dimensions = vocab.dimensions

        with self:
            # Main content storage with feedback for maintenance
            self.content = spa.State(
                vocab,
                feedback=feedback_strength,
                feedback_synapse=decay_rate,
                label=f"{label}_content" if label else "content",
            )

            # Gate for controlling whether new input is accepted
            self.gate = spa.Scalar(label=f"{label}_gate" if label else "gate")

            # Gated input connection
            self._input_node = nengo.Node(size_in=self.dimensions)

            # Multiply input by gate value
            self._gated_input = nengo.Ensemble(
                n_neurons=self.dimensions * 10,
                dimensions=self.dimensions + 1,
                label="gated_input",
            )

            nengo.Connection(self._input_node, self._gated_input[:-1])
            nengo.Connection(self.gate.output, self._gated_input[-1])

            def gate_function(x):
                """Gate the input based on gate value."""
                return x[:-1] * x[-1]

            nengo.Connection(
                self._gated_input,
                self.content.input,
                function=gate_function,
            )

    @property
    def input(self):
        return self._input_node

    @property
    def output(self):
        return self.content.output


class WorkingMemory(spa.Network):
    """Working memory system with multiple slots.

    This models the prefrontal cortex's role in maintaining
    active representations for cognitive operations.

    Features:
    - Multiple slots (default: 7, following Miller's law)
    - Automatic slot selection for new items
    - Decay without active rehearsal
    - Interference between similar items
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_slots: int = 7,
        feedback_strength: float = 0.95,
        decay_rate: float = 0.1,
        label: str = "working_memory",
        **kwargs
    ):
        """Initialize working memory.

        Args:
            vocab: Semantic pointer vocabulary.
            n_slots: Number of memory slots (capacity).
            feedback_strength: Maintenance strength.
            decay_rate: Passive decay rate.
            label: Network label.
        """
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.n_slots = n_slots
        self.dimensions = vocab.dimensions

        with self:
            # Create memory slots
            self.slots: List[WorkingMemorySlot] = []
            for i in range(n_slots):
                slot = WorkingMemorySlot(
                    vocab=vocab,
                    feedback_strength=feedback_strength,
                    decay_rate=decay_rate,
                    label=f"slot_{i}",
                )
                self.slots.append(slot)

            # Input buffer - receives new items
            self.input_buffer = spa.State(vocab, label="input_buffer")

            # Output buffer - combines slot contents
            self.output_buffer = spa.State(vocab, label="output_buffer")

            # Slot selector - determines which slot to use
            # Using a simple State for slot selection (simplified from WTAAssocMem)
            self.slot_selector = spa.State(vocab, label="slot_selector")

            # Connect slots to output (summed)
            for slot in self.slots:
                nengo.Connection(
                    slot.output,
                    self.output_buffer.input,
                    transform=1.0 / n_slots,  # Normalize
                )

            # Create probes for monitoring
            self._probes: Dict[str, nengo.Probe] = {}

    @property
    def input(self):
        """Main input to working memory."""
        return self.input_buffer.input

    @property
    def output(self):
        """Main output from working memory."""
        return self.output_buffer.output

    def add_probes(self, synapse: float = 0.01) -> Dict[str, nengo.Probe]:
        """Add probes for monitoring working memory activity.

        Args:
            synapse: Filtering time constant.

        Returns:
            Dictionary of probes.
        """
        with self:
            self._probes["input"] = nengo.Probe(
                self.input_buffer.output, synapse=synapse
            )
            self._probes["output"] = nengo.Probe(
                self.output_buffer.output, synapse=synapse
            )
            for i, slot in enumerate(self.slots):
                self._probes[f"slot_{i}"] = nengo.Probe(
                    slot.output, synapse=synapse
                )

        return self._probes


class CentralExecutive(spa.Network):
    """Central executive for controlling working memory.

    The central executive coordinates:
    - Attention allocation
    - Task switching
    - Interference resolution
    - Strategy selection
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        working_memory: WorkingMemory,
        label: str = "central_executive",
        **kwargs
    ):
        """Initialize the central executive.

        Args:
            vocab: Semantic pointer vocabulary.
            working_memory: Associated working memory system.
            label: Network label.
        """
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.working_memory = working_memory

        with self:
            # Goal state - what we're trying to achieve
            self.goal = spa.State(vocab, label="goal")

            # Attention focus - what we're currently attending to
            self.focus = spa.State(vocab, label="focus")

            # Control signals
            self.control = spa.BasalGanglia(
                spa.ActionSelection(
                    n_actions=4,  # Number of possible control actions
                ),
                label="control_bg",
            )
