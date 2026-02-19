"""
Cortical column implementation.

A cortical column is a vertical structure through the cortical layers
that functions as a basic processing unit. This implementation provides
a simplified model suitable for cognitive simulation.
"""

from typing import Optional
import nengo
import nengo_spa as spa


class CorticalColumn(spa.Network):
    """A cortical column with multiple layers.

    This is a simplified model of a cortical column that includes:
    - Input layer (Layer 4)
    - Processing layer (Layers 2/3)
    - Output layer (Layer 5/6)

    The column processes semantic pointer representations using
    the Semantic Pointer Architecture.
    """

    def __init__(
        self,
        dimensions: int = 256,
        n_neurons_per_layer: int = 100,
        vocab: Optional[spa.Vocabulary] = None,
        label: str = None,
        **kwargs
    ):
        """Initialize a cortical column.

        Args:
            dimensions: Dimensionality of semantic pointers.
            n_neurons_per_layer: Neurons per layer.
            vocab: Semantic pointer vocabulary (shared or created).
            label: Optional label for the column.
        """
        super().__init__(label=label, **kwargs)

        self.dimensions = dimensions
        self.n_neurons_per_layer = n_neurons_per_layer

        # Use provided vocabulary or create a new one
        if vocab is None:
            self.vocab = spa.Vocabulary(dimensions)
        else:
            self.vocab = vocab

        with self:
            # Input state (Layer 4 equivalent)
            self.input_state = spa.State(
                self.vocab,
                subdimensions=16,
                label=f"{label}_L4" if label else "L4",
            )

            # Processing state (Layers 2/3 equivalent)
            self.processing_state = spa.State(
                self.vocab,
                subdimensions=16,
                label=f"{label}_L23" if label else "L23",
            )

            # Output state (Layers 5/6 equivalent)
            self.output_state = spa.State(
                self.vocab,
                subdimensions=16,
                label=f"{label}_L56" if label else "L56",
            )

            # Feedforward connections
            nengo.Connection(
                self.input_state.output,
                self.processing_state.input,
            )
            nengo.Connection(
                self.processing_state.output,
                self.output_state.input,
            )

            # Recurrent connection in processing layer
            nengo.Connection(
                self.processing_state.output,
                self.processing_state.input,
                transform=0.5,  # Decay factor
                synapse=0.1,
            )

    @property
    def input(self):
        """Input node for the column."""
        return self.input_state.input

    @property
    def output(self):
        """Output node from the column."""
        return self.output_state.output


class AttractorNetwork(spa.Network):
    """An attractor network for maintaining persistent activity.

    Attractor networks are used in working memory to maintain
    representations over time through recurrent connections.
    """

    def __init__(
        self,
        dimensions: int = 256,
        n_neurons: int = 500,
        feedback_strength: float = 1.0,
        decay_time: float = 0.1,
        vocab: Optional[spa.Vocabulary] = None,
        label: str = None,
        **kwargs
    ):
        """Initialize an attractor network.

        Args:
            dimensions: Dimensionality of representations.
            n_neurons: Number of neurons.
            feedback_strength: Strength of recurrent connections.
            decay_time: Time constant for decay (seconds).
            vocab: Semantic pointer vocabulary.
            label: Optional label.
        """
        super().__init__(label=label, **kwargs)

        self.dimensions = dimensions
        self.feedback_strength = feedback_strength
        self.decay_time = decay_time

        if vocab is None:
            self.vocab = spa.Vocabulary(dimensions)
        else:
            self.vocab = vocab

        with self:
            # Main state with recurrent connections
            self.state = spa.State(
                self.vocab,
                feedback=feedback_strength,
                feedback_synapse=decay_time,
                label=f"{label}_state" if label else "attractor_state",
            )

            # Gate for controlling input
            self.gate = nengo.Node(size_in=1)
            self._gate_value = 1.0

        # Expose input/output
        self.input = self.state.input
        self.output = self.state.output

    def set_gate(self, value: float):
        """Set the gating value (0 = blocked, 1 = open)."""
        self._gate_value = value
