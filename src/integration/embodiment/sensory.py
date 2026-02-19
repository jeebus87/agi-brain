"""
Sensory Processing - Neural encoding of environmental observations

Converts raw sensory data to neural population codes compatible
with the AGI brain's Semantic Pointer Architecture.
"""

import numpy as np
import nengo
import nengo_spa as spa
from typing import Dict, Tuple, Optional, Callable
from .environment import Observation


class SensoryProcessor(spa.Network):
    """
    Base sensory processing network

    Encodes observations into semantic pointer representations
    that can be bound and manipulated by the cognitive core.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        obs_shapes: Dict[str, Tuple[int, ...]],
        n_neurons_per_dim: int = 50,
        label: str = "sensory",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)
        self.vocab = vocab
        self.obs_shapes = obs_shapes
        dims = vocab.dimensions

        with self:
            # Visual processing output
            self.vision_out = spa.State(vocab, label="vision_repr")

            # Position/proprioception output
            self.position_out = spa.State(vocab, label="position_repr")

            # Integrated sensory representation
            self.integrated = spa.State(vocab, label="integrated_sensory")

            # Bind vision and position into unified percept
            # PERCEPT = VISION * POSITION (circular convolution)
            self.bind = spa.Bind(vocab, unbind_right=False)
            nengo.Connection(self.vision_out.output, self.bind.input_left)
            nengo.Connection(self.position_out.output, self.bind.input_right)
            nengo.Connection(self.bind.output, self.integrated.input)

        # Declare inputs (to be connected externally)
        self.vision_input = self.vision_out.input
        self.position_input = self.position_out.input

        # Primary output
        self.output = self.integrated.output


class VisionProcessor(spa.Network):
    """
    Visual cortex simulation

    Processes 2D visual field into hierarchical features,
    similar to V1 -> V2 -> V4 -> IT pathway.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        vision_shape: Tuple[int, int],
        n_neurons_per_dim: int = 30,
        label: str = "vision",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vision_shape = vision_shape
        self.vocab = vocab
        dims = vocab.dimensions
        vision_size = vision_shape[0] * vision_shape[1]

        with self:
            # V1: Edge detection / basic features
            # Flattened visual input
            self.v1_input = nengo.Node(size_in=vision_size, label="v1_input")

            # V1 population (retinotopic map)
            self.v1 = nengo.Ensemble(
                n_neurons=vision_size * n_neurons_per_dim,
                dimensions=vision_size,
                radius=3.0,
                label="V1"
            )
            nengo.Connection(self.v1_input, self.v1)

            # V2: Feature combination
            v2_dims = min(vision_size, dims // 2)
            self.v2 = nengo.Ensemble(
                n_neurons=v2_dims * n_neurons_per_dim,
                dimensions=v2_dims,
                label="V2"
            )

            # Dimensionality reduction V1 -> V2
            nengo.Connection(
                self.v1, self.v2,
                transform=self._make_reduction_transform(vision_size, v2_dims)
            )

            # V4/IT: Semantic features -> Semantic pointer
            self.it = nengo.Ensemble(
                n_neurons=dims * n_neurons_per_dim,
                dimensions=dims,
                label="IT"
            )

            # V2 -> IT with expansion
            nengo.Connection(
                self.v2, self.it,
                transform=self._make_expansion_transform(v2_dims, dims)
            )

            # Output as SPA state
            self.output_state = spa.State(vocab, label="vision_semantic")
            nengo.Connection(self.it, self.output_state.input)

        self.input = self.v1_input
        self.output = self.output_state.output

    def _make_reduction_transform(self, in_dims: int, out_dims: int) -> np.ndarray:
        """Create dimensionality reduction matrix"""
        # Simple averaging pooling
        transform = np.zeros((out_dims, in_dims))
        pool_size = in_dims // out_dims
        for i in range(out_dims):
            start = i * pool_size
            end = start + pool_size
            transform[i, start:end] = 1.0 / pool_size
        return transform

    def _make_expansion_transform(self, in_dims: int, out_dims: int) -> np.ndarray:
        """Create dimensionality expansion with learned-like random projection"""
        # Random projection preserves distances (Johnson-Lindenstrauss)
        np.random.seed(42)  # Reproducible
        transform = np.random.randn(out_dims, in_dims) / np.sqrt(in_dims)
        return transform


class ProprioceptionProcessor(spa.Network):
    """
    Proprioceptive processing - body state awareness

    Encodes position, velocity, and internal state
    into semantic pointer representations.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_neurons_per_dim: int = 50,
        label: str = "proprioception",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        dims = vocab.dimensions
        self.vocab = vocab

        with self:
            # Position input (normalized x, y)
            self.position_input = nengo.Node(size_in=2, label="position_in")

            # Velocity input
            self.velocity_input = nengo.Node(size_in=2, label="velocity_in")

            # Internal state input
            self.internal_input = nengo.Node(size_in=4, label="internal_in")

            # Position encoding ensemble
            self.position_ens = nengo.Ensemble(
                n_neurons=100,
                dimensions=2,
                label="position_ens"
            )
            nengo.Connection(self.position_input, self.position_ens)

            # Velocity encoding
            self.velocity_ens = nengo.Ensemble(
                n_neurons=100,
                dimensions=2,
                label="velocity_ens"
            )
            nengo.Connection(self.velocity_input, self.velocity_ens)

            # Combined body state
            self.body_state = nengo.Ensemble(
                n_neurons=dims * n_neurons_per_dim // 2,
                dimensions=8,  # pos(2) + vel(2) + internal(4)
                label="body_state"
            )
            nengo.Connection(self.position_ens, self.body_state[:2])
            nengo.Connection(self.velocity_ens, self.body_state[2:4])
            nengo.Connection(self.internal_input, self.body_state[4:])

            # Project to semantic pointer space
            self.output_state = spa.State(vocab, label="body_semantic")

            # Learned projection body -> semantic
            nengo.Connection(
                self.body_state,
                self.output_state.input,
                transform=self._body_to_semantic_transform(8, dims)
            )

        self.output = self.output_state.output

    def _body_to_semantic_transform(self, body_dims: int, semantic_dims: int) -> np.ndarray:
        """Create body state to semantic pointer projection"""
        np.random.seed(43)
        return np.random.randn(semantic_dims, body_dims) / np.sqrt(body_dims)


class RewardProcessor(spa.Network):
    """
    Reward signal processing

    Converts scalar reward to neural representation
    that can modulate learning and decision making.
    """

    def __init__(
        self,
        n_neurons: int = 100,
        label: str = "reward",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        with self:
            # Raw reward input
            self.reward_input = nengo.Node(size_in=1, label="reward_in")

            # Reward encoding (positive and negative separately)
            self.reward_pos = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=1,
                encoders=[[1]] * n_neurons,  # All positive preferred
                intercepts=nengo.dists.Uniform(0, 0.9),
                label="reward_positive"
            )

            self.reward_neg = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=1,
                encoders=[[-1]] * n_neurons,  # All negative preferred
                intercepts=nengo.dists.Uniform(0, 0.9),
                label="reward_negative"
            )

            nengo.Connection(self.reward_input, self.reward_pos)
            nengo.Connection(self.reward_input, self.reward_neg)

            # Combined reward signal
            self.reward_combined = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=1,
                label="reward_combined"
            )
            nengo.Connection(self.reward_pos, self.reward_combined)
            nengo.Connection(self.reward_neg, self.reward_combined)

            # Dopamine-like modulation signal (for learning)
            self.dopamine = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=1,
                label="dopamine"
            )

            # Reward prediction error approximation
            # (actual reward - baseline)
            self.baseline = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=1,
                label="baseline"
            )

            # Slow integration for baseline
            nengo.Connection(self.reward_combined, self.baseline, synapse=1.0)

            # RPE = reward - baseline
            nengo.Connection(self.reward_combined, self.dopamine)
            nengo.Connection(self.baseline, self.dopamine, transform=-1)

        self.input = self.reward_input
        self.output = self.reward_combined
        self.modulation = self.dopamine


def create_observation_encoder(
    vocab: spa.Vocabulary,
    obs_shapes: Dict[str, Tuple[int, ...]],
    seed: int = 42
) -> Callable[[Observation], Dict[str, np.ndarray]]:
    """
    Create function to encode observations for neural input

    Returns a function that converts Observation to dict of arrays
    suitable for Nengo node inputs.
    """
    vision_shape = obs_shapes.get('vision', (7, 7))
    vision_size = vision_shape[0] * vision_shape[1]

    def encode(obs: Observation) -> Dict[str, np.ndarray]:
        # Flatten and normalize vision
        vision_flat = obs.vision.flatten().astype(np.float32)
        # Normalize: -1 (out of bounds), 0 (empty), 1 (wall), 2 (goal), 3 (hazard)
        vision_norm = vision_flat / 3.0  # Scale to roughly [-0.33, 1]

        return {
            'vision': vision_norm,
            'position': obs.position,
            'velocity': obs.velocity,
            'proprioception': obs.proprioception,
            'reward': np.array([obs.reward])
        }

    return encode
