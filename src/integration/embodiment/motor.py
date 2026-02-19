"""
Motor Control - Neural action generation

Converts cognitive decisions to motor commands.
Implements action selection via basal ganglia dynamics.
"""

import numpy as np
import nengo
import nengo_spa as spa
from typing import Tuple, Optional
from .environment import Action


class MotorController(spa.Network):
    """
    Motor cortex + basal ganglia action selection

    Takes cognitive goals/intentions and produces motor commands.
    Uses winner-take-all dynamics for discrete action selection.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_actions: int = 5,
        n_neurons_per_action: int = 100,
        label: str = "motor",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.n_actions = n_actions
        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # Motor intention input (from cognitive core)
            self.intention = spa.State(vocab, label="intention")

            # Action value neurons (one population per action)
            # Maps intention to action utilities
            self.action_values = nengo.Ensemble(
                n_neurons=n_actions * n_neurons_per_action,
                dimensions=n_actions,
                label="action_values"
            )

            # Learned mapping: intention -> action values
            # Each action has an associated semantic pointer
            self._setup_action_mapping(dims, n_actions)

            # Basal ganglia for action selection
            self.bg = nengo.networks.BasalGanglia(n_actions, label="basal_ganglia")
            nengo.Connection(self.action_values, self.bg.input)

            # Thalamus for action gating
            self.thalamus = nengo.networks.Thalamus(n_actions, label="thalamus")
            nengo.Connection(self.bg.output, self.thalamus.input)

            # Motor output (one-hot action selection)
            self.motor_output = nengo.Node(size_in=n_actions, label="motor_out")
            nengo.Connection(self.thalamus.output, self.motor_output)

            # Convert to continuous movement vector
            self.movement_output = nengo.Node(size_in=2, label="movement_out")

            # Action to movement mapping
            # up=0, right=1, down=2, left=3, stay=4
            action_to_movement = np.array([
                [-1, 0],   # up
                [0, 1],    # right
                [1, 0],    # down
                [0, -1],   # left
                [0, 0],    # stay
            ]).T  # Shape: (2, n_actions)

            nengo.Connection(
                self.thalamus.output,
                self.movement_output,
                transform=action_to_movement
            )

        self.input = self.intention.input
        self.output = self.motor_output
        self.movement = self.movement_output

    def _setup_action_mapping(self, dims: int, n_actions: int):
        """Create learned mapping from intention to action values"""
        # Random but fixed projection for each action
        np.random.seed(44)

        # Each action gets a "preferred" semantic pattern
        self.action_encoders = np.random.randn(n_actions, dims)
        self.action_encoders /= np.linalg.norm(self.action_encoders, axis=1, keepdims=True)

        # Connection: dot product of intention with each action encoder
        nengo.Connection(
            self.intention.output,
            self.action_values,
            transform=self.action_encoders
        )


class ReflexiveMotor(spa.Network):
    """
    Fast reflexive motor responses

    Bypasses full cognitive processing for rapid reactions.
    Similar to spinal reflexes in biological systems.
    """

    def __init__(
        self,
        n_actions: int = 5,
        label: str = "reflex",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        with self:
            # Direct sensory input (e.g., hazard detection)
            self.sensory_input = nengo.Node(size_in=4, label="sensory_in")

            # Reflex mapping
            self.reflex_neurons = nengo.Ensemble(
                n_neurons=200,
                dimensions=4,
                label="reflex_neurons"
            )
            nengo.Connection(self.sensory_input, self.reflex_neurons)

            # Reflex output
            self.reflex_output = nengo.Ensemble(
                n_neurons=100,
                dimensions=n_actions,
                label="reflex_out"
            )

            # Simple reflex rules:
            # If hazard ahead -> move away
            def reflex_transform(x):
                # x[0:4] = [up_clear, right_clear, down_clear, left_clear]
                # Negative means obstacle/hazard
                actions = np.zeros(5)
                # Prefer moving toward clear directions
                actions[0] = max(0, x[0])  # up if clear ahead
                actions[1] = max(0, x[1])  # right if clear
                actions[2] = max(0, x[2])  # down if clear
                actions[3] = max(0, x[3])  # left if clear
                actions[4] = 0.1  # Small bias for stay
                return actions

            nengo.Connection(
                self.reflex_neurons,
                self.reflex_output,
                function=reflex_transform
            )

        self.input = self.sensory_input
        self.output = self.reflex_output


class MotorLearner(spa.Network):
    """
    Motor skill learning through practice

    Implements procedural learning using error-driven plasticity.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_actions: int = 5,
        learning_rate: float = 1e-4,
        label: str = "motor_learner",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        dims = vocab.dimensions

        with self:
            # State input
            self.state_input = spa.State(vocab, label="state")

            # Target action (for supervised learning)
            self.target_action = nengo.Node(size_in=n_actions, label="target")

            # Intermediate ensemble for learning
            self.state_ens = nengo.Ensemble(
                n_neurons=dims * 20,
                dimensions=dims,
                label="state_ens"
            )
            nengo.Connection(self.state_input.output, self.state_ens)

            # Learned action output
            self.action_output = nengo.Ensemble(
                n_neurons=n_actions * 50,
                dimensions=n_actions,
                label="action_out"
            )

            # Learnable connection
            self.learning_conn = nengo.Connection(
                self.state_ens,
                self.action_output,
                learning_rule_type=nengo.PES(learning_rate=learning_rate),
                transform=np.zeros((n_actions, dims))  # Start with no knowledge
            )

            # Error signal
            self.error = nengo.Ensemble(
                n_neurons=n_actions * 50,
                dimensions=n_actions,
                label="error"
            )
            nengo.Connection(self.target_action, self.error)
            nengo.Connection(self.action_output, self.error, transform=-1)

            # Error drives learning
            nengo.Connection(self.error, self.learning_conn.learning_rule)

        self.input = self.state_input.input
        self.output = self.action_output
        self.target = self.target_action


def decode_action(motor_output: np.ndarray, threshold: float = 0.2) -> Action:
    """
    Decode neural motor output to Action object

    Args:
        motor_output: Array of action activations
        threshold: Minimum activation to count as selected

    Returns:
        Action object with discrete action and movement vector
    """
    # Find winning action
    if np.max(motor_output) < threshold:
        # No clear winner, default to stay
        discrete_action = 4
    else:
        discrete_action = int(np.argmax(motor_output))

    # Convert to movement
    action_to_movement = {
        0: np.array([-1, 0]),   # up
        1: np.array([0, 1]),    # right
        2: np.array([1, 0]),    # down
        3: np.array([0, -1]),   # left
        4: np.array([0, 0]),    # stay
    }

    movement = action_to_movement.get(discrete_action, np.array([0, 0]))

    return Action(
        movement=movement.astype(np.float32),
        discrete_action=discrete_action
    )
