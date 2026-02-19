"""
Embodied Agent - Complete brain-body-environment integration

Connects the neural cognitive architecture to simulated environments,
enabling perception-action loops and embodied learning.
"""

import numpy as np
import nengo
import nengo_spa as spa
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass

from .environment import Environment, Observation, Action, GridWorld
from .sensory import VisionProcessor, ProprioceptionProcessor, RewardProcessor
from .motor import MotorController, decode_action


@dataclass
class AgentConfig:
    """Configuration for embodied agent"""
    vocab_dimensions: int = 64
    n_neurons_per_dim: int = 30
    vision_range: int = 3
    learning_rate: float = 1e-4
    seed: int = 42


class EmbodiedAgent:
    """
    Complete embodied agent with neural brain

    Integrates:
    - Sensory processing (vision, proprioception, reward)
    - Cognitive core (working memory, decision making)
    - Motor output (action selection)
    - Learning (reward-modulated plasticity)
    """

    def __init__(
        self,
        environment: Environment,
        config: Optional[AgentConfig] = None
    ):
        self.env = environment
        self.config = config or AgentConfig()

        # Get environment specs
        self.obs_shapes = environment.get_observation_shape()
        self.n_actions = environment.n_actions

        # Build neural architecture
        self.model, self.components = self._build_model()

        # Simulation state
        self.sim: Optional[nengo.Simulator] = None
        self.current_obs: Optional[Observation] = None

        # For real-time input/output
        self._sensory_values: Dict[str, np.ndarray] = {}
        self._motor_output: np.ndarray = np.zeros(self.n_actions)

    def _build_model(self) -> Tuple[spa.Network, Dict]:
        """Build complete neural model"""

        config = self.config
        dims = config.vocab_dimensions
        vision_shape = self.obs_shapes['vision']
        vision_size = vision_shape[0] * vision_shape[1]

        # Create vocabulary with embodiment-relevant concepts
        vocab = spa.Vocabulary(dimensions=dims, pointer_gen=np.random.RandomState(config.seed))

        # Add action-related concepts
        vocab.populate('''
            UP; DOWN; LEFT; RIGHT; STAY;
            GOAL; OBSTACLE; HAZARD; EMPTY; AGENT;
            FORWARD; BACKWARD; EXPLORE; EXPLOIT;
            SAFE; DANGER; REWARD; PUNISH;
            POSITION; VELOCITY; STATE
        ''')

        components = {}

        model = spa.Network(seed=config.seed, label="embodied_agent")

        with model:
            # ============================================
            # SENSORY INPUT LAYER
            # ============================================

            # Vision input node (externally driven)
            model.vision_input = nengo.Node(
                output=lambda t: self._sensory_values.get('vision', np.zeros(vision_size)),
                size_out=vision_size,
                label="vision_input"
            )

            # Position input
            model.position_input = nengo.Node(
                output=lambda t: self._sensory_values.get('position', np.zeros(2)),
                size_out=2,
                label="position_input"
            )

            # Velocity input
            model.velocity_input = nengo.Node(
                output=lambda t: self._sensory_values.get('velocity', np.zeros(2)),
                size_out=2,
                label="velocity_input"
            )

            # Proprioception input
            model.proprioception_input = nengo.Node(
                output=lambda t: self._sensory_values.get('proprioception', np.zeros(4)),
                size_out=4,
                label="proprioception_input"
            )

            # Reward input
            model.reward_input = nengo.Node(
                output=lambda t: self._sensory_values.get('reward', np.zeros(1)),
                size_out=1,
                label="reward_input"
            )

            # ============================================
            # SENSORY PROCESSING
            # ============================================

            # Vision processing (simplified V1 -> semantic)
            model.vision_ens = nengo.Ensemble(
                n_neurons=vision_size * config.n_neurons_per_dim,
                dimensions=vision_size,
                radius=2.0,
                label="vision_ensemble"
            )
            nengo.Connection(model.vision_input, model.vision_ens)

            # Visual features -> semantic pointer
            model.vision_semantic = spa.State(vocab, label="vision_semantic")
            vision_transform = np.random.RandomState(config.seed).randn(dims, vision_size)
            vision_transform /= np.sqrt(vision_size)
            nengo.Connection(
                model.vision_ens,
                model.vision_semantic.input,
                transform=vision_transform
            )

            # Body state encoding
            model.body_state = nengo.Ensemble(
                n_neurons=8 * config.n_neurons_per_dim,
                dimensions=8,
                label="body_state"
            )
            nengo.Connection(model.position_input, model.body_state[:2])
            nengo.Connection(model.velocity_input, model.body_state[2:4])
            nengo.Connection(model.proprioception_input, model.body_state[4:])

            model.body_semantic = spa.State(vocab, label="body_semantic")
            body_transform = np.random.RandomState(config.seed + 1).randn(dims, 8)
            body_transform /= np.sqrt(8)
            nengo.Connection(
                model.body_state,
                model.body_semantic.input,
                transform=body_transform
            )

            # Reward processing
            model.reward_ens = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="reward_ensemble"
            )
            nengo.Connection(model.reward_input, model.reward_ens)

            # ============================================
            # COGNITIVE CORE
            # ============================================

            # Integrated percept (vision * body binding)
            model.percept_bind = spa.Bind(vocab, label="percept_bind")
            nengo.Connection(model.vision_semantic.output, model.percept_bind.input_left)
            nengo.Connection(model.body_semantic.output, model.percept_bind.input_right)

            # Working memory (persistent mode - perfect retention)
            model.working_memory = spa.State(
                vocab,
                feedback=1.0,  # Perfect retention
                label="working_memory"
            )
            nengo.Connection(model.percept_bind.output, model.working_memory.input, transform=0.3)

            # Goal representation
            model.goal = spa.State(vocab, feedback=0.95, label="goal")

            # Decision state (combines WM + goal + value)
            model.decision_state = spa.State(vocab, label="decision")

            # Decision integrates working memory and goal
            nengo.Connection(model.working_memory.output, model.decision_state.input, transform=0.5)
            nengo.Connection(model.goal.output, model.decision_state.input, transform=0.5)

            # ============================================
            # ACTION SELECTION (Basal Ganglia)
            # ============================================

            # Map decision state to action values
            model.action_values = nengo.Ensemble(
                n_neurons=self.n_actions * 100,
                dimensions=self.n_actions,
                label="action_values"
            )

            # Action encoders (what patterns favor which actions)
            action_encoders = np.random.RandomState(config.seed + 2).randn(self.n_actions, dims)
            action_encoders /= np.linalg.norm(action_encoders, axis=1, keepdims=True)

            nengo.Connection(
                model.decision_state.output,
                model.action_values,
                transform=action_encoders
            )

            # Basal ganglia winner-take-all
            model.bg = nengo.networks.BasalGanglia(self.n_actions, label="basal_ganglia")
            nengo.Connection(model.action_values, model.bg.input)

            model.thalamus = nengo.networks.Thalamus(self.n_actions, label="thalamus")
            nengo.Connection(model.bg.output, model.thalamus.input)

            # ============================================
            # MOTOR OUTPUT
            # ============================================

            # Output probe node
            model.motor_output = nengo.Node(
                output=lambda t, x: self._set_motor_output(x),
                size_in=self.n_actions,
                label="motor_output"
            )
            nengo.Connection(model.thalamus.output, model.motor_output)

            # ============================================
            # LEARNING (Reward-modulated)
            # ============================================

            # Value prediction
            model.value_pred = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="value_pred"
            )

            # Intermediate ensemble for learning (PES requires Ensemble pre)
            model.state_for_learning = nengo.Ensemble(
                n_neurons=dims * config.n_neurons_per_dim,
                dimensions=dims,
                label="state_for_learning"
            )
            nengo.Connection(model.working_memory.output, model.state_for_learning)

            # Learn to predict value from state
            model.value_conn = nengo.Connection(
                model.state_for_learning,
                model.value_pred,
                transform=np.zeros((1, dims)),
                learning_rule_type=nengo.PES(learning_rate=config.learning_rate)
            )

            # TD error = reward - predicted_value
            model.td_error = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="td_error"
            )
            nengo.Connection(model.reward_ens, model.td_error)
            nengo.Connection(model.value_pred, model.td_error, transform=-1)

            # Error signal for learning
            model.error_signal = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="error_signal"
            )
            nengo.Connection(model.td_error, model.error_signal)
            nengo.Connection(model.error_signal, model.value_conn.learning_rule)

            # ============================================
            # PROBES
            # ============================================

            model.p_vision = nengo.Probe(model.vision_semantic.output, synapse=0.01)
            model.p_body = nengo.Probe(model.body_semantic.output, synapse=0.01)
            model.p_wm = nengo.Probe(model.working_memory.output, synapse=0.01)
            model.p_decision = nengo.Probe(model.decision_state.output, synapse=0.01)
            model.p_actions = nengo.Probe(model.action_values, synapse=0.01)
            model.p_motor = nengo.Probe(model.thalamus.output, synapse=0.01)
            model.p_reward = nengo.Probe(model.reward_ens, synapse=0.01)
            model.p_td_error = nengo.Probe(model.td_error, synapse=0.01)

            components = {
                'vocab': vocab,
                'vision_semantic': model.vision_semantic,
                'body_semantic': model.body_semantic,
                'working_memory': model.working_memory,
                'goal': model.goal,
                'decision': model.decision_state,
                'bg': model.bg,
                'thalamus': model.thalamus,
            }

        return model, components

    def _set_motor_output(self, x: np.ndarray) -> np.ndarray:
        """Callback to store motor output"""
        self._motor_output = x.copy()
        return x

    def _encode_observation(self, obs: Observation):
        """Convert observation to neural inputs"""
        vision_flat = obs.vision.flatten().astype(np.float32)
        vision_norm = vision_flat / 3.0

        self._sensory_values = {
            'vision': vision_norm,
            'position': obs.position,
            'velocity': obs.velocity,
            'proprioception': obs.proprioception,
            'reward': np.array([obs.reward])
        }

    def reset(self) -> Observation:
        """Reset environment and agent state"""
        self.current_obs = self.env.reset()
        self._encode_observation(self.current_obs)
        self._motor_output = np.zeros(self.n_actions)
        return self.current_obs

    def step(self, sim_time: float = 0.1) -> Tuple[Observation, Action, float, bool]:
        """
        Run one step of the perception-action loop

        Args:
            sim_time: Neural simulation time per step (seconds)

        Returns:
            observation, action, reward, done
        """
        if self.sim is None:
            raise RuntimeError("Must call build_simulator() before step()")

        # Run neural simulation
        self.sim.run(sim_time)

        # Decode action from motor output
        action = decode_action(self._motor_output)

        # Execute in environment
        self.current_obs = self.env.step(action)

        # Encode new observation
        self._encode_observation(self.current_obs)

        return (
            self.current_obs,
            action,
            self.current_obs.reward,
            self.current_obs.done
        )

    def build_simulator(self, progress_bar: bool = False):
        """Build Nengo simulator"""
        self.sim = nengo.Simulator(self.model, progress_bar=progress_bar)

    def close(self):
        """Clean up resources"""
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def run_episode(
        self,
        max_steps: int = 100,
        sim_time_per_step: float = 0.1,
        render: bool = False
    ) -> Dict:
        """
        Run complete episode

        Returns:
            Episode statistics
        """
        if self.sim is None:
            self.build_simulator()

        obs = self.reset()
        total_reward = 0.0
        steps = 0
        actions_taken = []

        for step in range(max_steps):
            if render:
                print(f"\nStep {step}:")
                print(self.env.render_ascii())

            obs, action, reward, done = self.step(sim_time_per_step)

            total_reward += reward
            steps += 1
            actions_taken.append(action.discrete_action)

            if done:
                break

        return {
            'total_reward': total_reward,
            'steps': steps,
            'actions': actions_taken,
            'success': obs.info.get('goals_remaining', 1) == 0
        }

    def get_neuron_count(self) -> int:
        """Count total neurons in model"""
        count = 0
        for obj in self.model.all_ensembles:
            count += obj.n_neurons
        return count


def create_embodied_demo(
    grid_size: int = 8,
    seed: int = 42
) -> EmbodiedAgent:
    """Create a demo embodied agent with GridWorld"""

    env = GridWorld(
        size=grid_size,
        n_goals=1,
        n_obstacles=5,
        n_hazards=2,
        vision_range=3,
        max_steps=50,
        seed=seed
    )

    config = AgentConfig(
        vocab_dimensions=64,
        n_neurons_per_dim=30,
        seed=seed
    )

    agent = EmbodiedAgent(env, config)

    return agent
