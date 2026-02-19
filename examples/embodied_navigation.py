"""
Embodied Navigation Demo - Goal-directed navigation with learning

An improved embodied agent that uses:
1. Explicit goal direction encoding
2. Better action biasing toward goals
3. Reward-based action learning

Run with: python examples/embodied_navigation.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

from src.integration.embodiment.environment import GridWorld, Observation, Action


class NavigationAgent:
    """
    Goal-directed navigation agent

    Uses explicit spatial representations and learns
    from reward signals to improve navigation.
    """

    def __init__(
        self,
        env: GridWorld,
        dims: int = 64,
        n_neurons_per_dim: int = 30,
        learning_rate: float = 1e-3,
        seed: int = 42
    ):
        self.env = env
        self.dims = dims
        self.n_neurons_per_dim = n_neurons_per_dim
        self.learning_rate = learning_rate
        self.seed = seed
        self.n_actions = 5  # up, right, down, left, stay

        # State for real-time input
        self._goal_direction = np.zeros(2)
        self._local_obstacles = np.zeros(4)  # up, right, down, left
        self._local_hazards = np.zeros(4)
        self._reward = 0.0
        self._motor_output = np.zeros(self.n_actions)

        # Build model
        self.model = self._build_model()
        self.sim: Optional[nengo.Simulator] = None

    def _build_model(self) -> nengo.Network:
        """Build neural navigation model"""

        with nengo.Network(seed=self.seed, label="navigation_agent") as model:
            # ============================================
            # SENSORY INPUTS
            # ============================================

            # Goal direction (normalized dx, dy toward nearest goal)
            model.goal_direction = nengo.Node(
                output=lambda t: self._goal_direction,
                size_out=2,
                label="goal_direction"
            )

            # Local obstacle detection (binary: blocked in each direction)
            model.obstacles = nengo.Node(
                output=lambda t: self._local_obstacles,
                size_out=4,
                label="obstacles"
            )

            # Local hazard detection
            model.hazards = nengo.Node(
                output=lambda t: self._local_hazards,
                size_out=4,
                label="hazards"
            )

            # Reward signal
            model.reward_in = nengo.Node(
                output=lambda t: [self._reward],
                size_out=1,
                label="reward_input"
            )

            # ============================================
            # SPATIAL PROCESSING
            # ============================================

            # Encode goal direction
            model.goal_ens = nengo.Ensemble(
                n_neurons=200,
                dimensions=2,
                label="goal_encoding"
            )
            nengo.Connection(model.goal_direction, model.goal_ens)

            # Encode obstacles
            model.obstacle_ens = nengo.Ensemble(
                n_neurons=200,
                dimensions=4,
                label="obstacle_encoding"
            )
            nengo.Connection(model.obstacles, model.obstacle_ens)

            # Encode hazards
            model.hazard_ens = nengo.Ensemble(
                n_neurons=200,
                dimensions=4,
                label="hazard_encoding"
            )
            nengo.Connection(model.hazards, model.hazard_ens)

            # ============================================
            # ACTION VALUE COMPUTATION
            # ============================================

            # Action values based on goal direction
            model.goal_action_values = nengo.Ensemble(
                n_neurons=500,
                dimensions=self.n_actions,
                label="goal_action_values"
            )

            # Goal direction -> action preference
            # up=0, right=1, down=2, left=3, stay=4
            def goal_to_action(goal_dir):
                dx, dy = goal_dir
                values = np.zeros(5)
                # Prefer actions that move toward goal
                values[0] = -dx * 0.5  # up (negative row)
                values[1] = dy * 0.5   # right (positive col)
                values[2] = dx * 0.5   # down (positive row)
                values[3] = -dy * 0.5  # left (negative col)
                values[4] = -0.1       # slight penalty for staying
                return values

            nengo.Connection(
                model.goal_ens,
                model.goal_action_values,
                function=goal_to_action
            )

            # Obstacle avoidance values
            model.obstacle_action_values = nengo.Ensemble(
                n_neurons=500,
                dimensions=self.n_actions,
                label="obstacle_action_values"
            )

            def obstacles_to_action(obs):
                # obs = [up_blocked, right_blocked, down_blocked, left_blocked]
                values = np.zeros(5)
                values[0] = -obs[0] * 2.0  # Strong penalty if up blocked
                values[1] = -obs[1] * 2.0
                values[2] = -obs[2] * 2.0
                values[3] = -obs[3] * 2.0
                values[4] = 0.0
                return values

            nengo.Connection(
                model.obstacle_ens,
                model.obstacle_action_values,
                function=obstacles_to_action
            )

            # Hazard avoidance values
            model.hazard_action_values = nengo.Ensemble(
                n_neurons=500,
                dimensions=self.n_actions,
                label="hazard_action_values"
            )

            def hazards_to_action(haz):
                values = np.zeros(5)
                values[0] = -haz[0] * 1.5  # Penalty for hazard
                values[1] = -haz[1] * 1.5
                values[2] = -haz[2] * 1.5
                values[3] = -haz[3] * 1.5
                values[4] = 0.0
                return values

            nengo.Connection(
                model.hazard_ens,
                model.hazard_action_values,
                function=hazards_to_action
            )

            # Combined action values
            model.combined_values = nengo.Ensemble(
                n_neurons=500,
                dimensions=self.n_actions,
                label="combined_values"
            )
            nengo.Connection(model.goal_action_values, model.combined_values)
            nengo.Connection(model.obstacle_action_values, model.combined_values)
            nengo.Connection(model.hazard_action_values, model.combined_values)

            # ============================================
            # LEARNED ACTION BIAS
            # ============================================

            # State representation for learning
            model.state_ens = nengo.Ensemble(
                n_neurons=1000,
                dimensions=10,  # goal(2) + obstacles(4) + hazards(4)
                label="state"
            )
            nengo.Connection(model.goal_ens, model.state_ens[:2])
            nengo.Connection(model.obstacle_ens, model.state_ens[2:6])
            nengo.Connection(model.hazard_ens, model.state_ens[6:10])

            # Learned action adjustments
            model.learned_values = nengo.Ensemble(
                n_neurons=500,
                dimensions=self.n_actions,
                label="learned_values"
            )

            # Learnable connection
            model.learning_conn = nengo.Connection(
                model.state_ens,
                model.learned_values,
                transform=np.zeros((self.n_actions, 10)),
                learning_rule_type=nengo.PES(learning_rate=self.learning_rate)
            )

            # Add learned values to combined
            nengo.Connection(model.learned_values, model.combined_values, transform=0.5)

            # ============================================
            # ACTION SELECTION
            # ============================================

            # Basal ganglia for winner-take-all
            model.bg = nengo.networks.BasalGanglia(self.n_actions, label="bg")
            nengo.Connection(model.combined_values, model.bg.input)

            model.thalamus = nengo.networks.Thalamus(self.n_actions, label="thalamus")
            nengo.Connection(model.bg.output, model.thalamus.input)

            # Motor output
            model.motor_out = nengo.Node(
                output=lambda t, x: self._set_motor(x),
                size_in=self.n_actions,
                label="motor_output"
            )
            nengo.Connection(model.thalamus.output, model.motor_out)

            # ============================================
            # LEARNING SIGNAL
            # ============================================

            # Error = negative of action that led to bad outcome
            # (simplified: use reward as error modulation)
            model.reward_ens = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="reward"
            )
            nengo.Connection(model.reward_in, model.reward_ens)

            # Error signal based on reward
            model.error = nengo.Ensemble(
                n_neurons=self.n_actions * 100,
                dimensions=self.n_actions,
                label="error"
            )

            # Negative reward means we should have done something else
            def compute_error(reward):
                if reward[0] < 0:
                    # Punish current action
                    return -reward[0] * self._motor_output
                else:
                    # Small negative to slightly reduce current action
                    # (encourages exploration)
                    return -0.1 * self._motor_output

            nengo.Connection(
                model.reward_ens,
                model.error,
                function=compute_error
            )

            nengo.Connection(model.error, model.learning_conn.learning_rule)

            # ============================================
            # PROBES
            # ============================================
            model.p_goal = nengo.Probe(model.goal_ens, synapse=0.01)
            model.p_combined = nengo.Probe(model.combined_values, synapse=0.01)
            model.p_motor = nengo.Probe(model.thalamus.output, synapse=0.01)
            model.p_reward = nengo.Probe(model.reward_ens, synapse=0.01)

        return model

    def _set_motor(self, x: np.ndarray) -> np.ndarray:
        """Store motor output"""
        self._motor_output = x.copy()
        return x

    def _encode_observation(self, obs: Observation):
        """Convert observation to neural inputs"""
        # Compute goal direction from vision
        vision = obs.vision
        center = vision.shape[0] // 2

        # Find goals in vision
        goal_positions = np.where(vision == 2)
        if len(goal_positions[0]) > 0:
            # Average goal direction
            goal_rows = goal_positions[0] - center
            goal_cols = goal_positions[1] - center

            # Normalize
            dx = np.mean(goal_rows)
            dy = np.mean(goal_cols)
            norm = max(np.sqrt(dx**2 + dy**2), 0.1)
            self._goal_direction = np.array([dx / norm, dy / norm])
        else:
            # No goal visible - use proprioception
            # (distance info from proprioception[2])
            self._goal_direction = np.random.randn(2) * 0.3

        # Check obstacles in each direction
        self._local_obstacles = np.zeros(4)
        self._local_hazards = np.zeros(4)

        # up
        if center > 0:
            if vision[center-1, center] == 1:
                self._local_obstacles[0] = 1.0
            if vision[center-1, center] == 3:
                self._local_hazards[0] = 1.0
        else:
            self._local_obstacles[0] = 1.0  # boundary

        # right
        if center < vision.shape[1] - 1:
            if vision[center, center+1] == 1:
                self._local_obstacles[1] = 1.0
            if vision[center, center+1] == 3:
                self._local_hazards[1] = 1.0
        else:
            self._local_obstacles[1] = 1.0

        # down
        if center < vision.shape[0] - 1:
            if vision[center+1, center] == 1:
                self._local_obstacles[2] = 1.0
            if vision[center+1, center] == 3:
                self._local_hazards[2] = 1.0
        else:
            self._local_obstacles[2] = 1.0

        # left
        if center > 0:
            if vision[center, center-1] == 1:
                self._local_obstacles[3] = 1.0
            if vision[center, center-1] == 3:
                self._local_hazards[3] = 1.0
        else:
            self._local_obstacles[3] = 1.0

        self._reward = obs.reward

    def reset(self) -> Observation:
        """Reset environment"""
        obs = self.env.reset()
        self._encode_observation(obs)
        self._motor_output = np.zeros(self.n_actions)
        return obs

    def step(self, sim_time: float = 0.05) -> tuple:
        """Run one step"""
        if self.sim is None:
            raise RuntimeError("Call build_simulator() first")

        self.sim.run(sim_time)

        # Decode action
        action_idx = int(np.argmax(self._motor_output))
        action = Action(
            movement=np.zeros(2),
            discrete_action=action_idx
        )

        # Execute
        obs = self.env.step(action)
        self._encode_observation(obs)

        return obs, action, obs.reward, obs.done

    def build_simulator(self, progress_bar: bool = False):
        """Build simulator"""
        self.sim = nengo.Simulator(self.model, progress_bar=progress_bar)

    def close(self):
        """Clean up"""
        if self.sim:
            self.sim.close()
            self.sim = None

    def run_episode(self, max_steps: int = 50, sim_time: float = 0.05, render: bool = False) -> Dict:
        """Run full episode"""
        obs = self.reset()
        total_reward = 0.0
        steps = 0
        trajectory = [self.env.agent_pos.copy()]

        for _ in range(max_steps):
            if render:
                print(f"\nStep {steps}:")
                print(self.env.render_ascii())

            obs, action, reward, done = self.step(sim_time)
            total_reward += reward
            steps += 1
            trajectory.append(self.env.agent_pos.copy())

            if done:
                break

        return {
            'total_reward': total_reward,
            'steps': steps,
            'success': obs.info.get('goals_remaining', 1) == 0,
            'trajectory': trajectory
        }

    def get_neuron_count(self) -> int:
        """Count neurons"""
        return sum(e.n_neurons for e in self.model.all_ensembles)


def run_navigation_demo():
    """Run navigation demonstration"""

    print("=" * 60)
    print("  GOAL-DIRECTED NAVIGATION DEMO")
    print("=" * 60)
    print()
    print("  The agent uses explicit spatial reasoning to navigate:")
    print("  1. Goal direction encoding")
    print("  2. Obstacle/hazard avoidance")
    print("  3. Reward-based learning to improve")
    print()

    # Create environment
    env = GridWorld(
        size=8,
        n_goals=1,
        n_obstacles=4,
        n_hazards=2,
        vision_range=3,
        max_steps=50,
        seed=123
    )

    print("[1] Environment created")
    print(f"    Grid: {env.size}x{env.size}")
    print()
    print("    Initial layout:")
    print(env.render_ascii())
    print()

    # Create agent
    agent = NavigationAgent(
        env=env,
        dims=64,
        n_neurons_per_dim=30,
        learning_rate=1e-3,
        seed=42
    )

    print(f"[2] Agent created: {agent.get_neuron_count():,} neurons")
    print()

    # Build simulator
    print("[3] Building simulator...")
    agent.build_simulator(progress_bar=False)
    print("    Ready")
    print()

    # Run episodes
    print("[4] Running episodes...")
    print("-" * 60)

    n_episodes = 10
    stats_list = []

    for ep in range(n_episodes):
        # Vary seed for different layouts
        env.rng = np.random.default_rng(123 + ep)
        env._generate_grid()

        stats = agent.run_episode(max_steps=50, sim_time=0.03, render=False)
        stats_list.append(stats)

        status = "SUCCESS" if stats['success'] else "timeout"
        print(f"  Episode {ep+1:2d}: reward={stats['total_reward']:6.2f}, "
              f"steps={stats['steps']:2d}, {status}")

    print("-" * 60)

    # Summary
    print()
    print("[5] Summary:")
    avg_reward = np.mean([s['total_reward'] for s in stats_list])
    success_rate = np.mean([s['success'] for s in stats_list])
    avg_steps = np.mean([s['steps'] for s in stats_list])

    print(f"    Average reward: {avg_reward:.2f}")
    print(f"    Success rate: {success_rate * 100:.0f}%")
    print(f"    Average steps: {avg_steps:.1f}")
    print()

    # Visualize
    print("[6] Generating visualizations...")

    sim = agent.sim
    t = sim.trange()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Action selection over time
    ax1 = axes[0]
    motor = sim.data[agent.model.p_motor]
    labels = ['Up', 'Right', 'Down', 'Left', 'Stay']
    for i in range(5):
        ax1.plot(t, motor[:, i], label=labels[i], alpha=0.8)
    ax1.set_ylabel('Action Activation')
    ax1.set_title('Action Selection Over Time', fontweight='bold')
    ax1.legend(loc='upper right', ncol=5)
    ax1.grid(True, alpha=0.3)

    # Combined action values
    ax2 = axes[1]
    values = sim.data[agent.model.p_combined]
    for i in range(5):
        ax2.plot(t, values[:, i], label=labels[i], alpha=0.7)
    ax2.set_ylabel('Action Value')
    ax2.set_title('Combined Action Values (goal + obstacles + learned)', fontweight='bold')
    ax2.legend(loc='upper right', ncol=5)
    ax2.grid(True, alpha=0.3)

    # Reward
    ax3 = axes[2]
    reward = sim.data[agent.model.p_reward]
    ax3.plot(t, reward, color='green', linewidth=1.5)
    ax3.axhline(0, color='gray', linestyle='--')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Signal', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('navigation_demo.png', dpi=150, bbox_inches='tight')
    print("    Saved to navigation_demo.png")

    # Clean up
    agent.close()

    print()
    print("=" * 60)
    print("  NAVIGATION DEMO COMPLETE")
    print("=" * 60)
    print()


if __name__ == '__main__':
    run_navigation_demo()
