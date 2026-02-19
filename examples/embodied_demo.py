"""
Embodied Agent Demo - Brain connected to simulated world

Demonstrates the AGI brain navigating a GridWorld environment:
1. Sensory processing converts visual/body state to neural codes
2. Cognitive core integrates information in working memory
3. Basal ganglia selects actions
4. Motor output executes in environment
5. Learning improves behavior over episodes

Run with: python examples/embodied_demo.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Import embodiment components
from src.integration.embodiment.environment import GridWorld, Action
from src.integration.embodiment.agent import EmbodiedAgent, AgentConfig, create_embodied_demo


def visualize_episode(
    env: GridWorld,
    actions: List[int],
    title: str = "Episode Trajectory"
):
    """Visualize the agent's path through the grid"""
    action_symbols = ['^', '>', 'v', '<', 'o']  # up, right, down, left, stay

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw grid
    grid = env.grid.copy()
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'green', 'red'])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=3)

    # Add grid lines
    for i in range(env.size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='white',
                   markersize=15, label='Empty'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                   markersize=15, label='Wall'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green',
                   markersize=15, label='Goal'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red',
                   markersize=15, label='Hazard'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    return fig, ax


def run_demo():
    """Main demo function"""

    print("=" * 60)
    print("  EMBODIED AGENT DEMONSTRATION")
    print("=" * 60)
    print()
    print("  The AGI brain is connected to a simulated GridWorld.")
    print("  It perceives the environment through neural vision,")
    print("  processes information in working memory, and selects")
    print("  actions through basal ganglia dynamics.")
    print()

    # Create environment
    print("[1] Creating GridWorld environment...")
    env = GridWorld(
        size=8,
        n_goals=1,
        n_obstacles=6,
        n_hazards=3,
        vision_range=3,
        max_steps=50,
        seed=42
    )

    print(f"    Grid size: {env.size}x{env.size}")
    print(f"    Goals: {env.n_goals}")
    print(f"    Obstacles: {env.n_obstacles}")
    print(f"    Hazards: {env.n_hazards}")
    print(f"    Vision range: {env.vision_range}")
    print()
    print("    Initial grid:")
    print(env.render_ascii())
    print()

    # Create agent
    print("[2] Building neural embodied agent...")
    config = AgentConfig(
        vocab_dimensions=64,
        n_neurons_per_dim=30,
        learning_rate=1e-4,
        seed=42
    )

    agent = EmbodiedAgent(env, config)
    n_neurons = agent.get_neuron_count()
    print(f"    Total neurons: {n_neurons:,}")
    print(f"    Vocabulary dimensions: {config.vocab_dimensions}")
    print()

    # Build simulator
    print("[3] Building neural simulator...")
    agent.build_simulator(progress_bar=False)
    print("    Simulator ready")
    print()

    # Run episodes
    print("[4] Running episodes...")
    print("-" * 60)

    n_episodes = 5
    episode_stats = []

    for ep in range(n_episodes):
        print(f"\n  Episode {ep + 1}/{n_episodes}")

        # Reset environment with new layout each episode
        env.rng = np.random.default_rng(42 + ep)
        env._generate_grid()

        # Run episode
        stats = agent.run_episode(
            max_steps=50,
            sim_time_per_step=0.05,  # 50ms per action
            render=False
        )

        episode_stats.append(stats)

        status = "[SUCCESS]" if stats['success'] else "[ONGOING]"
        print(f"    Steps: {stats['steps']}")
        print(f"    Total reward: {stats['total_reward']:.2f}")
        print(f"    Status: {status}")

    print()
    print("-" * 60)

    # Summary
    print("\n[5] Summary across episodes:")
    avg_reward = np.mean([s['total_reward'] for s in episode_stats])
    avg_steps = np.mean([s['steps'] for s in episode_stats])
    success_rate = np.mean([s['success'] for s in episode_stats])

    print(f"    Average reward: {avg_reward:.2f}")
    print(f"    Average steps: {avg_steps:.1f}")
    print(f"    Success rate: {success_rate * 100:.0f}%")
    print()

    # Visualize neural activity
    print("[6] Generating visualizations...")

    # Get probe data
    sim = agent.sim
    t = sim.trange()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Motor output (action selection)
    ax1 = axes[0]
    motor_data = sim.data[agent.model.p_motor]
    action_names = ['Up', 'Right', 'Down', 'Left', 'Stay']
    for i in range(5):
        ax1.plot(t, motor_data[:, i], label=action_names[i], alpha=0.8)
    ax1.set_ylabel('Motor Activity')
    ax1.set_title('Action Selection (Thalamus Output)', fontweight='bold')
    ax1.legend(loc='upper right', ncol=5)
    ax1.grid(True, alpha=0.3)

    # Action values
    ax2 = axes[1]
    action_data = sim.data[agent.model.p_actions]
    for i in range(5):
        ax2.plot(t, action_data[:, i], label=action_names[i], alpha=0.7)
    ax2.set_ylabel('Action Value')
    ax2.set_title('Action Value Estimation', fontweight='bold')
    ax2.legend(loc='upper right', ncol=5)
    ax2.grid(True, alpha=0.3)

    # Reward signal
    ax3 = axes[2]
    reward_data = sim.data[agent.model.p_reward]
    ax3.plot(t, reward_data, color='green', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Reward')
    ax3.set_title('Reward Signal', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # TD Error (learning signal)
    ax4 = axes[3]
    td_data = sim.data[agent.model.p_td_error]
    ax4.plot(t, td_data, color='purple', linewidth=1.5)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('TD Error')
    ax4.set_title('Temporal Difference Error (Learning Signal)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('embodied_demo.png', dpi=150, bbox_inches='tight')
    print("    Saved to embodied_demo.png")

    # Working memory visualization
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8))

    # Vision semantic representation
    ax5 = axes2[0]
    vision_data = sim.data[agent.model.p_vision]
    # Show first 16 dimensions as heatmap
    im = ax5.imshow(vision_data[:, :16].T, aspect='auto', cmap='RdBu',
                    extent=[0, t[-1], 16, 0])
    ax5.set_ylabel('Dimension')
    ax5.set_title('Visual Semantic Representation (first 16 dims)', fontweight='bold')
    plt.colorbar(im, ax=ax5, label='Activation')

    # Working memory
    ax6 = axes2[1]
    wm_data = sim.data[agent.model.p_wm]
    im2 = ax6.imshow(wm_data[:, :16].T, aspect='auto', cmap='RdBu',
                     extent=[0, t[-1], 16, 0])
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Dimension')
    ax6.set_title('Working Memory State (first 16 dims)', fontweight='bold')
    plt.colorbar(im2, ax=ax6, label='Activation')

    plt.tight_layout()
    plt.savefig('embodied_memory.png', dpi=150, bbox_inches='tight')
    print("    Saved to embodied_memory.png")

    # Clean up
    agent.close()

    print()
    print("=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("  The embodied agent demonstrates:")
    print("    - Sensory encoding of visual/body information")
    print("    - Working memory with perfect retention (savant mode)")
    print("    - Action selection via basal ganglia")
    print("    - Reward-based learning signals")
    print()
    print("  Next steps:")
    print("    - Extend to MuJoCo/Unity for richer environments")
    print("    - Add exploration strategies")
    print("    - Scale with GPU acceleration")
    print()


if __name__ == '__main__':
    run_demo()
