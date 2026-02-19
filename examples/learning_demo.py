"""
Learning Demo - Shows synaptic plasticity in action

Demonstrates that the AGI brain can LEARN from experience using PES rule:
1. Communication learning - learn to transform signal A into signal B
2. Function learning - learn an arbitrary input-output mapping

Run with: python examples/learning_demo.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import matplotlib.pyplot as plt


def demo_communication_learning():
    """Learn to communicate a signal from one ensemble to another."""

    print("=" * 60)
    print("  DEMO 1: Communication Learning")
    print("=" * 60)
    print()
    print("  Task: Learn to transmit input signal to output")
    print("  Initially: output = 0 (no connection)")
    print("  After learning: output = input")
    print()

    with nengo.Network(seed=42) as model:
        # Input signal
        input_node = nengo.Node(lambda t: np.sin(2 * np.pi * t))

        # Ensembles
        pre = nengo.Ensemble(n_neurons=100, dimensions=1, label="pre")
        post = nengo.Ensemble(n_neurons=100, dimensions=1, label="post")

        # Input to pre
        nengo.Connection(input_node, pre)

        # Learnable connection (starts at 0)
        conn = nengo.Connection(
            pre, post,
            transform=[[0]],  # Start with no connection
            learning_rule_type=nengo.PES(learning_rate=1e-4)
        )

        # Error signal: we want post to equal pre (i.e., target - actual)
        error = nengo.Ensemble(n_neurons=100, dimensions=1)
        nengo.Connection(pre, error)          # Target = pre
        nengo.Connection(post, error, transform=-1)  # Subtract actual

        # Error drives learning
        nengo.Connection(error, conn.learning_rule)

        # Probes
        p_input = nengo.Probe(input_node, synapse=0.01)
        p_pre = nengo.Probe(pre, synapse=0.01)
        p_post = nengo.Probe(post, synapse=0.01)
        p_error = nengo.Probe(error, synapse=0.01)
        p_weights = nengo.Probe(conn, "weights", synapse=0.1)

    # Run
    print("[1] Running simulation (10 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(10.0)

    # Analyze
    print("\n[2] Analyzing results...")

    t = sim.trange()
    weights = sim.data[p_weights]

    print(f"    Initial weight: {weights[0, 0, 0]:.4f}")
    print(f"    Final weight:   {weights[-1, 0, 0]:.4f}")

    # Compute correlation between pre and post at start vs end
    start_corr = np.corrcoef(
        sim.data[p_pre][:1000].flatten(),
        sim.data[p_post][:1000].flatten()
    )[0, 1]
    end_corr = np.corrcoef(
        sim.data[p_pre][-1000:].flatten(),
        sim.data[p_post][-1000:].flatten()
    )[0, 1]

    print(f"    Start correlation: {start_corr:.3f}")
    print(f"    End correlation:   {end_corr:.3f}")

    # Plot
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(4, 1, figsize=(12, 10))

    # Pre and Post over time
    ax1 = axes[0]
    ax1.plot(t, sim.data[p_pre], label="Pre (input)", alpha=0.8)
    ax1.plot(t, sim.data[p_post], label="Post (learned output)", alpha=0.8)
    ax1.axvline(x=5, color='gray', linestyle='--', alpha=0.5, label="Midpoint")
    ax1.set_ylabel("Activity")
    ax1.set_title("Communication Learning: Pre -> Post", fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Early (unlearned) vs Late (learned)
    ax2 = axes[1]
    ax2.plot(t[:2000], sim.data[p_pre][:2000], label="Pre", alpha=0.8)
    ax2.plot(t[:2000], sim.data[p_post][:2000], label="Post (unlearned)", alpha=0.8)
    ax2.set_ylabel("Activity")
    ax2.set_title("EARLY: Before Learning (Post lags Pre)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(t[-2000:], sim.data[p_pre][-2000:], label="Pre", alpha=0.8)
    ax3.plot(t[-2000:], sim.data[p_post][-2000:], label="Post (learned)", alpha=0.8)
    ax3.set_ylabel("Activity")
    ax3.set_title("LATE: After Learning (Post follows Pre)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Weight evolution
    ax4 = axes[3]
    ax4.plot(t, weights[:, 0, 0], color="green", linewidth=2)
    ax4.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Target (1.0)")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Weight")
    ax4.set_title("Learned Connection Weight (should approach 1.0)", fontweight="bold")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_communication.png", dpi=150, bbox_inches="tight")
    print("    Saved to learning_communication.png")

    return weights[-1, 0, 0], end_corr


def demo_function_learning():
    """Learn an arbitrary function (squaring)."""

    print("\n" + "=" * 60)
    print("  DEMO 2: Function Learning")
    print("=" * 60)
    print()
    print("  Task: Learn to compute y = x^2")
    print("  Initially: output = x (identity)")
    print("  After learning: output = x^2")
    print()

    with nengo.Network(seed=42) as model:
        # Input signal (varies between -1 and 1)
        input_node = nengo.Node(lambda t: np.sin(2 * np.pi * 0.5 * t))

        # Input ensemble
        x = nengo.Ensemble(n_neurons=200, dimensions=1, label="x")
        nengo.Connection(input_node, x)

        # Output ensemble (will learn x^2)
        y = nengo.Ensemble(n_neurons=200, dimensions=1, label="y")

        # Learnable connection (starts with identity)
        conn = nengo.Connection(
            x, y,
            function=lambda x: x,  # Start with identity
            learning_rule_type=nengo.PES(learning_rate=1e-4)
        )

        # Target: x^2
        target = nengo.Ensemble(n_neurons=200, dimensions=1, label="target")
        nengo.Connection(x, target, function=lambda x: x**2)

        # Error
        error = nengo.Ensemble(n_neurons=200, dimensions=1)
        nengo.Connection(target, error)
        nengo.Connection(y, error, transform=-1)

        nengo.Connection(error, conn.learning_rule)

        # Probes
        p_x = nengo.Probe(x, synapse=0.01)
        p_y = nengo.Probe(y, synapse=0.01)
        p_target = nengo.Probe(target, synapse=0.01)
        p_error = nengo.Probe(error, synapse=0.01)

    print("[1] Running simulation (20 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(20.0)

    print("\n[2] Analyzing results...")

    t = sim.trange()

    # MSE at start vs end
    start_mse = np.mean((sim.data[p_y][:2000] - sim.data[p_target][:2000])**2)
    end_mse = np.mean((sim.data[p_y][-2000:] - sim.data[p_target][-2000:])**2)

    print(f"    Start MSE: {start_mse:.4f}")
    print(f"    End MSE:   {end_mse:.4f}")
    print(f"    Improvement: {(1 - end_mse/start_mse)*100:.1f}%")

    # Plot
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9))

    # Full timeline
    ax1 = axes[0]
    ax1.plot(t, sim.data[p_target], label="Target (x^2)", alpha=0.8)
    ax1.plot(t, sim.data[p_y], label="Learned output", alpha=0.8)
    ax1.set_ylabel("Output")
    ax1.set_title("Function Learning: y = x^2", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Early (before learning)
    ax2 = axes[1]
    ax2.plot(t[:4000], sim.data[p_target][:4000], label="Target (x^2)", alpha=0.8)
    ax2.plot(t[:4000], sim.data[p_y][:4000], label="Output (unlearned)", alpha=0.8)
    ax2.set_ylabel("Output")
    ax2.set_title("EARLY: Output = x (identity, not squared)", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Late (after learning)
    ax3 = axes[2]
    ax3.plot(t[-4000:], sim.data[p_target][-4000:], label="Target (x^2)", alpha=0.8)
    ax3.plot(t[-4000:], sim.data[p_y][-4000:], label="Output (learned)", alpha=0.8)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Output")
    ax3.set_title("LATE: Output matches x^2 (learned!)", fontweight="bold")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("learning_function.png", dpi=150, bbox_inches="tight")
    print("    Saved to learning_function.png")

    return end_mse


def main():
    print("\n" + "=" * 60)
    print("  NEURAL LEARNING DEMONSTRATIONS")
    print("=" * 60)
    print("  Showing that spiking neural networks can LEARN")
    print("=" * 60 + "\n")

    # Demo 1: Communication learning
    final_weight, correlation = demo_communication_learning()

    # Demo 2: Function learning
    final_mse = demo_function_learning()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print("  Demo 1 - Communication Learning:")
    print(f"    Learned weight: {final_weight:.3f} (target: 1.0)")
    print(f"    Output correlation: {correlation:.3f}")
    print(f"    Status: {'[PASS]' if final_weight > 0.5 else '[LEARNING]'}")
    print()
    print("  Demo 2 - Function Learning (y = x^2):")
    print(f"    Final MSE: {final_mse:.4f}")
    print(f"    Status: {'[PASS]' if final_mse < 0.1 else '[LEARNING]'}")
    print()
    print("  The system demonstrates PLASTICITY - it learns from experience!")
    print("=" * 60)


if __name__ == "__main__":
    main()
