"""
Working Memory Demonstration

This example demonstrates the working memory system:
1. Limited capacity (~7 items)
2. Maintenance through attractor dynamics
3. Decay without rehearsal

Run with: python examples/working_memory_demo.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt

from src.cognition.working_memory import WorkingMemory


def run_working_memory_demo():
    """Demonstrate working memory capacity and decay."""
    print("=" * 60)
    print("AGI Brain - Working Memory Demonstration")
    print("=" * 60)

    # Create vocabulary
    vocab = spa.Vocabulary(128)
    vocab.populate("ITEM1; ITEM2; ITEM3; ITEM4; ITEM5; ITEM6; ITEM7; ITEM8")

    print(f"\n[1] Created vocabulary with {len(vocab)} items")

    # Create working memory network
    with spa.Network(seed=42) as model:
        wm = WorkingMemory(
            vocab=vocab,
            n_slots=7,
            feedback_strength=0.95,
            decay_rate=0.1,
            label="working_memory",
        )

        # Input node for presenting items
        input_node = nengo.Node(size_in=vocab.dimensions)
        nengo.Connection(input_node, wm.input)

        # Create stimulus function
        def stimulus(t):
            """Present items sequentially."""
            items = ["ITEM1", "ITEM2", "ITEM3", "ITEM4", "ITEM5"]
            duration_per_item = 0.3
            gap = 0.1

            for i, item in enumerate(items):
                start = i * (duration_per_item + gap)
                end = start + duration_per_item
                if start <= t < end:
                    return vocab.parse(item).v

            return np.zeros(vocab.dimensions)

        input_node.output = stimulus

        # Probes
        input_probe = nengo.Probe(input_node, synapse=0.01)
        output_probe = nengo.Probe(wm.output, synapse=0.01)

    print("\n[2] Created working memory with 7 slots")

    # Run simulation
    print("\n[3] Running simulation (5 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(5.0)

    # Analyze results
    print("\n[4] Analyzing working memory performance...")
    plot_working_memory(
        sim.trange(),
        sim.data[input_probe],
        sim.data[output_probe],
        vocab,
    )


def plot_working_memory(t, input_data, output_data, vocab):
    """Plot working memory input and output."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    items = ["ITEM1", "ITEM2", "ITEM3", "ITEM4", "ITEM5"]

    # Plot input similarities
    ax1 = axes[0]
    for item in items:
        if item in vocab:
            vec = vocab[item].v
            sims = np.dot(input_data, vec) / (
                np.linalg.norm(input_data, axis=1, keepdims=True) * np.linalg.norm(vec) + 1e-8
            ).flatten()
            ax1.plot(t, sims, label=item, linewidth=2)

    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Similarity")
    ax1.set_title("Working Memory INPUT - Sequential Item Presentation")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.2)

    # Plot output similarities (what WM maintains)
    ax2 = axes[1]
    for item in items:
        if item in vocab:
            vec = vocab[item].v
            norm = np.linalg.norm(output_data, axis=1, keepdims=True)
            norm = np.where(norm > 0, norm, 1)  # Avoid division by zero
            sims = np.dot(output_data, vec) / (norm * np.linalg.norm(vec)).flatten()
            ax2.plot(t, sims, label=item, linewidth=2)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Similarity")
    ax2.set_title("Working Memory OUTPUT - Item Maintenance and Decay")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.2, 1.2)

    # Add annotations
    ax2.axhline(y=0.3, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax2.text(4.5, 0.35, "Retrieval\nThreshold", fontsize=9, color="red")

    plt.tight_layout()
    plt.savefig("working_memory_demo.png", dpi=150)
    print(f"    Saved plot to working_memory_demo.png")
    plt.show()


if __name__ == "__main__":
    run_working_memory_demo()
