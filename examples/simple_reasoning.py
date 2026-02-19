"""
Simple Reasoning Example

This example demonstrates the basic usage of the AGI Brain reasoning POC.
It shows how to:
1. Create a reasoning network
2. Present a simple syllogism
3. Extract the neural response

Run with: python examples/simple_reasoning.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt

from src.reasoning.reasoning_poc import create_reasoning_poc


def run_syllogism_example():
    """Run a simple syllogism reasoning example.

    Syllogism:
        All A are B (A → B)
        All B are C (B → C)
        Therefore: All A are C (A → C)
    """
    print("=" * 60)
    print("AGI Brain - Syllogistic Reasoning Example")
    print("=" * 60)

    # Create the reasoning model (scaled down for quick testing)
    print("\n[1] Creating reasoning network...")
    model, vocab = create_reasoning_poc(scale=0.01)  # 1% scale for testing

    print(f"    Vocabulary dimensions: {vocab.dimensions}")
    print(f"    Concepts: {list(vocab.keys())[:10]}...")

    # Create the input signal representing the premises
    # A IMPLIES B AND B IMPLIES C
    premise = vocab.parse("A*IMPLIES + IMPLIES*B").v  # Simplified encoding

    print("\n[2] Setting up simulation...")

    # Add input function (must accept t and x for Nengo Node)
    def input_func(t, x):
        if t < 0.2:
            return premise  # Present premise
        else:
            return np.zeros(vocab.dimensions)  # Allow processing

    with model:
        model.input.output = input_func

    # Run simulation
    print("\n[3] Running simulation for 1 second...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(1.0)

    # Analyze output
    print("\n[4] Analyzing results...")
    output_data = sim.data[model.output_probe]

    # Find which semantic pointer is most similar to the output
    final_output = output_data[-1]  # Last timestep

    similarities = {}
    for key in vocab.keys():
        similarity = np.dot(final_output, vocab[key].v) / (
            np.linalg.norm(final_output) * np.linalg.norm(vocab[key].v) + 1e-8
        )
        similarities[key] = similarity

    # Sort by similarity
    sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print("\n    Top 5 output similarities:")
    for concept, sim_val in sorted_sims[:5]:
        print(f"      {concept}: {sim_val:.3f}")

    # Plot results
    print("\n[5] Generating visualization...")
    plot_results(sim.trange(), output_data, vocab)

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


def plot_results(t, output_data, vocab):
    """Plot the reasoning output over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Plot raw output (first few dimensions)
    ax1 = axes[0]
    for i in range(min(5, output_data.shape[1])):
        ax1.plot(t, output_data[:, i], alpha=0.7, label=f"Dim {i}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Activity")
    ax1.set_title("Neural Output (First 5 Dimensions)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot similarity to key concepts
    ax2 = axes[1]
    key_concepts = ["A", "B", "C", "IMPLIES", "CONCLUSION"]
    for concept in key_concepts:
        if concept in vocab:
            concept_vec = vocab[concept].v
            similarities = np.dot(output_data, concept_vec) / (
                np.linalg.norm(output_data, axis=1) * np.linalg.norm(concept_vec) + 1e-8
            )
            ax2.plot(t, similarities, label=concept)

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Similarity")
    ax2.set_title("Output Similarity to Key Concepts")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig("reasoning_output.png", dpi=150)
    print(f"    Saved plot to reasoning_output.png")
    plt.show()


if __name__ == "__main__":
    run_syllogism_example()
