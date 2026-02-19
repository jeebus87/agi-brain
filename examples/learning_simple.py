"""
Simple Learning Demo - Clear demonstration of neural plasticity

Shows the network learning to produce a specific output value
when given a specific input, with clear before/after comparison.

Run with: python examples/learning_simple.py
"""

import numpy as np
import nengo
import matplotlib.pyplot as plt


def main():
    print("=" * 60)
    print("  NEURAL PLASTICITY DEMONSTRATION")
    print("=" * 60)
    print()
    print("  Task: Learn to output 0.8 when input is 1.0")
    print("  The network starts with random/zero weights and")
    print("  learns the correct mapping through experience.")
    print()

    with nengo.Network(seed=42) as model:
        # Constant input of 1.0
        stim = nengo.Node(output=1.0)

        # Pre-synaptic ensemble (receives input)
        pre = nengo.Ensemble(100, dimensions=1)
        nengo.Connection(stim, pre)

        # Post-synaptic ensemble (produces output)
        post = nengo.Ensemble(100, dimensions=1)

        # Learnable connection - starts with transform=0
        # Will learn to produce 0.8 output from 1.0 input
        conn = nengo.Connection(
            pre, post,
            transform=0,  # Start at zero - no signal passes
            learning_rule_type=nengo.PES(learning_rate=3e-4)
        )

        # Error signal: target (0.8) minus actual output
        # PES minimizes this error by adjusting weights
        error = nengo.Node(output=lambda t, x: x - 0.8, size_in=1)
        nengo.Connection(post, error)
        nengo.Connection(error, conn.learning_rule)

        # Probes
        pre_p = nengo.Probe(pre, synapse=0.01)
        post_p = nengo.Probe(post, synapse=0.01)
        error_p = nengo.Probe(error, synapse=0.01)
        weights_p = nengo.Probe(conn, 'weights', synapse=0.05)

    # Run simulation
    print("[1] Running learning simulation (10 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(10)

    # Analysis
    print("\n[2] Results:")

    t = sim.trange()

    # Get values at different times
    early_output = sim.data[post_p][500:1000].mean()  # 0.5-1.0s
    mid_output = sim.data[post_p][4500:5000].mean()   # 4.5-5.0s
    late_output = sim.data[post_p][-500:].mean()      # Last 0.5s

    early_weight = sim.data[weights_p][500, 0, 0]
    late_weight = sim.data[weights_p][-1, 0, 0]

    print(f"    Target output: 0.80")
    print(f"    Early output (t=0.5-1s):  {early_output:.3f}")
    print(f"    Mid output (t=4.5-5s):    {mid_output:.3f}")
    print(f"    Late output (t=9.5-10s):  {late_output:.3f}")
    print()
    print(f"    Early weight: {early_weight:.4f}")
    print(f"    Final weight: {late_weight:.4f}")
    print()

    # Success check
    if abs(late_output - 0.8) < 0.15:
        print("    [SUCCESS] Network learned to output ~0.8!")
    else:
        print(f"    [LEARNING] Output approaching target...")

    # Plot
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Output over time
    ax1 = axes[0]
    ax1.plot(t, sim.data[post_p], label='Learned Output', linewidth=2)
    ax1.axhline(y=0.8, color='red', linestyle='--', label='Target (0.8)', linewidth=2)
    ax1.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
    ax1.fill_between([0, 1], -1, 2, alpha=0.2, color='blue', label='Early (unlearned)')
    ax1.fill_between([9, 10], -1, 2, alpha=0.2, color='green', label='Late (learned)')
    ax1.set_ylabel('Output Value')
    ax1.set_title('Learning Progress: Output Approaches Target', fontweight='bold', fontsize=12)
    ax1.legend(loc='right')
    ax1.set_ylim(-0.5, 1.5)
    ax1.grid(True, alpha=0.3)

    # Error over time
    ax2 = axes[1]
    ax2.plot(t, sim.data[error_p], color='red', alpha=0.7)
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.set_ylabel('Error')
    ax2.set_title('Learning Error (should approach 0)', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Weight evolution
    ax3 = axes[2]
    ax3.plot(t, sim.data[weights_p][:, 0, 0], color='green', linewidth=2)
    ax3.axhline(y=0.8, color='red', linestyle='--', label='Expected (~0.8)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Connection Weight')
    ax3.set_title('Synaptic Weight Evolution', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_simple.png', dpi=150, bbox_inches='tight')
    print("    Saved to learning_simple.png")

    print("\n" + "=" * 60)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("  Key insight: The spiking neural network adjusted its")
    print("  synaptic weights to produce the desired output.")
    print("  This is LEARNING from experience!")
    print()


if __name__ == '__main__':
    main()
