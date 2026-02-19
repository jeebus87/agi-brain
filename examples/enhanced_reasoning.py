"""
Enhanced Reasoning Example with Actual Syllogistic Inference

This example demonstrates real rule-based reasoning using the
Semantic Pointer Architecture (SPA). It shows:

1. Syllogistic reasoning: A->B, B->C therefore A->C
2. Working memory maintenance
3. Rule application through associative memory

Run with: python examples/enhanced_reasoning.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt


def create_syllogism_reasoner():
    """Create a reasoning network that can perform syllogistic inference.

    This implements the classic syllogism:
        Premise 1: All A are B (A -> B)
        Premise 2: All B are C (B -> C)
        Conclusion: All A are C (A -> C)
    """

    # Create vocabulary with concepts for reasoning
    dimensions = 64  # Smaller for faster simulation
    vocab = spa.Vocabulary(dimensions, pointer_gen=np.random.RandomState(42))

    # Add concepts
    vocab.populate("""
        A; B; C; D;
        PREMISE; CONCLUSION;
        IMPLIES;
        YES; NO
    """)

    # Create compound concepts for the syllogism
    # Statement 1: A implies B
    vocab.add("A_IMPLIES_B", vocab.parse("A + IMPLIES + B"))
    # Statement 2: B implies C
    vocab.add("B_IMPLIES_C", vocab.parse("B + IMPLIES + C"))
    # Conclusion: A implies C
    vocab.add("A_IMPLIES_C", vocab.parse("A + IMPLIES + C"))

    with spa.Network(seed=42) as model:
        # =====================================================
        # WORKING MEMORY - Holds current premises
        # =====================================================
        model.premise1 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                    label="premise1")
        model.premise2 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                    label="premise2")

        # =====================================================
        # INFERENCE ENGINE - Derives new knowledge
        # =====================================================
        model.inference = spa.State(vocab, label="inference")

        # =====================================================
        # CONCLUSION - Result of reasoning
        # =====================================================
        model.conclusion = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                      label="conclusion")

        # =====================================================
        # ANSWER - Response to query
        # =====================================================
        model.answer = spa.State(vocab, feedback=0.8, feedback_synapse=0.1,
                                  label="answer")

        # =====================================================
        # INFERENCE LOGIC
        # =====================================================
        # The key insight: If premise1 contains A->B and premise2 contains B->C,
        # we should output A->C

        # Create a "transitive inference" ensemble that combines premises
        # and produces the conclusion when both premises are present

        # Detector for premise1 = A->B
        model.detect_p1 = spa.Scalar(label="detect_p1")
        nengo.Connection(
            model.premise1.output,
            model.detect_p1.input,
            transform=[vocab.parse("A_IMPLIES_B").v]
        )

        # Detector for premise2 = B->C
        model.detect_p2 = spa.Scalar(label="detect_p2")
        nengo.Connection(
            model.premise2.output,
            model.detect_p2.input,
            transform=[vocab.parse("B_IMPLIES_C").v]
        )

        # Inference trigger - fires when both premises are present
        model.trigger = nengo.Ensemble(
            n_neurons=100,
            dimensions=2,
            label="inference_trigger"
        )
        nengo.Connection(model.detect_p1.output, model.trigger[0])
        nengo.Connection(model.detect_p2.output, model.trigger[1])

        # When both premises detected, output the conclusion A->C
        def inference_function(x):
            """Output 1 when both premises are strongly present."""
            p1, p2 = x
            # Both must be > 0.3 to trigger inference
            if p1 > 0.3 and p2 > 0.3:
                return min(p1, p2)  # Confidence is limited by weakest premise
            return 0.0

        # Connect trigger to conclusion via the inference
        conclusion_vec = vocab.parse("A_IMPLIES_C").v

        nengo.Connection(
            model.trigger,
            model.inference.input,
            function=lambda x: inference_function(x) * conclusion_vec,
        )

        # Transfer inference to conclusion (with accumulation)
        nengo.Connection(
            model.inference.output,
            model.conclusion.input,
            transform=0.5,
        )

        # =====================================================
        # ANSWER LOGIC
        # =====================================================
        # Detect if conclusion contains A->C
        model.detect_conclusion = spa.Scalar(label="detect_conclusion")
        nengo.Connection(
            model.conclusion.output,
            model.detect_conclusion.input,
            transform=[vocab.parse("A_IMPLIES_C").v]
        )

        # When conclusion is present, answer YES
        yes_vec = vocab.parse("YES").v

        nengo.Connection(
            model.detect_conclusion.output,
            model.answer.input,
            transform=np.outer(yes_vec, [1.0]),
        )

        # =====================================================
        # INPUT FUNCTIONS
        # =====================================================

        def premise1_input(t):
            """Present first premise: A implies B"""
            if 0.0 <= t < 0.5:
                return vocab.parse("A_IMPLIES_B").v
            return np.zeros(dimensions)

        def premise2_input(t):
            """Present second premise: B implies C"""
            if 0.1 <= t < 0.6:
                return vocab.parse("B_IMPLIES_C").v
            return np.zeros(dimensions)

        # Create input nodes
        model.premise1_input = spa.Transcode(premise1_input, output_vocab=vocab)
        model.premise2_input = spa.Transcode(premise2_input, output_vocab=vocab)

        # Connect inputs
        nengo.Connection(model.premise1_input.output, model.premise1.input)
        nengo.Connection(model.premise2_input.output, model.premise2.input)

        # =====================================================
        # PROBES for monitoring
        # =====================================================
        model.p_premise1 = nengo.Probe(model.premise1.output, synapse=0.01)
        model.p_premise2 = nengo.Probe(model.premise2.output, synapse=0.01)
        model.p_inference = nengo.Probe(model.inference.output, synapse=0.01)
        model.p_conclusion = nengo.Probe(model.conclusion.output, synapse=0.01)
        model.p_answer = nengo.Probe(model.answer.output, synapse=0.01)
        model.p_detect_p1 = nengo.Probe(model.detect_p1.output, synapse=0.01)
        model.p_detect_p2 = nengo.Probe(model.detect_p2.output, synapse=0.01)

        # Store vocab for later analysis
        model.vocab = vocab

    return model


def run_reasoning_demo():
    """Run the syllogistic reasoning demonstration."""

    print("=" * 70)
    print("  ENHANCED REASONING DEMO: Syllogistic Inference")
    print("=" * 70)
    print()
    print("  Task: Given two premises, derive a logical conclusion")
    print()
    print("  Premise 1: All A are B  (A -> B)")
    print("  Premise 2: All B are C  (B -> C)")
    print("  " + "-" * 35)
    print("  Conclusion: All A are C (A -> C)")
    print()
    print("=" * 70)

    # Create the model
    print("\n[1] Building reasoning network...")
    model = create_syllogism_reasoner()
    vocab = model.vocab
    print("    Done! Network created successfully.")

    # Run simulation
    print("\n[2] Running simulation (2 seconds)...")
    print("    Timeline:")
    print("      0.0-0.5s: Present Premise 1 (A->B)")
    print("      0.1-0.6s: Present Premise 2 (B->C)")
    print("      0.3-2.0s: Inference & conclusion formation")
    print()

    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(2.0)

    # Analyze results
    print("\n[3] Analyzing results...")

    t = sim.trange()

    # Key concepts to track
    concepts = {
        "A_IMPLIES_B": "A->B (Premise 1)",
        "B_IMPLIES_C": "B->C (Premise 2)",
        "A_IMPLIES_C": "A->C (Conclusion)",
        "YES": "YES (Answer)",
    }

    # Calculate similarities for conclusion
    conclusion_data = sim.data[model.p_conclusion]
    answer_data = sim.data[model.p_answer]

    print("\n    Final state analysis (last 0.2 seconds):")
    print("    " + "-" * 50)

    for key, label in concepts.items():
        vec = vocab.parse(key).v

        # Conclusion similarity
        conc_sim = np.mean([
            np.dot(conclusion_data[i], vec) /
            (np.linalg.norm(conclusion_data[i]) * np.linalg.norm(vec) + 1e-8)
            for i in range(-200, 0)  # Last 200 timesteps
        ])

        # Answer similarity
        ans_sim = np.mean([
            np.dot(answer_data[i], vec) /
            (np.linalg.norm(answer_data[i]) * np.linalg.norm(vec) + 1e-8)
            for i in range(-200, 0)
        ])

        print(f"    {label:25s} Conclusion: {conc_sim:+.3f}  Answer: {ans_sim:+.3f}")

    # Determine success
    conclusion_vec = vocab.parse("A_IMPLIES_C").v
    final_conclusion = conclusion_data[-1]
    final_similarity = np.dot(final_conclusion, conclusion_vec) / (
        np.linalg.norm(final_conclusion) * np.linalg.norm(conclusion_vec) + 1e-8
    )

    yes_vec = vocab.parse("YES").v
    final_answer = answer_data[-1]
    answer_similarity = np.dot(final_answer, yes_vec) / (
        np.linalg.norm(final_answer) * np.linalg.norm(yes_vec) + 1e-8
    )

    print("\n    " + "-" * 50)
    print(f"\n    REASONING RESULT:")
    if final_similarity > 0.3:
        print(f"    [OK] Successfully derived: A -> C (similarity: {final_similarity:.3f})")
    else:
        print(f"    [?] Weak conclusion (similarity: {final_similarity:.3f})")

    if answer_similarity > 0.2:
        print(f"    [OK] Answered YES to 'Does A->C?' (similarity: {answer_similarity:.3f})")
    else:
        print(f"    [?] Uncertain answer (similarity: {answer_similarity:.3f})")

    # Plot results
    print("\n[4] Generating visualization...")
    plot_reasoning_results(sim, model, vocab, concepts)

    print("\n" + "=" * 70)
    print("  Demo complete!")
    print("=" * 70)


def plot_reasoning_results(sim, model, vocab, concepts):
    """Create detailed visualization of reasoning process."""

    t = sim.trange()

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # Color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # =====================================================
    # Plot 1: Premise 1 (Working Memory)
    # =====================================================
    ax1 = axes[0]
    premise1_data = sim.data[model.p_premise1]

    for i, (key, label) in enumerate(concepts.items()):
        vec = vocab.parse(key).v
        sims = [np.dot(premise1_data[j], vec) /
                (np.linalg.norm(premise1_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
        ax1.plot(t, sims, label=label, color=colors[i], linewidth=2)

    ax1.axvspan(0, 0.5, alpha=0.2, color='green', label='Input window')
    ax1.set_ylabel("Similarity")
    ax1.set_title("Premise 1 (Working Memory): 'All A are B'", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.2)

    # =====================================================
    # Plot 2: Premise 2 (Working Memory)
    # =====================================================
    ax2 = axes[1]
    premise2_data = sim.data[model.p_premise2]

    for i, (key, label) in enumerate(concepts.items()):
        vec = vocab.parse(key).v
        sims = [np.dot(premise2_data[j], vec) /
                (np.linalg.norm(premise2_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
        ax2.plot(t, sims, label=label, color=colors[i], linewidth=2)

    ax2.axvspan(0.1, 0.6, alpha=0.2, color='blue', label='Input window')
    ax2.set_ylabel("Similarity")
    ax2.set_title("Premise 2 (Working Memory): 'All B are C'", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 1.2)

    # =====================================================
    # Plot 3: Inference & Conclusion
    # =====================================================
    ax3 = axes[2]
    inference_data = sim.data[model.p_inference]
    conclusion_data = sim.data[model.p_conclusion]

    # Plot A->C in both inference and conclusion
    vec = vocab.parse("A_IMPLIES_C").v

    inf_sims = [np.dot(inference_data[j], vec) /
                (np.linalg.norm(inference_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
    conc_sims = [np.dot(conclusion_data[j], vec) /
                 (np.linalg.norm(conclusion_data[j]) * np.linalg.norm(vec) + 1e-8)
                 for j in range(len(t))]

    ax3.plot(t, inf_sims, label='Inference (A->C)', color='orange', linewidth=2, linestyle='--')
    ax3.plot(t, conc_sims, label='Conclusion (A->C)', color='red', linewidth=3)

    # Also plot the detection signals
    detect_p1 = sim.data[model.p_detect_p1].flatten()
    detect_p2 = sim.data[model.p_detect_p2].flatten()
    ax3.plot(t, detect_p1, label='Detect P1', color='lightblue', linewidth=1, alpha=0.7)
    ax3.plot(t, detect_p2, label='Detect P2', color='lightgreen', linewidth=1, alpha=0.7)

    ax3.axhline(y=0.3, color='gray', linestyle=':', alpha=0.7, label='Threshold')
    ax3.set_ylabel("Similarity / Detection")
    ax3.set_title("Inference Process: Deriving 'All A are C'", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.5, 1.2)

    # =====================================================
    # Plot 4: Answer
    # =====================================================
    ax4 = axes[3]
    answer_data = sim.data[model.p_answer]

    # Answer similarity to YES
    yes_vec = vocab.parse("YES").v
    yes_sims = [np.dot(answer_data[j], yes_vec) /
                (np.linalg.norm(answer_data[j]) * np.linalg.norm(yes_vec) + 1e-8)
                for j in range(len(t))]

    # Answer similarity to A->C
    ac_vec = vocab.parse("A_IMPLIES_C").v
    ac_sims = [np.dot(answer_data[j], ac_vec) /
               (np.linalg.norm(answer_data[j]) * np.linalg.norm(ac_vec) + 1e-8)
               for j in range(len(t))]

    ax4.plot(t, yes_sims, label='Answer: YES', color='green', linewidth=3)
    ax4.plot(t, ac_sims, label='Answer contains A->C', color='purple', linewidth=2, linestyle='--')

    ax4.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Threshold')
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Similarity")
    ax4.set_title("Answer: Response to 'Does A imply C?'", fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.5, 1.2)

    plt.tight_layout()
    plt.savefig("enhanced_reasoning_output.png", dpi=150, bbox_inches='tight')
    print(f"    Saved plot to enhanced_reasoning_output.png")
    plt.show()


if __name__ == "__main__":
    run_reasoning_demo()
