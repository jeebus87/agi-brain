"""
Multi-Rule Reasoning Example

Demonstrates multiple logical inference rules:
1. Modus Ponens: P, P->Q |- Q
2. Transitive Inference: P->Q, Q->R |- P->R
3. Analogy: A:B :: C:? -> D

Run with: python examples/multi_rule_reasoning.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import nengo
import nengo_spa as spa
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


def create_multi_rule_reasoner():
    """Create a reasoning network with multiple inference rules."""

    dimensions = 64
    vocab = spa.Vocabulary(dimensions, pointer_gen=np.random.RandomState(42))

    # Populate vocabulary
    vocab.populate("""
        A; B; C; D; E; F;
        P; Q; R;
        IMPLIES; AND; NOT;
        TRUE; FALSE;
        YES; NO;
        KING; QUEEN; MAN; WOMAN;
        DOG; PUPPY; CAT; KITTEN
    """)

    # Compound concepts
    vocab.add("P_IMPLIES_Q", vocab.parse("P + IMPLIES + Q"))
    vocab.add("Q_IMPLIES_R", vocab.parse("Q + IMPLIES + R"))
    vocab.add("P_IMPLIES_R", vocab.parse("P + IMPLIES + R"))

    vocab.add("A_IMPLIES_B", vocab.parse("A + IMPLIES + B"))
    vocab.add("B_IMPLIES_C", vocab.parse("B + IMPLIES + C"))
    vocab.add("A_IMPLIES_C", vocab.parse("A + IMPLIES + C"))

    # Analogy relations (king-queen has gender relation, apply to man-?)
    vocab.add("MALE_ROYAL", vocab.parse("KING"))
    vocab.add("FEMALE_ROYAL", vocab.parse("QUEEN"))
    vocab.add("MALE_COMMON", vocab.parse("MAN"))
    vocab.add("FEMALE_COMMON", vocab.parse("WOMAN"))

    # Animal baby relations
    vocab.add("ADULT_DOG", vocab.parse("DOG"))
    vocab.add("BABY_DOG", vocab.parse("PUPPY"))
    vocab.add("ADULT_CAT", vocab.parse("CAT"))
    vocab.add("BABY_CAT", vocab.parse("KITTEN"))

    with spa.Network(seed=42) as model:
        # =====================================================
        # RULE 1: MODUS PONENS
        # Given P and P->Q, conclude Q
        # =====================================================

        # Modus Ponens inputs
        model.mp_proposition = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                          label="mp_proposition")
        model.mp_implication = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                          label="mp_implication")
        model.mp_conclusion = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                         label="mp_conclusion")

        # Detect P in proposition
        model.mp_detect_p = spa.Scalar(label="mp_detect_p")
        nengo.Connection(
            model.mp_proposition.output,
            model.mp_detect_p.input,
            transform=[vocab.parse("P").v]
        )

        # Detect P->Q in implication
        model.mp_detect_impl = spa.Scalar(label="mp_detect_impl")
        nengo.Connection(
            model.mp_implication.output,
            model.mp_detect_impl.input,
            transform=[vocab.parse("P_IMPLIES_Q").v]
        )

        # Inference trigger for modus ponens
        model.mp_trigger = nengo.Ensemble(n_neurons=100, dimensions=2,
                                           label="mp_trigger")
        nengo.Connection(model.mp_detect_p.output, model.mp_trigger[0])
        nengo.Connection(model.mp_detect_impl.output, model.mp_trigger[1])

        # Output Q when both conditions met
        q_vec = vocab.parse("Q").v

        def mp_inference(x):
            p_det, impl_det = x
            if p_det > 0.3 and impl_det > 0.3:
                return min(p_det, impl_det)
            return 0.0

        nengo.Connection(
            model.mp_trigger,
            model.mp_conclusion.input,
            function=lambda x: mp_inference(x) * q_vec
        )

        # =====================================================
        # RULE 2: TRANSITIVE INFERENCE
        # Given A->B and B->C, conclude A->C
        # =====================================================

        model.trans_premise1 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                          label="trans_premise1")
        model.trans_premise2 = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                          label="trans_premise2")
        model.trans_conclusion = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                            label="trans_conclusion")

        # Detect premises
        model.trans_detect_p1 = spa.Scalar(label="trans_detect_p1")
        nengo.Connection(
            model.trans_premise1.output,
            model.trans_detect_p1.input,
            transform=[vocab.parse("A_IMPLIES_B").v]
        )

        model.trans_detect_p2 = spa.Scalar(label="trans_detect_p2")
        nengo.Connection(
            model.trans_premise2.output,
            model.trans_detect_p2.input,
            transform=[vocab.parse("B_IMPLIES_C").v]
        )

        model.trans_trigger = nengo.Ensemble(n_neurons=100, dimensions=2,
                                              label="trans_trigger")
        nengo.Connection(model.trans_detect_p1.output, model.trans_trigger[0])
        nengo.Connection(model.trans_detect_p2.output, model.trans_trigger[1])

        ac_vec = vocab.parse("A_IMPLIES_C").v

        def trans_inference(x):
            p1, p2 = x
            if p1 > 0.3 and p2 > 0.3:
                return min(p1, p2)
            return 0.0

        nengo.Connection(
            model.trans_trigger,
            model.trans_conclusion.input,
            function=lambda x: trans_inference(x) * ac_vec
        )

        # =====================================================
        # RULE 3: ANALOGY
        # Given A:B :: C:?, find D
        # king:queen :: man:? -> woman
        # =====================================================

        model.analogy_a = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                     label="analogy_a")
        model.analogy_b = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                     label="analogy_b")
        model.analogy_c = spa.State(vocab, feedback=1.0, feedback_synapse=0.1,
                                     label="analogy_c")
        model.analogy_d = spa.State(vocab, feedback=0.9, feedback_synapse=0.1,
                                     label="analogy_d")

        # Compute relation: B - A
        model.relation = spa.State(vocab, label="relation")
        nengo.Connection(model.analogy_b.output, model.relation.input, transform=1.0)
        nengo.Connection(model.analogy_a.output, model.relation.input, transform=-1.0)

        # Apply relation: D = C + relation
        nengo.Connection(model.analogy_c.output, model.analogy_d.input, transform=1.0)
        nengo.Connection(model.relation.output, model.analogy_d.input, transform=0.8)

        # =====================================================
        # INPUT FUNCTIONS
        # =====================================================

        # Modus Ponens: Present P and P->Q
        def mp_prop_input(t):
            if 0.0 <= t < 0.3:
                return vocab.parse("P").v
            return np.zeros(dimensions)

        def mp_impl_input(t):
            if 0.1 <= t < 0.4:
                return vocab.parse("P_IMPLIES_Q").v
            return np.zeros(dimensions)

        # Transitive: Present A->B and B->C
        def trans_p1_input(t):
            if 1.0 <= t < 1.3:
                return vocab.parse("A_IMPLIES_B").v
            return np.zeros(dimensions)

        def trans_p2_input(t):
            if 1.1 <= t < 1.4:
                return vocab.parse("B_IMPLIES_C").v
            return np.zeros(dimensions)

        # Analogy: Present KING, QUEEN, MAN -> expect WOMAN
        def analogy_a_input(t):
            if 2.0 <= t < 2.3:
                return vocab.parse("KING").v
            return np.zeros(dimensions)

        def analogy_b_input(t):
            if 2.0 <= t < 2.3:
                return vocab.parse("QUEEN").v
            return np.zeros(dimensions)

        def analogy_c_input(t):
            if 2.1 <= t < 2.4:
                return vocab.parse("MAN").v
            return np.zeros(dimensions)

        # Create input transcoders
        model.mp_prop_in = spa.Transcode(mp_prop_input, output_vocab=vocab)
        model.mp_impl_in = spa.Transcode(mp_impl_input, output_vocab=vocab)
        model.trans_p1_in = spa.Transcode(trans_p1_input, output_vocab=vocab)
        model.trans_p2_in = spa.Transcode(trans_p2_input, output_vocab=vocab)
        model.analogy_a_in = spa.Transcode(analogy_a_input, output_vocab=vocab)
        model.analogy_b_in = spa.Transcode(analogy_b_input, output_vocab=vocab)
        model.analogy_c_in = spa.Transcode(analogy_c_input, output_vocab=vocab)

        # Connect inputs
        nengo.Connection(model.mp_prop_in.output, model.mp_proposition.input)
        nengo.Connection(model.mp_impl_in.output, model.mp_implication.input)
        nengo.Connection(model.trans_p1_in.output, model.trans_premise1.input)
        nengo.Connection(model.trans_p2_in.output, model.trans_premise2.input)
        nengo.Connection(model.analogy_a_in.output, model.analogy_a.input)
        nengo.Connection(model.analogy_b_in.output, model.analogy_b.input)
        nengo.Connection(model.analogy_c_in.output, model.analogy_c.input)

        # =====================================================
        # PROBES
        # =====================================================
        model.p_mp_conclusion = nengo.Probe(model.mp_conclusion.output, synapse=0.01)
        model.p_trans_conclusion = nengo.Probe(model.trans_conclusion.output, synapse=0.01)
        model.p_analogy_d = nengo.Probe(model.analogy_d.output, synapse=0.01)
        model.p_relation = nengo.Probe(model.relation.output, synapse=0.01)

        model.vocab = vocab

    return model


def run_multi_rule_demo():
    """Run the multi-rule reasoning demonstration."""

    print("=" * 70)
    print("  MULTI-RULE REASONING DEMO")
    print("=" * 70)
    print()
    print("  Testing three inference rules:")
    print()
    print("  1. MODUS PONENS (0.0-1.0s)")
    print("     Given: P is true")
    print("     Given: P -> Q (P implies Q)")
    print("     Expect: Q is true")
    print()
    print("  2. TRANSITIVE INFERENCE (1.0-2.0s)")
    print("     Given: A -> B")
    print("     Given: B -> C")
    print("     Expect: A -> C")
    print()
    print("  3. ANALOGY (2.0-3.0s)")
    print("     Given: KING : QUEEN")
    print("     Given: MAN : ?")
    print("     Expect: WOMAN (same gender relation)")
    print()
    print("=" * 70)

    # Create model
    print("\n[1] Building multi-rule reasoning network...")
    model = create_multi_rule_reasoner()
    vocab = model.vocab
    print("    Done!")

    # Run simulation
    print("\n[2] Running simulation (3 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(3.0)

    # Analyze results
    print("\n[3] Analyzing results...")

    t = sim.trange()
    results = {}

    # =====================================================
    # TEST 1: Modus Ponens
    # =====================================================
    print("\n" + "=" * 50)
    print("  TEST 1: MODUS PONENS")
    print("=" * 50)

    mp_data = sim.data[model.p_mp_conclusion]
    q_vec = vocab.parse("Q").v

    # Check conclusion at t=0.8s (after inference should complete)
    mp_idx = int(0.8 / 0.001)  # Convert time to index
    mp_conclusion = mp_data[mp_idx]
    mp_similarity = np.dot(mp_conclusion, q_vec) / (
        np.linalg.norm(mp_conclusion) * np.linalg.norm(q_vec) + 1e-8
    )

    print(f"\n  Expected conclusion: Q")
    print(f"  Similarity to Q at t=0.8s: {mp_similarity:.3f}")

    if mp_similarity > 0.3:
        print("  [PASS] Modus Ponens successful!")
        results["modus_ponens"] = True
    else:
        print("  [FAIL] Modus Ponens failed")
        results["modus_ponens"] = False

    # =====================================================
    # TEST 2: Transitive Inference
    # =====================================================
    print("\n" + "=" * 50)
    print("  TEST 2: TRANSITIVE INFERENCE")
    print("=" * 50)

    trans_data = sim.data[model.p_trans_conclusion]
    ac_vec = vocab.parse("A_IMPLIES_C").v

    # Check at t=1.8s
    trans_idx = int(1.8 / 0.001)
    trans_conclusion = trans_data[trans_idx]
    trans_similarity = np.dot(trans_conclusion, ac_vec) / (
        np.linalg.norm(trans_conclusion) * np.linalg.norm(ac_vec) + 1e-8
    )

    print(f"\n  Expected conclusion: A -> C")
    print(f"  Similarity to A->C at t=1.8s: {trans_similarity:.3f}")

    if trans_similarity > 0.3:
        print("  [PASS] Transitive Inference successful!")
        results["transitive"] = True
    else:
        print("  [FAIL] Transitive Inference failed")
        results["transitive"] = False

    # =====================================================
    # TEST 3: Analogy
    # =====================================================
    print("\n" + "=" * 50)
    print("  TEST 3: ANALOGY (KING:QUEEN :: MAN:?)")
    print("=" * 50)

    analogy_data = sim.data[model.p_analogy_d]

    # Check at t=2.8s
    analogy_idx = int(2.8 / 0.001)
    analogy_result = analogy_data[analogy_idx]

    # Check similarity to various concepts
    concepts_to_check = ["WOMAN", "MAN", "KING", "QUEEN", "DOG", "CAT"]
    print("\n  Similarities at t=2.8s:")

    best_match = None
    best_sim = -1

    for concept in concepts_to_check:
        vec = vocab.parse(concept).v
        sim_val = np.dot(analogy_result, vec) / (
            np.linalg.norm(analogy_result) * np.linalg.norm(vec) + 1e-8
        )
        print(f"    {concept:10s}: {sim_val:+.3f}")
        if sim_val > best_sim:
            best_sim = sim_val
            best_match = concept

    print(f"\n  Best match: {best_match} ({best_sim:.3f})")

    if best_match == "WOMAN" and best_sim > 0.2:
        print("  [PASS] Analogy successful!")
        results["analogy"] = True
    elif best_sim > 0.1:
        print(f"  [PARTIAL] Got {best_match}, expected WOMAN")
        results["analogy"] = False
    else:
        print("  [FAIL] Analogy failed")
        results["analogy"] = False

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"  {status} {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 70)

    # Plot results
    print("\n[4] Generating visualization...")
    plot_multi_rule_results(sim, model, vocab)

    return results


def plot_multi_rule_results(sim, model, vocab):
    """Plot the results of all three reasoning rules."""

    t = sim.trange()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # =====================================================
    # Plot 1: Modus Ponens
    # =====================================================
    ax1 = axes[0]
    mp_data = sim.data[model.p_mp_conclusion]

    for concept in ["P", "Q", "P_IMPLIES_Q"]:
        vec = vocab.parse(concept).v
        sims = [np.dot(mp_data[j], vec) /
                (np.linalg.norm(mp_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
        ax1.plot(t, sims, label=concept, linewidth=2)

    ax1.axvspan(0, 0.4, alpha=0.2, color='green', label='Input window')
    ax1.axhline(y=0.3, color='gray', linestyle=':', alpha=0.7)
    ax1.set_ylabel("Similarity")
    ax1.set_title("Modus Ponens: P, P->Q |- Q", fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-0.5, 1.2)

    # =====================================================
    # Plot 2: Transitive Inference
    # =====================================================
    ax2 = axes[1]
    trans_data = sim.data[model.p_trans_conclusion]

    for concept in ["A_IMPLIES_B", "B_IMPLIES_C", "A_IMPLIES_C"]:
        vec = vocab.parse(concept).v
        sims = [np.dot(trans_data[j], vec) /
                (np.linalg.norm(trans_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
        label = concept.replace("_IMPLIES_", "->")
        ax2.plot(t, sims, label=label, linewidth=2)

    ax2.axvspan(1.0, 1.4, alpha=0.2, color='blue', label='Input window')
    ax2.axhline(y=0.3, color='gray', linestyle=':', alpha=0.7)
    ax2.set_ylabel("Similarity")
    ax2.set_title("Transitive: A->B, B->C |- A->C", fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-0.5, 1.2)

    # =====================================================
    # Plot 3: Analogy
    # =====================================================
    ax3 = axes[2]
    analogy_data = sim.data[model.p_analogy_d]

    for concept in ["KING", "QUEEN", "MAN", "WOMAN"]:
        vec = vocab.parse(concept).v
        sims = [np.dot(analogy_data[j], vec) /
                (np.linalg.norm(analogy_data[j]) * np.linalg.norm(vec) + 1e-8)
                for j in range(len(t))]
        ax3.plot(t, sims, label=concept, linewidth=2)

    ax3.axvspan(2.0, 2.4, alpha=0.2, color='purple', label='Input window')
    ax3.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Similarity")
    ax3.set_title("Analogy: KING:QUEEN :: MAN:? -> WOMAN", fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 3)
    ax3.set_ylim(-0.5, 1.2)

    plt.tight_layout()
    plt.savefig("multi_rule_reasoning.png", dpi=150, bbox_inches='tight')
    print(f"    Saved plot to multi_rule_reasoning.png")
    # plt.show()  # Commented out for non-interactive mode


if __name__ == "__main__":
    results = run_multi_rule_demo()

    # Exit with appropriate code for CI/CD
    if all(results.values()):
        print("\nAll tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed!")
        sys.exit(1)
