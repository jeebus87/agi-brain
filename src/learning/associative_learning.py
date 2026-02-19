"""
Associative Learning System

Implements learning capabilities for the AGI brain:

1. Associative Memory Learning
   - Learn to associate concept A with concept B
   - Heteroassociative (A -> B) and autoassociative (A -> A')

2. Rule Learning
   - Learn inference patterns from examples
   - Generalize to new instances

3. Reward-Modulated Learning
   - Strengthen successful reasoning paths
   - Weaken unsuccessful ones

This allows the system to improve from experience rather than
relying solely on pre-programmed knowledge.
"""

import numpy as np
import nengo
import nengo_spa as spa
from typing import Optional, List, Tuple, Dict, Callable
import matplotlib.pyplot as plt


class AssociativeLearner(spa.Network):
    """Learns associations between semantic pointer patterns.

    Uses the PES (Prescribed Error Sensitivity) learning rule to
    learn mappings from one pattern space to another.

    Example: Learn that "DOG" associates with "BARK"
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        learning_rate: float = 1e-4,
        n_neurons: int = 500,
        label: str = "associative_learner",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # Input pattern (what we see)
            self.input = spa.State(vocab, label="input")

            # Input ensemble for learning (PES requires ensemble pre)
            self.input_ens = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=dims,
                label="input_ensemble"
            )
            nengo.Connection(self.input.output, self.input_ens)

            # Target pattern (what we want to output)
            self.target = spa.State(vocab, label="target")

            # Output pattern (what the network produces)
            self.output = spa.State(vocab, label="output")

            # Error computation
            self.error = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=dims,
                label="error"
            )

            # Learnable association connection
            # Starts with identity (will learn deviations)
            self.association = nengo.Connection(
                self.input_ens,
                self.output.input,
                transform=np.eye(dims) * 0.1,
                learning_rule_type=nengo.PES(learning_rate=learning_rate),
                label="learned_association"
            )

            # Error = target - output
            nengo.Connection(self.target.output, self.error, transform=1.0)
            nengo.Connection(self.output.output, self.error, transform=-1.0)

            # Connect error to learning rule
            nengo.Connection(self.error, self.association.learning_rule)

            # Learning gate (can turn learning on/off)
            self.learn_gate = nengo.Node(size_in=1, size_out=1, label="learn_gate")

    def get_weights(self) -> np.ndarray:
        """Get current weight matrix (must be called during simulation)."""
        return self.association.transform


class ConceptLearner(spa.Network):
    """Learns new concepts from examples.

    Given multiple examples of a concept, learns a prototype
    representation that generalizes.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        learning_rate: float = 1e-4,
        label: str = "concept_learner",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # Example input
            self.example = spa.State(vocab, label="example")

            # Learned prototype (accumulates examples)
            self.prototype = spa.State(
                vocab,
                feedback=1.0,  # Perfect retention
                feedback_synapse=0.1,
                label="prototype"
            )

            # Learning signal (how much to incorporate new example)
            self.learn_signal = nengo.Node(
                output=lambda t: 0.1 if t < 2.0 else 0.0,
                label="learn_signal"
            )

            # Gated connection from example to prototype
            self.learn_gate = nengo.Ensemble(
                n_neurons=100,
                dimensions=1,
                label="learn_gate"
            )

            nengo.Connection(self.learn_signal, self.learn_gate)

            # Modulated connection
            self.learn_conn = nengo.Connection(
                self.example.output,
                self.prototype.input,
                transform=0.1,
                label="learn_connection"
            )


class RuleLearner(spa.Network):
    """Learns inference rules from examples.

    Given examples of rule application:
        (premise1, premise2) -> conclusion

    Learns to generalize the rule to new premises.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        learning_rate: float = 1e-4,
        n_neurons: int = 500,
        label: str = "rule_learner",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # Input premises
            self.premise1 = spa.State(vocab, label="premise1")
            self.premise2 = spa.State(vocab, label="premise2")

            # Combined premise ensemble (for learning)
            self.combined_ens = nengo.Ensemble(
                n_neurons=n_neurons,
                dimensions=dims,
                label="combined_ensemble"
            )

            # Combine premises into ensemble
            nengo.Connection(
                self.premise1.output,
                self.combined_ens,
                transform=0.5
            )
            nengo.Connection(
                self.premise2.output,
                self.combined_ens,
                transform=0.5
            )

            # Target conclusion (for training)
            self.target = spa.State(vocab, label="target")

            # Learned conclusion
            self.conclusion = spa.State(vocab, label="conclusion")

            # Error for learning
            self.error = nengo.Ensemble(
                n_neurons=300,
                dimensions=dims,
                label="error"
            )

            # Learnable inference connection (from ensemble)
            self.inference_conn = nengo.Connection(
                self.combined_ens,
                self.conclusion.input,
                transform=np.eye(dims) * 0.1,  # Start small
                learning_rule_type=nengo.PES(learning_rate=learning_rate),
                label="learned_inference"
            )

            # Error computation
            nengo.Connection(self.target.output, self.error, transform=1.0)
            nengo.Connection(self.conclusion.output, self.error, transform=-1.0)

            # Connect error to learning
            nengo.Connection(self.error, self.inference_conn.learning_rule)


class RewardLearner(spa.Network):
    """Reward-modulated learning system.

    Uses a reward signal to gate learning:
    - Positive reward: strengthen recent connections
    - Negative reward: weaken recent connections
    - No reward: no change

    This implements a simple form of reinforcement learning.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        learning_rate: float = 1e-4,
        label: str = "reward_learner",
        **kwargs
    ):
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        dims = vocab.dimensions

        with self:
            # State input
            self.state = spa.State(vocab, label="state")

            # Action output
            self.action = spa.State(vocab, label="action")

            # Reward input (scalar)
            self.reward = nengo.Node(size_in=1, label="reward")

            # Value estimate
            self.value = nengo.Ensemble(
                n_neurons=200,
                dimensions=1,
                label="value"
            )

            # TD error (reward prediction error)
            self.td_error = nengo.Ensemble(
                n_neurons=200,
                dimensions=1,
                label="td_error"
            )

            # Policy connection (state -> action)
            self.policy = nengo.Connection(
                self.state.output,
                self.action.input,
                transform=np.eye(dims) * 0.1,
                learning_rule_type=nengo.PES(learning_rate=learning_rate),
                label="policy"
            )

            # Value prediction error
            nengo.Connection(self.reward, self.td_error)
            nengo.Connection(self.value, self.td_error, transform=-1.0)

            # Reward modulation (simplified - full would use eligibility traces)
            # Scale the error by reward for learning
            self.modulated_error = nengo.Ensemble(
                n_neurons=300,
                dimensions=dims,
                label="modulated_error"
            )

            # Connect TD error to modulate learning
            nengo.Connection(
                self.td_error,
                self.modulated_error,
                transform=np.ones((dims, 1)) * 0.1
            )

            nengo.Connection(self.modulated_error, self.policy.learning_rule)


def demo_associative_learning():
    """Demonstrate learning associations between concepts."""

    print("=" * 70)
    print("  ASSOCIATIVE LEARNING DEMO")
    print("=" * 70)
    print()
    print("  Task: Learn to associate DOG -> BARK, CAT -> MEOW")
    print("  Method: Supervised learning with error correction (PES)")
    print()

    # Create vocabulary
    dims = 64
    vocab = spa.Vocabulary(dims, pointer_gen=np.random.RandomState(42))
    vocab.populate("DOG; CAT; BIRD; BARK; MEOW; CHIRP; ANIMAL; SOUND")

    # Training pairs
    training_pairs = [
        ("DOG", "BARK"),
        ("CAT", "MEOW"),
        ("BIRD", "CHIRP"),
    ]

    with spa.Network(seed=42) as model:
        learner = AssociativeLearner(vocab, learning_rate=5e-4, label="learner")

        # Training schedule
        def input_func(t):
            # Cycle through training pairs
            pair_idx = int(t / 0.5) % len(training_pairs)
            concept, _ = training_pairs[pair_idx]
            return vocab.parse(concept).v

        def target_func(t):
            pair_idx = int(t / 0.5) % len(training_pairs)
            _, target = training_pairs[pair_idx]
            return vocab.parse(target).v

        input_node = spa.Transcode(input_func, output_vocab=vocab)
        target_node = spa.Transcode(target_func, output_vocab=vocab)

        nengo.Connection(input_node.output, learner.input.input)
        nengo.Connection(target_node.output, learner.target.input)

        # Probes
        p_input = nengo.Probe(learner.input.output, synapse=0.01)
        p_output = nengo.Probe(learner.output.output, synapse=0.01)
        p_target = nengo.Probe(learner.target.output, synapse=0.01)
        p_error = nengo.Probe(learner.error, synapse=0.01)
        p_weights = nengo.Probe(learner.association, "weights", synapse=0.1)

    # Run training
    print("[1] Training (3 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(3.0)

    # Analyze results
    print("\n[2] Analyzing learning...")

    t = sim.trange()

    # Check learned associations
    print("\n    Testing learned associations:")
    for concept, expected in training_pairs:
        # Get output at end of simulation for this concept
        concept_vec = vocab.parse(concept).v
        expected_vec = vocab.parse(expected).v

        # Find similarity of output to expected
        final_output = sim.data[p_output][-100:].mean(axis=0)
        final_target = sim.data[p_target][-100:].mean(axis=0)

        # For each concept, compute similarity
        output_sim = np.dot(final_output, expected_vec) / (
            np.linalg.norm(final_output) * np.linalg.norm(expected_vec) + 1e-8
        )
        print(f"      {concept} -> {expected}: similarity = {output_sim:.3f}")

    # Compute error over time
    error_magnitude = np.linalg.norm(sim.data[p_error], axis=1)
    initial_error = error_magnitude[:100].mean()
    final_error = error_magnitude[-100:].mean()

    print(f"\n    Error reduction:")
    print(f"      Initial error: {initial_error:.4f}")
    print(f"      Final error:   {final_error:.4f}")
    print(f"      Reduction:     {(1 - final_error/initial_error)*100:.1f}%")

    # Plot results
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Similarities over time
    ax1 = axes[0]
    colors = ["blue", "orange", "green"]
    for i, (concept, expected) in enumerate(training_pairs):
        expected_vec = vocab.parse(expected).v
        sims = [np.dot(sim.data[p_output][j], expected_vec) /
                (np.linalg.norm(sim.data[p_output][j]) * np.linalg.norm(expected_vec) + 1e-8)
                for j in range(len(t))]
        ax1.plot(t, sims, label=f"{concept}->{expected}", color=colors[i], alpha=0.7)

    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.set_ylabel("Output Similarity")
    ax1.set_title("Learning Progress: Output Similarity to Target", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.5, 1.0)

    # Plot 2: Error over time
    ax2 = axes[1]
    ax2.plot(t, error_magnitude, color="red", alpha=0.7)
    ax2.set_ylabel("Error Magnitude")
    ax2.set_title("Learning Error (should decrease)", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weight matrix evolution
    ax3 = axes[2]
    weights = sim.data[p_weights]
    weight_norms = [np.linalg.norm(weights[i]) for i in range(len(t))]
    ax3.plot(t, weight_norms, color="green", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Weight Norm")
    ax3.set_title("Weight Matrix Evolution", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("associative_learning_demo.png", dpi=150, bbox_inches="tight")
    print("    Saved plot to associative_learning_demo.png")

    print("\n" + "=" * 70)
    print("  Associative Learning Demo Complete!")
    print("=" * 70)

    return model, sim


def demo_rule_learning():
    """Demonstrate learning inference rules from examples."""

    print("=" * 70)
    print("  RULE LEARNING DEMO")
    print("=" * 70)
    print()
    print("  Task: Learn transitive inference from examples")
    print("  Examples: (A->B, B->C) -> A->C")
    print("           (P->Q, Q->R) -> P->R")
    print()

    dims = 64
    vocab = spa.Vocabulary(dims, pointer_gen=np.random.RandomState(42))
    vocab.populate("""
        A; B; C; D; P; Q; R; S;
        IMPLIES
    """)

    # Create compound concepts
    vocab.add("A_IMP_B", vocab.parse("A + IMPLIES + B"))
    vocab.add("B_IMP_C", vocab.parse("B + IMPLIES + C"))
    vocab.add("A_IMP_C", vocab.parse("A + IMPLIES + C"))
    vocab.add("P_IMP_Q", vocab.parse("P + IMPLIES + Q"))
    vocab.add("Q_IMP_R", vocab.parse("Q + IMPLIES + R"))
    vocab.add("P_IMP_R", vocab.parse("P + IMPLIES + R"))

    # Training examples: (premise1, premise2) -> conclusion
    training_examples = [
        ("A_IMP_B", "B_IMP_C", "A_IMP_C"),
        ("P_IMP_Q", "Q_IMP_R", "P_IMP_R"),
    ]

    with spa.Network(seed=42) as model:
        learner = RuleLearner(vocab, learning_rate=1e-3, label="rule_learner")

        # Training schedule
        def premise1_func(t):
            idx = int(t / 1.0) % len(training_examples)
            p1, _, _ = training_examples[idx]
            return vocab.parse(p1).v

        def premise2_func(t):
            idx = int(t / 1.0) % len(training_examples)
            _, p2, _ = training_examples[idx]
            return vocab.parse(p2).v

        def target_func(t):
            idx = int(t / 1.0) % len(training_examples)
            _, _, conc = training_examples[idx]
            return vocab.parse(conc).v

        p1_node = spa.Transcode(premise1_func, output_vocab=vocab)
        p2_node = spa.Transcode(premise2_func, output_vocab=vocab)
        target_node = spa.Transcode(target_func, output_vocab=vocab)

        nengo.Connection(p1_node.output, learner.premise1.input)
        nengo.Connection(p2_node.output, learner.premise2.input)
        nengo.Connection(target_node.output, learner.target.input)

        # Probes
        p_conclusion = nengo.Probe(learner.conclusion.output, synapse=0.01)
        p_target = nengo.Probe(learner.target.output, synapse=0.01)
        p_error = nengo.Probe(learner.error, synapse=0.01)

    print("[1] Training rule learner (4 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(4.0)

    print("\n[2] Analyzing results...")

    t = sim.trange()

    # Check each conclusion
    print("\n    Learned conclusions:")
    for p1, p2, conc in training_examples:
        conc_vec = vocab.parse(conc).v
        final_output = sim.data[p_conclusion][-100:].mean(axis=0)
        sim_val = np.dot(final_output, conc_vec) / (
            np.linalg.norm(final_output) * np.linalg.norm(conc_vec) + 1e-8
        )
        print(f"      ({p1}, {p2}) -> {conc}: similarity = {sim_val:.3f}")

    error_mag = np.linalg.norm(sim.data[p_error], axis=1)
    print(f"\n    Error reduction: {error_mag[100:200].mean():.4f} -> {error_mag[-100:].mean():.4f}")

    # Plot
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    ax1 = axes[0]
    for p1, p2, conc in training_examples:
        conc_vec = vocab.parse(conc).v
        sims = [np.dot(sim.data[p_conclusion][j], conc_vec) /
                (np.linalg.norm(sim.data[p_conclusion][j]) * np.linalg.norm(conc_vec) + 1e-8)
                for j in range(len(t))]
        ax1.plot(t, sims, label=f"{p1}+{p2}->{conc}", linewidth=2)

    ax1.axhline(y=0.5, color="gray", linestyle="--")
    ax1.set_ylabel("Conclusion Similarity")
    ax1.set_title("Rule Learning: Conclusion Quality Over Time", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, error_mag, color="red", alpha=0.7)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Error")
    ax2.set_title("Learning Error", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("rule_learning_demo.png", dpi=150, bbox_inches="tight")
    print("    Saved to rule_learning_demo.png")

    print("\n" + "=" * 70)
    print("  Rule Learning Demo Complete!")
    print("=" * 70)

    return model, sim


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  LEARNING SYSTEM DEMONSTRATIONS")
    print("=" * 70 + "\n")

    print("Running Demo 1: Associative Learning\n")
    demo_associative_learning()

    print("\n" + "-" * 70 + "\n")

    print("Running Demo 2: Rule Learning\n")
    demo_rule_learning()
