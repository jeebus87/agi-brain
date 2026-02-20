"""
Phoneme Learner - STDP-based Speech Sound Recognition

Learns to recognize phonemes (basic speech sounds) through:
- Spike-timing dependent plasticity (STDP)
- Competitive learning (winner-take-all)
- Temporal pattern recognition
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhonemeConfig:
    """Configuration for phoneme learning."""
    n_input: int = 800           # Cochlea output size (80 mels x 10 time)
    n_phoneme_neurons: int = 4000  # Neurons per phoneme detector
    n_phonemes: int = 40         # Number of phonemes to learn
    learning_rate: float = 0.001
    tau_stdp: float = 0.020      # STDP time constant
    tau_trace: float = 0.050     # Eligibility trace time constant
    inhibition_strength: float = 0.5  # Lateral inhibition
    threshold: float = 0.8       # Detection threshold


class STDPSynapse:
    """
    Spike-Timing Dependent Plasticity synapse.

    Implements asymmetric STDP:
    - Pre before post: potentiation (LTP)
    - Post before pre: depression (LTD)
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        connectivity: float = 0.1,
        seed: int = 42
    ):
        self.n_pre = n_pre
        self.n_post = n_post
        self.rng = np.random.default_rng(seed)

        # Sparse connectivity
        n_conn = int(n_pre * n_post * connectivity)
        self.pre_idx = self.rng.integers(0, n_pre, n_conn)
        self.post_idx = self.rng.integers(0, n_post, n_conn)
        self.weights = self.rng.uniform(0, 0.5, n_conn).astype(np.float32)

        # STDP traces
        self.pre_trace = np.zeros(n_pre, dtype=np.float32)
        self.post_trace = np.zeros(n_post, dtype=np.float32)

    def propagate(self, pre_spikes: np.ndarray) -> np.ndarray:
        """Propagate spikes through synapse."""
        post_current = np.zeros(self.n_post, dtype=np.float32)

        # Only process active pre-synaptic neurons
        active = pre_spikes[self.pre_idx] > 0
        np.add.at(post_current, self.post_idx[active], self.weights[active])

        return post_current

    def update_stdp(
        self,
        pre_spikes: np.ndarray,
        post_spikes: np.ndarray,
        reward: float = 1.0,
        lr: float = 0.001,
        tau: float = 0.020,
        dt: float = 0.001
    ):
        """Apply STDP weight updates."""
        # Update traces
        decay = np.exp(-dt / tau)
        self.pre_trace *= decay
        self.post_trace *= decay

        self.pre_trace[pre_spikes > 0] = 1.0
        self.post_trace[post_spikes > 0] = 1.0

        # LTP: pre before post (potentiation)
        post_active = post_spikes[self.post_idx] > 0
        pre_trace_vals = self.pre_trace[self.pre_idx]
        self.weights[post_active] += lr * reward * pre_trace_vals[post_active]

        # LTD: post before pre (depression)
        pre_active = pre_spikes[self.pre_idx] > 0
        post_trace_vals = self.post_trace[self.post_idx]
        self.weights[pre_active] -= lr * reward * 0.5 * post_trace_vals[pre_active]

        # Bound weights
        np.clip(self.weights, 0, 1, out=self.weights)


class PhonemeDetector:
    """
    Single phoneme detector unit.

    A group of neurons that learns to respond to a specific phoneme
    through STDP and competitive inhibition.
    """

    def __init__(
        self,
        phoneme_id: int,
        n_neurons: int,
        n_input: int,
        seed: int = 42
    ):
        self.phoneme_id = phoneme_id
        self.n_neurons = n_neurons
        self.rng = np.random.default_rng(seed + phoneme_id)

        # Neuron state
        self.voltage = np.zeros(n_neurons, dtype=np.float32)
        self.spikes = np.zeros(n_neurons, dtype=np.bool_)
        self.refractory = np.zeros(n_neurons, dtype=np.float32)

        # Input synapse
        self.input_synapse = STDPSynapse(
            n_input, n_neurons,
            connectivity=0.2,
            seed=seed + phoneme_id
        )

        # Running statistics for confidence
        self.activation_history = []
        self.max_history = 100

    def step(
        self,
        input_spikes: np.ndarray,
        inhibition: float = 0.0,
        dt: float = 0.001,
        tau: float = 0.010,
        threshold: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Process one timestep.

        Returns:
            spikes: Output spike array
            activation: Mean activation level
        """
        # Decay voltage
        self.voltage *= np.exp(-dt / tau)

        # Add input current
        current = self.input_synapse.propagate(input_spikes)
        self.voltage += current

        # Apply lateral inhibition
        self.voltage -= inhibition

        # Update refractory
        self.refractory = np.maximum(0, self.refractory - dt)

        # Generate spikes
        can_spike = self.refractory <= 0
        self.spikes = (self.voltage >= threshold) & can_spike

        # Reset
        self.voltage[self.spikes] = 0
        self.refractory[self.spikes] = 0.002

        # Compute activation
        activation = self.spikes.mean()
        self.activation_history.append(activation)
        if len(self.activation_history) > self.max_history:
            self.activation_history.pop(0)

        return self.spikes, activation

    def learn(self, input_spikes: np.ndarray, reward: float, lr: float):
        """Apply STDP learning."""
        self.input_synapse.update_stdp(
            input_spikes,
            self.spikes.astype(np.float32),
            reward=reward,
            lr=lr
        )

    def get_confidence(self) -> float:
        """Get detection confidence (running average activation)."""
        if not self.activation_history:
            return 0.0
        return np.mean(self.activation_history[-20:])


class PhonemeLearner:
    """
    Complete phoneme learning system.

    Manages multiple phoneme detectors with:
    - Competitive learning (lateral inhibition)
    - Winner-take-all dynamics
    - Self-organizing map behavior
    """

    def __init__(self, config: Optional[PhonemeConfig] = None):
        self.config = config or PhonemeConfig()
        self.detectors: List[PhonemeDetector] = []

        # Create phoneme detectors
        for i in range(self.config.n_phonemes):
            detector = PhonemeDetector(
                phoneme_id=i,
                n_neurons=self.config.n_phoneme_neurons,
                n_input=self.config.n_input,
                seed=i * 1000
            )
            self.detectors.append(detector)

        # Phoneme labels (learned through association)
        self.phoneme_labels: Dict[int, str] = {}

        # Statistics
        self.detection_counts = np.zeros(self.config.n_phonemes)
        self.learning_step = 0

    def process(
        self,
        input_spikes: np.ndarray,
        learn: bool = True,
        target_phoneme: Optional[int] = None
    ) -> Tuple[int, float]:
        """
        Process input and optionally learn.

        Args:
            input_spikes: Cochlea output spikes
            learn: Whether to apply STDP
            target_phoneme: Supervised target (optional)

        Returns:
            detected_phoneme: ID of detected phoneme
            confidence: Detection confidence
        """
        activations = []
        all_spikes = []

        # First pass: compute activations
        for detector in self.detectors:
            spikes, activation = detector.step(
                input_spikes,
                inhibition=0,
                threshold=self.config.threshold
            )
            activations.append(activation)
            all_spikes.append(spikes)

        activations = np.array(activations)

        # Winner-take-all
        winner = np.argmax(activations)
        confidence = activations[winner]

        # Apply lateral inhibition (suppress losers)
        if confidence > 0:
            for i, detector in enumerate(self.detectors):
                if i != winner:
                    detector.voltage *= (1 - self.config.inhibition_strength)

        # Learning
        if learn:
            self.learning_step += 1

            if target_phoneme is not None:
                # Supervised learning
                for i, detector in enumerate(self.detectors):
                    if i == target_phoneme:
                        reward = 1.0  # Strengthen correct detector
                    elif i == winner and i != target_phoneme:
                        reward = -0.5  # Weaken incorrect winner
                    else:
                        reward = 0.0
                    detector.learn(input_spikes, reward, self.config.learning_rate)
            else:
                # Unsupervised (Hebbian) learning
                if confidence > 0.1:
                    self.detectors[winner].learn(
                        input_spikes,
                        reward=1.0,
                        lr=self.config.learning_rate
                    )
                    self.detection_counts[winner] += 1

        return winner, confidence

    def label_phoneme(self, detector_id: int, label: str):
        """Associate a label with a detector."""
        self.phoneme_labels[detector_id] = label

    def get_label(self, detector_id: int) -> str:
        """Get label for detector."""
        return self.phoneme_labels.get(detector_id, f"P{detector_id}")

    def detect_sequence(
        self,
        spike_sequence: np.ndarray,
        min_confidence: float = 0.2
    ) -> List[Tuple[int, str, float]]:
        """
        Detect phoneme sequence from spike stream.

        Args:
            spike_sequence: Shape (time, input_dim)
            min_confidence: Minimum confidence to report

        Returns:
            List of (time_idx, phoneme_label, confidence)
        """
        detections = []
        prev_phoneme = -1

        for t, spikes in enumerate(spike_sequence):
            phoneme_id, confidence = self.process(spikes, learn=False)

            if confidence >= min_confidence and phoneme_id != prev_phoneme:
                label = self.get_label(phoneme_id)
                detections.append((t, label, confidence))
                prev_phoneme = phoneme_id

        return detections

    def get_statistics(self) -> Dict:
        """Get learning statistics."""
        return {
            'total_steps': self.learning_step,
            'detection_counts': self.detection_counts.tolist(),
            'active_detectors': int(np.sum(self.detection_counts > 0)),
            'detector_confidences': [d.get_confidence() for d in self.detectors]
        }

    def save(self, path: str):
        """Save learned weights."""
        data = {
            'weights': [d.input_synapse.weights for d in self.detectors],
            'pre_idx': [d.input_synapse.pre_idx for d in self.detectors],
            'post_idx': [d.input_synapse.post_idx for d in self.detectors],
            'labels': self.phoneme_labels,
            'detection_counts': self.detection_counts
        }
        np.savez(path, **{k: np.array(v, dtype=object) for k, v in data.items()})

    def load(self, path: str):
        """Load learned weights."""
        data = np.load(path, allow_pickle=True)
        for i, detector in enumerate(self.detectors):
            detector.input_synapse.weights = data['weights'][i]
        self.phoneme_labels = data['labels'].item()
        self.detection_counts = data['detection_counts']


class TemporalPatternLearner:
    """
    Learns temporal sequences of phonemes.

    Uses a recurrent network to predict next phoneme
    based on recent history, implementing basic grammar learning.
    """

    def __init__(
        self,
        n_phonemes: int = 40,
        n_context: int = 5,
        n_neurons: int = 1000
    ):
        self.n_phonemes = n_phonemes
        self.n_context = n_context
        self.n_neurons = n_neurons

        # Context buffer (recent phoneme sequence)
        self.context = np.zeros((n_context, n_phonemes), dtype=np.float32)

        # Prediction weights
        self.rng = np.random.default_rng(42)
        input_size = n_context * n_phonemes
        self.weights = self.rng.standard_normal((n_phonemes, input_size)).astype(np.float32) * 0.1

        # Learning history
        self.predictions = []
        self.actuals = []

    def predict(self, current_phoneme: int) -> np.ndarray:
        """Predict next phoneme distribution."""
        # Update context
        self.context = np.roll(self.context, -1, axis=0)
        self.context[-1] = 0
        self.context[-1, current_phoneme] = 1.0

        # Compute prediction
        context_flat = self.context.flatten()
        logits = np.dot(self.weights, context_flat)
        probs = self._softmax(logits)

        return probs

    def learn(self, actual_next: int, lr: float = 0.01):
        """Learn from prediction error."""
        context_flat = self.context.flatten()

        # Get current prediction
        logits = np.dot(self.weights, context_flat)
        probs = self._softmax(logits)

        # Compute error (cross-entropy gradient)
        target = np.zeros(self.n_phonemes, dtype=np.float32)
        target[actual_next] = 1.0

        error = probs - target

        # Update weights
        self.weights -= lr * np.outer(error, context_flat)

        # Track accuracy
        self.predictions.append(np.argmax(probs))
        self.actuals.append(actual_next)

    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def get_accuracy(self, window: int = 100) -> float:
        """Get recent prediction accuracy."""
        if len(self.predictions) < window:
            return 0.0
        recent_pred = self.predictions[-window:]
        recent_actual = self.actuals[-window:]
        return np.mean(np.array(recent_pred) == np.array(recent_actual))


if __name__ == "__main__":
    print("Testing Phoneme Learner")
    print("=" * 50)

    # Create learner
    config = PhonemeConfig(n_input=800, n_phoneme_neurons=100, n_phonemes=10)
    learner = PhonemeLearner(config)

    # Simulate learning with random patterns
    print("Training on random patterns...")

    # Create 10 distinct patterns (simulating different phonemes)
    patterns = [np.random.rand(800).astype(np.float32) for _ in range(10)]
    for i, p in enumerate(patterns):
        p[p < 0.7] = 0  # Sparsify
        patterns[i] = p

    # Train for 1000 steps
    for step in range(1000):
        pattern_idx = step % 10
        pattern = patterns[pattern_idx] + np.random.rand(800).astype(np.float32) * 0.1

        detected, conf = learner.process(
            (pattern > 0.5).astype(np.float32),
            learn=True,
            target_phoneme=pattern_idx
        )

        if step % 200 == 0:
            stats = learner.get_statistics()
            print(f"Step {step}: active detectors = {stats['active_detectors']}")

    # Test
    print("\nTesting detection...")
    correct = 0
    for pattern_idx in range(10):
        pattern = patterns[pattern_idx]
        detected, conf = learner.process((pattern > 0.5).astype(np.float32), learn=False)
        if detected == pattern_idx:
            correct += 1
        print(f"Pattern {pattern_idx} -> Detected {detected} (conf: {conf:.2f})")

    print(f"\nAccuracy: {correct}/10 = {correct/10:.0%}")
