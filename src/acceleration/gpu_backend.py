"""
GPU Backend - TensorFlow-based acceleration for neural simulations

Provides GPU-accelerated computation for:
- Large-scale neural population dynamics
- Matrix operations (weights, transforms)
- Batch simulation of multiple trials

Works as a companion to Nengo, accelerating the heavy computations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None


def check_gpu_available() -> Dict[str, any]:
    """Check GPU availability and return device info"""
    info = {
        'tensorflow_available': TF_AVAILABLE,
        'gpu_available': False,
        'gpu_devices': [],
        'gpu_memory': [],
        'recommended_backend': 'cpu'
    }

    if not TF_AVAILABLE:
        return info

    gpus = tf.config.list_physical_devices('GPU')
    info['gpu_devices'] = [gpu.name for gpu in gpus]
    info['gpu_available'] = len(gpus) > 0

    if info['gpu_available']:
        info['recommended_backend'] = 'gpu'
        # Try to get memory info
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except:
            pass

    return info


@dataclass
class NeuronPopulationGPU:
    """GPU-accelerated neuron population"""
    n_neurons: int
    dimensions: int
    encoders: np.ndarray  # (n_neurons, dimensions)
    gains: np.ndarray     # (n_neurons,)
    biases: np.ndarray    # (n_neurons,)
    tau_rc: float = 0.02
    tau_ref: float = 0.002


class GPUAccelerator:
    """
    TensorFlow-based GPU accelerator for neural computations

    Accelerates:
    - Neural activity computation (encoding)
    - Weight matrix operations
    - Batch processing of multiple inputs
    """

    def __init__(self, device: str = 'auto'):
        """
        Args:
            device: 'gpu', 'cpu', or 'auto' (auto-detect)
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

        self.device_type = device
        if device == 'auto':
            info = check_gpu_available()
            self.device_type = info['recommended_backend']

        self.device = f'/{self.device_type.upper()}:0'
        self._compiled_functions: Dict[str, Callable] = {}

        print(f"GPUAccelerator initialized on: {self.device}")

    @tf.function
    def _compute_lif_rates(
        self,
        x: tf.Tensor,
        encoders: tf.Tensor,
        gains: tf.Tensor,
        biases: tf.Tensor,
        tau_rc: float,
        tau_ref: float
    ) -> tf.Tensor:
        """Compute LIF neuron firing rates (GPU-accelerated)"""
        # J = gain * (encoders . x) + bias
        J = gains * tf.reduce_sum(encoders * x, axis=-1) + biases

        # LIF rate equation: r = 1 / (tau_ref - tau_rc * ln(1 - 1/J))
        # Only for J > 1
        J_clipped = tf.maximum(J, 1.0 + 1e-6)
        rates = 1.0 / (tau_ref - tau_rc * tf.math.log(1.0 - 1.0 / J_clipped))
        rates = tf.where(J > 1.0, rates, 0.0)

        return rates

    def compute_population_activity(
        self,
        population: NeuronPopulationGPU,
        inputs: np.ndarray,  # Shape: (batch_size, dimensions) or (dimensions,)
    ) -> np.ndarray:
        """
        Compute neural population activity for given inputs

        Args:
            population: Neuron population parameters
            inputs: Input values (can be batched)

        Returns:
            Firing rates for each neuron
        """
        with tf.device(self.device):
            # Convert to tensors
            x = tf.constant(inputs, dtype=tf.float32)
            if len(x.shape) == 1:
                x = tf.expand_dims(x, 0)  # Add batch dimension

            encoders = tf.constant(population.encoders, dtype=tf.float32)
            gains = tf.constant(population.gains, dtype=tf.float32)
            biases = tf.constant(population.biases, dtype=tf.float32)

            # Compute for each batch element
            rates = tf.map_fn(
                lambda xi: self._compute_lif_rates(
                    xi, encoders, gains, biases,
                    population.tau_rc, population.tau_ref
                ),
                x
            )

            return rates.numpy()

    @tf.function
    def _batch_matmul(self, x: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        """Batched matrix multiplication"""
        return tf.matmul(x, weights)

    def transform_signal(
        self,
        signal: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        Apply weight matrix transformation (GPU-accelerated)

        Args:
            signal: Input signal (batch_size, in_dims) or (in_dims,)
            weights: Weight matrix (in_dims, out_dims)

        Returns:
            Transformed signal
        """
        with tf.device(self.device):
            x = tf.constant(signal, dtype=tf.float32)
            w = tf.constant(weights, dtype=tf.float32)

            if len(x.shape) == 1:
                result = tf.tensordot(x, w, axes=1)
            else:
                result = self._batch_matmul(x, w)

            return result.numpy()

    def batch_simulate(
        self,
        populations: List[NeuronPopulationGPU],
        inputs_sequence: np.ndarray,  # (time_steps, n_populations, dimensions)
        weights: List[np.ndarray],    # Connection weights between populations
        dt: float = 0.001,
        synapse_tau: float = 0.005
    ) -> Dict[str, np.ndarray]:
        """
        Simulate network dynamics on GPU

        Args:
            populations: List of neuron populations
            inputs_sequence: Input time series for first population
            weights: Connection weight matrices
            dt: Simulation timestep
            synapse_tau: Synaptic filter time constant

        Returns:
            Dict with activity traces for each population
        """
        n_steps = inputs_sequence.shape[0]
        n_pops = len(populations)

        # Pre-allocate output arrays
        activities = {
            f'pop_{i}': np.zeros((n_steps, pop.n_neurons))
            for i, pop in enumerate(populations)
        }

        # Synaptic filter coefficient
        alpha = dt / synapse_tau

        with tf.device(self.device):
            # Convert all parameters to tensors
            encoders_list = [tf.constant(p.encoders, dtype=tf.float32) for p in populations]
            gains_list = [tf.constant(p.gains, dtype=tf.float32) for p in populations]
            biases_list = [tf.constant(p.biases, dtype=tf.float32) for p in populations]
            weights_tf = [tf.constant(w, dtype=tf.float32) for w in weights]

            # Initialize filtered activities
            filtered = [tf.zeros(p.n_neurons, dtype=tf.float32) for p in populations]

            for t in range(n_steps):
                # Get input for this timestep
                x_input = tf.constant(inputs_sequence[t], dtype=tf.float32)

                for i, pop in enumerate(populations):
                    # Input to this population
                    if i == 0:
                        x = x_input
                    else:
                        # Transform from previous population
                        x = tf.tensordot(filtered[i-1], weights_tf[i-1], axes=1)

                    # Compute rates
                    rates = self._compute_lif_rates(
                        x, encoders_list[i], gains_list[i], biases_list[i],
                        pop.tau_rc, pop.tau_ref
                    )

                    # Synaptic filter
                    filtered[i] = filtered[i] * (1 - alpha) + rates * alpha

                    # Store
                    activities[f'pop_{i}'][t] = filtered[i].numpy()

        return activities

    def benchmark(
        self,
        n_neurons: int = 10000,
        dimensions: int = 64,
        n_steps: int = 1000,
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Benchmark GPU performance

        Returns:
            Dict with timing results
        """
        # Create random population
        rng = np.random.default_rng(42)
        pop = NeuronPopulationGPU(
            n_neurons=n_neurons,
            dimensions=dimensions,
            encoders=rng.standard_normal((n_neurons, dimensions)).astype(np.float32),
            gains=rng.uniform(1, 2, n_neurons).astype(np.float32),
            biases=rng.uniform(-1, 1, n_neurons).astype(np.float32)
        )
        pop.encoders /= np.linalg.norm(pop.encoders, axis=1, keepdims=True)

        # Random inputs
        inputs = rng.standard_normal((n_steps, batch_size, dimensions)).astype(np.float32)

        # Warmup
        _ = self.compute_population_activity(pop, inputs[0])

        # Benchmark
        start = time.perf_counter()
        for t in range(n_steps):
            _ = self.compute_population_activity(pop, inputs[t])
        elapsed = time.perf_counter() - start

        return {
            'n_neurons': n_neurons,
            'dimensions': dimensions,
            'n_steps': n_steps,
            'total_time': elapsed,
            'time_per_step': elapsed / n_steps,
            'neurons_per_second': (n_neurons * n_steps) / elapsed,
            'device': self.device_type
        }


class LargeScaleSimulator:
    """
    Simulator for million+ neuron networks

    Uses GPU for heavy computation, with optimized memory management
    for very large networks.
    """

    def __init__(
        self,
        n_populations: int,
        neurons_per_population: int,
        dimensions: int = 64,
        device: str = 'auto'
    ):
        self.n_populations = n_populations
        self.neurons_per_pop = neurons_per_population
        self.dimensions = dimensions
        self.total_neurons = n_populations * neurons_per_population

        self.accelerator = GPUAccelerator(device=device)

        # Initialize populations
        self.populations = self._create_populations()
        self.weights = self._create_weights()

        print(f"LargeScaleSimulator: {self.total_neurons:,} total neurons")

    def _create_populations(self) -> List[NeuronPopulationGPU]:
        """Create random neuron populations"""
        rng = np.random.default_rng(42)
        populations = []

        for i in range(self.n_populations):
            encoders = rng.standard_normal((self.neurons_per_pop, self.dimensions))
            encoders /= np.linalg.norm(encoders, axis=1, keepdims=True)

            pop = NeuronPopulationGPU(
                n_neurons=self.neurons_per_pop,
                dimensions=self.dimensions,
                encoders=encoders.astype(np.float32),
                gains=rng.uniform(1, 2, self.neurons_per_pop).astype(np.float32),
                biases=rng.uniform(-1, 1, self.neurons_per_pop).astype(np.float32)
            )
            populations.append(pop)

        return populations

    def _create_weights(self) -> List[np.ndarray]:
        """Create connection weights between populations"""
        rng = np.random.default_rng(43)
        weights = []

        for i in range(self.n_populations - 1):
            # Sparse random weights
            w = rng.standard_normal((self.neurons_per_pop, self.dimensions))
            w /= np.sqrt(self.neurons_per_pop)
            weights.append(w.astype(np.float32))

        return weights

    def run(
        self,
        input_signal: np.ndarray,
        dt: float = 0.001
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation with given input

        Args:
            input_signal: (n_steps, dimensions) input to first population
            dt: Simulation timestep

        Returns:
            Activity traces for all populations
        """
        return self.accelerator.batch_simulate(
            self.populations,
            input_signal[:, np.newaxis, :] if len(input_signal.shape) == 2 else input_signal,
            self.weights,
            dt=dt
        )
