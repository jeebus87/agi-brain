"""
Benchmarks - Performance comparison across backends

Compare simulation performance:
- Standard Nengo (CPU)
- GPU-accelerated (TensorFlow)
- Optimized CPU (multi-threaded)
- Scaling analysis
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import nengo
import nengo_spa as spa

from .gpu_backend import GPUAccelerator, NeuronPopulationGPU, check_gpu_available


@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    backend: str
    n_neurons: int
    dimensions: int
    sim_time: float
    wall_time: float
    neurons_per_second: float
    timesteps_per_second: float
    memory_mb: float


def run_benchmark(
    n_neurons: int = 10000,
    dimensions: int = 64,
    sim_time: float = 1.0,
    dt: float = 0.001,
    backend: str = 'nengo'
) -> BenchmarkResult:
    """
    Run performance benchmark

    Args:
        n_neurons: Number of neurons
        dimensions: Vector dimensions
        sim_time: Simulation time in seconds
        dt: Timestep
        backend: 'nengo', 'gpu', or 'optimized'

    Returns:
        BenchmarkResult with timing data
    """
    n_steps = int(sim_time / dt)

    if backend == 'nengo':
        return _benchmark_nengo(n_neurons, dimensions, sim_time, dt)
    elif backend == 'gpu':
        return _benchmark_gpu(n_neurons, dimensions, n_steps, dt)
    elif backend == 'optimized':
        return _benchmark_optimized(n_neurons, dimensions, sim_time, dt)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _benchmark_nengo(
    n_neurons: int,
    dimensions: int,
    sim_time: float,
    dt: float
) -> BenchmarkResult:
    """Benchmark standard Nengo simulator"""

    # Create simple network
    with nengo.Network(seed=42) as model:
        inp = nengo.Node(output=lambda t: np.sin(2 * np.pi * t))

        # Scale neurons across ensembles if very large
        if n_neurons <= 10000:
            ens = nengo.Ensemble(n_neurons, dimensions)
        else:
            # Split into multiple ensembles
            n_ens = (n_neurons + 9999) // 10000
            neurons_per = n_neurons // n_ens
            ensembles = [
                nengo.Ensemble(neurons_per, dimensions, label=f"ens_{i}")
                for i in range(n_ens)
            ]
            ens = ensembles[0]
            for e in ensembles[1:]:
                nengo.Connection(ens, e)

        nengo.Connection(inp, ens[0])
        probe = nengo.Probe(ens, synapse=0.01)

    # Run and time
    start_time = time.perf_counter()

    with nengo.Simulator(model, dt=dt, progress_bar=False) as sim:
        sim.run(sim_time)

    wall_time = time.perf_counter() - start_time

    return BenchmarkResult(
        backend='nengo',
        n_neurons=n_neurons,
        dimensions=dimensions,
        sim_time=sim_time,
        wall_time=wall_time,
        neurons_per_second=n_neurons * (sim_time / dt) / wall_time,
        timesteps_per_second=(sim_time / dt) / wall_time,
        memory_mb=0  # TODO: measure actual memory
    )


def _benchmark_gpu(
    n_neurons: int,
    dimensions: int,
    n_steps: int,
    dt: float
) -> BenchmarkResult:
    """Benchmark GPU backend"""

    try:
        accelerator = GPUAccelerator(device='auto')
    except ImportError:
        return BenchmarkResult(
            backend='gpu (unavailable)',
            n_neurons=n_neurons,
            dimensions=dimensions,
            sim_time=n_steps * dt,
            wall_time=float('inf'),
            neurons_per_second=0,
            timesteps_per_second=0,
            memory_mb=0
        )

    # Create population
    rng = np.random.default_rng(42)
    encoders = rng.standard_normal((n_neurons, dimensions)).astype(np.float32)
    encoders /= np.linalg.norm(encoders, axis=1, keepdims=True)

    pop = NeuronPopulationGPU(
        n_neurons=n_neurons,
        dimensions=dimensions,
        encoders=encoders,
        gains=rng.uniform(1, 2, n_neurons).astype(np.float32),
        biases=rng.uniform(-1, 1, n_neurons).astype(np.float32)
    )

    # Create input sequence
    t = np.arange(n_steps) * dt
    inputs = np.sin(2 * np.pi * t)[:, np.newaxis] * np.ones(dimensions)
    inputs = inputs.astype(np.float32)

    # Warmup
    _ = accelerator.compute_population_activity(pop, inputs[:10])

    # Benchmark
    start_time = time.perf_counter()
    for i in range(n_steps):
        _ = accelerator.compute_population_activity(pop, inputs[i])
    wall_time = time.perf_counter() - start_time

    return BenchmarkResult(
        backend=f'gpu ({accelerator.device_type})',
        n_neurons=n_neurons,
        dimensions=dimensions,
        sim_time=n_steps * dt,
        wall_time=wall_time,
        neurons_per_second=n_neurons * n_steps / wall_time,
        timesteps_per_second=n_steps / wall_time,
        memory_mb=0
    )


def _benchmark_optimized(
    n_neurons: int,
    dimensions: int,
    sim_time: float,
    dt: float
) -> BenchmarkResult:
    """Benchmark optimized CPU simulator"""

    # Use Nengo with optimization flag
    with nengo.Network(seed=42) as model:
        inp = nengo.Node(output=lambda t: np.sin(2 * np.pi * t))

        if n_neurons <= 10000:
            ens = nengo.Ensemble(n_neurons, dimensions)
        else:
            n_ens = (n_neurons + 9999) // 10000
            neurons_per = n_neurons // n_ens
            ensembles = [
                nengo.Ensemble(neurons_per, dimensions, label=f"ens_{i}")
                for i in range(n_ens)
            ]
            ens = ensembles[0]

        nengo.Connection(inp, ens[0])
        probe = nengo.Probe(ens, synapse=0.01)

    start_time = time.perf_counter()

    # Use optimized simulator settings
    with nengo.Simulator(
        model,
        dt=dt,
        progress_bar=False,
        optimize=True
    ) as sim:
        sim.run(sim_time)

    wall_time = time.perf_counter() - start_time

    return BenchmarkResult(
        backend='optimized',
        n_neurons=n_neurons,
        dimensions=dimensions,
        sim_time=sim_time,
        wall_time=wall_time,
        neurons_per_second=n_neurons * (sim_time / dt) / wall_time,
        timesteps_per_second=(sim_time / dt) / wall_time,
        memory_mb=0
    )


def compare_backends(
    neuron_counts: List[int] = [1000, 5000, 10000, 50000],
    dimensions: int = 64,
    sim_time: float = 1.0,
    backends: List[str] = ['nengo', 'gpu']
) -> Dict[str, List[BenchmarkResult]]:
    """
    Compare performance across backends and scales

    Args:
        neuron_counts: List of neuron counts to test
        dimensions: Vector dimensions
        sim_time: Simulation time
        backends: Backends to compare

    Returns:
        Dict mapping backend name to list of results
    """
    results = {b: [] for b in backends}

    for n_neurons in neuron_counts:
        print(f"\nBenchmarking {n_neurons:,} neurons...")

        for backend in backends:
            print(f"  {backend}...", end=" ", flush=True)
            try:
                result = run_benchmark(
                    n_neurons=n_neurons,
                    dimensions=dimensions,
                    sim_time=sim_time,
                    backend=backend
                )
                results[backend].append(result)
                print(f"{result.neurons_per_second/1e6:.2f}M neurons/sec")
            except Exception as e:
                print(f"Error: {e}")
                results[backend].append(None)

    return results


def scaling_analysis(
    max_neurons: int = 100000,
    steps: int = 5,
    backend: str = 'gpu'
) -> List[BenchmarkResult]:
    """
    Analyze how performance scales with network size

    Returns:
        List of benchmark results at different scales
    """
    neuron_counts = np.logspace(3, np.log10(max_neurons), steps).astype(int)
    results = []

    print(f"\nScaling analysis for {backend} backend:")
    print("-" * 50)

    for n in neuron_counts:
        result = run_benchmark(n_neurons=n, backend=backend, sim_time=0.5)
        results.append(result)
        print(f"  {n:>8,} neurons: {result.neurons_per_second/1e6:>8.2f}M neurons/sec "
              f"({result.wall_time:.3f}s)")

    return results


def print_benchmark_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print formatted benchmark summary"""

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Get all neuron counts
    all_counts = set()
    for backend_results in results.values():
        for r in backend_results:
            if r:
                all_counts.add(r.n_neurons)
    neuron_counts = sorted(all_counts)

    # Header
    print(f"\n{'Neurons':>10}", end="")
    for backend in results.keys():
        print(f" | {backend:>15}", end="")
    print(" |")
    print("-" * (12 + 18 * len(results)))

    # Data rows
    for n in neuron_counts:
        print(f"{n:>10,}", end="")
        for backend, backend_results in results.items():
            result = next((r for r in backend_results if r and r.n_neurons == n), None)
            if result:
                rate = result.neurons_per_second / 1e6
                print(f" | {rate:>12.2f} M/s", end="")
            else:
                print(f" | {'N/A':>15}", end="")
        print(" |")

    print("-" * (12 + 18 * len(results)))

    # Best performer
    print("\nBest performers:")
    for n in neuron_counts:
        best_backend = None
        best_rate = 0
        for backend, backend_results in results.items():
            result = next((r for r in backend_results if r and r.n_neurons == n), None)
            if result and result.neurons_per_second > best_rate:
                best_rate = result.neurons_per_second
                best_backend = backend
        if best_backend:
            print(f"  {n:>8,} neurons: {best_backend} ({best_rate/1e6:.2f}M neurons/sec)")
