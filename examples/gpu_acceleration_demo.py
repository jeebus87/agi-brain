"""
GPU Acceleration Demo - Scale up neural simulations

Demonstrates:
1. GPU availability check
2. Performance comparison (CPU vs GPU)
3. Scaling analysis
4. Large-scale network simulation

Run with: python examples/gpu_acceleration_demo.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import time
import matplotlib.pyplot as plt

from src.acceleration.gpu_backend import (
    GPUAccelerator,
    NeuronPopulationGPU,
    LargeScaleSimulator,
    check_gpu_available
)
from src.acceleration.benchmarks import (
    run_benchmark,
    compare_backends,
    scaling_analysis,
    print_benchmark_summary
)
from src.acceleration.optimized_sim import estimate_memory_usage


def demo_gpu_check():
    """Check GPU availability"""
    print("=" * 60)
    print("  GPU AVAILABILITY CHECK")
    print("=" * 60)
    print()

    info = check_gpu_available()

    print(f"  TensorFlow available: {info['tensorflow_available']}")
    print(f"  GPU available: {info['gpu_available']}")

    if info['gpu_devices']:
        print(f"  GPU devices: {info['gpu_devices']}")
    else:
        print("  GPU devices: None (will use CPU)")

    print(f"  Recommended backend: {info['recommended_backend']}")
    print()

    return info


def demo_basic_acceleration():
    """Demonstrate basic GPU acceleration"""
    print("=" * 60)
    print("  BASIC GPU ACCELERATION")
    print("=" * 60)
    print()

    # Create accelerator
    print("[1] Creating GPU accelerator...")
    try:
        accel = GPUAccelerator(device='auto')
    except ImportError as e:
        print(f"    Error: {e}")
        print("    Falling back to CPU-only demonstration")
        return

    # Create a test population
    print("\n[2] Creating neural population...")
    n_neurons = 10000
    dimensions = 64

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
    print(f"    Neurons: {n_neurons:,}")
    print(f"    Dimensions: {dimensions}")

    # Test computation
    print("\n[3] Computing neural activity...")
    test_input = np.random.randn(dimensions).astype(np.float32)

    # Warmup
    _ = accel.compute_population_activity(pop, test_input)

    # Timed run
    n_trials = 100
    start = time.perf_counter()
    for _ in range(n_trials):
        activity = accel.compute_population_activity(pop, test_input)
    elapsed = time.perf_counter() - start

    print(f"    Time for {n_trials} computations: {elapsed*1000:.2f} ms")
    print(f"    Time per computation: {elapsed/n_trials*1000:.3f} ms")
    print(f"    Neurons per second: {n_neurons * n_trials / elapsed / 1e6:.2f}M")
    print()

    return accel


def demo_performance_comparison():
    """Compare CPU vs GPU performance"""
    print("=" * 60)
    print("  PERFORMANCE COMPARISON")
    print("=" * 60)
    print()

    neuron_counts = [1000, 5000, 10000, 25000]

    print("Running benchmarks...")
    results = compare_backends(
        neuron_counts=neuron_counts,
        dimensions=64,
        sim_time=0.5,
        backends=['nengo', 'gpu']
    )

    print_benchmark_summary(results)

    # Visualize
    print("\n[Generating performance visualization...]")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Throughput comparison
    ax1 = axes[0]
    for backend, backend_results in results.items():
        valid_results = [r for r in backend_results if r]
        if valid_results:
            neurons = [r.n_neurons for r in valid_results]
            rates = [r.neurons_per_second / 1e6 for r in valid_results]
            ax1.plot(neurons, rates, 'o-', label=backend, linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Neurons')
    ax1.set_ylabel('Million Neurons/Second')
    ax1.set_title('Throughput Comparison', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')

    # Wall time comparison
    ax2 = axes[1]
    for backend, backend_results in results.items():
        valid_results = [r for r in backend_results if r]
        if valid_results:
            neurons = [r.n_neurons for r in valid_results]
            times = [r.wall_time for r in valid_results]
            ax2.plot(neurons, times, 'o-', label=backend, linewidth=2, markersize=8)

    ax2.set_xlabel('Number of Neurons')
    ax2.set_ylabel('Wall Time (seconds)')
    ax2.set_title('Simulation Time Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('gpu_benchmark.png', dpi=150, bbox_inches='tight')
    print("Saved to gpu_benchmark.png")

    return results


def demo_large_scale():
    """Demonstrate large-scale simulation"""
    print("\n" + "=" * 60)
    print("  LARGE-SCALE SIMULATION")
    print("=" * 60)
    print()

    # Memory estimation
    print("[1] Memory estimation for different scales:")
    scales = [
        (100000, "100K neurons (POC)"),
        (1000000, "1M neurons (small brain region)"),
        (10000000, "10M neurons (cortical area)"),
        (100000000, "100M neurons (mouse brain scale)"),
    ]

    for n_neurons, desc in scales:
        mem = estimate_memory_usage(n_neurons, 64, n_neurons // 100)
        print(f"    {desc}:")
        print(f"        Memory: {mem['total_gb']:.2f} GB")
    print()

    # Create large-scale simulator
    print("[2] Creating large-scale simulator...")
    try:
        sim = LargeScaleSimulator(
            n_populations=10,
            neurons_per_population=10000,  # 100K total
            dimensions=64,
            device='auto'
        )

        # Create input signal
        print("\n[3] Running simulation (1 second)...")
        t = np.linspace(0, 1, 1000)
        input_signal = np.sin(2 * np.pi * t)[:, np.newaxis] * np.ones(64)
        input_signal = input_signal.astype(np.float32)

        start = time.perf_counter()
        activities = sim.run(input_signal, dt=0.001)
        elapsed = time.perf_counter() - start

        print(f"    Simulation time: 1.0 seconds")
        print(f"    Wall time: {elapsed:.2f} seconds")
        print(f"    Speedup: {1.0/elapsed:.2f}x real-time")
        print(f"    Total neurons: {sim.total_neurons:,}")

        # Visualize
        print("\n[4] Generating activity visualization...")

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        # First population activity
        ax1 = axes[0]
        pop0_activity = activities['pop_0']
        # Show subset of neurons
        n_show = min(100, pop0_activity.shape[1])
        im = ax1.imshow(
            pop0_activity[:, :n_show].T,
            aspect='auto',
            cmap='hot',
            extent=[0, 1, n_show, 0]
        )
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Neuron Index')
        ax1.set_title('Population 0 Activity (first 100 neurons)', fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Firing Rate')

        # Population averages
        ax2 = axes[1]
        for i in range(min(5, len(activities))):
            pop_activity = activities[f'pop_{i}']
            avg = np.mean(pop_activity, axis=1)
            ax2.plot(t, avg, label=f'Pop {i}', alpha=0.8)

        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Mean Activity')
        ax2.set_title('Population Mean Activity Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('large_scale_activity.png', dpi=150, bbox_inches='tight')
        print("    Saved to large_scale_activity.png")

    except Exception as e:
        print(f"    Error: {e}")
        print("    (Large-scale simulation requires sufficient memory)")


def demo_scaling_analysis():
    """Analyze scaling behavior"""
    print("\n" + "=" * 60)
    print("  SCALING ANALYSIS")
    print("=" * 60)
    print()

    print("Analyzing how simulation scales with network size...")

    results = scaling_analysis(
        max_neurons=50000,
        steps=5,
        backend='gpu'
    )

    # Visualize scaling
    print("\n[Generating scaling visualization...]")

    fig, ax = plt.subplots(figsize=(10, 6))

    neurons = [r.n_neurons for r in results if r]
    rates = [r.neurons_per_second / 1e6 for r in results if r]

    ax.loglog(neurons, rates, 'bo-', linewidth=2, markersize=10)

    # Add ideal scaling line
    ideal = [rates[0] * (n / neurons[0]) for n in neurons]
    ax.loglog(neurons, ideal, 'r--', alpha=0.5, label='Ideal (linear) scaling')

    ax.set_xlabel('Number of Neurons')
    ax.set_ylabel('Million Neurons/Second')
    ax.set_title('Scaling Analysis', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('scaling_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved to scaling_analysis.png")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("  GPU ACCELERATION DEMONSTRATION")
    print("  Scaling Up the AGI Brain Simulation")
    print("=" * 60)
    print()

    # Check GPU
    info = demo_gpu_check()

    # Basic acceleration
    demo_basic_acceleration()

    # Performance comparison
    demo_performance_comparison()

    # Large-scale simulation
    demo_large_scale()

    # Scaling analysis
    demo_scaling_analysis()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print()
    print("  GPU acceleration infrastructure is ready:")
    print("    - TensorFlow-based GPU backend")
    print("    - Optimized CPU fallback")
    print("    - Benchmarking tools")
    print("    - Large-scale simulation support")
    print()
    print("  Next steps for production scaling:")
    print("    - Deploy on cloud GPU instances (A100)")
    print("    - Implement distributed simulation")
    print("    - Add checkpoint/resume for long runs")
    print()
    print("=" * 60)


if __name__ == '__main__':
    main()
