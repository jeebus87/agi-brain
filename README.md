# AGI Brain Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeebus87/agi-brain/blob/master/notebooks/agi_brain_colab.ipynb)

Spiking Neural Network cognitive architecture using Nengo/NengoSPA.

## Features

- **125K+ Neuron Architecture** - Working memory, rule application, analogy engine, executive control
- **Savant-Mode Memory** - Perfect retention with 12 memory slots
- **Neural Plasticity** - STDP, PES learning, reward-modulated plasticity
- **Embodied Agent** - GridWorld navigation with sensory processing and motor control
- **GPU Acceleration** - TensorFlow backend for large-scale simulation

## Quick Start

### Run on Google Colab (Recommended)

Click the badge above or [open the notebook directly](https://colab.research.google.com/github/jeebus87/agi-brain/blob/master/notebooks/agi_brain_colab.ipynb).

### Run Locally

```bash
# Clone
git clone https://github.com/jeebus87/agi-brain.git
cd agi-brain

# Install dependencies
pip install nengo nengo-spa tensorflow matplotlib numpy

# Run demos
python examples/multi_rule_reasoning.py    # Reasoning demo
python examples/learning_simple.py         # Learning demo
python examples/embodied_navigation.py     # Navigation demo
python examples/gpu_acceleration_demo.py   # GPU benchmark
```

## Architecture

```
agi-brain/
├── src/
│   ├── core/           # Neurons, synapses, populations
│   ├── cognition/      # Working memory, executive control
│   ├── reasoning/      # 100K POC, rules engine
│   ├── learning/       # STDP, associative learning
│   ├── integration/    # Embodiment, environment
│   └── acceleration/   # GPU backend, benchmarks
├── examples/           # Demo scripts
├── notebooks/          # Colab notebooks
└── tests/              # Unit tests
```

## Benchmarks

| Neurons | Nengo (CPU) | TensorFlow | Speedup |
|---------|-------------|------------|---------|
| 5,000   | 0.81 M/s    | 3.55 M/s   | 4.4x    |
| 10,000  | 0.39 M/s    | 5.47 M/s   | 14x     |
| 25,000  | 0.52 M/s    | 7.89 M/s   | 15x     |

## Components

### Reasoning POC (125K neurons)
- Problem Encoder (~20K neurons)
- Working Memory Buffer (~15K neurons, savant mode)
- Rule Application Engine (~20K neurons)
- Analogy Engine (~15K neurons)
- Executive Controller (~20K neurons)
- Response Generator (~10K neurons)

### Learning Rules
- **STDP** - Spike-timing-dependent plasticity
- **PES** - Prescribed Error Sensitivity
- **BCM** - Homeostatic plasticity
- **Reward-modulated** - Dopamine-like learning

### Embodiment
- GridWorld environment
- Vision processing (V1 -> IT pathway)
- Proprioception encoding
- Basal ganglia action selection
- 40% navigation success rate

## License

MIT
