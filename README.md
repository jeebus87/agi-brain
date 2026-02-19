# AGI Brain Simulation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeebus87/agi-brain/blob/master/notebooks/agi_brain_colab.ipynb)
[![Language Learning](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeebus87/agi-brain/blob/master/notebooks/agi_language_learning.ipynb)
[![Quick Chat](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jeebus87/agi-brain/blob/master/notebooks/agi_chat_quick.ipynb)

Spiking Neural Network cognitive architecture that learns language from scratch - no pre-trained LLMs.

## Features

- **10M+ Neuron Sparse Architecture** - Efficient memory usage, fits in free Colab GPU
- **Language Learning from Scratch** - No pre-trained models, learns through STDP
- **Speech I/O** - Cochlea-based audio encoding, formant speech synthesis
- **Semantic Memory** - Sparse Distributed Memory for word associations
- **YouTube/Web Learning** - Learn from videos and web pages
- **125K+ Neuron Reasoning** - Working memory, rule application, analogy engine
- **Embodied Agent** - GridWorld navigation with visual sensing
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
│   ├── language/       # Language learning from scratch
│   │   ├── sparse_network.py    # 10M neuron efficient SNN
│   │   ├── audio_encoder.py     # Cochlea simulation
│   │   ├── phoneme_learner.py   # STDP phoneme recognition
│   │   ├── semantic_memory.py   # Word-meaning associations
│   │   └── speech_generator.py  # Neural vocoder
│   ├── interface/      # Chat UI, voice I/O, learning pipeline
│   ├── integration/    # Embodiment, environment
│   └── acceleration/   # GPU backend, benchmarks
├── examples/           # Demo scripts
├── notebooks/          # Colab notebooks
└── tests/              # Unit tests
```

## Language Learning

The brain learns language without any pre-trained LLM:

| Component | Function |
|-----------|----------|
| **Cochlea Encoder** | Converts audio to spike patterns (like biological ear) |
| **Phoneme Learner** | Recognizes speech sounds via STDP |
| **Semantic Memory** | Associates words with meanings |
| **Speech Generator** | Produces speech from neural activity |

### Learning Sources
- **Conversation** - Learns from chat interactions
- **YouTube** - Downloads and transcribes videos
- **Web Pages** - Extracts and learns from text content

## Benchmarks

| Neurons | Nengo (CPU) | TensorFlow | Speedup |
|---------|-------------|------------|---------|
| 5,000   | 0.81 M/s    | 3.55 M/s   | 4.4x    |
| 10,000  | 0.39 M/s    | 5.47 M/s   | 14x     |
| 25,000  | 0.52 M/s    | 7.89 M/s   | 15x     |

## Components

### Reasoning POC (125K neurons)
- Problem Encoder (~20K neurons)
- Working Memory Buffer (~15K neurons, persistent mode)
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
- Goal-directed navigation with visual sensing

## License

MIT
