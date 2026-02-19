"""Core components for the AGI brain simulation."""

from .neurons import IzhikevichNeuron, LIFNeuron
from .populations import NeuralPopulation
from .networks import CorticalColumn

__all__ = [
    "IzhikevichNeuron",
    "LIFNeuron",
    "NeuralPopulation",
    "CorticalColumn",
]
