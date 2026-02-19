"""Neuron models for the spiking neural network."""

from .izhikevich import IzhikevichNeuron, IzhikevichNeuronType
from .lif import LIFNeuron

__all__ = ["IzhikevichNeuron", "IzhikevichNeuronType", "LIFNeuron"]
