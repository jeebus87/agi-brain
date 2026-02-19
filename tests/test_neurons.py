"""Tests for neuron models."""

import pytest
import numpy as np

# Skip tests if nengo not installed
pytest.importorskip("nengo")

import nengo
from src.core.neurons.izhikevich import (
    IzhikevichNeuron,
    IzhikevichNeuronType,
    IzhikevichParams,
)
from src.core.neurons.lif import LIFNeuron


class TestIzhikevichParams:
    """Tests for Izhikevich parameter presets."""

    def test_regular_spiking_params(self):
        """Test regular spiking neuron parameters."""
        params = IzhikevichParams.from_type(IzhikevichNeuronType.REGULAR_SPIKING)
        assert params.a == 0.02
        assert params.b == 0.2
        assert params.c == -65
        assert params.d == 8

    def test_fast_spiking_params(self):
        """Test fast spiking neuron parameters."""
        params = IzhikevichParams.from_type(IzhikevichNeuronType.FAST_SPIKING)
        assert params.a == 0.1
        assert params.b == 0.2
        assert params.c == -65
        assert params.d == 2

    def test_all_neuron_types_have_params(self):
        """Ensure all neuron types have defined parameters."""
        for neuron_type in IzhikevichNeuronType:
            params = IzhikevichParams.from_type(neuron_type)
            assert params.a > 0
            assert params.c < 0  # Reset voltage is negative


class TestIzhikevichNeuron:
    """Tests for Izhikevich neuron model."""

    def test_neuron_creation(self):
        """Test basic neuron creation."""
        neuron = IzhikevichNeuron()
        assert neuron is not None
        assert neuron.spiking is True

    def test_neuron_with_custom_params(self):
        """Test neuron with custom parameters."""
        params = IzhikevichParams(a=0.05, b=0.25, c=-60, d=4)
        neuron = IzhikevichNeuron(params=params)
        assert neuron.a == 0.05
        assert neuron.b == 0.25

    def test_neuron_in_ensemble(self):
        """Test that neuron works in a Nengo ensemble."""
        with nengo.Network() as model:
            ens = nengo.Ensemble(
                n_neurons=10,
                dimensions=1,
                neuron_type=IzhikevichNeuron(),
            )
        assert ens is not None
        assert ens.n_neurons == 10


class TestLIFNeuron:
    """Tests for LIF neuron model."""

    def test_cortical_preset(self):
        """Test cortical neuron preset."""
        neuron = LIFNeuron.cortical()
        assert neuron.tau_rc == 0.02
        assert neuron.tau_ref == 0.002

    def test_fast_inhibitory_preset(self):
        """Test fast inhibitory neuron preset."""
        neuron = LIFNeuron.fast_inhibitory()
        assert neuron.tau_rc == 0.01
        assert neuron.tau_ref == 0.001

    def test_neuron_in_ensemble(self):
        """Test that neuron works in a Nengo ensemble."""
        with nengo.Network() as model:
            ens = nengo.Ensemble(
                n_neurons=10,
                dimensions=1,
                neuron_type=LIFNeuron(),
            )
        assert ens is not None
