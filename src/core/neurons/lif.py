"""
Leaky Integrate-and-Fire (LIF) neuron model.

The LIF model is a simpler, more computationally efficient neuron model
that can be used as a fallback when Izhikevich neurons are too expensive.
"""

import numpy as np
import nengo


class LIFNeuron(nengo.neurons.LIF):
    """Extended LIF neuron with additional features.

    This wraps Nengo's built-in LIF neuron with additional functionality
    for compatibility with the AGI brain architecture.
    """

    def __init__(
        self,
        tau_rc: float = 0.02,
        tau_ref: float = 0.002,
        min_voltage: float = 0.0,
        amplitude: float = 1.0,
    ):
        """Initialize the LIF neuron.

        Args:
            tau_rc: Membrane time constant (seconds). Controls how quickly
                    the membrane potential decays.
            tau_ref: Absolute refractory period (seconds). The neuron cannot
                    spike during this period after each spike.
            min_voltage: Minimum membrane voltage (prevents unbounded decay).
            amplitude: Scaling factor for output spikes.
        """
        super().__init__(
            tau_rc=tau_rc,
            tau_ref=tau_ref,
            min_voltage=min_voltage,
            amplitude=amplitude,
        )

    @classmethod
    def cortical(cls) -> "LIFNeuron":
        """Create a LIF neuron with typical cortical neuron parameters."""
        return cls(tau_rc=0.02, tau_ref=0.002)

    @classmethod
    def fast_inhibitory(cls) -> "LIFNeuron":
        """Create a fast-spiking inhibitory neuron."""
        return cls(tau_rc=0.01, tau_ref=0.001)

    @classmethod
    def slow_excitatory(cls) -> "LIFNeuron":
        """Create a slow-spiking excitatory neuron."""
        return cls(tau_rc=0.05, tau_ref=0.003)
