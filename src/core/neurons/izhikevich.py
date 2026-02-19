"""
Izhikevich neuron model implementation.

The Izhikevich model provides a computationally efficient yet biologically
plausible neuron model that can reproduce many spiking patterns observed
in real cortical neurons.

Reference:
    Izhikevich, E.M. (2003). Simple model of spiking neurons.
    IEEE Transactions on Neural Networks, 14(6), 1569-1572.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
import nengo


class IzhikevichNeuronType(Enum):
    """Standard Izhikevich neuron types with preset parameters."""

    # Regular spiking (RS) - most common excitatory cortical neuron
    REGULAR_SPIKING = "RS"

    # Intrinsically bursting (IB) - pyramidal neurons in layer 5
    INTRINSICALLY_BURSTING = "IB"

    # Chattering (CH) - fast-rhythmic-bursting neurons
    CHATTERING = "CH"

    # Fast spiking (FS) - inhibitory interneurons (basket cells)
    FAST_SPIKING = "FS"

    # Low-threshold spiking (LTS) - inhibitory interneurons
    LOW_THRESHOLD_SPIKING = "LTS"

    # Thalamo-cortical (TC) - thalamic relay neurons
    THALAMOCORTICAL = "TC"

    # Resonator (RZ) - neurons that resonate at specific frequencies
    RESONATOR = "RZ"


@dataclass
class IzhikevichParams:
    """Parameters for the Izhikevich neuron model.

    The model is defined by:
        v' = 0.04v² + 5v + 140 - u + I
        u' = a(bv - u)

    With reset condition:
        if v >= 30mV: v = c, u = u + d

    Attributes:
        a: Time scale of recovery variable u. Smaller = slower recovery.
        b: Sensitivity of u to subthreshold v. >0 is saddle-node, <0 is Andronov-Hopf.
        c: After-spike reset value of v (mV).
        d: After-spike increment of u.
    """
    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_type(cls, neuron_type: IzhikevichNeuronType) -> "IzhikevichParams":
        """Create parameters for a standard neuron type."""
        params = {
            IzhikevichNeuronType.REGULAR_SPIKING: cls(a=0.02, b=0.2, c=-65, d=8),
            IzhikevichNeuronType.INTRINSICALLY_BURSTING: cls(a=0.02, b=0.2, c=-55, d=4),
            IzhikevichNeuronType.CHATTERING: cls(a=0.02, b=0.2, c=-50, d=2),
            IzhikevichNeuronType.FAST_SPIKING: cls(a=0.1, b=0.2, c=-65, d=2),
            IzhikevichNeuronType.LOW_THRESHOLD_SPIKING: cls(a=0.02, b=0.25, c=-65, d=2),
            IzhikevichNeuronType.THALAMOCORTICAL: cls(a=0.02, b=0.25, c=-65, d=0.05),
            IzhikevichNeuronType.RESONATOR: cls(a=0.1, b=0.26, c=-65, d=2),
        }
        return params[neuron_type]


class IzhikevichNeuron(nengo.neurons.NeuronType):
    """Izhikevich spiking neuron model for Nengo.

    This neuron type can be used in Nengo ensembles to create
    biologically plausible spiking neural networks.

    Example:
        >>> import nengo
        >>> from agi_brain.core.neurons import IzhikevichNeuron
        >>>
        >>> with nengo.Network() as model:
        ...     ens = nengo.Ensemble(
        ...         n_neurons=100,
        ...         dimensions=1,
        ...         neuron_type=IzhikevichNeuron()
        ...     )
    """

    # Nengo requires these class attributes
    probeable = ("spikes", "voltage", "recovery")
    negative = False
    spiking = True

    def __init__(
        self,
        neuron_type: IzhikevichNeuronType = IzhikevichNeuronType.REGULAR_SPIKING,
        params: IzhikevichParams = None,
        tau_rc: float = 0.02,
        tau_ref: float = 0.002,
    ):
        """Initialize the Izhikevich neuron.

        Args:
            neuron_type: Preset neuron type (ignored if params provided).
            params: Custom Izhikevich parameters.
            tau_rc: Membrane time constant (for compatibility).
            tau_ref: Refractory period (for compatibility).
        """
        super().__init__()

        if params is None:
            params = IzhikevichParams.from_type(neuron_type)

        self.params = params
        self.tau_rc = tau_rc
        self.tau_ref = tau_ref
        self._neuron_type = neuron_type

    @property
    def a(self) -> float:
        return self.params.a

    @property
    def b(self) -> float:
        return self.params.b

    @property
    def c(self) -> float:
        return self.params.c

    @property
    def d(self) -> float:
        return self.params.d

    def gain_bias(
        self, max_rates: np.ndarray, intercepts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the gain and bias for achieving target max_rates and intercepts.

        This is required by Nengo to set up the neural ensemble.
        """
        # Simplified gain/bias calculation
        # In practice, would need more sophisticated calibration
        max_rates = np.array(max_rates, dtype=float)
        intercepts = np.array(intercepts, dtype=float)

        gain = max_rates / (1.0 - intercepts)
        bias = -intercepts * gain

        return gain, bias

    def max_rates_intercepts(
        self, gain: np.ndarray, bias: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute max_rates and intercepts from gain and bias."""
        intercepts = -bias / gain
        max_rates = gain * (1.0 - intercepts)
        return max_rates, intercepts

    def step(
        self, dt: float, J: np.ndarray, output: np.ndarray,
        voltage: np.ndarray, recovery: np.ndarray
    ) -> None:
        """Advance the neuron state by one timestep.

        Args:
            dt: Timestep size in seconds.
            J: Input current to each neuron.
            output: Output array for spike counts.
            voltage: Membrane voltage state (modified in-place).
            recovery: Recovery variable state (modified in-place).
        """
        # Convert dt from seconds to ms for Izhikevich equations
        dt_ms = dt * 1000.0

        # Izhikevich model equations
        # v' = 0.04v² + 5v + 140 - u + I
        # u' = a(bv - u)
        dv = 0.04 * voltage * voltage + 5 * voltage + 140 - recovery + J
        du = self.a * (self.b * voltage - recovery)

        voltage += dv * dt_ms
        recovery += du * dt_ms

        # Spike detection and reset
        spiked = voltage >= 30.0
        output[:] = spiked / dt  # Spike rate

        # Reset spiked neurons
        voltage[spiked] = self.c
        recovery[spiked] += self.d


def create_izhikevich_population(
    n_neurons: int,
    dimensions: int,
    neuron_type: IzhikevichNeuronType = IzhikevichNeuronType.REGULAR_SPIKING,
    excitatory_ratio: float = 0.8,
    label: str = None,
) -> nengo.Ensemble:
    """Create a population of Izhikevich neurons with mixed types.

    In biological neural networks, approximately 80% of neurons are
    excitatory (glutamatergic) and 20% are inhibitory (GABAergic).

    Args:
        n_neurons: Total number of neurons in the population.
        dimensions: Dimensionality of the represented signal.
        neuron_type: Primary neuron type for excitatory neurons.
        excitatory_ratio: Fraction of excitatory neurons (default 0.8).
        label: Optional label for the ensemble.

    Returns:
        A Nengo ensemble with Izhikevich neurons.
    """
    # Note: In a full implementation, we would mix neuron types
    # For now, use a single type
    return nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=dimensions,
        neuron_type=IzhikevichNeuron(neuron_type=neuron_type),
        label=label,
    )
