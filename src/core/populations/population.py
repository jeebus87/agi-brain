"""
Neural population implementation.

A neural population is a group of neurons that collectively encode
information using the Neural Engineering Framework (NEF) principles.
"""

from typing import Optional, Union
import numpy as np
import nengo

from ..neurons import IzhikevichNeuron, IzhikevichNeuronType


class NeuralPopulation:
    """A biologically-inspired neural population.

    This class wraps Nengo ensembles with additional features for
    creating realistic cortical populations.

    Attributes:
        n_neurons: Number of neurons in the population.
        dimensions: Dimensionality of the encoded signal.
        ensemble: The underlying Nengo ensemble.
    """

    def __init__(
        self,
        n_neurons: int,
        dimensions: int,
        neuron_type: str = "izhikevich",
        excitatory_ratio: float = 0.8,
        max_rates: tuple = (100, 200),
        intercepts: tuple = (-1, 1),
        label: Optional[str] = None,
    ):
        """Initialize a neural population.

        Args:
            n_neurons: Number of neurons.
            dimensions: Dimensionality of the encoded signal.
            neuron_type: Type of neuron model ("izhikevich" or "lif").
            excitatory_ratio: Fraction of excitatory neurons.
            max_rates: Range of maximum firing rates (Hz).
            intercepts: Range of x-intercepts for tuning curves.
            label: Optional label for identification.
        """
        self.n_neurons = n_neurons
        self.dimensions = dimensions
        self.excitatory_ratio = excitatory_ratio
        self.label = label

        # Create neuron type
        if neuron_type == "izhikevich":
            self._neuron_type = IzhikevichNeuron(
                neuron_type=IzhikevichNeuronType.REGULAR_SPIKING
            )
        else:
            self._neuron_type = nengo.LIF()

        # Store max_rates and intercepts for ensemble creation
        self._max_rates = nengo.dists.Uniform(max_rates[0], max_rates[1])
        self._intercepts = nengo.dists.Uniform(intercepts[0], intercepts[1])

        # The ensemble will be created when added to a network
        self.ensemble: Optional[nengo.Ensemble] = None

    def build(self, network: Optional[nengo.Network] = None) -> nengo.Ensemble:
        """Build the Nengo ensemble for this population.

        Args:
            network: Optional Nengo network context. If None, must be called
                    within a 'with nengo.Network()' context.

        Returns:
            The created Nengo ensemble.
        """
        if network is not None:
            ctx = network
        else:
            ctx = nengo.Network()

        with ctx:
            self.ensemble = nengo.Ensemble(
                n_neurons=self.n_neurons,
                dimensions=self.dimensions,
                neuron_type=self._neuron_type,
                max_rates=self._max_rates,
                intercepts=self._intercepts,
                label=self.label,
            )

        return self.ensemble

    @property
    def n_excitatory(self) -> int:
        """Number of excitatory neurons."""
        return int(self.n_neurons * self.excitatory_ratio)

    @property
    def n_inhibitory(self) -> int:
        """Number of inhibitory neurons."""
        return self.n_neurons - self.n_excitatory


class CorticalLayer:
    """A cortical layer with excitatory and inhibitory populations.

    Cortical layers have characteristic connectivity patterns:
    - Excitatory neurons project broadly
    - Inhibitory neurons provide local feedback
    """

    def __init__(
        self,
        n_excitatory: int,
        n_inhibitory: int,
        dimensions: int,
        label: str = None,
    ):
        """Initialize a cortical layer.

        Args:
            n_excitatory: Number of excitatory neurons.
            n_inhibitory: Number of inhibitory neurons.
            dimensions: Dimensionality of the encoded signal.
            label: Optional label.
        """
        self.excitatory = NeuralPopulation(
            n_neurons=n_excitatory,
            dimensions=dimensions,
            neuron_type="izhikevich",
            label=f"{label}_exc" if label else None,
        )
        self.inhibitory = NeuralPopulation(
            n_neurons=n_inhibitory,
            dimensions=dimensions,
            neuron_type="izhikevich",
            label=f"{label}_inh" if label else None,
        )
        self.dimensions = dimensions
        self.label = label

    def build(self, network: Optional[nengo.Network] = None) -> None:
        """Build both populations and their connections."""
        ctx = network if network is not None else nengo.Network()

        with ctx:
            self.excitatory.build()
            self.inhibitory.build()

            # Excitatory to inhibitory
            nengo.Connection(
                self.excitatory.ensemble,
                self.inhibitory.ensemble,
                transform=1.0,
            )

            # Inhibitory feedback (negative)
            nengo.Connection(
                self.inhibitory.ensemble,
                self.excitatory.ensemble,
                transform=-1.0,
            )
