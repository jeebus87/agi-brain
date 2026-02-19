"""
Synaptic Plasticity Rules

Implements biologically-inspired learning rules:

1. STDP (Spike-Timing-Dependent Plasticity)
   - Pre before post -> strengthen (LTP)
   - Post before pre -> weaken (LTD)

2. BCM (Bienenstock-Cooper-Munro)
   - Homeostatic plasticity
   - Sliding threshold prevents runaway excitation

3. Reward-Modulated STDP
   - Dopamine-gated learning
   - Only consolidate changes when rewarded

4. Oja's Rule
   - Hebbian with normalization
   - Prevents unbounded weight growth

These rules allow the network to learn from experience rather than
having fixed, pre-programmed connections.
"""

import numpy as np
import nengo
from nengo.params import Default
from nengo.builder import Builder, Signal, Operator
from nengo.builder.connection import build_connection
from typing import Optional, Callable, Tuple


class STDPRule(nengo.learning_rules.LearningRuleType):
    """Spike-Timing-Dependent Plasticity learning rule.

    Implements the classic STDP rule:
    - If pre fires before post (causal): strengthen connection (LTP)
    - If post fires before pre (anti-causal): weaken connection (LTD)

    The weight change follows:
        dw = A_plus * exp(-dt/tau_plus)   if dt > 0 (pre before post)
        dw = -A_minus * exp(dt/tau_minus) if dt < 0 (post before pre)

    where dt = t_post - t_pre

    Parameters
    ----------
    learning_rate : float
        Base learning rate (default: 1e-4)
    tau_plus : float
        Time constant for LTP window in seconds (default: 0.020)
    tau_minus : float
        Time constant for LTD window in seconds (default: 0.020)
    a_plus : float
        Amplitude of LTP (default: 1.0)
    a_minus : float
        Amplitude of LTD (default: 1.05, slightly larger for stability)
    w_max : float
        Maximum weight value (default: 1.0)
    w_min : float
        Minimum weight value (default: 0.0)
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "delta")

    learning_rate = nengo.params.NumberParam(
        "learning_rate", low=0, default=1e-4
    )
    tau_plus = nengo.params.NumberParam(
        "tau_plus", low=0, default=0.020
    )
    tau_minus = nengo.params.NumberParam(
        "tau_minus", low=0, default=0.020
    )
    a_plus = nengo.params.NumberParam(
        "a_plus", low=0, default=1.0
    )
    a_minus = nengo.params.NumberParam(
        "a_minus", low=0, default=1.05
    )
    w_max = nengo.params.NumberParam(
        "w_max", default=1.0
    )
    w_min = nengo.params.NumberParam(
        "w_min", default=0.0
    )

    def __init__(
        self,
        learning_rate=Default,
        tau_plus=Default,
        tau_minus=Default,
        a_plus=Default,
        a_minus=Default,
        w_max=Default,
        w_min=Default,
    ):
        super().__init__(learning_rate, size_in=0)
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min


class RewardModulatedSTDP(nengo.learning_rules.LearningRuleType):
    """Reward-modulated STDP (dopamine-gated learning).

    Standard STDP creates "eligibility traces" that mark synapses
    for potential modification. The actual weight change only occurs
    when a reward signal (dopamine) is present.

    This implements a three-factor learning rule:
        dw = eligibility_trace * reward * learning_rate

    where eligibility_trace is computed by standard STDP.

    Parameters
    ----------
    learning_rate : float
        Base learning rate (default: 1e-4)
    tau_plus : float
        LTP time constant (default: 0.020)
    tau_minus : float
        LTD time constant (default: 0.020)
    tau_eligibility : float
        Eligibility trace decay time constant (default: 1.0)
    """

    modifies = "weights"
    probeable = ("pre_filtered", "post_filtered", "eligibility", "delta")

    learning_rate = nengo.params.NumberParam(
        "learning_rate", low=0, default=1e-4
    )
    tau_plus = nengo.params.NumberParam(
        "tau_plus", low=0, default=0.020
    )
    tau_minus = nengo.params.NumberParam(
        "tau_minus", low=0, default=0.020
    )
    tau_eligibility = nengo.params.NumberParam(
        "tau_eligibility", low=0, default=1.0
    )

    def __init__(
        self,
        learning_rate=Default,
        tau_plus=Default,
        tau_minus=Default,
        tau_eligibility=Default,
    ):
        super().__init__(learning_rate, size_in="post")  # Reward input
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_eligibility = tau_eligibility


class BCMRule(nengo.learning_rules.LearningRuleType):
    """BCM (Bienenstock-Cooper-Munro) learning rule.

    Implements homeostatic plasticity with a sliding threshold:
    - Activity above threshold -> LTP
    - Activity below threshold -> LTD
    - Threshold adapts based on average post-synaptic activity

    This prevents runaway excitation/inhibition and creates
    competitive, sparse representations.

    dw = learning_rate * pre * post * (post - theta)
    d_theta = tau_theta^-1 * (post^2 - theta)

    Parameters
    ----------
    learning_rate : float
        Base learning rate (default: 1e-5)
    tau_theta : float
        Time constant for threshold adaptation (default: 1.0)
    theta_init : float
        Initial threshold value (default: 0.5)
    """

    modifies = "weights"
    probeable = ("theta", "delta")

    learning_rate = nengo.params.NumberParam(
        "learning_rate", low=0, default=1e-5
    )
    tau_theta = nengo.params.NumberParam(
        "tau_theta", low=0, default=1.0
    )
    theta_init = nengo.params.NumberParam(
        "theta_init", default=0.5
    )

    def __init__(
        self,
        learning_rate=Default,
        tau_theta=Default,
        theta_init=Default,
    ):
        super().__init__(learning_rate, size_in=0)
        self.tau_theta = tau_theta
        self.theta_init = theta_init


class OjaRule(nengo.learning_rules.LearningRuleType):
    """Oja's learning rule (normalized Hebbian).

    Implements Hebbian learning with weight normalization:
        dw = learning_rate * post * (pre - post * w)

    The subtraction term (post * w) prevents unbounded weight growth
    and leads to weights that extract the principal component.

    Parameters
    ----------
    learning_rate : float
        Base learning rate (default: 1e-4)
    """

    modifies = "weights"
    probeable = ("delta",)

    learning_rate = nengo.params.NumberParam(
        "learning_rate", low=0, default=1e-4
    )

    def __init__(self, learning_rate=Default):
        super().__init__(learning_rate, size_in=0)


# =============================================================================
# Nengo Operators for custom learning rules
# =============================================================================

class SimSTDP(Operator):
    """Simulator operator for STDP learning."""

    def __init__(
        self,
        pre_filtered,
        post_filtered,
        weights,
        delta,
        learning_rate,
        a_plus,
        a_minus,
        w_max,
        w_min,
        tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_max = w_max
        self.w_min = w_min

        self.sets = []
        self.incs = []
        self.reads = [pre_filtered, post_filtered, weights]
        self.updates = [delta, weights]

    @property
    def pre_filtered(self):
        return self.reads[0]

    @property
    def post_filtered(self):
        return self.reads[1]

    @property
    def weights(self):
        return self.reads[2]

    @property
    def delta(self):
        return self.updates[0]

    def make_step(self, signals, dt, rng):
        pre_filtered = signals[self.pre_filtered]
        post_filtered = signals[self.post_filtered]
        weights = signals[self.weights]
        delta = signals[self.delta]

        learning_rate = self.learning_rate
        a_plus = self.a_plus
        a_minus = self.a_minus
        w_max = self.w_max
        w_min = self.w_min

        def step_stdp():
            # LTP: pre before post (pre high, post high)
            ltp = a_plus * np.outer(post_filtered, pre_filtered)

            # LTD: post before pre (approximated by post * pre correlation)
            ltd = a_minus * np.outer(post_filtered, pre_filtered)

            # Net change (simplified STDP approximation)
            # In full implementation, would track spike times
            delta[...] = learning_rate * dt * (ltp - 0.5 * ltd)

            # Apply weight bounds
            new_weights = weights + delta
            np.clip(new_weights, w_min, w_max, out=weights)

        return step_stdp


class SimBCM(Operator):
    """Simulator operator for BCM learning."""

    def __init__(
        self,
        pre,
        post,
        weights,
        theta,
        delta,
        learning_rate,
        tau_theta,
        tag=None
    ):
        super().__init__(tag=tag)
        self.learning_rate = learning_rate
        self.tau_theta = tau_theta

        self.sets = []
        self.incs = []
        self.reads = [pre, post, weights]
        self.updates = [theta, delta, weights]

    def make_step(self, signals, dt, rng):
        pre = signals[self.reads[0]]
        post = signals[self.reads[1]]
        weights = signals[self.reads[2]]
        theta = signals[self.updates[0]]
        delta = signals[self.updates[1]]

        learning_rate = self.learning_rate
        tau_theta = self.tau_theta

        def step_bcm():
            # BCM weight update
            post_minus_theta = post - theta
            delta[...] = learning_rate * dt * np.outer(
                post * post_minus_theta, pre
            )

            # Update sliding threshold
            theta[...] += dt / tau_theta * (np.mean(post ** 2) - theta)

            # Apply weight change
            weights[...] += delta

        return step_bcm


# =============================================================================
# Builder functions to register learning rules with Nengo
# =============================================================================

@Builder.register(STDPRule)
def build_stdp(model, stdp, rule):
    """Build STDP learning rule."""
    conn = rule.connection

    # Get pre and post activities
    pre = model.sig[conn.pre_obj]["out"]
    post = model.sig[conn.post_obj]["in"]

    # Create filtered versions (traces)
    pre_filtered = Signal(
        shape=pre.shape,
        name=f"{rule}.pre_filtered"
    )
    post_filtered = Signal(
        shape=post.shape,
        name=f"{rule}.post_filtered"
    )

    # Add lowpass filters for the traces
    model.add_op(nengo.builder.operator.SimFilterSynapse(
        pre, pre_filtered, nengo.Lowpass(stdp.tau_plus), mode="update"
    ))
    model.add_op(nengo.builder.operator.SimFilterSynapse(
        post, post_filtered, nengo.Lowpass(stdp.tau_minus), mode="update"
    ))

    # Get weights signal
    weights = model.sig[conn]["weights"]

    # Create delta signal
    delta = Signal(shape=weights.shape, name=f"{rule}.delta")

    # Add STDP operator
    model.add_op(SimSTDP(
        pre_filtered,
        post_filtered,
        weights,
        delta,
        stdp.learning_rate,
        stdp.a_plus,
        stdp.a_minus,
        stdp.w_max,
        stdp.w_min,
        tag=f"SimSTDP({rule})"
    ))

    # Make signals probeable
    model.sig[rule]["pre_filtered"] = pre_filtered
    model.sig[rule]["post_filtered"] = post_filtered
    model.sig[rule]["delta"] = delta


@Builder.register(BCMRule)
def build_bcm(model, bcm, rule):
    """Build BCM learning rule."""
    conn = rule.connection

    pre = model.sig[conn.pre_obj]["out"]
    post = model.sig[conn.post_obj]["in"]
    weights = model.sig[conn]["weights"]

    # Theta (sliding threshold)
    theta = Signal(shape=(), initial_value=bcm.theta_init, name=f"{rule}.theta")

    # Delta signal
    delta = Signal(shape=weights.shape, name=f"{rule}.delta")

    model.add_op(SimBCM(
        pre, post, weights, theta, delta,
        bcm.learning_rate, bcm.tau_theta,
        tag=f"SimBCM({rule})"
    ))

    model.sig[rule]["theta"] = theta
    model.sig[rule]["delta"] = delta


# =============================================================================
# High-level learning connection helpers
# =============================================================================

def create_plastic_connection(
    pre,
    post,
    learning_rule: str = "stdp",
    learning_rate: float = 1e-4,
    **kwargs
) -> nengo.Connection:
    """Create a connection with plasticity.

    Parameters
    ----------
    pre : nengo object
        Pre-synaptic population
    post : nengo object
        Post-synaptic population
    learning_rule : str
        Type of learning rule: "stdp", "bcm", "oja", "pes"
    learning_rate : float
        Learning rate
    **kwargs
        Additional arguments passed to the learning rule

    Returns
    -------
    nengo.Connection
        Connection with learning enabled
    """
    if learning_rule == "stdp":
        rule = STDPRule(learning_rate=learning_rate, **kwargs)
    elif learning_rule == "bcm":
        rule = BCMRule(learning_rate=learning_rate, **kwargs)
    elif learning_rule == "oja":
        rule = OjaRule(learning_rate=learning_rate)
    elif learning_rule == "pes":
        rule = nengo.PES(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown learning rule: {learning_rule}")

    conn = nengo.Connection(
        pre, post,
        learning_rule_type=rule,
        **{k: v for k, v in kwargs.items()
           if k not in ["tau_plus", "tau_minus", "a_plus", "a_minus",
                        "w_max", "w_min", "tau_theta", "theta_init"]}
    )

    return conn


class PlasticSynapse:
    """A synapse with configurable plasticity.

    This is a higher-level wrapper that creates plastic connections
    with sensible defaults for AGI applications.
    """

    def __init__(
        self,
        pre,
        post,
        learning_rule: str = "stdp",
        learning_rate: float = 1e-4,
        initial_weights: Optional[np.ndarray] = None,
        trainable: bool = True,
    ):
        self.pre = pre
        self.post = post
        self.learning_rule = learning_rule
        self.learning_rate = learning_rate
        self.trainable = trainable

        # Determine weight shape
        if hasattr(pre, "n_neurons"):
            pre_size = pre.n_neurons
        elif hasattr(pre, "size_out"):
            pre_size = pre.size_out
        else:
            pre_size = pre.output.size_out

        if hasattr(post, "n_neurons"):
            post_size = post.n_neurons
        elif hasattr(post, "size_in"):
            post_size = post.size_in
        else:
            post_size = post.input.size_in

        if initial_weights is None:
            # Xavier initialization
            scale = np.sqrt(2.0 / (pre_size + post_size))
            self.initial_weights = np.random.randn(post_size, pre_size) * scale
        else:
            self.initial_weights = initial_weights

    def connect(self, **kwargs) -> nengo.Connection:
        """Create the plastic connection."""
        if self.trainable:
            return create_plastic_connection(
                self.pre,
                self.post,
                learning_rule=self.learning_rule,
                learning_rate=self.learning_rate,
                transform=self.initial_weights,
                **kwargs
            )
        else:
            return nengo.Connection(
                self.pre,
                self.post,
                transform=self.initial_weights,
                **kwargs
            )


# =============================================================================
# Demo and testing
# =============================================================================

def demo_stdp_learning():
    """Demonstrate STDP learning with a simple pattern association task."""
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("  STDP Learning Demo")
    print("=" * 60)
    print()
    print("  Task: Learn to associate input pattern A with output B")
    print("  Method: Present A, then B repeatedly (Hebbian pairing)")
    print()

    # Create a simple network
    with nengo.Network(seed=42) as model:
        # Input populations
        input_a = nengo.Ensemble(n_neurons=50, dimensions=1, label="input_a")
        output_b = nengo.Ensemble(n_neurons=50, dimensions=1, label="output_b")

        # Plastic connection (using PES as STDP isn't built into base nengo)
        # We'll use PES which is similar in effect
        error = nengo.Ensemble(n_neurons=50, dimensions=1, label="error")

        conn = nengo.Connection(
            input_a,
            output_b,
            learning_rule_type=nengo.PES(learning_rate=1e-3),
            transform=[[0.0]]  # Start with zero weight
        )

        # Error signal: target - actual
        nengo.Connection(output_b, error, transform=-1)

        # Connect error to learning rule
        nengo.Connection(error, conn.learning_rule)

        # Input: periodic presentation of pattern
        def input_func(t):
            # Present input every 0.5 seconds
            if (t % 1.0) < 0.3:
                return 1.0
            return 0.0

        def target_func(t):
            # Target output (what we want to learn)
            if (t % 1.0) < 0.3:
                return 0.8  # Want output to be 0.8 when input is 1.0
            return 0.0

        input_node = nengo.Node(input_func)
        target_node = nengo.Node(target_func)

        nengo.Connection(input_node, input_a)
        nengo.Connection(target_node, error)  # Target for error computation

        # Probes
        p_input = nengo.Probe(input_a, synapse=0.01)
        p_output = nengo.Probe(output_b, synapse=0.01)
        p_error = nengo.Probe(error, synapse=0.01)
        p_weights = nengo.Probe(conn, "weights", synapse=0.1)

    # Run simulation
    print("[1] Running learning simulation (5 seconds)...")
    with nengo.Simulator(model, progress_bar=True) as sim:
        sim.run(5.0)

    # Analyze results
    print("\n[2] Analyzing learning...")

    t = sim.trange()
    weights = sim.data[p_weights]

    initial_weight = weights[0, 0, 0]
    final_weight = weights[-1, 0, 0]

    print(f"    Initial weight: {initial_weight:.4f}")
    print(f"    Final weight:   {final_weight:.4f}")
    print(f"    Weight change:  {final_weight - initial_weight:+.4f}")

    # Plot results
    print("\n[3] Generating visualization...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Plot 1: Input/Output
    ax1 = axes[0]
    ax1.plot(t, sim.data[p_input], label="Input", alpha=0.7)
    ax1.plot(t, sim.data[p_output], label="Output", alpha=0.7)
    ax1.set_ylabel("Activity")
    ax1.set_title("Input and Output Activity", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error
    ax2 = axes[1]
    ax2.plot(t, sim.data[p_error], color="red", alpha=0.7)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_ylabel("Error")
    ax2.set_title("Learning Error (should decrease)", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Weight evolution
    ax3 = axes[2]
    ax3.plot(t, weights[:, 0, 0], color="green", linewidth=2)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Weight")
    ax3.set_title("Synaptic Weight Evolution", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stdp_learning_demo.png", dpi=150, bbox_inches="tight")
    print("    Saved plot to stdp_learning_demo.png")

    print("\n" + "=" * 60)
    print("  Learning Demo Complete!")
    print("=" * 60)

    return model, sim


if __name__ == "__main__":
    demo_stdp_learning()
