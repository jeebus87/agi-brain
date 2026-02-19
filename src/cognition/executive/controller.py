"""
Executive Controller implementation.

The executive controller implements the cortex-basal ganglia-thalamus loop
for action selection, goal maintenance, and cognitive control.

Reference:
    Frank, M. J. (2011). Computational models of motivated action selection
    in corticostriatal circuits. Current Opinion in Neurobiology.
"""

from typing import Optional, List, Callable, Dict, Any
import nengo
import nengo_spa as spa


class ActionRule:
    """A production rule for action selection.

    Production rules have the form:
        IF condition THEN action

    In the SPA framework, this is implemented as:
        IF dot(state, PATTERN) > threshold THEN effect
    """

    def __init__(
        self,
        name: str,
        condition: str,
        action: str,
        utility: float = 1.0,
    ):
        """Initialize an action rule.

        Args:
            name: Human-readable name for the rule.
            condition: Semantic pointer pattern to match.
            action: Semantic pointer to produce.
            utility: Expected utility/reward of this action.
        """
        self.name = name
        self.condition = condition
        self.action = action
        self.utility = utility


class ExecutiveController(spa.Network):
    """Executive controller using basal ganglia for action selection.

    The executive controller:
    1. Maintains goals in prefrontal cortex
    2. Selects actions via basal ganglia
    3. Routes information through thalamus
    4. Learns from reward signals

    This implements a simplified version of the cortex-BG-thalamus loop.
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        n_actions: int = 5,
        label: str = "executive",
        **kwargs
    ):
        """Initialize the executive controller.

        Args:
            vocab: Semantic pointer vocabulary.
            n_actions: Maximum number of concurrent action rules.
            label: Network label.
        """
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.n_actions = n_actions
        self.dimensions = vocab.dimensions

        with self:
            # Prefrontal Cortex - Goal maintenance
            self.goal_state = spa.State(
                vocab,
                feedback=1.0,
                feedback_synapse=0.1,
                label="goal",
            )

            # Current state representation
            self.current_state = spa.State(vocab, label="current")

            # Action output
            self.action_state = spa.State(vocab, label="action")

            # Basal Ganglia for action selection
            # This will be configured with rules later
            self._rules: List[ActionRule] = []
            self._action_selection = None

            # Thalamus for gating
            self.thalamus = spa.Thalamus(
                action_count=n_actions,
                label="thalamus",
            )

            # Dopamine signal (reward prediction error)
            self.reward_signal = spa.Scalar(label="reward")

            # Utility estimates for actions
            self._utilities = [1.0] * n_actions

    def add_rule(self, rule: ActionRule) -> None:
        """Add a production rule to the controller.

        Args:
            rule: The action rule to add.
        """
        self._rules.append(rule)

    def compile_rules(self) -> None:
        """Compile the added rules into basal ganglia actions.

        This must be called after all rules are added and before
        the network is simulated.
        """
        if not self._rules:
            raise ValueError("No rules added to executive controller")

        with self:
            # Create action selection with the defined rules
            actions = spa.Actions()
            for i, rule in enumerate(self._rules):
                # Create the action specification
                action_spec = f"{rule.action}"
                condition = f"dot(current, {rule.condition})"
                actions.add(utility=f"{rule.utility} * {condition}", **{f"action_{i}": action_spec})

            # Rebuild basal ganglia with new actions
            self._action_selection = spa.ActionSelection(actions)
            self.basal_ganglia = spa.BasalGanglia(
                self._action_selection,
                label="basal_ganglia",
            )

    def set_goal(self, goal_pattern: str) -> None:
        """Set the current goal.

        Args:
            goal_pattern: Semantic pointer name for the goal.
        """
        # This would be implemented with a Node that sets the goal
        pass

    def update_utility(self, rule_index: int, delta: float) -> None:
        """Update the utility estimate for a rule based on reward.

        Args:
            rule_index: Index of the rule to update.
            delta: Change in utility (positive = reward, negative = punishment).
        """
        if 0 <= rule_index < len(self._utilities):
            self._utilities[rule_index] += delta
            # Keep utilities positive
            self._utilities[rule_index] = max(0.1, self._utilities[rule_index])


class GoalStack(spa.Network):
    """A goal stack for hierarchical planning.

    The goal stack allows:
    - Pushing subgoals
    - Popping completed goals
    - Maintaining goal hierarchy
    """

    def __init__(
        self,
        vocab: spa.Vocabulary,
        max_depth: int = 5,
        label: str = "goal_stack",
        **kwargs
    ):
        """Initialize the goal stack.

        Args:
            vocab: Semantic pointer vocabulary.
            max_depth: Maximum stack depth.
            label: Network label.
        """
        super().__init__(label=label, **kwargs)

        self.vocab = vocab
        self.max_depth = max_depth

        with self:
            # Stack levels
            self.levels: List[spa.State] = []
            for i in range(max_depth):
                level = spa.State(
                    vocab,
                    feedback=1.0,
                    feedback_synapse=0.1,
                    label=f"level_{i}",
                )
                self.levels.append(level)

            # Stack pointer (which level is active)
            self.pointer = spa.Scalar(label="pointer")

            # Top of stack output
            self.top = spa.State(vocab, label="top")

            # Connect top level to output
            # (In a full implementation, this would be pointer-dependent)
            nengo.Connection(self.levels[0].output, self.top.input)
