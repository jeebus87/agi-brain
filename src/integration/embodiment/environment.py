"""
Environment interfaces for embodied simulation

Provides abstract base class and concrete implementations:
- GridWorld: Simple 2D navigation with goals and obstacles
- Future: MuJoCo, Unity, AI2-THOR wrappers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Any, Optional
import numpy as np


@dataclass
class Observation:
    """Sensory observation from environment"""
    vision: np.ndarray          # Visual field (grid or image)
    position: np.ndarray        # Agent position (x, y, z) or (x, y)
    velocity: np.ndarray        # Agent velocity
    proprioception: np.ndarray  # Internal body state
    reward: float = 0.0         # Reward signal
    done: bool = False          # Episode complete
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    """Motor action to execute"""
    movement: np.ndarray    # Movement vector
    discrete_action: int = -1  # For discrete action spaces


class Environment(ABC):
    """Abstract base class for simulation environments"""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state"""
        pass

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Execute action and return new observation"""
        pass

    @abstractmethod
    def get_observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        """Return shapes of observation components"""
        pass

    @abstractmethod
    def get_action_shape(self) -> Tuple[int, ...]:
        """Return shape of action space"""
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of discrete actions (0 if continuous)"""
        pass


class GridWorld(Environment):
    """
    Simple 2D grid navigation environment

    The agent navigates a grid to reach goals while avoiding obstacles.
    Perfect for testing embodied cognition without external dependencies.

    Grid cell values:
        0 = empty
        1 = wall/obstacle
        2 = goal (positive reward)
        3 = hazard (negative reward)
        4 = agent position
    """

    # Actions: 0=up, 1=right, 2=down, 3=left, 4=stay
    ACTION_DELTAS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
        4: (0, 0),   # stay
    }
    ACTION_NAMES = ['up', 'right', 'down', 'left', 'stay']

    def __init__(
        self,
        size: int = 10,
        n_goals: int = 1,
        n_obstacles: int = 5,
        n_hazards: int = 2,
        vision_range: int = 3,
        max_steps: int = 100,
        seed: Optional[int] = None
    ):
        """
        Args:
            size: Grid dimension (size x size)
            n_goals: Number of goal locations
            n_obstacles: Number of wall obstacles
            n_hazards: Number of hazard locations
            vision_range: How far agent can see (creates 2*range+1 vision field)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        self.size = size
        self.n_goals = n_goals
        self.n_obstacles = n_obstacles
        self.n_hazards = n_hazards
        self.vision_range = vision_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        # State
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.agent_pos = np.array([0, 0])
        self.goal_positions: List[Tuple[int, int]] = []
        self.steps = 0
        self.total_reward = 0.0

        # Initialize
        self._generate_grid()

    def _generate_grid(self):
        """Generate random grid layout"""
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)

        # Place agent at random position
        self.agent_pos = self.rng.integers(0, self.size, size=2)

        # Place goals
        self.goal_positions = []
        for _ in range(self.n_goals):
            pos = self._find_empty_cell()
            if pos:
                self.grid[pos] = 2
                self.goal_positions.append(pos)

        # Place obstacles
        for _ in range(self.n_obstacles):
            pos = self._find_empty_cell()
            if pos:
                self.grid[pos] = 1

        # Place hazards
        for _ in range(self.n_hazards):
            pos = self._find_empty_cell()
            if pos:
                self.grid[pos] = 3

    def _find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """Find random empty cell not occupied by agent"""
        for _ in range(100):  # Max attempts
            pos = tuple(self.rng.integers(0, self.size, size=2))
            if self.grid[pos] == 0 and not np.array_equal(pos, self.agent_pos):
                return pos
        return None

    def _get_vision_field(self) -> np.ndarray:
        """Get local vision around agent"""
        r = self.vision_range
        vision_size = 2 * r + 1
        vision = np.ones((vision_size, vision_size), dtype=np.float32) * -1  # -1 = out of bounds

        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                gi = self.agent_pos[0] + di
                gj = self.agent_pos[1] + dj

                if 0 <= gi < self.size and 0 <= gj < self.size:
                    vi = di + r
                    vj = dj + r
                    vision[vi, vj] = self.grid[gi, gj]

        return vision

    def _compute_proprioception(self) -> np.ndarray:
        """Compute internal state representation"""
        # Normalized position
        norm_pos = self.agent_pos / self.size

        # Distance to nearest goal
        if self.goal_positions:
            distances = [np.linalg.norm(self.agent_pos - np.array(g))
                        for g in self.goal_positions]
            min_dist = min(distances) / (self.size * np.sqrt(2))
        else:
            min_dist = 1.0

        # Steps remaining (normalized)
        steps_remaining = (self.max_steps - self.steps) / self.max_steps

        return np.array([norm_pos[0], norm_pos[1], min_dist, steps_remaining], dtype=np.float32)

    def reset(self) -> Observation:
        """Reset to new random configuration"""
        self._generate_grid()
        self.steps = 0
        self.total_reward = 0.0

        return Observation(
            vision=self._get_vision_field(),
            position=self.agent_pos.astype(np.float32) / self.size,
            velocity=np.zeros(2, dtype=np.float32),
            proprioception=self._compute_proprioception(),
            reward=0.0,
            done=False,
            info={'grid': self.grid.copy(), 'step': 0}
        )

    def step(self, action: Action) -> Observation:
        """Execute action and return observation"""
        self.steps += 1

        # Get action index
        if action.discrete_action >= 0:
            act_idx = action.discrete_action
        else:
            # Convert continuous movement to discrete
            if np.linalg.norm(action.movement) < 0.1:
                act_idx = 4  # stay
            else:
                # Find closest cardinal direction
                angle = np.arctan2(action.movement[1], action.movement[0])
                act_idx = int((angle + np.pi) / (np.pi / 2)) % 4

        act_idx = np.clip(act_idx, 0, 4)

        # Compute new position
        delta = self.ACTION_DELTAS[act_idx]
        new_pos = self.agent_pos + np.array(delta)

        # Check bounds and obstacles
        reward = -0.01  # Small step penalty to encourage efficiency
        valid_move = True

        if not (0 <= new_pos[0] < self.size and 0 <= new_pos[1] < self.size):
            valid_move = False
            reward -= 0.1  # Penalty for hitting boundary
        elif self.grid[tuple(new_pos)] == 1:  # Wall
            valid_move = False
            reward -= 0.1  # Penalty for hitting wall

        if valid_move:
            old_pos = self.agent_pos.copy()
            self.agent_pos = new_pos
            velocity = (new_pos - old_pos).astype(np.float32)

            # Check what we stepped on
            cell = self.grid[tuple(self.agent_pos)]
            if cell == 2:  # Goal
                reward = 1.0
                self.grid[tuple(self.agent_pos)] = 0  # Remove goal
                self.goal_positions = [g for g in self.goal_positions
                                       if not np.array_equal(g, self.agent_pos)]
            elif cell == 3:  # Hazard
                reward = -0.5
        else:
            velocity = np.zeros(2, dtype=np.float32)

        self.total_reward += reward

        # Check if done
        done = (len(self.goal_positions) == 0) or (self.steps >= self.max_steps)

        return Observation(
            vision=self._get_vision_field(),
            position=self.agent_pos.astype(np.float32) / self.size,
            velocity=velocity,
            proprioception=self._compute_proprioception(),
            reward=reward,
            done=done,
            info={
                'grid': self.grid.copy(),
                'step': self.steps,
                'action': self.ACTION_NAMES[act_idx],
                'total_reward': self.total_reward,
                'goals_remaining': len(self.goal_positions)
            }
        )

    def get_observation_shape(self) -> Dict[str, Tuple[int, ...]]:
        vision_size = 2 * self.vision_range + 1
        return {
            'vision': (vision_size, vision_size),
            'position': (2,),
            'velocity': (2,),
            'proprioception': (4,)
        }

    def get_action_shape(self) -> Tuple[int, ...]:
        return (2,)  # Continuous movement vector

    @property
    def n_actions(self) -> int:
        return 5  # up, right, down, left, stay

    def render_ascii(self) -> str:
        """Render grid as ASCII art"""
        symbols = {0: '.', 1: '#', 2: 'G', 3: 'X', 4: 'A'}
        lines = []

        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if np.array_equal(self.agent_pos, [i, j]):
                    row += 'A '
                else:
                    row += symbols[self.grid[i, j]] + ' '
            lines.append(row)

        return '\n'.join(lines)


class MultiRoomGridWorld(GridWorld):
    """
    Extended grid world with multiple connected rooms

    Tests spatial reasoning and navigation planning
    """

    def __init__(
        self,
        room_size: int = 5,
        n_rooms_x: int = 2,
        n_rooms_y: int = 2,
        door_size: int = 1,
        **kwargs
    ):
        self.room_size = room_size
        self.n_rooms_x = n_rooms_x
        self.n_rooms_y = n_rooms_y
        self.door_size = door_size

        # Calculate total size including walls between rooms
        total_size = room_size * n_rooms_x + (n_rooms_x - 1)

        super().__init__(size=total_size, **kwargs)

    def _generate_grid(self):
        """Generate room layout with connecting doors"""
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)

        # Add walls between rooms
        for rx in range(self.n_rooms_x - 1):
            wall_x = (rx + 1) * self.room_size + rx
            self.grid[wall_x, :] = 1

            # Add door in each wall
            for ry in range(self.n_rooms_y):
                door_start = ry * (self.room_size + 1) + self.room_size // 2
                for d in range(self.door_size):
                    if door_start + d < self.size:
                        self.grid[wall_x, door_start + d] = 0

        for ry in range(self.n_rooms_y - 1):
            wall_y = (ry + 1) * self.room_size + ry
            self.grid[:, wall_y] = 1

            # Add door
            for rx in range(self.n_rooms_x):
                door_start = rx * (self.room_size + 1) + self.room_size // 2
                for d in range(self.door_size):
                    if door_start + d < self.size:
                        self.grid[door_start + d, wall_y] = 0

        # Place agent
        self.agent_pos = self._find_empty_cell_array()

        # Place goals in different rooms
        self.goal_positions = []
        for _ in range(self.n_goals):
            pos = self._find_empty_cell()
            if pos:
                self.grid[pos] = 2
                self.goal_positions.append(pos)

        # Place hazards
        for _ in range(self.n_hazards):
            pos = self._find_empty_cell()
            if pos:
                self.grid[pos] = 3

    def _find_empty_cell_array(self) -> np.ndarray:
        """Find empty cell and return as array"""
        pos = self._find_empty_cell()
        if pos:
            return np.array(pos)
        return np.array([1, 1])
