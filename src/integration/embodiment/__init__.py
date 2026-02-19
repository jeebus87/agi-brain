# Embodiment layer - connects brain to simulated environments
from .environment import Environment, GridWorld
from .sensory import SensoryProcessor, VisionProcessor
from .motor import MotorController
from .agent import EmbodiedAgent
