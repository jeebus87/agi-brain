"""
Interface Module

Provides chat and voice interfaces for interacting with the AGI brain.
"""

from .chat_ui import ChatInterface, create_gradio_app
from .voice_io import VoiceInterface, Microphone, Speaker
from .learning_pipeline import YouTubeLearner, WebLearner, LearningPipeline
