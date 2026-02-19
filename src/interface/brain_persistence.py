"""
Brain Persistence - Save and Load Brain State

Enables persistent storage of the AGI brain:
- Save to Google Drive for Colab sessions
- Load previous state on startup
- Automatic checkpointing during learning
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class BrainPersistence:
    """
    Handles saving and loading the complete brain state.

    Saves:
    - Semantic memory (concepts, associations)
    - Phoneme learner weights
    - Conversation history
    - Learning statistics
    """

    def __init__(self, save_dir: str = "/content/drive/MyDrive/agi-brain"):
        """
        Initialize brain persistence.

        Args:
            save_dir: Directory for saving brain state.
                      Default is Google Drive path for Colab.
        """
        self.save_dir = Path(save_dir)
        self.metadata_file = self.save_dir / "brain_metadata.json"

    def _ensure_dir(self):
        """Create save directory if it doesn't exist."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def mount_drive(self):
        """Mount Google Drive (for Colab)."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully!")
            return True
        except ImportError:
            print("Not running in Colab - using local storage")
            return False
        except Exception as e:
            print(f"Failed to mount Drive: {e}")
            return False

    def save_brain(
        self,
        brain,
        name: str = "default",
        include_history: bool = True
    ) -> str:
        """
        Save complete brain state.

        Args:
            brain: AGIBrain or SimpleBrain instance
            name: Name for this save (allows multiple saves)
            include_history: Whether to save conversation history

        Returns:
            Path to saved state
        """
        self._ensure_dir()

        save_path = self.save_dir / name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save semantic memory
        if hasattr(brain, 'semantic_memory'):
            sem_path = str(save_path / "semantic_memory.npz")
            brain.semantic_memory.save(sem_path)
            print(f"  Saved semantic memory: {brain.semantic_memory.get_vocabulary_size()} words")

        # Save phoneme learner
        if hasattr(brain, 'phoneme_learner'):
            phon_path = str(save_path / "phoneme_learner.npz")
            brain.phoneme_learner.save(phon_path)
            print(f"  Saved phoneme learner")

        # Save conversation history
        if include_history and hasattr(brain, 'conversation_history'):
            history_path = save_path / "conversation_history.json"
            with open(history_path, 'w') as f:
                json.dump(brain.conversation_history, f)
            print(f"  Saved {len(brain.conversation_history)} conversation turns")

        # Save metadata
        metadata = {
            'name': name,
            'saved_at': datetime.now().isoformat(),
            'vocabulary_size': brain.semantic_memory.get_vocabulary_size() if hasattr(brain, 'semantic_memory') else 0,
            'concepts': len(brain.semantic_memory.concepts) if hasattr(brain, 'semantic_memory') else 0,
            'conversation_turns': len(brain.conversation_history) if hasattr(brain, 'conversation_history') else 0,
        }

        meta_path = save_path / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update global metadata
        self._update_global_metadata(name, metadata)

        print(f"\nBrain saved to: {save_path}")
        return str(save_path)

    def load_brain(self, brain, name: str = "default") -> bool:
        """
        Load brain state from saved files.

        Args:
            brain: AGIBrain or SimpleBrain instance to load into
            name: Name of save to load

        Returns:
            True if loaded successfully, False otherwise
        """
        save_path = self.save_dir / name

        if not save_path.exists():
            print(f"No saved brain found at: {save_path}")
            return False

        print(f"Loading brain from: {save_path}")

        # Load semantic memory
        sem_path = save_path / "semantic_memory.npz"
        if sem_path.exists() and hasattr(brain, 'semantic_memory'):
            brain.semantic_memory.load(str(sem_path))
            print(f"  Loaded semantic memory: {brain.semantic_memory.get_vocabulary_size()} words")

        # Load phoneme learner
        phon_path = save_path / "phoneme_learner.npz"
        if phon_path.exists() and hasattr(brain, 'phoneme_learner'):
            brain.phoneme_learner.load(str(phon_path))
            print(f"  Loaded phoneme learner")

        # Load conversation history
        history_path = save_path / "conversation_history.json"
        if history_path.exists() and hasattr(brain, 'conversation_history'):
            with open(history_path, 'r') as f:
                brain.conversation_history = json.load(f)
            print(f"  Loaded {len(brain.conversation_history)} conversation turns")

        # Load metadata
        meta_path = save_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            print(f"\nBrain loaded (saved at: {metadata.get('saved_at', 'unknown')})")

        return True

    def _update_global_metadata(self, name: str, metadata: Dict):
        """Update global metadata file with save info."""
        global_meta = {}
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                global_meta = json.load(f)

        global_meta[name] = metadata

        with open(self.metadata_file, 'w') as f:
            json.dump(global_meta, f, indent=2)

    def list_saves(self) -> Dict[str, Dict]:
        """List all saved brain states."""
        if not self.metadata_file.exists():
            return {}

        with open(self.metadata_file, 'r') as f:
            return json.load(f)

    def delete_save(self, name: str) -> bool:
        """Delete a saved brain state."""
        import shutil

        save_path = self.save_dir / name
        if save_path.exists():
            shutil.rmtree(save_path)

            # Update metadata
            global_meta = self.list_saves()
            if name in global_meta:
                del global_meta[name]
                with open(self.metadata_file, 'w') as f:
                    json.dump(global_meta, f, indent=2)

            print(f"Deleted save: {name}")
            return True
        return False

    def auto_checkpoint(
        self,
        brain,
        interval_turns: int = 50,
        max_checkpoints: int = 3
    ):
        """
        Create automatic checkpoint based on conversation turns.

        Call this periodically during learning.
        """
        turns = len(brain.conversation_history) if hasattr(brain, 'conversation_history') else 0

        if turns > 0 and turns % interval_turns == 0:
            checkpoint_name = f"checkpoint_{turns}"
            self.save_brain(brain, checkpoint_name, include_history=True)

            # Clean up old checkpoints
            saves = self.list_saves()
            checkpoints = sorted([k for k in saves.keys() if k.startswith("checkpoint_")])

            while len(checkpoints) > max_checkpoints:
                oldest = checkpoints.pop(0)
                self.delete_save(oldest)


def setup_persistence_for_colab(save_name: str = "my_brain"):
    """
    Helper function to set up persistence in Colab.

    Usage in Colab:
        from src.interface.brain_persistence import setup_persistence_for_colab
        persistence, brain = setup_persistence_for_colab("my_brain")

        # Use brain...

        # Save when done:
        persistence.save_brain(brain, "my_brain")
    """
    persistence = BrainPersistence()

    # Mount Google Drive
    persistence.mount_drive()

    # Check for existing save
    saves = persistence.list_saves()

    if save_name in saves:
        print(f"\nFound existing brain: {save_name}")
        print(f"  Vocabulary: {saves[save_name].get('vocabulary_size', 0)} words")
        print(f"  Saved at: {saves[save_name].get('saved_at', 'unknown')}")
        return persistence, save_name
    else:
        print(f"\nNo existing brain found. Will create: {save_name}")
        return persistence, None


if __name__ == "__main__":
    print("Brain Persistence Module")
    print("=" * 50)

    # Test locally
    persistence = BrainPersistence(save_dir="./test_brain_saves")

    # Create a mock brain for testing
    class MockBrain:
        def __init__(self):
            from ..language.semantic_memory import SemanticMemory, SemanticConfig
            self.semantic_memory = SemanticMemory(SemanticConfig(n_hard_locations=100))
            self.conversation_history = []

            # Add some test data
            self.semantic_memory.create_concept(word="test")
            self.semantic_memory.create_concept(word="hello")
            self.conversation_history.append(("hi", "hello"))

    print("\nTesting save/load...")
    brain = MockBrain()

    persistence.save_brain(brain, "test_save")

    saves = persistence.list_saves()
    print(f"\nSaved brains: {list(saves.keys())}")

    # Test load
    brain2 = MockBrain()
    persistence.load_brain(brain2, "test_save")
