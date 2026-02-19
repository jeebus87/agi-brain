"""
AGI Brain Web App - Modern Chat Interface

Run with: python web_app/app.py
Open: http://localhost:8000
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.language.semantic_memory import SemanticMemory, SemanticConfig
from src.language.speech_generator import SpeechGenerator
from src.language.phoneme_learner import PhonemeLearner, PhonemeConfig

# Initialize app
app = FastAPI(title="AGI Brain")

# Brain storage directory
BRAIN_DIR = Path(__file__).parent / "brain_data"
BRAIN_DIR.mkdir(exist_ok=True)


class Brain:
    """AGI Brain that learns from conversation."""

    def __init__(self):
        self.semantic_memory = None
        self.speech_generator = None
        self.phoneme_learner = None
        self.conversation_history = []
        self.initialized = False

    def initialize(self):
        """Initialize brain components."""
        if self.initialized:
            return

        sem_config = SemanticConfig(n_hard_locations=5000, address_size=500, data_size=500)
        self.semantic_memory = SemanticMemory(sem_config)
        self.speech_generator = SpeechGenerator()
        phoneme_config = PhonemeConfig(n_input=800, n_phoneme_neurons=500, n_phonemes=20)
        self.phoneme_learner = PhonemeLearner(phoneme_config)
        self.conversation_history = []
        self.initialized = True
        print("Brain initialized!")

    def process(self, text: str) -> str:
        """Process input and generate response."""
        if not self.initialized:
            self.initialize()

        words = text.lower().split()

        # Learn from input
        self.semantic_memory.learn_from_sentence(words)

        # Find associations
        all_activations = {}
        for word in words:
            if self.semantic_memory.lookup(word):
                activations = self.semantic_memory.spread_activation(word, depth=2)
                for w, a in activations.items():
                    all_activations[w] = all_activations.get(w, 0) + a

        # Generate response
        if all_activations:
            top_words = sorted(all_activations.items(), key=lambda x: -x[1])[:5]
            associated = [w for w, a in top_words if w not in words]
            if associated:
                response = f"When you mention {words[0] if words else 'that'}, I think of: {', '.join(associated[:3])}"
            else:
                response = "I understand. Tell me more so I can learn."
        else:
            if words:
                response = f"I'm learning about '{words[0]}'. What else can you tell me?"
            else:
                response = "I'm listening and learning."

        self.conversation_history.append({"role": "user", "content": text})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def save(self, name: str = "default"):
        """Save brain state to disk."""
        if not self.initialized:
            return False

        save_path = BRAIN_DIR / name
        save_path.mkdir(exist_ok=True)

        # Save semantic memory
        self.semantic_memory.save(str(save_path / "semantic_memory.npz"))

        # Save conversation history
        with open(save_path / "history.json", "w") as f:
            json.dump(self.conversation_history, f)

        # Save metadata
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "vocabulary_size": self.semantic_memory.get_vocabulary_size(),
            "concepts": len(self.semantic_memory.concepts),
            "conversations": len(self.conversation_history) // 2
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        return True

    def load(self, name: str = "default") -> bool:
        """Load brain state from disk."""
        save_path = BRAIN_DIR / name

        if not save_path.exists():
            return False

        self.initialize()

        # Load semantic memory
        sem_path = save_path / "semantic_memory.npz"
        if sem_path.exists():
            self.semantic_memory.load(str(sem_path))

        # Load conversation history
        history_path = save_path / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                self.conversation_history = json.load(f)

        return True

    def get_status(self) -> dict:
        """Get brain status."""
        if not self.initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "vocabulary_size": self.semantic_memory.get_vocabulary_size(),
            "concepts": len(self.semantic_memory.concepts),
            "conversations": len(self.conversation_history) // 2
        }

    def list_saves(self) -> list:
        """List all saved brains."""
        saves = []
        for path in BRAIN_DIR.iterdir():
            if path.is_dir():
                meta_path = path / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["name"] = path.name
                    saves.append(meta)
        return saves


# Global brain instance
brain = Brain()


# API Models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class SaveRequest(BaseModel):
    name: str = "default"

class LoadRequest(BaseModel):
    name: str = "default"


# API Routes
@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    response = brain.process(request.message)
    return {"response": response}

@app.post("/api/save")
async def save(request: SaveRequest):
    """Save the brain."""
    success = brain.save(request.name)
    if success:
        return {"status": "saved", "name": request.name}
    raise HTTPException(status_code=500, detail="Failed to save")

@app.post("/api/load")
async def load(request: LoadRequest):
    """Load a saved brain."""
    success = brain.load(request.name)
    if success:
        return {
            "status": "loaded",
            "name": request.name,
            "history": brain.conversation_history
        }
    raise HTTPException(status_code=404, detail="Brain not found")

@app.get("/api/status")
async def status():
    """Get brain status."""
    return brain.get_status()

@app.get("/api/saves")
async def list_saves():
    """List all saved brains."""
    return brain.list_saves()

@app.post("/api/new")
async def new_brain():
    """Start a new brain."""
    brain.initialized = False
    brain.initialize()
    return {"status": "created"}

@app.get("/api/history")
async def get_history():
    """Get conversation history."""
    return {"history": brain.conversation_history}


# Serve frontend
@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


if __name__ == "__main__":
    print("\n" + "="*50)
    print("AGI Brain Web App")
    print("="*50)
    print("\nStarting server...")
    print("Open http://localhost:8000 in your browser\n")

    # Try to load default brain
    if brain.load("default"):
        print(f"Loaded saved brain: {brain.semantic_memory.get_vocabulary_size()} words")
    else:
        brain.initialize()
        print("Started with fresh brain")

    uvicorn.run(app, host="0.0.0.0", port=8000)
