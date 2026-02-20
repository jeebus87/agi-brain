"""
Xemsa - Autonomous AI Brain

Run with: python web_app/app.py
Open: http://localhost:8000
"""

import os
import json
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from web_app.full_brain import FullAGIBrain, BrainConfig

# Initialize app
app = FastAPI(title="Xemsa - Autonomous AI")

# Storage directories
BRAIN_DIR = Path(__file__).parent / "brain_data"
CHATS_DIR = BRAIN_DIR / "chats"
TRASH_DIR = BRAIN_DIR / "trash"
BRAIN_DIR.mkdir(exist_ok=True)
CHATS_DIR.mkdir(exist_ok=True)
TRASH_DIR.mkdir(exist_ok=True)

# Global brain instance with full capabilities
brain = FullAGIBrain(BrainConfig(
    n_neurons=100_000,
    enable_reasoning=True,
    enable_audio=True,
    enable_learning=True
))

# Current active chat ID
current_chat_id: Optional[str] = None


# API Models
class ChatRequest(BaseModel):
    message: str

class LearnURLRequest(BaseModel):
    url: str

class ChatRenameRequest(BaseModel):
    title: str


# Chat Management Helpers
def generate_chat_id() -> str:
    """Generate a unique chat ID."""
    return str(uuid.uuid4())[:8]


def get_chat_path(chat_id: str, trash: bool = False) -> Path:
    """Get the file path for a chat."""
    directory = TRASH_DIR if trash else CHATS_DIR
    return directory / f"{chat_id}.json"


def load_chat(chat_id: str, trash: bool = False) -> Optional[dict]:
    """Load a chat from disk."""
    path = get_chat_path(chat_id, trash)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return None


def save_chat(chat_data: dict, trash: bool = False):
    """Save a chat to disk."""
    chat_id = chat_data['id']
    path = get_chat_path(chat_id, trash)
    chat_data['updated_at'] = datetime.now().isoformat()
    with open(path, 'w') as f:
        json.dump(chat_data, f, indent=2)


def delete_chat_file(chat_id: str, trash: bool = False):
    """Delete a chat file from disk."""
    path = get_chat_path(chat_id, trash)
    if path.exists():
        path.unlink()


def list_chats(trash: bool = False) -> List[dict]:
    """List all chats (or trashed chats)."""
    directory = TRASH_DIR if trash else CHATS_DIR
    chats = []
    for path in directory.glob("*.json"):
        try:
            with open(path, 'r') as f:
                chat = json.load(f)
                # Return summary info only
                chats.append({
                    'id': chat['id'],
                    'title': chat.get('title', 'Untitled'),
                    'created_at': chat.get('created_at'),
                    'updated_at': chat.get('updated_at'),
                    'deleted_at': chat.get('deleted_at'),
                    'message_count': len(chat.get('messages', []))
                })
        except Exception:
            continue
    # Sort by updated_at descending (most recent first)
    chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return chats


def create_new_chat(title: Optional[str] = None) -> dict:
    """Create a new chat."""
    chat_id = generate_chat_id()
    now = datetime.now().isoformat()
    chat_data = {
        'id': chat_id,
        'title': title or f"Chat {chat_id}",
        'created_at': now,
        'updated_at': now,
        'messages': []
    }
    save_chat(chat_data)
    return chat_data


def generate_title_from_message(message: str) -> str:
    """Generate a chat title from the first message."""
    # Take first 40 chars, clean up
    title = message.strip()[:40]
    if len(message) > 40:
        title += "..."
    return title


# API Routes - Chat Management

@app.get("/api/chats")
async def list_all_chats():
    """List all active chats."""
    return {"chats": list_chats(trash=False)}


@app.post("/api/chats")
async def create_chat():
    """Create a new chat and set it as current."""
    global current_chat_id
    chat = create_new_chat()
    current_chat_id = chat['id']

    # Clear brain's conversation history for new chat
    brain.conversation_history = []
    brain.working_memory = []

    return {"chat": chat, "current": True}


@app.get("/api/chats/current")
async def get_current_chat():
    """Get the current active chat."""
    global current_chat_id
    if not current_chat_id:
        # Create a new chat if none exists
        chat = create_new_chat()
        current_chat_id = chat['id']
        return {"chat": chat}

    chat = load_chat(current_chat_id)
    if not chat:
        # Chat was deleted, create new one
        chat = create_new_chat()
        current_chat_id = chat['id']

    return {"chat": chat}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Get a specific chat by ID."""
    chat = load_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": chat}


@app.post("/api/chats/{chat_id}/switch")
async def switch_chat(chat_id: str):
    """Switch to a different chat."""
    global current_chat_id

    chat = load_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    current_chat_id = chat_id

    # Load chat history into brain
    brain.conversation_history = chat.get('messages', [])
    brain.working_memory = []

    return {"chat": chat, "switched": True}


@app.put("/api/chats/{chat_id}")
async def update_chat(chat_id: str, request: ChatRenameRequest):
    """Update chat title."""
    chat = load_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    chat['title'] = request.title
    save_chat(chat)

    return {"chat": chat}


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Move a chat to trash (soft delete)."""
    global current_chat_id

    chat = load_chat(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    # Mark as deleted and move to trash
    chat['deleted_at'] = datetime.now().isoformat()
    save_chat(chat, trash=True)
    delete_chat_file(chat_id, trash=False)

    # If this was the current chat, clear it
    if current_chat_id == chat_id:
        current_chat_id = None
        brain.conversation_history = []
        brain.working_memory = []

    return {"deleted": True, "chat_id": chat_id}


# API Routes - Trash Management

@app.get("/api/trash")
async def list_trash():
    """List all chats in trash."""
    return {"chats": list_chats(trash=True)}


@app.post("/api/trash/{chat_id}/restore")
async def restore_chat(chat_id: str):
    """Restore a chat from trash."""
    chat = load_chat(chat_id, trash=True)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found in trash")

    # Remove deleted_at and move back to chats
    chat.pop('deleted_at', None)
    save_chat(chat, trash=False)
    delete_chat_file(chat_id, trash=True)

    return {"restored": True, "chat": chat}


@app.delete("/api/trash/{chat_id}")
async def permanently_delete_chat(chat_id: str):
    """Permanently delete a chat from trash."""
    chat = load_chat(chat_id, trash=True)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found in trash")

    delete_chat_file(chat_id, trash=True)

    return {"permanently_deleted": True, "chat_id": chat_id}


@app.delete("/api/trash")
async def empty_trash():
    """Permanently delete all chats in trash."""
    count = 0
    for path in TRASH_DIR.glob("*.json"):
        path.unlink()
        count += 1

    return {"emptied": True, "count": count}


# API Routes - Chat Messages

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    global current_chat_id

    # Ensure we have a current chat
    if not current_chat_id:
        chat_data = create_new_chat()
        current_chat_id = chat_data['id']

    # Get response from brain
    response = brain.process(request.message)

    # Load current chat and update it
    chat_data = load_chat(current_chat_id)
    if not chat_data:
        chat_data = create_new_chat()
        current_chat_id = chat_data['id']

    # Add messages to chat history
    chat_data['messages'].append({'role': 'user', 'content': request.message})
    chat_data['messages'].append({'role': 'assistant', 'content': response})

    # Auto-generate title from first user message
    if len(chat_data['messages']) == 2:  # First exchange
        chat_data['title'] = generate_title_from_message(request.message)

    # Save chat
    save_chat(chat_data)

    # Auto-save brain knowledge
    brain.save(str(BRAIN_DIR / "default"))

    return {"response": response, "chat_id": current_chat_id}


@app.post("/api/chat/json")
async def chat_json(request: ChatRequest):
    """Send a message and get a structured JSON response with reasoning."""
    global current_chat_id

    # Ensure we have a current chat
    if not current_chat_id:
        chat_data = create_new_chat()
        current_chat_id = chat_data['id']

    result = brain.process_json(request.message)

    # Load current chat and update it
    chat_data = load_chat(current_chat_id)
    if not chat_data:
        chat_data = create_new_chat()
        current_chat_id = chat_data['id']

    # Add messages to chat history
    chat_data['messages'].append({'role': 'user', 'content': request.message})
    chat_data['messages'].append({'role': 'assistant', 'content': result.get('response', '')})

    # Auto-generate title from first user message
    if len(chat_data['messages']) == 2:
        chat_data['title'] = generate_title_from_message(request.message)

    save_chat(chat_data)
    brain.save(str(BRAIN_DIR / "default"))

    return result


@app.get("/api/status")
async def status():
    """Get brain status."""
    return brain.get_status()


@app.get("/api/history")
async def get_history():
    """Get conversation history for current chat."""
    global current_chat_id
    if current_chat_id:
        chat = load_chat(current_chat_id)
        if chat:
            return {"history": chat.get('messages', [])}
    return {"history": brain.conversation_history}


@app.post("/api/new")
async def new_chat():
    """Start a new chat - creates a new chat and preserves knowledge."""
    global current_chat_id

    # Create new chat
    chat_data = create_new_chat()
    current_chat_id = chat_data['id']

    # Clear brain's conversation history for new chat
    brain.conversation_history = []
    brain.working_memory = []

    # Save the brain state (preserving knowledge)
    brain.save(str(BRAIN_DIR / "default"))

    vocab_size = brain.semantic_memory.get_vocabulary_size()
    kg_stats = brain.knowledge_graph.get_stats() if brain.knowledge_graph else {}

    return {
        "status": "new_chat",
        "chat": chat_data,
        "message": "New chat created. All knowledge preserved.",
        "vocabulary": vocab_size,
        "knowledge_facts": kg_stats.get('relationships', 0)
    }


@app.post("/api/reset")
async def reset_brain():
    """FULL RESET - Creates a completely new brain (use with caution)."""
    global brain
    brain = FullAGIBrain(BrainConfig(
        n_neurons=100_000,
        enable_reasoning=True,
        enable_audio=True,
        enable_learning=True
    ))
    brain.initialize()

    # Load dictionary
    from src.language.dictionary_loader import DictionaryLoader
    loader = DictionaryLoader()
    loader.load_into_brain(brain, use_wordnet=loader.wordnet_available)

    brain.save(str(BRAIN_DIR / "default"))

    return {"status": "reset", "vocabulary": brain.semantic_memory.get_vocabulary_size()}


@app.post("/api/learn-url")
async def learn_from_url(request: LearnURLRequest):
    """Learn from a web page."""
    try:
        stats = brain.learn_from_url(request.url)
        brain.save(str(BRAIN_DIR / "default"))
        return {
            "status": "learned",
            "url": request.url,
            "words_learned": stats.get("words_learned", 0),
            "sentences": stats.get("sentences", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/learn-youtube")
async def learn_from_youtube(request: LearnURLRequest):
    """Learn from a YouTube video."""
    try:
        stats = brain.learn_from_youtube(request.url)
        brain.save(str(BRAIN_DIR / "default"))
        return {
            "status": "learned",
            "url": request.url,
            "title": stats.get("title", "Unknown"),
            "words_learned": stats.get("words_learned", 0),
            "segments": stats.get("segments", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend
@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse(Path(__file__).parent / "index.html")


# Startup: Load most recent chat or create new one
@app.on_event("startup")
async def startup_event():
    global current_chat_id

    # Load brain
    if brain.load(str(BRAIN_DIR / "default")):
        print(f"Loaded saved brain: {brain.semantic_memory.get_vocabulary_size()} words")
    else:
        brain.initialize()
        print("Started with fresh brain")

        # Auto-load dictionary if brain has few words
        if brain.semantic_memory.get_vocabulary_size() < 100:
            print("Loading dictionary vocabulary...")
            from src.language.dictionary_loader import DictionaryLoader
            loader = DictionaryLoader()
            stats = loader.load_into_brain(brain, use_wordnet=loader.wordnet_available)
            print(f"Loaded {stats['words_loaded']} words, {stats['associations_created']} associations")
            brain.save(str(BRAIN_DIR / "default"))

    # Load most recent chat if exists
    chats = list_chats()
    if chats:
        most_recent = chats[0]
        current_chat_id = most_recent['id']
        chat_data = load_chat(current_chat_id)
        if chat_data:
            brain.conversation_history = chat_data.get('messages', [])
            print(f"Loaded chat: {most_recent['title']}")

    status_data = brain.get_status()
    print(f"\nBrain ready:")
    print(f"  - Vocabulary: {status_data['vocabulary_size']} words")
    print(f"  - Concepts: {status_data['concepts']}")
    print(f"  - Reasoning: {status_data['reasoning_enabled']}")
    print(f"  - Active chats: {len(chats)}")
    print("\nBrain auto-saves after each message.\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("     Xemsa - Autonomous AI")
    print("="*60)
    print(f"\n  Neurons: {brain.config.n_neurons:,}")
    print(f"  Reasoning: {'Enabled' if brain.config.enable_reasoning else 'Disabled'}")
    print(f"  Audio I/O: {'Enabled' if brain.config.enable_audio else 'Disabled'}")
    print("\nStarting server...")
    print("Open http://localhost:8000 in your browser\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
