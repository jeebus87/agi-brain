"""
Xemsa - Autonomous AI Brain

Deploy with: modal deploy modal_app.py
Run locally: modal serve modal_app.py

Named after Emily, Savannah, Seth, and Xavier.
"""

import modal

# Create the Modal app
app = modal.App("xemsa")

# Create a persistent volume for brain data (survives redeployments)
brain_volume = modal.Volume.from_name("xemsa-data", create_if_missing=True)

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "numpy>=1.24.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
    )
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("web_app", remote_path="/app/web_app")
)


@app.function(
    image=image,
    volumes={"/data/brain": brain_volume},
    gpu="T4",  # 16GB VRAM - plenty for 100K neurons
    timeout=600,
    scaledown_window=300,  # Keep warm for 5 min
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app(custom_domains=["xemsa.com"])
def web():
    """Serve the AGI Brain FastAPI application."""
    import os
    import sys
    sys.path.insert(0, "/app")
    sys.path.insert(0, "/app/src")

    # Set environment variable for brain data location
    os.environ["BRAIN_DATA_DIR"] = "/data/brain"

    # Ensure brain_data directories exist
    from pathlib import Path
    brain_dir = Path("/data/brain")
    (brain_dir / "chats").mkdir(parents=True, exist_ok=True)
    (brain_dir / "trash").mkdir(parents=True, exist_ok=True)

    # Import and return the FastAPI app
    from web_app.app import app as fastapi_app

    # Commit volume changes on each request
    brain_volume.commit()

    return fastapi_app


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("AGI Brain deployed!")
    print("Run 'modal serve modal_app.py' for local dev")
    print("Run 'modal deploy modal_app.py' to deploy to production")
