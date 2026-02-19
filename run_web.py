#!/usr/bin/env python
"""
Quick launcher for AGI Brain Web App

Usage:
    python run_web.py

Then open http://localhost:8000 in your browser.
"""

import subprocess
import sys

def main():
    # Check dependencies
    try:
        import fastapi
        import uvicorn
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "web_app/requirements.txt"])

    # Run the app
    print("\n" + "="*50)
    print("AGI Brain Web App")
    print("="*50)
    print("\nOpen http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop\n")

    subprocess.run([sys.executable, "web_app/app.py"])

if __name__ == "__main__":
    main()
