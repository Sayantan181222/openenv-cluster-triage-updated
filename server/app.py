import sys
import os
import uvicorn

# Point back to the root directory so it can find your actual app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the fully mounted FastAPI + Gradio app from the root app.py
from app import fastapi_app as app

def main():
    """Entry point required by the OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()