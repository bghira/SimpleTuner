"""Create app instance for running with uvicorn."""

from .app import create_app, ServerMode

# Create the app instance
app = create_app(mode=ServerMode.TRAINER)
