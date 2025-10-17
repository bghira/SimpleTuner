"""Create app instance for running with uvicorn."""

from .app import create_unified_app

# Create the app instance
app = create_unified_app()
