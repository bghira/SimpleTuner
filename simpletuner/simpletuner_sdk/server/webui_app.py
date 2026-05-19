"""Create app instance for running with uvicorn."""

import os

from .app import create_unified_app
from .services.cloud.storage.base import get_local_state_dir

os.environ.setdefault("SIMPLETUNER_STATE_DIR", str(get_local_state_dir()))

# Create the app instance
app = create_unified_app()
