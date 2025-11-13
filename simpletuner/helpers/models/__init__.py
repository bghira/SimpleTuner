import importlib
import logging
import os
from pathlib import Path

# Auto-discover and import all model modules
model_dir = Path(__file__).parent
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))

strict_model_imports = os.environ.get("SIMPLETUNER_STRICT_MODEL_IMPORTS") == "1"
for subdir in model_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("__"):
        model_file = subdir / "model.py"
        if model_file.exists():
            module_name = f"simpletuner.helpers.models.{subdir.name}.model"
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                if strict_model_imports:
                    raise
                logger.warning("Skipping optional model '%s' due to import error: %s", subdir.name, exc)
