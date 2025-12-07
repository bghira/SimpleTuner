import importlib
import logging
import os
from pathlib import Path

model_dir = Path(__file__).parent
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "WARNING"))

for subdir in model_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith("__"):
        model_file = subdir / "model.py"
        if model_file.exists():
            module_name = f"simpletuner.helpers.models.{subdir.name}.model"
            try:
                importlib.import_module(module_name)
            except Exception as exc:
                logger.exception(
                    "Failed to import model '%s'; continuing without it. Resolve the missing dependency if you need this model.",
                    subdir.name,
                )
