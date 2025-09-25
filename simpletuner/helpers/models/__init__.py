import importlib
import os
from pathlib import Path

# Auto-discover and import all model modules
model_dir = Path(__file__).parent
for subdir in model_dir.iterdir():
    if subdir.is_dir() and not subdir.name.startswith('__'):
        model_file = subdir / 'model.py'
        if model_file.exists():
            module_name = f'simpletuner.helpers.models.{subdir.name}.model'
            try:
                importlib.import_module(module_name)
            except ImportError:
                pass