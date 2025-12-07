import json
import logging
import os
from multiprocessing.managers import DictProxy

logger = logging.getLogger("BucketStateManager")
from simpletuner.helpers.training.multi_process import should_log

logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


class BucketStateManager:
    def __init__(self, id: str):
        self.id = id

    def mangle_state_path(self, state_path):
        # When saving the state, it goes into the checkpoint dir.
        # However, we need to save a single state for each data backend.
        # Thus, we split the state_path from its extension, add self.id to the end of the name, and rejoin:
        if self.id in os.path.basename(state_path):
            return state_path
        filename, ext = os.path.splitext(state_path)
        return f"{filename}-{self.id}{ext}"

    def load_seen_images(self, state_path: str):
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_seen_images(self, seen_images, state_path: str):
        with open(state_path, "w") as f:
            json.dump(seen_images, f)

    def deep_convert_dict(self, d):
        if isinstance(d, dict):
            return {key: self.deep_convert_dict(value) for key, value in d.items()}
        elif isinstance(d, list):
            return [self.deep_convert_dict(value) for value in d]
        elif isinstance(d, DictProxy):
            return self.deep_convert_dict(dict(d))
        else:
            return d

    def save_state(self, state: dict, state_path: str):
        if state_path is None:
            raise ValueError("state_path must be specified")
        state_path = self.mangle_state_path(state_path)
        logger.debug(f"Saving trainer state to {state_path}")
        final_state = self.deep_convert_dict(state)
        with open(state_path, "w") as f:
            json.dump(final_state, f)

    def load_state(self, state_path: str):
        if state_path is None:
            raise ValueError("state_path must be specified")
        state_path = self.mangle_state_path(state_path)
        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                return json.load(f)
        else:
            logger.debug(f"load_state found no file: {state_path}")
            return {}
