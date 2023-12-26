import json, os, multiprocessing, logging
from multiprocessing.managers import DictProxy

logger = logging.getLogger("BucketStateManager")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class BucketStateManager:
    def __init__(self, id: str, state_path, seen_images_path):
        self.id = id
        self.state_path = self.mangle_state_path(state_path)
        # seen_images_path is pre-mangled by the dataset factory
        self.seen_images_path = seen_images_path

    def mangle_state_path(self, state_path):
        # When saving the state, it goes into the checkpoint dir.
        # However, we need to save a single state for each data backend.
        # Thus, we split the state_path from its extension, add self.id to the end of the name, and rejoin:
        if self.id in state_path:
            return state_path
        filename, ext = os.path.splitext(state_path)
        return f"{filename}-{self.id}{ext}"

    def load_seen_images(self):
        if os.path.exists(self.seen_images_path):
            with open(self.seen_images_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def save_seen_images(self, seen_images):
        with open(self.seen_images_path, "w") as f:
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

    def save_state(self, state: dict, state_path: str = None):
        final_state = state
        if state_path is None:
            state_path = self.state_path
        else:
            state_path = self.mangle_state_path(state_path)
        logger.debug(f"Type of state: {type(state)}")
        final_state = self.deep_convert_dict(state)
        logger.info(f"Saving trainer state to {state_path}")
        with open(state_path, "w") as f:
            json.dump(final_state, f)

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                return json.load(f)
        else:
            logger.debug(f"load_state found no file: {self.state_path}")
            return {}
