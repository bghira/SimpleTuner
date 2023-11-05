import json, os, multiprocessing, logging
from multiprocessing.managers import DictProxy

logger = logging.getLogger("BucketStateManager")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class BucketStateManager:
    def __init__(self, state_path, seen_images_path):
        self.state_path = state_path
        self.seen_images_path = seen_images_path

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
