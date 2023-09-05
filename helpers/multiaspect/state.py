import json, os, multiprocessing, logging
logger = logging.getLogger('BucketStateManager')
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
            logger.debug(f'Found a dictionary to convert: {d}')
            return {key: self.deep_convert_dict(value) for key, value in d.items()}
        elif isinstance(d, list):
            logger.debug(f'Found a list to convert: {d}')
            return [self.deep_convert_dict(value) for value in d]
        elif isinstance(d, multiprocessing.managers.DictProxy):
            logger.debug(f'Found a DictProxy to convert: {d}')
            return self.deep_convert_dict(dict(d))
        else:
            logger.debug(f'Returning straight-through type {type(d)}')
            return d

    def save_state(self, state: dict, state_path: str = None):
        final_state = state
        if state_path is None:
            state_path = self.state_path
        logger.debug(f'Type of state: {type(state)}')
        final_state = self.deep_convert_dict(state)
        logger.info(f'Saving trainer state to {state_path}')
        with open(state_path, "w") as f:
            json.dump(final_state, f)

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                return json.load(f)
        else:
            return {}
