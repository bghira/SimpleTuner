import json, os, multiprocessing


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

    def save_state(self, state: dict, state_path: str = None):
        if state_path is None:
            state_path = self.state_path
        if type(state) == multiprocessing.Manager:
            state = state.deepcopy()
        print(f'Saving state type: {type(state)}')
        with open(state_path, "w") as f:
            json.dump(state, f)

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                return json.load(f)
        else:
            return {}
