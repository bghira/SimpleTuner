import json
import os
import tempfile
import unittest

from simpletuner.helpers.multiaspect.state import BucketStateManager


class TestBucketStateManager(unittest.TestCase):
    def test_state_path_is_per_dataset(self):
        manager = BucketStateManager("photos")

        self.assertEqual(
            manager.mangle_state_path(os.path.join("checkpoint-10", "training_state.json")),
            os.path.join("checkpoint-10", "training_state-photos.json"),
        )
        self.assertEqual(
            manager.mangle_state_path(os.path.join("checkpoint-10", "training_state-rank1.json")),
            os.path.join("checkpoint-10", "training_state-rank1-photos.json"),
        )

    def test_dataset_id_inside_trainer_filename_keeps_its_own_file(self):
        trainer_path = os.path.join("checkpoint-10", "training_state.json")
        for dataset_id in ("train", "training", "state", "training_state", "json", "a"):
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(
                    BucketStateManager(dataset_id).mangle_state_path(trainer_path),
                    os.path.join("checkpoint-10", f"training_state-{dataset_id}.json"),
                )

    def test_save_state_leaves_trainer_state_untouched(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer_path = os.path.join(checkpoint_dir, "training_state.json")
            trainer_state = {"global_step": 10}
            with open(trainer_path, "w") as handle:
                json.dump(trainer_state, handle)

            manager = BucketStateManager("train")
            sampler_state = {"current_epoch": 2}
            manager.save_state(sampler_state, trainer_path)

            with open(trainer_path) as handle:
                self.assertEqual(json.load(handle), trainer_state)
            self.assertEqual(manager.load_state(trainer_path), sampler_state)


if __name__ == "__main__":
    unittest.main()
