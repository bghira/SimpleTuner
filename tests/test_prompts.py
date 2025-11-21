import unittest
from unittest.mock import patch

from simpletuner.helpers.prompts import PromptHandler


class _DummyBackend:
    def __init__(self, files):
        self.id = "test-backend"
        self._files = files

    def list_files(self, instance_data_dir=None, file_extensions=None):
        return self._files


class PromptHandlerTests(unittest.TestCase):
    def test_instanceprompt_returns_entry_per_image(self):
        backend = _DummyBackend(["a.jpg", "b.jpg", "c.jpg"])
        with patch(
            "simpletuner.helpers.prompts.StateTracker.get_image_files",
            return_value=None,
        ):
            captions, missing, paths = PromptHandler.get_all_captions(
                instance_data_dir="",
                use_captions=False,
                prepend_instance_prompt=False,
                data_backend=backend,
                caption_strategy="instanceprompt",
                instance_prompt="minecraft",
                return_image_paths=True,
            )
        self.assertEqual(missing, [])
        self.assertEqual(captions, ["minecraft", "minecraft", "minecraft"])
        self.assertEqual(paths, ["a.jpg", "b.jpg", "c.jpg"])


if __name__ == "__main__":
    unittest.main()
