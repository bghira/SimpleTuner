import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend


class _CaptionConfig:
    def validate(self, *_args, **_kwargs):
        return None


class _CaptionMetadata(CaptionMetadataBackend):
    def __init__(self, events):
        self.events = events

    def load_image_metadata(self):
        self.events.append("load")

    def list_metadata_ids(self):
        return []

    def __len__(self):
        return 0


class FactoryCaptionBackendTests(unittest.TestCase):
    def _configure_backend(self, *, global_batch_size, dataset_batch_size=None):
        events = []
        metadata_backend = _CaptionMetadata(events)
        accelerator = SimpleNamespace(
            is_local_main_process=True,
            num_processes=2,
            process_index=0,
            wait_for_everyone=lambda: events.append("wait"),
        )
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(
            train_batch_size=global_batch_size,
            seed=123,
            skip_file_discovery="caption",
        )
        factory.accelerator = accelerator
        factory.caption_backends = {}

        backend = {
            "id": "captions",
            "type": "local",
            "dataset_type": "caption",
            "instance_data_dir": "data/captions",
        }
        if dataset_batch_size is not None:
            backend["train_batch_size"] = dataset_batch_size

        def init_backend_config(config, _args, _accelerator):
            runtime_config = {"repeats": 0}
            if "train_batch_size" in config:
                runtime_config["train_batch_size"] = config["train_batch_size"]
            return {
                "id": config["id"],
                "config": runtime_config,
                "dataset_type": config["dataset_type"],
            }

        with (
            patch("simpletuner.helpers.data_backend.factory.create_backend_config", return_value=_CaptionConfig()),
            patch("simpletuner.helpers.data_backend.factory.init_backend_config", side_effect=init_backend_config),
            patch(
                "simpletuner.helpers.data_backend.factory.build_backend_from_config",
                return_value={
                    "data_backend": MagicMock(),
                    "metadata_backend": metadata_backend,
                    "instance_data_dir": backend["instance_data_dir"],
                },
            ),
            patch("simpletuner.helpers.data_backend.factory.StateTracker") as state_tracker,
        ):
            state_tracker.register_data_backend.side_effect = lambda _backend: events.append("register")
            factory._configure_caption_backend(backend)
            registered_backend = state_tracker.register_data_backend.call_args.args[0]

        return registered_backend, events

    def test_caption_backend_uses_dataset_batch_size_override(self):
        registered_backend, events = self._configure_backend(global_batch_size=8, dataset_batch_size=3)

        self.assertIsInstance(registered_backend["sampler"], CaptionSampler)
        self.assertEqual(registered_backend["sampler"].batch_size, 3)
        self.assertIs(registered_backend["train_dataloader"].sampler, registered_backend["sampler"])
        self.assertEqual(events, ["wait", "load", "register"])

    def test_caption_backend_uses_global_batch_size_fallback(self):
        registered_backend, _events = self._configure_backend(global_batch_size=5)

        self.assertIsInstance(registered_backend["sampler"], CaptionSampler)
        self.assertEqual(registered_backend["sampler"].batch_size, 5)
        self.assertIn("train_dataloader", registered_backend)


if __name__ == "__main__":
    unittest.main()
