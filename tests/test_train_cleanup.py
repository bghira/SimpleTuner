import runpy
import unittest
from unittest.mock import MagicMock, patch

try:
    from tests import test_setup
except ModuleNotFoundError:
    import test_setup  # noqa: F401


class _DummyFetcher:
    def __init__(self):
        self.stopped = False

    def stop_fetching(self):
        self.stopped = True


class _FakeTrainer:
    instances = []

    def __init__(self, *args, **kwargs):
        self.cleanup_called = False
        self.bf = _DummyFetcher()
        self.config = MagicMock()
        _FakeTrainer.instances.append(self)

    def configure_webhook(self, *_, **__):
        return None

    def init_noise_schedule(self, *_, **__):
        return None

    def init_seed(self, *_, **__):
        return None

    def init_huggingface_hub(self, *_, **__):
        return None

    def init_preprocessing_models(self, *_, **__):
        return None

    def init_precision(self, *_, **__):
        return None

    def init_data_backend(self, *_, **__):
        return None

    def init_unload_text_encoder(self, *_, **__):
        return None

    def init_unload_vae(self, *_, **__):
        return None

    def init_load_base_model(self, *_, **__):
        return None

    def init_controlnet_model(self, *_, **__):
        return None

    def init_tread_model(self, *_, **__):
        return None

    def init_freeze_models(self, *_, **__):
        return None

    def init_trainable_peft_adapter(self, *_, **__):
        return None

    def init_ema_model(self, *_, **__):
        return None

    def move_models(self, *_, **__):
        return None

    def init_distillation(self, *_, **__):
        return None

    def init_validations(self, *_, **__):
        return None

    def init_benchmark_base_model(self, *_, **__):
        return None

    def init_delete_model_caches(self, *_, **__):
        return None

    def resume_and_prepare(self, *_, **__):
        return None

    def init_trackers(self, *_, **__):
        return None

    def train(self, *_, **__):
        # Simulate a runtime failure after initial setup
        raise RuntimeError("simulated training failure")

    def cleanup(self):
        self.cleanup_called = True


class TrainEntryCleanupTest(unittest.TestCase):
    def test_train_main_invokes_cleanup_on_failure(self):
        """Train entrypoint should call trainer.cleanup when a failure occurs."""
        _FakeTrainer.instances.clear()

        with (
            patch("simpletuner.helpers.training.trainer.Trainer", _FakeTrainer),
            patch("simpletuner.train.AttentionBackendController.apply", MagicMock()),
            patch("multiprocessing.set_start_method", MagicMock()),
        ):
            with self.assertRaises(RuntimeError):
                runpy.run_module("simpletuner.train", run_name="__main__")

        self.assertTrue(_FakeTrainer.instances, "Fake trainer was not constructed")
        trainer = _FakeTrainer.instances[0]
        self.assertTrue(
            trainer.cleanup_called,
            "train.py did not invoke trainer.cleanup() after a training failure",
        )


if __name__ == "__main__":
    unittest.main()
