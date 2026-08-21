import os
import runpy
import signal
import tempfile
import unittest
from pathlib import Path
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
        self.init_order = []
        self.startup_validation_called = False
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

    def init_diffusion_blocks_model(self, *_, **__):
        self.init_order.append("diffusion_blocks")
        return None

    def init_freeze_models(self, *_, **__):
        return None

    def init_distillation_adapter_modules(self, *_, **__):
        self.init_order.append("distillation_adapter_modules")
        return None

    def init_trainable_peft_adapter(self, *_, **__):
        self.init_order.append("peft_adapter")
        return None

    def init_diffusion_blocks_trainable_filter(self, *_, **__):
        self.init_order.append("diffusion_blocks_filter")
        return None

    def init_ema_model(self, *_, **__):
        return None

    def move_models(self, *_, **__):
        return None

    def init_distillation(self, *_, **__):
        self.init_order.append("distillation")
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

    def run_startup_validation(self, *_, **__):
        self.startup_validation_called = True

    def train(self, *_, **__):
        # Simulate a runtime failure after initial setup
        self.init_order.append("train")
        raise RuntimeError("simulated training failure")

    def cleanup(self):
        self.cleanup_called = True


class TrainEntryCleanupTest(unittest.TestCase):
    def test_rank_local_inductor_cache_uses_local_rank(self):
        import simpletuner.train as train_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {
                    "SIMPLETUNER_RANK_LOCAL_INDUCTOR_CACHE_ROOT": tmp_dir,
                    "LOCAL_RANK": "2",
                },
                clear=False,
            ):
                os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

                train_module._configure_rank_local_inductor_cache()

                expected_cache_dir = Path(tmp_dir) / "rank-2"
                self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], str(expected_cache_dir))
                self.assertTrue(expected_cache_dir.is_dir())

    def test_faulthandler_uses_rank_local_output_and_timeout(self):
        import simpletuner.train as train_module

        previous_stream = train_module._faulthandler_stream
        with tempfile.TemporaryDirectory() as tmp_dir:
            with (
                patch.dict(
                    os.environ,
                    {
                        "SIMPLETUNER_FAULTHANDLER_DIR": tmp_dir,
                        "SIMPLETUNER_FAULTHANDLER_TIMEOUT_SECONDS": "11",
                        "RANK": "3",
                    },
                    clear=False,
                ),
                patch("faulthandler.enable") as mock_enable,
                patch("faulthandler.register") as mock_register,
                patch("faulthandler.dump_traceback_later") as mock_dump_traceback_later,
            ):
                try:
                    train_module._faulthandler_stream = None
                    train_module._configure_faulthandler()

                    output_file = Path(tmp_dir) / "rank-3.log"
                    self.assertTrue(output_file.exists())
                    self.assertEqual(train_module._faulthandler_stream.name, str(output_file))
                    mock_enable.assert_called_once()
                    if hasattr(signal, "SIGUSR1"):
                        mock_register.assert_called_once()
                    else:
                        mock_register.assert_not_called()
                    mock_dump_traceback_later.assert_called_once()
                    self.assertEqual(mock_dump_traceback_later.call_args.args[0], 11)
                finally:
                    if train_module._faulthandler_stream is not None:
                        train_module._faulthandler_stream.close()
                    train_module._faulthandler_stream = previous_stream

    def test_faulthandler_rejects_invalid_timeout(self):
        import simpletuner.train as train_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(
                os.environ,
                {
                    "SIMPLETUNER_FAULTHANDLER_DIR": tmp_dir,
                    "SIMPLETUNER_FAULTHANDLER_TIMEOUT_SECONDS": "abc",
                },
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, "SIMPLETUNER_FAULTHANDLER_TIMEOUT_SECONDS.*'abc'"):
                    train_module._configure_faulthandler()
                if train_module._faulthandler_stream is not None:
                    train_module._faulthandler_stream.close()
                    train_module._faulthandler_stream = None

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
        self.assertIn(
            "distillation_adapter_modules",
            trainer.init_order,
            "distillation adapter module setup was not called",
        )
        self.assertIn("peft_adapter", trainer.init_order, "PEFT adapter setup was not called")
        self.assertIn("diffusion_blocks", trainer.init_order, "DiffusionBlocks setup was not called")
        self.assertIn("diffusion_blocks_filter", trainer.init_order, "DiffusionBlocks filtering was not called")
        self.assertIn("distillation", trainer.init_order, "distillation setup was not called")
        self.assertLess(
            trainer.init_order.index("distillation_adapter_modules"),
            trainer.init_order.index("peft_adapter"),
            "distillation adapter modules should be initialized before PEFT setup",
        )
        self.assertLess(
            trainer.init_order.index("peft_adapter"),
            trainer.init_order.index("diffusion_blocks_filter"),
            "DiffusionBlocks filtering should run after PEFT setup",
        )
        self.assertLess(
            trainer.init_order.index("diffusion_blocks_filter"),
            trainer.init_order.index("distillation"),
            "distillation setup should run after PEFT setup",
        )
        self.assertTrue(
            trainer.startup_validation_called,
            "train.py did not dispatch startup validation before training",
        )


if __name__ == "__main__":
    unittest.main()
