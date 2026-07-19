import types
import unittest
from unittest.mock import patch

from simpletuner.helpers.training.trainer import Trainer
from simpletuner.helpers.training.validation import Validation


class DummyModel:
    def __init__(self):
        self.unload_calls = 0

    def unload_text_encoder(self):
        self.unload_calls += 1


class TextEncoderUnloadTestCase(unittest.TestCase):
    def test_untracked_text_embed_caches_are_cleared(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = types.SimpleNamespace(model_type="full", train_text_encoder=False)
        trainer.accelerator = types.SimpleNamespace(is_main_process=True)
        trainer.model = DummyModel()
        trainer.stats_memory_used = lambda: 0.0
        trainer._report_cuda_usage = lambda *_args, **_kwargs: None
        trainer._clear_pipeline_caches = lambda: None

        cache = types.SimpleNamespace(text_encoders=["encoder"], pipeline="pipeline")

        with (
            patch("simpletuner.helpers.training.trainer.StateTracker.get_data_backends", return_value={}),
            patch("simpletuner.helpers.training.trainer.StateTracker.any_text_cache_uses_ondemand", return_value=False),
            patch("simpletuner.helpers.training.trainer.reclaim_memory") as mock_reclaim,
            patch("simpletuner.helpers.training.trainer.TextEmbeddingCache.active_caches", return_value=[cache]),
        ):
            trainer.init_unload_text_encoder()

        self.assertIsNone(cache.text_encoders)
        self.assertIsNone(cache.pipeline)
        self.assertEqual(trainer.model.unload_calls, 1)
        mock_reclaim.assert_called_once()

    def test_global_ondemand_keeps_text_encoders_loaded(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = types.SimpleNamespace(
            model_type="full",
            train_text_encoder=False,
            text_cache_ondemand=True,
            text_cache_disable=False,
        )
        trainer.model = DummyModel()

        with patch("simpletuner.helpers.training.trainer.StateTracker.any_text_cache_uses_ondemand", return_value=False):
            trainer.init_unload_text_encoder()

        self.assertEqual(trainer.model.unload_calls, 0)

    def test_backend_ondemand_keeps_text_encoders_loaded(self):
        trainer = Trainer.__new__(Trainer)
        trainer.config = types.SimpleNamespace(
            model_type="full",
            train_text_encoder=False,
            text_cache_ondemand=False,
            text_cache_disable=False,
        )
        trainer.model = DummyModel()

        with patch("simpletuner.helpers.training.trainer.StateTracker.any_text_cache_uses_ondemand", return_value=True):
            trainer.init_unload_text_encoder()

        self.assertEqual(trainer.model.unload_calls, 0)

    def test_validation_cleanup_keeps_text_encoders_for_backend_ondemand(self):
        validation = Validation.__new__(Validation)
        validation.config = types.SimpleNamespace(text_cache_ondemand=False, text_cache_disable=False)
        validation.model = DummyModel()

        with patch("simpletuner.helpers.training.validation.StateTracker.any_text_cache_uses_ondemand", return_value=True):
            validation.clear_text_encoders()

        self.assertEqual(validation.model.unload_calls, 0)


if __name__ == "__main__":
    unittest.main()
