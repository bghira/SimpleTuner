import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import simpletuner.helpers.training.validation as validation_module
from simpletuner.helpers.training.validation import Validation


class SchedulerConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class LoadedScheduler:
    def __init__(self, **config):
        self.config = SchedulerConfig(config)


class ConfigBackedScheduler:
    from_config_calls = []
    from_pretrained_calls = []

    def __init__(self, config, load_kwargs):
        self.config = SchedulerConfig(config)
        self.load_kwargs = load_kwargs

    @classmethod
    def reset(cls):
        cls.from_config_calls = []
        cls.from_pretrained_calls = []

    @classmethod
    def from_config(cls, config, **kwargs):
        cls.from_config_calls.append((config, kwargs))
        return cls(config=config, load_kwargs=kwargs)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        raise AssertionError("from_pretrained should not be called when a loaded scheduler config is available")


class RequestedScheduler(ConfigBackedScheduler):
    pass


class LoadedSubclassScheduler(RequestedScheduler):
    pass


class PretrainedScheduler:
    from_pretrained_calls = []

    def __init__(self, path, load_kwargs):
        self.config = SchedulerConfig()
        self.path = path
        self.load_kwargs = load_kwargs

    @classmethod
    def reset(cls):
        cls.from_pretrained_calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls(path=args[0], load_kwargs=kwargs)


class ValidationSchedulerCacheTests(unittest.TestCase):
    def setUp(self):
        ConfigBackedScheduler.reset()
        RequestedScheduler.reset()
        LoadedSubclassScheduler.reset()
        PretrainedScheduler.reset()

    def _validation(self, *, scheduler, scheduler_name="cached", model_path="repo/model"):
        validation = Validation.__new__(Validation)
        validation.distiller = None
        validation.config = SimpleNamespace(
            model_family="sdxl",
            pretrained_model_name_or_path=model_path,
            validation_noise_scheduler=scheduler_name,
            prediction_type="epsilon",
            inference_scheduler_timestep_spacing="trailing",
            rescale_betas_zero_snr=True,
            revision="main",
            cache_dir="/tmp/hf-cache",
            local_files_only=True,
        )
        validation.model = SimpleNamespace(
            DEFAULT_NOISE_SCHEDULER=None,
            PREDICTION_TYPE=SimpleNamespace(value="epsilon"),
            pipeline=SimpleNamespace(scheduler=scheduler),
            noise_schedule=None,
            requires_special_scheduler_setup=lambda: False,
        )
        return validation

    def test_setup_scheduler_clones_loaded_pipeline_scheduler_config(self):
        loaded_scheduler = LoadedScheduler(variance_type="learned")
        validation = self._validation(scheduler=loaded_scheduler)

        with patch.dict(validation_module.SCHEDULER_NAME_MAP, {"cached": ConfigBackedScheduler}):
            scheduler = validation.setup_scheduler()

        self.assertIs(validation.model.pipeline.scheduler, scheduler)
        self.assertEqual(ConfigBackedScheduler.from_pretrained_calls, [])
        self.assertEqual(len(ConfigBackedScheduler.from_config_calls), 1)
        _config, kwargs = ConfigBackedScheduler.from_config_calls[0]
        self.assertEqual(kwargs["prediction_type"], "epsilon")
        self.assertEqual(kwargs["variance_type"], "fixed_small")
        self.assertEqual(kwargs["timestep_spacing"], "trailing")
        self.assertTrue(kwargs["rescale_betas_zero_snr"])

    def test_setup_scheduler_uses_loaded_noise_schedule_when_pipeline_is_absent(self):
        validation = self._validation(scheduler=None)
        validation.model.pipeline = None
        validation.model.noise_schedule = LoadedScheduler(variance_type="fixed_small")

        with patch.dict(validation_module.SCHEDULER_NAME_MAP, {"cached": ConfigBackedScheduler}):
            scheduler = validation.setup_scheduler()

        self.assertIsInstance(scheduler, ConfigBackedScheduler)
        self.assertEqual(ConfigBackedScheduler.from_pretrained_calls, [])
        self.assertEqual(len(ConfigBackedScheduler.from_config_calls), 1)

    def test_setup_scheduler_preserves_loaded_scheduler_subclass(self):
        loaded_scheduler = LoadedSubclassScheduler(config=SchedulerConfig(variance_type="fixed_small"), load_kwargs={})
        validation = self._validation(scheduler=loaded_scheduler, scheduler_name="requested")

        with patch.dict(validation_module.SCHEDULER_NAME_MAP, {"requested": RequestedScheduler}):
            scheduler = validation.setup_scheduler()

        self.assertIsInstance(scheduler, LoadedSubclassScheduler)
        self.assertEqual(RequestedScheduler.from_pretrained_calls, [])
        self.assertEqual(len(RequestedScheduler.from_config_calls), 0)
        self.assertEqual(len(LoadedSubclassScheduler.from_config_calls), 1)

    def test_fallback_from_pretrained_uses_resolved_path_and_cache_flags(self):
        validation = self._validation(scheduler=None, scheduler_name="pretrained", model_path="repo/model.safetensors")
        validation.model.pipeline = None
        validation.model._model_config_path = MagicMock(return_value="/tmp/model-cache/snapshot")

        with patch.dict(validation_module.SCHEDULER_NAME_MAP, {"pretrained": PretrainedScheduler}):
            scheduler = validation.setup_scheduler()

        self.assertIsInstance(scheduler, PretrainedScheduler)
        validation.model._model_config_path.assert_called_once_with()
        self.assertEqual(len(PretrainedScheduler.from_pretrained_calls), 1)
        args, kwargs = PretrainedScheduler.from_pretrained_calls[0]
        self.assertEqual(args, ("/tmp/model-cache/snapshot",))
        self.assertEqual(kwargs["subfolder"], "scheduler")
        self.assertEqual(kwargs["revision"], "main")
        self.assertEqual(kwargs["timestep_spacing"], "trailing")
        self.assertTrue(kwargs["rescale_betas_zero_snr"])
        self.assertEqual(kwargs["cache_dir"], "/tmp/hf-cache")
        self.assertTrue(kwargs["local_files_only"])
        self.assertEqual(kwargs["prediction_type"], "epsilon")


if __name__ == "__main__":
    unittest.main()
