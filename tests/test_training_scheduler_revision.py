import unittest
from types import SimpleNamespace
from unittest.mock import patch

from diffusers import DDPMScheduler, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler

from simpletuner.helpers.models.common import ModelFoundation, PredictionTypes
from simpletuner.helpers.models.cosmos3.model import Cosmos3Image
from simpletuner.helpers.models.sana.model import Sana
from simpletuner.helpers.models.sanavideo.model import SanaVideo


class TrainingSchedulerRevisionTests(unittest.TestCase):
    def _model(self, revision, prediction_type=PredictionTypes.FLOW_MATCHING):
        return SimpleNamespace(
            config=SimpleNamespace(
                model_family="flux",
                pretrained_model_name_or_path="repo/model",
                revision=revision,
                flow_schedule_shift=3.0,
                rescale_betas_zero_snr=True,
                training_scheduler_timestep_spacing="trailing",
                prediction_type=None,
            ),
            PREDICTION_TYPE=prediction_type,
            uses_noise_schedule=lambda: True,
        )

    def test_shared_training_scheduler_loads_use_model_revision(self):
        cases = (
            (PredictionTypes.FLOW_MATCHING, FlowMatchEulerDiscreteScheduler, {"shift": 3.0}),
            (PredictionTypes.FLOW_MAP, FlowMatchEulerDiscreteScheduler, {"shift": 3.0}),
            (PredictionTypes.EPSILON, DDPMScheduler, {"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"}),
            (PredictionTypes.V_PREDICTION, DDPMScheduler, {"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"}),
            (PredictionTypes.SAMPLE, DDPMScheduler, {"rescale_betas_zero_snr": True, "timestep_spacing": "trailing"}),
        )
        for prediction_type, scheduler_class, overrides in cases:
            for revision in ("pinned-model-commit", None):
                with self.subTest(prediction_type=prediction_type, revision=revision):
                    model = self._model(revision, prediction_type)
                    loaded = scheduler_class(num_train_timesteps=10)
                    with patch.object(scheduler_class, "from_pretrained", return_value=loaded) as loader:
                        config, scheduler = ModelFoundation.setup_training_noise_schedule(model)

                    loader.assert_called_once_with("repo/model", subfolder="scheduler", revision=revision, **overrides)
                    self.assertIs(config, model.config)
                    self.assertIs(scheduler, loaded)
                    self.assertIs(model.noise_schedule, loaded)
                    if scheduler_class is DDPMScheduler:
                        self.assertEqual(config.prediction_type, loaded.config.prediction_type)

    def test_training_scheduler_overrides_use_model_revision(self):
        cases = (
            (Sana, FlowMatchEulerDiscreteScheduler),
            (SanaVideo, FlowMatchEulerDiscreteScheduler),
            (Cosmos3Image, UniPCMultistepScheduler),
        )
        for model_class, scheduler_class in cases:
            for revision in ("pinned-model-commit", None):
                with self.subTest(model=model_class.__name__, revision=revision):
                    model = self._model(revision)
                    loaded = scheduler_class(num_train_timesteps=10)
                    with patch.object(scheduler_class, "from_pretrained", return_value=loaded) as loader:
                        config, scheduler = model_class.setup_training_noise_schedule(model)

                    loader.assert_called_once_with("repo/model", subfolder="scheduler", revision=revision)
                    self.assertIs(config, model.config)
                    self.assertIs(scheduler, loaded)


if __name__ == "__main__":
    unittest.main()
