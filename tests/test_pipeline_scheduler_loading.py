import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler

from simpletuner.helpers.models.ltxvideo.model import LTXVideo
from simpletuner.helpers.models.sana.model import Sana
from simpletuner.helpers.models.sanavideo.model import SanaVideo


class TestPipelineSchedulerLoading(unittest.TestCase):
    def test_lightweight_pipelines_load_checkpoint_schedulers(self):
        cases = (
            ("sana", Sana, DPMSolverMultistepScheduler),
            ("sanavideo", SanaVideo, DPMSolverMultistepScheduler),
            ("ltxvideo", LTXVideo, FlowMatchEulerDiscreteScheduler),
        )
        for family, model_class, scheduler_class in cases:
            with self.subTest(model=family), tempfile.TemporaryDirectory() as checkpoint:
                scheduler_class(num_train_timesteps=32).save_pretrained(Path(checkpoint) / "scheduler")
                model = model_class.__new__(model_class)
                model.config = SimpleNamespace(
                    model_family=family,
                    pretrained_model_name_or_path=checkpoint,
                    revision=None,
                    local_files_only=True,
                    controlnet=False,
                )
                model.model = None
                model.vae = None
                model.get_vae = Mock(return_value=None)
                model.text_encoders = None
                model.tokenizers = None
                model.pipelines = {}
                model.noise_schedule = FlowMatchEulerDiscreteScheduler(num_train_timesteps=64)

                pipeline = model.get_pipeline(load_base_model=False)

                self.assertIsInstance(pipeline.scheduler, scheduler_class)
                self.assertEqual(pipeline.scheduler.config.num_train_timesteps, 32)
                self.assertEqual(model.noise_schedule.config.num_train_timesteps, 64)
