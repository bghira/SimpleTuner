import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.models.wan.model import Wan


class WanModelTests(unittest.TestCase):
    def test_special_scheduler_setup_loads_pipeline_scheduler(self):
        model = object.__new__(Wan)
        model.config = SimpleNamespace(flow_schedule_shift=5.0)
        model._model_config_path = lambda: "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        scheduler = object()

        with (
            patch(
                "simpletuner.helpers.models.wan.model.FlowMatchEulerDiscreteScheduler.from_pretrained",
                return_value=scheduler,
            ) as from_pretrained,
            patch(
                "simpletuner.helpers.models.wan.model.fix_flow_match_euler_schedule_bounds",
                side_effect=lambda value: value,
            ) as fix_bounds,
        ):
            result = model._load_scheduler_for_pipeline("text2img")

        self.assertIs(result, scheduler)
        from_pretrained.assert_called_once_with(
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
            subfolder="scheduler",
            shift=5.0,
        )
        fix_bounds.assert_called_once_with(scheduler)


if __name__ == "__main__":
    unittest.main()
