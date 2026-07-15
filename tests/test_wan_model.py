import unittest
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.models.wan.model import Wan


class WanModelTests(unittest.TestCase):
    def _animegen_config(self, flavour: str):
        return SimpleNamespace(
            model_family="wan",
            model_flavour=flavour,
            pretrained_model_name_or_path=None,
            pretrained_vae_model_name_or_path=None,
            pretrained_transformer_model_name_or_path=None,
            pretrained_transformer_subfolder="transformer",
            vae_path=None,
            flow_schedule_shift=5.0,
            validation_num_inference_steps=40,
            validation_guidance=3.5,
        )

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

    def test_animegen_high_flavour_uses_high_noise_single_file_stage(self):
        model = object.__new__(Wan)
        model.config = self._animegen_config("animegen-t2v-high")

        Wan.setup_model_flavour(model)

        self.assertEqual(model.config.pretrained_model_name_or_path, Wan.WAN22_T2V_A14B_PATH)
        self.assertEqual(model.config.pretrained_transformer_model_name_or_path, Wan.ANIMEGEN_T2V_HIGH_PATH)
        self.assertIsNone(model.config.pretrained_transformer_subfolder)
        self.assertEqual(model.config.wan_trained_stage, "high")
        self.assertIsNone(model.config.wan_stage_other_subfolder)
        self.assertEqual(model.config.flow_schedule_shift, 3.0)
        self.assertEqual(model.config.validation_num_inference_steps, 8)
        self.assertEqual(model.config.validation_guidance, 1.0)
        self.assertEqual(model.config.wan_boundary_ratio, 0.875)

    def test_animegen_low_flavour_uses_low_noise_single_file_stage(self):
        model = object.__new__(Wan)
        model.config = self._animegen_config("animegen-t2v-low")

        Wan.setup_model_flavour(model)

        self.assertEqual(model.config.pretrained_model_name_or_path, Wan.WAN22_T2V_A14B_PATH)
        self.assertEqual(model.config.pretrained_transformer_model_name_or_path, Wan.ANIMEGEN_T2V_LOW_PATH)
        self.assertIsNone(model.config.pretrained_transformer_subfolder)
        self.assertEqual(model.config.wan_trained_stage, "low")
        self.assertIsNone(model.config.wan_stage_other_subfolder)
        self.assertEqual(model.config.flow_schedule_shift, 3.0)
        self.assertEqual(model.config.validation_num_inference_steps, 8)
        self.assertEqual(model.config.validation_guidance, 1.0)
        self.assertEqual(model.config.wan_boundary_ratio, 0.875)

    def test_animegen_flavours_are_model_flavour_choices(self):
        choices = Wan.get_flavour_choices()

        self.assertIn("animegen-t2v-high", choices)
        self.assertIn("animegen-t2v-low", choices)


if __name__ == "__main__":
    unittest.main()
