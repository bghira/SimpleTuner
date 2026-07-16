import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.models.common import PipelineTypes, VideoModelFoundation
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
            validation_num_video_frames=81,
            wan_validation_load_other_stage=False,
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

    def _stage_model(self, flavour: str, *, load_other: bool):
        model = object.__new__(Wan)
        model.config = self._animegen_config(flavour)
        model.config.wan_validation_load_other_stage = load_other
        model._wan_expand_timesteps = False
        model._wan_cached_stage_modules = {}
        model.unwrap_model = MagicMock(side_effect=lambda model=None, **kwargs: model)
        return model

    def test_high_stage_validation_loads_low_stage_as_transformer_2(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=True)
        pipeline = SimpleNamespace(config=SimpleNamespace(), transformer="trained-high", transformer_2=None)
        other_stage = object()
        model._get_or_load_wan_stage_module = MagicMock(return_value=other_stage)

        with patch.object(VideoModelFoundation, "get_pipeline", return_value=pipeline):
            result = Wan.get_pipeline(model, PipelineTypes.IMG2VIDEO, load_base_model=False)

        self.assertIs(result, pipeline)
        self.assertEqual(pipeline.transformer, "trained-high")
        self.assertIs(pipeline.transformer_2, other_stage)
        self.assertEqual(pipeline.config.boundary_ratio, 0.90)
        model._get_or_load_wan_stage_module.assert_called_once_with("transformer", None)

    def test_low_stage_validation_loads_high_stage_as_transformer(self):
        model = self._stage_model("i2v-14b-2.2-low", load_other=True)
        pipeline = SimpleNamespace(config=SimpleNamespace(), transformer="trained-low", transformer_2=None)
        other_stage = object()
        model._get_or_load_wan_stage_module = MagicMock(return_value=other_stage)

        with patch.object(VideoModelFoundation, "get_pipeline", return_value=pipeline):
            result = Wan.get_pipeline(model, PipelineTypes.IMG2VIDEO, load_base_model=False)

        self.assertIs(result, pipeline)
        self.assertIs(pipeline.transformer, other_stage)
        self.assertEqual(pipeline.transformer_2, "trained-low")
        self.assertEqual(pipeline.config.boundary_ratio, 0.90)
        model._get_or_load_wan_stage_module.assert_called_once_with("transformer_2", None)

    def test_single_stage_validation_does_not_load_other_stage(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=False)
        pipeline = SimpleNamespace(config=SimpleNamespace(), transformer="trained-high", transformer_2="stale")
        model._get_or_load_wan_stage_module = MagicMock()

        with patch.object(VideoModelFoundation, "get_pipeline", return_value=pipeline):
            Wan.get_pipeline(model, PipelineTypes.IMG2VIDEO, load_base_model=False)

        self.assertEqual(pipeline.transformer, "trained-high")
        self.assertIsNone(pipeline.transformer_2)
        self.assertIsNone(pipeline.config.boundary_ratio)
        model._get_or_load_wan_stage_module.assert_not_called()

    def test_non_validation_pipeline_does_not_load_other_stage(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=True)
        pipeline = SimpleNamespace(config=SimpleNamespace(), transformer="trained-high", transformer_2="stale")
        model._get_or_load_wan_stage_module = MagicMock()

        with patch.object(VideoModelFoundation, "get_pipeline", return_value=pipeline):
            Wan.get_pipeline(model, PipelineTypes.IMG2VIDEO, load_base_model=True)

        self.assertEqual(pipeline.transformer, "trained-high")
        self.assertIsNone(pipeline.transformer_2)
        self.assertIsNone(pipeline.config.boundary_ratio)
        model._get_or_load_wan_stage_module.assert_not_called()

    def test_update_pipeline_call_kwargs_includes_peer_stage_guidance(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=True)

        result = Wan.update_pipeline_call_kwargs(model, {"image": "frame"})

        self.assertEqual(result["num_inference_steps"], 40)
        self.assertEqual(result["guidance_scale"], 3.5)
        self.assertEqual(result["guidance_scale_2"], 3.5)
        self.assertEqual(result["output_type"], "pil")

    def test_wan_multistage_validation_support_tracks_peer_stage_loading(self):
        self.assertTrue(self._stage_model("i2v-14b-2.2-high", load_other=True).supports_multistage_validation())
        self.assertFalse(self._stage_model("i2v-14b-2.2-high", load_other=False).supports_multistage_validation())
        self.assertFalse(self._stage_model("t2v-480p-1.3b-2.1", load_other=True).supports_multistage_validation())

    def test_wan_run_multistage_validation_uses_single_pipeline(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=True)
        model.pipeline = object()
        calls = []

        result = Wan.run_multistage_validation(
            model,
            {"prompt_embeds": "embeds"},
            lambda pipeline, kwargs, target_stage=None: calls.append((pipeline, kwargs, target_stage)) or "result",
        )

        self.assertEqual(result, "result")
        self.assertEqual(calls, [(model.pipeline, {"prompt_embeds": "embeds"}, ("high", "low"))])

    def test_unload_validation_models_clears_cached_peer_stages(self):
        model = self._stage_model("i2v-14b-2.2-high", load_other=True)
        model._wan_cached_stage_modules["peer"] = object()

        with patch.object(VideoModelFoundation, "unload_validation_models", autospec=True) as super_unload:
            Wan.unload_validation_models(model)

        super_unload.assert_called_once_with(model)
        self.assertEqual(model._wan_cached_stage_modules, {})


if __name__ == "__main__":
    unittest.main()
