import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PipelineTypes, VideoModelFoundation
from simpletuner.helpers.models.wan.model import Wan, add_first_frame_latent_conditioning, time_text_monkeypatch
from simpletuner.helpers.models.wan.transformer import WanTimeTextImageEmbedding


class _RecordingWanTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        r_timestep=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "r_timestep": r_timestep,
            **kwargs,
        }
        return (torch.zeros_like(hidden_states),)


class WanModelTests(unittest.TestCase):
    def test_flowmap_gate_survives_ddp_buffer_broadcast_during_accumulation(self):
        for use_monkeypatch in (False, True):
            with self.subTest(use_monkeypatch=use_monkeypatch):
                embedder = WanTimeTextImageEmbedding(
                    dim=8,
                    time_freq_dim=4,
                    time_proj_dim=48,
                    text_embed_dim=6,
                )
                embedder.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
                if use_monkeypatch:
                    embedder.forward = partial(time_text_monkeypatch, embedder)

                timestep = torch.tensor([0.75])
                r_timestep = torch.tensor([0.25])
                encoder_hidden_states = torch.randn(1, 2, 6)
                first_temb = embedder(timestep, encoder_hidden_states, r_timestep=r_timestep)[0]

                # DDP broadcasts model buffers in place before each forward.
                with torch.no_grad():
                    embedder.flowmap_delta_emb_gate.copy_(embedder.flowmap_delta_emb_gate)

                second_temb = embedder(timestep, encoder_hidden_states, r_timestep=r_timestep)[0]
                (first_temb.sum() + second_temb.sum()).backward()

                self.assertIsNotNone(embedder.delta_embedder.linear_1.weight.grad)

    def _lora_target_model(self, *, distillation_method=None, anyflow_config=None):
        model = object.__new__(Wan)
        model.config = SimpleNamespace(
            distillation_method=distillation_method,
            distillation_config={"anyflow": anyflow_config or {}},
            lora_type="standard",
            peft_lora_target_modules=None,
            slider_lora_target=False,
            controlnet=False,
        )
        return model

    def test_anyflow_lora_targets_match_wan_reference_scope(self):
        model = self._lora_target_model(distillation_method="anyflow")

        self.assertEqual(
            model.get_lora_target_layers(),
            [
                "attn1.to_q",
                "attn1.to_k",
                "attn1.to_v",
                "attn1.to_out.0",
                "ffn.net.0.proj",
                "ffn.net.2",
                "condition_embedder.time_embedder.linear_1",
                "condition_embedder.time_embedder.linear_2",
                "condition_embedder.delta_embedder.linear_1",
                "condition_embedder.delta_embedder.linear_2",
            ],
        )

    def test_anyflow_lora_targets_respect_time_embedder_flags(self):
        model = self._lora_target_model(
            distillation_method="anyflow",
            anyflow_config={"train_time_embedder": False, "train_delta_embedder": False},
        )

        targets = model.get_lora_target_layers()

        self.assertIn("ffn.net.0.proj", targets)
        self.assertFalse(any("time_embedder" in target for target in targets))

    def test_standard_wan_lora_targets_are_unchanged(self):
        model = self._lora_target_model()

        self.assertEqual(model.get_lora_target_layers(), Wan.DEFAULT_LORA_TARGET)

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

    def test_latent_i2v_conditioning_builds_36_channel_input(self):
        latent_model_input = torch.zeros(1, 16, 3, 4, 5)
        clean_latents = torch.arange(1 * 16 * 3 * 4 * 5, dtype=torch.float32).view(1, 16, 3, 4, 5)
        vae = SimpleNamespace(config=SimpleNamespace(temperal_downsample=[1, 1]))

        conditioned = add_first_frame_latent_conditioning(latent_model_input, clean_latents, vae)

        self.assertEqual(tuple(conditioned.shape), (1, 36, 3, 4, 5))
        self.assertTrue(torch.equal(conditioned[:, :16], latent_model_input))
        self.assertTrue(torch.all(conditioned[:, 16:20, 0] == 1))
        self.assertTrue(torch.all(conditioned[:, 16:20, 1:] == 0))
        self.assertTrue(torch.equal(conditioned[:, 20:, :1], clean_latents[:, :, :1]))
        self.assertTrue(torch.all(conditioned[:, 20:, 1:] == 0))

    def test_latent_i2v_conditioning_uses_config_temporal_downsample(self):
        latent_model_input = torch.zeros(1, 16, 3, 4, 5)
        clean_latents = torch.zeros(1, 16, 3, 4, 5)
        vae = SimpleNamespace(config=SimpleNamespace(temperal_downsample=[0, 1]))

        conditioned = add_first_frame_latent_conditioning(latent_model_input, clean_latents, vae)

        self.assertEqual(tuple(conditioned.shape), (1, 34, 3, 4, 5))
        self.assertTrue(torch.all(conditioned[:, 16:18, 0] == 1))
        self.assertTrue(torch.all(conditioned[:, 16:18, 1:] == 0))

    def test_i2v_conditioning_uses_cached_latents_without_warning(self):
        model = object.__new__(Wan)
        model._is_i2v_like_flavour = MagicMock(return_value=False)
        model._extract_conditioning_frames = MagicMock(return_value=(None, None))
        model.get_vae = MagicMock(return_value=SimpleNamespace(config=SimpleNamespace(temperal_downsample=[1, 1])))

        hidden_states = torch.zeros(1, 16, 3, 4, 5)
        clean_latents = torch.arange(1 * 16 * 3 * 4 * 5, dtype=torch.float32).view(1, 16, 3, 4, 5)
        transformer_kwargs = {"hidden_states": hidden_states.clone()}
        prepared_batch = {"is_i2v_data": True, "latents": clean_latents}

        with patch("simpletuner.helpers.models.wan.model.logger.warning") as warning:
            Wan._apply_i2v_conditioning_to_kwargs(model, prepared_batch, transformer_kwargs)

        conditioned = transformer_kwargs["hidden_states"]
        self.assertEqual(tuple(conditioned.shape), (1, 36, 3, 4, 5))
        self.assertTrue(torch.equal(conditioned[:, :16], hidden_states))
        self.assertTrue(torch.equal(conditioned[:, 20:, :1], clean_latents[:, :, :1]))
        self.assertTrue(torch.all(conditioned[:, 20:, 1:] == 0))
        warning.assert_not_called()

    def test_model_predict_forwards_anyflow_r_timestep(self):
        model = object.__new__(Wan)
        transformer = _RecordingWanTransformer()
        model.model = transformer
        model.config = SimpleNamespace(
            controlnet=False,
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            tread_config=None,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.crepa_regularizer = None
        model.unwrap_model = MagicMock(side_effect=lambda model=None, **_: model)
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model._build_grounding_position_net_kwargs = MagicMock(return_value=None)
        model._apply_i2v_conditioning_to_kwargs = MagicMock()
        r_timesteps = torch.tensor([0.25])

        result = Wan.model_predict(
            model,
            {
                "noisy_latents": torch.randn(1, 16, 1, 2, 2),
                "encoder_hidden_states": torch.randn(1, 4, 8),
                "timesteps": torch.tensor([0.75]),
                Wan.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps,
            },
        )

        self.assertIs(transformer.last_kwargs["r_timestep"], r_timesteps)
        self.assertEqual(result["model_prediction"].shape, (1, 16, 1, 2, 2))


if __name__ == "__main__":
    unittest.main()
