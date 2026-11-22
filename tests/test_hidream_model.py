import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedTokenizerFast

from simpletuner.helpers.models.hidream.controlnet import HiDreamControlNetModel, HiDreamControlNetPipeline
from simpletuner.helpers.models.hidream.model import HiDream
from simpletuner.helpers.models.hidream.transformer import HiDreamImageTransformer2DModel
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class HiDreamModelTests(unittest.TestCase):
    def setUp(self):
        self.model = HiDream.__new__(HiDream)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            input_perturbation=0.0,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            crepa_self_flow=False,
            crepa_feature_source=None,
            loss_type="l2",
            huber_schedule="constant",
            huber_c=0.1,
        )
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped
        self.model.model = MagicMock(config=SimpleNamespace(patch_size=2, max_seq=4))
        self.model.diff2flow_bridge = None

    def _enable_xm(self, candidate_count: int = 2):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )

    def _make_controlnet(self, *, use_shared_modules=True):
        transformer = HiDreamImageTransformer2DModel(
            patch_size=2,
            in_channels=2,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            caption_channels=[8, 8],
            text_emb_dim=8,
            num_routed_experts=0,
            axes_dims_rope=(2, 2, 4),
            max_resolution=(8, 8),
            llama_layers=[0, 1],
        )
        transformer.final_layer.linear.reset_parameters()
        transformer.register_to_config(pooled_projection_dim=8)
        return transformer, HiDreamControlNetModel.from_transformer(transformer, use_shared_modules=use_shared_modules)

    def test_controlnet_predict_aligns_residuals_for_rectangular_batch(self):
        self.model.model, self.model.controlnet = self._make_controlnet()
        projections = [self.model.controlnet.controlnet_blocks[0], self.model.controlnet.controlnet_single_blocks[0]]
        with torch.no_grad():
            for projection in projections:
                projection.bias.fill_(1)

        residuals = []
        self.model.controlnet.register_forward_hook(lambda _module, _args, outputs: residuals.extend(outputs))
        latents = torch.randn(2, 2, 4, 6)
        batch = {
            "noisy_latents": latents,
            "conditioning_latents": torch.randn_like(latents),
            "timesteps": torch.tensor([250.0, 750.0]),
            "text_encoder_output": {
                "t5_prompt_embeds": torch.randn(2, 3, 8),
                "llama_prompt_embeds": torch.randn(2, 2, 3, 8),
                "pooled_prompt_embeds": torch.randn(2, 8),
            },
        }

        prediction = self.model.controlnet_predict(batch)["model_prediction"]

        self.assertEqual(prediction.shape, latents.shape)
        for samples in residuals:
            sample = samples[0]
            self.assertEqual(sample.shape, (2, self.model.model.max_seq, self.model.model.inner_dim))
            torch.testing.assert_close(sample[:, :6], torch.ones_like(sample[:, :6]))
            torch.testing.assert_close(sample[:, 6:], torch.zeros_like(sample[:, 6:]))
        prediction.square().mean().backward()
        for projection in projections:
            self.assertGreater(projection.weight.grad.abs().sum().item(), 0)

    def test_controlnet_checkpoints_blocks_and_only_projects_selected_llama_layers(self):
        _, controlnet = self._make_controlnet()
        with torch.no_grad():
            for projection in (*controlnet.controlnet_blocks, *controlnet.controlnet_single_blocks):
                projection.weight.normal_()

        inputs = dict(
            hidden_states=torch.randn(1, 2, 4, 6),
            controlnet_cond=torch.randn(1, 2, 4, 6),
            timesteps=torch.tensor([0.5]),
            t5_hidden_states=torch.randn(1, 3, 8),
            llama_hidden_states=torch.randn(4, 1, 3, 8),
            pooled_embeds=torch.randn(1, 8),
            return_dict=False,
        )

        def run():
            controlnet.zero_grad(set_to_none=True)
            outputs = controlnet(**inputs)
            sum(sample.square().mean() for stream in outputs for sample in stream).backward()
            return outputs, {name: param.grad for name, param in controlnet.named_parameters() if param.grad is not None}

        eager_outputs, eager_gradients = run()
        controlnet.enable_gradient_checkpointing(lambda block, *args: checkpoint(block, *args, use_reentrant=False))
        with (
            patch.object(controlnet.llama_embedder, "forward", wraps=controlnet.llama_embedder.forward) as project_llama,
            patch.object(
                controlnet, "_gradient_checkpointing_func", wraps=controlnet._gradient_checkpointing_func
            ) as checkpoint_block,
        ):
            checkpoint_outputs, checkpoint_gradients = run()

        self.assertEqual(project_llama.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in checkpoint_block.call_args_list],
            [controlnet.double_stream_blocks[0], controlnet.single_stream_blocks[0]],
        )
        torch.testing.assert_close(checkpoint_outputs, eager_outputs)
        torch.testing.assert_close(checkpoint_gradients, eager_gradients)

    def test_controlnet_checkpoints_frozen_eval_transformer_for_residual_gradients(self):
        transformer, controlnet = self._make_controlnet(use_shared_modules=False)
        self.model.config.controlnet = True
        self.model.model = transformer.requires_grad_(False).eval()
        self.model.controlnet = controlnet
        control_projection = controlnet.controlnet_blocks[0].weight
        latents = torch.randn(1, 2, 4, 6)
        batch = {
            "noisy_latents": latents,
            "conditioning_latents": torch.randn_like(latents),
            "timesteps": torch.tensor([250.0]),
            "text_encoder_output": {
                "t5_prompt_embeds": torch.randn(1, 3, 8),
                "llama_prompt_embeds": torch.randn(2, 1, 3, 8),
                "pooled_prompt_embeds": torch.randn(1, 8),
            },
        }

        def run():
            controlnet.zero_grad(set_to_none=True)
            prediction = self.model.controlnet_predict(batch)["model_prediction"]
            prediction.square().mean().backward()
            return prediction, control_projection.grad

        expected, expected_gradient = run()
        self.assertGreater(expected_gradient.abs().sum().item(), 0)
        self.model.enable_gradient_checkpointing()
        base_block = transformer.single_stream_blocks[0]
        with patch.object(base_block, "forward", wraps=base_block.forward) as forward:
            actual, actual_gradient = run()

        self.assertFalse(transformer.training)
        self.assertEqual(forward.call_count, 2)
        self.assertTrue(all(param.grad is None for param in transformer.parameters()))
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_controlnet_pipeline_denoises_rectangular_batch(self):
        transformer, controlnet = self._make_controlnet()
        pipeline = HiDreamControlNetPipeline(
            transformer=transformer.eval(),
            controlnet=controlnet.eval(),
            scheduler=FlowMatchEulerDiscreteScheduler(),
            vae=AutoencoderKL(
                block_out_channels=(8,),
                layers_per_block=1,
                latent_channels=2,
                norm_num_groups=1,
                shift_factor=0.0,
            ).eval(),
            text_encoder=None,
            text_encoder_2=None,
            text_encoder_3=None,
            text_encoder_4=None,
            tokenizer=None,
            tokenizer_2=None,
            tokenizer_3=None,
            tokenizer_4=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"</s>": 0})), eos_token="</s>"),
        )
        pipeline.set_progress_bar_config(disable=True)
        t5 = torch.randn(2, 3, 8)
        llama = torch.randn(2, 2, 3, 8)
        pooled = torch.randn(2, 8)

        def check_controlnet_inputs(_module, _args, kwargs):
            hidden_states = kwargs["hidden_states"]
            self.assertEqual(hidden_states.shape[-2:], (4, 6))
            self.assertEqual(hidden_states.shape, kwargs["controlnet_cond"].shape)

        controlnet.register_forward_pre_hook(check_controlnet_inputs, with_kwargs=True)
        for guidance_scale in (1.0, 3.0):
            with self.subTest(guidance_scale=guidance_scale):
                latents = pipeline(
                    control_image=torch.randn(2, 3, 4, 6),
                    t5_prompt_embeds=t5,
                    llama_prompt_embeds=llama,
                    pooled_prompt_embeds=pooled,
                    negative_t5_prompt_embeds=torch.zeros_like(t5),
                    negative_llama_prompt_embeds=torch.zeros_like(llama),
                    negative_pooled_prompt_embeds=torch.zeros_like(pooled),
                    guidance_scale=guidance_scale,
                    num_inference_steps=1,
                    output_type="latent",
                ).images

                self.assertEqual(latents.shape, (2, 2, 4, 6))
                self.assertTrue(torch.isfinite(latents).all())

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.config.crepa_self_flow_mask_ratio = 0.5
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([750.0], dtype=torch.float32), torch.tensor([750.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4),
            "input_noise": torch.ones(1, 16, 4, 4),
            "sigmas": torch.tensor([250.0], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 4]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))
        self.assertEqual(updated["crepa_teacher_noisy_latents"].shape, torch.Size([1, 16, 4, 4]))

    def test_model_predict_accepts_tokenwise_timesteps_and_capture_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 16, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(patch_size=2, max_seq=4))

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 250.0, 750.0]], dtype=torch.float32),
            "text_encoder_output": {
                "t5_prompt_embeds": torch.randn(1, 2, 16),
                "llama_prompt_embeds": torch.randn(2, 1, 2, 16),
                "pooled_prompt_embeds": torch.randn(1, 16),
            },
            "crepa_capture_block_index": 7,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timesteps"], prepared_batch["timesteps"]))

    def test_check_user_config_keeps_bundled_quanto_text_encoder_with_quanto_base(self):
        self.model.config = SimpleNamespace(
            base_model_precision="int8-quanto",
            text_encoder_4_precision="int4-quanto",
            tokenizer_max_length=0,
            i_know_what_i_am_doing=False,
            aspect_bucket_alignment=32,
        )

        self.model.check_user_config()

        self.assertEqual(self.model.config.text_encoder_4_precision, "int4-quanto")

    def test_check_user_config_disables_bundled_quanto_text_encoder_for_torchao_base(self):
        self.model.config = SimpleNamespace(
            base_model_precision="fp8-torchao",
            text_encoder_4_precision="int4-quanto",
            tokenizer_max_length=0,
            i_know_what_i_am_doing=False,
            aspect_bucket_alignment=32,
        )

        self.model.check_user_config()

        self.assertEqual(self.model.config.text_encoder_4_precision, "no_change")

    def test_check_user_config_disables_bundled_quanto_text_encoder_for_sdnq_base(self):
        self.model.config = SimpleNamespace(
            base_model_precision="int8-sdnq",
            text_encoder_4_precision="int4-quanto",
            tokenizer_max_length=0,
            i_know_what_i_am_doing=False,
            aspect_bucket_alignment=32,
        )

        self.model.check_user_config()

        self.assertEqual(self.model.config.text_encoder_4_precision, "no_change")

    def test_check_user_config_rejects_non_default_mixed_text_encoder_backend(self):
        self.model.config = SimpleNamespace(
            base_model_precision="fp8-torchao",
            text_encoder_4_precision="int8-quanto",
            tokenizer_max_length=0,
            i_know_what_i_am_doing=False,
            aspect_bucket_alignment=32,
        )

        with self.assertRaisesRegex(ValueError, "cannot mix base model precision"):
            self.model.check_user_config()

    def test_xm_noise_candidates_expand_text_encoder_output_candidate_major(self):
        self._enable_xm(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 1, 2, 2),
            "noise": torch.zeros(2, 1, 2, 2),
            "input_noise": torch.zeros(2, 1, 2, 2),
            "noisy_latents": torch.zeros(2, 1, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "text_encoder_output": {
                "t5_prompt_embeds": torch.randn(2, 4, 16),
                "llama_prompt_embeds": torch.randn(2, 3, 16),
                "pooled_prompt_embeds": torch.randn(2, 16),
            },
        }

        self.model._prepare_xm_noise_candidates(batch, family_name="HiDream")

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertEqual(tuple(batch["text_encoder_output"]["t5_prompt_embeds"].shape), (6, 4, 16))
        expected_noisy = 0.75 * batch["latents"] + 0.25 * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))

    def test_xm_loss_selects_winners_and_trims_hidream_outputs(self):
        self._enable_xm(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_xm_rejects_crepa_self_flow(self):
        self._enable_xm(candidate_count=2)
        self.model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            self.model._validate_xm_support()


if __name__ == "__main__":
    unittest.main()
