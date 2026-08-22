import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.hidream.model import HiDream
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
