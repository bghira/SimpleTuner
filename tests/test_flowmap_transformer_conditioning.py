import unittest
import warnings

import torch
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings

from simpletuner.helpers.models.flowmap import clone_flowmap_embedder, prepare_flowmap_delta_timestep
from simpletuner.helpers.models.flux2.transformer import Flux2TimestepGuidanceEmbeddings
from simpletuner.helpers.models.flux.transformer import _flux_tokenwise_conditioning, _flux_tokenwise_flowmap_conditioning
from simpletuner.helpers.models.sd3.transformer import _sd3_tokenwise_conditioning, _sd3_tokenwise_flowmap_conditioning
from simpletuner.helpers.models.unet_flowmap import FlowMapUNet2DConditionModel


class TestFlowMapTransformerConditioning(unittest.TestCase):
    def test_prepare_delta_timestep_supports_batch_and_tokenwise(self):
        timestep = torch.tensor([0.8, 0.4])
        r_timestep = torch.tensor([0.2])

        delta = prepare_flowmap_delta_timestep(timestep, r_timestep, "r", model_name="Test")
        self.assertTrue(torch.equal(delta, torch.tensor([0.2, 0.2])))

        delta = prepare_flowmap_delta_timestep(timestep, r_timestep, "t-r", model_name="Test")
        self.assertTrue(torch.allclose(delta, torch.tensor([0.6, 0.2])))

        tokenwise_timestep = torch.tensor([[0.8, 0.7], [0.4, 0.3]])
        tokenwise_delta = prepare_flowmap_delta_timestep(
            tokenwise_timestep,
            torch.tensor([0.2, 0.1]),
            "r",
            model_name="Test",
        )
        expected = torch.tensor([[0.2, 0.2], [0.1, 0.1]])
        self.assertTrue(torch.equal(tokenwise_delta, expected))

    def test_flux_flowmap_equal_r_matches_base_conditioning(self):
        torch.manual_seed(0)
        conditioning = CombinedTimestepTextProjEmbeddings(embedding_dim=32, pooled_projection_dim=16)
        delta_embedder = clone_flowmap_embedder(conditioning.timestep_embedder)
        timestep = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=torch.float32)
        pooled = torch.randn(2, 16)
        gate = torch.tensor([0.25])

        base = _flux_tokenwise_conditioning(conditioning, timestep, pooled)
        flowmap = _flux_tokenwise_flowmap_conditioning(
            conditioning,
            delta_embedder,
            timestep,
            pooled,
            timestep,
            "r",
            gate,
        )

        self.assertTrue(torch.allclose(flowmap, base, atol=1e-6))

    def test_flux_flowmap_different_r_changes_conditioning(self):
        torch.manual_seed(1)
        conditioning = CombinedTimestepTextProjEmbeddings(embedding_dim=32, pooled_projection_dim=16)
        delta_embedder = clone_flowmap_embedder(conditioning.timestep_embedder)
        timestep = torch.tensor([0.2, 0.4], dtype=torch.float32)
        pooled = torch.randn(2, 16)
        gate = torch.tensor([0.25])

        base = _flux_tokenwise_conditioning(conditioning, timestep, pooled)
        flowmap = _flux_tokenwise_flowmap_conditioning(
            conditioning,
            delta_embedder,
            timestep,
            pooled,
            torch.zeros_like(timestep),
            "r",
            gate,
        )

        self.assertFalse(torch.allclose(flowmap, base, atol=1e-6))

    def test_sd3_flowmap_equal_r_matches_base_conditioning(self):
        torch.manual_seed(2)
        conditioning = CombinedTimestepTextProjEmbeddings(embedding_dim=32, pooled_projection_dim=16)
        delta_embedder = clone_flowmap_embedder(conditioning.timestep_embedder)
        timestep = torch.tensor([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=torch.float32)
        pooled = torch.randn(2, 16)
        gate = torch.tensor([0.25])

        base = _sd3_tokenwise_conditioning(conditioning, timestep, pooled)
        flowmap = _sd3_tokenwise_flowmap_conditioning(
            conditioning,
            delta_embedder,
            timestep,
            pooled,
            timestep,
            "r",
            gate,
        )

        self.assertTrue(torch.allclose(flowmap, base, atol=1e-6))

    def test_flux2_embedding_requires_enable_for_r_timestep(self):
        embedding = Flux2TimestepGuidanceEmbeddings(in_channels=16, embedding_dim=32, guidance_embeds=True)
        timestep = torch.tensor([0.2, 0.4], dtype=torch.float32)
        guidance = torch.tensor([0.5, 0.6], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "enable_flowmap_time_conditioning"):
            embedding(timestep, guidance, r_timestep=timestep)

    def test_flux2_flowmap_equal_r_matches_base_conditioning(self):
        torch.manual_seed(3)
        embedding = Flux2TimestepGuidanceEmbeddings(in_channels=16, embedding_dim=32, guidance_embeds=True)
        embedding.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        timestep = torch.tensor([0.2, 0.4], dtype=torch.float32)
        guidance = torch.tensor([0.5, 0.6], dtype=torch.float32)

        base = embedding(timestep, guidance)
        flowmap = embedding(timestep, guidance, r_timestep=timestep)

        self.assertTrue(torch.allclose(flowmap, base, atol=1e-6))

    def test_flux2_flowmap_different_r_changes_conditioning(self):
        torch.manual_seed(4)
        embedding = Flux2TimestepGuidanceEmbeddings(in_channels=16, embedding_dim=32, guidance_embeds=True)
        embedding.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
        timestep = torch.tensor([0.2, 0.4], dtype=torch.float32)
        guidance = torch.tensor([0.5, 0.6], dtype=torch.float32)

        base = embedding(timestep, guidance)
        flowmap = embedding(timestep, guidance, r_timestep=torch.zeros_like(timestep))

        self.assertFalse(torch.allclose(flowmap, base, atol=1e-6))

    def test_unet_flowmap_requires_enable_for_r_timestep(self):
        model = self._tiny_unet()
        sample, timestep, encoder_hidden_states = self._tiny_unet_inputs()

        with self.assertRaisesRegex(ValueError, "enable_flowmap_time_conditioning"):
            model(sample, timestep, encoder_hidden_states, r_timestep=timestep)

    def test_unet_flowmap_equal_r_matches_base_output(self):
        torch.manual_seed(5)
        model = self._tiny_unet()
        model.eval()
        sample, timestep, encoder_hidden_states = self._tiny_unet_inputs()

        with torch.no_grad():
            base = model(sample, timestep, encoder_hidden_states).sample
            model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
            flowmap = model(sample, timestep, encoder_hidden_states, r_timestep=timestep).sample

        self.assertTrue(torch.allclose(flowmap, base, atol=1e-6))

    def test_unet_flowmap_different_r_changes_output(self):
        torch.manual_seed(6)
        model = self._tiny_unet()
        model.eval()
        sample, timestep, encoder_hidden_states = self._tiny_unet_inputs()

        with torch.no_grad():
            base = model(sample, timestep, encoder_hidden_states).sample
            model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")
            flowmap = model(sample, timestep, encoder_hidden_states, r_timestep=torch.zeros_like(timestep)).sample

        self.assertFalse(torch.allclose(flowmap, base, atol=1e-6))

    def test_unet_flowmap_from_config_restores_delta_embedding(self):
        model = self._tiny_unet()
        model.enable_flowmap_time_conditioning(gate_value=0.25, deltatime_type="r")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clone = FlowMapUNet2DConditionModel.from_config(model.config)

        flowmap_config_warnings = [
            warning for warning in caught if "deltatime_type" in str(warning.message) or "gate_value" in str(warning.message)
        ]
        self.assertEqual(flowmap_config_warnings, [])
        self.assertNotIn("deltatime_type", model.config.get("_use_default_values", []))
        self.assertEqual(clone.flowmap_deltatime_type, "r")
        self.assertIsNotNone(clone.delta_time_embedding)

    def _tiny_unet(self):
        return FlowMapUNet2DConditionModel(
            sample_size=4,
            in_channels=4,
            out_channels=4,
            down_block_types=("DownBlock2D",),
            up_block_types=("UpBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=4,
            cross_attention_dim=8,
            attention_head_dim=1,
            mid_block_type=None,
        )

    def _tiny_unet_inputs(self):
        sample = torch.randn(2, 4, 4, 4)
        timestep = torch.tensor([10, 20])
        encoder_hidden_states = torch.randn(2, 1, 8)
        return sample, timestep, encoder_hidden_states


if __name__ == "__main__":
    unittest.main()
