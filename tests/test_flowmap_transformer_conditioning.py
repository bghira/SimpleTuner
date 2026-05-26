import unittest

import torch
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings

from simpletuner.helpers.models.flowmap import clone_flowmap_embedder, prepare_flowmap_delta_timestep
from simpletuner.helpers.models.flux2.transformer import Flux2TimestepGuidanceEmbeddings
from simpletuner.helpers.models.flux.transformer import _flux_tokenwise_conditioning, _flux_tokenwise_flowmap_conditioning
from simpletuner.helpers.models.sd3.transformer import _sd3_tokenwise_conditioning, _sd3_tokenwise_flowmap_conditioning


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


if __name__ == "__main__":
    unittest.main()
