import unittest

import torch
import torch.nn as nn

from simpletuner.helpers.models.ltxvideo2.connectors import LTX2TextConnectors


class _IdentityConnector(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return hidden_states, attention_mask


class TestLTX2TextConnectors(unittest.TestCase):
    def test_forward_accepts_additive_mask_and_matches_binary_mask(self):
        connectors = LTX2TextConnectors(
            caption_channels=2,
            text_proj_in_factor=2,
            video_connector_num_attention_heads=1,
            video_connector_attention_head_dim=4,
            video_connector_num_layers=1,
            video_connector_num_learnable_registers=None,
            audio_connector_num_attention_heads=1,
            audio_connector_attention_head_dim=4,
            audio_connector_num_layers=1,
            audio_connector_num_learnable_registers=None,
            per_modality_projections=True,
            video_hidden_dim=4,
            audio_hidden_dim=4,
        )
        connectors.video_connector = _IdentityConnector()
        connectors.audio_connector = _IdentityConnector()

        hidden_states = torch.arange(1, 13, dtype=torch.float32).reshape(1, 3, 2, 2)
        binary_mask = torch.tensor([[1, 1, 0]], dtype=torch.int64)
        additive_mask = (1 - binary_mask.to(torch.float32)) * -1000000.0

        video_binary, audio_binary, returned_binary_mask = connectors(hidden_states, binary_mask)
        video_additive, audio_additive, returned_additive_mask = connectors(hidden_states, additive_mask, additive_mask=True)

        self.assertTrue(torch.allclose(video_binary, video_additive))
        self.assertTrue(torch.allclose(audio_binary, audio_additive))
        self.assertTrue(torch.equal(returned_binary_mask, returned_additive_mask))
        self.assertTrue(torch.equal(returned_binary_mask, binary_mask))
        self.assertTrue(torch.count_nonzero(video_binary[:, -1, :]) == 0)
        self.assertTrue(torch.count_nonzero(audio_binary[:, -1, :]) == 0)


if __name__ == "__main__":
    unittest.main()
