import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.ltxvideo2 import ltx2_rope_freqs_dtype
from simpletuner.helpers.models.ltxvideo2.model import (
    LTXVideo2,
    _align_ltx2_connector_attention_mask,
    _pad_ltx2_audio_sequence_for_cp,
)
from simpletuner.helpers.models.ltxvideo2.transformer import LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor


class _FakeLTX2Connectors:
    def __call__(self, encoder_hidden_states, additive_attention_mask, additive_mask=False):
        del additive_attention_mask, additive_mask
        batch_size = encoder_hidden_states.shape[0]
        device = encoder_hidden_states.device
        dtype = encoder_hidden_states.dtype
        video_embeds = torch.zeros(batch_size, 2, 8, device=device, dtype=dtype)
        audio_embeds = torch.zeros(batch_size, 3, 8, device=device, dtype=dtype)
        attention_mask = torch.ones(batch_size, 3, device=device, dtype=torch.bool)
        return video_embeds, audio_embeds, attention_mask


class _RecordingLTX2Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch_size=1, patch_size_t=1)
        self.last_kwargs = None

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        timestep,
        audio_timestep,
        r_timestep=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "audio_encoder_hidden_states": audio_encoder_hidden_states,
            "timestep": timestep,
            "audio_timestep": audio_timestep,
            "r_timestep": r_timestep,
            **kwargs,
        }
        return torch.zeros_like(hidden_states), torch.zeros_like(audio_hidden_states)


class TestLTXVideo2ModelHelpers(unittest.TestCase):
    def test_align_connector_attention_mask_keeps_matching_length(self):
        attention_mask = torch.tensor([[1, 0, 1]])

        aligned = _align_ltx2_connector_attention_mask(attention_mask, 3)

        self.assertIs(aligned, attention_mask)

    def test_align_connector_attention_mask_crops_left_padding(self):
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])

        aligned = _align_ltx2_connector_attention_mask(attention_mask, 3)

        self.assertTrue(torch.equal(aligned, torch.tensor([[1, 1, 1]])))

    def test_align_connector_attention_mask_rejects_short_mask(self):
        attention_mask = torch.tensor([[1, 1]])

        with self.assertRaisesRegex(ValueError, "shorter than connector sequence length"):
            _align_ltx2_connector_attention_mask(attention_mask, 3)

    def test_pad_audio_sequence_for_alltoall_cp(self):
        audio = torch.ones(2, 17, 4)

        padded, audio_num_frames = _pad_ltx2_audio_sequence_for_cp(audio, 17, 2, "alltoall")

        self.assertEqual(padded.shape, (2, 18, 4))
        self.assertEqual(audio_num_frames, 18)
        self.assertTrue(torch.equal(padded[:, :17], audio))
        self.assertTrue(torch.equal(padded[:, 17:], torch.zeros_like(padded[:, 17:])))

    def test_pad_audio_sequence_for_cp_ignores_allgather(self):
        audio = torch.ones(2, 17, 4)

        padded, audio_num_frames = _pad_ltx2_audio_sequence_for_cp(audio, 17, 2, "allgather")

        self.assertIs(padded, audio)
        self.assertEqual(audio_num_frames, 17)

    def test_perturbed_processor_reuses_attention_output_flattening(self):
        self.assertIs(
            LTX2PerturbedAttnProcessor._flatten_attention_output,
            LTX2AudioVideoAttnProcessor._flatten_attention_output,
        )

    def test_rope_freqs_dtype_keeps_double_precision_off_mps(self):
        self.assertEqual(ltx2_rope_freqs_dtype(True, torch.device("cpu")), torch.float64)
        self.assertEqual(ltx2_rope_freqs_dtype(True, torch.device("mps")), torch.float32)
        self.assertEqual(ltx2_rope_freqs_dtype(False, torch.device("mps")), torch.float32)

    def test_model_predict_forwards_anyflow_r_timestep(self):
        model = LTXVideo2.__new__(LTXVideo2)
        transformer = _RecordingLTX2Transformer()
        model.model = transformer
        model.config = SimpleNamespace(
            controlnet=False,
            weight_dtype=torch.float32,
            framerate=None,
            tread_config=None,
            twinflow_enabled=False,
            context_parallel_size=1,
            context_parallel_comm_strategy="allgather",
            ltx2_intrinsic_conditioning=None,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.connectors = _FakeLTX2Connectors()
        model.crepa_regularizer = None
        model.unwrap_model = MagicMock(side_effect=lambda model=None, **_: model)
        model._load_connectors = MagicMock()
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model._build_grounding_position_net_kwargs = MagicMock(return_value=None)
        r_timesteps = torch.tensor([0.25])

        result = LTXVideo2.model_predict(
            model,
            {
                "noisy_latents": torch.randn(1, 128, 1, 2, 2),
                "audio_latents": torch.randn(1, 8, 3, 4),
                "audio_noisy_latents": torch.randn(1, 8, 3, 4),
                "encoder_hidden_states": torch.randn(1, 4, 8),
                "timesteps": torch.tensor([0.75]),
                LTXVideo2.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps,
            },
        )

        self.assertIs(transformer.last_kwargs["r_timestep"], r_timesteps)
        self.assertEqual(result["model_prediction"].shape, (1, 128, 1, 2, 2))
        self.assertEqual(result["audio_prediction"].shape, (1, 8, 3, 4))


if __name__ == "__main__":
    unittest.main()
