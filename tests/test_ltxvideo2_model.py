import unittest

import torch

from simpletuner.helpers.models.ltxvideo2.model import _align_ltx2_connector_attention_mask, _pad_ltx2_audio_sequence_for_cp
from simpletuner.helpers.models.ltxvideo2.transformer import LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor


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


if __name__ == "__main__":
    unittest.main()
