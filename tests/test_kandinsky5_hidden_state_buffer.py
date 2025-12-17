import unittest

import torch

from simpletuner.helpers.models.kandinsky5_video.transformer_kandinsky5 import Kandinsky5Transformer3DModel
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer


class Kandinsky5HiddenStateBufferTest(unittest.TestCase):
    def test_forward_accepts_hidden_states_buffer_and_populates_layers(self):
        model = Kandinsky5Transformer3DModel(
            in_visual_dim=4,
            in_text_dim=16,
            in_text_dim2=8,
            time_dim=8,
            out_visual_dim=4,
            patch_size=(1, 2, 2),
            model_dim=32,
            ff_dim=64,
            num_text_blocks=1,
            num_visual_blocks=2,
            axes_dims=(4, 2, 2),
        )

        batch_size = 1
        frames = 2
        height = 4
        width = 4
        text_len = 2

        hidden_states = torch.randn(batch_size, frames, height, width, 4)
        encoder_hidden_states = torch.randn(batch_size, text_len, 16)
        pooled_projections = torch.randn(batch_size, 8)
        timestep = torch.tensor([0.5])

        visual_rope_pos = (
            torch.arange(frames // 1),
            torch.arange(height // 2),
            torch.arange(width // 2),
        )
        text_rope_pos = torch.arange(text_len)

        hidden_states_buffer = HiddenStateBuffer()
        hidden_states_buffer.capture_layers = {0, 1}

        with torch.no_grad():
            out = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                pooled_projections=pooled_projections,
                visual_rope_pos=visual_rope_pos,
                text_rope_pos=text_rope_pos,
                return_dict=True,
                hidden_states_buffer=hidden_states_buffer,
            )

        self.assertTrue(hasattr(out, "sample"))
        self.assertIn("layer_0", hidden_states_buffer)
        self.assertIn("layer_1", hidden_states_buffer)
        self.assertEqual(tuple(hidden_states_buffer["layer_0"].shape), (batch_size, frames, 4, 32))


if __name__ == "__main__":
    unittest.main()
