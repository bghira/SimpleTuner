import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.models.lumina2.transformer import Lumina2Transformer2DModel


class Lumina2TransformerTests(unittest.TestCase):
    def _make_model(self) -> Lumina2Transformer2DModel:
        return Lumina2Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            hidden_size=96,
            num_layers=1,
            num_refiner_layers=1,
            num_attention_heads=4,
            num_kv_heads=4,
            multiple_of=8,
            cap_feat_dim=16,
            axes_dim_rope=(4, 4, 4),
            axes_lens=(16, 16, 16),
        )

    def test_forward_accepts_tokenwise_timesteps(self):
        model = self._make_model()

        with patch(
            "simpletuner.helpers.models.lumina2.transformer.apply_rotary_emb", side_effect=lambda x, *args, **kwargs: x
        ):
            output = model(
                hidden_states=torch.randn(1, 4, 8, 8),
                timestep=torch.tensor(
                    [
                        [
                            100.0,
                            900.0,
                            250.0,
                            750.0,
                            100.0,
                            900.0,
                            250.0,
                            750.0,
                            100.0,
                            900.0,
                            250.0,
                            750.0,
                            100.0,
                            900.0,
                            250.0,
                            750.0,
                        ]
                    ]
                ),
                encoder_hidden_states=torch.randn(1, 3, 16),
                encoder_attention_mask=torch.ones(1, 3, dtype=torch.int32),
                return_dict=False,
            )[0]

        self.assertEqual(output.shape, (1, 4, 8, 8))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        model = self._make_model()

        with self.assertRaisesRegex(ValueError, "tokenwise timesteps expected shape"):
            model(
                hidden_states=torch.randn(1, 4, 8, 8),
                timestep=torch.tensor([[100.0, 900.0]]),
                encoder_hidden_states=torch.randn(1, 3, 16),
                encoder_attention_mask=torch.ones(1, 3, dtype=torch.int32),
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()
