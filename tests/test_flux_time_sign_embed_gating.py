import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel
from simpletuner.helpers.training.tread import TREADRouter


def _tiny_flux_transformer(**extra_kwargs) -> FluxTransformer2DModel:
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
        musubi_blocks_to_swap=0,
        musubi_block_swap_device="cpu",
        **extra_kwargs,
    )


class FluxTimeSignEmbedGatingTestCase(unittest.TestCase):
    def test_tread_routes_tokenwise_timesteps_in_both_streams(self):
        model = _tiny_flux_transformer()
        router = TREADRouter(seed=9, device="cpu")
        image_tokens = torch.randn(2, 4, 4, requires_grad=True)
        inputs = {
            "hidden_states": image_tokens,
            "encoder_hidden_states": torch.randn(2, 2, 16),
            "pooled_projections": torch.randn(2, 8),
            "timestep": torch.tensor([[0.1, 0.3, 0.5, 0.7], [0.2, 0.4, 0.6, 0.8]]),
            "img_ids": torch.tensor([[0, 0, i] for i in range(4)]),
            "txt_ids": torch.zeros(2, 3, dtype=torch.long),
        }
        modulation = []

        def record_modulation(_block, _args, kwargs):
            modulation.append(kwargs["temb"])

        for block in (*model.transformer_blocks, *model.single_transformer_blocks):
            self.addCleanup(block.register_forward_pre_hook(record_modulation, with_kwargs=True).remove)

        model(**inputs)
        dense_double, dense_single = modulation
        for start in (0, 1):
            with self.subTest(start=start), patch.object(router, "end_route", wraps=router.end_route) as restore:
                modulation.clear()
                model.set_router(router, [{"selection_ratio": 0.5, "start_layer_idx": start, "end_layer_idx": 1}])
                output = model(**inputs).sample
                self.assertEqual(output.shape, image_tokens.shape)
                output.square().mean().backward()

            mask = restore.call_args.args[1]
            expected_image = router.start_route(dense_double, mask)
            torch.testing.assert_close(modulation[0], expected_image if start == 0 else dense_double)
            torch.testing.assert_close(modulation[1], torch.cat((dense_single[:, :2], expected_image), dim=1))

    def test_forward_accepts_tokenwise_timesteps(self):
        model = _tiny_flux_transformer()
        hidden_states = torch.randn(1, 4, 4)
        encoder_hidden_states = torch.randn(1, 2, 16)
        pooled_projections = torch.randn(1, 8)
        timestep = torch.tensor([[0.1, 0.3, 0.5, 0.7]], dtype=torch.float32)
        img_ids = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=torch.int64,
        )
        txt_ids = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=torch.int64,
        )

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (1, 4, 4))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        model = _tiny_flux_transformer()
        with self.assertRaisesRegex(ValueError, "tokenwise timesteps"):
            model(
                hidden_states=torch.randn(1, 4, 4),
                encoder_hidden_states=torch.randn(1, 2, 16),
                pooled_projections=torch.randn(1, 8),
                timestep=torch.tensor([[0.1, 0.3, 0.5]], dtype=torch.float32),
                img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.int64),
                txt_ids=torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64),
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()
