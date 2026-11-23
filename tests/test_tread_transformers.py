import os
import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

os.environ["SIMPLETUNER_LOG_LEVEL"] = "CRITICAL"

from simpletuner.helpers.training.tread import TREADRouter


class TestTREADRouter(unittest.TestCase):
    def test_random_routing_skips_importance_and_preserves_seeded_mask(self):
        seed = 9
        router = TREADRouter(seed=seed, device="cpu")
        forced = torch.zeros(2, 6, dtype=torch.bool)
        forced[0, 4] = True
        forced[1, 2] = True
        hidden_states = torch.arange(36, dtype=torch.bfloat16).reshape(2, 6, 3)
        noise = torch.rand((2, 6), dtype=hidden_states.dtype, generator=torch.Generator().manual_seed(seed))

        with patch.object(router, "_importance", side_effect=AssertionError("random routing should only use noise")):
            info = router.get_mask(hidden_states, mask_ratio=0.5, force_keep=forced)

        torch.testing.assert_close(info.ids_keep, noise.masked_fill(forced, -1).argsort(dim=1)[:, :3])

    def test_routed_storage_only_contains_kept_tokens(self):
        router = TREADRouter(device="cpu")
        original = torch.arange(36, dtype=torch.float32).reshape(2, 6, 3).requires_grad_()
        info = router.get_mask(original, mask_ratio=0.5)
        routed = router.start_route(original, info)

        expected = torch.stack([original[index, keep] for index, keep in enumerate(info.ids_keep)])
        torch.testing.assert_close(routed, expected)
        self.assertEqual(routed.untyped_storage().nbytes(), routed.numel() * routed.element_size())

        dropped = info.mask.unsqueeze(-1)
        restored = router.end_route(routed * 2, info, original_x=original)
        torch.testing.assert_close(restored, torch.where(dropped, original, original * 2))
        restored.sum().backward()
        torch.testing.assert_close(original.grad, torch.where(dropped, 1.0, 2.0).expand_as(original))

        masked = router.end_route(routed, info, mask_token=-1)
        torch.testing.assert_close(masked, original.masked_fill(dropped, -1))


class TestTREADTransformers(unittest.TestCase):
    """Test TREAD support added to transformer models."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_router = Mock(spec=TREADRouter)
        self.test_routes = [
            {
                "selection_ratio": 0.5,
                "start_layer_idx": 2,
                "end_layer_idx": 4,
            }
        ]

    def test_auraflow_tread_integration(self):
        from simpletuner.helpers.models.auraflow.transformer import AuraFlowTransformer2DModel

        model = AuraFlowTransformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_mmdit_layers=1,
            num_single_dit_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            caption_projection_dim=16,
            pos_embed_max_size=16,
        )
        router = TREADRouter(device="cpu")
        model.set_router(router, [{"selection_ratio": 0.5, "start_layer_idx": -2, "end_layer_idx": -1}])
        image = torch.randn(1, 4, 8, 8)
        context = torch.randn(1, 3, 16)
        for training in (True, False):
            with self.subTest(training=training), patch.object(router, "end_route", wraps=router.end_route) as restore:
                model.train(training)
                output = model(image, context, torch.tensor([1.0])).sample
                self.assertEqual(output.shape, image.shape)
                if training:
                    restore.assert_called_once()
                    self.assertEqual(restore.call_args.args[0].shape[1], 8)
                    self.assertEqual(restore.call_args.kwargs["original_x"].shape[1], 16)
                else:
                    restore.assert_not_called()

    def test_flux2_routes_both_streams_with_per_sample_rotary_positions(self):
        from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel

        model = Flux2Transformer2DModel(
            in_channels=4,
            num_layers=1,
            num_single_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=4,
            mlp_ratio=2.0,
            axes_dims_rope=(2, 2, 2, 2),
        )
        router = TREADRouter(device="cpu")
        model.set_router(router, [{"selection_ratio": 2 / 3, "start_layer_idx": 0, "end_layer_idx": -2}])

        image_tokens = torch.randn(2, 6, 4, requires_grad=True)
        text_tokens = torch.randn(2, 2, 16)
        image_ids = torch.tensor([[0, 0, i, 0] for i in range(6)])
        text_ids = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 1]])
        force_keep = torch.zeros(2, 6, dtype=torch.bool)
        force_keep[0, [0, 2]] = True
        force_keep[1, [1, 4]] = True

        block_inputs = []

        def record_inputs(_block, _args, kwargs):
            block_inputs.append(kwargs)

        for block in (*model.transformer_blocks, *model.single_transformer_blocks):
            self.addCleanup(block.register_forward_pre_hook(record_inputs, with_kwargs=True).remove)

        with (
            patch.object(router, "get_mask", wraps=router.get_mask) as choose_mask,
            patch.object(router, "end_route", wraps=router.end_route) as end_route,
        ):
            output = model(
                hidden_states=image_tokens,
                encoder_hidden_states=text_tokens,
                timestep=torch.rand(2, 6),
                img_ids=image_ids,
                txt_ids=text_ids,
                guidance=torch.ones(2),
                force_keep_mask=force_keep,
            ).sample

        self.assertEqual(output.shape, image_tokens.shape)
        choose_mask.assert_called_once()
        end_route.assert_called_once()
        double_stream, single_stream, restored_stream = block_inputs
        self.assertEqual(double_stream["hidden_states"].shape[:2], (2, 2))
        self.assertEqual(double_stream["encoder_hidden_states"].shape[:2], (2, 2))
        self.assertEqual(single_stream["hidden_states"].shape[:2], (2, 4))
        self.assertEqual(restored_stream["hidden_states"].shape[:2], (2, 8))

        image_rope = model.pos_embed(image_ids)
        text_rope = model.pos_embed(text_ids)
        routed, mask_info, original = end_route.call_args.args
        self.assertTrue(torch.equal(~mask_info.mask, force_keep))
        for block in (double_stream, single_stream):
            for actual, text_table, image_table in zip(block["image_rotary_emb"], text_rope, image_rope):
                expected = torch.cat((text_table.expand(2, -1, -1), image_table[mask_info.ids_keep]), dim=1)
                torch.testing.assert_close(actual, expected)

        expected = original.scatter(1, mask_info.ids_keep.unsqueeze(-1).expand_as(routed), routed)
        torch.testing.assert_close(restored_stream["hidden_states"][:, 2:], expected)

        output.square().mean().backward()
        self.assertTrue(torch.isfinite(image_tokens.grad).all())

    def test_sd3_tread_integration(self):
        """Test TREAD integration in SD3 transformer."""
        with patch("simpletuner.helpers.models.sd3.transformer.TREADRouter"):
            from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

            # Create model with minimal config
            model = SD3Transformer2DModel(
                sample_size=64,
                patch_size=2,
                in_channels=16,
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=12,
                joint_attention_dim=4096,
                caption_projection_dim=1152,
                pooled_projection_dim=2048,
                out_channels=16,
                pos_embed_max_size=96,
            )

            # Test set_router method
            model.set_router(self.mock_router, self.test_routes)
            self.assertEqual(model._tread_router, self.mock_router)
            self.assertEqual(model._tread_routes, self.test_routes)

            # Test TREAD attributes exist
            self.assertIsNotNone(model._tread_router)
            self.assertIsNotNone(model._tread_routes)

    def test_pixart_tread_integration(self):
        """Test TREAD integration in PixArt transformer."""
        with patch("simpletuner.helpers.models.pixart.transformer.TREADRouter"):
            from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel

            # Create model with minimal config
            model = PixArtTransformer2DModel(
                num_attention_heads=12,
                attention_head_dim=64,
                in_channels=4,
                out_channels=4,
                num_layers=2,
                dropout=0.0,
                norm_num_groups=32,
                cross_attention_dim=1152,
                attention_bias=True,
                sample_size=64,
                patch_size=2,
                activation_fn="gelu-approximate",
                num_embeds_ada_norm=1000,
                upcast_attention=False,
                norm_type="ada_norm_single",
                norm_elementwise_affine=False,
                norm_eps=1e-6,
            )

            # Test set_router method
            model.set_router(self.mock_router, self.test_routes)
            self.assertEqual(model._tread_router, self.mock_router)
            self.assertEqual(model._tread_routes, self.test_routes)

            # Test TREAD attributes exist
            self.assertIsNotNone(model._tread_router)
            self.assertIsNotNone(model._tread_routes)

    def test_tread_force_keep_mask(self):
        """Test force_keep_mask parameter handling."""
        with patch("simpletuner.helpers.models.sd3.transformer.TREADRouter"):
            from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

            model = SD3Transformer2DModel(
                sample_size=64,
                patch_size=2,
                in_channels=16,
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=12,
                joint_attention_dim=4096,
                caption_projection_dim=1152,
                pooled_projection_dim=2048,
                out_channels=16,
                pos_embed_max_size=96,
            )

            model.set_router(self.mock_router, self.test_routes)

            # Test that force_keep_mask parameter exists in forward method
            import inspect

            forward_signature = inspect.signature(model.forward)
            self.assertIn("force_keep_mask", forward_signature.parameters)

            # Test parameter is optional
            param = forward_signature.parameters["force_keep_mask"]
            self.assertEqual(param.default, None)

    def test_flux_single_stream_route_accepts_image_and_combined_keep_masks(self):
        from tests.test_flux_time_sign_embed_gating import _tiny_flux_transformer

        model = _tiny_flux_transformer()
        router = TREADRouter(device="cpu")
        model.set_router(router, [{"selection_ratio": 0.5, "start_layer_idx": 1, "end_layer_idx": 1}])
        image_tokens = torch.randn(2, 4, 4)
        force_keep = torch.tensor([[True, False, True, False], [False, True, False, True]])
        text_mask = torch.ones(2, 2, dtype=torch.bool)

        for keep_mask in (force_keep, torch.cat((text_mask, force_keep), dim=1)):
            with (
                self.subTest(mask_width=keep_mask.shape[1]),
                patch.object(router, "end_route", wraps=router.end_route) as restore,
            ):
                output = model(
                    hidden_states=image_tokens,
                    encoder_hidden_states=torch.randn(2, 2, 16),
                    timestep=torch.tensor([0.25, 0.75]),
                    pooled_projections=torch.randn(2, 8),
                    img_ids=torch.tensor([[0, 0, i] for i in range(4)]),
                    txt_ids=torch.zeros(2, 3, dtype=torch.long),
                    force_keep_mask=keep_mask,
                ).sample

            self.assertEqual(output.shape, image_tokens.shape)
            torch.testing.assert_close(~restore.call_args.args[1].mask, force_keep)

    def test_empty_routes_handling(self):
        """Test handling when no routes are configured."""
        with patch("simpletuner.helpers.models.pixart.transformer.TREADRouter"):
            from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel

            model = PixArtTransformer2DModel(
                num_attention_heads=12,
                attention_head_dim=64,
                in_channels=4,
                out_channels=4,
                num_layers=2,
                dropout=0.0,
                norm_num_groups=32,
                cross_attention_dim=1152,
                attention_bias=True,
                sample_size=64,
                patch_size=2,
                activation_fn="gelu-approximate",
                num_embeds_ada_norm=1000,
                upcast_attention=False,
                norm_type="ada_norm_single",
                norm_elementwise_affine=False,
                norm_eps=1e-6,
            )

            # Test with empty routes
            model.set_router(self.mock_router, [])
            self.assertEqual(model._tread_routes, [])

            # Test with None routes
            model.set_router(self.mock_router, None)
            self.assertEqual(model._tread_routes, None)

    def test_hidream_tread_integration(self):
        from simpletuner.helpers.models.hidream.transformer import HiDreamImageTransformer2DModel

        model = HiDreamImageTransformer2DModel(
            patch_size=2,
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=16,
            num_attention_heads=2,
            caption_channels=[16, 16],
            text_emb_dim=16,
            num_routed_experts=0,
            llama_layers=[0],
            axes_dims_rope=(4, 6, 6),
            max_resolution=(4, 4),
        )
        # Fresh HiDream gates and the output head are zero, so routing mistakes would be invisible.
        for block in [*model.double_stream_blocks, *model.single_stream_blocks]:
            block.block.adaLN_modulation[1].reset_parameters()
        model.final_layer.linear.reset_parameters()
        image = torch.randn(2, 4, 4, 4, requires_grad=True)
        self._check_routed_forward(
            model,
            image,
            dict(
                timesteps=torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
                t5_hidden_states=torch.randn(2, 3, 16),
                llama_hidden_states=torch.randn(1, 2, 2, 16),
                pooled_embeds=torch.randn(2, 16),
            ),
        )

    def test_cosmos_tread_integration(self):
        from simpletuner.helpers.models.cosmos.transformer import CosmosTransformer3DModel

        model = CosmosTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=16,
            num_layers=2,
            mlp_ratio=2,
            text_embed_dim=16,
            adaln_lora_dim=8,
            max_size=(1, 4, 4),
            concat_padding_mask=False,
        )
        image = torch.randn(2, 4, 1, 4, 4, requires_grad=True)
        self._check_routed_forward(
            model,
            image,
            dict(
                timestep=torch.tensor([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]]),
                encoder_hidden_states=torch.randn(2, 3, 16),
            ),
        )

    def _check_routed_forward(self, model, image, conditioning):
        router = TREADRouter(device="cpu")
        model.set_router(router, [{"selection_ratio": 0.5, "start_layer_idx": 0, "end_layer_idx": -1}])
        keep = torch.tensor([[True, False, True, False], [False, True, False, True]])
        with (
            patch.object(router, "get_mask", wraps=router.get_mask) as select,
            patch.object(router, "end_route", wraps=router.end_route) as restore,
        ):
            output = model(hidden_states=image, force_keep_mask=keep, **conditioning).sample
        select.assert_called_once()
        routed_tokens, mask = restore.call_args.args
        self.assertEqual(routed_tokens.shape[:2], (2, 2))
        torch.testing.assert_close(~mask.mask, keep)
        self.assertEqual(output.shape, image.shape)
        grad = torch.autograd.grad(output.square().mean(), image)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)

        model.enable_gradient_checkpointing()
        checkpointed = model(hidden_states=image, force_keep_mask=keep, **conditioning).sample
        torch.testing.assert_close(checkpointed, output)
        torch.testing.assert_close(torch.autograd.grad(checkpointed.square().mean(), image)[0], grad)
        model.disable_gradient_checkpointing()

        # Keeping all tokens permutes image positions. RoPE, conditioning and text
        # must stay aligned for the routed result to match the unrouted forward.
        model.set_router(router, [])
        reference = model(hidden_states=image, **conditioning).sample
        self.assertFalse(torch.allclose(output, reference))
        model.set_router(router, [{"selection_ratio": 0.0, "start_layer_idx": 0, "end_layer_idx": -1}])
        permuted = model(hidden_states=image, **conditioning).sample
        torch.testing.assert_close(permuted, reference, rtol=1e-4, atol=1e-5)

    def test_sana_tread_integration(self):
        """Test TREAD integration in SANA transformer."""
        with patch("simpletuner.helpers.models.sana.transformer.TREADRouter"):
            from simpletuner.helpers.models.sana.transformer import SanaTransformer2DModel

            # Create model with minimal config
            model = SanaTransformer2DModel(
                in_channels=32,
                out_channels=32,
                num_attention_heads=70,
                attention_head_dim=32,
                num_layers=2,
                num_cross_attention_heads=20,
                cross_attention_head_dim=112,
                cross_attention_dim=2240,
                caption_channels=2304,
                sample_size=32,
                patch_size=1,
            )

            # Test set_router method
            model.set_router(self.mock_router, self.test_routes)
            self.assertEqual(model._tread_router, self.mock_router)
            self.assertEqual(model._tread_routes, self.test_routes)

            # Test TREAD attributes exist
            self.assertIsNotNone(model._tread_router)
            self.assertIsNotNone(model._tread_routes)

    def test_qwen_image_tread_integration(self):
        """Test TREAD integration in Qwen Image transformer."""
        with patch("simpletuner.helpers.models.qwen_image.transformer.TREADRouter"):
            from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel

            # Create model with minimal config
            model = QwenImageTransformer2DModel(
                patch_size=2,
                in_channels=16,
                out_channels=16,
                num_layers=2,
                num_attention_heads=12,
                attention_head_dim=64,
                joint_attention_dim=2048,
            )

            # Test set_router method
            model.set_router(self.mock_router, self.test_routes)
            self.assertEqual(model._tread_router, self.mock_router)
            self.assertEqual(model._tread_routes, self.test_routes)

            # Test TREAD attributes exist
            self.assertIsNotNone(model._tread_router)
            self.assertIsNotNone(model._tread_routes)

    def test_ernie_tread_integration(self):
        """Test TREAD integration in ERNIE transformer."""
        with patch("simpletuner.helpers.models.ernie.transformer.TREADRouter"):
            from simpletuner.helpers.models.ernie.transformer import ErnieImageTransformer2DModel

            model = ErnieImageTransformer2DModel(
                hidden_size=128,
                num_attention_heads=8,
                num_layers=2,
                ffn_hidden_size=256,
                in_channels=16,
                out_channels=16,
                text_in_dim=64,
                rope_axes_dim=(8, 4, 4),
            )

            model.set_router(self.mock_router, self.test_routes)
            self.assertEqual(model._tread_router, self.mock_router)
            self.assertEqual(model._tread_routes, self.test_routes)
            self.assertIsNotNone(model._tread_router)
            self.assertIsNotNone(model._tread_routes)


class TestTREADModelInitialization(unittest.TestCase):
    """Test TREAD initialization methods in model classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_config = Mock()
        self.mock_config.tread_config = {
            "routes": [
                {
                    "selection_ratio": 0.5,
                    "start_layer_idx": 2,
                    "end_layer_idx": 4,
                }
            ]
        }
        self.mock_config.seed = 42

    def test_auraflow_tread_init(self):
        """Test TREAD initialization in AuraFlow model."""
        import importlib

        import simpletuner.helpers.models.auraflow as _af_pkg

        # Ensure the submodule is registered on the package so patch() can resolve the dotted path.
        # A full import may fail due to optional dependencies (e.g. diffusers.models.controlnet),
        # so we guard it and skip when the environment lacks them.
        if not hasattr(_af_pkg, "model"):
            try:
                importlib.import_module("simpletuner.helpers.models.auraflow.model")
            except ImportError as exc:
                self.skipTest(f"Cannot import auraflow.model: {exc}")

        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.auraflow.model.logger"):
                from simpletuner.helpers.models.auraflow.model import Auraflow

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cpu"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(Auraflow, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = Auraflow
                Auraflow.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="cpu")

    def test_sd3_tread_init(self):
        """Test TREAD initialization in SD3 model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.sd3.model.logger"):
                from simpletuner.helpers.models.sd3.model import SD3

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cuda"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(SD3, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = SD3
                SD3.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="cuda")

    def test_pixart_tread_init(self):
        """Test TREAD initialization in PixArt model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.pixart.model.logger"):
                from simpletuner.helpers.models.pixart.model import PixartSigma

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "mps"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(PixartSigma, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = PixartSigma
                PixartSigma.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="mps")

    def test_tread_init_missing_config(self):
        """Test TREAD initialization handles missing config correctly."""
        try:
            from simpletuner.helpers.models.auraflow.model import Auraflow
        except ImportError as exc:
            self.skipTest(f"Cannot import auraflow.model: {exc}")

        self.assertTrue(hasattr(Auraflow, "tread_init"))

        # Test with valid config (successful path)
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.auraflow.model.logger"):
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cpu"
                model_instance.unwrap_model.return_value = Mock()
                model_instance.__class__ = Auraflow

                # This should succeed without error
                Auraflow.tread_init(model_instance)
                mock_tread_router.assert_called_once()

    def test_tread_init_empty_routes(self):
        """Test TREAD initialization fails with empty routes."""
        with patch("simpletuner.helpers.training.tread.TREADRouter"):
            with patch("simpletuner.helpers.models.sd3.model.logger") as mock_logger:
                with patch("sys.exit") as mock_exit:
                    from simpletuner.helpers.models.sd3.model import SD3

                    # Create mock model instance with empty routes
                    model_instance = Mock()
                    model_instance.config = Mock()
                    model_instance.config.tread_config = {"routes": None}

                    model_instance.__class__ = SD3

                    # Test that empty routes raises SystemExit
                    SD3.tread_init(model_instance)
                    mock_exit.assert_called_once_with(1)

                    # Verify error was logged
                    mock_logger.error.assert_called_once()

    def test_hidream_tread_init(self):
        """Test TREAD initialization in HiDream model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.hidream.model.logger"):
                from simpletuner.helpers.models.hidream.model import HiDream

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cpu"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(HiDream, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = HiDream
                HiDream.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="cpu")

    def test_cosmos_tread_init(self):
        """Test TREAD initialization in Cosmos model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.cosmos.model.logger"):
                from simpletuner.helpers.models.cosmos.model import Cosmos2Image

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cuda"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(Cosmos2Image, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = Cosmos2Image
                Cosmos2Image.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="cuda")

    def test_sana_tread_init(self):
        """Test TREAD initialization in SANA model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.sana.model.logger"):
                from simpletuner.helpers.models.sana.model import Sana

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "mps"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(Sana, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = Sana
                Sana.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="mps")

    def test_qwen_image_tread_init(self):
        """Test TREAD initialization in Qwen Image model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.qwen_image.model.logger"):
                from simpletuner.helpers.models.qwen_image.model import QwenImage

                # Create mock model instance
                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cuda"
                model_instance.unwrap_model.return_value = Mock()

                # Test tread_init method exists
                self.assertTrue(hasattr(QwenImage, "tread_init"))

                # Test successful initialization
                model_instance.__class__ = QwenImage
                QwenImage.tread_init(model_instance)

                # Verify TREADRouter was called with correct parameters
                mock_tread_router.assert_called_once_with(seed=42, device="cuda")

    def test_ernie_tread_init(self):
        """Test TREAD initialization in ERNIE model."""
        with patch("simpletuner.helpers.training.tread.TREADRouter") as mock_tread_router:
            with patch("simpletuner.helpers.models.ernie.model.logger"):
                from simpletuner.helpers.models.ernie.model import Ernie

                model_instance = Mock()
                model_instance.config = self.mock_config
                model_instance.accelerator = Mock()
                model_instance.accelerator.device = "cpu"
                model_instance.unwrap_model.return_value = Mock()

                self.assertTrue(hasattr(Ernie, "tread_init"))

                model_instance.__class__ = Ernie
                Ernie.tread_init(model_instance)

                mock_tread_router.assert_called_once_with(seed=42, device="cpu")


if __name__ == "__main__":
    unittest.main()
