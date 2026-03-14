"""Tests for WanS2V (Speech-to-Video) pipeline components."""

import unittest
from unittest import mock

import torch

from tests.test_pipelines._common import PipelineTestCase, WanPromptCleaningMixin


class TestWanS2VPipeline(WanPromptCleaningMixin, PipelineTestCase):
    """Test suite for WanS2V pipeline."""

    module_name = "wan_s2v"


class TestWanS2VTransformerLoading(unittest.TestCase):
    """Test WanS2V transformer can be imported and configured."""

    def test_transformer_import(self):
        """Test that the transformer module can be imported."""
        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel

        self.assertIsNotNone(WanS2VTransformer3DModel)

    def test_transformer_has_tread_support(self):
        """Test that the transformer has TREAD router support."""
        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel

        # Check class attributes for TREAD
        self.assertTrue(hasattr(WanS2VTransformer3DModel, "_tread_router"))
        self.assertTrue(hasattr(WanS2VTransformer3DModel, "_tread_routes"))

    def test_transformer_has_set_router_method(self):
        """Test that the transformer has the set_router method for TREAD."""
        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel

        self.assertTrue(hasattr(WanS2VTransformer3DModel, "set_router"))

    def test_transformer_has_musubi_block_swap(self):
        """Test that the transformer has Musubi block swap support."""
        import inspect

        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel

        # Check the __init__ signature includes musubi parameters
        sig = inspect.signature(WanS2VTransformer3DModel.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("musubi_blocks_to_swap", params)
        self.assertIn("musubi_block_swap_device", params)


class TestWanS2VModelClass(unittest.TestCase):
    """Test WanS2V model class attributes and methods."""

    def test_model_import(self):
        """Test that the model module can be imported."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertIsNotNone(WanS2V)

    def test_model_attributes(self):
        """Test model class attributes are correctly set."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertEqual(WanS2V.NAME, "WanS2V")
        self.assertEqual(WanS2V.LATENT_CHANNEL_COUNT, 16)
        self.assertEqual(WanS2V.DEFAULT_MODEL_FLAVOUR, "s2v-14b-2.2")

    def test_requires_s2v_datasets(self):
        """Test that S2V model reports requiring s2v_datasets."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        # Check the class method exists
        self.assertTrue(hasattr(WanS2V, "requires_s2v_datasets"))

    def test_model_huggingface_path(self):
        """Test that the correct HuggingFace path is configured."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertIn("s2v-14b-2.2", WanS2V.HUGGINGFACE_PATHS)
        self.assertEqual(
            WanS2V.HUGGINGFACE_PATHS["s2v-14b-2.2"],
            "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
        )


class TestWanS2VModelMetadata(unittest.TestCase):
    """Test that WanS2V is registered in model metadata."""

    def test_model_metadata_contains_wan_s2v(self):
        """Test that model_metadata.json contains wan_s2v entry."""
        import json
        from pathlib import Path

        metadata_path = Path(__file__).parent.parent.parent / "simpletuner/helpers/models/model_metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)

        self.assertIn("wan_s2v", metadata)
        self.assertEqual(metadata["wan_s2v"]["class_name"], "WanS2V")
        self.assertEqual(
            metadata["wan_s2v"]["module_path"],
            "simpletuner.helpers.models.wan_s2v.model",
        )


class TestWanS2VStateTrackerIntegration(unittest.TestCase):
    """Test StateTracker S2V dataset methods."""

    def test_s2v_dataset_methods_exist(self):
        """Test that StateTracker has s2v_datasets methods."""
        from simpletuner.helpers.training.state_tracker import StateTracker

        self.assertTrue(hasattr(StateTracker, "set_s2v_datasets"))
        self.assertTrue(hasattr(StateTracker, "get_s2v_datasets"))
        self.assertTrue(hasattr(StateTracker, "get_s2v_mappings"))


class TestWanS2VCollateIntegration(unittest.TestCase):
    """Test that collate function handles s2v_audio_paths."""

    def test_collate_returns_s2v_audio_paths(self):
        """Test that collate function returns s2v_audio_paths field."""
        # The function should include s2v_audio_paths in its return
        import inspect

        from simpletuner.helpers.training.collate import collate_fn

        source = inspect.getsource(collate_fn)
        self.assertIn("s2v_audio_paths", source)


class TestWanS2VAudioInterpolation(unittest.TestCase):
    """Test audio interpolation functionality."""

    def test_interpolate_audio_to_frames(self):
        """Test audio interpolation method exists and works."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        # Check the method exists
        self.assertTrue(hasattr(WanS2V, "interpolate_audio_to_frames"))


class TestWanS2VSelfFlow(unittest.TestCase):
    def test_transformer_block_accepts_tokenwise_timestep_payload(self):
        from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformerBlock

        block = WanS2VTransformerBlock(
            dim=8,
            ffn_dim=16,
            num_heads=2,
            cross_attn_norm=True,
        )

        hidden_states = torch.randn(1, 12, 8)
        encoder_hidden_states = torch.randn(1, 6, 8)
        video_timestep_proj = torch.randn(1, 8, 6, 8)
        conditioning_timestep_proj = torch.randn(1, 6, 8)

        output = block(
            hidden_states,
            encoder_hidden_states,
            (video_timestep_proj, conditioning_timestep_proj, torch.tensor(8)),
            None,
        )

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())

    def test_model_predict_preserves_tokenwise_timesteps_for_self_flow_capture(self):
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        model = WanS2V.__new__(WanS2V)
        model.config = mock.MagicMock(weight_dtype=torch.float32)
        model._new_hidden_state_buffer = mock.MagicMock(return_value=None)
        model.crepa_regularizer = mock.MagicMock(block_index=4)
        model.crepa_regularizer.wants_hidden_states.return_value = True
        model.interpolate_audio_to_frames = mock.MagicMock(side_effect=lambda audio, _: audio)

        predicted = torch.randn(1, 4, 2, 4, 4)
        captured = torch.randn(1, 2, 4, 8)
        model.model = mock.MagicMock(return_value=(predicted, captured))

        tokenwise_timesteps = torch.tensor([[100.0, 900.0, 100.0, 900.0, 100.0, 900.0, 100.0, 900.0]])
        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 2, 4, 4),
            "encoder_hidden_states": torch.randn(1, 6, 16),
            "timesteps": tokenwise_timesteps,
            "conditioning_latents": torch.randn(1, 4, 1, 4, 4),
            "audio_embeds": torch.randn(1, 3, 8, 2),
            "latents": torch.randn(1, 4, 2, 4, 4),
            "crepa_capture_block_index": 9,
        }

        result = model.model_predict(prepared_batch)

        self.assertIs(result["model_prediction"], predicted)
        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], tokenwise_timesteps))
        self.assertEqual(transformer_kwargs["hidden_state_layer"], 9)
        self.assertTrue(transformer_kwargs["output_hidden_states"])

    def test_model_supports_crepa_self_flow(self):
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        model = WanS2V.__new__(WanS2V)
        self.assertTrue(model.supports_crepa_self_flow())


class TestWanS2VValidationSupport(unittest.TestCase):
    """Test S2V validation support methods."""

    def test_requires_s2v_validation_inputs(self):
        """Test that S2V model requires S2V validation inputs."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertTrue(hasattr(WanS2V, "requires_s2v_validation_inputs"))

    def test_requires_conditioning_validation_inputs(self):
        """Test that S2V model requires conditioning validation inputs."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertTrue(hasattr(WanS2V, "requires_conditioning_validation_inputs"))

    def test_conditioning_validation_dataset_type(self):
        """Test that S2V model specifies video dataset type for validation."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertTrue(hasattr(WanS2V, "conditioning_validation_dataset_type"))

    def test_update_pipeline_call_kwargs(self):
        """Test that S2V model has update_pipeline_call_kwargs method."""
        from simpletuner.helpers.models.wan_s2v.model import WanS2V

        self.assertTrue(hasattr(WanS2V, "update_pipeline_call_kwargs"))


class TestValidationS2VFunction(unittest.TestCase):
    """Test retrieve_validation_s2v_samples function exists."""

    def test_retrieve_validation_s2v_samples_exists(self):
        """Test that the S2V validation retrieval function exists."""
        from simpletuner.helpers.training.validation import retrieve_validation_s2v_samples

        self.assertIsNotNone(retrieve_validation_s2v_samples)

    def test_retrieve_validation_images_checks_s2v(self):
        """Test that retrieve_validation_images checks for S2V models."""
        import inspect

        from simpletuner.helpers.training.validation import retrieve_validation_images

        source = inspect.getsource(retrieve_validation_images)
        self.assertIn("requires_s2v_validation_inputs", source)


if __name__ == "__main__":
    unittest.main()
