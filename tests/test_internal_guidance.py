import unittest
from types import SimpleNamespace

import torch
from torch import nn

from simpletuner.helpers.models.common import ModelFoundation, ModelTypes, PredictionTypes
from simpletuner.helpers.training.internal_guidance import (
    InternalGuidanceHead,
    InternalGuidanceRegularizer,
    attach_internal_guidance_head_from_state_dict,
    infer_internal_guidance_block_count,
    infer_internal_guidance_output_features,
    internal_guidance_lora_state_dict,
)
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer
from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry


class _Transformer(nn.Module):
    def __init__(self, *, hidden_size=8, channels=4, patch_size=(2, 2), layers=8):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            in_channels=channels,
            out_channels=channels,
            patch_size=patch_size,
        )
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(layers)])
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, *, hidden_states_buffer=None):
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states)
            if hidden_states_buffer is not None:
                hidden_states_buffer[f"layer_{layer_idx}"] = hidden_states
        batch, tokens, _ = hidden_states.shape
        side = int(tokens**0.5) * 2
        return torch.ones(batch, 4, side, side, device=hidden_states.device, dtype=hidden_states.dtype)


class _Foundation:
    NAME = "test transformer"
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self, **overrides):
        values = {
            "internal_guidance_enabled": True,
            "internal_guidance_loss_weight": 0.5,
            "internal_guidance_block_index": 1,
            "validation_internal_guidance_scale": 1.0,
            "lora_type": "standard",
            "weight_dtype": torch.float32,
        }
        values.update(overrides)
        self.config = SimpleNamespace(**values)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model = _Transformer()
        self.internal_guidance_regularizer = None

    def unwrap_model(self, model=None):
        return model or self.model

    def get_trained_component(self, unwrap_model=False):
        return self.model

    def _infer_transformer_hidden_size(self):
        return ModelFoundation._infer_transformer_hidden_size(self)

    def get_prediction_target(self, prepared_batch):
        return prepared_batch["target"]

    def loss(self, prepared_batch, model_output, apply_conditioning_mask=True):
        return torch.nn.functional.mse_loss(model_output["model_prediction"], prepared_batch["target"])


class InternalGuidanceHeadTests(unittest.TestCase):
    def test_zero_initialised_image_head_matches_target_shape(self):
        head = InternalGuidanceHead(hidden_size=8, output_features=16, block_index=3)
        hidden_states = torch.randn(2, 16, 8)
        target = torch.randn(2, 4, 8, 8)

        prediction = head.unpatchify(head(hidden_states), target, preferred_patch_size=(2, 2))

        self.assertEqual(prediction.shape, target.shape)
        self.assertEqual(head.block_index.item(), 3)
        self.assertTrue(torch.count_nonzero(prediction).item() == 0)

    def test_video_head_unpatchifies_temporal_and_spatial_tokens(self):
        head = InternalGuidanceHead(hidden_size=8, output_features=16)
        hidden_states = torch.randn(2, 48, 8)
        target = torch.randn(2, 4, 3, 8, 8)

        prediction = head.unpatchify(head(hidden_states), target, preferred_patch_size=(1, 2, 2))

        self.assertEqual(prediction.shape, target.shape)

    def test_shape_mismatch_fails_loudly(self):
        target = torch.randn(1, 4, 8, 8)
        prediction_tokens = torch.randn(1, 15, 16)

        with self.assertRaisesRegex(ValueError, "token count does not match"):
            InternalGuidanceHead.unpatchify(prediction_tokens, target)


class InternalGuidanceRegularizerTests(unittest.TestCase):
    def setUp(self):
        self.foundation = _Foundation()
        ModelFoundation._init_internal_guidance_regularizer(self.foundation)

    def test_infers_transformer_dimensions_and_default_depth(self):
        model = _Transformer(layers=12)

        self.assertEqual(infer_internal_guidance_block_count(model), 12)
        self.assertEqual(infer_internal_guidance_output_features(model), 16)

    def test_auxiliary_loss_backpropagates_into_head_and_backbone(self):
        regularizer = self.foundation.internal_guidance_regularizer
        nn.init.normal_(regularizer.head.proj.weight, std=0.02)
        hidden_states = torch.randn(2, 16, 8, requires_grad=True)
        target = torch.randn(2, 4, 8, 8)
        buffer = HiddenStateBuffer(capture_layers={1})
        buffer["layer_1"] = hidden_states

        loss, logs = regularizer.compute_loss(buffer, {"target": target}, self.foundation)
        loss.backward()

        self.assertIsNotNone(hidden_states.grad)
        self.assertIsNotNone(regularizer.head.proj.weight.grad)
        self.assertIn("internal_guidance_loss", logs)

    def test_inference_context_applies_extrapolation_and_restores_forward(self):
        regularizer = self.foundation.internal_guidance_regularizer
        hidden_states = torch.randn(1, 16, 8)
        original_forward = self.foundation.model.forward

        with regularizer.inference_context(scale=1.5):
            guided = self.foundation.model(hidden_states)

        self.assertTrue(torch.allclose(guided, torch.full_like(guided, 1.5)))
        self.assertEqual(self.foundation.model.forward, original_forward)

    def test_foundation_inference_context_rejects_explicit_zero_scale(self):
        self.foundation.config.validation_internal_guidance_scale = 0.0

        with self.assertRaisesRegex(ValueError, "must be greater than zero"):
            with ModelFoundation.internal_guidance_inference_context(self.foundation):
                pass

    def test_peft_modules_to_save_include_head(self):
        self.assertEqual(
            ModelFoundation.get_lora_save_layers(self.foundation),
            [InternalGuidanceRegularizer.MODULE_NAME],
        )

    def test_vanilla_loader_can_reconstruct_head_before_adapter_load(self):
        model = _Transformer()
        projection = torch.randn(16, 8)
        state_dict = {
            "transformer.internal_guidance_head.norm.weight": torch.randn(8),
            "transformer.internal_guidance_head.norm.bias": torch.randn(8),
            "transformer.internal_guidance_head.proj.weight": projection,
            "transformer.internal_guidance_head.proj.bias": torch.randn(16),
            "transformer.internal_guidance_head.block_index": torch.tensor(3),
            "transformer.blocks.0.attn.to_q.lora_A.weight": torch.randn(2, 8),
        }

        head = attach_internal_guidance_head_from_state_dict(model, state_dict)
        lora_state_dict = internal_guidance_lora_state_dict(state_dict)

        self.assertIs(model.internal_guidance_head, head)
        self.assertEqual(head.proj.weight.shape, (16, 8))
        self.assertTrue(torch.equal(head.proj.weight, projection))
        self.assertEqual(head.block_index.item(), 3)
        self.assertEqual(list(lora_state_dict), ["transformer.blocks.0.attn.to_q.lora_A.weight"])

    def test_rejects_unet_autoregressive_and_lycoris_modes(self):
        cases = (
            (ModelTypes.UNET, PredictionTypes.EPSILON, "standard", "diffusion transformer"),
            (ModelTypes.TRANSFORMER, PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN, "standard", "autoregressive"),
            (ModelTypes.TRANSFORMER, PredictionTypes.FLOW_MATCHING, "lycoris", "standard PEFT LoRA"),
        )
        for model_type, prediction_type, lora_type, message in cases:
            with self.subTest(model_type=model_type, prediction_type=prediction_type, lora_type=lora_type):
                foundation = _Foundation(lora_type=lora_type)
                foundation.MODEL_TYPE = model_type
                foundation.PREDICTION_TYPE = prediction_type
                with self.assertRaisesRegex(ValueError, message):
                    ModelFoundation._init_internal_guidance_regularizer(foundation)


class HiddenStateBufferTests(unittest.TestCase):
    def test_capture_filter_retains_only_requested_layers(self):
        buffer = HiddenStateBuffer(capture_layers={2})
        buffer["layer_1"] = torch.tensor(1)
        buffer["layer_2"] = torch.tensor(2)
        buffer["metadata"] = "kept"

        self.assertNotIn("layer_1", buffer)
        self.assertEqual(buffer["layer_2"].item(), 2)
        self.assertEqual(buffer["metadata"], "kept")


class InternalGuidanceFieldTests(unittest.TestCase):
    def test_registry_exposes_dit_fields_and_excludes_autoregressive_family(self):
        registry = FieldRegistry()
        enabled = registry.get_field("internal_guidance_enabled")

        self.assertIn("anima", enabled.model_specific)
        self.assertIn("wan", enabled.model_specific)
        self.assertIn("minimaxmusic", enabled.model_specific)
        self.assertNotIn("heartmula", enabled.model_specific)
        self.assertIsNotNone(registry.get_field("validation_internal_guidance_scale"))


if __name__ == "__main__":
    unittest.main()
