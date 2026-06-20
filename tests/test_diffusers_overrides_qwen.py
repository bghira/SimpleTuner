import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from diffusers import QwenImagePipeline
from diffusers.models import attention_dispatch
from diffusers.models.attention_dispatch import _HUB_KERNELS_REGISTRY, AttentionBackendName
from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0

from simpletuner.helpers.training import diffusers_overrides


class FakeBatchEncoding(SimpleNamespace):
    def to(self, device):
        converted = {}
        for key, value in self.__dict__.items():
            if torch.is_tensor(value):
                converted[key] = value.to(device)
            else:
                converted[key] = value
        return FakeBatchEncoding(**converted)


class FakeTokenizer:
    def __init__(self, attention_mask: torch.Tensor):
        self.attention_mask = attention_mask
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        input_ids = torch.arange(self.attention_mask.numel(), dtype=torch.long).view_as(self.attention_mask)
        return FakeBatchEncoding(input_ids=input_ids, attention_mask=self.attention_mask.clone())


class FakeProcessor:
    def __init__(self, attention_mask: torch.Tensor):
        self.attention_mask = attention_mask
        self.calls = []

    def __call__(self, text, images=None, **kwargs):
        self.calls.append({"text": text, "images": images, **kwargs})
        input_ids = torch.arange(self.attention_mask.numel(), dtype=torch.long).view_as(self.attention_mask)
        return FakeBatchEncoding(
            input_ids=input_ids,
            attention_mask=self.attention_mask.clone(),
            pixel_values=torch.ones(self.attention_mask.shape[0], 3, 4, 4),
            image_grid_thw=torch.ones(self.attention_mask.shape[0], 3, dtype=torch.long),
        )


class FakeTextEncoder:
    def __init__(self, hidden_states: torch.Tensor):
        self.hidden_states = hidden_states
        self.dtype = hidden_states.dtype
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(hidden_states=[None, self.hidden_states.clone()])


class FakeTokenizerPipeline:
    def __init__(self, attention_mask: torch.Tensor, hidden_states: torch.Tensor):
        self._execution_device = torch.device("cpu")
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "PROMPT:{}"
        self.prompt_template_encode_start_idx = 2
        self.tokenizer = FakeTokenizer(attention_mask)
        self.text_encoder = FakeTextEncoder(hidden_states)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)


class FakeProcessorPipeline:
    def __init__(self, attention_mask: torch.Tensor, hidden_states: torch.Tensor):
        self._execution_device = torch.device("cpu")
        self.tokenizer_max_length = 1024
        self.prompt_template_encode = "EDIT:{}"
        self.prompt_template_encode_start_idx = 2
        self.processor = FakeProcessor(attention_mask)
        self.text_encoder = FakeTextEncoder(hidden_states)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)


class IdentityProjection(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class FakeAttention:
    def __init__(self):
        self.heads = 1
        self.to_q = IdentityProjection()
        self.to_k = IdentityProjection()
        self.to_v = IdentityProjection()
        self.add_q_proj = IdentityProjection()
        self.add_k_proj = IdentityProjection()
        self.add_v_proj = IdentityProjection()
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None
        self.to_out = [nn.Identity()]
        self.to_add_out = nn.Identity()


class QwenDiffusersOverrideTests(unittest.TestCase):
    def test_flash_attn2_hub_registry_uses_published_kernel_attrs(self):
        flash_config = _HUB_KERNELS_REGISTRY[AttentionBackendName.FLASH_HUB]
        varlen_config = _HUB_KERNELS_REGISTRY[AttentionBackendName.FLASH_VARLEN_HUB]
        original_attrs = (
            flash_config.wrapped_forward_attr,
            flash_config.wrapped_backward_attr,
            varlen_config.wrapped_forward_attr,
            varlen_config.wrapped_backward_attr,
        )
        try:
            flash_config.wrapped_forward_attr = "flash_attn_interface._wrapped_flash_attn_forward"
            flash_config.wrapped_backward_attr = "flash_attn_interface._wrapped_flash_attn_backward"
            varlen_config.wrapped_forward_attr = "flash_attn_interface._wrapped_flash_attn_varlen_forward"
            varlen_config.wrapped_backward_attr = "flash_attn_interface._wrapped_flash_attn_varlen_backward"

            diffusers_overrides.patch_flash_attn2_hub_kernel_attrs()

            self.assertEqual(flash_config.wrapped_forward_attr, "flash_attn_interface._flash_attn_forward")
            self.assertEqual(flash_config.wrapped_backward_attr, "flash_attn_interface._flash_attn_backward")
            self.assertEqual(varlen_config.wrapped_forward_attr, "flash_attn_interface._flash_attn_varlen_forward")
            self.assertEqual(varlen_config.wrapped_backward_attr, "flash_attn_interface._flash_attn_varlen_backward")
        finally:
            (
                flash_config.wrapped_forward_attr,
                flash_config.wrapped_backward_attr,
                varlen_config.wrapped_forward_attr,
                varlen_config.wrapped_backward_attr,
            ) = original_attrs

    def test_ring_anything_attention_accepts_four_dim_lse(self):
        class FakeRingMesh:
            def get_group(self):
                return object()

        parallel_config = SimpleNamespace(
            context_parallel_config=SimpleNamespace(
                _ring_mesh=FakeRingMesh(),
                _ring_local_rank=0,
                ring_degree=2,
                convert_to_fp32=False,
            )
        )
        query = torch.randn(1, 2, 1, 4)
        key = torch.randn(1, 2, 1, 4)
        value = torch.randn(1, 2, 1, 4)

        def forward_op(ctx, query, key, value, *args, **kwargs):
            out = torch.ones_like(query) * (1.0 if kwargs["_save_ctx"] else 2.0)
            lse = torch.zeros(query.shape[0], query.shape[1], query.shape[2], 1, dtype=query.dtype)
            return out, lse

        kv_buffer = torch.cat([key.flatten(), value.flatten()]).contiguous()

        with (
            patch.object(attention_dispatch, "gather_size_by_comm", return_value=[2, 2]),
            patch.object(attention_dispatch.funcol, "all_gather_tensor", return_value=torch.cat([kv_buffer, kv_buffer])),
        ):
            output, lse = diffusers_overrides._patched_ring_anything_attention_forward(
                SimpleNamespace(),
                query,
                key,
                value,
                None,
                0.0,
                False,
                None,
                False,
                True,
                forward_op,
                object(),
                parallel_config,
            )

        self.assertEqual(output.shape, query.shape)
        self.assertEqual(lse.shape, (1, 2, 1))

    def test_diffusers_qwen_pipeline_is_monkey_patched(self):
        self.assertIs(
            QwenImagePipeline._get_qwen_prompt_embeds,
            diffusers_overrides._patched_qwen_prompt_embeds_from_tokenizer,
        )

    def test_tokenizer_prompt_patch_always_pads_to_1024(self):
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        hidden_states = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)
        pipeline = FakeTokenizerPipeline(attention_mask, hidden_states)

        prompt_embeds, prompt_mask = diffusers_overrides._patched_qwen_prompt_embeds_from_tokenizer(
            pipeline,
            prompt=["alpha", "beta"],
        )

        self.assertEqual(prompt_embeds.shape, torch.Size([2, 1024, 3]))
        self.assertEqual(prompt_mask.shape, torch.Size([2, 1024]))
        self.assertEqual(prompt_mask[0].sum().item(), 2)
        self.assertEqual(prompt_mask[1].sum().item(), 4)
        self.assertTrue(torch.equal(prompt_embeds[0, 2:], torch.zeros_like(prompt_embeds[0, 2:])))
        self.assertTrue(torch.equal(prompt_embeds[1, 4:], torch.zeros_like(prompt_embeds[1, 4:])))
        self.assertEqual(pipeline.tokenizer.calls[0]["padding"], "max_length")
        self.assertEqual(pipeline.tokenizer.calls[0]["max_length"], 1026)

    def test_processor_prompt_patch_always_pads_to_1024(self):
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype=torch.long,
        )
        hidden_states = torch.arange(2 * 5 * 4, dtype=torch.float32).view(2, 5, 4)
        pipeline = FakeProcessorPipeline(attention_mask, hidden_states)

        prompt_embeds, prompt_mask = diffusers_overrides._patched_qwen_prompt_embeds_from_processor(
            pipeline,
            prompt=["alpha", "beta"],
            image=[object(), object()],
        )

        self.assertEqual(prompt_embeds.shape, torch.Size([2, 1024, 4]))
        self.assertEqual(prompt_mask.shape, torch.Size([2, 1024]))
        self.assertEqual(prompt_mask[0].sum().item(), 1)
        self.assertEqual(prompt_mask[1].sum().item(), 3)
        self.assertTrue(torch.equal(prompt_embeds[0, 1:], torch.zeros_like(prompt_embeds[0, 1:])))
        self.assertTrue(torch.equal(prompt_embeds[1, 3:], torch.zeros_like(prompt_embeds[1, 3:])))
        self.assertEqual(pipeline.processor.calls[0]["padding"], "max_length")
        self.assertEqual(pipeline.processor.calls[0]["max_length"], 1026)
        self.assertTrue(pipeline.processor.calls[0]["truncation"])
        self.assertEqual(
            pipeline.processor.calls[0]["text"],
            [
                "EDIT:Picture 1: <|vision_start|><|image_pad|><|vision_end|>alpha",
                "EDIT:Picture 1: <|vision_start|><|image_pad|><|vision_end|>beta",
            ],
        )

    def test_processor_prompt_patch_preserves_single_prompt_multi_image_placeholders(self):
        attention_mask = torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long)
        hidden_states = torch.arange(1 * 5 * 4, dtype=torch.float32).view(1, 5, 4)
        pipeline = FakeProcessorPipeline(attention_mask, hidden_states)

        diffusers_overrides._patched_qwen_prompt_embeds_from_processor(
            pipeline,
            prompt=["alpha"],
            image=[object(), object()],
        )

        self.assertEqual(
            pipeline.processor.calls[0]["text"],
            [
                "EDIT:Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
                "Picture 2: <|vision_start|><|image_pad|><|vision_end|>alpha"
            ],
        )

    def test_patched_double_stream_attention_zeros_padded_text_positions(self):
        processor = QwenDoubleStreamAttnProcessor2_0()
        attn = FakeAttention()

        hidden_states = torch.tensor(
            [
                [[0.5, 0.25]],
                [[1.5, 0.75]],
            ],
            dtype=torch.float32,
        )
        encoder_hidden_states = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [99.0, 99.0], [98.0, 98.0]],
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [97.0, 97.0]],
            ],
            dtype=torch.float32,
        )
        joint_attention_mask = torch.tensor(
            [
                [[[1, 1, 0, 0, 1]]],
                [[[1, 1, 1, 0, 1]]],
            ],
            dtype=torch.bool,
        )

        img_attn_output, txt_attn_output = processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=joint_attention_mask,
        )

        self.assertEqual(img_attn_output.shape, torch.Size([2, 1, 2]))
        self.assertEqual(txt_attn_output.shape, torch.Size([2, 4, 2]))
        self.assertTrue(torch.equal(txt_attn_output[0, 2:], torch.zeros_like(txt_attn_output[0, 2:])))
        self.assertTrue(torch.equal(txt_attn_output[1, 3:], torch.zeros_like(txt_attn_output[1, 3:])))


if __name__ == "__main__":
    unittest.main()
