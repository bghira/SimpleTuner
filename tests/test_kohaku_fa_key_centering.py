import unittest

import torch

from scripts.benchmark_attention_precision import fp64_reference, metrics
from simpletuner.helpers.training.kohaku_fa_cute import attention
from simpletuner.helpers.training.kohaku_fa_cute.api import _center_keys


class KeyCenteringTests(unittest.TestCase):
    def test_outputs_and_original_key_gradients_match_uncentered_attention(self):
        torch.manual_seed(42)
        for causal in (False, True):
            for scale in (0.0, -0.3, 0.3):
                with self.subTest(causal=causal, scale=scale):
                    q = torch.randn(2, 4, 5, 7, requires_grad=True)
                    k = (torch.randn(2, 2, 9, 7) + 32).requires_grad_()
                    v = torch.randn_like(k, requires_grad=True)
                    mask = torch.rand(5, 9) > 0.2
                    mask[0] = False
                    if causal:
                        mask &= torch.arange(9)[None, :] <= torch.arange(5)[:, None]
                    centered = _center_keys(k)
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        out = torch.nn.functional.scaled_dot_product_attention(
                            q, centered, v, attn_mask=mask, scale=scale, enable_gqa=True
                        )
                    dout = torch.randn_like(out)
                    gradients = torch.autograd.grad(out, (q, k, v), dout)
                    reference = fp64_reference((q, k, v), dout, mask, scale=scale)
                    for actual, expected in zip((out, *gradients), reference):
                        torch.testing.assert_close(actual.double(), expected, atol=3e-6, rtol=3e-5)
                    torch.testing.assert_close(gradients[1].sum(-2), torch.zeros_like(k[..., 0, :]), atol=1e-6, rtol=0)

    def test_centering_handles_noncontiguous_keys_without_mutating_them(self):
        torch.manual_seed(43)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                keys = (torch.randn(2, 9, 3, 7) + 32).to(dtype).transpose(1, 2)
                original = keys.clone()
                centered = _center_keys(keys)
                self.assertEqual(centered.dtype, dtype)
                self.assertEqual(centered.shape, keys.shape)
                self.assertLess(centered.float().mean(-2).abs().max().item(), 0.005)
                torch.testing.assert_close(keys, original, atol=0, rtol=0)

    def test_centered_strict_cpu_calls_still_reject_unsupported_device(self):
        q = torch.randn(1, 2, 5, 16)
        with self.assertRaisesRegex(ValueError, "CUDA"):
            attention(q, q, q, center_keys=True)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() in {(8, 9), (9, 0), (10, 0), (10, 3), (12, 0)},
    "Requires centered Ada, Hopper or Blackwell kernels",
)
class CuTeKeyCenteringTests(unittest.TestCase):
    def test_compiled_checkpointed_production_block_lora_updates(self):
        from copy import deepcopy
        from unittest.mock import patch

        from diffusers.models.attention import BasicTransformerBlock
        from peft import LoraConfig, get_peft_model
        from torch.utils.checkpoint import checkpoint

        from scripts.benchmark_attention_precision import centered_sdpa
        from simpletuner.helpers.training.kohaku_fa_cute import scaled_dot_product_attention

        torch.manual_seed(47)
        model = get_peft_model(
            BasicTransformerBlock(1280, 20, 64, cross_attention_dim=2048).cuda().bfloat16(),
            LoraConfig(r=8, lora_alpha=8, target_modules="all-linear"),
        ).train()
        x = torch.randn(1, 129, 1280, device="cuda", dtype=torch.bfloat16)
        context = torch.randn(1, 77, 2048, device="cuda", dtype=x.dtype)
        target = torch.randn_like(x)
        histories, updates = [], []
        for sdpa in (scaled_dot_product_attention, centered_sdpa):
            current = deepcopy(model)
            parameters = tuple(p for p in current.parameters() if p.requires_grad)
            initial = torch.cat([p.detach().flatten() for p in parameters])
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            compiled = torch.compile(current, fullgraph=True)
            losses = []
            with patch("torch.nn.functional.scaled_dot_product_attention", sdpa):
                for _ in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    output = checkpoint(compiled, x, encoder_hidden_states=context, use_reentrant=False)
                    loss = (output.float() - target).square().mean()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters))
                    optimizer.step()
                    losses.append(loss.detach())
            histories.append(torch.stack(losses))
            updates.append(torch.cat([p.detach().flatten() for p in parameters]) - initial)
        torch.testing.assert_close(histories[1], histories[0], atol=0.001, rtol=0.01)
        self.assertGreater(updates[1].float().norm().item(), 0)
        self.assertLess((updates[1] - updates[0]).float().norm().item(), 0.05 * updates[0].float().norm().item())
        self.assertLess(histories[1][-1].item(), histories[1][0].item())

    def compare(self, fn, dtype, causal, masked):
        torch.manual_seed(44)
        q = torch.randn(2, 4, 33, 64, device="cuda", dtype=dtype)
        k = (torch.randn(2, 2, 49, 64, device="cuda") + 4).to(dtype)
        v = torch.randn_like(k)
        # Projection layout, GQA, unequal lengths and an all-hidden row.
        inputs = [t.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_() for t in (q, k, v)]
        mask = None
        if masked:
            mask = torch.rand(33, 49, device="cuda") > 0.2
            mask[0] = False
        dout = torch.randn_like(q)
        expected = fp64_reference(inputs, dout, mask, causal, scale=-0.125)
        out = fn(*inputs, mask=mask, causal=causal, scale=-0.125, center_keys=True)
        gradients = torch.autograd.grad(out, inputs, dout)
        for name, actual, reference in zip(("out", "dq", "dk", "dv"), (out, *gradients), expected):
            result = metrics(actual, reference)
            self.assertTrue(result["finite"], name)
            self.assertLess(result["relative_error"], 0.04, name)

    def test_dense_causal_masked_outputs_and_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            for causal, masked in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(dtype=dtype, causal=causal, masked=masked):
                    self.compare(attention, dtype, causal, masked)

    def test_compiled_key_centering(self):
        compiled = torch.compile(attention, fullgraph=True)
        self.compare(compiled, torch.bfloat16, False, False)
        self.compare(compiled, torch.bfloat16, True, True)

    def test_cuda_graph_replay_recomputes_mean_for_updated_keys(self):
        torch.manual_seed(45)
        q, k, v = [torch.randn(1, 2, 17, 64, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                attention(q, k, v, center_keys=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = attention(q, k, v, center_keys=True)
        k.copy_((torch.randn_like(k.float()) + 8).to(k.dtype))
        graph.replay()
        torch.testing.assert_close(out, attention(q, k, v, center_keys=True), atol=0, rtol=0)
