import logging
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager, _module_on_device, prepare_musubi_model_for_ddp


class _QuantizedBuffer(torch.Tensor):
    @staticmethod
    def __new__(cls, weight: torch.Tensor, scale: torch.Tensor):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            weight.shape,
            strides=weight.stride(),
            storage_offset=weight.storage_offset(),
            dtype=weight.dtype,
            device=weight.device,
        )

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor):
        self.weight = weight
        self.scale = scale

    def __tensor_flatten__(self):
        return ["weight", "scale"], None

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, _metadata, outer_size=None, outer_stride=None):
        del outer_size, outer_stride
        return cls(inner_tensors["weight"], inner_tensors["scale"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(f"{cls.__name__} does not implement {func}")


_QuantizedBuffer.__module__ = "sdnq.testing"


class MusubiBlockSwapTests(unittest.TestCase):
    def test_prepare_for_ddp_ignores_only_frozen_state(self):
        module = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        module[0].weight.requires_grad_(False)
        module[0].bias.requires_grad_(False)
        module.register_buffer("frozen_scale", torch.ones(1))

        moved, ignored = prepare_musubi_model_for_ddp(module, torch.device("cpu"))

        self.assertEqual(moved, 0)
        self.assertEqual(ignored, 3)
        self.assertEqual(
            module._ddp_params_and_buffers_to_ignore,
            {"0.weight", "0.bias", "frozen_scale"},
        )

    def test_prepare_for_ddp_preserves_existing_ignore_names(self):
        module = nn.Linear(4, 4)
        module.weight.requires_grad_(False)
        module._ddp_params_and_buffers_to_ignore = {"existing"}

        _moved, ignored = prepare_musubi_model_for_ddp(module, torch.device("cpu"))

        self.assertEqual(ignored, 1)
        self.assertEqual(module._ddp_params_and_buffers_to_ignore, {"existing", "weight"})

    def _accelerator_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        self.skipTest("No accelerator device available for block swap transfer tests")

    def test_quanto_qlinear_streams_without_apply_swap(self):
        try:
            from optimum.quanto import freeze, qint8, quantize
            from optimum.quanto.nn.qlinear import QLinear
        except ImportError as exc:
            self.skipTest(f"Quanto int8 quantization is unavailable: {exc}")

        device = self._accelerator_device()
        block = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        quantize(block, weights=qint8)
        freeze(block)
        qlinear = block[0]
        self.assertIsInstance(qlinear, QLinear)

        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )

        with patch.object(QLinear, "_apply", side_effect=RuntimeError("_apply(): Couldn't swap QLinear.weight")):
            manager.stream_in(block, device)
            self.assertTrue(_module_on_device(block, device))
            self.assertEqual(qlinear.weight.device.type, device.type)
            self.assertEqual(qlinear.weight._data.device.type, device.type)
            self.assertEqual(qlinear.weight._scale.device.type, device.type)
            output = block(torch.randn(2, 4, device=device))
            self.assertEqual(output.device.type, device.type)

            manager.stream_out(block)
            self.assertTrue(_module_on_device(block, torch.device("cpu")))
            self.assertEqual(qlinear.weight.device.type, "cpu")
            self.assertEqual(qlinear.weight._data.device.type, "cpu")
            self.assertEqual(qlinear.weight._scale.device.type, "cpu")

    def test_sdnq_module_streams_without_apply_swap(self):
        device = self._accelerator_device()

        class FakeSDNQLinear(nn.Linear):
            pass

        FakeSDNQLinear.__module__ = "sdnq.training.layers.linear"
        block = nn.Sequential(FakeSDNQLinear(4, 4), nn.SiLU(), FakeSDNQLinear(4, 4))
        for param in block.parameters():
            param.requires_grad_(False)
            param.sdnq_dequantizer = object()
            param.weight = param.detach().clone()
            param.scale = torch.ones(param.shape[0], 1, device=param.device)
        block.to(device)

        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )

        with patch.object(FakeSDNQLinear, "_apply", side_effect=RuntimeError("_apply(): Couldn't swap SDNQLinear.weight")):
            manager.stream_out(block)
            self.assertTrue(_module_on_device(block, torch.device("cpu")))
            self.assertEqual(block[0].weight.device.type, "cpu")
            self.assertEqual(block[0].weight.scale.device.type, "cpu")

            manager.stream_in(block, device)
            self.assertTrue(_module_on_device(block, device))
            self.assertEqual(block[0].weight.device.type, device.type)
            self.assertEqual(block[0].weight.scale.device.type, device.type)
            output = block(torch.randn(2, 4, device=device))
            self.assertEqual(output.device.type, device.type)

    def test_stream_out_keeps_trainable_params_on_accelerator(self):
        device = self._accelerator_device()
        block = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        for param in block.parameters():
            param.requires_grad_(False)
        block.register_parameter("adapter_weight", nn.Parameter(torch.ones(4, device=device)))
        block.register_buffer("adapter_scalar", torch.ones((), device=device))
        block.to(device)

        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )

        manager.stream_out(block)

        self.assertEqual(block.adapter_weight.device.type, device.type)
        self.assertEqual(block.adapter_scalar.device.type, device.type)
        self.assertEqual(block[0].weight.device.type, "cpu")

    @unittest.skipUnless(
        torch.cuda.is_available(),
        "CUDA is required for the block-swap training lifecycle test",
    )
    def test_checkpointed_training_reoffloads_frozen_blocks_after_backward(self):
        device = torch.device("cuda")

        class AdapterBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(64, 64, bias=False)
                self.adapter_down = nn.Linear(64, 8, bias=False)
                self.adapter_up = nn.Linear(8, 64, bias=False)
                self.base.weight.requires_grad_(False)

            def forward(self, hidden_states):
                return torch.nn.functional.silu(self.base(hidden_states) + self.adapter_up(self.adapter_down(hidden_states)))

        blocks = nn.ModuleList([AdapterBlock().to(device) for _ in range(3)])
        manager = MusubiBlockSwapManager(
            block_indices=list(range(len(blocks))),
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )
        manager.activate(blocks, device, grad_enabled=True)
        optimizer = torch.optim.SGD(
            [parameter for block in blocks for parameter in block.parameters() if parameter.requires_grad],
            lr=0.01,
        )

        hidden_states = torch.randn(2, 16, 64, device=device, requires_grad=True)
        for block in blocks:
            manager.stream_in(block, device)
            hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            manager.stream_out(block)

        self.assertTrue(all(block.base.weight.device.type == "cpu" for block in blocks))
        loss = hidden_states.square().mean()
        loss.backward()

        self.assertTrue(all(block.base.weight.device.type == "cpu" for block in blocks))
        self.assertTrue(all(block.adapter_down.weight.device.type == "cuda" for block in blocks))
        self.assertTrue(all(block.adapter_up.weight.device.type == "cuda" for block in blocks))
        self.assertTrue(all(block.adapter_down.weight.grad is not None for block in blocks))
        self.assertTrue(all(block.adapter_up.weight.grad is not None for block in blocks))
        optimizer.step()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the H2D ring test")
    def test_checkpointed_blocks_reuse_h2d_ring_storage(self):
        device = torch.device("cuda")

        class AdapterBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Linear(64, 64, bias=False)
                self.adapter = nn.Linear(64, 64, bias=False)
                self.base.weight.requires_grad_(False)

            def forward(self, hidden_states):
                return torch.nn.functional.silu(self.base(hidden_states) + self.adapter(hidden_states))

        blocks = nn.ModuleList([AdapterBlock().to(device) for _ in range(2)])
        manager = MusubiBlockSwapManager(
            block_indices=[0, 1],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )
        manager.activate(blocks, device, grad_enabled=True)

        hidden_states = torch.randn(2, 8, 64, device=device, requires_grad=True)
        ring_pointers = []
        for block in blocks:
            manager.stream_in(block, device, checkpointed=True)
            state = manager._h2d_block_states[id(block)]
            ring_pointers.append(state.slot.leaves[0].data_ptr())
            self.assertEqual(state.slot.flat.untyped_storage().data_ptr(), state.slot.leaves[0].untyped_storage().data_ptr())
            self.assertFalse(state.cpu_flat.is_pinned())
            hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
            manager.stream_out(block)

        self.assertEqual(ring_pointers[0], ring_pointers[1])
        copier = manager._h2d_copiers[device]
        self.assertTrue(all(staging.is_pinned() for pool in copier._staging.values() for staging in pool))
        self.assertTrue(all(block.base.weight.device.type == "cpu" for block in blocks))
        hidden_states.square().mean().backward()
        self.assertTrue(all(block.base.weight.device.type == "cpu" for block in blocks))
        self.assertTrue(all(block.adapter.weight.grad is not None for block in blocks))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the quantized H2D ring test")
    def test_checkpointed_quanto_payload_uses_h2d_ring(self):
        try:
            from optimum.quanto import freeze, qint8, quantize
        except ImportError as exc:
            self.skipTest(f"Quanto int8 quantization is unavailable: {exc}")

        device = torch.device("cuda")

        class QuantizedAdapterBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Sequential(nn.Linear(32, 32, bias=False))
                self.adapter = nn.Linear(32, 32, bias=False)

            def forward(self, hidden_states):
                return self.base(hidden_states) + self.adapter(hidden_states)

        block = QuantizedAdapterBlock()
        quantize(block.base, weights=qint8)
        freeze(block.base)
        block.adapter.to(device)
        _moved, ignored = prepare_musubi_model_for_ddp(block, device)
        self.assertIn("base.0.weight", block._ddp_params_and_buffers_to_ignore)
        self.assertGreaterEqual(ignored, 1)
        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )
        manager.activate([block], device, grad_enabled=True)

        hidden_states = torch.randn(2, 8, 32, device=device, requires_grad=True)
        manager.stream_in(block, device, checkpointed=True)
        self.assertEqual(block.base[0].weight.device.type, "cuda")
        self.assertEqual(block.base[0].weight._data.device.type, "cuda")
        hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
        manager.stream_out(block)
        self.assertEqual(block.base[0].weight.device.type, "cpu")
        self.assertEqual(block.base[0].weight._data.device.type, "cpu")

        hidden_states.square().mean().backward()
        self.assertEqual(block.base[0].weight.device.type, "cpu")
        self.assertIsNotNone(block.adapter.weight.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the quantized H2D ring test")
    def test_checkpointed_quantized_buffer_payload_uses_h2d_ring(self):
        device = torch.device("cuda")

        class QuantizedAdapterBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "quantized_weight",
                    _QuantizedBuffer(torch.randn(32, 32), torch.ones(32, 1)),
                )
                self.adapter = nn.Linear(32, 32, bias=False)

            def forward(self, hidden_states):
                base = hidden_states @ self.quantized_weight.weight.t()
                return base * self.quantized_weight.scale.mean() + self.adapter(hidden_states)

        block = QuantizedAdapterBlock()
        block.adapter.to(device)
        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )
        manager.activate([block], device, grad_enabled=True)

        hidden_states = torch.randn(2, 8, 32, device=device, requires_grad=True)
        manager.stream_in(block, device, checkpointed=True)
        self.assertEqual(block.quantized_weight.weight.device.type, "cuda")
        self.assertEqual(block.quantized_weight.scale.device.type, "cuda")
        hidden_states = checkpoint(block, hidden_states, use_reentrant=False)
        manager.stream_out(block)
        self.assertEqual(block.quantized_weight.weight.device.type, "cpu")
        self.assertEqual(block.quantized_weight.scale.device.type, "cpu")

        hidden_states.square().mean().backward()
        self.assertEqual(block.quantized_weight.weight.device.type, "cpu")
        self.assertIsNotNone(block.adapter.weight.grad)


if __name__ == "__main__":
    unittest.main()
