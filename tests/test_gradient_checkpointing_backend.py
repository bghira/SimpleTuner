"""
Tests for gradient checkpointing backend selection.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn


class SimpleModule(nn.Module):
    """Simple module for testing checkpointing."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class TestOffloadedGradientCheckpointer(unittest.TestCase):
    """Tests for the offloaded gradient checkpointer."""

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dim = 64
        self.batch_size = 4

    def test_offloaded_checkpoint_forward_pass(self):
        """Test that offloaded checkpoint produces correct forward output."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        module = SimpleModule(self.dim).to(self.device)
        x = torch.randn(self.batch_size, self.dim, device=self.device)

        # Direct forward
        expected = module(x)

        # Checkpointed forward
        result = offloaded_checkpoint(module, x, use_reentrant=False)

        self.assertTrue(torch.allclose(expected, result, atol=1e-6))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for offload test")
    def test_offloaded_checkpoint_backward_pass(self):
        """Test that offloaded checkpoint computes correct gradients."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        # Create two identical modules
        module1 = SimpleModule(self.dim).to(self.device)
        module2 = SimpleModule(self.dim).to(self.device)
        module2.load_state_dict(module1.state_dict())

        x1 = torch.randn(self.batch_size, self.dim, device=self.device, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)

        # Direct backward
        out1 = module1(x1)
        loss1 = out1.sum()
        loss1.backward()

        # Checkpointed backward
        out2 = offloaded_checkpoint(module2, x2, use_reentrant=False)
        loss2 = out2.sum()
        loss2.backward()

        # Check gradients match
        for (n1, p1), (n2, p2) in zip(module1.named_parameters(), module2.named_parameters()):
            self.assertTrue(
                torch.allclose(p1.grad, p2.grad, atol=1e-5),
                f"Gradient mismatch for {n1}",
            )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for offload test")
    def test_offloaded_checkpoint_tuple_output(self):
        """Test that offloaded checkpoint handles tuple outputs correctly."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        class TupleModule(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.linear = nn.Linear(dim, dim)

            def forward(self, x, y):
                return self.linear(x), self.linear(y)

        module = TupleModule(self.dim).to(self.device)
        x = torch.randn(self.batch_size, self.dim, device=self.device)
        y = torch.randn(self.batch_size, self.dim, device=self.device)

        # Direct forward
        expected_x, expected_y = module(x, y)

        # Checkpointed forward
        result_x, result_y = offloaded_checkpoint(module, x, y, use_reentrant=False)

        self.assertTrue(torch.allclose(expected_x, result_x, atol=1e-6))
        self.assertTrue(torch.allclose(expected_y, result_y, atol=1e-6))

    def test_cpu_offload_hooks_pack_unpack(self):
        """Test that CPUOffloadHooks correctly packs and unpacks tensors."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks

        hooks = CPUOffloadHooks()

        if torch.cuda.is_available():
            tensor = torch.randn(4, 4, device="cuda", requires_grad=True) * 2
            packed = hooks.pack(tensor)
            # Pack returns (cpu_tensor, original_device[, pool_key]) tuple
            self.assertIsInstance(packed, tuple)
            self.assertGreaterEqual(len(packed), 2)
            cpu_tensor, original_device = packed[:2]
            self.assertEqual(cpu_tensor.device.type, "cpu")
            self.assertEqual(original_device.type, "cuda")

            unpacked = hooks.unpack(packed)
            self.assertEqual(unpacked.device.type, "cuda")
        else:
            # On CPU, tensors should pass through with None device
            tensor = torch.randn(4, 4)
            packed = hooks.pack(tensor)
            self.assertIsInstance(packed, tuple)
            cpu_tensor, original_device = packed
            self.assertEqual(cpu_tensor.device.type, "cpu")
            self.assertIsNone(original_device)

    def test_activation_offload_does_not_recompute_forward(self):
        """Activation offload preserves the original forward graph instead of rematerializing it."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import activation_offload

        class CountingModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
                self.linear = nn.Linear(4, 4)

            def forward(self, x):
                self.calls += 1
                return torch.relu(self.linear(x))

        module = CountingModule()
        x = torch.randn(2, 4, requires_grad=True)
        activation_offload(module, x).sum().backward()

        self.assertEqual(module.calls, 1)

    def test_activation_offload_pin_memory_bucket_setting(self):
        """Pinned bucket limit is configurable."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            get_activation_offload_pin_memory_max_buckets,
            set_activation_offload_pin_memory_max_buckets,
        )

        original = get_activation_offload_pin_memory_max_buckets()
        try:
            set_activation_offload_pin_memory_max_buckets(7)
            self.assertEqual(get_activation_offload_pin_memory_max_buckets(), 7)
            set_activation_offload_pin_memory_max_buckets(0)
            self.assertEqual(get_activation_offload_pin_memory_max_buckets(), 0)
        finally:
            set_activation_offload_pin_memory_max_buckets(original)

    def test_activation_offload_copy_stream_counts_are_configurable(self):
        """Activation offload copy stream pools expose bounded tunable widths."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            get_activation_offload_copy_stream_stats,
            get_activation_offload_d2h_copy_stream_count,
            get_activation_offload_h2d_prefetch_stream_count,
            set_activation_offload_d2h_copy_stream_count,
            set_activation_offload_h2d_prefetch_stream_count,
        )

        original_d2h = get_activation_offload_d2h_copy_stream_count()
        original_h2d = get_activation_offload_h2d_prefetch_stream_count()
        try:
            set_activation_offload_d2h_copy_stream_count(3)
            set_activation_offload_h2d_prefetch_stream_count(5)

            self.assertEqual(get_activation_offload_d2h_copy_stream_count(), 3)
            self.assertEqual(get_activation_offload_h2d_prefetch_stream_count(), 5)
            stats = get_activation_offload_copy_stream_stats()
            self.assertEqual(stats["d2h"]["width"], 3)
            self.assertEqual(stats["h2d_prefetch"]["width"], 5)

            set_activation_offload_d2h_copy_stream_count(0)
            set_activation_offload_h2d_prefetch_stream_count(-1)
            self.assertEqual(get_activation_offload_d2h_copy_stream_count(), 1)
            self.assertEqual(get_activation_offload_h2d_prefetch_stream_count(), 1)
        finally:
            set_activation_offload_d2h_copy_stream_count(original_d2h)
            set_activation_offload_h2d_prefetch_stream_count(original_h2d)

    def test_pinned_memory_pool_tracks_reuse_stats(self):
        """Pinned pool stats track allocation, release, and buffer reuse per bucket."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import _PinnedMemoryPool

        pool = _PinnedMemoryPool(max_buckets=1)
        pool._allocate = lambda key: torch.empty_strided(key.size, key.stride, dtype=key.dtype, layout=key.layout)
        key = pool.key_for(torch.empty(2, 3))

        first = pool.checkout(key)
        pool.release_after_cuda_copy(key, first, torch.device("cpu"))
        second = pool.checkout(key)

        self.assertIs(first, second)
        snapshot = pool.snapshot()
        bucket = snapshot["buckets"][0]
        self.assertEqual(snapshot["total_accesses"], 2)
        self.assertEqual(snapshot["total_allocations"], 1)
        self.assertEqual(snapshot["total_buffer_reuses"], 1)
        self.assertEqual(bucket["accesses"], 2)
        self.assertEqual(bucket["allocations"], 1)
        self.assertEqual(bucket["buffer_reuses"], 1)
        self.assertEqual(bucket["releases"], 1)

    def test_pinned_memory_pool_eviction_uses_persisted_stats(self):
        """Repeated non-resident shapes can evict colder resident buckets without losing old counters."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import _PinnedMemoryPool

        pool = _PinnedMemoryPool(max_buckets=1)
        pool._allocate = lambda key: torch.empty_strided(key.size, key.stride, dtype=key.dtype, layout=key.layout)
        key_a = pool.key_for(torch.empty(2, 3))
        key_b = pool.key_for(torch.empty(4, 5))

        first = pool.checkout(key_a)
        pool.release_after_cuda_copy(key_a, first, torch.device("cpu"))

        self.assertIsNone(pool.checkout(key_b))
        second = pool.checkout(key_b)

        self.assertIsNotNone(second)
        snapshot = pool.snapshot()
        buckets = {(bucket["size"], bucket["stride"]): bucket for bucket in snapshot["buckets"]}
        bucket_a = buckets[((2, 3), (3, 1))]
        bucket_b = buckets[((4, 5), (5, 1))]

        self.assertFalse(bucket_a["resident"])
        self.assertEqual(bucket_a["evictions"], 1)
        self.assertEqual(bucket_a["accesses"], 1)
        self.assertTrue(bucket_b["resident"])
        self.assertEqual(bucket_b["accesses"], 2)
        self.assertEqual(bucket_b["cap_misses"], 1)
        self.assertEqual(bucket_b["admissions"], 1)
        self.assertEqual(snapshot["tracked_buckets"], 2)
        self.assertEqual(snapshot["total_evictions"], 1)

    def test_cpu_offload_dense_noncontiguous_views_use_flat_transfer(self):
        """Dense views can transfer as flat storage and restore their original logical stride."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks

        hooks = CPUOffloadHooks()
        base = torch.arange(12).view(3, 4)
        transposed = base.t()

        flat = hooks._flat_storage_view(transposed)

        self.assertIsNotNone(flat)
        self.assertEqual(tuple(flat.shape), (12,))
        self.assertEqual(tuple(flat.stride()), (1,))
        self.assertEqual(flat.storage_offset(), transposed.storage_offset())

        transfer_tensor, restore_view = hooks._transfer_view(transposed)
        restored = hooks.unpack((transfer_tensor.clone(), torch.device("cpu"), None, restore_view))

        self.assertEqual(tuple(transfer_tensor.shape), (12,))
        self.assertEqual(tuple(restored.shape), tuple(transposed.shape))
        self.assertEqual(tuple(restored.stride()), tuple(transposed.stride()))
        self.assertTrue(torch.equal(restored, transposed))

    def test_cpu_offload_sparse_storage_views_keep_original_layout(self):
        """Views with holes in storage are not flattened because storage-order transfer would lose layout."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks

        hooks = CPUOffloadHooks()
        sparse_view = torch.arange(12).view(3, 4)[:, ::2]

        flat = hooks._flat_storage_view(sparse_view)
        transfer_tensor, restore_view = hooks._transfer_view(sparse_view)

        self.assertIsNone(flat)
        self.assertIs(transfer_tensor, sparse_view)
        self.assertIsNone(restore_view)

    def test_cpu_offload_flat_transfer_key_ignores_dense_view_stride(self):
        """Pinned bucket keys are based on transfer shape, not dense view logical stride."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks, _PinnedMemoryPool

        hooks = CPUOffloadHooks()
        pool = _PinnedMemoryPool(max_buckets=2)
        base = torch.empty(3, 4)
        transposed = base.t()

        base_transfer, base_restore = hooks._transfer_view(base)
        transposed_transfer, transposed_restore = hooks._transfer_view(transposed)
        base_key = pool.key_for(base_transfer)
        transposed_key = pool.key_for(transposed_transfer)

        self.assertEqual(base_key, transposed_key)
        self.assertEqual(base_key.size, (12,))
        self.assertEqual(base_key.stride, (1,))
        self.assertEqual(base_restore.size, (3, 4))
        self.assertEqual(transposed_restore.size, (4, 3))
        self.assertEqual(transposed_restore.stride, (1, 4))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for pinned offload test")
    def test_activation_offload_pin_memory_bucket_limit(self):
        """New shapes fall back to pageable CPU once the pinned bucket cap is reached."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            CPUOffloadHooks,
            get_activation_offload_pin_memory_max_buckets,
            get_activation_offload_pin_memory_stats,
            reset_activation_offload_pin_memory_stats,
            set_activation_offload_pin_memory_max_buckets,
        )

        original = get_activation_offload_pin_memory_max_buckets()
        try:
            set_activation_offload_pin_memory_max_buckets(0)
            reset_activation_offload_pin_memory_stats()
            set_activation_offload_pin_memory_max_buckets(1)
            hooks = CPUOffloadHooks()

            first = hooks.pack(torch.randn(4, 4, device="cuda", requires_grad=True) * 2)
            second = hooks.pack(torch.randn(8, 8, device="cuda", requires_grad=True) * 2)

            self.assertTrue(first[0].is_pinned())
            self.assertIsNotNone(first[2])
            self.assertIsNone(second[2])
            self.assertEqual(get_activation_offload_pin_memory_stats()["total_cap_misses"], 1)
        finally:
            set_activation_offload_pin_memory_max_buckets(original)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for copy stream offload test")
    def test_cpu_offload_uses_copy_stream_ready_event(self):
        """Pinned CUDA offload returns a ready event and restores dense view strides."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import CPUOffloadHooks

        hooks = CPUOffloadHooks()
        tensor = (torch.arange(12, device="cuda", dtype=torch.float32, requires_grad=True) * 2).view(3, 4).t()

        packed = hooks.pack(tensor)

        self.assertGreaterEqual(len(packed), 5)
        self.assertIsNotNone(packed[2])
        self.assertIsNotNone(packed[3])
        self.assertIsNotNone(packed[4])

        unpacked = hooks.unpack(packed)
        torch.cuda.synchronize()

        self.assertEqual(unpacked.device.type, "cuda")
        self.assertEqual(tuple(unpacked.shape), tuple(tensor.shape))
        self.assertEqual(tuple(unpacked.stride()), tuple(tensor.stride()))
        self.assertTrue(torch.equal(unpacked, tensor.detach()))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for copy stream pool test")
    def test_activation_offload_copy_stream_pools_round_robin_by_direction(self):
        """D2H offload and H2D prefetch use independent round-robin stream pools."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            _ACTIVATION_PREFETCH_RUNTIME,
            _PINNED_MEMORY_POOL,
            CPUOffloadHooks,
            _OffloadedActivationRecord,
            get_activation_offload_copy_stream_stats,
            get_activation_offload_d2h_copy_stream_count,
            get_activation_offload_h2d_prefetch_stream_count,
            reset_activation_offload_copy_stream_stats,
            reset_activation_offload_prefetch_stats,
            set_activation_offload_d2h_copy_stream_count,
            set_activation_offload_h2d_prefetch_stream_count,
        )

        original_d2h = get_activation_offload_d2h_copy_stream_count()
        original_h2d = get_activation_offload_h2d_prefetch_stream_count()
        try:
            set_activation_offload_d2h_copy_stream_count(2)
            set_activation_offload_h2d_prefetch_stream_count(2)
            reset_activation_offload_prefetch_stats()
            reset_activation_offload_copy_stream_stats()

            hooks = CPUOffloadHooks()
            hooks.pack(torch.randn(32, 32, device="cuda", requires_grad=True) * 2)
            hooks.pack(torch.randn(32, 32, device="cuda", requires_grad=True) * 3)

            pinned_a = torch.empty(32, 32, pin_memory=True)
            pinned_b = torch.empty(32, 32, pin_memory=True)
            pool_key = _PINNED_MEMORY_POOL.key_for(pinned_a)
            for logical_id, tensor in (("a", pinned_a), ("b", pinned_b)):
                _ACTIVATION_PREFETCH_RUNTIME.register(
                    _OffloadedActivationRecord(
                        logical_id=logical_id,
                        predictor_id=logical_id,
                        generation=0,
                        tensor=tensor,
                        original_device=torch.device("cuda"),
                        pool_key=pool_key,
                        restore_view=None,
                        ready_event=None,
                    )
                )
                self.assertTrue(_ACTIVATION_PREFETCH_RUNTIME.prefetch(0, logical_id))

            torch.cuda.synchronize()
            stats = get_activation_offload_copy_stream_stats()
            d2h_devices = list(stats["d2h"]["devices"].values())
            h2d_devices = list(stats["h2d_prefetch"]["devices"].values())

            self.assertEqual(d2h_devices[0]["uses"], [1, 1])
            self.assertEqual(h2d_devices[0]["uses"], [1, 1])
        finally:
            reset_activation_offload_prefetch_stats()
            set_activation_offload_d2h_copy_stream_count(original_d2h)
            set_activation_offload_h2d_prefetch_stream_count(original_h2d)

    def test_activation_offload_prefetch_learns_stable_successors(self):
        """Activation prefetch learns on stable predictor ids instead of unique payload ids."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            _ACTIVATION_PREFETCH_RUNTIME,
            _OffloadedActivationRecord,
            get_activation_offload_prefetch_enabled,
            reset_activation_offload_prefetch_stats,
            set_activation_offload_prefetch_enabled,
        )

        reset_activation_offload_prefetch_stats()
        original_enabled = get_activation_offload_prefetch_enabled()
        try:
            set_activation_offload_prefetch_enabled(False)
            for generation in range(2):
                first = _OffloadedActivationRecord(
                    logical_id=f"{generation}:a:payload",
                    predictor_id="block.attn:0",
                    generation=generation,
                    tensor=torch.empty(1),
                    original_device=torch.device("cpu"),
                    pool_key=None,
                    restore_view=None,
                    ready_event=None,
                )
                second = _OffloadedActivationRecord(
                    logical_id=f"{generation}:b:payload",
                    predictor_id="block.attn:1",
                    generation=generation,
                    tensor=torch.empty(1),
                    original_device=torch.device("cpu"),
                    pool_key=None,
                    restore_view=None,
                    ready_event=None,
                )
                _ACTIVATION_PREFETCH_RUNTIME.register(first)
                _ACTIVATION_PREFETCH_RUNTIME.register(second)
                _ACTIVATION_PREFETCH_RUNTIME.consume(first)
                _ACTIVATION_PREFETCH_RUNTIME.consume(second)

            stats = _ACTIVATION_PREFETCH_RUNTIME.snapshot()
            self.assertEqual(_ACTIVATION_PREFETCH_RUNTIME.successors["block.attn:0"], "block.attn:1")
            self.assertEqual(stats["learned_successors"], 1)
            self.assertEqual(stats["transition_updates"], 1)
        finally:
            set_activation_offload_prefetch_enabled(original_enabled)
            reset_activation_offload_prefetch_stats()

    def test_activation_offload_prefetch_retires_unconsumed_generation_records(self):
        """Unpacked-only subsets should not strand pinned buffers in the prefetch runtime."""
        from simpletuner.helpers.training import offloaded_gradient_checkpointer as offload

        original_pool = offload._PINNED_MEMORY_POOL
        original_runtime = offload._ACTIVATION_PREFETCH_RUNTIME
        pool = offload._PinnedMemoryPool(max_buckets=1)
        pool._allocate = lambda key: torch.empty_strided(key.size, key.stride, dtype=key.dtype, layout=key.layout)
        runtime = offload._ActivationOffloadPrefetchRuntime()
        try:
            offload._PINNED_MEMORY_POOL = pool
            offload._ACTIVATION_PREFETCH_RUNTIME = runtime
            key = pool.key_for(torch.empty(2, 3))
            cpu_tensor = pool.checkout(key)
            record = offload._OffloadedActivationRecord(
                logical_id="0:unused:payload",
                predictor_id="unused",
                generation=0,
                tensor=cpu_tensor,
                original_device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                pool_key=key,
                restore_view=None,
                ready_event=None,
            )

            runtime.register(record)
            runtime.saw_unpack_since_pack = True

            self.assertEqual(runtime.snapshot()["active_records"], 1)
            self.assertEqual(runtime.next_generation_for_pack(), 1)

            snapshot = runtime.snapshot()
            pool_snapshot = pool.snapshot()
            self.assertEqual(snapshot["active_records"], 0)
            self.assertTrue(record.cpu_released)
            self.assertEqual(pool_snapshot["buckets"][0]["releases"], 1)
            self.assertEqual(pool_snapshot["buckets"][0]["available_buffers"], 1)
        finally:
            offload._PINNED_MEMORY_POOL = original_pool
            offload._ACTIVATION_PREFETCH_RUNTIME = original_runtime

    def test_activation_offload_prefetch_autotune_disables_worse_prefetch(self):
        """Autotune disables prefetch when measured waits are not better than JIT restore."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            _ACTIVATION_PREFETCH_RUNTIME,
            reset_activation_offload_prefetch_stats,
        )

        reset_activation_offload_prefetch_stats()
        try:
            for _ in range(8):
                _ACTIVATION_PREFETCH_RUNTIME.record_jit_restore_ms(1.0)
                _ACTIVATION_PREFETCH_RUNTIME.record_prefetch_wait_ms(1.1)

            stats = _ACTIVATION_PREFETCH_RUNTIME.snapshot()
            self.assertTrue(stats["autotune_disabled"])
            self.assertEqual(stats["autotune_decision"], "jit")
        finally:
            reset_activation_offload_prefetch_stats()


class TestGradientCheckpointingBackend(unittest.TestCase):
    """Tests for the gradient checkpointing backend module."""

    def test_trainer_prefetch_autotune_uses_model_predict_and_discards_gradients(self):
        """Trainer-level prefetch autotune probes the real prediction/loss path without retaining grads."""
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import (
            activation_offload_prefetch_autotune_decision,
            reset_activation_offload_prefetch_stats,
        )
        from simpletuner.helpers.training.trainer import Trainer

        class FakeAccelerator:
            def __init__(self):
                self.backward_calls = 0

            def backward(self, loss):
                self.backward_calls += 1
                loss.backward()

        class FakeModel:
            def __init__(self, component):
                self.component = component

            def get_trained_component(self, unwrap_model=False):
                return self.component

            def loss_with_logs(self, prepared_batch, model_output, apply_conditioning_mask=True):
                return model_output.square().mean(), {}

            def auxiliary_loss(self, prepared_batch, model_output, loss):
                return loss, {}

        trainer = Trainer.__new__(Trainer)
        trainer.config = SimpleNamespace(
            gradient_checkpointing_offload_prefetch=True,
            gradient_checkpointing_offload_attention=True,
            disable_accelerator=False,
            distillation_method=None,
        )
        trainer.probe_component = nn.Linear(2, 2)
        trainer.model = FakeModel(trainer.probe_component)
        trainer.optimizer = torch.optim.SGD(trainer.probe_component.parameters(), lr=0.1)
        trainer.sidecar_optimizer = None
        trainer.accelerator = FakeAccelerator()
        trainer.predict_calls = 0

        def model_predict(prepared_batch):
            trainer.predict_calls += 1
            return trainer.probe_component(prepared_batch["x"])

        trainer.model_predict = model_predict
        prepared_batch = {"x": torch.ones(1, 2)}

        reset_activation_offload_prefetch_stats()
        with (
            mock.patch("simpletuner.helpers.training.trainer.torch.cuda.is_available", return_value=True),
            mock.patch("simpletuner.helpers.training.trainer.torch.cuda.synchronize"),
            mock.patch("simpletuner.helpers.training.trainer.torch.cuda.get_rng_state_all", return_value=[]),
            mock.patch("simpletuner.helpers.training.trainer.torch.cuda.set_rng_state_all"),
        ):
            trainer._maybe_autotune_activation_offload_prefetch(prepared_batch)

        self.assertEqual(trainer.predict_calls, 3)
        self.assertEqual(trainer.accelerator.backward_calls, 3)
        self.assertIsNone(trainer.probe_component.weight.grad)
        self.assertIsNone(trainer.probe_component.bias.grad)
        self.assertIn(activation_offload_prefetch_autotune_decision(), {"jit", "prefetch"})
        reset_activation_offload_prefetch_stats()

    def test_set_checkpoint_backend(self):
        """Test that checkpoint backend can be set."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            set_checkpoint_backend,
        )

        # Default should be torch
        original = get_checkpoint_backend()

        try:
            set_checkpoint_backend("unsloth")
            self.assertEqual(get_checkpoint_backend(), "unsloth")

            set_checkpoint_backend("torch-ffn")
            self.assertEqual(get_checkpoint_backend(), "torch-ffn")

            set_checkpoint_backend("unsloth-ffn")
            self.assertEqual(get_checkpoint_backend(), "unsloth-ffn")

            set_checkpoint_backend("torch")
            self.assertEqual(get_checkpoint_backend(), "torch")
        finally:
            # Restore original
            set_checkpoint_backend(original)

    def test_set_checkpoint_backend_validation(self):
        """Test that invalid backend values raise ValueError."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import set_checkpoint_backend

        with self.assertRaises(ValueError) as cm:
            set_checkpoint_backend("invalid_backend")

        self.assertIn("invalid_backend", str(cm.exception))
        self.assertIn("torch", str(cm.exception))
        self.assertIn("unsloth", str(cm.exception))

    def test_checkpoint_backend_scope(self):
        """Test backend scope parsing."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend_base,
            get_checkpoint_backend_scope,
        )

        self.assertEqual(get_checkpoint_backend_base("torch"), "torch")
        self.assertEqual(get_checkpoint_backend_scope("torch"), "layer")
        self.assertEqual(get_checkpoint_backend_base("torch-ffn"), "torch")
        self.assertEqual(get_checkpoint_backend_scope("torch-ffn"), "ffn")
        self.assertEqual(get_checkpoint_backend_base("unsloth-ffn"), "unsloth")
        self.assertEqual(get_checkpoint_backend_scope("unsloth-ffn"), "ffn")

    def test_get_checkpoint_function_torch(self):
        """Test that get_checkpoint_function returns torch checkpoint for torch backend."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )

        original_backend = get_checkpoint_backend()

        try:
            set_checkpoint_backend("torch")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, torch.utils.checkpoint.checkpoint)

            set_checkpoint_backend("torch-ffn")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, torch.utils.checkpoint.checkpoint)
        finally:
            set_checkpoint_backend(original_backend)

    def test_get_checkpoint_function_unsloth(self):
        """Test that get_checkpoint_function returns offloaded checkpoint for unsloth backend."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )
        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

        original_backend = get_checkpoint_backend()

        try:
            set_checkpoint_backend("unsloth")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, offloaded_checkpoint)

            set_checkpoint_backend("unsloth-ffn")
            checkpoint_fn = get_checkpoint_function()
            self.assertEqual(checkpoint_fn, offloaded_checkpoint)
        finally:
            set_checkpoint_backend(original_backend)

    def test_checkpoint_function_produces_correct_output(self):
        """Test that checkpoint functions produce correct forward output."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import (
            get_checkpoint_backend,
            get_checkpoint_function,
            set_checkpoint_backend,
        )

        original_backend = get_checkpoint_backend()

        try:
            module = SimpleModule(32)
            x = torch.randn(2, 32)

            # Direct forward
            expected = module(x)

            # Test with torch backend
            set_checkpoint_backend("torch")
            checkpoint_fn = get_checkpoint_function()
            result_torch = checkpoint_fn(module, x, use_reentrant=False)
            self.assertTrue(torch.allclose(expected, result_torch, atol=1e-6))

            # Test with unsloth backend (only if CUDA available)
            if torch.cuda.is_available():
                module = module.cuda()
                x = x.cuda()
                expected = module(x)

                set_checkpoint_backend("unsloth")
                checkpoint_fn = get_checkpoint_function()
                result_unsloth = checkpoint_fn(module, x, use_reentrant=False)
                self.assertTrue(torch.allclose(expected, result_unsloth, atol=1e-6))
        finally:
            set_checkpoint_backend(original_backend)

    def test_checkpoint_sequential_state_matches_direct_gradients(self):
        """Test segmented checkpointing over a tuple-carrying block sequence."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state

        class TupleBlock(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.x_proj = nn.Linear(dim, dim)
                self.y_proj = nn.Linear(dim, dim)

            def forward(self, x, y):
                next_x = torch.relu(self.x_proj(x) + y)
                next_y = torch.relu(self.y_proj(y) + next_x)
                return next_x, next_y

        direct_blocks = nn.ModuleList([TupleBlock(8) for _ in range(4)])
        checkpointed_blocks = nn.ModuleList([TupleBlock(8) for _ in range(4)])
        checkpointed_blocks.load_state_dict(direct_blocks.state_dict())

        direct_x = torch.randn(2, 8, requires_grad=True)
        direct_y = torch.randn(2, 8, requires_grad=True)
        checkpointed_x = direct_x.detach().clone().requires_grad_(True)
        checkpointed_y = direct_y.detach().clone().requires_grad_(True)

        x, y = direct_x, direct_y
        for block in direct_blocks:
            x, y = block(x, y)
        direct_loss = x.sum() + y.sum()
        direct_loss.backward()

        def run_block(_index, block, x, y):
            return block(x, y)

        x, y = checkpoint_sequential_state(
            list(checkpointed_blocks),
            2,
            (checkpointed_x, checkpointed_y),
            run_block,
            torch.utils.checkpoint.checkpoint,
            {"use_reentrant": False},
        )
        checkpointed_loss = x.sum() + y.sum()
        checkpointed_loss.backward()

        self.assertTrue(torch.allclose(direct_x.grad, checkpointed_x.grad, atol=1e-6))
        self.assertTrue(torch.allclose(direct_y.grad, checkpointed_y.grad, atol=1e-6))
        for direct_block, checkpointed_block in zip(direct_blocks, checkpointed_blocks):
            for direct_param, checkpointed_param in zip(direct_block.parameters(), checkpointed_block.parameters()):
                self.assertTrue(torch.allclose(direct_param.grad, checkpointed_param.grad, atol=1e-6))

    def test_checkpoint_sequential_state_uses_contiguous_chunks(self):
        """Test that segment_size controls contiguous chunk boundaries."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state

        calls = []

        def checkpoint_fn(function, *args, **_kwargs):
            calls.append("checkpoint")
            return function(*args)

        def run_block(index, block, x):
            calls.append(index)
            return x + block

        (result,) = checkpoint_sequential_state(
            [1, 2, 3, 4, 5],
            2,
            (torch.tensor(0),),
            run_block,
            checkpoint_fn,
            {"use_reentrant": False},
        )

        self.assertEqual(result.item(), 15)
        self.assertEqual(calls, ["checkpoint", 0, 1, "checkpoint", 2, 3, "checkpoint", 4])

    def test_checkpoint_sequential_state_segment_stride_runs_gaps_without_checkpoint(self):
        """Test that segment_stride leaves deterministic eager gaps between chunks."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state

        calls = []

        def checkpoint_fn(function, *args, **_kwargs):
            calls.append("checkpoint")
            return function(*args)

        def run_block(index, block, x):
            calls.append(index)
            return x + block

        (result,) = checkpoint_sequential_state(
            [1, 2, 3, 4, 5, 6],
            2,
            (torch.tensor(0),),
            run_block,
            checkpoint_fn,
            {"use_reentrant": False},
            segment_stride=4,
        )

        self.assertEqual(result.item(), 21)
        self.assertEqual(calls, ["checkpoint", 0, 1, 2, 3, "checkpoint", 4, 5])

    def test_checkpoint_sequential_state_rejects_overlapping_stride(self):
        """Test that overlapping segment schedules are rejected."""
        from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state

        with self.assertRaisesRegex(ValueError, "segment_stride"):
            checkpoint_sequential_state(
                [1, 2],
                2,
                (torch.tensor(0),),
                lambda _index, block, x: x + block,
                lambda function, *args, **_kwargs: function(*args),
                segment_stride=1,
            )


class TestConfigFieldIntegration(unittest.TestCase):
    """Tests for the configuration field integration."""

    def test_gradient_checkpointing_backend_field_exists(self):
        """Test that the gradient_checkpointing_backend field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_backend")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, "torch")
        self.assertIn({"value": "torch", "label": "PyTorch layer (recompute)"}, field.choices)
        self.assertIn({"value": "torch-ffn", "label": "PyTorch FFN-only (recompute)"}, field.choices)
        self.assertIn({"value": "unsloth", "label": "Unsloth layer (CPU offload)"}, field.choices)
        self.assertIn({"value": "unsloth-ffn", "label": "Unsloth FFN-only (CPU offload)"}, field.choices)

    def test_gradient_checkpointing_offload_attention_field_exists(self):
        """Test that the attention activation offload field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_offload_attention")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, False)
        self.assertEqual(field.arg_name, "--gradient_checkpointing_offload_attention")
        self.assertEqual(field.dependencies, [])

    def test_gradient_checkpointing_offload_pin_memory_max_buckets_field_exists(self):
        """Test that the attention offload pinned bucket field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_offload_pin_memory_max_buckets")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, 12)
        self.assertEqual(field.arg_name, "--gradient_checkpointing_offload_pin_memory_max_buckets")
        self.assertEqual(len(field.dependencies), 1)
        self.assertEqual(field.dependencies[0].field, "gradient_checkpointing_offload_attention")

    def test_gradient_checkpointing_offload_prefetch_field_exists(self):
        """Test that the attention offload prefetch field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_offload_prefetch")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, False)
        self.assertEqual(field.arg_name, "--gradient_checkpointing_offload_prefetch")

    def test_gradient_checkpointing_backend_validation(self):
        """Test that invalid backend values are rejected."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_backend")

        # Check validation rules
        choices_rule = None
        for rule in field.validation_rules:
            if rule.rule_type.value == "choices":
                choices_rule = rule
                break

        self.assertIsNotNone(choices_rule)
        self.assertIn("torch", choices_rule.value)
        self.assertIn("torch-ffn", choices_rule.value)
        self.assertIn("unsloth", choices_rule.value)
        self.assertIn("unsloth-ffn", choices_rule.value)

    def test_gradient_checkpointing_segment_stride_field_exists(self):
        """Test that the segmented checkpointing stride field is registered."""
        from simpletuner.simpletuner_sdk.server.services.field_registry import FieldRegistry

        registry = FieldRegistry()
        field = registry.get_field("gradient_checkpointing_segment_stride")

        self.assertIsNotNone(field)
        self.assertEqual(field.default_value, None)
        self.assertEqual(field.arg_name, "--gradient_checkpointing_segment_stride")


class TestTransformerBackendAttribute(unittest.TestCase):
    """Tests that transformer models have the backend attribute and setter."""

    def test_flux_transformer_has_backend_attribute(self):
        """Test that FluxTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel

        self.assertTrue(hasattr(FluxTransformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(getattr(FluxTransformer2DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(FluxTransformer2DModel, "_supports_attention_activation_offload", False))

    def test_flux2_transformer_has_checkpointing_support_flags(self):
        """Test that Flux2Transformer2DModel exposes attention offload support."""
        from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel

        self.assertTrue(hasattr(Flux2Transformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(Flux2Transformer2DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(Flux2Transformer2DModel, "_supports_attention_activation_offload", False))

    def test_chroma_transformer_has_checkpointing_support_flags(self):
        """Test that ChromaTransformer2DModel exposes FFN and attention offload support."""
        from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel

        self.assertTrue(hasattr(ChromaTransformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(ChromaTransformer2DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(ChromaTransformer2DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(ChromaTransformer2DModel, "_supports_attention_activation_offload", False))

    def test_krea2_transformer_has_checkpointing_support_flags(self):
        """Test that Krea2Transformer2DModel exposes FFN and attention offload support."""
        from simpletuner.helpers.models.krea2.transformer import Krea2Transformer2DModel

        self.assertTrue(hasattr(Krea2Transformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(Krea2Transformer2DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(Krea2Transformer2DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(Krea2Transformer2DModel, "_supports_attention_activation_offload", False))

    def test_flux_blocks_support_ffn_checkpoint_scope(self):
        """Test that Flux blocks preserve output values with FFN-only checkpointing."""
        from simpletuner.helpers.models.flux.transformer import FluxSingleTransformerBlock, FluxTransformerBlock

        double_block = FluxTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).train()
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        encoder_hidden = torch.randn(2, 3, 16, requires_grad=True)
        temb = torch.randn(2, 16)

        expected_encoder, expected_hidden = double_block(hidden, encoder_hidden, temb)
        actual_encoder, actual_hidden = double_block(
            hidden,
            encoder_hidden,
            temb,
            checkpoint_ffn=True,
            checkpoint_fn=torch.utils.checkpoint.checkpoint,
            offload_attention=True,
        )
        self.assertTrue(torch.allclose(expected_encoder, actual_encoder, atol=1e-6))
        self.assertTrue(torch.allclose(expected_hidden, actual_hidden, atol=1e-6))

        single_block = FluxSingleTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).train()
        hidden = torch.randn(2, 7, 16, requires_grad=True)
        temb = torch.randn(2, 16)

        expected_hidden = single_block(hidden, temb)
        actual_hidden = single_block(
            hidden,
            temb,
            checkpoint_ffn=True,
            checkpoint_fn=torch.utils.checkpoint.checkpoint,
            offload_attention=True,
        )
        self.assertTrue(torch.allclose(expected_hidden, actual_hidden, atol=1e-6))

    def test_sana_transformer_has_backend_attribute(self):
        """Test that SanaTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.sana.transformer import SanaTransformer2DModel

        self.assertTrue(hasattr(SanaTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_sd3_transformer_has_backend_attribute(self):
        """Test that SD3Transformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

        self.assertTrue(hasattr(SD3Transformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(SD3Transformer2DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(SD3Transformer2DModel, "_supports_attention_activation_offload", False))

    def test_chroma_transformer_has_backend_attribute(self):
        """Test that ChromaTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel

        self.assertTrue(hasattr(ChromaTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_auraflow_transformer_has_backend_attribute(self):
        """Test that AuraFlowTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.auraflow.transformer import AuraFlowTransformer2DModel

        self.assertTrue(hasattr(AuraFlowTransformer2DModel, "set_gradient_checkpointing_backend"))

    def test_mageflow_transformer_has_backend_attribute(self):
        """Test that MageFlowTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.mageflow.transformer import MageFlowTransformer2DModel

        self.assertTrue(hasattr(MageFlowTransformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(MageFlowTransformer2DModel, "set_gradient_checkpointing_interval"))
        self.assertTrue(getattr(MageFlowTransformer2DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(MageFlowTransformer2DModel, "_supports_attention_activation_offload", False))

    def test_ltx2_transformer_has_checkpointing_support_flags(self):
        """Test that LTX2VideoTransformer3DModel exposes FFN and attention offload support."""
        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        self.assertTrue(hasattr(LTX2VideoTransformer3DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(LTX2VideoTransformer3DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(LTX2VideoTransformer3DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(LTX2VideoTransformer3DModel, "_supports_attention_activation_offload", False))

    def test_wan_transformer_has_segmented_checkpointing_setters(self):
        """Test that WanTransformer3DModel exposes segmented checkpointing controls."""
        from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel

        self.assertTrue(hasattr(WanTransformer3DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(WanTransformer3DModel, "set_gradient_checkpointing_interval"))
        self.assertTrue(hasattr(WanTransformer3DModel, "set_gradient_checkpointing_segment_stride"))
        self.assertTrue(hasattr(WanTransformer3DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(WanTransformer3DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(WanTransformer3DModel, "_supports_attention_activation_offload", False))

    def test_z_image_transformer_has_checkpointing_support_flags(self):
        """Test that ZImageTransformer2DModel exposes FFN and attention offload support."""
        from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel

        self.assertTrue(hasattr(ZImageTransformer2DModel, "set_gradient_checkpointing_backend"))
        self.assertTrue(hasattr(ZImageTransformer2DModel, "set_gradient_checkpointing_offload_attention"))
        self.assertTrue(getattr(ZImageTransformer2DModel, "_supports_ffn_gradient_checkpointing", False))
        self.assertTrue(getattr(ZImageTransformer2DModel, "_supports_attention_activation_offload", False))

    def test_qwen_image_transformer_has_backend_attribute(self):
        """Test that QwenImageTransformer2DModel has gradient_checkpointing_backend."""
        from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel

        self.assertTrue(hasattr(QwenImageTransformer2DModel, "set_gradient_checkpointing_backend"))


if __name__ == "__main__":
    unittest.main()
