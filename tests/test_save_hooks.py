import json
import multiprocessing
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from accelerate.utils import DistributedType

from simpletuner.helpers.training.save_hooks import LORA_SAFETENSORS_FILENAME, SaveHookManager
from simpletuner.helpers.training.state_tracker import StateTracker


def _concurrent_writer(directory, barrier, iterations, errors):
    from simpletuner.helpers.training.state_tracker import StateTracker as tracker

    for _ in range(iterations):
        barrier.wait()
        try:
            tracker.save_ramtorch_prefetch_orders(directory)
        except FileNotFoundError:
            with errors.get_lock():
                errors.value += 1


class SaveHookManagerTests(unittest.TestCase):
    def _run_save_model_hook(self, is_local_main_process):
        output_dir = "/tmp/checkpoint"

        with (
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_training_state") as save_state,
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_ramtorch_prefetch_orders") as save_orders,
        ):
            manager = object.__new__(SaveHookManager)
            manager.accelerator = SimpleNamespace(
                distributed_type=DistributedType.NO,
                is_main_process=is_local_main_process,
                is_local_main_process=is_local_main_process,
            )
            manager.args = SimpleNamespace(model_type="full")
            manager.training_state_path = "training_state.json"
            manager._offload_models_during_save = Mock(return_value=nullcontext())
            manager._save_ema_state = Mock()
            manager._is_fsdp2 = Mock(return_value=False)
            manager._save_full_model = Mock()

            manager.save_model_hook([], [], output_dir)

        return save_state, save_orders

    def test_ramtorch_prefetch_orders_are_saved_only_on_local_main_process(self):
        for is_local_main_process in (True, False):
            with self.subTest(is_local_main_process=is_local_main_process):
                _, save_orders = self._run_save_model_hook(is_local_main_process)
                self.assertEqual(save_orders.call_count, int(is_local_main_process))

    def test_ramtorch_prefetch_orders_ignore_global_main_flag(self):
        # A non-zero node's local main has is_main_process=False but must still
        # write, so the file lands on that node's storage in multi-node runs.
        output_dir = "/tmp/checkpoint"

        with (
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_training_state"),
            patch("simpletuner.helpers.training.save_hooks.StateTracker.save_ramtorch_prefetch_orders") as save_orders,
        ):
            manager = object.__new__(SaveHookManager)
            manager.accelerator = SimpleNamespace(
                distributed_type=DistributedType.NO,
                is_main_process=False,
                is_local_main_process=True,
            )
            manager.args = SimpleNamespace(model_type="full")
            manager.training_state_path = "training_state.json"
            manager._offload_models_during_save = Mock(return_value=nullcontext())
            manager._save_ema_state = Mock()
            manager._is_fsdp2 = Mock(return_value=False)
            manager._save_full_model = Mock()

            manager.save_model_hook([], [], output_dir)

        self.assertEqual(save_orders.call_count, 1)

    def test_training_state_is_saved_on_every_rank(self):
        for is_local_main_process in (True, False):
            with self.subTest(is_local_main_process=is_local_main_process):
                save_state, _ = self._run_save_model_hook(is_local_main_process)
                self.assertEqual(save_state.call_count, 1)

    def _adapter_hook_manager(self, output_dir, is_main_process=True):
        lycoris_config_path = Path(output_dir) / "source_lycoris.json"
        lycoris_config_path.write_text(json.dumps({"algo": "lokr"}))
        network = torch.nn.Module()
        network.lokr_w1 = torch.nn.Parameter(torch.ones(2))
        manager = object.__new__(SaveHookManager)
        manager.accelerator = SimpleNamespace(
            distributed_type=DistributedType.FSDP,
            is_main_process=is_main_process,
            is_local_main_process=is_main_process,
            _lycoris_wrapped_network=network,
            wait_for_everyone=Mock(),
        )
        manager.args = SimpleNamespace(
            model_type="lora", lora_type="lycoris", lycoris_config=str(lycoris_config_path), use_ema=False
        )
        manager.training_state_path = "training_state.json"
        manager._build_modelspec_metadata = Mock(return_value={})
        return manager

    def test_fsdp2_lycoris_save_gathers_on_every_rank_and_writes_on_main(self):
        for is_main_process in (True, False):
            with self.subTest(is_main_process=is_main_process), tempfile.TemporaryDirectory() as tmpdir:
                manager = self._adapter_hook_manager(tmpdir, is_main_process=is_main_process)
                manager._offload_models_during_save = Mock(return_value=nullcontext())
                manager._is_fsdp2 = Mock(return_value=True)
                network = manager.accelerator._lycoris_wrapped_network
                network.state_dict = Mock(wraps=network.state_dict)
                with patch("simpletuner.helpers.training.save_hooks.StateTracker"):
                    manager.save_model_hook([], [], tmpdir)

                network.state_dict.assert_called_once()
                self.assertEqual((Path(tmpdir) / LORA_SAFETENSORS_FILENAME).exists(), is_main_process)

    def test_fsdp2_adapter_resume_relies_on_accelerate_shards(self):
        for lora_type, loader in (("lycoris", "_load_lycoris"), ("standard", "_load_lora")):
            for is_fsdp2 in (True, False):
                with self.subTest(lora_type=lora_type, is_fsdp2=is_fsdp2), tempfile.TemporaryDirectory() as tmpdir:
                    manager = self._adapter_hook_manager(tmpdir)
                    manager.args.lora_type = lora_type
                    manager._is_fsdp2 = Mock(return_value=is_fsdp2)
                    setattr(manager, loader, Mock())
                    with patch("simpletuner.helpers.training.save_hooks.StateTracker"):
                        manager.load_model_hook([], tmpdir)

                    self.assertEqual(getattr(manager, loader).call_count, int(not is_fsdp2))


class RamtorchPrefetchOrderWriterTests(unittest.TestCase):
    def test_writer_leaves_only_the_final_file_behind(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            StateTracker.reset_ramtorch_prefetch_orders()
            StateTracker.save_ramtorch_prefetch_orders(tmpdir)

            entries = sorted(p.name for p in Path(tmpdir).iterdir())
            self.assertEqual(entries, ["ramtorch_prefetch_orders.json"])
            with (Path(tmpdir) / "ramtorch_prefetch_orders.json").open() as handle:
                self.assertEqual(json.load(handle), {"version": 1, "components": {}})

    def test_concurrent_writers_do_not_race(self):
        # Two node-mains on a shared filesystem write the same path at once.
        # With a shared temp filename this raises FileNotFoundError almost every
        # barrier-synchronised round; the process-unique temp name makes every
        # write an atomic last-writer-wins rename.
        ctx = multiprocessing.get_context("spawn")
        with tempfile.TemporaryDirectory() as tmpdir:
            barrier = ctx.Barrier(2, timeout=60)
            errors = ctx.Value("i", 0)
            workers = [ctx.Process(target=_concurrent_writer, args=(tmpdir, barrier, 30, errors)) for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(120)

            self.assertEqual(errors.value, 0)
            entries = sorted(p.name for p in Path(tmpdir).iterdir())
            self.assertEqual(entries, ["ramtorch_prefetch_orders.json"])
            with (Path(tmpdir) / "ramtorch_prefetch_orders.json").open() as handle:
                json.load(handle)


if __name__ == "__main__":
    unittest.main()
