import logging
import os
import random
import socket
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.multiprocessing as mp


def _checkpoint_worker(rank, port, output_dir):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
        OMP_NUM_THREADS="1",
    )
    from accelerate import Accelerator

    from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
    from simpletuner.helpers.multiaspect.state import BucketStateManager
    from simpletuner.helpers.training.save_hooks import SaveHookManager
    from simpletuner.helpers.training.state_tracker import StateTracker
    from simpletuner.helpers.training.trainer import Trainer
    from simpletuner.helpers.utils.checkpoint_manager import CheckpointManager

    torch.set_default_device("cpu")

    class CPUAccelerator(Accelerator):
        def wait_for_everyone(self):
            # Gloo barriers must not receive MPS device IDs from Accelerate.
            torch.distributed.barrier()

    accelerator = CPUAccelerator(cpu=True)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model, optimizer = accelerator.prepare(model, optimizer)
    optimizer.zero_grad()
    accelerator.backward(model(torch.ones(1, 2)).sum())
    optimizer.step()
    trainer = object.__new__(Trainer)
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(
        output_dir=output_dir,
        checkpointing_use_tempdir=True,
        disk_low_threshold=None,
        checkpoints_total_limit=0,
        checkpoints_rolling_total_limit=0,
        num_train_epochs=1,
    )
    trainer.state = {"global_step": 1}
    trainer.job_id = None
    trainer.model = SimpleNamespace()
    trainer.distiller = None
    trainer.hub_manager = None
    trainer.checkpoint_manager = CheckpointManager(output_dir)
    for name in [
        "mark_optimizer_eval",
        "mark_optimizer_train",
        "_emit_event",
        "_run_post_checkpoint_script",
        "_run_post_upload_script",
        "_export_dynamo_cache",
    ]:
        setattr(trainer, name, Mock())
    trainer._prepare_training_progress_payload = Mock(return_value=({}, {}))
    hooks = object.__new__(SaveHookManager)
    hooks.accelerator = accelerator
    hooks.training_state_path = "training_state.json" if rank == 0 else f"training_state-rank{rank}.json"
    hooks.args = SimpleNamespace(model_type="full")
    hooks._is_fsdp2 = lambda: False
    hooks._offload_models_during_save = lambda main: nullcontext()
    hooks._save_ema_state = Mock()
    hooks._save_full_model = Mock()
    hooks.get_modelspec_architecture = Mock(return_value="test/full")
    trainer.model_hooks = hooks
    accelerator.register_save_state_pre_hook(hooks.save_model_hook)

    sampler = object.__new__(MultiAspectSampler)
    sampler.accelerator = accelerator
    sampler.id = "dataset"
    sampler.batch_size = 1
    sampler.buckets = ["1.0"]
    sampler.exhausted_buckets = []
    sampler.current_bucket = 0
    sampler.current_epoch = 3
    sampler.sample_type_strs = "images"
    sampler.logger = logging.getLogger("checkpoint-test")
    sampler.state_manager = BucketStateManager(sampler.id)
    sampler.metadata_backend = SimpleNamespace(
        aspect_ratio_bucket_indices={"1.0": [f"rank{rank}-a", f"rank{rank}-b"]}, seen_images={f"rank{rank}-a": 1}
    )
    StateTracker.set_args(trainer.config)
    original_parameters = [p.detach().clone() for p in model.parameters()]
    for tempdir, rolling in [(True, False), (False, True)]:
        trainer.config.checkpointing_use_tempdir = tempdir
        trainer.state["global_step"] += 1
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        with patch.object(StateTracker, "get_data_backends", return_value={"dataset": {"sampler": sampler}}):
            path = trainer._save_rolling_checkpoint() if rolling else trainer._run_standard_checkpoint(None, None, 0)
        expected = (random.random(), np.random.random(), torch.rand(4))
        manifest = trainer.checkpoint_manager.load_manifest(path)
        for saved_rank in range(2):
            assert f"random_states_{saved_rank}.pkl" in manifest["files"]
            stem = "training_state" if saved_rank == 0 else f"training_state-rank{saved_rank}"
            assert f"{stem}-dataset.json" in manifest["files"]
        assert (Path(path) / ".guard").exists()
        saved_sampler = sampler.state_manager.load_state(str(Path(path) / hooks.training_state_path))
        sampler.metadata_backend.aspect_ratio_bucket_indices = {"wrong": []}
        sampler.metadata_backend.seen_images.clear()
        sampler.current_epoch = 0
        sampler.current_bucket = 99
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(10)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        accelerator.load_state(path)
        actual = (random.random(), np.random.random(), torch.rand(4))
        assert expected[:2] == actual[:2]
        assert torch.equal(expected[2], actual[2])
        assert all(torch.equal(p, old) for p, old in zip(model.parameters(), original_parameters))
        sampler.load_states(str(Path(path) / hooks.training_state_path))
        assert sampler.metadata_backend.aspect_ratio_bucket_indices == saved_sampler["aspect_ratio_bucket_indices"]
        assert sampler.metadata_backend.seen_images == saved_sampler["seen_images"]
        assert sampler.current_epoch == saved_sampler["current_epoch"]
        assert sampler.current_bucket == saved_sampler["current_bucket"]
        accelerator.wait_for_everyone()
    accelerator.end_training()


class TestDistributedCheckpoint(unittest.TestCase):
    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Requires Gloo distributed backend"
    )
    def test_ddp_restores_each_rank_rng_and_sampler(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as output_dir:
            context = mp.spawn(_checkpoint_worker, args=(port, output_dir), nprocs=2, join=False)
            try:
                deadline = time.monotonic() + 120
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Distributed checkpoint test timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)


class TestSharedDistillerCheckpoint(unittest.TestCase):
    def test_shared_distiller_files_have_one_writer(self):
        from simpletuner.helpers.distillation.assistant_lora.distiller import AssistantLoRADistiller
        from simpletuner.helpers.distillation.dmd.distiller import DMDDistiller

        for cls in (AssistantLoRADistiller, DMDDistiller):
            for main in (False, True):
                with self.subTest(distiller=cls.__name__, main=main), tempfile.TemporaryDirectory() as directory:
                    distiller = object.__new__(cls)
                    distiller.teacher_model = SimpleNamespace(accelerator=SimpleNamespace(is_main_process=main))
                    if cls is DMDDistiller:
                        distiller.fake_score_transformer = torch.nn.Linear(2, 1)
                        distiller.fake_score_optimizer = torch.optim.AdamW(distiller.fake_score_transformer.parameters())
                        expected = {"fake_score_transformer.safetensors", "fake_score_transformer_optim.pt"}
                    else:
                        distiller._generation_index = 3
                        distiller._generated_samples = 6
                        distiller._seed = 42
                        distiller.config = {"resolutions": [[512, 512]], "num_inference_steps": 4}
                        expected = {"assistant_lora_state.json"}
                    distiller.on_save_checkpoint(3, directory)
                    self.assertEqual({p.name for p in Path(directory).iterdir()}, expected if main else set())
