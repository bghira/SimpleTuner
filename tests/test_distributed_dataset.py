import json
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
import torch.multiprocessing as mp
from PIL import Image

WORLD_SIZE = 2
DROPPED = "img00.png"


class _StubModel:
    """A model with no VAE, no text encoders and no conditioning requirements."""

    AUTOENCODER_CLASS = None

    def __getattr__(self, name):
        if name.startswith(("requires_", "uses_", "supports_", "is_")):
            return lambda *args, **kwargs: False
        raise AttributeError(name)


def _configure_worker(rank, port, root):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
        OMP_NUM_THREADS="1",
    )
    # Tests that import simpletuner.cli leave this in the environment; a trainer never runs with it.
    os.environ.pop("SIMPLETUNER_SKIP_TORCH", None)
    from accelerate import Accelerator

    from simpletuner.helpers.data_backend.factory import FactoryRegistry
    from simpletuner.helpers.training.state_tracker import StateTracker
    from simpletuner.helpers.training.trainer import Trainer

    torch.set_default_device("cpu")

    class CPUAccelerator(Accelerator):
        def wait_for_everyone(self):
            # Gloo barriers must not receive MPS device IDs from Accelerate.
            torch.distributed.barrier()

    accelerator = CPUAccelerator(cpu=True)
    args = MagicMock()
    options = dict(
        model_type="full",
        model_family="stub",
        resolution=64,
        resolution_type="pixel",
        minimum_image_size=0,
        maximum_image_size=None,
        target_downsample_size=None,
        cache_dir=os.path.join(root, "cache"),
        cache_dir_text=os.path.join(root, "text"),
        cache_dir_vae=os.path.join(root, "vae"),
        output_dir=os.path.join(root, "output"),
        train_batch_size=2,
        gradient_accumulation_steps=1,
        skip_file_discovery="",
        caption_strategy="filename",
        prepend_instance_prompt=False,
        instance_prompt=None,
        only_instance_prompt=False,
        seed=None,
        allow_dataset_oversubscription=False,
        eval_dataset_id=None,
        override_dataset_config=False,
        cache_file_suffix=None,
        delete_problematic_images=False,
        delete_unwanted_images=False,
        debug_aspect_buckets=False,
        metadata_update_interval=3600,
        disable_bucket_pruning=False,
        aspect_bucket_rounding=2,
        aspect_bucket_alignment=8,
        enable_multiprocessing=False,
        max_workers=2,
        image_processing_batch_size=4,
        ignore_missing_files=False,
        dataloader_prefetch=False,
        ltx_train_mode=None,
        datasets_base_dir=None,
    )
    for name, value in options.items():
        setattr(args, name, value)
    os.makedirs(args.output_dir, exist_ok=True)
    StateTracker.set_args(args)
    StateTracker.set_accelerator(accelerator)
    factory = FactoryRegistry(args=args, accelerator=accelerator, text_encoders=[], tokenizers=[], model=_StubModel())

    scan = factory._handle_error_scanning_and_metadata

    def scan_and_drop_a_sample(backend, init_backend, conditioning_type):
        # Stands in for any preparation step that rejects a sample: it happens on the main process only.
        if accelerator.is_main_process and init_backend["id"] == "train":
            init_backend["metadata_backend"].remove_image(os.path.join(root, "train", DROPPED))
        scan(backend, init_backend, conditioning_type)

    factory._handle_error_scanning_and_metadata = scan_and_drop_a_sample
    common = {"type": "local", "resolution": 64, "resolution_type": "pixel", "minimum_image_size": 0, "crop": False}
    train = {"id": "train", "dataset_type": "image", "instance_data_dir": os.path.join(root, "train"), **common}
    control = {
        "id": "control",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "train",
        "instance_data_dir": os.path.join(root, "control"),
        **common,
    }
    for backend in (train, control):
        factory._configure_single_data_backend(backend, [train, control], [], [], False)

    report = {}
    for dataset_id in ("train", "control"):
        cursor = StateTracker.get_dataset(dataset_id).cursor()
        metadata_backend = StateTracker.get_data_backend(dataset_id)["metadata_backend"]
        report[dataset_id] = {
            "share": cursor.rank_share(),
            "batches": len(StateTracker.get_data_backend(dataset_id)["sampler"]),
            "seed": cursor.seed,
            "buckets": metadata_backend.aspect_ratio_bucket_indices,
            "read_only": metadata_backend.read_only,
        }
    trainer = object.__new__(Trainer)
    trainer.distiller = None
    trainer.lr_scheduler = None
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(
        distillation_method=None,
        gradient_accumulation_steps=1,
        max_train_steps=0,
        num_train_epochs=2,
        strict_epoch_limit=False,
        train_batch_size=2,
    )
    trainer._recalculate_training_steps()
    report["steps_per_epoch"] = trainer.config.num_update_steps_per_epoch
    report["max_train_steps"] = trainer.config.max_train_steps
    Path(root, f"report-{rank}.json").write_text(json.dumps(report))
    accelerator.wait_for_everyone()
    accelerator.end_training()


class TestDistributedDatasetConfiguration(unittest.TestCase):
    @unittest.skipUnless(
        torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Requires Gloo distributed backend"
    )
    def test_ranks_agree_on_the_dataset_and_divide_it(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as root:
            names = [f"img{index:02d}.png" for index in range(12)]
            for folder in ("train", "control"):
                os.makedirs(os.path.join(root, folder))
                for index, name in enumerate(names):
                    Image.new("RGB", (64, 64) if index < 8 else (96, 64)).save(os.path.join(root, folder, name))
            # A mirrored dataset is copied from its source on every start; what an earlier run left
            # behind must not replace the copy.
            stale = {"config": {}, "aspect_ratio_bucket_indices": {"1.0": [os.path.join(root, "control", "stale.png")]}}
            Path(root, "control", "aspect_ratio_bucket_indices_control.json").write_text(json.dumps(stale))

            context = mp.spawn(_configure_worker, args=(port, root), nprocs=WORLD_SIZE, join=False)
            try:
                deadline = time.monotonic() + 120
                while not context.join(timeout=1):
                    if time.monotonic() > deadline:
                        self.fail("Distributed dataset configuration timed out")
            finally:
                for process in context.processes:
                    if process.is_alive():
                        process.terminate()
            reports = [json.loads(Path(root, f"report-{rank}.json").read_text()) for rank in range(WORLD_SIZE)]

            kept = [os.path.join(root, "train", name) for name in names if name != DROPPED]
            train = [report["train"] for report in reports]
            # Only the main process dropped the sample; every rank has to end up without it.
            for report in train:
                self.assertEqual(sorted(path for bucket in report["buckets"].values() for path in bucket), kept)
            self.assertEqual([report["read_only"] for report in train], [False, True])
            # The run is unseeded, so the ranks had to agree on a seed to divide the dataset.
            self.assertEqual(train[0]["seed"], train[1]["seed"])
            shares = [[path for bucket in report["share"].values() for path in bucket] for report in train]
            self.assertEqual(sorted(shares[0] + shares[1]), kept)
            self.assertEqual({bucket: len(paths) for bucket, paths in train[0]["share"].items()}, {"1.0": 4, "1.5": 2})
            self.assertEqual({bucket: len(paths) for bucket, paths in train[1]["share"].items()}, {"1.0": 3, "1.5": 2})
            # A rank with the shorter share wraps, so every rank steps the same number of times.
            self.assertEqual([report["batches"] for report in train], [3, 3])
            # The trainer plans its epochs from those batches; the conditioning dataset adds none.
            self.assertEqual([report["steps_per_epoch"] for report in reports], [3, 3])
            self.assertEqual([report["max_train_steps"] for report in reports], [6, 6])

            # The conditioning dataset mirrors the training dataset and is whole on every rank.
            mirrored = [path.replace(os.path.join(root, "train"), os.path.join(root, "control"), 1) for path in kept]
            for report in (report["control"] for report in reports):
                self.assertEqual(sorted(path for bucket in report["buckets"].values() for path in bucket), mirrored)
                self.assertEqual(sorted(path for bucket in report["share"].values() for path in bucket), mirrored)
