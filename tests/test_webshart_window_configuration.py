"""Configuring a Webshart dataset that is read a window of shards at a time."""

import json
import math
import os
import socket
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.multiprocessing as mp

from simpletuner.helpers.data_backend.config.image import ImageBackendConfig
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime, shuffle_window
from simpletuner.helpers.data_backend.factory import FactoryRegistry, init_backend_config
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from tests.test_webshart_units import USABLE, write_parquet_source

SHARDS = 5
WINDOW = 2


class _StubModel:
    """A model with no VAE, no text encoders and no conditioning requirements."""

    AUTOENCODER_CLASS = None

    def __getattr__(self, name):
        if name.startswith(("requires_", "uses_", "supports_", "is_")):
            return lambda *args, **kwargs: False
        raise AttributeError(name)


def windowed_dataset(root, **overrides):
    return {
        "id": "train",
        "type": "webshart",
        "dataset_type": "image",
        "source": os.path.join(root, "source"),
        "caption_strategy": "webshart",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 32,
        "crop": False,
        "webshart": {"format": "parquet", "shuffle_window": WINDOW, "cache_dir": os.path.join(root, "webshart")},
        **overrides,
    }


def _configure_worker(rank, port, root, world_size, max_train_steps):
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
        OMP_NUM_THREADS="1",
    )
    # Tests that import simpletuner.cli leave this in the environment; a trainer never runs with it.
    os.environ.pop("SIMPLETUNER_SKIP_TORCH", None)
    from accelerate import Accelerator

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
        max_train_steps=max_train_steps,
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
        vae_cache_ondemand=False,
        vae_cache_disable=False,
        print_sampler_statistics=False,
    )
    for name, value in options.items():
        setattr(args, name, value)
    os.makedirs(args.output_dir, exist_ok=True)
    StateTracker.set_args(args)
    StateTracker.set_accelerator(accelerator)
    factory = FactoryRegistry(args=args, accelerator=accelerator, text_encoders=[], tokenizers=[], model=_StubModel())
    train = windowed_dataset(root)
    report_path = Path(root, f"report-{rank}.json")
    try:
        factory._configure_single_data_backend(train, [train], [], [], False)
    except ValueError as error:
        report_path.write_text(json.dumps({"refused": str(error)}))
        return

    backend = StateTracker.get_data_backend("train")
    cursor = StateTracker.get_dataset("train").cursor()
    metadata_backend = backend["metadata_backend"]
    report = {
        "seed": cursor.seed,
        "samples_per_unit": cursor.samples_per_unit,
        "first_window": cursor.units_in_window(),
        "open_after_configuring": sorted(metadata_backend._open_units),
        "batches": len(backend["sampler"]),
        "buckets": metadata_backend.aspect_ratio_bucket_indices,
        "json_files": sorted(path.name for path in Path(root).rglob("*.json") if "source" not in path.parts),
    }
    trainer = object.__new__(Trainer)
    trainer.distiller = None
    trainer.lr_scheduler = None
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(
        distillation_method=None,
        gradient_accumulation_steps=1,
        max_train_steps=max_train_steps,
        num_train_epochs=0,
        strict_epoch_limit=False,
        train_batch_size=2,
    )
    trainer._recalculate_training_steps()
    report["steps_per_epoch"] = trainer.config.num_update_steps_per_epoch
    report["strict_epoch_limit"] = trainer.config.strict_epoch_limit

    # One epoch, read the way the trainer reads it.
    batches = []
    try:
        for batch in backend["sampler"]:
            batches.append(
                [(sample["image_path"], list(sample["target_size"]), sample["instance_prompt_text"]) for sample in batch]
            )
    except MultiDatasetExhausted:
        pass
    report["epoch"] = batches
    report["open_after_the_epoch"] = sorted(metadata_backend._open_units)
    report["epoch_after"] = cursor.epoch
    report_path.write_text(json.dumps(report))
    accelerator.wait_for_everyone()
    accelerator.end_training()


@unittest.skipUnless(
    torch.distributed.is_available() and torch.distributed.is_gloo_available(), "Requires Gloo distributed backend"
)
class TestWindowedDatasetConfiguration(unittest.TestCase):
    def configure(self, world_size, max_train_steps):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        os.makedirs(os.path.join(root.name, "source"))
        write_parquet_source(Path(root.name, "source"), shards=SHARDS)
        context = mp.spawn(
            _configure_worker, args=(port, root.name, world_size, max_train_steps), nprocs=world_size, join=False
        )
        try:
            deadline = time.monotonic() + 120
            while not context.join(timeout=1):
                if time.monotonic() > deadline:
                    self.fail("Windowed dataset configuration timed out")
        finally:
            for process in context.processes:
                if process.is_alive():
                    process.terminate()
        return [json.loads(Path(root.name, f"report-{rank}.json").read_text()) for rank in range(world_size)]

    def test_two_ranks_read_their_own_shards_and_plan_the_same_epoch(self):
        reports = self.configure(world_size=2, max_train_steps=20)
        main, other = reports

        # The run is unseeded, so the ranks had to agree on a seed to deal the shards.
        self.assertEqual(main["seed"], other["seed"])
        # Five shards between two ranks: three and two, none shared.
        self.assertEqual(len(main["first_window"]), WINDOW)
        self.assertEqual(len(other["first_window"]), WINDOW)
        self.assertEqual(set(main["first_window"]) & set(other["first_window"]), set())
        # The main process opened its first window to measure it; the other rank was told the figure.
        self.assertEqual(main["open_after_configuring"], sorted(main["first_window"]))
        self.assertEqual(other["open_after_configuring"], [])
        usable_in_first_window = sum(len(USABLE) + shard for shard in main["first_window"])
        self.assertEqual(main["samples_per_unit"], usable_in_first_window / WINDOW)
        self.assertEqual(other["samples_per_unit"], main["samples_per_unit"])
        # Both plan for the rank with two shards.
        self.assertEqual(main["batches"], other["batches"])
        self.assertEqual(main["batches"], math.ceil(math.ceil(2 * main["samples_per_unit"]) / 2))
        self.assertEqual([report["steps_per_epoch"] for report in reports], [main["batches"]] * 2)
        self.assertEqual([report["strict_epoch_limit"] for report in reports], [False, False])

        trained = []
        for report in reports:
            shards = set()
            for batch in report["epoch"]:
                self.assertEqual(len(batch), 2)
                self.assertEqual(len({tuple(size) for _, size, _ in batch}), 1)
                for path, _, caption in batch:
                    shards.add(int(path.split("/")[2]))
                    self.assertIn(caption, [caption for _, _, caption in USABLE] + [f"extra {n}" for n in range(SHARDS)])
            trained.append(shards)
            # Nothing describing the whole dataset exists, in memory or on disk; only the measured figure.
            self.assertEqual(report["buckets"], {})
            self.assertEqual(report["json_files"], ["samples_per_unit.json"])
            self.assertEqual(report["open_after_the_epoch"], [])
            self.assertEqual(report["epoch_after"], 2)
        self.assertEqual(sorted(len(shards) for shards in trained), [2, 3])
        self.assertEqual(trained[0] | trained[1], set(range(SHARDS)))
        self.assertEqual(trained[0] & trained[1], set())
        samples = [{path for batch in report["epoch"] for path, _, _ in batch} for report in reports]
        self.assertEqual(len(samples[0] | samples[1]), sum(len(USABLE) + shard for shard in range(SHARDS)))

    def test_a_run_measured_in_epochs_is_refused_when_the_dataset_is_configured(self):
        (report,) = self.configure(world_size=1, max_train_steps=0)
        self.assertRegex(report["refused"], r"\(id=train\) webshart.shuffle_window=2 .* Set --max_train_steps\.")


class TestWhatAShuffleWindowRulesOut(unittest.TestCase):
    def factory(self, model=None, **args):
        factory = object.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(
            **{"max_train_steps": 100, "eval_dataset_id": None, "aspect_bucket_disable_rebuild": False, **args}
        )
        factory.model = model or _StubModel()
        return factory

    def refusal(self, factory=None, text_embed_cache=None, **dataset):
        backend = windowed_dataset("/datasets", **dataset)
        args = SimpleNamespace(
            resolution=64, resolution_type="pixel", caption_strategy="filename", vae_cache_ondemand=False, cache_dir="cache"
        )
        init_backend = init_backend_config(backend, args, accelerator=None)
        if text_embed_cache is not None:
            init_backend["text_embed_cache"] = text_embed_cache
        try:
            (factory or self.factory())._refuse_what_a_shuffle_window_rules_out(backend, init_backend)
        except ValueError as error:
            return str(error)
        return None

    def test_a_step_driven_image_dataset_is_accepted(self):
        self.assertIsNone(self.refusal())
        self.assertIsNone(self.refusal(repeats=3))
        self.assertIsNone(self.refusal(text_embed_cache=SimpleNamespace(text_cache_ondemand=True)))

    def test_the_refusal_names_the_dataset_and_the_window(self):
        self.assertTrue(
            self.refusal(max_num_samples=10).startswith(
                "(id=train) webshart.shuffle_window=2 has the dataset indexed a shard at a time while training. "
            )
        )

    def test_a_run_measured_in_epochs_is_refused(self):
        for steps in (0, None):
            factory = self.factory()
            factory.args.max_train_steps = steps
            self.assertIn("Set --max_train_steps.", self.refusal(factory))

    def test_only_image_datasets_are_read_this_way(self):
        self.assertIn("its dataset_type is video", self.refusal(dataset_type="video"))
        self.assertIn("its dataset_type is conditioning", self.refusal(dataset_type="conditioning"))

    def test_the_evaluation_dataset_is_refused(self):
        self.assertIn("--eval_dataset_id", self.refusal(self.factory(eval_dataset_id="train")))
        self.assertIn("--eval_dataset_id", self.refusal(self.factory(eval_dataset_id=["other", "train"])))
        self.assertIsNone(self.refusal(self.factory(eval_dataset_id="other")))

    def test_conditioning_pairs_are_refused(self):
        self.assertIn("conditioning datasets", self.refusal(conditioning_data="control"))
        self.assertIn("conditioning datasets", self.refusal(conditioning=[{"type": "canny"}]))

    def test_settings_that_need_the_whole_dataset_indexed_are_refused(self):
        for key in ("max_num_samples", "scan_for_errors", "vae_cache_clear_each_epoch"):
            with self.subTest(key=key):
                self.assertIn(f"{key} needs the whole dataset indexed before training", self.refusal(**{key: 1}))

    def test_latents_have_to_be_encoded_while_training(self):
        model = _StubModel()
        model.AUTOENCODER_CLASS = object
        self.assertIn("Set vae_cache_ondemand or vae_cache_disable", self.refusal(self.factory(model)))
        self.assertIsNone(self.refusal(self.factory(model), vae_cache_ondemand=True))
        self.assertIsNone(self.refusal(self.factory(model), vae_cache_disable=True))

    def test_latents_are_not_kept_for_a_dataset_cropped_anew_each_epoch(self):
        model = _StubModel()
        model.AUTOENCODER_CLASS = object
        recropped = dict(crop=True, crop_aspect="random", crop_aspect_buckets=[0.5, 1.0, 2.0])
        refusal = self.refusal(self.factory(model), vae_cache_ondemand=True, **recropped)
        self.assertIn("crop_aspect=random gives its samples a new size and crop each epoch", refusal)
        self.assertIn("Set vae_cache_disable on the dataset, or --aspect_bucket_disable_rebuild", refusal)

        self.assertIsNone(self.refusal(self.factory(model), vae_cache_disable=True, **recropped))
        self.assertIsNone(
            self.refusal(self.factory(model, aspect_bucket_disable_rebuild=True), vae_cache_ondemand=True, **recropped)
        )
        self.assertIsNone(self.refusal(self.factory(), **recropped))
        for kept in (dict(crop=True, crop_aspect="square"), dict(recropped, crop=False)):
            with self.subTest(kept=kept):
                self.assertIsNone(self.refusal(self.factory(model), vae_cache_ondemand=True, **kept))

    def test_captions_have_to_be_encoded_while_training(self):
        self.assertIn(
            "Set text_cache_ondemand on its text embed backend",
            self.refusal(text_embed_cache=SimpleNamespace(text_cache_ondemand=False)),
        )

    def test_a_model_that_needs_conditioning_image_embeds_is_refused(self):
        model = _StubModel()
        model.requires_conditioning_image_embeds = lambda: True
        self.assertIn("conditioning image embeds", self.refusal(self.factory(model)))

    def test_a_dataset_without_a_window_is_never_refused(self):
        factory = self.factory()
        factory.args.max_train_steps = 0
        self.assertIsNone(self.refusal(factory, webshart={"format": "parquet"}, max_num_samples=10, scan_for_errors=True))


class TestLatentCacheOfAWindowedDataset(unittest.TestCase):
    def test_the_dataset_is_not_listed_to_map_cache_files_to_samples(self):
        factory = object.__new__(FactoryRegistry)
        factory.args = MagicMock()
        factory.model = MagicMock()
        factory.accelerator = SimpleNamespace(is_local_main_process=True)
        factory._move_text_encoders_for_vae_cache = lambda device: None
        data_backend = MagicMock()
        data_backend.list_files.side_effect = AssertionError("every sample of the dataset was listed")
        backend = windowed_dataset("/datasets", vae_cache_ondemand=True, cache_dir_vae="cache/vae/train")
        init_backend = {
            "id": "train",
            "dataset_type": "image",
            "config": {"webshart": {"shuffle_window": WINDOW}, "vae_cache_ondemand": True},
            "data_backend": data_backend,
            "metadata_backend": MagicMock(),
            "dataset_runtime": MagicMock(),
            "instance_data_dir": "cache/webshart",
        }
        with (
            patch("simpletuner.helpers.data_backend.factory.VAECache") as vae_cache,
            patch("simpletuner.helpers.data_backend.factory.StateTracker") as state_tracker,
        ):
            vae_cache.return_value.vae_cache_ondemand = True
            state_tracker.get_image_files.return_value = None
            factory._configure_vae_cache(backend, init_backend, init_backend, [], [], None)

        vae_cache.return_value.build_vae_cache_filename_map.assert_called_once_with(all_image_files=[])
        data_backend.list_files.assert_not_called()
        state_tracker.get_image_files.assert_not_called()
        state_tracker.set_image_files.assert_not_called()


class TestShuffleWindowOption(unittest.TestCase):
    def test_the_window_is_read_from_the_webshart_block(self):
        self.assertEqual(shuffle_window({"webshart": {"shuffle_window": 4}}), 4)
        self.assertIsNone(shuffle_window({"webshart": {"format": "parquet"}}))
        self.assertIsNone(shuffle_window({"webshart": None}))
        self.assertIsNone(shuffle_window({"type": "local"}))

    def test_the_dataset_config_carries_the_window_to_the_backend(self):
        payload = windowed_dataset("/datasets")
        config = ImageBackendConfig.from_dict(payload, {})
        self.assertEqual(config.webshart_shuffle_window, WINDOW)
        self.assertEqual(config.to_dict()["config"]["webshart"]["shuffle_window"], WINDOW)
        self.assertEqual(shuffle_window(config.to_dict()["config"]), WINDOW)
        from simpletuner.helpers.data_backend.builders.webshart import WebshartBackendBuilder

        with patch("simpletuner.helpers.data_backend.builders.webshart.WebshartDataBackend") as backend:
            WebshartBackendBuilder(None, {})._create_backend(config)
        self.assertEqual(backend.call_args.kwargs["shuffle_window"], WINDOW)

    def test_a_dataset_without_the_option_has_no_window(self):
        payload = windowed_dataset("/datasets", webshart={"format": "parquet"})
        config = ImageBackendConfig.from_dict(payload, {})
        self.assertIsNone(config.webshart_shuffle_window)
        self.assertNotIn("shuffle_window", config.to_dict()["config"]["webshart"])
        args = SimpleNamespace(resolution=64, resolution_type="pixel", caption_strategy="filename", cache_dir="cache")
        self.assertIsNone(shuffle_window(init_backend_config(payload, args, accelerator=None)["config"]))

    def test_the_window_is_named_only_inside_the_webshart_block(self):
        payload = windowed_dataset("/datasets", webshart={"format": "parquet"}, shuffle_window=4)
        args = SimpleNamespace(resolution=64, resolution_type="pixel", caption_strategy="filename", cache_dir="cache")
        self.assertIsNone(shuffle_window(init_backend_config(payload, args, accelerator=None)["config"]))
        self.assertIsNone(ImageBackendConfig.from_dict(payload, {}).webshart_shuffle_window)

    def test_a_window_that_is_not_a_whole_number_of_shards_is_refused_in_the_config(self):
        for value in (0, "4", 1.5, True):
            payload = windowed_dataset("/datasets", webshart={"format": "parquet", "shuffle_window": value})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "whole number of shards, 1 or more"):
                ImageBackendConfig.from_dict(payload, {})


class TestRuntimeWithAWindow(unittest.TestCase):
    """What configure_cursor does with the dataset's unit interface."""

    def runtime(self, pools_per_unit, units=4, dp=(1, 0), repeats=0, recorded=None, main=True):
        self.calls = []
        metadata = MagicMock()
        metadata.unit_count.return_value = units
        metadata.recorded_samples_per_unit.return_value = recorded
        metadata.record_samples_per_unit.side_effect = lambda value: self.calls.append(("record", value))
        metadata.check_window.side_effect = lambda window, ranks: self.calls.append(("check", sorted(window), ranks))

        def open_unit(unit, epoch):
            self.calls.append(("open", unit, epoch))
            return {
                pool: [f"webshart://{unit}/{index}/{pool}.png" for index in range(size)]
                for pool, size in pools_per_unit.items()
            }

        metadata.open_unit.side_effect = open_unit
        runtime = DatasetRuntime("train", MagicMock(), metadata, {"webshart": {"shuffle_window": WINDOW}})
        args = SimpleNamespace(seed=11, gradient_accumulation_steps=1, allow_dataset_oversubscription=False)
        with (
            patch("simpletuner.helpers.data_backend.dataset_runtime.StateTracker.get_args", return_value=args),
            patch(
                "simpletuner.helpers.data_backend.dataset_runtime.StateTracker.get_data_backend_config",
                return_value={"repeats": repeats},
            ),
            patch("simpletuner.helpers.data_backend.dataset_runtime.get_cp_aware_dp_info", return_value=(*dp, 1)),
            patch(
                "simpletuner.helpers.data_backend.dataset_runtime.broadcast_object_from_main",
                side_effect=lambda value: value if main else 5.0,
            ),
        ):
            runtime.configure_cursor(
                accelerator=SimpleNamespace(is_main_process=main), batch_size=2, shared_by_all_ranks=False
            )
        return runtime

    def test_the_window_is_checked_against_the_shard_cache_before_it_is_opened(self):
        runtime = self.runtime({"64x64": 2}, dp=(2, 1))
        window = sorted(runtime.cursor().units_in_window())
        self.assertEqual(self.calls[0], ("check", window, 2))
        self.assertEqual(sorted(self.calls[1:-1]), [("open", unit, 1) for unit in window])

    def test_the_first_measure_of_samples_per_unit_is_recorded(self):
        cursor = self.runtime({"64x64": 4, "96x64": 2}).cursor()
        self.assertEqual(self.calls[-1], ("record", 6.0))
        self.assertEqual(cursor.samples_per_unit, 6.0)

    def test_a_recorded_measure_is_used_and_nothing_is_opened_for_it(self):
        cursor = self.runtime({"64x64": 4}, recorded=7.5).cursor()
        self.assertEqual(cursor.samples_per_unit, 7.5)
        self.assertEqual([call[0] for call in self.calls], ["check"])
        self.assertEqual(len(cursor), 15)  # 4 shards x 7.5 samples / 2
        cursor.next_batch()
        self.assertEqual([call[0] for call in self.calls], ["check", "open", "open"])

    def test_other_ranks_take_the_measure_of_the_main_process(self):
        cursor = self.runtime({"64x64": 4}, dp=(2, 1), main=False).cursor()
        self.assertEqual(cursor.samples_per_unit, 5.0)
        self.assertEqual([call[0] for call in self.calls], ["check"])

    def test_the_cursor_reads_the_dataset_through_its_units_with_the_dataset_repeats(self):
        cursor = self.runtime({"64x64": 4, "96x64": 2}, repeats=1).cursor()
        self.assertEqual((cursor.window, cursor.unit_count, cursor.repeats, cursor.seed), (WINDOW, 4, 1, 11))
        self.assertEqual(cursor.samples_per_unit, 6.0)
        self.assertEqual(len(cursor), 24)  # 4 shards x 6 samples x 2 passes / 2

    def test_bucket_lists_changing_does_not_replace_the_cursor(self):
        runtime = self.runtime({"64x64": 4})
        cursor = runtime.cursor()
        cursor.next_batch()
        runtime.rebuild_cursor()
        self.assertIs(runtime.cursor(), cursor)
        self.assertEqual(cursor.progress()[0], 2)

    def test_a_first_window_without_usable_samples_is_refused_and_not_recorded(self):
        with self.assertRaisesRegex(ValueError, r"\(id=train\) The first 2 shard\(s\) opened hold no usable sample"):
            self.runtime({})
        self.assertNotIn("record", [call[0] for call in self.calls])

    def test_a_dataset_with_fewer_shards_than_ranks_is_refused(self):
        with self.assertRaisesRegex(ValueError, "fewer than the 8 ranks"):
            self.runtime({"64x64": 4}, units=4, dp=(8, 0))


if __name__ == "__main__":
    unittest.main()
