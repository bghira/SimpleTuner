import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.prompts import CaptionNotFoundError
from simpletuner.helpers.training.state_tracker import StateTracker


class DatasetRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.metadata = {
            "original_size": (512, 512),
            "target_size": (512, 512),
            "crop_coordinates": (0, 0),
            "captions": [" first ", "second"],
            "webshart": {"shard_idx": 2, "sample_idx": 19},
        }
        self.metadata_backend = Mock()
        self.metadata_backend.get_metadata_by_filepath.return_value = self.metadata
        self.metadata_backend.caption_cache_entry.return_value = self.metadata["captions"]
        self.data_backend = Mock(id="images")
        self.config = {
            "dataset_type": "image",
            "caption_strategy": "webshart",
            "use_captions": True,
            "prepend_instance_prompt": False,
            "instance_prompt": None,
            "disable_multiline_split": False,
        }
        self.runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, self.config)

    def test_caption_traversal_is_lazy_and_keeps_all_alternatives(self):
        visited = []

        def selection():
            for ref in ("one.jpg", "two.jpg", "one.jpg"):
                visited.append(ref)
                yield ref

        records = self.runtime.iter_caption_records(selection())
        self.assertEqual(visited, [])
        self.assertEqual(next(records), ("first", "one.jpg"))
        self.assertEqual(next(records), ("second", "one.jpg"))
        self.assertEqual(visited, ["one.jpg"])
        self.assertEqual(
            list(records), [("first", "two.jpg"), ("second", "two.jpg"), ("first", "one.jpg"), ("second", "one.jpg")]
        )

    def test_caption_traversal_reports_missing_samples_without_collecting_them(self):
        self.metadata_backend.caption_cache_entry.side_effect = [None, "caption"]
        missing = []
        records = self.runtime.iter_caption_records(["missing.jpg", "ok.jpg"], on_missing_caption=missing.append)
        self.assertEqual(list(records), [("caption", "ok.jpg")])
        self.assertEqual(missing, ["missing.jpg"])

    def test_text_preparation_preserves_model_keys_metadata_and_grounding_labels(self):
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        self.metadata["bbox_entities"] = [{"label": "cat"}, {"label": ""}]
        config = {**self.config, "instance_data_dir": "dataset"}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        model = SimpleNamespace(
            text_embed_cache_key=lambda: TextEmbedCacheKey.DATASET_AND_FILENAME,
            text_embed_cache_metadata_for_filepath=lambda **kwargs: {"height": 512},
            text_embed_cache_key_value=lambda prompt, default_key, metadata: f"{default_key}:{metadata['height']}",
        )
        records = list(runtime.iter_text_embed_records(model, iter(["sample.jpg"]), init_backend={}))
        self.assertEqual([record["prompt"] for record in records], ["first", "cat", "second", "cat"])
        self.assertEqual(records[0]["key"], "images:sample.jpg:512")
        self.assertEqual(records[0]["metadata"]["dataset_relative_path"], "sample.jpg")
        self.assertEqual(records[1], {"prompt": "cat", "key": "sample.jpg__bbox_0", "metadata": {"grounding_entity": True}})

    def test_reads_preserve_order_and_duplicates_with_independent_nested_metadata(self):
        refs = ["webshart://2/19/a.jpg", "webshart://0/3/b.jpg", "webshart://2/19/a.jpg"]
        records = self.runtime.read_samples(refs)
        self.assertEqual([record["image_path"] for record in records], refs)
        self.assertEqual([record["data_backend_id"] for record in records], ["images"] * 3)
        records[0]["captions"].append("third")
        records[0]["webshart"]["sample_idx"] = 99
        self.assertEqual(records[2]["captions"], [" first ", "second"])
        self.assertEqual(self.metadata["webshart"]["sample_idx"], 19)
        self.assertNotIn("image_path", self.metadata)
        self.data_backend.list_files.assert_not_called()
        self.data_backend.read_image.assert_not_called()

    def test_describe_returns_the_geometry_the_dataset_holds_or_none(self):
        self.assertIs(self.runtime.describe("one.jpg"), self.metadata)
        self.metadata_backend.get_metadata_by_filepath.assert_called_once_with("one.jpg")
        self.metadata_backend.get_metadata_by_filepath.return_value = None
        self.assertIsNone(self.runtime.describe("missing.jpg"))

    def test_missing_metadata_preserves_sample_identity_for_sampler_validation(self):
        self.metadata_backend.get_metadata_by_filepath.return_value = None
        self.assertEqual(
            self.runtime.read_samples(["missing.jpg"]),
            [{"dataset_type": "image", "data_backend_id": "images", "image_path": "missing.jpg"}],
        )

    def test_metadata_caption_strategies_use_injected_dependencies(self):
        for strategy in ("webshart", "huggingface"):
            with self.subTest(strategy=strategy):
                self.config["caption_strategy"] = strategy
                runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, self.config)
                with (
                    patch.object(StateTracker, "get_data_backend", side_effect=AssertionError("registry lookup")),
                    patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("config lookup")),
                ):
                    self.assertEqual(runtime.resolve_caption("sample.jpg"), ["first", "second"])

    def test_missing_caption_is_reported(self):
        self.metadata_backend.caption_cache_entry.return_value = None
        with self.assertRaises(CaptionNotFoundError):
            self.runtime.resolve_caption("webshart://2/19/a.jpg")

    def test_caption_shuffle_expands_alternatives_before_prepend(self):
        config = {
            **self.config,
            "caption_shuffle": {"enable": True},
            "prepend_instance_prompt": True,
            "instance_prompt": "style",
        }
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        with patch(
            "simpletuner.helpers.prompts.CaptionShuffler.expand_with_shuffles",
            side_effect=[["first", "first variant"], ["second", "second variant"]],
        ) as expand:
            self.assertEqual(
                runtime.resolve_caption("sample.jpg"),
                ["style first", "style first variant", "style second", "style second variant"],
            )
        self.assertEqual([call.args[0] for call in expand.call_args_list], ["first", "second"])

    def test_parquet_caption_uses_injected_metadata_and_root(self):
        self.metadata_backend.parquet_database = {}
        self.metadata_backend.parquet_config = {"filename_column": "file", "caption_column": "caption"}
        config = {**self.config, "caption_strategy": "parquet", "instance_data_dir": "images"}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        with (
            patch.object(StateTracker, "get_parquet_database", side_effect=AssertionError("database lookup")),
            patch.object(StateTracker, "get_data_backend", side_effect=AssertionError("registry lookup")),
            patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("config lookup")),
        ):
            self.assertEqual(runtime.resolve_caption("images/nested/sample.jpg"), ["first", "second"])
        self.metadata_backend.caption_cache_entry.assert_called_once_with("nested/sample")

    def test_config_is_a_snapshot_and_empty_shuffle_config_needs_no_registry(self):
        self.config["caption_shuffle"] = {"enable": False}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, self.config)
        self.config["caption_shuffle"]["enable"] = True
        self.config["instance_prompt"] = "changed"
        with patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("config lookup")):
            self.assertEqual(runtime.resolve_caption("sample.jpg"), ["first", "second"])

    def test_filename_textfile_csv_and_instance_prompt_semantics(self):
        self.data_backend.read.return_value = b" first \n second "
        self.data_backend.get_caption.return_value = ["csv one", "csv two"]
        cases = (
            ("filename", {}, "a photo"),
            ("textfile", {}, ["first", "second"]),
            ("textfile", {"disable_multiline_split": True}, "first \n second"),
            ("csv", {}, ["csv one", "csv two"]),
            ("instanceprompt", {"instance_prompt": " style "}, "style"),
            ("webshart", {"use_captions": False, "instance_prompt": "style"}, "style"),
            ("webshart", {"prepend_instance_prompt": True, "instance_prompt": "style"}, ["style first", "style second"]),
        )
        for strategy, overrides, expected in cases:
            with self.subTest(strategy=strategy, overrides=overrides):
                config = {**self.config, "caption_strategy": strategy, **overrides}
                runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
                self.assertEqual(runtime.resolve_caption("a_photo.jpg"), expected)

    def test_sampler_keeps_occurrence_captions_independent(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "images"
        sampler.dataset_type = "image"
        sampler.metadata_backend = self.metadata_backend
        sampler.dataset_runtime = self.runtime
        sampler.debug_log = Mock()
        with (
            patch.object(StateTracker, "get_args", return_value=SimpleNamespace(model_family="sdxl")),
            patch("simpletuner.helpers.multiaspect.sampler.random.choice", side_effect=["first", "second"]),
        ):
            records = sampler._validate_and_yield_images_from_samples(["sample.jpg", "sample.jpg"], "1.0")
        self.assertEqual([record["instance_prompt_text"] for record in records], ["first", "second"])
        self.assertNotIn("instance_prompt_text", self.metadata)
        records[1]["webshart"]["sample_idx"] = 20
        self.assertEqual(records[0]["webshart"]["sample_idx"], 19)

    def test_sampler_accepts_runtime_without_reading_global_config(self):
        self.metadata_backend.id = "images"
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["sample.jpg"]}
        with (
            patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("config lookup")),
            patch("simpletuner.helpers.multiaspect.sampler.get_logger", return_value=Mock()),
        ):
            sampler = MultiAspectSampler(self.runtime, model=None, accelerator=Mock(), batch_size=1)
        self.assertIs(sampler.dataset_runtime, self.runtime)
        self.assertEqual(sampler.caption_strategy, "webshart")

    def test_validation_isolates_metadata_and_resolves_source_without_source_sampler(self):
        for source_id in (None, "source"):
            with self.subTest(source_id=source_id):
                sampler = object.__new__(MultiAspectSampler)
                sampler.id = "images"
                sampler.model = None
                sampler.dataset_runtime = self.runtime
                sampler.data_backend = self.data_backend
                sampler.source_dataset_id = source_id
                sampler.metadata_backend = SimpleNamespace(aspect_ratio_bucket_indices={"1.0": ["sample.jpg"]})
                sampler._val_cursor = 0
                sampler.logger = Mock()
                sampler.debug_log = Mock()
                source_runtime = SimpleNamespace(resolve_caption=Mock(return_value="source caption"))

                def training_sample(**kwargs):
                    kwargs["image_metadata"]["webshart"]["sample_idx"] = 99
                    sample = Mock(image="prepared image")
                    sample.training_sample_path.return_value = "source.jpg"
                    return sample

                with (
                    patch("simpletuner.helpers.multiaspect.sampler.TrainingSample", side_effect=training_sample),
                    patch.object(StateTracker, "get_data_backend", return_value={"dataset_runtime": source_runtime}),
                    patch("simpletuner.helpers.multiaspect.sampler.random.choice", side_effect=lambda choices: choices[0]),
                ):
                    results = sampler.retrieve_validation_set(1)
                self.assertEqual(results[0][1], "source caption" if source_id else "first")
                self.assertEqual(self.metadata["webshart"]["sample_idx"], 19)
                if source_id:
                    source_runtime.resolve_caption.assert_called_once_with("source.jpg")
                else:
                    source_runtime.resolve_caption.assert_not_called()

    def test_conditioning_isolates_metadata_and_uses_source_runtime(self):
        config = {**self.config, "caption_strategy": None}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "images"
        sampler.model = None
        sampler.dataset_runtime = runtime
        sampler.data_backend = self.data_backend
        sampler.metadata_backend = self.metadata_backend
        sampler.metadata_backend.instance_data_dir = "/conditioning"
        sampler.conditioning_type = "mask"
        sampler.caption_strategy = None
        sampler.source_dataset_id = "source"
        sampler.logger = Mock()
        sampler.debug_log = Mock()
        source_runtime = SimpleNamespace(resolve_caption=Mock(return_value="source caption"))

        def training_sample(**kwargs):
            kwargs["image_metadata"]["webshart"]["sample_idx"] = 99
            sample = Mock(image=None)
            sample.training_sample_path.return_value = "source.jpg"
            return sample

        with (
            patch("simpletuner.helpers.multiaspect.sampler.TrainingSample", side_effect=training_sample),
            patch.object(StateTracker, "get_data_backend", return_value={"dataset_runtime": source_runtime}),
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
        ):
            sample = sampler.get_conditioning_sample("sample.jpg")
        sample.set_caption.assert_called_once_with("source caption")
        source_runtime.resolve_caption.assert_called_once_with("source.jpg")
        self.assertEqual(self.metadata["webshart"]["sample_idx"], 19)

    def test_sampler_still_rejects_missing_crop_metadata(self):
        self.metadata_backend.get_metadata_by_filepath.return_value = {"original_size": (512, 512)}
        sampler = object.__new__(MultiAspectSampler)
        sampler.dataset_type = "image"
        sampler.dataset_runtime = self.runtime
        sampler.metadata_backend = self.metadata_backend
        with patch.object(StateTracker, "get_args", return_value=SimpleNamespace(model_family="sdxl")):
            with self.assertRaisesRegex(Exception, "did not have its metadata"):
                sampler._validate_and_yield_images_from_samples(["sample.jpg"], "1.0")
        self.metadata_backend.caption_cache_entry.assert_not_called()


class DatasetRuntimeCursorTests(unittest.TestCase):
    """The runtime owns the dataset's traversal: one rank's share of the bucket lists."""

    def setUp(self):
        self.metadata_backend = SimpleNamespace(
            aspect_ratio_bucket_indices={"1.0": [f"image{index}.jpg" for index in range(8)]}
        )
        self.runtime = DatasetRuntime("images", Mock(id="images"), self.metadata_backend, {})
        self.args = SimpleNamespace(seed=11, gradient_accumulation_steps=1, allow_dataset_oversubscription=False)
        self.config = {}
        for target, value in (("get_args", self.args), ("get_data_backend_config", self.config)):
            patcher = patch.object(StateTracker, target, return_value=value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _configure(self, runtime=None, *, dp_size=1, dp_rank=0, is_main_process=True, **options):
        accelerator = SimpleNamespace(num_processes=dp_size, process_index=dp_rank, is_main_process=is_main_process)
        options = {"batch_size": 2, "shared_by_all_ranks": False, **options}
        (runtime or self.runtime).configure_cursor(accelerator=accelerator, **options)

    def test_a_dataset_without_a_configured_traversal_has_no_cursor(self):
        with self.assertRaisesRegex(RuntimeError, "configure_cursor"):
            self.runtime.cursor()

    def test_ranks_take_disjoint_shares_of_the_bucket_lists(self):
        shares = []
        for rank in range(4):
            runtime = DatasetRuntime("images", Mock(id="images"), self.metadata_backend, {})
            self._configure(runtime, dp_size=4, dp_rank=rank, batch_size=1)
            shares.append(runtime.cursor().rank_share()["1.0"])
        self.assertEqual([len(share) for share in shares], [2] * 4)
        self.assertEqual(
            sorted(ref for share in shares for ref in share), self.metadata_backend.aspect_ratio_bucket_indices["1.0"]
        )

    def test_context_parallel_ranks_of_one_group_take_the_same_share(self):
        with patch(
            "simpletuner.helpers.data_backend.dataset_runtime.get_cp_aware_dp_info", return_value=(2, 1, 2)
        ) as dp_info:
            self._configure(dp_size=4, dp_rank=3, batch_size=1)
        dp_info.assert_called_once()
        state = self.runtime.cursor().state()
        self.assertEqual((state["layout"]["dp_size"], state["layout"]["dp_rank"]), (2, 1))
        self.assertEqual(len(self.runtime.cursor().rank_share()["1.0"]), 4)

    def test_the_cursor_is_built_from_the_bucket_lists_at_first_use(self):
        self._configure()
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.5": ["wide0.jpg", "wide1.jpg"]}
        self.assertEqual(sorted(self.runtime.cursor().rank_share()["1.5"]), ["wide0.jpg", "wide1.jpg"])
        self.assertIs(self.runtime.cursor(), self.runtime.cursor())

    def test_dataset_options_reach_the_cursor(self):
        self.config["repeats"] = 2
        self.args.gradient_accumulation_steps = 2
        self._configure()
        self.assertEqual(len(self.runtime.cursor()), 12)
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["only.jpg"]}
        self._configure(batch_size=4)
        with self.assertRaisesRegex(ValueError, "Effective batch size: 8"):
            self.runtime.cursor()
        self.args.allow_dataset_oversubscription = True
        self.config["repeats"] = 0
        self._configure(batch_size=4)
        self.assertEqual(len(self.runtime.cursor()), 2)
        self._configure(batch_size=4, shared_by_all_ranks=True)
        self.assertEqual(len(self.runtime.cursor()), 1)

    def test_a_seeded_run_uses_its_seed(self):
        with patch("simpletuner.helpers.data_backend.dataset_runtime.broadcast_object_from_main") as broadcast:
            self._configure()
        broadcast.assert_not_called()
        self.assertEqual(self.runtime.cursor().seed, 11)

    def test_an_unseeded_run_takes_the_seed_the_main_process_drew(self):
        self.args.seed = None
        for is_main_process in (True, False):
            with (
                self.subTest(is_main_process=is_main_process),
                patch(
                    "simpletuner.helpers.data_backend.dataset_runtime.broadcast_object_from_main", return_value=1234
                ) as broadcast,
            ):
                self._configure(is_main_process=is_main_process)
                (drawn,), _kwargs = broadcast.call_args
                self.assertEqual(drawn is None, not is_main_process)
                self.assertEqual(self.runtime.cursor().seed, 1234)

    def test_a_rank_the_seed_did_not_reach_refuses_to_divide_the_dataset(self):
        self.args.seed = None
        with patch(
            "simpletuner.helpers.data_backend.dataset_runtime.broadcast_object_from_main", side_effect=lambda value: value
        ):
            self._configure(is_main_process=True)
            with self.assertRaisesRegex(RuntimeError, "did not receive the shuffle seed"):
                self._configure(is_main_process=False)

    def test_rebuilding_reads_the_new_bucket_lists_and_keeps_the_epoch_and_seed(self):
        self.args.seed = None
        self._configure()
        cursor = self.runtime.cursor()
        cursor.finish_epoch()
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.5": ["wide0.jpg", "wide1.jpg"]}
        self.runtime.rebuild_cursor()
        rebuilt = self.runtime.cursor()
        self.assertIsNot(rebuilt, cursor)
        self.assertEqual((rebuilt.epoch, rebuilt.seed), (2, cursor.seed))
        self.assertEqual(
            {bucket: sorted(refs) for bucket, refs in rebuilt.rank_share().items()}, {"1.5": ["wide0.jpg", "wide1.jpg"]}
        )

    def test_rebuilding_keeps_the_seed_a_resumed_run_took_from_its_checkpoint(self):
        self._configure()
        saved = self.runtime.cursor().state()
        self.args.seed = 99
        self._configure()
        self.runtime.cursor().load_state(saved, allow_batch_size_change=False)
        self.runtime.rebuild_cursor()
        self.assertEqual(self.runtime.cursor().seed, 11)

    def test_rebuilding_before_first_use_builds_nothing(self):
        self._configure()
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["only.jpg"]}
        self.runtime.rebuild_cursor()
        self.assertIsNone(self.runtime._cursor)

    def test_state_tracker_locates_the_runtime_by_dataset_id(self):
        with patch.object(StateTracker, "data_backends", {"images": {"dataset_runtime": self.runtime}}):
            self.assertIs(StateTracker.get_dataset("images"), self.runtime)
