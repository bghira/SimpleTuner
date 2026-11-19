import unittest
from contextlib import closing, nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.prompts import CaptionNotFoundError
from simpletuner.helpers.training.state_tracker import StateTracker


class DatasetRuntimeTests(unittest.TestCase):
    def test_standalone_local_preparation_uses_its_bound_geometry_config(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import _ScanArgsNamespace

        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            Image.new("RGB", (96, 64)).save(image_dir / "sample.png")
            args = _ScanArgsNamespace(
                {"output_dir": str(root), "resolution": 64, "aspect_bucket_alignment": 8, "model_family": "sd1x"}
            )
            accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, is_local_main_process=True)
            config = {
                "id": "standalone",
                "type": "local",
                "instance_data_dir": str(image_dir),
                "preserve_data_backend_cache": True,
                "crop": True,
                "crop_style": "center",
                "crop_aspect": "square",
                "resolution": 64,
                "resolution_type": "pixel",
            }
            with (
                patch.object(StateTracker, "get_args", return_value=args),
                patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("global dataset config")),
                closing(DatasetRuntime.from_config(config, args, accelerator)) as dataset,
            ):
                dataset.prepare_view()
                refs, total = dataset.read_page(limit=1)
                self.assertEqual(total, 1)
                (record,) = dataset.read_samples(refs)
                self.assertEqual(tuple(record["target_size"]), (64, 64))

    def setUp(self):
        self.metadata = {
            "original_size": (512, 512),
            "target_size": (512, 512),
            "crop_coordinates": (0, 0),
            "captions": [" first ", "second"],
            "webshart": {"shard_idx": 2, "sample_idx": 19},
        }
        self.metadata_backend = Mock(epoch_provider=None)
        self.metadata_backend.get_metadata_batch.side_effect = lambda refs: MetadataBackend.get_metadata_batch(
            self.metadata_backend, refs
        )
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

    def test_registry_exposes_catalogue_without_sampler_or_legacy_bundle(self):
        with (
            patch.object(StateTracker, "datasets", {}),
            patch.object(StateTracker, "_pending_dataset_configs", {}),
        ):
            StateTracker.register_dataset(self.runtime)
            self.assertIs(StateTracker.get_dataset("images"), self.runtime)
            self.assertIs(StateTracker.get_metadata_by_filepath("ref", "images"), self.metadata)
            updated = {**self.runtime.config, "probability": 0.25}
            StateTracker.set_data_backend_config("images", updated)
            self.assertIs(StateTracker.get_data_backend_config("images"), updated)
            self.assertNotIn("images", StateTracker._pending_dataset_configs)

    def test_registration_retires_the_unbound_configuration(self):
        with patch.object(StateTracker, "datasets", {}), patch.object(StateTracker, "_pending_dataset_configs", {}):
            StateTracker.set_data_backend_config(self.runtime.id, {"probability": 0.5})
            StateTracker.register_dataset(self.runtime)
            self.assertNotIn(self.runtime.id, StateTracker._pending_dataset_configs)
            self.assertIs(StateTracker.get_data_backend_config(self.runtime.id), self.runtime.config)

    def test_factory_failure_closes_opened_datasets_and_clears_pending_state(self):
        from contextlib import ExitStack

        from simpletuner.helpers.data_backend.factory import FactoryRegistry, configure_multi_databackend

        def fail_startup(_):
            StateTracker.own_dataset_resource(self.runtime)
            StateTracker.register_dataset(self.runtime)
            StateTracker.set_data_backend_config("second", {})
            raise ValueError("preparation failed")

        with (
            patch.object(StateTracker, "_dataset_resources", ExitStack()),
            patch.object(StateTracker, "datasets", {}),
            patch.object(StateTracker, "_pending_dataset_configs", {}),
            patch.object(StateTracker, "dataset_schedule", {}),
            patch.object(StateTracker, "default_text_embed_cache", None),
            patch.object(StateTracker, "consumed_dataset_mixer", None),
            patch.object(FactoryRegistry, "__init__", return_value=None),
            patch.object(FactoryRegistry, "configure", side_effect=fail_startup),
        ):
            with self.assertRaisesRegex(ValueError, "preparation failed"):
                configure_multi_databackend({}, SimpleNamespace(is_main_process=True), None, None, None)
            self.assertEqual(StateTracker.get_datasets(), {})
            self.assertEqual(StateTracker._pending_dataset_configs, {})
        self.metadata_backend.close.assert_called_once_with()

    def test_run_closes_shared_writer_once_after_dataset_handles(self):
        from contextlib import ExitStack

        writer = Mock()
        second = DatasetRuntime("other", self.data_backend, Mock(), {})
        order = []
        writer.close.side_effect = lambda: order.append("writer")
        with (
            patch.object(StateTracker, "_dataset_resources", ExitStack()),
            patch.object(StateTracker, "datasets", {}),
            patch.object(StateTracker, "_pending_dataset_configs", {}),
            patch.object(StateTracker, "dataset_schedule", {}),
            patch.object(StateTracker, "default_text_embed_cache", None),
            patch.object(StateTracker, "consumed_dataset_mixer", None),
        ):
            StateTracker.own_dataset_resource(writer)
            for dataset in (self.runtime, second):
                dataset.metadata_backend.close.side_effect = lambda dataset=dataset: order.append(dataset.id)
                dataset.text_embed_cache = writer
                StateTracker.register_dataset(dataset)
                StateTracker.own_dataset_resource(dataset)
            StateTracker.clear_data_backends()
            StateTracker.clear_data_backends()
        self.assertEqual(order, ["other", "images", "writer"])

    def test_occurrence_revision_pins_binding_order_views_captions_and_dropout(self):
        from simpletuner.helpers.metadata.preparation import DatasetDescription

        args = SimpleNamespace(conditioning_multidataset_sampling="random", caption_dropout_probability=0.1)
        a = DatasetRuntime("a", None, Mock(), {"instance_prompt": "first"})
        b = DatasetRuntime("b", None, Mock(), {})
        for related in (a, b):
            related.metadata_backend.describe.return_value = DatasetDescription(view_revision="view-1")
        self.runtime.conditioning_datasets = (a, b)
        original = self.runtime.occurrence_revision(args)
        self.runtime.conditioning_datasets = (b, a)
        self.assertNotEqual(self.runtime.occurrence_revision(args), original)
        self.runtime.conditioning_datasets = (a, b)
        a.metadata_backend.describe.return_value = DatasetDescription(view_revision="view-2")
        self.assertNotEqual(self.runtime.occurrence_revision(args), original)
        a.metadata_backend.describe.return_value = DatasetDescription(view_revision="view-1")
        with patch.dict(a.config, instance_prompt="second"):
            self.assertNotEqual(self.runtime.occurrence_revision(args), original)
        args.caption_dropout_probability = 0.5
        self.assertNotEqual(self.runtime.occurrence_revision(args), original)
        args.caption_dropout_probability = 0.1
        with patch.dict(self.runtime.config, aws_secret_access_key="test-value"):
            self.assertEqual(self.runtime.occurrence_revision(args), original)

    def test_published_preparation_groups_ignore_rank_local_training_buckets(self):
        import json
        from io import BytesIO
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from simpletuner.helpers.data_backend.aws import S3DataBackend
        from simpletuner.helpers.data_backend.csv_url_list import CSVDataBackend
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        class Provider(MetadataBackend):
            pass

        accepted = {"square": ["a", "b", "c"], "wide": ["d", "e", "f"]}
        published = json.dumps({"aspect_ratio_bucket_indices": accepted})
        with TemporaryDirectory() as directory:
            root = Path(directory)
            csv_file = root / "source.csv"
            csv_file.write_text("url,caption\nhttps://example.com/sample.png,caption\n")
            local = LocalDataBackend(None, "local")
            csv = CSVDataBackend(None, "csv", csv_file)
            with patch("simpletuner.helpers.data_backend.aws.boto3.client"):
                aws = S3DataBackend("aws", "datasets", None)
            aws.client.get_object.side_effect = lambda **kwargs: {"Body": BytesIO(published.encode())}

            for backend in (local, csv, aws):
                with self.subTest(backend=backend.id):
                    provider = object.__new__(Provider)
                    provider.read_only = True
                    provider.cache_file = str(root / f"{backend.id}-buckets.json")
                    provider.data_backend = backend
                    if backend is not aws:
                        backend.write(provider.cache_file, published)
                    provider.aspect_ratio_bucket_indices = {"square": ["b", "b"], "wide": ["e"]}
                    dataset = DatasetRuntime(backend.id, backend, provider, {})
                    with patch.object(backend, "read", side_effect=AssertionError("full bucket index read")):
                        self.assertEqual(dataset.read_page(offset=2, limit=3), (["c", "d", "e"], 6))
                        for rank, expected in enumerate((["a", "c", "e"], ["b", "d", "f"])):
                            with provider.open_preparation_view(worker=rank, workers=2) as view:
                                self.assertEqual(list(view), expected)
                            batches = list(dataset.iter_preparation_batches(worker=rank, workers=2, batch_size=2))
                            self.assertEqual([ref for _, group in batches for ref in group], expected)
                            self.assertTrue(all(set(group) <= set(accepted[bucket]) for bucket, group in batches))
                    self.assertEqual(provider.aspect_ratio_bucket_indices, {"square": ["b", "b"], "wide": ["e"]})

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
        self.metadata_backend.instance_data_dir = "dataset"
        prepared_metadata = Mock(get_metadata_by_filepath=Mock(return_value=self.metadata))
        view = Mock(metadata=prepared_metadata)
        view.__iter__ = lambda _: iter(["sample.jpg"])
        self.metadata_backend.open_preparation_view.return_value = nullcontext(view)
        self.metadata_backend.get_metadata_by_filepath.side_effect = AssertionError("live metadata read")
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)

        def metadata_for_preparation(*, read_metadata, image_path, **kwargs):
            return {"height": read_metadata(image_path)["target_size"][1]}

        model = SimpleNamespace(
            text_embed_cache_key=lambda: TextEmbedCacheKey.DATASET_AND_FILENAME,
            text_embed_cache_metadata_for_preparation=metadata_for_preparation,
            text_embed_cache_key_value=lambda prompt, default_key, metadata: f"{default_key}:{metadata['height']}",
        )
        records = list(runtime.iter_preparation_text_records(model))
        self.assertEqual([record["prompt"] for record in records], ["first", "cat", "second", "cat"])
        self.assertEqual(records[0]["key"], "images:sample.jpg:512")
        self.assertEqual(records[0]["metadata"]["dataset_relative_path"], "sample.jpg")
        self.assertEqual(records[1], {"prompt": "cat", "key": "sample.jpg__bbox_0", "metadata": {"grounding_entity": True}})

    def test_random_mask_and_reference_emit_the_selected_caption_context_and_unconditional(self):
        from simpletuner.helpers.data_backend.dataset_runtime import TextRequestBuilder
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        model = SimpleNamespace(
            text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION, uses_image_context_dropout_caption_cache=lambda: True
        )
        args = SimpleNamespace(conditioning_multidataset_sampling="random", caption_dropout_probability=0)
        self.metadata_backend.instance_data_dir = ""
        source = DatasetRuntime("images", self.data_backend, self.metadata_backend, self.config)
        source.resolve_caption = lambda path: ["source A", "source B"]
        mask = SimpleNamespace(
            id="mask",
            config={"conditioning_type": "mask", "source_dataset_id": "images", "caption_strategy": None},
            conditioning_ref=lambda ref: "m/" + ref,
        )
        reference = SimpleNamespace(
            id="ref",
            config={"conditioning_type": "reference_strict"},
            conditioning_ref=lambda ref: "r/" + ref,
            resolve_caption=lambda path: "reference",
        )
        with patch(
            "simpletuner.helpers.distillation.factory.DistillerFactory.training_batch_requirements",
            return_value={"unconditional_text_embeddings"},
        ):
            builder = TextRequestBuilder(model, source, conditioning_datasets=[mask, reference], args=args)
        requests = list(
            builder.records([("source A", "file.png")], read_metadata=self.metadata_backend.get_metadata_by_filepath)
        )
        contexts = [(item["prompt"], item["metadata"].get("image_paths")) for item in requests]
        self.assertEqual(
            contexts,
            [
                ("reference", ["r/file.png"]),
                ("", ["r/file.png"]),
                ("source A", None),
                ("", None),
                ("source B", None),
                ("", None),
            ],
        )

    def test_caption_preparation_only_reads_geometry_when_grounding_is_enabled(self):
        from simpletuner.helpers.data_backend.dataset_runtime import TextRequestBuilder
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        model = SimpleNamespace(text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION)
        self.metadata_backend.instance_data_dir = ""
        read_metadata = Mock(return_value={"bbox_entities": [{"label": "cat"}]})
        caption = [("caption", "image.jpg")]
        builder = TextRequestBuilder(model, self.runtime, args=SimpleNamespace(max_grounding_entities=0))
        self.assertEqual([record["prompt"] for record in builder.records(caption, read_metadata=read_metadata)], ["caption"])
        read_metadata.assert_not_called()
        builder = TextRequestBuilder(
            model, self.runtime, conditioning_datasets=[], args=SimpleNamespace(max_grounding_entities=1)
        )
        self.assertEqual(
            [record["prompt"] for record in builder.records(caption, read_metadata=read_metadata)], ["caption", "cat"]
        )
        read_metadata.assert_called_once_with("image.jpg")

    def test_projected_payloads_preserve_order_duplicates_and_identify_failed_references(self):
        self.data_backend.read_image_batch.return_value = (["a", "a", "b"], [1, 2, 3])
        records = self.runtime.read_samples(["a", "b", "a"], fields={"image"})
        self.assertEqual([record["image"] for record in records], [1, 3, 2])
        self.metadata_backend.get_metadata_by_filepath.assert_not_called()
        with self.assertRaisesRegex(OSError, "missing"):
            self.runtime.read_samples(["a", "missing"], fields={"image"})

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
                    patch.object(StateTracker, "get_dataset", side_effect=AssertionError("registry lookup")),
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
        self.metadata_backend.parquet_config = {"filename_column": "file", "caption_column": "caption"}
        config = {**self.config, "caption_strategy": "parquet", "instance_data_dir": "images"}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        with (
            patch.object(StateTracker, "get_dataset", side_effect=AssertionError("registry lookup")),
            patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("config lookup")),
        ):
            self.assertEqual(runtime.resolve_caption("images/nested/sample.jpg"), ["first", "second"])
        self.metadata_backend.caption_cache_entry.assert_called_once_with("nested/sample")

    def test_parquet_provider_shares_one_side_table_with_caption_reads(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        import pandas as pd

        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend

        with TemporaryDirectory() as directory:
            root = Path(directory)
            table = root / "metadata.parquet"
            pd.DataFrame(
                {"file": ["a/sample.jpg", "b/sample.jpg"], "caption": ["primary", None], "tags": ["unused", "fallback"]}
            ).to_parquet(table)
            config = {**self.config, "caption_strategy": "parquet", "instance_data_dir": directory}
            accelerator = SimpleNamespace()
            storage = LocalDataBackend(accelerator, "images")
            with (
                patch.object(StateTracker, "get_data_backend_config", return_value=config),
                patch.object(storage, "read", wraps=storage.read) as reads,
            ):
                metadata = ParquetMetadataBackend(
                    id="images",
                    instance_data_dir=directory,
                    cache_file=str(root / "buckets"),
                    metadata_file=str(root / "geometry"),
                    data_backend=storage,
                    accelerator=accelerator,
                    batch_size=1,
                    resolution=512,
                    resolution_type="pixel",
                    parquet_config={
                        "path": str(table),
                        "filename_column": "file",
                        "caption_column": "caption",
                        "fallback_caption_column": "tags",
                    },
                )
                dataset = DatasetRuntime("images", storage, metadata, config)
                self.assertEqual(
                    dataset.resolve_captions([str(root / "b/sample.jpg"), str(root / "a/sample.jpg")]),
                    ["fallback", "primary"],
                )
                reads.assert_called_once_with(str(table))

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
        self.metadata_backend.caption_projection_strategy = "webshart"
        self.metadata_backend.read_caption_batch.return_value = [self.metadata["captions"], self.metadata["captions"]]
        self.metadata["conditioning"] = {"references": ["base.png"]}
        self.metadata["crop_coordinates"] = [0, 0]
        with (
            patch.object(StateTracker, "get_args", return_value=SimpleNamespace(model_family="sdxl")),
            patch("simpletuner.helpers.multiaspect.sampler.random.choice", side_effect=["first", "second"]),
        ):
            records = sampler._validate_and_yield_images_from_samples(["sample.jpg", "sample.jpg"], "1.0")
        self.assertEqual([record["instance_prompt_text"] for record in records], ["first", "second"])
        self.metadata_backend.read_caption_batch.assert_called_once_with(["sample.jpg", "sample.jpg"])
        records[0]["crop_coordinates"][0] = 8
        records[1]["conditioning"]["references"].append("alternative.png")
        self.assertEqual(records[1]["crop_coordinates"], [0, 0])
        self.assertEqual(records[0]["conditioning"]["references"], ["base.png"])
        self.assertEqual(self.metadata["conditioning"]["references"], ["base.png"])
        self.assertEqual(self.metadata["crop_coordinates"], [0, 0])
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

    def test_provider_schedule_is_independent_of_storage_config(self):
        @dataclass(frozen=True)
        class Schedule:
            epoch: int
            batch_size: int
            occurrence_revision: str | None = None

        provider = Mock()
        provider.epoch_spec.side_effect = Schedule
        provider.scheduled_batches.return_value = 9
        self.metadata_backend.epoch_provider = provider
        self.metadata_backend.id = "images"
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, {})
        sampler = MultiAspectSampler(runtime, model=None, accelerator=Mock(), batch_size=2)

        self.assertTrue(runtime.uses_native_schedule)
        self.assertEqual(len(sampler), 9)
        provider.scheduled_batches.assert_called_once_with(Schedule(1, 2))
        self.assertIs(sampler.validation_cursor, provider.open_validation.return_value)
        with patch.object(runtime, "occurrence_revision", return_value="revision"):
            self.assertIs(sampler._open_training_cursor(), provider.open_epoch.return_value)
        provider.open_epoch.assert_called_once_with(Schedule(1, 2, "revision"))
        self.metadata_backend.open_epoch.assert_not_called()

        self.metadata_backend.epoch_provider = None
        generic = DatasetRuntime(
            "images", self.data_backend, self.metadata_backend, {"webshart": {"sampling_policy": "global"}}
        )
        self.assertFalse(generic.uses_native_schedule)

    def test_validation_isolates_metadata_and_resolves_source_without_source_sampler(self):
        for source_id in (None, "source"):
            with self.subTest(source_id=source_id):
                sampler = object.__new__(MultiAspectSampler)
                sampler.id = "images"
                sampler.model = None
                sampler.dataset_runtime = self.runtime
                sampler.data_backend = self.data_backend
                sampler.validation_cursor = SimpleNamespace(next_refs=Mock(return_value=["sample.jpg"]))
                self.data_backend.read_image_batch.return_value = (["sample.jpg"], ["image"])
                sampler.logger = Mock()
                sampler.debug_log = Mock()
                source_runtime = SimpleNamespace(id="source", resolve_captions=Mock(return_value=["source caption"]))
                self.runtime.source_dataset = source_runtime if source_id else None

                def training_sample(**kwargs):
                    kwargs["image_metadata"]["webshart"]["sample_idx"] = 99
                    sample = Mock(image="prepared image")
                    sample.training_sample_path.return_value = "source.jpg"
                    return sample

                with (
                    patch("simpletuner.helpers.multiaspect.sampler.TrainingSample", side_effect=training_sample),
                    patch.object(StateTracker, "get_dataset", side_effect=AssertionError("global dataset lookup")),
                    patch("simpletuner.helpers.multiaspect.sampler.random.choice", side_effect=lambda choices: choices[0]),
                ):
                    results = sampler.retrieve_validation_set(1)
                self.assertEqual(results[0].prompt, "source caption" if source_id else "first")
                self.assertEqual(self.metadata["webshart"]["sample_idx"], 19)
                if source_id:
                    source_runtime.resolve_captions.assert_called_once_with(["source.jpg"])
                else:
                    source_runtime.resolve_captions.assert_not_called()

    def test_conditioning_isolates_metadata_and_uses_source_runtime(self):
        config = {**self.config, "caption_strategy": None, "source_dataset_id": "source", "conditioning_type": "mask"}
        runtime = DatasetRuntime("images", self.data_backend, self.metadata_backend, config)
        self.metadata_backend.instance_data_dir = "/conditioning"
        self.metadata_backend.conditioning_ref.return_value = "sample.jpg"
        source_runtime = SimpleNamespace(id="source", resolve_caption=Mock(return_value="source caption"))

        def training_sample(**kwargs):
            kwargs["image_metadata"]["webshart"]["sample_idx"] = 99
            sample = Mock(image=None)
            sample.training_sample_path.return_value = "source.jpg"
            return sample

        with (
            patch("simpletuner.helpers.data_backend.dataset_runtime.TrainingSample", side_effect=training_sample),
            patch.object(StateTracker, "get_dataset", side_effect=AssertionError("global dataset lookup")),
        ):
            sample = runtime.read_conditioning_sample("sample.jpg", source=source_runtime, source_ref="source.jpg")
        self.assertEqual(sample.paired_source, ("source", "source.jpg"))
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
