"""Tests for DatasetViewerService cache file discovery and summary generation."""

import base64
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from PIL import Image

from simpletuner.simpletuner_sdk.server.services.dataset_viewer_service import (
    DatasetFileMutationUnavailable,
    DatasetViewChanged,
    DatasetViewerService,
    _data_backend_cache,
    generate_thumbnail,
    generate_thumbnail_from_pil,
)


class TestWebshartMembershipPages(unittest.TestCase):
    def test_native_caption_and_preview_are_read_with_prepared_source_pins(self):
        from simpletuner.helpers.metadata.backends.webshart_geometry import snapshot_path, stage_geometry
        from simpletuner.helpers.metadata.backends.webshart_membership import publish_indexed_membership

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "id": "images",
                "type": "webshart",
                "caption_strategy": "webshart",
                "webshart": {"source": "s3://example", "cache_dir": directory, "format": "parquet"},
            }
            metadata = root / "geometry.json"
            ref = "webshart://0/0/data/0000.jpg"
            with stage_geometry(metadata) as (writer, staging):
                writer.pin_shard(0, "pinned-object")
                writer.add("square", ref, {"original_size": [32, 32]}, shard=0)
                writer.finish()
                writer.db.commit()
                revision = writer.digest.hexdigest()
                staging.replace(snapshot_path(metadata, revision))
            publication = root / "aspect_ratio_bucket_indices_source_0123456789abcdef_images.json"
            publish_indexed_membership(
                publication,
                {
                    "view_revision": revision,
                    "source_revision": "catalogue",
                    "source_pins": True,
                    "geometry_name": metadata.name,
                    "geometry_sha256": revision,
                    "geometry_format": "sqlite-v1",
                    "bucket_counts": {"square": 1},
                    "config": config,
                },
            )

            service = DatasetViewerService(project_root=root)
            backend = MagicMock()
            backend.dataset.fingerprint = "catalogue"
            pins = []
            backend.dataset.bind_source_revisions.side_effect = pins.append

            def caption(path):
                self.assertEqual(path, ref)
                self.assertEqual(pins[-1](0), "pinned-object")
                return ["first", "alternate"]

            def image(paths):
                self.assertEqual(paths, [ref])
                self.assertEqual(pins[-1](0), "pinned-object")
                return [ref], [Image.new("RGB", (32, 32), "red")]

            backend.get_caption.side_effect = caption
            backend.read_image_batch.side_effect = image
            with patch.object(service, "_create_data_backend", return_value=backend):
                entry = service.get_file_metadata(config, ref, view_revision=revision)
                self.assertEqual(entry.caption, "first\nalternate")
                self.assertIsNone(pins[-1])
                self.assertIsNotNone(service.get_image_preview(config, ref, metadata=entry.model_dump()))
                self.assertIsNone(service.get_image_preview(config, "webshart://0/1/data/other.jpg", view_revision=revision))
                self.assertIsNone(pins[-1])
                thumbnails = service.get_dataset_thumbnails(config, view_revision=revision)
                self.assertIsNotNone(thumbnails[0].thumbnail)
                self.assertIsNone(pins[-1])
                self.assertTrue(service._get_thumbnail_cache_path(config, ref, revision).exists())
                self.assertFalse(service._get_thumbnail_cache_path(config, ref).exists())
                self.assertEqual(backend.read_image_batch.call_count, 2)
                service.get_dataset_thumbnails(config, view_revision=revision)
                self.assertEqual(backend.read_image_batch.call_count, 2)
                backend.dataset.fingerprint = "new-catalogue"
                with self.assertRaisesRegex(DatasetViewChanged, "catalogue changed"):
                    service.get_file_metadata(config, ref, view_revision=revision)

    def test_latest_matching_catalogue_is_selected_across_publications(self):
        from simpletuner.helpers.metadata.backends.webshart_membership import write_membership

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "id": "images",
                "type": "webshart",
                "webshart": {
                    "source": "s3://example",
                    "cache_dir": directory,
                    "format": "parquet",
                    "manifest": "capture.json",
                },
            }

            def publish(fingerprint, ref, order, source_config):
                path = root / f"aspect_ratio_bucket_indices_source_{fingerprint}_images.json"
                write_membership(path, {"square": [ref]}, {"source_revision": fingerprint, "config": source_config})
                os.utime(path, ns=(order, order))
                return path

            service = DatasetViewerService(project_root=root)
            old = publish("z", "old.jpg", 1_000_000_000, config)
            selected = service.get_dataset_summary(config).view_revision
            publish("a", "new.jpg", 2_000_000_000, config)
            other = {**config, "webshart": {**config["webshart"], "manifest": "other.json"}}
            publish("later", "other.jpg", 3_000_000_000, other)

            summary = service.get_dataset_summary(config)
            self.assertEqual(summary.source_revision, "a")
            self.assertEqual(service.get_dataset_files(config, view_revision=summary.view_revision).files, ["new.jpg"])
            self.assertEqual(service.get_dataset_files(other).files, ["other.jpg"])
            with self.assertRaises(DatasetViewChanged):
                service.get_dataset_files(config, view_revision=selected)
            self.assertTrue(old.is_file())

    def test_incomplete_webshart_scan_is_isolated_from_other_dataset_summaries(self):
        import asyncio

        from simpletuner.helpers.metadata.backends.webshart_membership import write_membership
        from simpletuner.simpletuner_sdk.server.routes import dataset_viewer

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "id": "images",
                "type": "webshart",
                "webshart": {"source": "s3://example", "cache_dir": directory, "format": "parquet"},
            }
            local = {"id": "local", "type": "local", "instance_data_dir": directory}
            (root / "aspect_ratio_bucket_indices_local.json").write_text(
                json.dumps({"aspect_ratio_bucket_indices": {"square": ["local.jpg"]}})
            )
            path = root / "aspect_ratio_bucket_indices_source_123_images.json"
            write_membership(path, {"square": ["partial.jpg"]}, {"config": config, "scan_in_progress": True})

            service = DatasetViewerService(project_root=root)
            with (
                patch.object(dataset_viewer, "_load_plan", return_value=[config, local]),
                patch.object(dataset_viewer, "_get_viewer_service", return_value=service),
            ):
                native, legacy = asyncio.run(dataset_viewer.get_viewer_summaries())
            self.assertTrue(native.scan_incomplete)
            self.assertIsNone(native.accepted_samples)
            self.assertEqual(legacy.accepted_samples, 1)
            self.assertEqual(service.get_dataset_files(config).files, [])
            with self.assertRaises(DatasetViewChanged):
                service.get_dataset_files(config, view_revision="previous")

            write_membership(path, {"square": ["ready.jpg"]}, {"config": config})
            ready = service.get_dataset_summary(config)
            self.assertTrue(ready.has_cache)
            self.assertFalse(ready.scan_incomplete)
            self.assertEqual(service.get_dataset_files(config, view_revision=ready.view_revision).files, ["ready.jpg"])

    def test_native_caption_health_uses_admission_provenance_without_reading_rows(self):
        from simpletuner.helpers.metadata.backends.webshart_membership import write_membership

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "id": "images",
                "type": "webshart",
                "caption_strategy": "webshart",
                "only_instance_prompt": True,
                "instance_prompt": "shared prompt",
                "webshart": {"source": "s3://example", "cache_dir": directory, "format": "parquet"},
            }
            path = root / "aspect_ratio_bucket_indices_source_123_images.json"
            properties = {"config": config}
            service = DatasetViewerService(project_root=root)

            with patch.object(service, "get_dataset_files", side_effect=AssertionError("caption inventory")):
                write_membership(path, {"square": ["caption-missing.jpg"]}, properties)
                old = service.get_caption_status(config)
                self.assertIsNone(old.captioned_files)
                self.assertIsNone(old.coverage_ratio)

                properties["source_captions_required"] = False
                write_membership(path, {"square": ["caption-missing.jpg"]}, properties)
                status = service.get_caption_status(config)
                self.assertEqual(status.total_files, 1)
                self.assertIsNone(status.captioned_files)
                self.assertIsNone(status.coverage_ratio)

                properties["config"] = {**config, "only_instance_prompt": False}
                properties["source_captions_required"] = True
                write_membership(path, {"square": ["caption-present.jpg"]}, properties)
                status = service.get_caption_status(properties["config"])
                self.assertEqual(status.captioned_files, 1)
                self.assertEqual(status.coverage_ratio, 1.0)

    def test_published_pages_are_bounded_and_survive_replacement(self):
        from simpletuner.helpers.metadata.backends.webshart_membership import MembershipPages, write_membership

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {
                "id": "images",
                "type": "webshart",
                "webshart": {
                    "source": "s3://example",
                    "cache_dir": directory,
                    "format": "parquet",
                    "manifest": "capture.json",
                },
            }
            path = root / "aspect_ratio_bucket_indices_source_0123456789abcdef_images.json"
            refs = [f'path/{i} "snow 雪"\n' for i in range(900)]
            buckets = {"empty": [], "square": refs, "wide": ["repeat", "repeat", "last"]}
            write_membership(path, buckets, {"config": config})
            self.assertEqual(json.loads(path.read_text())["aspect_ratio_bucket_indices"], buckets)
            service = DatasetViewerService(project_root=root)
            with (
                patch.object(service, "_load_cache", side_effect=AssertionError("full JSON read")),
                path.open("rb") as source,
            ):
                old = MembershipPages.open(source)
                for key, offset, limit, expected, total in [
                    ("square", 255, 3, refs[255:258], 900),
                    (None, 899, 8, [refs[-1], *buckets["wide"]], 903),
                    ("wide", 1, 1, ["repeat"], 3),
                    ("missing", 1, 8, [], 0),
                    (None, 0, 0, [], 903),
                    (None, 2000, 1, [], 903),
                ]:
                    page = service.get_dataset_files(config, bucket_key=key, offset=offset, limit=limit)
                    self.assertEqual((page.files, page.total), (expected, total))
                self.assertEqual(service.get_dataset_summary(config).total_files, 903)
                with patch.object(service, "get_dataset_files", side_effect=AssertionError("caption inventory")):
                    coverage = service.get_caption_status({**config, "caption_strategy": "filename"})
                    self.assertEqual((coverage.captioned_files, coverage.coverage_ratio), (903, 1.0))
                    coverage = service.get_caption_status({**config, "caption_strategy": "csv"})
                    self.assertIsNone(coverage.coverage_ratio)
                with patch.object(old.source, "readline", wraps=old.source.readline) as lines:
                    self.assertEqual(old.read_page(None, 868, 3), (refs[868:871], 903))
                    self.assertEqual(lines.call_count, 1)
                self.assertEqual(list(old.partition(6, 7)), refs[774:] + buckets["wide"])
                write_membership(path, {"square": ["new"]}, {"config": config})
                self.assertEqual(old.read_page(None, 900, 3)[0], buckets["wide"])
                with self.assertRaises(DatasetViewChanged):
                    service.get_dataset_files(config, view_revision=old.header["view_revision"])
                new_page = service.get_dataset_files(config)
                self.assertEqual(new_page.files, ["new"])
                self.assertEqual(service.get_dataset_files(config, view_revision=new_page.view_revision), new_page)
                other = {**config, "webshart": {**config["webshart"], "manifest": "another.json"}}
                self.assertFalse(service.get_dataset_summary(other).has_cache)

    def test_conditioning_queries_page_native_views_and_apply_exclusions(self):
        import sqlite3
        from contextlib import closing

        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.metadata.backends.webshart_geometry import snapshot_path, stage_geometry
        from simpletuner.helpers.metadata.backends.webshart_membership import MembershipPages, publish_indexed_membership

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DatasetViewerService(project_root=root)
            configs = []
            refs = [f"webshart://{i}/0/image.jpg" for i in range(5)]
            for name, accepted in [("source", refs[:4]), ("conditioning", refs[1:])]:
                config = {"id": name, "type": "webshart", "webshart": {"cache_dir": directory, "format": "parquet"}}
                configs.append(config)
                metadata = root / f"geometry-{name}.json"
                publication = root / f"aspect_ratio_bucket_indices_source_test_{name}.json"
                with stage_geometry(metadata, shards_per_partition=2) as (writer, staging):
                    pruned = refs[0] if name == "conditioning" else refs[-1]
                    for partition, start in enumerate(range(0, len(refs), 2)):
                        with writer.stage_partition(partition) as (group, part):
                            for shard, ref in enumerate(refs[start : start + 2], start):
                                if ref in accepted or ref == pruned:
                                    group.add(
                                        "pruned" if ref == pruned else "wide", ref, {"original_size": [8, 16]}, shard=shard
                                    )
                        writer.adopt_partition(partition, part)
                    writer.finish()
                    digest = writer.digest.hexdigest()
                    writer.db.commit()
                    writer.publish(metadata)
                    staging.replace(snapshot_path(metadata, digest))
                publish_indexed_membership(
                    publication,
                    {
                        "view_revision": digest,
                        "bucket_counts": {"wide": len(accepted)},
                        "config": config,
                        "geometry_format": "sqlite-v1",
                        "geometry_partition_shards": 2,
                        "geometry_name": metadata.name,
                        "geometry_sha256": digest,
                    },
                )
            with publication.open("rb") as source, closing(sqlite3.connect("")) as db:
                pages = MembershipPages.open(source)
                ledger = Exclusions(db)
                ledger.add(refs[2], "wide", "nsfw")
                _publish(publication, pages.header["view_revision"], db)
            self.assertEqual(service.get_dataset_summary(configs[1]).total_files, 3)
            self.assertEqual(service.get_file_metadata(configs[1], refs[4]).original_size, [8, 16])
            filtered_page = service.get_dataset_files(configs[1], bucket_key="wide", offset=1, limit=1)
            self.assertEqual((filtered_page.files, filtered_page.total), ([refs[3]], 3))
            with self.assertRaises(DatasetViewChanged):
                service.get_file_metadata(configs[1], refs[4], view_revision=pages.header["view_revision"])
            metadata = service.get_file_metadata(configs[1], refs[4], view_revision=filtered_page.view_revision)
            self.assertEqual(metadata.view_revision, filtered_page.view_revision)
            self.assertFalse(metadata.metadata_editable)
            for action, args in (
                (service.update_crop_coordinates, (configs[1], refs[4], [0, 0])),
                (service.update_bbox_entities, (configs[1], refs[4], [])),
                (service.rebuild_file_metadata, (configs[1], {}, refs[4])),
                (service.delete_vae_cache_file, (configs[1], refs[4])),
            ):
                with self.subTest(action=action.__name__):
                    with self.assertRaises(DatasetViewChanged):
                        action(*args, view_revision=pages.header["view_revision"])
                    with self.assertRaisesRegex(DatasetFileMutationUnavailable, "Rescan"):
                        action(*args, view_revision=filtered_page.view_revision)
            with patch.object(service, "get_dataset_files", side_effect=AssertionError("unbounded listing")):
                report = service.get_conditioning_pairs(*configs, offset=1, limit=2)
                self.assertEqual(report["pairs"], [{"source_path": refs[1], "conditioning_path": refs[1]}])
                self.assertEqual((report["source_total"], report["total_pairs"]), (4, None))
                self.assertEqual(report["conditioning_view_revision"], filtered_page.view_revision)
                self.assertIsNotNone(report["source_view_revision"])
                self.assertEqual(
                    service.get_conditioning_pairs(
                        *configs,
                        offset=1,
                        limit=2,
                        source_view_revision=report["source_view_revision"],
                        conditioning_view_revision=report["conditioning_view_revision"],
                    ),
                    report,
                )
                for key in ("source_view_revision", "conditioning_view_revision"):
                    with self.subTest(key=key), self.assertRaises(DatasetViewChanged):
                        service.get_conditioning_pairs(*configs, **{key: "replaced-view"})
                orphans = service.get_orphaned_files(*configs)
                self.assertEqual((orphans.source_orphans, orphans.conditioning_orphans), ([refs[0], refs[2]], [refs[4]]))
                self.assertEqual(service.get_conditioning_pairs(configs[0], {"type": "local"})["scope"], "unavailable")
                from PIL import Image

                image = root / "image.png"
                Image.new("RGB", (8, 8)).save(image)
                local = {
                    "id": "frames",
                    "type": "local",
                    "instance_data_dir": directory,
                    "conditioning_config": {"type": "i2v_first_frame"},
                }
                linked = {**configs[0], "conditioning_data": ["frames"]}
                match = service.get_conditioning_file_match(linked, [local], refs[1])
                self.assertEqual(match["file_path"], str(image.resolve()))
                self.assertIsNone(service.get_conditioning_file_match(linked, [local], refs[-1]))


class TestPreviewSourceScope(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required for media preview")
    def test_remote_audio_preview_uses_provider_stream_and_renders_short_clips(self):
        service = DatasetViewerService()
        config = {"id": "audio", "type": "aws", "aws_data_prefix": "clips/"}
        audio = io.BytesIO()
        with wave.open(audio, "wb") as stream:
            stream.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            stream.writeframes(b"\x00\x00" * 80)
        audio.seek(0)
        backend = MagicMock()
        backend.open_read_stream.return_value = audio
        with patch.object(service, "_get_or_create_data_backend", return_value=backend):
            self.assertIsNone(service.get_audio_preview(config, "other/clip.wav").waveform)
            result = service.get_audio_preview(config, "clips/clip.wav")
        backend.open_read_stream.assert_called_once_with("clips/clip.wav")
        self.assertTrue(audio.closed)
        self.assertEqual((result.sample_rate, result.channels, result.duration), (8000, 1, 0.01))
        self.assertTrue(result.waveform.startswith("data:image/png;base64,"))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    def test_remote_video_preview_extracts_frames_from_provider_bytes(self):
        service = DatasetViewerService()
        config = {"id": "video", "type": "aws", "aws_data_prefix": ["videos/"]}
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "clip.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=size=32x32:rate=4:duration=1",
                    "-c:v",
                    "mpeg4",
                    str(video),
                ],
                check=True,
                capture_output=True,
            )
            with patch.object(service, "_get_or_create_data_backend") as open_backend:
                open_backend.return_value.open_read_stream.return_value = io.BytesIO(video.read_bytes())
                result = service.get_video_preview(config, "videos/clip.mp4", frame_count=2)
        self.assertEqual((result.resolution, result.fps, result.duration), ([32, 32], 4.0, 1.0))
        self.assertEqual(len(result.frames), 2)

    def test_local_preview_resolves_dataset_entries_and_explicit_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            outside = root / "outside.jpg"
            Image.new("RGB", (16, 16)).save(outside)
            entry = images / "linked.jpg"
            entry.symlink_to(outside)
            service = DatasetViewerService(project_root=root)
            config = {"id": "images", "instance_data_dir": str(images)}

            self.assertIsNotNone(service.get_image_preview(config, "linked.jpg"))
            self.assertIsNotNone(service.get_image_preview(config, str(entry)))
            self.assertIsNone(service.get_image_preview(config, str(outside)))
            self.assertIsNone(service.get_image_preview(config, "../outside.jpg"))

    def test_s3_preview_stays_within_the_configured_prefixes(self):
        service = DatasetViewerService()
        config = {"type": "aws", "aws_data_prefix": ["images/", "extra/"]}
        backend = MagicMock()
        backend.read_image.return_value = Image.new("RGB", (16, 16))
        with patch.object(service, "_get_or_create_data_backend", return_value=backend):
            self.assertIsNone(service.get_image_preview(config, "outside/file.jpg"))
            self.assertIsNotNone(service.get_image_preview(config, "extra/file.jpg"))
        backend.read_image.assert_called_once_with("extra/file.jpg")


class TestLegacyMembershipPages(unittest.TestCase):
    def test_viewer_reads_legacy_page_without_loading_full_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            service = DatasetViewerService(project_root=Path(directory))
            publication = Path(directory) / "buckets.json"
            publication.write_text(
                json.dumps(
                    {
                        "aspect_ratio_bucket_indices": {"wide": ["a", "b", "c"], "square": ["d"]},
                        "config": {"resolution": 0.5},
                        "filtering_statistics": {"skipped": {"too_small": 2}},
                    }
                )
            )
            with (
                patch.object(service, "_find_cache_file", return_value=publication),
                patch.object(service, "_load_cache", side_effect=AssertionError("full cache read")),
            ):
                page = service.get_dataset_files({"id": "images"}, offset=2, limit=1)
                square = service.get_dataset_files({"id": "images"}, bucket_key="square")
                summary = service.get_dataset_summary({"id": "images"})
            self.assertEqual((page.files, page.total), (["c"], 4))
            self.assertEqual((square.files, square.total), (["d"], 1))
            self.assertEqual(summary.total_files, 4)
            self.assertEqual(summary.config, {"resolution": 0.5})
            self.assertEqual(summary.filtering_statistics, {"skipped": {"too_small": 2}})


class TestLatentFileInvalidation(unittest.TestCase):
    def test_viewer_uses_training_keys_and_the_selected_artifact_store(self):
        from simpletuner.helpers.caching.vae import VAECache
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.metadata.preparation import DatasetViewChangedError

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            service = DatasetViewerService(project_root=root)
            source = root / "images"
            latents = root / "latents"
            sample = str(source / "nested" / "image.jpg")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({sample: {"original_size": [32, 32]}}))
            dataset = {
                "id": "images",
                "type": "local",
                "instance_data_dir": str(source),
                "cache_dir_vae": str(latents),
                "image_embeds": "separate",
            }
            store = {"id": "separate", "dataset_type": "image_embeds", "type": "local"}
            cache = VAECache.__new__(VAECache)
            cache.image_data_backend = LocalDataBackend(None, "images")
            cache.cache_data_backend = LocalDataBackend(None, "separate")
            cache.cache_dir = str(latents)
            cache.instance_data_dir = str(source)
            cache.hash_filenames = True
            artifact = Path(cache.cache_path_for_image(sample))
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"fixture")

            with patch.object(service, "_find_cache_file", return_value=metadata):
                current = service.get_file_metadata(dataset, sample)
                with self.assertRaises(DatasetViewChangedError):
                    service.delete_vae_cache_file(dataset, sample, dataset_plan=[store], metadata_revision="old")
                remote = {**store, "type": "aws"}
                self.assertFalse(service.delete_vae_cache_file(dataset, sample, dataset_plan=[remote])["deleted"])
                self.assertFalse(service.delete_vae_cache_file(dataset, "missing.jpg", dataset_plan=[store])["deleted"])
                self.assertTrue(artifact.exists())
                result = service.delete_vae_cache_file(
                    dataset, sample, dataset_plan=[store], metadata_revision=current.metadata_revision
                )
            self.assertEqual(result, {"deleted": True, "cache_path": str(artifact.resolve())})
            self.assertFalse(artifact.exists())


class TestFindCacheFile(unittest.TestCase):
    """Verify _find_cache_file locates cache files across all backend types."""

    def setUp(self):
        self.service = DatasetViewerService()
        self.tmpdir = tempfile.mkdtemp()

    def _write_cache(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["a.jpg", "b.jpg"]},
                    "filtering_statistics": {"total_processed": 2, "skipped": {}},
                }
            )
        )

    def test_finds_cache_in_instance_data_dir(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertEqual(result, cache_file)

    def test_finds_cache_in_cache_dir(self):
        cache_dir = Path(self.tmpdir) / "cache_vae"
        cache_file = cache_dir / "aspect_ratio_bucket_indices_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "cache_dir": str(cache_dir)}
        result = self.service._find_cache_file(config, "indices")
        self.assertEqual(result, cache_file)

    def test_finds_cache_in_huggingface_metadata_dir(self):
        """HuggingFace backends store cache at cache/huggingface/{id}/huggingface_metadata/{id}/."""
        hf_dir = Path(self.tmpdir) / "cache" / "huggingface" / "hf-ds" / "huggingface_metadata" / "hf-ds"
        cache_file = hf_dir / "aspect_ratio_bucket_indices_hf-ds.json"
        self._write_cache(cache_file)

        config = {"id": "hf-ds"}
        # The service uses relative Path("cache"), so chdir to tmpdir
        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            result = self.service._find_cache_file(config, "indices")
            self.assertIsNotNone(result)
            self.assertEqual(result.resolve(), cache_file.resolve())
        finally:
            os.chdir(original_cwd)

    def test_finds_metadata_file_type(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_metadata_test-ds.json"
        self._write_cache(cache_file)

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "metadata")
        self.assertEqual(result, cache_file)

    def test_returns_none_when_no_cache_exists(self):
        config = {"id": "nonexistent-ds", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertIsNone(result)

    def test_returns_none_for_empty_id(self):
        config = {"id": "", "instance_data_dir": self.tmpdir}
        result = self.service._find_cache_file(config, "indices")
        self.assertIsNone(result)

    def test_fallback_rglob_under_cache_dir(self):
        """Deeply nested cache files should be found via rglob fallback."""
        nested_dir = Path(self.tmpdir) / "cache" / "deep" / "nested" / "path"
        cache_file = nested_dir / "aspect_ratio_bucket_indices_deep-ds.json"
        self._write_cache(cache_file)

        config = {"id": "deep-ds"}
        service = DatasetViewerService(project_root=Path(self.tmpdir))
        result = service._find_cache_file(config, "indices")
        self.assertIsNotNone(result)
        self.assertEqual(result.resolve(), cache_file.resolve())


class TestGetDatasetSummary(unittest.TestCase):
    """Verify summary generation from cache files."""

    def setUp(self):
        self.service = DatasetViewerService()
        self.tmpdir = tempfile.mkdtemp()

    def test_summary_with_no_cache(self):
        config = {"id": "no-cache", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertEqual(summary.dataset_id, "no-cache")
        self.assertFalse(summary.has_cache)
        self.assertEqual(summary.total_files, 0)

    def test_summary_with_valid_cache(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_test-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {
                        "1.0": ["a.jpg", "b.jpg", "c.jpg"],
                        "0.75": ["d.jpg", "e.jpg"],
                    },
                    "filtering_statistics": {
                        "total_processed": 10,
                        "skipped": {"too_small": 5},
                    },
                }
            )
        )

        config = {"id": "test-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertTrue(summary.has_cache)
        self.assertEqual(summary.total_files, 5)
        self.assertEqual(len(summary.buckets), 2)
        self.assertEqual(summary.filtering_statistics["total_processed"], 10)

    def test_summary_with_empty_buckets(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_empty-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {},
                    "filtering_statistics": {
                        "total_processed": 50,
                        "skipped": {"metadata_missing": 50},
                    },
                }
            )
        )

        config = {"id": "empty-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertTrue(summary.has_cache)
        self.assertEqual(summary.total_files, 0)
        self.assertEqual(len(summary.buckets), 0)

    def test_summary_with_corrupt_json(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_bad-ds.json"
        cache_file.write_text("not valid json{{{")

        config = {"id": "bad-ds", "instance_data_dir": self.tmpdir}
        summary = self.service.get_dataset_summary(config)
        self.assertFalse(summary.has_cache)


class TestSourceShardListing(unittest.TestCase):
    def test_capture_shards_are_filterable_and_pageable_without_preparation(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            service = DatasetViewerService(root)
            capture = root / "capture.json"
            files = root / "files.jsonl"
            capture.write_text(
                json.dumps({"format": "parquet-image-capture", "version": 2, "files_manifest": "files.jsonl"})
            )
            config = {
                "id": "capture",
                "type": "webshart",
                "source": str(root),
                "webshart": {"format": "parquet", "manifest": "capture.json", "subfolder": "train"},
            }
            records = [
                {"key": f"{folder}/{name}.parquet", "size_bytes": size, "num_rows": rows}
                for folder, name, size, rows in [
                    ("train", "B", 32, 7),
                    ("train", "A", 64, 12),
                    ("test", "A", 16, 4),
                ]
            ]
            files.write_text("\n".join(json.dumps(item) for item in records))
            with patch(
                "simpletuner.helpers.data_backend.dataset_runtime.DatasetRuntime.from_config",
                side_effect=AssertionError("Shard browsing must not open sample metadata."),
            ):
                summary = service.get_dataset_summary(config)
                self.assertTrue(summary.supports_shard_listing)
                self.assertFalse(summary.has_cache)
                page = service.list_source_shards(config, {}, limit=1, offset=0)
                self.assertEqual((page.total, page.total_shards), (2, 2))
                self.assertEqual(page.shards[0].path, "train/A.parquet")
                self.assertEqual((page.shards[0].size_bytes, page.shards[0].rows), (64, 12))
                page = service.list_source_shards(config, {}, limit=1, offset=1)
                self.assertEqual(page.shards[0].path, "train/B.parquet")
                page = service.list_source_shards(config, {}, limit=1, offset=0, query="b.PARQUET")
                self.assertEqual((page.total, page.total_shards), (1, 2))
                existing = files.read_text()
                files.write_text(existing + "\n" + json.dumps({"key": "train/C.parquet", "size_bytes": 8, "num_rows": 2}))
                self.assertEqual(service.list_source_shards(config, {}, limit=1, offset=0).total, 2)
                self.assertEqual(service.list_source_shards(config, {}, limit=1, offset=0, refresh=True).total, 3)

    def test_reads_selected_parquet_rows_without_opening_dataset_preparation(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = io.BytesIO()
            Image.new("RGB", (640, 320), "red").save(payload, format="PNG")
            pq.write_table(
                pa.table({"pixels": [{"bytes": payload.getvalue(), "path": "first.png"}, None, None]}),
                root / "chosen.parquet",
            )
            (root / "unopened.parquet").touch()
            config = {
                "id": "test-shards",
                "type": "webshart",
                "source": directory,
                "webshart": {"format": "parquet", "columns": {"image": "pixels"}},
            }
            service = DatasetViewerService(root)
            with patch(
                "simpletuner.helpers.data_backend.dataset_runtime.DatasetRuntime.from_config",
                side_effect=AssertionError("Shard browsing must not open dataset preparation."),
            ):
                page = service.get_shard_samples(config, {}, "chosen.parquet", offset=0, limit=2)
                self.assertEqual(page.total, 3)
                self.assertEqual([sample.filename for sample in page.samples], ["Row 1", "Row 2"])
                self.assertTrue(page.samples[0].thumbnail.startswith("data:image/jpeg;base64,"))
                self.assertIsNone(page.samples[1].thumbnail)
                preview = service.get_shard_preview(config, {}, "chosen.parquet", offset=0)
                with Image.open(io.BytesIO(preview)) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.size, (640, 320))
                with self.assertRaises(FileNotFoundError):
                    service.get_shard_preview(config, {}, "chosen.parquet", offset=1)
                self.assertEqual(
                    service.get_shard_samples(config, {}, "chosen.parquet", offset=2, limit=2).samples[0].path, "2"
                )
                self.assertEqual(service.get_shard_samples(config, {}, "chosen.parquet", offset=3, limit=2).samples, [])
                with self.assertRaises(FileNotFoundError):
                    service.get_shard_samples(config, {}, "../chosen.parquet", offset=0, limit=2)

    def test_reads_samples_from_an_indexed_tar_shard(self):
        import tarfile

        from webshart import MetadataExtractor

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = io.BytesIO()
            Image.new("RGB", (8, 8), "red").save(image, format="PNG")
            data = image.getvalue()
            with tarfile.open(root / "samples.tar", "w") as archive:
                member = tarfile.TarInfo("sample.png")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

            MetadataExtractor().extract_metadata(directory, directory)
            config = {"id": "samples", "type": "webshart", "source": directory, "webshart": {"format": "tar"}}
            page = DatasetViewerService(root).get_shard_samples(config, {}, "samples.tar", offset=0, limit=1)

            self.assertEqual(page.total, 1)
            self.assertEqual(page.samples[0].filename, "sample.png")
            self.assertTrue(page.samples[0].thumbnail.startswith("data:image/jpeg;base64,"))

    def test_s3_capture_reads_the_catalogue_through_the_aws_client(self):
        from hashlib import sha256

        files = b'{"key":"train/A.parquet","size_bytes":64,"num_rows":12}\n'
        descriptor = json.dumps(
            {
                "format": "parquet-image-capture",
                "version": 2,
                "files_manifest": "files.jsonl",
                "manifest_sha256": sha256(files).hexdigest(),
            }
        ).encode()
        config = {
            "id": "capture",
            "type": "webshart",
            "source": "s3://example/captures",
            "webshart": {"format": "parquet", "manifest": "capture.json"},
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("webshart.parquet._S3Store") as native_store,
            patch("simpletuner.helpers.data_backend.aws._s3_client") as aws_client,
        ):
            native_store.return_value.head.return_value = {"size": len(descriptor)}
            native_store.return_value.read.return_value = descriptor
            boto = aws_client.return_value
            boto.head_object.return_value = {"ContentLength": len(files)}
            contents = iter((files, files.replace(b"64", b"65")))
            boto.download_file.side_effect = lambda bucket, key, filename, **kwargs: Path(filename).write_bytes(
                next(contents)
            )
            viewer = DatasetViewerService(Path(directory))
            page = viewer.list_source_shards(config, {}, limit=5, offset=0)
            self.assertEqual(
                [("train/A.parquet", 64, 12)], [(shard.path, shard.size_bytes, shard.rows) for shard in page.shards]
            )
            boto.head_object.assert_called_once_with(Bucket="example", Key="captures/files.jsonl")
            self.assertEqual(boto.download_file.call_args.args[:2], ("example", "captures/files.jsonl"))
            native_store.return_value.download_manifest.assert_not_called()
            with self.assertRaisesRegex(ValueError, "Manifest SHA-256 mismatch"):
                viewer.list_source_shards(config, {}, limit=5, offset=0, refresh=True)


class TestThumbnailGeneration(unittest.TestCase):
    """Verify thumbnail generation from file paths and PIL images."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def _create_test_image(self, size=(100, 100), mode="RGB") -> Path:
        img = Image.new(mode, size, (128, 64, 32))
        path = Path(self.tmpdir) / "test.jpg"
        img.save(path, format="JPEG")
        return path

    def test_generate_thumbnail_from_file(self):
        path = self._create_test_image()
        result = generate_thumbnail(path)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))

    def test_generate_thumbnail_from_pil_image(self):
        img = Image.new("RGB", (200, 200), (255, 0, 0))
        result = generate_thumbnail_from_pil(img)
        self.assertIsNotNone(result)
        self.assertTrue(result.startswith("data:image/jpeg;base64,"))

    def test_generate_thumbnail_from_rgba(self):
        img = Image.new("RGBA", (100, 100), (255, 0, 0, 128))
        result = generate_thumbnail_from_pil(img)
        self.assertIsNotNone(result)

    def test_generate_thumbnail_nonexistent_file(self):
        result = generate_thumbnail(Path("/nonexistent/image.jpg"))
        self.assertIsNone(result)


class TestLocalThumbnails(unittest.TestCase):
    """Verify get_dataset_thumbnails for local backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService()

    def _setup_local_dataset(self):
        """Create a local dataset with an image and bucket cache."""
        img = Image.new("RGB", (100, 100), (128, 64, 32))
        img.save(Path(self.tmpdir) / "photo.jpg", format="JPEG")

        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_local-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["photo.jpg"]},
                }
            )
        )
        return {"id": "local-ds", "type": "local", "instance_data_dir": self.tmpdir}

    def test_local_thumbnail_generated(self):
        config = self._setup_local_dataset()
        result = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].thumbnail)
        self.assertIsNone(result[0].error)

    def test_local_missing_file_returns_error(self):
        cache_file = Path(self.tmpdir) / "aspect_ratio_bucket_indices_missing-ds.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["gone.jpg"]},
                }
            )
        )
        config = {"id": "missing-ds", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].error, "File not found")


class TestHuggingfaceDisplayLabels(unittest.TestCase):
    """Verify viewer labels use dataset row metadata for Hugging Face backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

    def _write_hf_cache(
        self, dataset_id: str, indices: Dict[str, List[str]], metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        hf_dir = Path(self.tmpdir) / "cache" / "huggingface" / dataset_id / "huggingface_metadata" / dataset_id
        hf_dir.mkdir(parents=True, exist_ok=True)
        (hf_dir / f"aspect_ratio_bucket_indices_{dataset_id}.json").write_text(
            json.dumps({"config": {}, "aspect_ratio_bucket_indices": indices})
        )
        if metadata is not None:
            (hf_dir / f"aspect_ratio_bucket_metadata_{dataset_id}.json").write_text(json.dumps(metadata))

    def test_huggingface_thumbnail_uses_row_filename_label(self):
        dataset_id = "hf-ds"
        self._write_hf_cache(dataset_id, {"1.0": ["1.jpg"]})
        config = {"id": dataset_id, "type": "huggingface", "image_column": "image"}

        backend = MagicMock()
        backend._get_index_from_path.return_value = 1
        backend.get_dataset_item.return_value = {"file_name": "nested/path/real_photo.png"}

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with (
                patch.object(self.service, "_get_or_create_data_backend", return_value=backend),
                patch.object(self.service, "_read_remote_thumbnail", return_value=None),
            ):
                thumbnails = self.service.get_dataset_thumbnails(config, bucket_key="1.0")
        finally:
            os.chdir(original_cwd)

        self.assertEqual(len(thumbnails), 1)
        self.assertEqual(thumbnails[0].path, "1.jpg")
        self.assertEqual(thumbnails[0].filename, "real_photo.png")

    def test_huggingface_file_metadata_exposes_display_name(self):
        dataset_id = "hf-meta"
        self._write_hf_cache(
            dataset_id,
            {"1.0": ["7.jpg"]},
            metadata={
                "7.jpg": {
                    "original_size": [1024, 768],
                    "aspect_ratio": 1.3333,
                }
            },
        )
        config = {"id": dataset_id, "type": "huggingface", "image_column": "image"}

        backend = MagicMock()
        backend._get_index_from_path.return_value = 7
        backend.get_dataset_item.return_value = {"image": {"path": "/data/archive/folder/actual_name.webp"}}

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            with patch.object(self.service, "_get_or_create_data_backend", return_value=backend):
                metadata = self.service.get_file_metadata(config, "7.jpg")
        finally:
            os.chdir(original_cwd)

        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.path, "7.jpg")
        self.assertEqual(metadata.display_name, "actual_name.webp")


class TestImagePreview(unittest.TestCase):
    """Verify get_image_preview crop simulation pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

    def _create_image(self, size=(800, 600)):
        img = Image.new("RGB", size, (128, 64, 32))
        path = Path(self.tmpdir) / "photo.jpg"
        img.save(path, format="JPEG")
        return path

    def test_preview_without_metadata(self):
        self._create_image()
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_image_preview(config, "photo.jpg")
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["original"])
        self.assertIsNone(result["intermediary"])
        self.assertIsNone(result["cropped"])

    def test_remote_detail_uses_source_pixels_after_thumbnail_is_cached(self):
        config = {"id": "remote", "type": "aws"}
        image = Image.new("RGB", (640, 384), "red")
        image.paste("blue", (320, 0, 640, 384))
        backend = MagicMock()
        backend.read_image.return_value = image
        metadata = {"intermediary_size": (640, 384), "crop_coordinates": (0, 320), "target_size": (320, 384)}
        with patch.object(self.service, "_get_or_create_data_backend", return_value=backend):
            self.service._read_remote_thumbnail(config, "photo.jpg", 128)
            preview = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertEqual(backend.read_image.call_count, 2)
        with Image.open(io.BytesIO(base64.b64decode(preview["original"].split(",", 1)[1]))) as original:
            self.assertEqual(original.size, (640, 384))
        with Image.open(io.BytesIO(base64.b64decode(preview["cropped"].split(",", 1)[1]))) as cropped:
            self.assertEqual(cropped.size, (320, 384))
            red, _, blue = cropped.getpixel((100, 100))
            self.assertGreater(blue, red)

    def test_preview_with_crop_metadata(self):
        self._create_image(size=(2000, 1500))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [1200, 900],
            "crop_coordinates": [100, 50],
            "target_size": [1024, 768],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result["original"])
        self.assertIsNotNone(result["intermediary"])
        self.assertIsNotNone(result["cropped"])
        # All should be base64 data URLs
        for key in ("original", "intermediary", "cropped"):
            self.assertTrue(result[key].startswith("data:image/jpeg;base64,"))

    def test_preview_with_zero_crop_offset(self):
        """Center or corner crop with (0,0) offset should work."""
        self._create_image(size=(1024, 1024))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [1024, 1024],
            "crop_coordinates": [0, 0],
            "target_size": [1024, 1024],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])

    def test_preview_missing_file(self):
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        result = self.service.get_image_preview(config, "missing.jpg")
        self.assertIsNone(result)

    def test_preview_partial_metadata_skips_crop(self):
        """If only some crop fields are present, skip simulation."""
        self._create_image()
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {"intermediary_size": [400, 300]}  # missing crop_coordinates and target_size
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["original"])
        self.assertIsNone(result["intermediary"])
        self.assertIsNone(result["cropped"])

    def test_preview_clamps_crop_to_intermediary_bounds(self):
        """Crop coordinates that would exceed intermediary bounds get clamped.

        Reproduces the user-reported case: 810x1080 original, 512x682 intermediary,
        512x512 target at crop_coordinates (143, 0) in (top, left) order.
        Croppers and the batch crop path store (top, left) matching SDXL conditioning.
        """
        self._create_image(size=(810, 1080))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        metadata = {
            "intermediary_size": [512, 682],
            "crop_coordinates": [143, 0],  # (top=143, left=0)
            "target_size": [512, 512],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        # Decode the cropped image to verify dimensions
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # Should be 512x512 since (top=143, left=0) fits in 512w x 682h
        self.assertEqual(cropped_img.size[0], 512)
        self.assertEqual(cropped_img.size[1], 512)

    def test_preview_clamps_when_crop_truly_overflows(self):
        """When crop genuinely overflows, the result should be clamped smaller."""
        self._create_image(size=(400, 400))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        # Intermediary 200x200, target 200x200, but crop starts at (50, 50)
        # Clamped: right = 50 + min(200, 200-50) = 200, bottom = 50 + min(200, 200-50) = 200
        # Result: 150x150 (clamped)
        metadata = {
            "intermediary_size": [200, 200],
            "crop_coordinates": [50, 50],  # (top=50, left=50) — symmetric so order doesn't matter
            "target_size": [200, 200],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # Clamped: 150x150
        self.assertEqual(cropped_img.size[0], 150)
        self.assertEqual(cropped_img.size[1], 150)

    def test_preview_crop_coordinate_order_asymmetric(self):
        """Verify (top, left) order: left=0 should produce full-width crop on portrait."""
        self._create_image(size=(512, 800))
        config = {"id": "test", "type": "local", "instance_data_dir": self.tmpdir}
        # Portrait intermediary 512w x 682h, target 512x512
        # crop_coordinates = (top=211, left=0) → full width, offset vertically
        metadata = {
            "intermediary_size": [512, 682],
            "crop_coordinates": [211, 0],  # (top=211, left=0)
            "target_size": [512, 512],
        }
        result = self.service.get_image_preview(config, "photo.jpg", metadata=metadata)
        self.assertIsNotNone(result["cropped"])
        b64 = result["cropped"].split(",", 1)[1]
        cropped_img = Image.open(io.BytesIO(base64.b64decode(b64)))
        # left=0, width=512 (full width); top=211, height=min(512, 682-211)=471
        self.assertEqual(cropped_img.size[0], 512)
        self.assertEqual(cropped_img.size[1], 471)


class TestFileCaption(unittest.TestCase):
    """Verify caption loading in get_file_metadata."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.dataset_id = "cap-test"
        # Create image dir with image, caption sidecar, and metadata cache
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (800, 600)).save(self.img_dir / "photo.jpg")
        (self.img_dir / "photo.txt").write_text("a painting of a sunset")
        # Place metadata cache in instance_data_dir so _find_cache_file finds it
        meta_file = self.img_dir / f"aspect_ratio_bucket_metadata_{self.dataset_id}.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "target_size": [512, 512],
                        "aspect_ratio": 1.333,
                    }
                }
            )
        )

    def _config(self, **overrides):
        base = {
            "id": self.dataset_id,
            "type": "local",
            "instance_data_dir": str(self.img_dir),
            "caption_strategy": "textfile",
        }
        base.update(overrides)
        return base

    def test_textfile_caption_loaded(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "a painting of a sunset")

    def test_filename_caption(self):
        meta = self.service.get_file_metadata(self._config(caption_strategy="filename"), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "photo")

    def test_instanceprompt_caption(self):
        meta = self.service.get_file_metadata(
            self._config(caption_strategy="instanceprompt", instance_prompt="a dog"),
            "photo.jpg",
        )
        self.assertIsNotNone(meta)
        self.assertEqual(meta.caption, "a dog")

    def test_missing_textfile_returns_none(self):
        """When caption file doesn't exist, caption should be None."""
        (self.img_dir / "photo.txt").unlink()
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta)
        self.assertIsNone(meta.caption)


class TestUpdateCropCoordinates(unittest.TestCase):
    """Verify crop coordinate persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        meta_file = self.img_dir / "aspect_ratio_bucket_metadata_crop-ds.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "intermediary_size": [512, 384],
                        "target_size": [512, 384],
                        "crop_coordinates": [0, 0],
                        "aspect_ratio": 1.333,
                    }
                }
            )
        )

    def _config(self):
        return {"id": "crop-ds", "type": "local", "instance_data_dir": str(self.img_dir)}

    def test_update_saves_to_cache(self):
        ok = self.service.update_crop_coordinates(self._config(), "photo.jpg", [50, 30])
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertEqual(meta.crop_coordinates, [50, 30])

    def test_edits_reject_stale_metadata_and_publish_the_new_revision(self):
        from simpletuner.helpers.metadata.preparation import DatasetViewChangedError

        config = self._config()
        original = self.service.get_file_metadata(config, "photo.jpg")
        edit = self.service.update_crop_coordinates(
            config, "photo.jpg", [12, 34], metadata_revision=original.metadata_revision
        )
        current = self.service.get_file_metadata(config, "photo.jpg")
        self.assertEqual(current.metadata_revision, edit.revision)
        self.assertNotEqual(current.metadata_revision, original.metadata_revision)
        with self.assertRaises(DatasetViewChangedError):
            self.service.update_bbox_entities(config, "photo.jpg", [], metadata_revision=original.metadata_revision)
        with self.assertRaises(TypeError):
            self.service.update_bbox_entities(
                config, "photo.jpg", [{"invalid": object()}], metadata_revision=current.metadata_revision
            )
        self.assertEqual(self.service.get_file_metadata(config, "photo.jpg"), current)
        newer = self.service.update_crop_coordinates(config, "photo.jpg", [45, 67], metadata_revision=edit.revision)
        self.assertEqual(self.service.get_file_metadata(config, "photo.jpg").metadata_revision, newer.revision)

    def test_rebuild_uses_viewer_settings_without_changing_the_registered_dataset(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = dict(self._config(), crop=True, crop_style="center", crop_aspect="square")
        active = DatasetRuntime(config["id"], None, None, {"resolution": 1024})
        with (
            patch.object(StateTracker, "datasets", {active.id: active}),
            patch.object(StateTracker, "set_args", side_effect=AssertionError("changed trainer args")),
            patch.object(StateTracker, "set_data_backend_config", side_effect=AssertionError("changed trainer dataset")),
        ):
            updated = self.service.rebuild_file_metadata(
                config, {"resolution": 64, "resolution_type": "pixel", "aspect_bucket_alignment": 8}, "photo.jpg"
            )
            self.assertEqual(updated.record["target_size"], [64, 64])
            self.assertEqual(StateTracker.get_dataset(active.id).config["resolution"], 1024)
        self.assertEqual(self.service.get_file_metadata(config, "photo.jpg").target_size, [64, 64])

    def test_update_missing_file_returns_false(self):
        ok = self.service.update_crop_coordinates(self._config(), "missing.jpg", [10, 10])
        self.assertFalse(ok)

    def test_update_no_cache_returns_false(self):
        config = {"id": "no-cache", "type": "local", "instance_data_dir": self.tmpdir}
        ok = self.service.update_crop_coordinates(config, "photo.jpg", [10, 10])
        self.assertFalse(ok)


class TestUpdateBboxEntities(unittest.TestCase):
    """Verify bbox entity persistence."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        self.img_dir = Path(self.tmpdir) / "images"
        self.img_dir.mkdir(parents=True, exist_ok=True)
        meta_file = self.img_dir / "aspect_ratio_bucket_metadata_bbox-ds.json"
        meta_file.write_text(
            json.dumps(
                {
                    "photo.jpg": {
                        "original_size": [800, 600],
                        "target_size": [512, 384],
                        "aspect_ratio": 1.333,
                        "bbox_entities": [
                            {"label": "cat", "bbox": [0.1, 0.2, 0.5, 0.8]},
                        ],
                    }
                }
            )
        )

    def _config(self):
        return {"id": "bbox-ds", "type": "local", "instance_data_dir": str(self.img_dir)}

    def test_get_file_metadata_returns_bbox_entities(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNotNone(meta.bbox_entities)
        self.assertEqual(len(meta.bbox_entities), 1)
        self.assertEqual(meta.bbox_entities[0]["label"], "cat")
        self.assertAlmostEqual(meta.bbox_entities[0]["bbox"][0], 0.1)

    def test_bbox_entities_not_in_extra(self):
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertNotIn("bbox_entities", meta.extra)

    def test_update_saves_to_cache(self):
        new_entities = [
            {"label": "dog", "bbox": [0.2, 0.3, 0.6, 0.9]},
            {"label": "cat", "bbox": [0.0, 0.0, 0.5, 0.5]},
        ]
        ok = self.service.update_bbox_entities(self._config(), "photo.jpg", new_entities)
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertEqual(len(meta.bbox_entities), 2)
        self.assertEqual(meta.bbox_entities[0]["label"], "dog")

    def test_update_with_none_removes_entities(self):
        ok = self.service.update_bbox_entities(self._config(), "photo.jpg", None)
        self.assertTrue(ok)
        meta = self.service.get_file_metadata(self._config(), "photo.jpg")
        self.assertIsNone(meta.bbox_entities)

    def test_update_missing_file_returns_false(self):
        ok = self.service.update_bbox_entities(self._config(), "missing.jpg", [])
        self.assertFalse(ok)

    def test_update_no_cache_returns_false(self):
        config = {"id": "no-cache", "type": "local", "instance_data_dir": self.tmpdir}
        ok = self.service.update_bbox_entities(config, "photo.jpg", [])
        self.assertFalse(ok)


class TestConditioningFileMatch(unittest.TestCase):
    """Verify conditioning file matching by provider identity."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))

        # Source dataset
        self.src_dir = Path(self.tmpdir) / "source_images"
        self.src_dir.mkdir(parents=True)
        # Create a dummy source image
        img = Image.new("RGB", (64, 64), color="red")
        img.save(self.src_dir / "photo123.jpg")

        # Conditioning dataset (in plan)
        self.cond_dir = Path(self.tmpdir) / "cond_images"
        self.cond_dir.mkdir(parents=True)
        # Create a matching conditioning image
        cond_img = Image.new("RGB", (64, 64), color="green")
        cond_img.save(self.cond_dir / "photo123.jpg")

        # Bucket cache for conditioning dataset
        cond_cache = self.cond_dir / "aspect_ratio_bucket_indices_cond-ds.json"
        cond_cache.write_text(
            json.dumps(
                {
                    "aspect_ratio_bucket_indices": {"1.0": ["photo123.jpg"]},
                }
            )
        )

    def _src_config(self):
        return {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning_data": ["cond-ds"],
        }

    def _cond_config(self):
        return {
            "id": "cond-ds",
            "type": "local",
            "instance_data_dir": str(self.cond_dir),
            "conditioning_type": "controlnet",
        }

    def test_match_found_via_conditioning_data(self):
        all_datasets = [self._src_config(), self._cond_config()]
        result = self.service.get_conditioning_file_match(self._src_config(), all_datasets, "photo123.jpg")
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "cond-ds")
        self.assertEqual(result["conditioning_type"], "controlnet")
        self.assertIn("thumbnail", result)

    def test_no_match_for_missing_path(self):
        all_datasets = [self._src_config(), self._cond_config()]
        result = self.service.get_conditioning_file_match(self._src_config(), all_datasets, "nonexistent.jpg")
        self.assertIsNone(result)

    def test_nested_relative_paths_keep_same_name_pairs_and_orphans_distinct(self):
        source, target = self._src_config(), self._cond_config()
        for directory in (self.src_dir, self.cond_dir):
            for prefix in ("first", "second"):
                (directory / prefix).mkdir()
                Image.new("RGB", (64, 64), color="red").save(directory / prefix / "same.png")
        for directory, config, refs in (
            (self.src_dir, source, ["first/same.png", "second/same.png", "source_only/same.png"]),
            (self.cond_dir, target, ["first/same.png", "second/same.png", "target_only/same.png"]),
        ):
            (directory / f"aspect_ratio_bucket_indices_{config['id']}.json").write_text(
                json.dumps({"aspect_ratio_bucket_indices": {"1.0": refs}})
            )

        page = self.service.get_conditioning_pairs(source, target, offset=1, limit=1)
        self.assertEqual(page["total_pairs"], 2)
        self.assertEqual(page["pairs"][0]["conditioning_path"], "second/same.png")
        report = self.service.get_orphaned_files(source, target)
        self.assertEqual(report.source_orphans, ["source_only/same.png"])
        self.assertEqual(report.conditioning_orphans, ["target_only/same.png"])
        match = self.service.get_conditioning_file_match(source, [target], "second/same.png")
        self.assertEqual(Path(match["file_path"]), self.cond_dir / "second/same.png")
        self.assertIsNone(self.service.get_conditioning_file_match(source, [target], "source_only/same.png"))

    def test_no_match_without_conditioning(self):
        config = {"id": "plain-ds", "type": "local", "instance_data_dir": str(self.src_dir)}
        result = self.service.get_conditioning_file_match(config, [config], "photo123.jpg")
        self.assertIsNone(result)

    def test_match_via_filesystem_discovery(self):
        """Auto-generated conditioning images found on disk with explicit id."""
        gen_dir = Path(self.tmpdir) / "conditioning_data" / "auto-gen-canny"
        gen_dir.mkdir(parents=True)
        gen_img = Image.new("RGB", (64, 64), color="blue")
        gen_img.save(gen_dir / "photo123.jpg")

        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning": [
                {"id": "auto-gen-canny", "type": "canny", "instance_data_dir": str(gen_dir)},
            ],
        }
        result = self.service.get_conditioning_file_match(src_config, [src_config], "photo123.jpg")
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "src-ds_conditioning_canny")
        self.assertEqual(result["conditioning_type"], "canny")

    def test_match_via_derived_id_and_path(self):
        """Generator config without explicit id uses {source_id}_conditioning_{type}."""
        # Simulate the directory structure the duplicator would create
        gen_dir = Path(self.tmpdir) / "conditioning_data" / "src-ds_conditioning_canny"
        gen_dir.mkdir(parents=True)
        gen_img = Image.new("RGB", (64, 64), color="blue")
        gen_img.save(gen_dir / "photo123.jpg")

        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "cache_dir_vae": str(Path(self.tmpdir) / "independent-latent-store"),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
        }
        result = self.service.get_conditioning_file_match(
            src_config, [src_config], "photo123.jpg", global_config={"cache_dir": self.tmpdir}
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "src-ds_conditioning_canny")
        self.assertEqual(result["conditioning_type"], "controlnet")

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_viewer_service.GENERATOR_REGISTRY_AVAILABLE", True)
    def test_on_the_fly_generation(self):
        """When conditioning images don't exist, generate preview on-the-fly."""
        src_config = {
            "id": "src-ds",
            "type": "local",
            "instance_data_dir": str(self.src_dir),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
        }

        # Mock the generator to avoid needing trainingsample C extension
        mock_result = Image.new("RGB", (64, 64), color="white")
        mock_generator = MagicMock()
        mock_generator.transform_batch.return_value = [mock_result]

        mock_registry = {"canny": MagicMock(return_value=mock_generator)}
        with patch(
            "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service.GENERATOR_REGISTRY",
            mock_registry,
        ):
            result = self.service.get_conditioning_file_match(src_config, [src_config], "photo123.jpg")

        self.assertIsNotNone(result)
        self.assertEqual(result["conditioning_id"], "src-ds_conditioning_canny")
        self.assertEqual(result["conditioning_type"], "controlnet")
        self.assertTrue(result.get("generated_on_the_fly"))
        self.assertIn("thumbnail", result)


class TestRemoteThumbnails(unittest.TestCase):
    """Verify on-demand thumbnail fetching for remote backends."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetViewerService(project_root=Path(self.tmpdir))
        # Clear the module-level backend cache between tests
        _data_backend_cache.clear()

    def tearDown(self):
        _data_backend_cache.clear()

    def _setup_hf_cache(self, dataset_id="hf-ds"):
        """Create a bucket cache for a HuggingFace dataset."""
        cache_dir = Path(self.tmpdir) / "cache" / "huggingface" / dataset_id / "huggingface_metadata" / dataset_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"aspect_ratio_bucket_indices_{dataset_id}.json"
        cache_file.write_text(
            json.dumps(
                {
                    "config": {},
                    "aspect_ratio_bucket_indices": {"1.0": ["0.jpg", "1.jpg"]},
                }
            )
        )
        return cache_dir

    def test_cached_thumbnail_served_from_disk(self):
        """Pre-cached thumbnails should be returned without creating a backend."""
        self._setup_hf_cache()
        config = {"id": "hf-ds", "type": "huggingface"}
        path = self.service._get_thumbnail_cache_path(config, "0.jpg")
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64), (255, 0, 0)).save(path, format="JPEG")

        with patch.object(self.service, "_get_or_create_data_backend", side_effect=AssertionError("opened reader")):
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)
        self.assertEqual(len(result), 1)
        self.assertIsNotNone(result[0].thumbnail)
        self.assertIsNone(result[0].error)

    def test_reused_dataset_ids_do_not_share_readers_or_thumbnails_between_sources(self):
        self._setup_hf_cache()
        root = Path(self.tmpdir)
        other_project = DatasetViewerService(project_root=root, plan_path=root / "second-plan.json")
        config = {"id": "hf-ds", "type": "huggingface", "dataset_name": "repo-a"}
        changed = {**config, "dataset_name": "repo-b"}
        requests = [(self.service, config), (self.service, changed), (other_project, config)]
        readers = []
        for color in ("red", "blue", "green"):
            reader = MagicMock()
            reader.read_image.return_value = Image.new("RGB", (64, 64), color)
            readers.append(reader)

        with patch.object(DatasetViewerService, "_create_data_backend", side_effect=readers) as create:
            pages = [service.get_dataset_thumbnails(source, limit=1) for service, source in requests]
            self.assertEqual(create.call_count, 3)
            self.assertEqual(len({page[0].thumbnail for page in pages}), 3)
            for service, source in requests:
                self.assertTrue(service._get_thumbnail_cache_path(source, "0.jpg").exists())
                service.get_dataset_thumbnails(source, limit=1)
            self.assertEqual(create.call_count, 3)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_on_demand_fetch_generates_and_caches_thumbnail(self, mock_get_backend):
        """When no cached thumbnail, should fetch from backend, cache, and return."""
        self._setup_hf_cache()
        mock_backend = MagicMock()
        mock_backend.read_image.return_value = Image.new("RGB", (200, 200), (0, 255, 0))
        mock_get_backend.return_value = mock_backend

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNotNone(result[0].thumbnail)
            self.assertIsNone(result[0].error)
            mock_backend.read_image.assert_called_once_with("0.jpg")

            # Verify thumbnail was cached to disk
            cached = self.service._get_thumbnail_cache_path(config, "0.jpg")
            self.assertTrue(cached.is_file())
        finally:
            os.chdir(original_cwd)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_backend_failure_returns_error(self, mock_get_backend):
        """If backend creation fails, thumbnail should report error."""
        self._setup_hf_cache()
        mock_get_backend.side_effect = RuntimeError("Cannot connect")

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].thumbnail)
            self.assertEqual(result[0].error, "Failed to load remote image")
        finally:
            os.chdir(original_cwd)

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_viewer_service"
        ".DatasetViewerService._get_or_create_data_backend"
    )
    def test_read_image_returns_none(self, mock_get_backend):
        """If read_image returns None, thumbnail should report error."""
        self._setup_hf_cache()
        mock_backend = MagicMock()
        mock_backend.read_image.return_value = None
        mock_get_backend.return_value = mock_backend

        import os

        original_cwd = os.getcwd()
        os.chdir(self.tmpdir)
        try:
            config = {"id": "hf-ds", "type": "huggingface"}
            result = self.service.get_dataset_thumbnails(config, bucket_key="1.0", limit=1)

            self.assertEqual(len(result), 1)
            self.assertIsNone(result[0].thumbnail)
            self.assertEqual(result[0].error, "Failed to load remote image")
        finally:
            os.chdir(original_cwd)


class TestClearVaeCache(unittest.TestCase):
    """Verify VAE cache clearing in DatasetScanService."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_preserves_unrelated_files(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        vae_dir = Path(self.tmpdir) / "cache" / "vae" / "test-ds"
        vae_dir.mkdir(parents=True, exist_ok=True)
        latent = vae_dir / ("1" * 64 + ".pt")
        latent.write_bytes(b"fake")
        checkpoint = vae_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"keep")

        service = DatasetScanService()
        config = {"id": "test-ds", "cache_dir_vae": str(vae_dir)}
        result = service.clear_vae_cache(config, {})
        self.assertIsNotNone(result)
        self.assertFalse(latent.exists())
        self.assertTrue(checkpoint.is_file())

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_no_dir_returns_none(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        service = DatasetScanService()
        config = {"id": "test-ds", "cache_dir_vae": "/nonexistent/path"}
        result = service.clear_vae_cache(config, {})
        self.assertIsNone(result)

    def test_clear_vae_cache_no_config_returns_none(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        service = DatasetScanService()
        config = {"id": "test-ds"}
        result = service.clear_vae_cache(config, {})
        self.assertIsNone(result)

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service" ".DatasetScanService._resolve_scan_output_dir")
    def test_clear_vae_cache_resolves_template_variables(self, mock_resolve):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService

        mock_resolve.return_value = self.tmpdir
        vae_dir = Path(self.tmpdir) / "cache" / "vae" / "flux" / "my-ds"
        vae_dir.mkdir(parents=True, exist_ok=True)
        latent = vae_dir / ("1" * 64 + ".pt")
        latent.write_bytes(b"fake")

        service = DatasetScanService()
        config = {
            "id": "my-ds",
            "cache_dir_vae": "{output_dir}/cache/vae/{model_family}/{id}",
        }
        global_config = {"model_family": "flux"}
        result = service.clear_vae_cache(config, global_config)
        self.assertIsNotNone(result)
        self.assertFalse(latent.exists())


if __name__ == "__main__":
    unittest.main()
