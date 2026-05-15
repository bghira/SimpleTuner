import threading
import unittest

import numpy as np
from PIL import Image

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.image_manipulation.nsfw_classifier import (
    DEFAULT_NSFW_CHECK_MODELS_CSV,
    NsfwClassifierModelStore,
    NsfwModelSpec,
    extract_classifier_frames,
    parse_nsfw_model_specs,
    score_sum,
)


class FakeClassifierStore:
    def __init__(self, rejected_paths):
        self.rejected_paths = set(rejected_paths)

    def classify_sample(self, sample, *, filepath=None):
        rejected = filepath in self.rejected_paths
        verdict = "nsfw" if rejected else "sfw"
        return {
            "filepath": filepath,
            "frames_scanned": 1,
            "frame_results": [
                {
                    "classifiers": [
                        {
                            "key": "fake/nsfw",
                            "model_id": "fake/nsfw",
                            "verdict": verdict,
                        }
                    ]
                }
            ],
            "summary": {
                "flagged_frames": 1 if rejected else 0,
                "rejected": rejected,
            },
        }


class FakeImageBackend:
    id = "train"
    type = "local"

    def __init__(self, samples):
        self.samples = samples
        self.deleted = []

    def read_image_batch(self, filepaths, delete_problematic_images=False):
        return list(filepaths), [self.samples[filepath] for filepath in filepaths]

    def delete(self, filepath):
        self.deleted.append(filepath)


class FakeAccelerator:
    is_main_process = True
    is_local_main_process = True
    num_processes = 1
    device = "cpu"

    def __init__(self):
        self.waits = 0

    def wait_for_everyone(self):
        self.waits += 1


class FakeMetadataBackend:
    def __init__(self):
        self.filtering_statistics = None
        self.bucket_report = None
        self.removed = []
        self.saved = False

    def remove_image(self, filepath, bucket=None):
        self.removed.append((filepath, bucket))

    def save_cache(self, enforce_constraints=False):
        self.saved = True


class FakeReadOnlyMetadataBackend:
    def __init__(self, split_cache, unsplit_cache):
        self.aspect_ratio_bucket_indices = {key: list(value) for key, value in split_cache.items()}
        self.unsplit_cache = {key: list(value) for key, value in unsplit_cache.items()}
        self.saved_cache = None
        self.filtering_statistics = None
        self.bucket_report = None
        self.read_only = True

    def reload_cache(self, set_config=True):
        self.aspect_ratio_bucket_indices = {key: list(value) for key, value in self.unsplit_cache.items()}

    def save_cache(self, enforce_constraints=False):
        self.saved_cache = {key: list(value) for key, value in self.aspect_ratio_bucket_indices.items()}


class FakeVotingStore(NsfwClassifierModelStore):
    def __init__(self, verdicts_by_model, **kwargs):
        specs = [NsfwModelSpec(model_id=model_id, threshold=0.5) for model_id in verdicts_by_model]
        super().__init__(model_specs=specs, min_votes=2, device="cpu", **kwargs)
        self.verdicts_by_model = verdicts_by_model

    def _classify_frame_with_model(self, image, model_spec):
        verdict = self.verdicts_by_model[model_spec.model_id]
        return {
            "key": model_spec.key,
            "model_id": model_spec.model_id,
            "threshold": model_spec.threshold,
            "top_label": "unsafe" if verdict == "nsfw" else "safe",
            "top_score": 0.9,
            "nsfw_score": 0.9 if verdict == "nsfw" else 0.1,
            "sfw_score": 0.1 if verdict == "nsfw" else 0.9,
            "verdict": verdict,
            "elapsed_ms": 0.0,
        }


def make_vae_for_nsfw_filter(samples, rejected_paths, delete_nsfw_images=False):
    vae = VAECache.__new__(VAECache)
    vae.id = "train"
    vae.dataset_type = "image"
    vae.nsfw_check_enabled = True
    vae.read_batch_size = 2
    vae.delete_problematic_images = False
    vae.delete_nsfw_images = delete_nsfw_images
    vae.accelerator = FakeAccelerator()
    vae.image_data_backend = FakeImageBackend(samples)
    vae.metadata_backend = FakeMetadataBackend()
    vae._nsfw_classifier_store = FakeClassifierStore(rejected_paths)
    vae._nsfw_lock = threading.Lock()
    vae._nsfw_scan_report = {
        "images_scanned": 0,
        "images_rejected": 0,
        "deleted_images": 0,
        "errors": [],
        "rejected_images": [],
        "classifier_verdicts": {},
    }
    return vae


class NsfwClassifierHelperTest(unittest.TestCase):
    def test_default_models_are_transformers_only(self):
        self.assertEqual(
            DEFAULT_NSFW_CHECK_MODELS_CSV,
            "Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5",
        )
        self.assertNotIn("Marqo/", DEFAULT_NSFW_CHECK_MODELS_CSV)

    def test_parse_nsfw_model_specs_supports_thresholds(self):
        specs = parse_nsfw_model_specs("foo/bar:threshold=0.75,bar/baz")
        self.assertEqual([spec.model_id for spec in specs], ["foo/bar", "bar/baz"])
        self.assertEqual(specs[0].threshold, 0.75)
        self.assertEqual(specs[1].threshold, 0.5)

    def test_parse_nsfw_model_specs_rejects_bad_thresholds(self):
        with self.assertRaises(ValueError):
            parse_nsfw_model_specs("foo/bar:threshold=1.5")

    def test_score_sum_matches_unsafe_label_hints(self):
        scores = [
            {"label": "neutral", "score": 0.2},
            {"label": "pornographic_content", "score": 0.55},
            {"label": "sexy", "score": 0.15},
        ]
        self.assertAlmostEqual(score_sum(scores, ("porn", "sexy")), 0.7)

    def test_score_sum_does_not_count_nsfw_as_sfw(self):
        scores = [
            {"label": "nsfw", "score": 0.9},
            {"label": "sfw", "score": 0.1},
        ]
        self.assertAlmostEqual(score_sum(scores, ("sfw", "safe")), 0.1)

    def test_video_frame_selection_modes(self):
        frames = np.stack([np.full((2, 2, 3), index, dtype=np.uint8) for index in range(5)])

        uniform = extract_classifier_frames(frames, frame_count=3, selection="uniform")
        first = extract_classifier_frames(frames, frame_count=3, selection="first")
        middle = extract_classifier_frames(frames, frame_count=3, selection="middle")

        self.assertEqual([frame.getpixel((0, 0))[0] for frame in uniform], [0, 2, 4])
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in first], [0, 1, 2])
        self.assertEqual([frame.getpixel((0, 0))[0] for frame in middle], [1, 2, 3])

    def test_classifier_store_uses_min_votes_for_rejection(self):
        image = Image.new("RGB", (4, 4), color="white")
        accepted_store = FakeVotingStore({"model/a": "nsfw", "model/b": "sfw"})
        rejected_store = FakeVotingStore({"model/a": "nsfw", "model/b": "nsfw"})

        self.assertFalse(accepted_store.classify_sample(image)["summary"]["rejected"])
        self.assertTrue(rejected_store.classify_sample(image)["summary"]["rejected"])

    def test_classifier_store_rejects_impossible_video_frame_threshold(self):
        with self.assertRaises(ValueError):
            NsfwClassifierModelStore(
                model_specs=[NsfwModelSpec(model_id="model/a")],
                min_votes=1,
                video_frame_count=1,
                video_min_flagged_frames=2,
                device="cpu",
            )


class VaeNsfwFilterTest(unittest.TestCase):
    def test_filter_removes_rejected_sample_from_metadata_without_deleting_file(self):
        good = "good.png"
        bad = "bad.png"
        samples = {
            good: Image.new("RGB", (4, 4), color="white"),
            bad: Image.new("RGB", (4, 4), color="black"),
        }
        vae = make_vae_for_nsfw_filter(samples, rejected_paths={bad})

        safe_files = vae._filter_nsfw_relevant_files([good, bad], bucket="1.0")

        self.assertEqual(safe_files, [good])
        self.assertEqual(vae.metadata_backend.removed, [(bad, "1.0")])
        self.assertEqual(vae.image_data_backend.deleted, [])
        self.assertEqual(vae.metadata_backend.filtering_statistics["skipped"]["nsfw"], 1)
        self.assertEqual(vae._nsfw_scan_report["images_scanned"], 2)
        self.assertEqual(vae._nsfw_scan_report["images_rejected"], 1)

    def test_filter_deletes_rejected_sample_when_enabled(self):
        bad = "bad.png"
        samples = {bad: Image.new("RGB", (4, 4), color="black")}
        vae = make_vae_for_nsfw_filter(samples, rejected_paths={bad}, delete_nsfw_images=True)

        safe_files = vae._filter_nsfw_relevant_files([bad], bucket="1.0")
        vae._finalize_deferred_metadata_filters()

        self.assertEqual(safe_files, [])
        self.assertEqual(vae.image_data_backend.deleted, [bad])
        self.assertEqual(vae._nsfw_scan_report["deleted_images"], 1)

    def test_read_only_metadata_cache_is_reloaded_before_deleting_rejected_sample(self):
        bad = "bad.png"
        vae = make_vae_for_nsfw_filter({}, rejected_paths=set(), delete_nsfw_images=True)
        vae.metadata_backend = FakeReadOnlyMetadataBackend(
            split_cache={"1.0": [bad]},
            unsplit_cache={"1.0": ["good.png", bad], "2.0": ["other.png"]},
        )
        vae._queue_metadata_filter_action(
            filepath=bad,
            bucket="1.0",
            reason="nsfw",
            delete_from_backend=True,
        )

        vae._finalize_deferred_metadata_filters()

        self.assertEqual(vae.metadata_backend.aspect_ratio_bucket_indices, {"1.0": [bad]})
        self.assertEqual(vae.metadata_backend.saved_cache, {"1.0": ["good.png"], "2.0": ["other.png"]})
        self.assertEqual(vae.image_data_backend.deleted, [bad])
        self.assertEqual(vae._nsfw_scan_report["deleted_images"], 1)

    def test_eval_dataset_type_is_not_scanned_even_when_sample_types_all(self):
        vae = VAECache.__new__(VAECache)
        vae.nsfw_check_requested = True
        vae.nsfw_check_backend_types = "all"
        vae.nsfw_check_sample_types = "all"
        vae.image_data_backend = FakeImageBackend({})
        vae.dataset_type = "eval"
        vae.debug_log = lambda *_args, **_kwargs: None

        self.assertFalse(vae._resolve_nsfw_check_enabled())


if __name__ == "__main__":
    unittest.main()
