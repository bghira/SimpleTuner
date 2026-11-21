import copy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.training.collate import (
    check_latent_shapes,
    collate_fn,
    compute_latents,
    compute_prompt_embeddings,
    describe_missing_conditioning_pairs,
    pixel_batch,
)
from simpletuner.helpers.training.state_tracker import StateTracker


class _StubModel:
    AUTOENCODER_CLASS = object

    def __init__(
        self,
        requires_conditioning: bool = False,
        use_reference_embeds: bool = False,
        requires_conditioning_latents: bool = False,
        requires_conditioning_dataset: bool = False,
        requires_text_embed_image_context: bool = False,
        uses_text_embeddings_cache: bool = True,
    ):
        self._requires_conditioning = requires_conditioning
        self._use_reference_embeds = use_reference_embeds
        self._requires_conditioning_latents = requires_conditioning_latents
        self._requires_conditioning_dataset = requires_conditioning_dataset
        self._requires_text_embed_context = requires_text_embed_image_context
        self._uses_text_embeddings_cache = uses_text_embeddings_cache

    def requires_conditioning_image_embeds(self):
        return self._requires_conditioning

    def requires_conditioning_latents(self):
        return self._requires_conditioning_latents

    def requires_conditioning_dataset(self):
        return self._requires_conditioning_dataset

    def collate_prompt_embeds(self, encoder_outputs):
        return {}

    def conditioning_image_embeds_use_reference_dataset(self):
        return self._use_reference_embeds

    def get_transforms(self, dataset_type: str | None = None):
        return None

    def requires_text_embed_image_context(self):
        return self._requires_text_embed_context

    def uses_text_embeddings_cache(self):
        return self._uses_text_embeddings_cache


class _StubConditioningSample:
    def __init__(
        self,
        training_path: str,
        backend_id: str,
        caption=None,
        paired_training_path: str = None,
        conditioning_type: str | None = None,
    ):
        self._training_path = training_path
        self._source_dataset_id = backend_id
        self._image_path = training_path
        self.caption = caption
        self.data_backend_id = backend_id
        self._paired_training_path = paired_training_path or training_path
        self._conditioning_type = conditioning_type

    def training_sample_path(self, training_dataset_id: str):
        return self._paired_training_path

    def get_conditioning_type(self):
        return self._conditioning_type

    def image_path(self, basename_only=False):
        return self._training_path


def _make_stub_data_backend():
    return SimpleNamespace(
        read_image=lambda *_: Image.new("RGB", (8, 8), color=0),
        get_abs_path=lambda path: path,
    )


class CollateFunctionTests(unittest.TestCase):
    def setUp(self):
        self.base_args = SimpleNamespace(
            caption_dropout_probability=0.0,
            controlnet=False,
            model_family="flux",
            vae_cache_ondemand=False,
            data_aesthetic_score=0.0,
            conditioning_multidataset_sampling=None,
            distillation_method=None,
            distillation_config={},
        )
        self.base_batch = [
            {
                "training_samples": [
                    {
                        "image_path": "sample.png",
                        "instance_prompt_text": "caption",
                        "original_size": (64, 64),
                        "image_data": MagicMock(),
                        "crop_coordinates": [0, 0, 32, 32],
                        "data_backend_id": "backend-1",
                        "aspect_ratio": 1.0,
                        "target_size": (32, 32),
                        "intermediary_size": (64, 64),
                    }
                ],
                "conditioning_samples": [],
            }
        ]

    def _patch_state_tracker(self, *, model, data_backend, text_outputs, backend_lookup=None):
        backend_lookup = backend_lookup or {}

        def make_runtime(dataset_id, entry):
            config = entry.get("config", {})
            runtime = DatasetRuntime(
                dataset_id,
                entry.get("data_backend"),
                SimpleNamespace(instance_data_dir=config.get("instance_data_dir")),
                config,
            )
            runtime.text_embed_cache = entry.get("text_embed_cache")
            runtime.vae_cache = entry.get("vaecache")
            runtime.conditioning_image_embed_cache = entry.get("conditioning_image_embed_cache")
            return runtime

        def _get_dataset(dataset_id):
            entry = backend_lookup.get(dataset_id, data_backend)
            runtime = make_runtime(dataset_id, entry)
            runtime.conditioning_datasets = tuple(
                make_runtime(link["id"], backend_lookup.get(link["id"], link)) for link in entry.get("conditioning_data", ())
            )
            return runtime

        backend_patcher = patch.object(StateTracker, "get_dataset", side_effect=_get_dataset)
        config_patcher = patch.object(
            StateTracker,
            "get_data_backend_config",
            return_value={"resolution": 1024, "resolution_type": "pixel", "instance_data_dir": "/train"},
        )
        patchers = [
            patch.object(StateTracker, "get_args", return_value=self.base_args),
            patch.object(StateTracker, "get_model", return_value=model),
            patch.object(StateTracker, "get_model_family", return_value="flux"),
            patch.object(StateTracker, "get_accelerator", return_value=SimpleNamespace(device="cpu")),
            patch.object(StateTracker, "get_weight_dtype", return_value=torch.float32),
            backend_patcher,
            config_patcher,
            patch("simpletuner.helpers.training.collate.compute_latents", return_value=[torch.zeros(1, 4, 4, 4)]),
            patch(
                "simpletuner.helpers.training.collate.check_latent_shapes",
                side_effect=lambda latents, *_, **__: latents,
            ),
            patch(
                "simpletuner.helpers.training.collate.compute_prompt_embeddings",
                return_value=text_outputs,
            ),
            patch("simpletuner.helpers.training.collate.gather_conditional_sdxl_size_features"),
            patch("simpletuner.helpers.training.collate.gather_conditional_pixart_size_features"),
            patch("torch.backends.mps.is_available", return_value=False),
        ]
        return patchers, backend_patcher

    def test_collate_fn_basic_path_uses_backend_for_text_cache(self):
        text_outputs = {
            "prompt_embeds": torch.zeros(1, 1),
            "pooled_prompt_embeds": torch.zeros(1, 1),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }

        model = _StubModel(requires_conditioning=False)
        patchers, backend_patcher = self._patch_state_tracker(  # pylint: disable=unused-variable
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )
        with ExitStack() as stack:
            active_mocks = [stack.enter_context(patcher) for patcher in patchers]
            result = collate_fn(self.base_batch)

        self.assertIn("latent_batch", result)
        self.assertTrue(torch.equal(result["prompt_embeds"], text_outputs["prompt_embeds"]))
        self.assertTrue(torch.equal(result["add_text_embeds"], text_outputs["pooled_prompt_embeds"]))
        backend_mock = active_mocks[5]
        backend_mock.assert_called()

    def test_collate_returns_selected_s2v_audio_reference_and_dataset(self):
        dataset = DatasetRuntime("backend-1", _make_stub_data_backend(), None, {"dataset_type": "video"})
        paired_audio = DatasetRuntime("speech", None, None, {"dataset_type": "audio"})
        dataset.audio_datasets = (paired_audio,)
        model = _StubModel(uses_text_embeddings_cache=False)
        example = self.base_batch[0]["training_samples"][0]
        example.update(dataset_type="video", s2v_audio_path="speech.wav", s2v_audio_backend_id=paired_audio.id)
        patchers, _ = self._patch_state_tracker(model=model, data_backend={}, text_outputs={})

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            output = collate_fn(self.base_batch, dataset=dataset)

        self.assertEqual(output["s2v_audio_paths"], ["speech.wav"])
        self.assertEqual(output["s2v_audio_backend_ids"], ["speech"])

    def test_audio_token_batch_keeps_source_identity_and_sampler_continuation(self):
        from simpletuner.helpers.models.heartmula.gen_config import HeartMuLaGenConfig
        from simpletuner.helpers.models.heartmula.model import HeartMuLa

        model = HeartMuLa.__new__(HeartMuLa)
        model.gen_config = HeartMuLaGenConfig(text_bos_id=1, text_eos_id=2)
        model.tokenizer = MagicMock()
        model.tokenizer.encode.return_value = SimpleNamespace(ids=[3])
        model.model = SimpleNamespace(config=SimpleNamespace(audio_num_codebooks=2))
        model.accelerator = SimpleNamespace(device="cpu")
        dataset = DatasetRuntime("music", None, None, {"dataset_type": "audio"})
        continuation = {"next_batch": 1}
        examples = [
            {
                "image_path": "track.wav",
                "data_backend_id": dataset.id,
                "dataset_continuation": continuation,
                "tags": "instrumental",
                "lyrics": "a quiet verse",
                "audio_tokens": [[2, 3], [4, 5], [6, 7]],
            },
        ]
        batch = [{"training_samples": examples, "conditioning_samples": []}]

        with (
            patch.object(StateTracker, "get_model", return_value=model),
            patch.object(StateTracker, "get_args", return_value=self.base_args),
        ):
            collated = collate_fn(batch, dataset=dataset)
        prepared = model.prepare_batch(collated, state={})

        self.assertEqual(prepared["data_backend_id"], dataset.id)
        self.assertIs(prepared["dataset_continuation"], continuation)
        self.assertEqual(prepared["audio_frame_mask"].sum().item(), 3)

    def test_mixed_s2v_occurrences_use_their_bound_audio_cache(self):
        dataset = DatasetRuntime("backend-1", _make_stub_data_backend(), None, {"dataset_type": "video"})
        metadata = SimpleNamespace(epoch_provider=None, requires_complete_preparation=False)
        audio = DatasetRuntime("speech", None, metadata, {"dataset_type": "audio"})
        audio.vae_cache = object()
        dataset.audio_datasets = (audio,)
        model = _StubModel(uses_text_embeddings_cache=False)
        model.uses_audio_latents_for_data_backend = lambda _: True
        examples = self.base_batch[0]["training_samples"]
        examples[0].update(dataset_type="video", s2v_audio_path="selected.wav")
        examples.append({key: value for key, value in examples[0].items() if key != "s2v_audio_path"})
        examples[1]["image_path"] = "second.mp4"
        patchers, _ = self._patch_state_tracker(model=model, data_backend={}, text_outputs={})

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            resolve = stack.enter_context(patch.object(dataset, "resolve_s2v_audio", return_value=(audio, "resolved.wav")))
            latents = stack.enter_context(
                patch("simpletuner.helpers.training.collate.compute_latents", return_value=[torch.zeros(1, 4, 4, 4)] * 2)
            )
            output = collate_fn(self.base_batch, dataset=dataset)

        resolve.assert_called_once_with("second.mp4")
        latents.assert_called_with(
            ["selected.wav", "resolved.wav"], "speech", model, vae_cache=audio.vae_cache, dataset=audio, view=None
        )
        self.assertEqual(output["s2v_audio_backend_ids"], ["speech", "speech"])

    def test_collate_passes_all_selected_reference_paths_to_text_encoding(self):
        from threading import Thread

        from simpletuner.helpers.caching.text_embeds import TextEmbeddingCache
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.models.flux3_image.model import Flux3Image

        self.base_args.conditioning_multidataset_sampling = "combined"
        references = [
            {
                "id": backend_id,
                "data_backend": _make_stub_data_backend(),
                "config": {
                    "instance_data_dir": f"/refs/{backend_id}",
                    "conditioning_type": "reference_loose" if backend_id == "second" else None,
                },
                "text_embed_cache": SimpleNamespace(disabled=False),
            }
            for backend_id in ("first", "second")
        ]
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        for entry in references:
            metadata = MetadataBackend.__new__(MetadataBackend)
            runtime = DatasetRuntime(
                entry["id"], entry["data_backend"], metadata, {**entry["config"], "caption_strategy": "textfile"}
            )
            runtime.resolve_caption = MagicMock()
            entry["dataset_runtime"] = runtime
        data_backend = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
            "conditioning_data": references,
        }
        model = _StubModel(requires_conditioning_latents=True, requires_text_embed_image_context=True)
        model.requires_text_embed_conditioning_pixels = lambda: False
        image_model = Flux3Image.__new__(Flux3Image)
        image_model.config = SimpleNamespace(flux3_prompt_upsample=True, resolution=512)
        model.text_embed_cache_key = image_model.text_embed_cache_key
        model.text_embed_cache_metadata_for_sample = image_model.text_embed_cache_metadata_for_sample
        self.base_batch[0]["conditioning_samples"] = [
            _StubConditioningSample(
                training_path=f"/refs/{reference['id']}/sample.png",
                backend_id=reference["id"],
                paired_training_path="/train/sample.png",
                conditioning_type=reference["config"]["conditioning_type"],
            )
            for reference in references
        ]
        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=data_backend,
            text_outputs={"prompt_embeds": torch.zeros(1, 1)},
            backend_lookup={"backend-1": data_backend, **{backend["id"]: backend for backend in references}},
        )
        with ExitStack() as stack:
            mocks = [stack.enter_context(patcher) for patcher in patchers]
            stack.enter_context(
                patch("simpletuner.helpers.training.collate.conditioning_pixels", side_effect=AssertionError)
            )
            collate_fn(self.base_batch)
        record = mocks[9].call_args.args[0][0]
        self.assertEqual(record["metadata"]["image_paths"], ["/refs/second/sample.png", "/refs/first/sample.png"])
        self.assertEqual(record["metadata"]["data_backend_ids"], ["second", "first"])

        cache = TextEmbeddingCache.__new__(TextEmbeddingCache)
        cache.batch_write_thread = Thread()
        cache.model = image_model
        cache.model_type = "flux3_image"
        cache.key_type = TextEmbedCacheKey.DATASET_AND_FILENAME
        cache._requires_path_based_keys = True
        cache.caption_filters = ()
        cache.cache_dir = "cache"
        precache = MagicMock()
        registry = FactoryRegistry.__new__(FactoryRegistry)
        registry.model = image_model
        registry.args = self.base_args
        registry.args.caption_strategy = "textfile"
        registry.args.prepend_instance_prompt = False
        registry.args.instance_prompt = ""
        registry.text_encoders = []
        registry.accelerator = SimpleNamespace(device="cpu", process_index=0, num_processes=1)
        init_backend = dict(
            data_backend,
            id="backend-1",
            instance_data_dir="/train",
            text_embed_cache=precache,
            metadata_backend=SimpleNamespace(
                instance_data_dir="/train", get_metadata_by_filepath=lambda _: {"target_size": (32, 32)}
            ),
        )
        from contextlib import nullcontext

        from simpletuner.helpers.metadata.preparation import PreparationView

        metadata = init_backend["metadata_backend"]
        metadata.open_preparation_view = lambda **kwargs: nullcontext(PreparationView(["/train/sample.png"], metadata))
        runtime = DatasetRuntime("backend-1", init_backend["data_backend"], metadata, {"caption_strategy": "textfile"})
        runtime.conditioning_datasets = tuple(ref["dataset_runtime"] for ref in references)
        runtime.text_embed_cache = precache
        runtime.resolve_caption = MagicMock(return_value="caption")
        init_backend["dataset_runtime"] = runtime
        prefetched = []
        precache.read_batch_size = precache.text_encoder_batch_size = 2
        precache.precompute_embeddings.side_effect = lambda records, **kwargs: prefetched.extend(records)
        registry._deferred_text_embed_datasets = [runtime]
        with patch("simpletuner.helpers.data_backend.factory.move_text_encoders"):
            registry._process_deferred_text_embeddings()
        self.assertEqual(cache.hash_prompt_with_path(record), cache.hash_prompt_with_path(prefetched[0]))

        self.base_args.conditioning_multidataset_sampling = "random"
        runtime.resolve_caption.return_value = ["source"]
        reference_captions = {"first": ["first edit", "alternate first edit"], "second": ["second edit"]}
        for entry in references:
            entry["dataset_runtime"].resolve_caption.return_value = reference_captions[entry["id"]]
        registry._deferred_text_embed_datasets = [runtime]
        prefetched.clear()
        with patch("simpletuner.helpers.data_backend.factory.move_text_encoders"):
            registry._process_deferred_text_embeddings()
        precomputed_keys = {cache.hash_prompt_with_path(entry) for entry in prefetched}
        all_samples = self.base_batch[0]["conditioning_samples"]
        selected_keys = set()
        for sample in all_samples:
            for caption in reference_captions[sample.data_backend_id]:
                sample.caption = caption
                self.base_batch[0]["conditioning_samples"] = [sample]
                with ExitStack() as stack:
                    mocks = [stack.enter_context(patcher) for patcher in patchers]
                    collate_fn(self.base_batch)
                selected_keys.add(cache.hash_prompt_with_path(mocks[9].call_args.args[0][0]))
        self.assertEqual(selected_keys, precomputed_keys)
        self.assertEqual(len(selected_keys), 3)

    def test_random_reference_context_follows_each_occurrence_of_the_same_sample(self):
        self.base_args.conditioning_multidataset_sampling = "random"
        cache = SimpleNamespace(disabled=False)
        dataset = {
            "id": "backend-1",
            "config": {"instance_data_dir": "/train"},
            "text_embed_cache": cache,
            "conditioning_data": [{"id": key, "text_embed_cache": cache} for key in ("first", "second")],
        }
        refs = ("/refs/second.png", "/refs/first.png")
        batch = copy.deepcopy(self.base_batch)
        batch[0]["training_samples"].extend(copy.deepcopy(batch[0]["training_samples"]))
        batch[0]["conditioning_samples"] = [
            _StubConditioningSample(
                path, key, caption=key, paired_training_path="sample.png", conditioning_type="reference_loose"
            )
            for path, key in zip(refs, ("second", "first"), strict=True)
        ]
        model = _StubModel(requires_text_embed_image_context=True)
        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=dataset, text_outputs={"prompt_embeds": torch.zeros(2, 1)}
        )
        with ExitStack() as stack:
            mocks = [stack.enter_context(patcher) for patcher in patchers]
            mocks[7].return_value = [torch.zeros(4, 4, 4), torch.zeros(4, 4, 4)]
            stack.enter_context(
                patch(
                    "simpletuner.helpers.training.collate.conditioning_pixels",
                    side_effect=lambda pairs, **_: torch.stack(
                        [
                            torch.full((3, 8, 8), 0.25 if reference.image_path() == refs[0] else 0.75)
                            for reference, _, _ in pairs
                        ]
                    ),
                )
            )
            for dropout, prompts in ((0.0, ["second", "first"]), (1.0, ["", ""])):
                self.base_args.caption_dropout_probability = dropout
                collate_fn(copy.deepcopy(batch))
                requests = mocks[9].call_args.args[0]
                self.assertEqual([record["prompt"] for record in requests], prompts)
                self.assertEqual([record["metadata"]["image_paths"] for record in requests], [[path] for path in refs])
                self.assertEqual(
                    [record["metadata"]["conditioning_pixel_values"][0, 0, 0].item() for record in requests],
                    [0.25, 0.75],
                )

    def test_collate_fn_loads_anyflow_unconditional_embeddings(self):
        self.base_args.distillation_method = "anyflow"
        self.base_args.distillation_config = {"anyflow": {"fuse_guidance_scale": 3.0}}
        positive = {"prompt_embeds": torch.ones(1, 1)}
        unconditional = {"prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }
        model = _StubModel(requires_conditioning=False)
        model.text_embed_cache_metadata_for_sample = lambda **kwargs: {
            "prompt_signature": f"sig:{kwargs['prompt'] or 'empty'}"
        }
        model.text_embed_cache_key_value = lambda prompt, default_key, metadata: metadata["prompt_signature"]
        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=positive,
            backend_lookup={"backend-1": backend_dict},
        )

        with ExitStack() as stack:
            active_mocks = [stack.enter_context(patcher) for patcher in patchers]
            embed_mock = active_mocks[9]
            embed_mock.side_effect = [positive, unconditional]
            result = collate_fn(self.base_batch)

        self.assertTrue(torch.equal(result["prompt_embeds"], positive["prompt_embeds"]))
        self.assertTrue(torch.equal(result["negative_prompt_embeds"], unconditional["prompt_embeds"]))
        self.assertEqual(embed_mock.call_count, 2)
        conditional_requests = embed_mock.call_args_list[0].args[0]
        unconditional_requests = embed_mock.call_args_list[1].args[0]
        self.assertEqual(conditional_requests[0]["metadata"]["prompt_signature"], "sig:caption")
        self.assertEqual(unconditional_requests[0]["metadata"]["prompt_signature"], "sig:empty")
        self.assertIsNot(conditional_requests[0]["metadata"], unconditional_requests[0]["metadata"])
        self.assertEqual(unconditional_requests[0]["prompt"], "")
        self.assertEqual(unconditional_requests[0]["key"], "sig:empty")

    def test_collate_fn_preserves_prompts_without_text_cache(self):
        backend_dict = {
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }

        model = _StubModel(requires_conditioning=False, uses_text_embeddings_cache=False)
        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs={}, backend_lookup={"backend-1": backend_dict}
        )
        with ExitStack() as stack:
            active_mocks = [stack.enter_context(patcher) for patcher in patchers]
            result = collate_fn(self.base_batch)

        self.assertEqual(result["prompts"], ["caption"])
        self.assertEqual(result["text_encoder_output"], {})
        self.assertIsNone(result["prompt_embeds"])
        active_mocks[9].assert_not_called()

    def test_text_conditioning_mask_tracks_selected_captions(self):
        backend_dict = {
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }
        model = _StubModel(uses_text_embeddings_cache=False)
        for caption, dropout, expected in (("caption", 0.0, True), ("", 0.0, False), ("caption", 1.0, False)):
            with self.subTest(caption=caption, dropout=dropout):
                self.base_args.caption_dropout_probability = dropout
                batch = copy.deepcopy(self.base_batch)
                batch[0]["training_samples"][0]["instance_prompt_text"] = caption
                patchers, _ = self._patch_state_tracker(
                    model=model, data_backend=backend_dict, text_outputs={}, backend_lookup={"backend-1": backend_dict}
                )
                with ExitStack() as stack:
                    for patcher in patchers:
                        stack.enter_context(patcher)
                    result = collate_fn(batch)

                torch.testing.assert_close(result["text_conditioning_mask"], torch.tensor([expected]))

    def test_prefetched_legacy_caption_dropout_uses_seed_before_continuation_is_attached(self):
        backend = {"data_backend": _make_stub_data_backend(), "config": {"instance_data_dir": "/train"}}
        patchers, _ = self._patch_state_tracker(
            model=_StubModel(uses_text_embeddings_cache=False), data_backend=backend, text_outputs={}
        )
        self.base_args.caption_dropout_probability = 0.5
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(np.random, "rand", side_effect=AssertionError("used ambient RNG")))
            for seed, expected in ((0, True), (3, False), (0, True)):
                batch = copy.deepcopy(self.base_batch)
                batch[0]["training_samples"][0]["caption_dropout_seed"] = seed
                result = collate_fn(batch)
                torch.testing.assert_close(result["text_conditioning_mask"], torch.tensor([expected]))

    def test_compute_latents_uses_backend_ondemand_mode(self):
        vae_cache = SimpleNamespace(
            vae_cache_ondemand=True,
            encode_images=MagicMock(return_value=["latent"]),
        )

        with (
            patch.object(StateTracker, "get_args", return_value=self.base_args),
            patch.object(StateTracker, "get_dataset", return_value=SimpleNamespace(vae_cache=vae_cache)),
        ):
            latents = compute_latents(["sample.png"], "backend-1", MagicMock())

        self.assertEqual(latents, ["latent"])
        vae_cache.encode_images.assert_called_once_with([None], ["sample.png"], view=None)

    def test_models_without_an_autoencoder_train_on_pixels_regardless_of_family(self):
        dataset = SimpleNamespace(vae_cache=None)
        pixels = torch.zeros(1, 3, 8, 8)
        model = SimpleNamespace(AUTOENCODER_CLASS=None)
        with (
            patch.object(StateTracker, "get_args", return_value=self.base_args),
            patch("simpletuner.helpers.training.collate.pixel_batch", return_value=pixels) as read_pixels,
        ):
            result = compute_latents(["sample.png"], "backend-1", model, dataset=dataset)
        self.assertIs(result, pixels)
        read_pixels.assert_called_once_with(["sample.png"], "backend-1", model, dataset=dataset, view=None)

    def test_pixel_batches_skip_the_latent_cache_check(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1), "pooled_prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }
        model = _StubModel()
        model.AUTOENCODER_CLASS = None
        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            check = stack.enter_context(patch("simpletuner.helpers.training.collate.check_latent_shapes"))
            collate_fn(self.base_batch)
        check.assert_not_called()

    def test_pixel_batch_reads_ordered_payloads_and_geometry_together(self):
        from simpletuner.helpers.metadata.preparation import DatasetReadView

        view = DatasetReadView("revision", epoch=3)
        image = Image.new("RGB", (8, 8))
        examples = [{"image": image, "image_path": path} for path in ("b.png", "a.png", "b.png")]
        dataset = SimpleNamespace(config={"resolution": 8}, read_samples=MagicMock(return_value=examples))
        samples = [SimpleNamespace(prepare=lambda **_: SimpleNamespace(image=torch.zeros(3, 8, 8))) for _ in examples]
        with (
            patch.object(StateTracker, "get_dataset", side_effect=AssertionError("global dataset lookup")),
            patch("simpletuner.helpers.training.collate.TrainingSample", side_effect=samples) as constructor,
        ):
            pixels = pixel_batch(["b.png", "a.png", "b.png"], "backend-1", MagicMock(), dataset=dataset, view=view)
        self.assertEqual(pixels.shape, (3, 3, 8, 8))
        dataset.read_samples.assert_called_once_with(["b.png", "a.png", "b.png"], fields=("image", "metadata"), view=view)
        self.assertEqual([call.kwargs["image_path"] for call in constructor.call_args_list], ["b.png", "a.png", "b.png"])

    def test_invalid_latent_is_deleted_through_the_bound_cache(self):
        from simpletuner.helpers.metadata.preparation import DatasetReadView

        view = DatasetReadView("revision", epoch=3)
        cache = MagicMock()
        cache.generate_vae_cache_filename.return_value = ("artifact.pt", None)
        with patch.object(StateTracker, "get_dataset", side_effect=AssertionError("global dataset lookup")):
            with self.assertRaisesRegex(ValueError, "Deleted cache file artifact.pt"):
                check_latent_shapes(
                    [torch.tensor([float("nan")])], ["sample.png"], "backend-1", [], vae_cache=cache, view=view
                )
        cache.generate_vae_cache_filename.assert_called_once_with("sample.png", view=view)
        cache.cache_data_backend.delete.assert_called_once_with("artifact.pt")

    def test_compute_prompt_embeddings_batches_precomputed_cache_access(self):
        text_cache = MagicMock(text_cache_ondemand=False)
        model = MagicMock()
        model.collate_prompt_embeds.return_value = {"prompt_embeds": torch.ones(2, 1, 1)}
        outputs = [
            {"prompt_embeds": torch.ones(1, 1, 1)},
            {"prompt_embeds": torch.ones(1, 1, 1)},
        ]

        text_cache.read_training_batch.return_value = outputs
        with patch.object(StateTracker, "get_default_text_embed_cache", return_value=text_cache):
            result = compute_prompt_embeddings(["one", "two"], text_cache, model)

        self.assertEqual([record["prompt"] for record in text_cache.read_training_batch.call_args.args[0]], ["one", "two"])
        self.assertIsNone(text_cache.read_training_batch.call_args.kwargs["device"])
        self.assertEqual(result["prompt_embeds"].shape, torch.Size([2, 1, 1]))

    def test_compute_prompt_embeddings_preserves_order_between_online_and_precomputed_stores(self):
        main = MagicMock(text_cache_ondemand=False)
        default = MagicMock(text_cache_ondemand=True, _requires_path_based_keys=False)
        default.accelerator.device = torch.device("cpu")
        main.read_training_batch.return_value = [
            {"prompt_embeds": torch.tensor([[[1.0]]])},
            {"prompt_embeds": torch.tensor([[[3.0]]])},
        ]
        default.read_training_batch.return_value = [{"prompt_embeds": torch.tensor([[[2.0]]])}]
        model = MagicMock()
        model.collate_prompt_embeds.return_value = {}

        with patch.object(StateTracker, "get_default_text_embed_cache", return_value=default):
            result = compute_prompt_embeddings(["one", "", "three"], main, model)

        self.assertEqual(result["prompt_embeds"].flatten().tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(main.read_training_batch.call_args.kwargs["device"], default.accelerator.device)
        self.assertEqual(default.read_training_batch.call_args.kwargs["device"], default.accelerator.device)

    def test_empty_prompt_uses_active_model_dropout_cache_policy(self):
        class FalseyModel:
            def __bool__(self):
                return False

            def use_text_cache_dropout_sentinel(self):
                return False

            def collate_prompt_embeds(self, entries):
                return {}

        dataset_cache = MagicMock()
        default_cache = MagicMock(text_cache_ondemand=False)
        default_cache._requires_path_based_keys = True
        default_cache.model = SimpleNamespace()
        default_cache.read_training_batch.return_value = [{"prompt_embeds": torch.ones(1, 1, 1)}]
        model = FalseyModel()
        prompt_entry = {"prompt": "", "key": "dataset:sample.mp4", "metadata": {"context": "sample"}}

        with patch.object(StateTracker, "get_default_text_embed_cache", return_value=default_cache):
            compute_prompt_embeddings([prompt_entry], dataset_cache, model)

        self.assertEqual(prompt_entry["key"], "dataset:sample.mp4")
        default_cache.read_training_batch.assert_called_once_with([prompt_entry], device=None)

    def test_contextual_dropout_reads_from_the_dataset_artifact_store(self):
        from simpletuner.helpers.models.minimaxh3.model import MiniMaxH3

        model = MiniMaxH3.__new__(MiniMaxH3)
        dataset_cache = MagicMock(text_cache_ondemand=False)
        dataset_cache.read_training_batch.return_value = [
            {"prompt_embeds": torch.ones(1, 1, 1), "text_token_tags": torch.ones(1, 1, dtype=torch.long)}
        ]
        prompt = {
            "prompt": "",
            "key": "dataset:sample.mp4:__caption_dropout__",
            "metadata": {"image_paths": ["reference.png"], "data_backend_ids": ["references"]},
        }

        with patch.object(StateTracker, "get_default_text_embed_cache", side_effect=AssertionError("wrong store")):
            compute_prompt_embeddings([prompt], dataset_cache, model)

        dataset_cache.read_training_batch.assert_called_once_with([prompt], device=None)

    def test_collate_fn_stacks_conditioning_image_embeds(self):
        conditioning_tensor = torch.ones(2, 4)
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=True),
            "conditioning_image_embed_cache": MagicMock(retrieve_from_cache=MagicMock(return_value=conditioning_tensor)),
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/train"},
        }
        model = _StubModel(requires_conditioning=True)

        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            result = collate_fn(self.base_batch)

        cache = backend_dict["conditioning_image_embed_cache"]
        cache.retrieve_from_cache.assert_called_once_with("sample.png", caption=None)
        self.assertIsNotNone(result["conditioning_image_embeds"])

    def test_collate_fn_uses_reference_conditioning_embeds(self):
        conditioning_tensor = torch.ones(2, 4)
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=True),
            "conditioning_image_embed_cache": MagicMock(),
            "conditioning_data": [{"id": "cond-1"}],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        cond_backend = {
            "conditioning_image_embed_cache": MagicMock(retrieve_from_cache=MagicMock(return_value=conditioning_tensor)),
            "data_backend": SimpleNamespace(read_image=lambda *_: Image.new("RGB", (8, 8), color=0)),
            "config": {"instance_data_dir": "/cond"},
        }
        model = _StubModel(requires_conditioning=True, use_reference_embeds=True)
        conditioning_sample = _StubConditioningSample(
            "/cond/sample.png",
            "cond-1",
            caption="reference caption",
            paired_training_path="sample.png",
        )

        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            result = collate_fn(batch)

        cond_cache = cond_backend["conditioning_image_embed_cache"]
        cond_cache.retrieve_from_cache.assert_called_once_with("/cond/sample.png", caption="reference caption")
        self.assertEqual(result["conditioning_captions"], ["reference caption"])

    def test_conditioning_pixels_use_training_sample_path_mapping(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        cond_backend = {
            "id": "cond-1",
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/cond"},
            "conditioning_image_embed_cache": SimpleNamespace(),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "conditioning_data": [cond_backend],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(
            requires_conditioning=False,
            requires_conditioning_latents=True,
            requires_conditioning_dataset=True,
            requires_text_embed_image_context=True,
        )

        conditioning_sample = _StubConditioningSample(
            "/cond/ref.png",
            "cond-1",
            paired_training_path="/train/custom.png",
        )
        batch = copy.deepcopy(self.base_batch)
        batch[0]["training_samples"][0]["image_path"] = "/train/custom.png"
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )

        fake_pixels = torch.stack([torch.zeros(3, 8, 8)], dim=0)

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            with patch(
                "simpletuner.helpers.training.collate.conditioning_pixels",
                return_value=fake_pixels,
            ) as pixels_mock:
                result = collate_fn(batch)

        pixels_mock.assert_called_once()
        (pair,) = pixels_mock.call_args.args[0]
        self.assertEqual(pair[1], "/train/custom.png")
        self.assertEqual(pair[2]["image_path"], "/train/custom.png")
        self.assertIsNotNone(result["conditioning_pixel_values"])
        self.assertIsNotNone(result["conditioning_latents"])

    def test_pixel_alignment_reuses_loaded_mask_and_source_occurrence_crop(self):
        from torchvision.transforms import ToTensor

        from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
        from simpletuner.helpers.training.collate import conditioning_pixels

        image = Image.linear_gradient("L").rotate(90).resize((16, 8)).convert("RGB")
        geometry = {
            "original_size": (16, 8),
            "target_size": (8, 8),
            "intermediary_size": (16, 8),
            "aspect_ratio": 1.0,
            "crop_coordinates": (0, 3),
        }
        config = {"resolution_type": "pixel", "resolution": 8, "crop": True, "crop_style": "random", "crop_aspect": "square"}
        model = SimpleNamespace(get_transforms=lambda **kwargs: ToTensor(), MAXIMUM_CANVAS_SIZE=None)
        dataset = SimpleNamespace(
            id="images", config=config, read_samples=MagicMock(side_effect=AssertionError("source read"))
        )
        mask = TrainingSample(
            image=image,
            data_backend_id="masks",
            image_path="mask.png",
            image_metadata={**geometry, "crop_coordinates": (0, 0)},
            data_backend_config={**config, "crop": False},
            conditioning_type="mask",
            model=model,
        )
        with patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("global config")):
            (pixels,) = conditioning_pixels([(mask, "sample.png", geometry)], dataset=dataset, model=model)
        torch.testing.assert_close(pixels, ToTensor()(image.crop((3, 0, 11, 8))))
        self.assertEqual(mask.image.size, (16, 8))
        self.assertEqual(tuple(mask.image_metadata["crop_coordinates"]), (0, 0))

    def test_collate_fn_handles_slash_prefixed_training_pairs(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        cond_backend = {
            "id": "cond-1",
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/cond"},
            "conditioning_image_embed_cache": SimpleNamespace(),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "conditioning_data": [cond_backend],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(
            requires_conditioning=False,
            requires_conditioning_latents=True,
            requires_conditioning_dataset=True,
            requires_text_embed_image_context=True,
        )

        conditioning_sample = _StubConditioningSample("/cond/ref.png", "cond-1", paired_training_path="/sample.png")
        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )

        fake_pixels = torch.stack([torch.zeros(3, 8, 8)], dim=0)

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(torch.Tensor, "pin_memory", lambda self: self))
            with patch(
                "simpletuner.helpers.training.collate.conditioning_pixels",
                return_value=fake_pixels,
            ):
                result = collate_fn(batch)

        self.assertIsNotNone(result["conditioning_pixel_values"])
        self.assertIsNotNone(result["conditioning_latents"])

    def test_flow_dpo_collects_reference_latents_without_model_requirement(self):
        text_outputs = {"prompt_embeds": torch.zeros(1, 1)}
        cond_backend = {
            "id": "cond-1",
            "data_backend": _make_stub_data_backend(),
            "config": {"instance_data_dir": "/cond"},
            "conditioning_image_embed_cache": SimpleNamespace(),
        }
        backend_dict = {
            "text_embed_cache": SimpleNamespace(disabled=False),
            "conditioning_data": [cond_backend],
            "config": {"instance_data_dir": "/train"},
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(
            requires_conditioning=False,
            requires_conditioning_latents=False,
            requires_conditioning_dataset=False,
        )
        conditioning_sample = _StubConditioningSample(
            "/cond/ref.png",
            "cond-1",
            paired_training_path="sample.png",
            conditioning_type="reference_strict",
        )
        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [conditioning_sample]

        patchers, _ = self._patch_state_tracker(
            model=model,
            data_backend=backend_dict,
            text_outputs=text_outputs,
            backend_lookup={"backend-1": backend_dict, "cond-1": cond_backend},
        )

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            stack.enter_context(patch.object(StateTracker, "get_distillation_method", return_value="flow_dpo"))
            result = collate_fn(batch)

        self.assertEqual(result["conditioning_type"], "reference_strict")
        self.assertIsNotNone(result["conditioning_latents"])

    def test_describe_missing_conditioning_pairs_reports_backend(self):
        examples = [{"image_path": "sample.png", "data_backend_id": "backend-1"}]
        conditioning_backends = [{"id": "cond-1"}, {"id": "cond-2"}]
        conditioning_examples = [_StubConditioningSample("/train/sample.png", "cond-1")]

        messages = describe_missing_conditioning_pairs(
            examples=examples,
            conditioning_examples=conditioning_examples,
            conditioning_backends=conditioning_backends,
            training_backend_id="backend-1",
            training_root="/train",
        )

        self.assertTrue(messages)
        self.assertIn("cond-2 missing 1 pair(s)", messages[0])
        self.assertIn("sample.png", messages[0])

    def test_collate_fn_includes_missing_conditioning_detail(self):
        text_outputs = {
            "prompt_embeds": torch.zeros(1, 1),
            "pooled_prompt_embeds": torch.zeros(1, 1),
        }
        backend_dict = {
            "config": {"instance_data_dir": "/train"},
            "conditioning_data": [{"id": "cond-1"}, {"id": "cond-2"}],
            "text_embed_cache": SimpleNamespace(disabled=False),
            "data_backend": _make_stub_data_backend(),
        }
        model = _StubModel(requires_conditioning=False)
        patchers, _ = self._patch_state_tracker(
            model=model, data_backend=backend_dict, text_outputs=text_outputs, backend_lookup={"backend-1": backend_dict}
        )

        batch = copy.deepcopy(self.base_batch)
        batch[0]["conditioning_samples"] = [_StubConditioningSample("/train/sample.png", "cond-1")]

        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            with self.assertRaisesRegex(ValueError, "cond-2 missing 1 pair\\(s\\)"):
                collate_fn(batch)


if __name__ == "__main__":
    unittest.main()
