"""Materialize caption prompts as ordinary image training data."""

import hashlib
import inspect
import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from huggingface_hub import HfApi
from PIL import Image

from simpletuner.helpers.caching.memory import reclaim_memory
from simpletuner.helpers.data_backend.config.validators import validate_caption_data_generator
from simpletuner.helpers.models.generation import base_model_generation_context

logger = logging.getLogger("CaptionImageGenerator")


class CaptionImageGenerator:
    def __init__(self, model, config, records, output_dir):
        self.model = model
        self.config = validate_caption_data_generator(config)
        self.records = list(records)
        if not self.records:
            raise ValueError("data_generator requires at least one caption.")
        recipe = {key: value for key, value in self.config.items() if key not in {"batch_size", "output_dir"}}
        model_config = model.config
        model_sources = {
            key: getattr(model_config, key, None)
            for key in (
                "pretrained_model_name_or_path",
                "pretrained_transformer_model_name_or_path",
                "pretrained_unet_model_name_or_path",
                "pretrained_vae_model_name_or_path",
            )
        }
        revision = getattr(model_config, "revision", None)
        source_identities = {key: self._source_identity(source, revision) for key, source in model_sources.items() if source}
        identity = {
            "recipe": recipe,
            "sources": source_identities,
            "captions": [record.caption_text for record in self.records],
            "model": {
                key: getattr(model_config, key, None)
                for key in (
                    "pretrained_model_name_or_path",
                    "pretrained_transformer_model_name_or_path",
                    "pretrained_unet_model_name_or_path",
                    "revision",
                    "variant",
                    "model_family",
                    "model_flavour",
                )
            },
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.output_dir = Path(output_dir) / fingerprint[:24]
        self.manifest_path = self.output_dir / "generation.json"
        self.manifest = {"version": 1, "fingerprint": fingerprint, "recipe": recipe, "batch_sizes": {}}
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest["fingerprint"] != fingerprint:
                raise ValueError("Generated caption cache fingerprint does not match its recipe.")

    @staticmethod
    def _source_identity(source, revision):
        path = Path(source)
        if path.exists():
            files = (
                [path]
                if path.is_file()
                else sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file() and item.suffix in {".safetensors", ".bin", ".pt", ".json"}
                )
            )
            return [
                (item.name if path.is_file() else str(item.relative_to(path)), item.stat().st_size, item.stat().st_mtime_ns)
                for item in files
            ]
        return HfApi().model_info(source, revision=revision).sha

    @contextmanager
    def _pipeline(self):
        model = self.model
        loaded_here = model.model is None
        try:
            if loaded_here:
                model.load_model(move_to_device=True)
            pipeline = model.get_pipeline(load_base_model=False)
            with base_model_generation_context(model):
                yield pipeline
        finally:
            if loaded_here and model.model is not None:
                component = model.unwrap_model(model=model.model)
                for pipeline in model.pipelines.values():
                    if getattr(pipeline, model.MODEL_TYPE.value, None) is component:
                        setattr(pipeline, model.MODEL_TYPE.value, None)
                model.model = None
                del component
                reclaim_memory()

    def _save_manifest(self):
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.manifest_path)

    def _sample(self, bucket_index, record_index):
        resolution = self.config["resolutions"][bucket_index]
        stem = self.output_dir / resolution / f"{record_index:08d}"
        seed = (self.config["seed"] + bucket_index * len(self.records) + record_index) % (2**63)
        return self.records[record_index], stem, seed

    @staticmethod
    def _is_complete(stem):
        return stem.with_suffix(".png").is_file() and stem.with_suffix(".txt").is_file()

    def _generate_batch(self, pipeline, samples, width, height):
        kwargs = {
            "prompt": [record.caption_text for record, _, _ in samples],
            "width": width,
            "height": height,
            "num_inference_steps": self.config["num_inference_steps"],
            "generator": [torch.Generator(device="cpu").manual_seed(seed) for _, _, seed in samples],
            "output_type": "pil",
        }
        kwargs = self.model.update_pipeline_call_kwargs(kwargs)
        parameters = inspect.signature(pipeline.__call__).parameters
        guidance = self.config["guidance_scale"]
        if "true_cfg_scale" in parameters:
            kwargs["true_cfg_scale"] = guidance
            kwargs["negative_prompt"] = [""] * len(samples) if guidance > 1 else None
        if "guidance_scale" in parameters:
            kwargs["guidance_scale"] = guidance
        images = pipeline(**kwargs).images
        if len(images) != len(samples):
            raise ValueError("Caption generation returned an unexpected image count.")
        if any(not isinstance(image, Image.Image) or image.size != (width, height) for image in images):
            raise ValueError("Caption generation must return PIL images matching the requested resolution.")
        return images

    def generate(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pending = {}
        for bucket_index, resolution in enumerate(self.config["resolutions"]):
            (self.output_dir / resolution).mkdir(exist_ok=True)
            pending[bucket_index] = [
                index for index in range(len(self.records)) if not self._is_complete(self._sample(bucket_index, index)[1])
            ]
        if not any(pending.values()):
            return self.bucket_directories()

        with self._pipeline() as pipeline:
            for bucket_index, record_indices in pending.items():
                resolution = self.config["resolutions"][bucket_index]
                width, height = map(int, resolution.split("x"))
                batch_size = min(
                    self.config["batch_size"], self.manifest["batch_sizes"].get(resolution, self.config["batch_size"])
                )
                cursor = 0
                while cursor < len(record_indices):
                    indices = record_indices[cursor : cursor + batch_size]
                    samples = [self._sample(bucket_index, index) for index in indices]
                    oom = False
                    try:
                        images = self._generate_batch(pipeline, samples, width, height)
                    except torch.OutOfMemoryError:
                        if len(samples) == 1:
                            raise
                        oom = True
                    except RuntimeError as error:
                        if not str(error).startswith("MPS backend out of memory") or len(samples) == 1:
                            raise
                        oom = True
                    if oom:
                        batch_size = max(1, len(samples) // 2)
                        self.manifest["batch_sizes"][resolution] = batch_size
                        self._save_manifest()
                        reclaim_memory()
                        logger.warning(
                            "Caption generation OOM at %s; retrying the same samples with batch size %s.",
                            resolution,
                            batch_size,
                        )
                        continue

                    for image, (record, stem, _) in zip(images, samples):
                        image_tmp = stem.with_suffix(".png.tmp")
                        caption_tmp = stem.with_suffix(".txt.tmp")
                        image.save(image_tmp, format="PNG")
                        caption_tmp.write_text(record.caption_text, encoding="utf-8")
                        os.replace(image_tmp, stem.with_suffix(".png"))
                        os.replace(caption_tmp, stem.with_suffix(".txt"))
                    self.manifest["batch_sizes"][resolution] = batch_size
                    self._save_manifest()
                    cursor += len(samples)
        return self.bucket_directories()

    def bucket_directories(self):
        return {resolution: str(self.output_dir / resolution) for resolution in self.config["resolutions"]}


def generated_image_backend_configs(backend, bucket_directories):
    """Reuse normal image preprocessing with each generated bucket's native dimensions."""
    result = []
    for resolution, directory in bucket_directories.items():
        width, height = map(int, resolution.split("x"))
        generated = dict(backend)
        for key in (
            "data_generator",
            "parquet",
            "huggingface",
            "webshart",
            "dataset_name",
            "dataset_config",
            "data_files",
            "caption_column",
            "fallback_caption_column",
            "aws_data_prefix",
            "maximum_image_size",
            "target_downsample_size",
            "crop_aspect_buckets",
            "cache_file_suffix",
            "instance_data_root",
        ):
            generated.pop(key, None)
        generated.update(
            id=f"{backend['id']}-generated-{resolution}",
            type="local",
            dataset_type="image",
            instance_data_dir=directory,
            caption_strategy="textfile",
            metadata_backend="discovery",
            resolution=min(width, height),
            resolution_type="pixel",
            crop=False,
            disable_multiline_split=True,
            probability=float(backend.get("probability", 1.0)) / len(bucket_directories),
        )
        for cache_key in ("cache_dir_vae", "cache_dir_text"):
            if generated.get(cache_key):
                generated[cache_key] = os.path.join(generated[cache_key], resolution)
        result.append(generated)
    return result
