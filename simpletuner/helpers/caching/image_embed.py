import logging
import os
from hashlib import sha256
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

from simpletuner.helpers.training import image_file_extensions
from simpletuner.helpers.training.multi_process import rank_info, should_log
from simpletuner.helpers.training.state_tracker import StateTracker

try:
    from simpletuner.helpers.webhooks.mixin import WebhookMixin
except Exception:  # pragma: no cover - optional dependency guard

    class WebhookMixin:  # type: ignore
        """Fallback mixin used when webhook dependencies are unavailable."""

        def set_webhook_handler(self, webhook_handler):
            self.webhook_handler = webhook_handler


logger = logging.getLogger("ConditioningImageEmbedCache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class ImageEmbedCache(WebhookMixin):
    def __init__(
        self,
        id: str,
        dataset_type: str,
        model,
        accelerator,
        metadata_backend,
        image_data_backend,
        cache_data_backend=None,
        instance_data_dir: str = "",
        cache_dir: str = "",
        write_batch_size: int = 16,
        read_batch_size: int = 16,
        embed_batch_size: int = 4,
        hash_filenames: bool = True,
    ):
        self.id = id
        self.dataset_type = dataset_type
        self.model = model
        self.accelerator = accelerator
        self.metadata_backend = metadata_backend
        self.image_data_backend = image_data_backend
        self.cache_data_backend = cache_data_backend if cache_data_backend is not None else image_data_backend
        self.instance_data_dir = instance_data_dir or ""
        self.cache_dir = cache_dir or ""
        if self.cache_data_backend and self.cache_data_backend.type in ["local", "huggingface"] and self.cache_dir:
            self.cache_dir = os.path.abspath(self.cache_dir)
            self.cache_data_backend.create_directory(self.cache_dir)
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.embed_batch_size = self._ensure_positive_batch_size(embed_batch_size, default=1)
        self.hash_filenames = hash_filenames
        self.rank_info = rank_info()

        self.webhook_handler = None

        self.image_path_to_embed_path: dict[str, str] = {}
        self.embed_path_to_image_path: dict[str, str] = {}

        self.pipeline = None
        self.image_encoder = None
        self.image_processor = None
        self.compute_device = (
            getattr(self.accelerator, "device", torch.device("cpu")) if self.accelerator else torch.device("cpu")
        )

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}{msg}")

    def set_webhook_handler(self, webhook_handler):
        self.webhook_handler = webhook_handler

    @staticmethod
    def _ensure_positive_batch_size(value, default: int) -> int:
        try:
            batch_size = int(value)
        except (TypeError, ValueError):
            return default
        return batch_size if batch_size > 0 else default

    def _ensure_model_components(self):
        if self.image_encoder is not None and self.image_processor is not None:
            return

        pipeline = self.model.get_pipeline()
        if getattr(pipeline, "image_encoder", None) is None or getattr(pipeline, "image_processor", None) is None:
            raise ValueError("Pipeline does not provide image encoder components required for conditioning embeddings.")
        self.pipeline = pipeline
        self.image_encoder = pipeline.image_encoder
        self.image_processor = pipeline.image_processor
        self.image_encoder.to(self.compute_device)
        self.image_encoder.eval()

    def generate_embed_filename(self, filepath: str) -> Tuple[str, str]:
        if filepath.endswith(".pt"):
            return filepath, os.path.basename(filepath)
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        if self.hash_filenames:
            base_filename = sha256(str(base_filename).encode()).hexdigest()
        base_filename = f"{base_filename}.pt"

        subfolders = ""
        if self.instance_data_dir:
            subfolders = os.path.dirname(filepath).replace(self.instance_data_dir, "", 1)
            subfolders = subfolders.lstrip(os.sep)

        if subfolders:
            full_filename = os.path.join(self.cache_dir, subfolders, base_filename)
        else:
            full_filename = os.path.join(self.cache_dir, base_filename)
        return full_filename, base_filename

    def build_embed_filename_map(self, all_image_files: List[str]) -> None:
        self.image_path_to_embed_path.clear()
        self.embed_path_to_image_path.clear()

        for image_file in all_image_files:
            cache_filename, _ = self.generate_embed_filename(image_file)
            if self.cache_data_backend.type == "local":
                cache_filename = os.path.abspath(cache_filename)
            self.image_path_to_embed_path[image_file] = cache_filename
            self.embed_path_to_image_path[cache_filename] = image_file

    def discover_all_files(self) -> List[str]:
        all_image_files = StateTracker.get_image_files(data_backend_id=self.id) or StateTracker.set_image_files(
            self.image_data_backend.list_files(
                instance_data_dir=self.instance_data_dir,
                file_extensions=image_file_extensions,
            ),
            data_backend_id=self.id,
        )
        StateTracker.get_conditioning_image_embed_files(self.id) or StateTracker.set_conditioning_image_embed_files(
            self.cache_data_backend.list_files(
                instance_data_dir=self.cache_dir,
                file_extensions=["pt"],
            ),
            data_backend_id=self.id,
        )
        self.debug_log(f"ConditioningImageEmbedCache discover_all_files found {len(all_image_files)} sources")
        return all_image_files

    def discover_unprocessed_files(self) -> List[str]:
        if not self.image_path_to_embed_path:
            return []

        pending = []
        for image_path, embed_path in self.image_path_to_embed_path.items():
            test_path = embed_path
            if self.cache_data_backend.type == "local":
                test_path = os.path.abspath(embed_path)
            if not self.cache_data_backend.exists(test_path):
                pending.append(image_path)

        return pending

    def _load_image_for_embedding(self, filepath: str) -> Image.Image:
        sample = self.image_data_backend.read_image(filepath)
        if isinstance(sample, Image.Image):
            return sample.convert("RGB")
        if isinstance(sample, np.ndarray):
            if sample.ndim == 4:
                first_frame = sample[0]
            elif sample.ndim == 3:
                first_frame = sample
            else:
                raise ValueError(f"Unsupported numpy shape for conditioning embed: {sample.shape}")
            if first_frame.dtype != np.uint8:
                first_frame = np.clip(first_frame, 0, 255).astype(np.uint8)
            return Image.fromarray(first_frame).convert("RGB")
        raise ValueError(f"Unsupported sample type for conditioning embed: {type(sample)}")

    def _encode_batch(self, filepaths: List[str]) -> Tuple[List[str], List[torch.Tensor]]:
        self._ensure_model_components()
        valid_paths: List[str] = []
        images: List[Image.Image] = []
        for fp in filepaths:
            try:
                images.append(self._load_image_for_embedding(fp))
                valid_paths.append(fp)
            except FileNotFoundError:
                self.debug_log(f"Skipping missing file during conditioning embed generation: {fp}")
            except ValueError as exc:
                self.debug_log(f"Skipping unsupported sample {fp}: {exc}")
        if not images:
            return [], []
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.compute_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.image_encoder(**inputs, output_hidden_states=True)
        embeds = outputs.hidden_states[-2].detach().cpu()
        return valid_paths, list(embeds)

    def _write_embed(self, filepath: str, embedding: torch.Tensor) -> None:
        cache_path = self.image_path_to_embed_path.get(filepath)
        if cache_path is None:
            cache_path, _ = self.generate_embed_filename(filepath)
            self.image_path_to_embed_path[filepath] = cache_path
            self.embed_path_to_image_path[cache_path] = filepath

        directory = os.path.dirname(cache_path)
        if directory and self.cache_data_backend.type == "local":
            os.makedirs(directory, exist_ok=True)

        self.cache_data_backend.torch_save(embedding, cache_path)

        current_cache = StateTracker.get_conditioning_image_embed_files(self.id)
        if isinstance(current_cache, dict):
            current_cache[cache_path] = True

    def process_files(self, filepaths: List[str]) -> None:
        if not filepaths:
            return
        for idx in range(0, len(filepaths), self.embed_batch_size):
            batch_paths = filepaths[idx : idx + self.embed_batch_size]
            valid_paths, embeddings = self._encode_batch(batch_paths)
            for fp, embed in zip(valid_paths, embeddings):
                self._write_embed(fp, embed)
        if self.cache_dir:
            StateTracker.set_conditioning_image_embed_files(
                self.cache_data_backend.list_files(
                    instance_data_dir=self.cache_dir,
                    file_extensions=["pt"],
                ),
                data_backend_id=self.id,
            )

    def retrieve_from_cache(self, filepath: str) -> torch.Tensor:
        if filepath not in self.image_path_to_embed_path:
            cache_path, _ = self.generate_embed_filename(filepath)
            self.image_path_to_embed_path[filepath] = cache_path
            self.embed_path_to_image_path[cache_path] = filepath

        cache_path = self.image_path_to_embed_path[filepath]
        if not self.cache_data_backend.exists(cache_path):
            self.process_files([filepath])
        return self.cache_data_backend.torch_load(cache_path)
