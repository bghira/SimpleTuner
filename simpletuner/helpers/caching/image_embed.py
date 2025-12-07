import logging
import os
from hashlib import sha256
from typing import Any, List, Optional, Tuple

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


logger = logging.getLogger(logging.getLogger("ConditioningImageEmbedCache"))
logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


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

        self.embedder = None

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

    def _ensure_embedder(self):
        if self.embedder is not None:
            return

        provider_factory = getattr(self.model, "get_conditioning_image_embedder", None)
        if not callable(provider_factory):
            raise ValueError(
                "Model does not expose a conditioning image embed provider. "
                "Ensure the active model implements 'get_conditioning_image_embedder'."
            )

        embedder = provider_factory()
        if embedder is None:
            raise ValueError("Model reported support for conditioning image embeddings but did not return a provider.")
        if not hasattr(embedder, "encode") or not callable(embedder.encode):
            raise ValueError("Conditioning image embed provider must implement an 'encode(images)' method.")
        self.embedder = embedder

    def generate_embed_filename(self, filepath: str) -> Tuple[str, str]:
        if filepath.endswith(".pt"):
            return filepath, os.path.basename(filepath)
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        if self.hash_filenames:
            base_filename = sha256(str(base_filename).encode()).hexdigest()
        base_filename = f"{base_filename}.pt"

        subfolders = ""
        if self.instance_data_dir:
            normalized_filepath = os.path.abspath(os.path.dirname(filepath))
            normalized_instance = os.path.abspath(self.instance_data_dir)
            if normalized_filepath.startswith(normalized_instance):
                subfolders = normalized_filepath.replace(normalized_instance, "", 1).lstrip(os.sep)

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

    def _encode_batch(
        self, filepaths: List[str], captions: Optional[List[Optional[str]]] = None
    ) -> Tuple[List[str], List[Any]]:
        self._ensure_embedder()
        valid_paths: List[str] = []
        images: List[Image.Image] = []
        captions_for_batch: Optional[List[Optional[str]]]
        if captions is not None:
            captions_for_batch = []
        else:
            captions_for_batch = None
        for idx, fp in enumerate(filepaths):
            try:
                images.append(self._load_image_for_embedding(fp))
                valid_paths.append(fp)
                if captions_for_batch is not None:
                    caption_value = None
                    if captions is not None and idx < len(captions):
                        caption_value = captions[idx]
                    captions_for_batch.append(caption_value)
            except FileNotFoundError:
                self.debug_log(f"Skipping missing file during conditioning embed generation: {fp}")
            except ValueError as exc:
                self.debug_log(f"Skipping unsupported sample {fp}: {exc}")
        if not images:
            return [], []
        with torch.no_grad():
            encode_kwargs = {}
            if captions_for_batch is not None:
                encode_kwargs["captions"] = captions_for_batch
            embeddings = self.embedder.encode(images, **encode_kwargs)
        if embeddings is None:
            return [], []

        def _detach_tensor(value):
            return value.detach().cpu()

        if isinstance(embeddings, (list, tuple)):
            if len(embeddings) == 0:
                return [], []
            if all(torch.is_tensor(item) for item in embeddings):
                stacked = torch.stack(embeddings, dim=0)
                stacked = _detach_tensor(stacked)
                return valid_paths, [stacked[i] for i in range(stacked.shape[0])]
            if all(isinstance(item, dict) for item in embeddings):
                processed: List[dict] = []
                for entry in embeddings:
                    processed.append(
                        {key: (_detach_tensor(value) if torch.is_tensor(value) else value) for key, value in entry.items()}
                    )
                return valid_paths, processed
            raise ValueError(
                "Conditioning image embed provider returned a sequence with unsupported element types. "
                "Expected tensors or dictionaries."
            )

        if isinstance(embeddings, dict):
            processed_dict = {
                key: (_detach_tensor(value) if torch.is_tensor(value) else value) for key, value in embeddings.items()
            }
            return valid_paths, [processed_dict]

        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)

        if torch.is_tensor(embeddings):
            embeddings = _detach_tensor(embeddings)
            return valid_paths, [embeddings[i] for i in range(embeddings.shape[0])]

        raise ValueError("Conditioning image embed provider returned an unsupported embedding type.")

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

    def process_files(self, filepaths: List[str], captions: Optional[List[Optional[str]]] = None) -> None:
        if not filepaths:
            self.debug_log("process_files called with empty filepaths list")
            return
        self.debug_log(f"process_files processing {len(filepaths)} files in batches of {self.embed_batch_size}")
        for idx in range(0, len(filepaths), self.embed_batch_size):
            batch_paths = filepaths[idx : idx + self.embed_batch_size]
            batch_captions = None
            if captions is not None:
                batch_captions = captions[idx : idx + self.embed_batch_size]
            self.debug_log(f"Processing batch {idx // self.embed_batch_size + 1}: {len(batch_paths)} files")
            valid_paths, embeddings = self._encode_batch(batch_paths, captions=batch_captions)
            self.debug_log(f"Encoded {len(valid_paths)} valid embeddings from batch")
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

    def retrieve_from_cache(self, filepath: str, caption: Optional[str] = None) -> torch.Tensor:
        if filepath not in self.image_path_to_embed_path:
            cache_path, _ = self.generate_embed_filename(filepath)
            self.image_path_to_embed_path[filepath] = cache_path
            self.embed_path_to_image_path[cache_path] = filepath

        cache_path = self.image_path_to_embed_path[filepath]
        if not self.cache_data_backend.exists(cache_path):
            caption_batch = [caption] if caption is not None else None
            self.process_files([filepath], captions=caption_batch)
        return self.cache_data_backend.torch_load(cache_path)
