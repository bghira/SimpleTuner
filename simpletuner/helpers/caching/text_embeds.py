import gc
import hashlib
import logging
import os
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Thread
from typing import Any, Dict, List, Optional, Sequence, TypedDict
from weakref import WeakSet

import torch
from tqdm import tqdm

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.models.common import ModelFoundation, PipelineTypes, TextEmbedCacheKey
from simpletuner.helpers.prompts import PromptHandler
from simpletuner.helpers.training.multi_process import rank_info
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.wrappers import gather_dict_of_tensors_shapes, move_dict_of_tensors_to_device
from simpletuner.helpers.webhooks.mixin import WebhookMixin

logger = logging.getLogger("TextEmbeddingCache")
from simpletuner.helpers.training.multi_process import _get_rank as get_rank
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class PromptCacheRecord(TypedDict, total=False):
    prompt: str
    key: str
    metadata: Dict[str, Any]


class TextEmbeddingCache(WebhookMixin):
    prompt_records: List[PromptCacheRecord] = []
    _instances: "WeakSet[TextEmbeddingCache]" = WeakSet()

    def __init__(
        self,
        id: str,
        data_backend: BaseDataBackend,
        text_encoders,
        tokenizers,
        accelerator,
        webhook_progress_interval: int = 100,
        cache_dir: str = "cache",
        model_type: str = None,
        prompt_handler: PromptHandler = None,
        write_batch_size: int = 128,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        text_encoder_batch_size: int = 4,
        max_workers: int = 32,
        model: ModelFoundation = None,
    ):
        self.id = id
        if data_backend.id != id:
            raise ValueError(f"TextEmbeddingCache received incorrect data_backend: {data_backend}")
        self.should_abort = False
        self.data_backend = data_backend
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.model = model
        self.pipeline = None
        self.prompt_handler = prompt_handler
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.write_thread_bar = None
        self.text_encoder_batch_size = text_encoder_batch_size
        self.max_workers = max_workers
        self.rank_info = rank_info()
        self.key_type = model.text_embed_cache_key() if model is not None else TextEmbedCacheKey.CAPTION
        self._requires_path_based_keys = self.key_type in (
            TextEmbedCacheKey.FILENAME,
            TextEmbedCacheKey.DATASET_AND_FILENAME,
        )
        self.prompt_records: List[PromptCacheRecord] = []
        if self.data_backend.type == "local":
            self.cache_dir = os.path.abspath(self.cache_dir)
        self.data_backend.create_directory(self.cache_dir)
        self.write_queue = Queue()
        self.process_write_batches = True
        self.batch_write_thread = Thread(
            target=self.batch_write_embeddings,
            name=f"batch_write_thread_{self.id}",
            daemon=True,
        )
        self.batch_write_thread.start()
        self.webhook_progress_interval = webhook_progress_interval
        self.disabled = False  # whether to skip or not at training time.
        TextEmbeddingCache._instances.add(self)

    @classmethod
    def active_caches(cls) -> list["TextEmbeddingCache"]:
        return [cache for cache in cls._instances if cache is not None]

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}(id={self.id}) {msg}")

    def _num_processes(self) -> int:
        """Return accelerator num_processes as an int with safe fallback."""
        if not self.accelerator:
            return 1
        value = getattr(self.accelerator, "num_processes", 1)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 1

    def _normalize_key_value(self, key_value):
        if key_value is None:
            return ""
        normalized = str(key_value)
        if self.key_type is TextEmbedCacheKey.FILENAME:
            if "://" not in normalized:
                normalized = os.path.normcase(os.path.abspath(os.path.normpath(normalized)))
        elif self.key_type is TextEmbedCacheKey.DATASET_AND_FILENAME:
            # Keys already include dataset identifiers; leave as-is.
            pass
        return normalized

    def create_hash(self, key_value):
        normalized_key = self._normalize_key_value(key_value)
        # Precomputed part of the format string
        hash_format = f"-{self.model_type}"

        # Reuse the hash object
        md5_hash = hashlib.md5()
        md5_hash.update(str(normalized_key).encode())
        # logger.debug(f"Hashing caption: {caption}")
        result = md5_hash.hexdigest() + hash_format
        # logger.debug(f"-> {result}")
        return result

    def _resolve_cache_key_value(self, prompt_record: PromptCacheRecord) -> str:
        # Check if 'key' exists in the record (not just if it's truthy)
        # Empty strings are valid values for dropout captions
        if "key" in prompt_record:
            return prompt_record["key"]

        if self._requires_path_based_keys:
            prompt_identifier = prompt_record.get("prompt")
            raise ValueError(
                "Prompt record is missing 'key' but model requires filename-based text embeddings. "
                f"Record metadata: {prompt_record.get('metadata')} prompt={prompt_identifier}"
            )

        # Check if 'prompt' exists in the record (not just if it's truthy)
        # Empty strings are valid values for dropout captions
        if "prompt" in prompt_record:
            return prompt_record["prompt"]

        raise ValueError("Prompt record is missing both 'key' and 'prompt' values.")

    def hash_prompt_with_path(self, prompt_record: PromptCacheRecord):
        key_value = self._resolve_cache_key_value(prompt_record)
        return os.path.join(self.cache_dir, self.create_hash(key_value) + ".pt")

    def hash_prompt(self, prompt_record: PromptCacheRecord):
        key_value = prompt_record.get("key") or prompt_record.get("prompt")
        return self.create_hash(key_value) + ".pt"

    def _normalize_prompt_records(self, prompts: Optional[Sequence[Any]]) -> List[PromptCacheRecord]:
        normalized: List[PromptCacheRecord] = []
        if prompts is None:
            return normalized
        for entry in prompts:
            if isinstance(entry, dict):
                prompt = entry.get("prompt", "")
                metadata = entry.get("metadata") or {}
                key_value = entry.get("key")
                if not key_value and self._requires_path_based_keys:
                    dataset_id = metadata.get("data_backend_id")
                    dataset_path = metadata.get("dataset_relative_path") or metadata.get("image_path")
                    if dataset_id and dataset_path:
                        key_value = f"{dataset_id}:{dataset_path}"
                    else:
                        logger.warning(
                            f"{self.rank_info}(id={self.id}) Prompt record missing dataset metadata while "
                            "path-based text embeddings are required. "
                            f"metadata={metadata} prompt='{prompt}'"
                        )
                        key_value = prompt
                elif not key_value:
                    key_value = prompt
            else:
                prompt = entry
                key_value = entry
                metadata = {}
            normalized.append({"prompt": prompt, "key": key_value, "metadata": metadata})
        return normalized

    def discover_all_files(self):
        """Identify all files in the data backend."""
        logger.info(f"{self.rank_info}(id={self.id}) Listing all text embed cache entries")
        # This isn't returned, because we merely check if it's stored, or, store it.
        (
            StateTracker.get_text_cache_files(data_backend_id=self.id)
            or StateTracker.set_text_cache_files(
                self.data_backend.list_files(
                    instance_data_dir=self.cache_dir,
                    file_extensions=["pt"],
                ),
                data_backend_id=self.id,
            )
        )
        self.debug_log(" -> done listing all text embed cache entries")

    def save_to_cache(self, filename, embeddings):
        """Add write requests to the queue instead of writing directly."""
        if not self.batch_write_thread.is_alive():
            logger.debug("Restarting background write thread.")
            # Start the thread again.
            self.process_write_batches = True
            self.batch_write_thread = Thread(target=self.batch_write_embeddings)
            self.batch_write_thread.start()
        packed = self.model.pack_text_embeddings_for_cache(embeddings)
        self.write_queue.put((packed, filename))
        logger.debug(
            f"save_to_cache called for {filename}, write queue has {self.write_queue.qsize()} items, and the write thread's status: {self.batch_write_thread.is_alive()}"
        )

    def encode_dropout_caption(self):
        """Encode and cache the dropout (null/empty) caption embedding."""
        # Use a special sentinel key for dropout captions with filename-based cache keys
        # This ensures consistent retrieval during training collation
        key = "__caption_dropout__" if self._requires_path_based_keys else ""
        prompt_record: PromptCacheRecord = {"prompt": "", "key": key, "metadata": {}}
        filename = self.hash_prompt_with_path(prompt_record)
        logger.debug(f"Encoding dropout caption to {filename}")
        dropout_embeds = self.model.encode_dropout_caption()
        self.save_to_cache(filename, dropout_embeds)
        logger.debug(f"Dropout caption encoded and saved to cache")

    def encode_validation_negative_prompt(self, negative_prompt: str):
        """Encode and cache the validation negative prompt embedding."""
        logger.debug(f"Encoding validation negative prompt: {negative_prompt}")
        negative_embeds = self.model.encode_validation_negative_prompt(negative_prompt)

        # Use a special sentinel key for negative prompts with filename-based cache keys
        # This ensures consistent retrieval during validation
        key = f"__validation_negative__{negative_prompt}" if self._requires_path_based_keys else negative_prompt
        prompt_record: PromptCacheRecord = {"prompt": negative_prompt, "key": key, "metadata": {}}
        filename = self.hash_prompt_with_path(prompt_record)

        self.save_to_cache(filename, negative_embeds)
        logger.debug(f"Validation negative prompt encoded and saved to cache")
        return negative_embeds

    def batch_write_embeddings(self):
        """Process write requests in batches."""
        batch = []
        written_elements = 0
        while True:
            try:
                # Block until an item is available or timeout occurs
                first_item = self.write_queue.get(timeout=1)
                batch = [first_item]

                # Try to get more items without blocking
                while not self.write_queue.empty() and len(batch) < self.write_batch_size:
                    logger.debug("Retrieving more items from the queue.")
                    items = self.write_queue.get_nowait()
                    batch.append(items)
                    logger.debug(f"Batch now contains {len(batch)} items.")

                self.process_write_batch(batch)
                if self.write_thread_bar is not None:
                    self.write_thread_bar.update(len(batch))
                logger.debug("Processed batch write.")
                written_elements += len(batch)

            except queue.Empty:
                # Timeout occurred, no items were ready
                if not self.process_write_batches:
                    if len(batch) > 0:
                        self.process_write_batch(batch)
                        if self.write_thread_bar is not None:
                            self.write_thread_bar.update(len(batch))
                    logger.debug(f"Exiting batch write thread, no more work to do after writing {written_elements} elements")
                    break
                # logger.debug(
                #     f"Queue is empty. Retrieving new entries. Should retrieve? {self.process_write_batches}"
                # )
                pass
            except Exception:
                logger.exception("An error occurred while writing embeddings to disk.")
        logger.debug("Exiting background batch write thread.")

    def process_write_batch(self, batch):
        """Write a batch of embeddings to the cache."""
        logger.debug(f"Writing {len(batch)} items to disk")
        # logger.debug(f"Batch: {batch}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.data_backend.torch_save, *args) for args in batch]
            for future in futures:
                future.result()  # Wait for all writes to complete
        logger.debug(f"Completed write batch of {len(batch)} items")

    def load_from_cache(self, filename):
        result = self.data_backend.torch_load(filename)

        # Handle backward compatibility: old cache files stored tuples, new format uses dicts
        if isinstance(result, tuple):
            # Convert old tuple format to new dict format using model's _format_text_embedding
            if hasattr(self.model, "_format_text_embedding"):
                result = self.model._format_text_embedding(result)
            else:
                logger.warning(
                    f"Loaded tuple format from cache but model doesn't have _format_text_embedding method. "
                    f"This cache file may be incompatible: {filename}"
                )
        if self.model is None:
            self.model = StateTracker.get_model()
        return self.model.unpack_text_embeddings_from_cache(result)

    def encode_wan_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        is_validation: bool = False,
    ):
        """
        Encode a prompt for a Wan model.

        Args:
            text_encoders: List of text encoders.
            tokenizers: List of tokenizers.
            prompt: The prompt to encode.
            num_images_per_prompt: The number of images to generate per prompt.
            is_validation: Whether the prompt is for validation.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds).
        """
        if self.model is None:
            raise ValueError("Wan prompt encoding requires an attached model.")
        pipeline = self.model.get_pipeline(
            PipelineTypes.TEXT2IMG,
            load_base_model=False,
            cache_pipeline=False,
        )
        prompt_embeds, masks = pipeline.encode_prompt(
            prompt=prompt,
            device=self.accelerator.device,
            # max_sequence_length=StateTracker.get_args().tokenizer_max_length,
        )

        return prompt_embeds, masks

    def compute_embeddings_for_prompts(
        self,
        all_prompts,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
        is_negative_prompt: bool = False,
    ):
        if self.model is None:
            self.model = StateTracker.get_model()
        if self.model.TEXT_ENCODER_CONFIGURATION == {}:
            # This is a model that doesn't use text encoders.
            self.debug_log(f"Model type {self.model_type} does not use text encoders, skipping text embed caching.")
            self.disabled = True
            return None
        normalized_inputs = self._normalize_prompt_records(all_prompts)
        if normalized_inputs:
            self.prompt_records = normalized_inputs
        prompt_records = self.prompt_records
        if not prompt_records:
            self.debug_log("No prompt records were provided to text embed cache.")
            return None
        self.debug_log("Initialising text embed calculator...")
        if not self.batch_write_thread.is_alive():
            self.debug_log("Restarting background write thread.")
            # Start the thread again.
            self.process_write_batches = True
            self.batch_write_thread = Thread(target=self.batch_write_embeddings)
            self.batch_write_thread.start()

        existing_cache_filenames = list(StateTracker.get_text_cache_files(data_backend_id=self.id).keys())

        all_cache_filenames = [self.hash_prompt_with_path(record) for record in prompt_records]

        # Create a set for faster lookups
        existing_cache_filenames_set = set(existing_cache_filenames)

        # Determine which prompts are not cached
        uncached_records = [
            record
            for record, filename in zip(prompt_records, all_cache_filenames)
            if filename not in existing_cache_filenames_set
        ]

        # If all prompts are cached and certain conditions are met, return None
        if not uncached_records and not return_concat:
            self.debug_log(
                f"All prompts are cached, ignoring (is_validation={is_validation}, return_concat={return_concat}, existing={existing_cache_filenames})"
            )
            return None
        else:
            self.debug_log(
                f"(uncached_prompts={len(uncached_records)}, is_validation={is_validation}, return_concat={return_concat})"
            )

        # Proceed with uncached prompts
        raw_records = uncached_records if uncached_records else prompt_records

        # Only check for metadata if we're going to encode (not just load from cache)
        requires_context = self.model.requires_text_embed_image_context()
        should_encode = not load_from_cache
        if requires_context and uncached_records and should_encode:
            contextless_records = [record for record in raw_records if not record.get("metadata")]
            if contextless_records:
                logger.warning(
                    f"{self.rank_info}(id={self.id}) Prompt records require image context but metadata "
                    f"was missing for {len(contextless_records)} entries. "
                    f"Sample record: {contextless_records[0]}"
                )
                if not return_concat:
                    return None
                raw_records = [record for record in raw_records if record.get("metadata")]
                if not raw_records:
                    return None

        output = None
        if self.model is not None:
            output = self.compute_prompt_embeddings_with_model(
                raw_records,
                return_concat=return_concat,
                is_validation=is_validation,
                load_from_cache=load_from_cache,
                is_negative_prompt=is_negative_prompt,
            )
        else:
            raise ValueError(f"No such text encoding backend for model type '{self.model_type}'")
        # logger.debug(f"Returning output: {output}")
        return output

    def split_prompt_records_between_processes(self, prompt_records: List[PromptCacheRecord]):
        with self.accelerator.split_between_processes(prompt_records) as split:
            split_records = split
        self.debug_log(
            f"Before splitting, we had {len(prompt_records)} prompts. After splitting, we have {len(split_records)} entries."
        )
        # # Print the first 5 as a debug log:
        self.debug_log(f"Local unprocessed prompts: {split_records[:5]} (truncated)")
        return split_records

    def compute_prompt_embeddings_with_model(
        self,
        prompt_records: List[PromptCacheRecord] = None,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
        is_negative_prompt: bool = False,
    ):
        prompt_embeds_all = []
        should_encode = not load_from_cache
        args = StateTracker.get_args()
        records = prompt_records or self.prompt_records
        if should_encode:
            local_records = self.split_prompt_records_between_processes(records)
        else:
            local_records = records

        if self.webhook_handler is not None:
            last_reported_index = 0
            self.send_progress_update(
                type="init_cache_text_embeds_status_update",
                readable_type="Text Embedding Caching in Progress",
                progress=int(0 // len(local_records)),
                total=len(local_records),
                current=0,
            )

        self.write_thread_bar = tqdm(
            desc="Write embeds to disk",
            leave=False,
            ncols=125,
            disable=return_concat or len(local_records) < 100,
            total=len(local_records),
            position=get_rank(),
        )
        current_idx = -1
        with torch.no_grad():
            text_encoder_output = None
            for record in tqdm(
                local_records,
                desc="Processing prompts",
                disable=return_concat,
                miniters=50,
                leave=False,
                ncols=125,
                position=get_rank() + self._num_processes() + 1,
            ):
                current_idx += 1
                filename = self.hash_prompt_with_path(record)
                prompt = record.get("prompt")
                debug_msg = f"Processing file: {filename}, prompt: {prompt}, records: {record}"
                prompt = PromptHandler.filter_caption(self.data_backend, prompt)
                debug_msg = f"{debug_msg}\n -> filtered prompt: {prompt}"
                if prompt is None:
                    logger.error(f"Filename {filename} does not have a caption.")
                    continue
                logger.debug(debug_msg)
                encode_current_prompt = should_encode
                if return_concat and load_from_cache and not encode_current_prompt:
                    try:
                        # We attempt to load.
                        text_encoder_output = self.load_from_cache(filename)
                        embed_shapes = gather_dict_of_tensors_shapes(tensors=text_encoder_output)
                        logger.debug(f"Cached text embeds: {embed_shapes}")
                        logger.debug(
                            f"Filename {filename} prompt embeds: {embed_shapes}, keys: {text_encoder_output.keys()}"
                        )
                    except Exception as e:
                        # We failed to load. Now encode the prompt.
                        logger.error(
                            f"Failed retrieving prompt from cache:"
                            f"\n-> prompt: {prompt}"
                            f"\n-> filename: {filename}"
                            f"\n-> error: {e}"
                            f"\n-> id: {self.id}, data_backend id: {self.data_backend.id}"
                        )
                        raise Exception(
                            "Cache retrieval for text embed file failed. Ensure your dataloader config value for "
                            "skip_file_discovery does not contain 'text', and that preserve_data_backend_cache is "
                            "disabled or unset."
                        )
                if encode_current_prompt:
                    # If load_from_cache is True, should_encode would be False unless we failed to load.
                    self.debug_log(
                        f"Encoding filename {filename} :: device {self.text_encoders[0].device} :: prompt {prompt}"
                    )
                    prompt_contexts = [record.get("metadata") or {}]
                    text_encoder_output = self.model.encode_text_batch(
                        [prompt], is_negative_prompt=is_negative_prompt, prompt_contexts=prompt_contexts
                    )
                    logger.debug(
                        f"Filename {filename} prompt embeds: {gather_dict_of_tensors_shapes(tensors=text_encoder_output)}, keys: {text_encoder_output.keys()}"
                    )
                    # Get the current size of the queue.
                    current_size = self.write_queue.qsize()
                    if current_size >= 2048:
                        log_msg = str(
                            f"[WARNING] Write queue size is {current_size}. This is quite large."
                            " Consider increasing the write batch size. Delaying encode so that writes can catch up."
                        )
                        self.write_thread_bar.write(log_msg)
                        while self.write_queue.qsize() > 100:
                            time.sleep(0.1)
                    self.debug_log(f"Adding embed to write queue: {filename}")
                    self.save_to_cache(filename, text_encoder_output)

                    if (
                        self.webhook_handler is not None
                        and int(self.write_thread_bar.n % self.webhook_progress_interval) < 10
                    ):
                        self.send_progress_update(
                            type="init_cache_text_embeds_status_update",
                            progress=int(current_idx / len(local_records) * 100),
                            total=len(local_records),
                            current=current_idx,
                        )

                    if return_concat:
                        # we're returning the embeds for training, so we'll prepare them
                        text_encoder_output = move_dict_of_tensors_to_device(
                            tensors=text_encoder_output, device=self.accelerator.device
                        )
                    else:
                        # if we're not returning them, we'll just nuke them
                        text_encoder_output = move_dict_of_tensors_to_device(tensors=text_encoder_output, device="meta")
                        del text_encoder_output
                        continue

                if return_concat:
                    prompt_embeds_all.append(text_encoder_output)

            while self.write_queue.qsize() > 0:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            if self.webhook_handler is not None:
                self.send_progress_update(
                    type="init_cache_text_embeds_status_update",
                    progress=100,
                    total=len(local_records),
                    current=len(local_records),
                    readable_type="Text Embedding Caching Complete",
                )

            # Close the tqdm progress bar after the loop
            self.write_thread_bar.close()

            if not return_concat:
                del prompt_embeds_all
                return

            logger.debug(f"Returning all {(len(prompt_embeds_all))} prompt embeds")
            # If only one embedding, return it directly without concatenation
            if len(prompt_embeds_all) == 1:
                return prompt_embeds_all[0]

            # Concatenate multiple embeddings
            if prompt_embeds_all and "prompt_embeds" in prompt_embeds_all[0]:
                all_prompt_embeds = [embed["prompt_embeds"] for embed in prompt_embeds_all]
                text_encoder_output["prompt_embeds"] = torch.cat(all_prompt_embeds, dim=0)
            if prompt_embeds_all and "add_text_embeds" in prompt_embeds_all[0]:
                all_pooled_embeds = [embed["add_text_embeds"] for embed in prompt_embeds_all]
                text_encoder_output["add_text_embeds"] = torch.cat(all_pooled_embeds, dim=0)

        return text_encoder_output

    def __del__(self):
        """Ensure that the batch write thread is properly closed."""
        if self.batch_write_thread.is_alive():
            self.batch_write_thread.join()
