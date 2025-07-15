import os
import torch
import hashlib
import logging
import time
import gc
from tqdm import tqdm
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from helpers.training.wrappers import (
    move_dict_of_tensors_to_device,
    gather_dict_of_tensors_shapes,
)
from helpers.prompts import PromptHandler
from helpers.training.multi_process import rank_info
from queue import Queue
import queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from helpers.training.multi_process import _get_rank as get_rank, should_log
from helpers.webhooks.mixin import WebhookMixin
from helpers.models.common import ModelFoundation

logger = logging.getLogger("TextEmbeddingCache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class TextEmbeddingCache(WebhookMixin):
    prompts = {}

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
            raise ValueError(
                f"TextEmbeddingCache received incorrect data_backend: {data_backend}"
            )
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

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}(id={self.id}) {msg}")

    def create_hash(self, caption):
        if caption is None:
            # It's gross, but some images do not have captions.
            caption = ""
        # Precomputed part of the format string
        hash_format = f"-{self.model_type}"

        # Reuse the hash object
        md5_hash = hashlib.md5()
        md5_hash.update(str(caption).encode())
        # logger.debug(f"Hashing caption: {caption}")
        result = md5_hash.hexdigest() + hash_format
        # logger.debug(f"-> {result}")
        return result

    def hash_prompt_with_path(self, caption):
        return os.path.join(self.cache_dir, self.create_hash(caption) + ".pt")

    def hash_prompt(self, caption):
        return self.create_hash(caption) + ".pt"

    def discover_all_files(self):
        """Identify all files in the data backend."""
        logger.info(
            f"{self.rank_info}(id={self.id}) Listing all text embed cache entries"
        )
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
        self.write_queue.put((embeddings, filename))
        logger.debug(
            f"save_to_cache called for {filename}, write queue has {self.write_queue.qsize()} items, and the write thread's status: {self.batch_write_thread.is_alive()}"
        )

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
                while (
                    not self.write_queue.empty() and len(batch) < self.write_batch_size
                ):
                    logger.debug("Retrieving more items from the queue.")
                    items = self.write_queue.get_nowait()
                    batch.append(items)
                    logger.debug(f"Batch now contains {len(batch)} items.")

                self.process_write_batch(batch)
                self.write_thread_bar.update(len(batch))
                logger.debug("Processed batch write.")
                written_elements += len(batch)

            except queue.Empty:
                # Timeout occurred, no items were ready
                if not self.process_write_batches:
                    if len(batch) > 0:
                        self.process_write_batch(batch)
                        self.write_thread_bar.update(len(batch))
                    logger.debug(
                        f"Exiting batch write thread, no more work to do after writing {written_elements} elements"
                    )
                    break
                logger.debug(
                    f"Queue is empty. Retrieving new entries. Should retrieve? {self.process_write_batches}"
                )
                pass
            except Exception:
                logger.exception("An error occurred while writing embeddings to disk.")
        logger.debug("Exiting background batch write thread.")

    def process_write_batch(self, batch):
        """Write a batch of embeddings to the cache."""
        logger.debug(f"Writing {len(batch)} items to disk")
        # logger.debug(f"Batch: {batch}")
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.data_backend.torch_save, *args) for args in batch
            ]
            for future in futures:
                future.result()  # Wait for all writes to complete
        logger.debug(f"Completed write batch of {len(batch)} items")

    def load_from_cache(self, filename):
        result = self.data_backend.torch_load(filename)
        return result

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
        from diffusers import WanPipeline

        pipe = WanPipeline(
            scheduler=self.pipeline.scheduler,
            vae=self.pipeline.vae,
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer,
            transformer=self.pipeline.transformer,
        )

        prompt_embeds, masks = pipe.encode_prompt(
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
        if self.model.TEXT_ENCODER_CONFIGURATION == {}:
            # This is a model that doesn't use text encoders.
            self.disabled = True
            return None
        logger.debug("Initialising text embed calculator...")
        if not self.batch_write_thread.is_alive():
            logger.debug("Restarting background write thread.")
            # Start the thread again.
            self.process_write_batches = True
            self.batch_write_thread = Thread(target=self.batch_write_embeddings)
            self.batch_write_thread.start()

        existing_cache_filenames = list(
            StateTracker.get_text_cache_files(data_backend_id=self.id).keys()
        )

        # Parallel processing for hashing
        with ThreadPoolExecutor() as executor:
            all_cache_filenames = list(
                executor.map(self.hash_prompt_with_path, all_prompts)
            )

        # Create a set for faster lookups
        existing_cache_filenames_set = set(existing_cache_filenames)

        # Determine which prompts are not cached
        uncached_prompts = [
            prompt
            for prompt, filename in zip(all_prompts, all_cache_filenames)
            if filename not in existing_cache_filenames_set
        ]

        # If all prompts are cached and certain conditions are met, return None
        if not uncached_prompts and not return_concat:
            self.debug_log(
                f"All prompts are cached, ignoring (uncached_prompts={uncached_prompts}, is_validation={is_validation}, return_concat={return_concat})"
            )
            return None
        else:
            self.debug_log(
                f"(uncached_prompts={uncached_prompts}, is_validation={is_validation}, return_concat={return_concat})"
            )

        # Proceed with uncached prompts
        raw_prompts = uncached_prompts if uncached_prompts else all_prompts
        output = None
        if self.model is not None:
            output = self.compute_prompt_embeddings_with_model(
                raw_prompts,
                return_concat=return_concat,
                is_validation=is_validation,
                load_from_cache=load_from_cache,
                is_negative_prompt=is_negative_prompt,
            )
        else:
            raise ValueError(
                f"No such text encoding backend for model type '{self.model_type}'"
            )
        # logger.debug(f"Returning output: {output}")
        return output

    def split_captions_between_processes(self, all_captions: list):
        with self.accelerator.split_between_processes(all_captions) as split:
            split_captions = split
        self.debug_log(
            f"Before splitting, we had {len(all_captions)} captions. After splitting, we have {len(split_captions)} unprocessed files."
        )
        # # Print the first 5 as a debug log:
        self.debug_log(f"Local unprocessed captions: {split_captions[:5]} (truncated)")
        return split_captions

    def compute_prompt_embeddings_with_model(
        self,
        prompts: list = None,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
        is_negative_prompt: bool = False,
    ):
        prompt_embeds_all = []
        should_encode = not load_from_cache
        args = StateTracker.get_args()
        if should_encode:
            local_caption_split = self.split_captions_between_processes(
                prompts or self.prompts
            )
        else:
            local_caption_split = prompts or self.prompts
        if (
            hasattr(args, "cache_clear_validation_prompts")
            and args.cache_clear_validation_prompts
            and is_validation
        ):
            # If --cache_clear_validation_prompts was provided, we will forcibly overwrite them.
            load_from_cache = False
            should_encode = True

        if self.webhook_handler is not None:
            last_reported_index = 0
            self.send_progress_update(
                type="init_cache_text_embeds_started",
                progress=int(0 // len(local_caption_split)),
                total=len(local_caption_split),
                current=0,
            )

        self.write_thread_bar = tqdm(
            desc="Write embeds to disk",
            leave=False,
            ncols=125,
            disable=return_concat,
            total=len(local_caption_split),
            position=get_rank(),
        )
        with torch.no_grad():
            last_reported_index = 0
            text_encoder_output = None
            for prompt in tqdm(
                local_caption_split,
                desc="Processing prompts",
                disable=return_concat,
                miniters=50,
                leave=False,
                ncols=125,
                position=get_rank() + self.accelerator.num_processes + 1,
            ):
                filename = os.path.join(self.cache_dir, self.hash_prompt(prompt))
                debug_msg = f"Processing file: {filename}, prompt: {prompt}"
                prompt = PromptHandler.filter_caption(self.data_backend, prompt)
                debug_msg = f"{debug_msg}\n -> filtered prompt: {prompt}"
                if prompt is None:
                    logger.error(f"Filename {filename} does not have a caption.")
                    continue
                logger.debug(debug_msg)
                if return_concat and load_from_cache:
                    try:
                        # We attempt to load.
                        text_encoder_output = self.load_from_cache(filename)
                        embed_shapes = gather_dict_of_tensors_shapes(
                            tensors=text_encoder_output
                        )
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
                        should_encode = True
                        raise Exception(
                            "Cache retrieval for text embed file failed. Ensure your dataloader config value for skip_file_discovery does not contain 'text', and that preserve_data_backend_cache is disabled or unset."
                        )
                if should_encode:
                    # If load_from_cache is True, should_encode would be False unless we failed to load.
                    self.debug_log(
                        f"Encoding filename {filename} :: device {self.text_encoders[0].device} :: prompt {prompt}"
                    )
                    text_encoder_output = self.model.encode_text_batch(
                        [prompt], is_negative_prompt=is_negative_prompt
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
                        and int(
                            self.write_thread_bar.n % self.webhook_progress_interval
                        )
                        < 10
                    ):
                        last_reported_index = int(
                            self.write_thread_bar.n % self.webhook_progress_interval
                        )
                        self.send_progress_update(
                            type="init_cache_text_embeds_status_update",
                            progress=int(
                                self.write_thread_bar.n
                                // len(local_caption_split)
                                * 100
                            ),
                            total=len(local_caption_split),
                            current=0,
                        )

                    if return_concat:
                        # we're returning the embeds for training, so we'll prepare them
                        text_encoder_output = move_dict_of_tensors_to_device(
                            tensors=text_encoder_output, device=self.accelerator.device
                        )
                    else:
                        # if we're not returning them, we'll just nuke them
                        text_encoder_output = move_dict_of_tensors_to_device(
                            tensors=text_encoder_output, device="meta"
                        )
                        del text_encoder_output
                        continue

                if return_concat:
                    prompt_embeds_all.append(text_encoder_output)

            while self.write_queue.qsize() > 0:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            if self.webhook_handler is not None:
                self.send_progress_update(
                    type="init_cache_text_embeds_status_complete",
                    progress=100,
                    total=len(local_caption_split),
                    current=len(local_caption_split),
                )

            # Close the tqdm progress bar after the loop
            self.write_thread_bar.close()

            if not return_concat:
                del prompt_embeds_all
                return

            logger.debug(f"Returning all {(len(prompt_embeds_all))} prompt embeds")
            if (
                text_encoder_output is not None
                and "prompt_embeds" in text_encoder_output
            ):
                all_prompt_embeds = tuple(
                    [v for v in text_encoder_output["prompt_embeds"]]
                )
                text_encoder_output["prompt_embeds"] = torch.cat(
                    all_prompt_embeds, dim=0
                )
            if (
                text_encoder_output is not None
                and "add_text_embeds" in text_encoder_output
            ):
                all_pooled_embeds = tuple(
                    [v for v in text_encoder_output["add_text_embeds"]]
                )
                text_encoder_output["add_text_embeds"] = torch.cat(
                    all_pooled_embeds, dim=0
                )

        return text_encoder_output

    def __del__(self):
        """Ensure that the batch write thread is properly closed."""
        if self.batch_write_thread.is_alive():
            self.batch_write_thread.join()
