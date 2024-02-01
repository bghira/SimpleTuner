import os, torch, hashlib, logging, time
from tqdm import tqdm
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from helpers.prompts import PromptHandler
from helpers.training.multi_process import rank_info
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("TextEmbeddingCache")
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class TextEmbeddingCache:
    prompts = {}

    def __init__(
        self,
        id: str,
        data_backend: BaseDataBackend,
        text_encoders,
        tokenizers,
        accelerator,
        cache_dir: str = "cache",
        model_type: str = "sdxl",
        prompt_handler: PromptHandler = None,
        write_batch_size: int = 128,
        read_batch_size: int = 25,
        process_queue_size: int = 16,
        text_encoder_batch_size: int = 4,
        max_workers: int = 32,
    ):
        self.id = id
        if data_backend.id != id:
            raise ValueError(
                f"TextEmbeddingCache received incorrect data_backend: {data_backend}"
            )
        self.data_backend = data_backend
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.prompt_handler = prompt_handler
        self.write_batch_size = write_batch_size
        self.read_batch_size = read_batch_size
        self.process_queue_size = process_queue_size
        self.text_encoder_batch_size = text_encoder_batch_size
        self.max_workers = max_workers
        self.rank_info = rank_info()
        self.data_backend.create_directory(self.cache_dir)
        self.write_queue = Queue()
        self.process_write_batches = True
        self.batch_write_thread = Thread(
            target=self.batch_write_embeddings,
            name=f"batch_write_thread_{self.id}",
            daemon=True,
        )
        self.batch_write_thread.start()
        # Ensure the cache has the currently-existing file list.
        self.discover_all_files()

    def debug_log(self, msg: str):
        logger.debug(f"{self.rank_info}(id={self.id}) {msg}")

    def create_hash(self, caption):
        # Precomputed part of the format string
        hash_format = f"-{self.model_type}"

        # Reuse the hash object
        md5_hash = hashlib.md5()
        md5_hash.update(caption.encode())
        return md5_hash.hexdigest() + hash_format

    def hash_prompt(self, caption):
        return self.create_hash(caption) + ".pt"

    def discover_all_files(self):
        """Identify all files in the data backend."""
        logger.info(f"(id={self.id}) Listing all text embed cache entries")
        # This isn't returned, because we merely check if it's stored, or, store it.
        (
            StateTracker.get_text_cache_files(data_backend_id=self.id)
            or StateTracker.set_text_cache_files(
                self.data_backend.list_files(
                    instance_data_root=self.cache_dir,
                    str_pattern="*.pt",
                ),
                data_backend_id=self.id,
            )
        )
        self.debug_log(" -> done listing all text embed cache entries")

    def save_to_cache(self, filename, embeddings):
        """Add write requests to the queue instead of writing directly."""
        self.write_queue.put((embeddings, filename))

    def batch_write_embeddings(self):
        """Process write requests in batches."""
        while True:
            batch = []
            while not self.write_queue.empty() and len(batch) < self.write_batch_size:
                items = self.write_queue.get()
                logger.debug(
                    f"Data backend: {self.data_backend.id}, Adding item to batch: {items[1]}"
                )
                batch.append(items)

            if len(batch) >= self.write_batch_size:
                self.process_write_batch(batch)
            elif self.write_queue.empty() and len(batch) > 0:
                self.process_write_batch(batch)

            check_interval = 5
            # How many items exist in the queue should influence the interval. If the queue is empty, we can wait longer.
            if self.write_queue.qsize() > 0:
                check_interval = 0.001
            time.sleep(
                float(
                    os.environ.get(
                        "SIMPLETUNER_BATCH_WRITE_SLEEP_INTERVAL", check_interval
                    )
                )
            )  # Prevents the thread from being too busy-waiting

    def process_write_batch(self, batch):
        """Write a batch of embeddings to the cache."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.data_backend.torch_save, *args) for args in batch
            ]
            for future in futures:
                future.result()  # Wait for all writes to complete

    def load_from_cache(self, filename):
        result = self.data_backend.torch_load(filename)
        return result

    def encode_legacy_prompt(self, text_encoder, tokenizer, prompt):
        input_tokens = tokenizer(
            PromptHandler.filter_caption(self.data_backend, prompt),
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(text_encoder.device)
        output = text_encoder(input_tokens)[0]
        self.debug_log(f"Legacy prompt shape: {output.shape}")
        self.debug_log(f"Legacy prompt encoded: {output}")
        return output

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_sdxl_prompt
    def encode_sdxl_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt,
        is_validation: bool = False,
    ):
        prompt_embeds_list = []

        emitted_warning = False
        # If prompt_handler (Compel) is available, use it for all prompts
        if self.prompt_handler and is_validation:
            positive_prompt = self.prompt_handler.process_long_prompt(prompt)
            return positive_prompt
        try:
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt, padding="max_length", truncation=True, return_tensors="pt"
                )
                text_input_ids = text_inputs.input_ids

                untruncated_ids = tokenizer(
                    prompt, padding="longest", return_tensors="pt"
                ).input_ids

                if untruncated_ids.shape[
                    -1
                ] > tokenizer.model_max_length and not torch.equal(
                    text_input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    if not emitted_warning:
                        # Only print this once. It's a bit spammy otherwise.
                        emitted_warning = True
                        logger.warning(
                            f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                        )

                prompt_embeds_output = text_encoder(
                    text_input_ids.to(text_encoder.device), output_hidden_states=True
                )
                # We are always interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds_output[0]
                prompt_embeds = prompt_embeds_output.hidden_states[-2]
                del prompt_embeds_output
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

                # Clear out anything we moved to the text encoder device
                text_input_ids.to("cpu")
                del text_input_ids

                prompt_embeds_list.append(prompt_embeds)
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to encode prompt: {prompt}\n-> error: {e}\n-> traceback: {traceback.format_exc()}"
            )
            raise e

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
    def encode_sdxl_prompts(
        self,
        text_encoders,
        tokenizers,
        prompts,
        is_validation: bool = False,
    ):
        prompt_embeds_all = []
        pooled_prompt_embeds_all = []

        for prompt in prompts:
            prompt_embeds, pooled_prompt_embeds = self.encode_sdxl_prompt(
                text_encoders, tokenizers, prompt, is_validation
            )
            prompt_embeds_all.append(prompt_embeds)
            pooled_prompt_embeds_all.append(pooled_prompt_embeds)

        return torch.stack(prompt_embeds_all).squeeze(dim=1), torch.stack(
            pooled_prompt_embeds_all
        ).squeeze(dim=1)

    def encode_prompt(self, prompt: str, is_validation: bool = False):
        if self.model_type == "sdxl":
            return self.encode_sdxl_prompt(
                self.text_encoders, self.tokenizers, prompt, is_validation
            )
        else:
            return self.encode_legacy_prompt(
                self.text_encoders[0], self.tokenizers[0], prompt
            )

    def compute_embeddings_for_prompts(
        self,
        all_prompts,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
    ):
        if not self.batch_write_thread.is_alive():
            # Start the thread again.
            self.process_write_batches = True
            self.batch_write_thread = Thread(target=self.batch_write_embeddings)
            self.batch_write_thread.start()

        existing_cache_filenames = list(
            StateTracker.get_text_cache_files(data_backend_id=self.id).keys()
        )

        # Parallel processing for hashing
        with ThreadPoolExecutor() as executor:
            all_cache_filenames = list(executor.map(self.hash_prompt, all_prompts))

        # Create a set for faster lookups
        existing_cache_filenames_set = set(existing_cache_filenames)

        # Determine which prompts are not cached
        uncached_prompts = [
            prompt
            for prompt, filename in zip(all_prompts, all_cache_filenames)
            if filename not in existing_cache_filenames_set
        ]

        # If all prompts are cached and certain conditions are met, return None
        if not uncached_prompts and not is_validation and not return_concat:
            logger.debug(f"(id={self.id}) All prompts are cached, ignoring.")
            return None

        # Proceed with uncached prompts
        raw_prompts = uncached_prompts if uncached_prompts else all_prompts
        if self.model_type == "sdxl":
            return self.compute_embeddings_for_sdxl_prompts(
                raw_prompts,
                return_concat=return_concat,
                is_validation=is_validation,
                load_from_cache=load_from_cache,
            )
        elif self.model_type == "legacy":
            return self.compute_embeddings_for_legacy_prompts(
                raw_prompts,
                return_concat=return_concat,
                load_from_cache=load_from_cache,
            )

    def compute_embeddings_for_sdxl_prompts(
        self,
        prompts: list = None,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
    ):
        prompt_embeds_all = []
        add_text_embeds_all = []
        should_encode = not load_from_cache
        args = StateTracker.get_args()
        if (
            hasattr(args, "cache_clear_validation_prompts")
            and args.cache_clear_validation_prompts
            and is_validation
        ):
            # If --cache_clear_validation_prompts was provided, we will forcibly overwrite them.
            load_from_cache = False
            should_encode = True
        logger.debug(
            f"compute_embeddings_for_sdxl_prompts received list of prompts: {list(prompts)[:5]}"
        )
        with torch.no_grad():
            for prompt in tqdm(
                prompts or self.prompts,
                desc="Processing prompts",
                disable=return_concat,
                leave=False,
                ncols=125,
            ):
                filename = os.path.join(
                    self.cache_dir, self.create_hash(prompt) + ".pt"
                )
                prompt = PromptHandler.filter_caption(self.data_backend, prompt)
                if return_concat and load_from_cache:
                    try:
                        # We attempt to load.
                        prompt_embeds, add_text_embeds = self.load_from_cache(filename)
                    except Exception as e:
                        # We failed to load. Now encode the prompt.
                        logger.warning(
                            f"Failed retrieving prompt from cache:"
                            f"\n-> prompt: {prompt}"
                            f"\n-> filename: {filename}"
                            f"\n-> error: {e}"
                        )
                        should_encode = True
                        raise Exception("This won't work. We cannot continue.")
                if should_encode:
                    # If load_from_cache is True, should_encode would be False unless we failed to load.
                    self.debug_log(f"Encoding prompt: {prompt}")
                    prompt_embeds, pooled_prompt_embeds = self.encode_sdxl_prompts(
                        self.text_encoders,
                        self.tokenizers,
                        [prompt],
                        is_validation,
                    )
                    add_text_embeds = pooled_prompt_embeds
                    self.debug_log(f"Adding embed to write queue: {filename}")
                    self.save_to_cache(filename, (prompt_embeds, add_text_embeds))
                    if return_concat:
                        prompt_embeds = prompt_embeds.to(self.accelerator.device)
                        add_text_embeds = add_text_embeds.to(self.accelerator.device)
                    else:
                        del prompt_embeds
                        del add_text_embeds
                        del pooled_prompt_embeds
                        continue

                if return_concat:
                    prompt_embeds_all.append(prompt_embeds)
                    add_text_embeds_all.append(add_text_embeds)

            # Wait for the batch write thread to finish, showing a spinner
            while self.write_queue.qsize() > 0:
                # Loading spinner
                time.sleep(0.1)
            self.process_write_batches = False

            if not return_concat:
                del prompt_embeds_all
                del add_text_embeds_all
                return

            prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
            add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)

        return prompt_embeds_all, add_text_embeds_all

    def compute_embeddings_for_legacy_prompts(
        self,
        prompts: list = None,
        return_concat: bool = True,
        load_from_cache: bool = True,
    ):
        prompt_embeds_all = []

        with torch.no_grad():
            for prompt in tqdm(
                prompts or self.prompts,
                desc="Processing prompts",
                leave=False,
                ncols=125,
                disable=return_concat,
            ):
                filename = os.path.join(
                    self.cache_dir, self.create_hash(prompt) + ".pt"
                )
                prompt = PromptHandler.filter_caption(self.data_backend, prompt)

                if (
                    load_from_cache
                    and self.data_backend.exists(filename)
                    and not return_concat
                ):
                    continue
                if self.data_backend.exists(filename):
                    self.debug_log(f"Loading from cache: {filename}")
                    prompt_embeds = self.load_from_cache(filename)
                else:
                    self.debug_log(f"Encoding prompt: {prompt}")
                    prompt_embeds = self.encode_legacy_prompt(
                        self.text_encoders[0], self.tokenizers[0], [prompt]
                    )
                    prompt_embeds = prompt_embeds.to(self.accelerator.device)
                    self.save_to_cache(filename, prompt_embeds)

                prompt_embeds_all.append(prompt_embeds)

            # Wait for the batch write thread to finish, showing a spinner
            while self.write_queue.qsize() > 0:
                # Loading spinner
                time.sleep(0.1)
            self.process_write_batches = False

            if not return_concat:
                logger.info(
                    "Not returning embeds, since we just concatenated a whackload of them."
                )
                del prompt_embeds_all
                return

        return prompt_embeds_all

    def split_cache_between_processes(self, prompts: list):
        # Use the accelerator to split the data
        with self.accelerator.split_between_processes(prompts) as split_files:
            self.prompts = split_files

    def __del__(self):
        """Ensure that the batch write thread is properly closed."""
        if self.batch_write_thread.is_alive():
            self.batch_write_thread.join()
