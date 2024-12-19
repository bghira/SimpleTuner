import os
import torch
import hashlib
import logging
import time
import gc
from tqdm import tqdm
from helpers.data_backend.base import BaseDataBackend
from helpers.training.state_tracker import StateTracker
from helpers.prompts import PromptHandler
from helpers.training.multi_process import rank_info
from queue import Queue
import queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from helpers.training.multi_process import _get_rank as get_rank, should_log
from helpers.webhooks.mixin import WebhookMixin

logger = logging.getLogger("TextEmbeddingCache")
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def _encode_sd3_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    zero_padding_tokens: bool = True,
    max_sequence_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    attention_mask = text_inputs.attention_mask.to(device)

    if zero_padding_tokens:
        # for some reason, SAI's reference code doesn't bother to mask the prompt embeddings.
        # this can lead to a problem where the model fails to represent short and long prompts equally well.
        # additionally, the model learns the bias of the prompt embeds' noise.
        return prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
    else:
        return prompt_embeds


def _encode_sd3_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    max_token_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


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
        self.should_abort = False
        self.data_backend = data_backend
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        self.model_type = model_type
        self.pipeline = None
        if self.model_type == "sana":
            from diffusers.pipelines.sana import SanaPipeline

            self.pipeline = SanaPipeline.from_pretrained(
                pretrained_model_name_or_path=StateTracker.get_args().pretrained_model_name_or_path,
                text_encoder=text_encoders[0],
                tokenizer=tokenizers[0],
                vae=None,
                transformer=None,
            )
        if self.model_type == "flux":
            from diffusers.pipelines.flux import FluxPipeline

            self.pipeline = FluxPipeline.from_pretrained(
                pretrained_model_name_or_path=StateTracker.get_args().pretrained_model_name_or_path,
                text_encoder=text_encoders[0],
                text_encoder_2=text_encoders[1],
                tokenizer=tokenizers[0],
                tokenizer_2=tokenizers[1],
                transformer=None,
                vae=None,
            )
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
        logger.debug(f"Batch: {batch}")
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

    def encode_flux_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        is_validation: bool = False,
        zero_padding_tokens: bool = True,
    ):
        """
        Encode a prompt for a Flux model.

        Args:
            text_encoders: List of text encoders.
            tokenizers: List of tokenizers.
            prompt: The prompt to encode.
            num_images_per_prompt: The number of images to generate per prompt.
            is_validation: Whether the prompt is for validation. No-op for SD3.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds).
        """
        from helpers.models.flux import FluxPipeline

        pipe = FluxPipeline(
            self.pipeline.scheduler,
            self.pipeline.vae,
            self.pipeline.text_encoder,
            self.pipeline.tokenizer,
            self.pipeline.text_encoder_2,
            self.pipeline.tokenizer_2,
            self.pipeline.transformer,
        )

        prompt_embeds, pooled_prompt_embeds, time_ids, masks = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=self.accelerator.device,
            max_sequence_length=StateTracker.get_args().tokenizer_max_length,
        )
        if zero_padding_tokens:
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(
                device=prompt_embeds.device
            ).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, pooled_prompt_embeds, time_ids, masks

    # Adapted from pipelines.StableDiffusion3Pipeline.encode_prompt
    def encode_sd3_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        is_validation: bool = False,
        zero_padding_tokens: bool = False,
    ):
        """
        Encode a prompt for an SD3 model.

        Args:
            text_encoders: List of text encoders.
            tokenizers: List of tokenizers.
            prompt: The prompt to encode.
            num_images_per_prompt: The number of images to generate per prompt.
            is_validation: Whether the prompt is for validation. No-op for SD3.

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds).
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        num_images_per_prompt = 1

        clip_tokenizers = tokenizers[:2]
        clip_text_encoders = text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = _encode_sd3_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                device=self.accelerator.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

        t5_prompt_embed = _encode_sd3_prompt_with_t5(
            text_encoders[-1],
            tokenizers[-1],
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=self.accelerator.device,
            zero_padding_tokens=zero_padding_tokens,
            max_sequence_length=StateTracker.get_args().tokenizer_max_length,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds

    def encode_legacy_prompt(self, text_encoder, tokenizer, prompt):
        input_tokens = tokenizer(
            PromptHandler.filter_caption(self.data_backend, prompt),
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(self.accelerator.device)
        output = text_encoder(input_tokens)[0]
        # self.debug_log(f"Legacy prompt shape: {output.shape}")
        # self.debug_log(f"Legacy prompt encoded: {output}")
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
        try:
            for tokenizer, text_encoder in zip(tokenizers, text_encoders):
                if tokenizer is None or text_encoder is None:
                    # SDXL Refiner only has one text encoder and tokenizer
                    continue
                if type(prompt) is not str and type(prompt) is not list:
                    prompt = str(prompt)
                max_seq_len = 256 if self.model_type == "kolors" else 77
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_seq_len,
                )
                untruncated_ids = tokenizer(
                    prompt,
                    padding="longest",
                    return_tensors="pt",
                    max_length=max_seq_len,
                ).input_ids

                if untruncated_ids.shape[
                    -1
                ] > tokenizer.model_max_length and not torch.equal(
                    text_inputs.input_ids, untruncated_ids
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
                if self.model_type == "sdxl":
                    prompt_embeds_output = text_encoder(
                        text_inputs.input_ids.to(self.accelerator.device),
                        output_hidden_states=True,
                    )
                    # We are always interested in the pooled output of the final text encoder
                    pooled_prompt_embeds = prompt_embeds_output[0]
                    prompt_embeds = prompt_embeds_output.hidden_states[-2]
                elif self.model_type == "kolors":
                    # we pass the attention mask into the text encoder. it transforms the embeds but does not attend to them.
                    # unfortunately, kolors does not return the attention mask for later use by the U-net to avoid attending to the padding tokens.
                    prompt_embeds_output = text_encoder(
                        input_ids=text_inputs["input_ids"].to(self.accelerator.device),
                        attention_mask=text_inputs["attention_mask"].to(
                            self.accelerator.device
                        ),
                        position_ids=text_inputs["position_ids"],
                        output_hidden_states=True,
                    )
                    # the ChatGLM encoder output is hereby mangled in fancy ways for Kolors to be useful.
                    prompt_embeds = (
                        prompt_embeds_output.hidden_states[-2].permute(1, 0, 2).clone()
                    )
                    # [max_sequence_length, batch, hidden_size] -> [batch, hidden_size]
                    pooled_prompt_embeds = prompt_embeds_output.hidden_states[-1][
                        -1, :, :
                    ].clone()
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

                # Clear out anything we moved to the text encoder device
                text_inputs.input_ids.to("cpu")
                del prompt_embeds_output
                del text_inputs

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
        if self.model_type == "sdxl" or self.model_type == "kolors":
            return self.encode_sdxl_prompt(
                self.text_encoders, self.tokenizers, prompt, is_validation
            )
        elif self.model_type == "sd3":
            return self.encode_sd3_prompt(
                self.text_encoders,
                self.tokenizers,
                prompt,
                is_validation,
                zero_padding_tokens=(
                    True if StateTracker.get_args().t5_padding == "zero" else False
                ),
            )
        else:
            return self.encode_legacy_prompt(
                self.text_encoders[0], self.tokenizers[0], prompt
            )

    def tokenize_t5_prompt(self, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            # prevent runaway token length sizes.
            # huge captions aren't very helpful, and if you want them, use --tokenizer_max_length
            max_length = 144

        text_inputs = self.tokenizers[0](
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return text_inputs

    def encode_t5_prompt(self, input_ids, attention_mask):
        text_input_ids = input_ids.to(self.text_encoders[0].device)
        attention_mask = attention_mask.to(self.text_encoders[0].device)
        prompt_embeds = self.text_encoders[0](
            text_input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]
        prompt_embeds = prompt_embeds.to("cpu")

        return prompt_embeds

    def compute_t5_prompt(self, prompt: str):
        """
        Tokenise, encode, optionally mask, and then return a prompt_embed for a T5 model.

        Args:
            prompt: The prompt to encode.
        Returns:
            Tuple of (prompt_embeds, attention_mask)
        """
        logger.debug(f"Computing T5 caption for: {prompt}")
        text_inputs = self.tokenize_t5_prompt(
            prompt, tokenizer_max_length=StateTracker.get_args().tokenizer_max_length
        )
        result = self.encode_t5_prompt(
            text_inputs.input_ids,
            text_inputs.attention_mask,
        )
        attn_mask = text_inputs.attention_mask
        del text_inputs

        return result, attn_mask

    def compute_gemma_prompt(self, prompt: str, is_negative_prompt: bool):
        prompt_embeds, prompt_attention_mask, _, _ = self.pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            device=self.accelerator.device,
            clean_caption=False,
            max_sequence_length=300,
            complex_human_instruction=(
                StateTracker.get_args().sana_complex_human_instruction
                if not is_negative_prompt
                else None
            ),
        )

        return prompt_embeds, prompt_attention_mask

    def compute_embeddings_for_prompts(
        self,
        all_prompts,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
        is_negative_prompt: bool = False,
    ):
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
        if self.model_type == "sdxl" or self.model_type == "kolors":
            output = self.compute_embeddings_for_sdxl_prompts(
                raw_prompts,
                return_concat=return_concat,
                is_validation=is_validation,
                load_from_cache=load_from_cache,
            )
        elif (
            self.model_type == "legacy"
            or self.model_type == "pixart_sigma"
            or self.model_type == "smoldit"
        ):
            # both sd1.x/2.x and t5 style models like pixart use this flow.
            output = self.compute_embeddings_for_legacy_prompts(
                raw_prompts,
                return_concat=return_concat,
                load_from_cache=load_from_cache,
            )
        elif self.model_type == "sd3":
            output = self.compute_embeddings_for_sd3_prompts(
                raw_prompts,
                return_concat=return_concat,
                load_from_cache=load_from_cache,
            )
        elif self.model_type == "flux":
            output = self.compute_embeddings_for_flux_prompts(
                raw_prompts,
                return_concat=return_concat,
                load_from_cache=load_from_cache,
            )
        elif self.model_type == "sana":
            output = self.compute_embeddings_for_sana_prompts(
                raw_prompts,
                return_concat=return_concat,
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
        # self.debug_log(
        #     f"compute_embeddings_for_sdxl_prompts received list of prompts: {list(prompts)[:5]}"
        # )
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
                logger.debug(debug_msg)
                if return_concat and load_from_cache:
                    try:
                        # We attempt to load.
                        prompt_embeds, add_text_embeds = self.load_from_cache(filename)
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
                    # self.debug_log(f"Encoding prompt: {prompt}")
                    prompt_embeds, pooled_prompt_embeds = self.encode_sdxl_prompts(
                        self.text_encoders,
                        self.tokenizers,
                        [prompt],
                        is_validation,
                    )
                    add_text_embeds = pooled_prompt_embeds
                    # If the prompt is empty, zero out the embeddings
                    if prompt == "":
                        prompt_embeds = torch.zeros_like(prompt_embeds)
                        add_text_embeds = torch.zeros_like(add_text_embeds)
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
                    self.save_to_cache(filename, (prompt_embeds, add_text_embeds))

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

            while self.write_queue.qsize() > 0:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            logger.debug(
                f"Exiting text cache write busy-loop, {self.write_queue.qsize()} items remaining."
            )

            if self.webhook_handler is not None:
                self.send_progress_update(
                    type="init_cache_text_embeds_status_complete",
                    progress=100,
                    total=len(local_caption_split),
                    current=len(local_caption_split),
                )

            # Close the tqdm progress bar after the loop
            self.write_thread_bar.close()
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
        logger.debug(
            f"compute_embeddings_for_legacy_prompts arguments: prompts={prompts}, return_concat={return_concat}, load_from_cache={load_from_cache}"
        )
        prompt_embeds_all = []
        prompt_embeds_all = []
        should_encode = not load_from_cache
        args = StateTracker.get_args()
        if (
            hasattr(args, "cache_clear_validation_prompts")
            and args.cache_clear_validation_prompts
            and not load_from_cache
        ):
            # If --cache_clear_validation_prompts was provided, we will forcibly overwrite them.
            should_encode = True
            logger.debug("Setting should_encode = True")
        # self.debug_log(
        #     f"compute_embeddings_for_legacy_prompts received list of prompts: {list(prompts)[:5]}"
        # )
        if should_encode:
            local_caption_split = self.split_captions_between_processes(
                prompts or self.prompts
            )
        else:
            local_caption_split = prompts or self.prompts

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
            attention_mask = None
            attention_masks_all = []
            last_reported_index = 0
            for prompt in tqdm(
                local_caption_split,
                desc="Processing prompts",
                leave=False,
                ncols=125,
                disable=return_concat,
                position=get_rank() + self.accelerator.num_processes + 1,
            ):
                filename = os.path.join(self.cache_dir, self.hash_prompt(prompt))
                if prompt != "":
                    prompt = PromptHandler.filter_caption(self.data_backend, prompt)
                if prompt is None:
                    continue

                if return_concat and load_from_cache:
                    try:
                        # We attempt to load.
                        logging.debug("Loading embed from cache.")
                        prompt_embeds = self.load_from_cache(filename)
                        if type(prompt_embeds) is tuple and len(prompt_embeds) == 2:
                            # we have an attention mask stored with the embed.
                            prompt_embeds, attention_mask = prompt_embeds
                        logging.debug(f"Loaded embeds: {prompt_embeds.shape}")
                    except Exception as e:
                        # We failed to load. Now encode the prompt.
                        logger.error(
                            f"Failed retrieving prompt from cache:"
                            f"\n-> prompt: {prompt}"
                            f"\n-> filename: {filename}"
                            f"\n-> error: {e}"
                        )
                        should_encode = True
                        raise Exception(
                            "Cache retrieval for text embed file failed. Ensure your dataloader config value for skip_file_discovery does not contain 'text', and that preserve_data_backend_cache is disabled or unset."
                        )

                if should_encode:
                    # self.debug_log(f"Encoding prompt: {prompt}")
                    # Get the current size of the queue.
                    current_size = self.write_queue.qsize()
                    if current_size >= 2048:
                        log_msg = str(
                            f"[WARNING] Write queue size is {current_size}. This is quite large."
                            " Consider increasing the write batch size. Delaying encode so that writes can catch up."
                        )
                        self.write_thread_bar.write(log_msg)
                        while self.write_queue.qsize() > 100:
                            logger.debug("Waiting for write thread to catch up.")
                            time.sleep(5)
                    if (
                        "deepfloyd" in StateTracker.get_args().model_type
                        or self.model_type == "pixart_sigma"
                        or self.model_type == "smoldit"
                    ):
                        # TODO: Batch this
                        prompt_embeds, attention_mask = self.compute_t5_prompt(
                            prompt=prompt,
                        )
                        if "deepfloyd" not in StateTracker.get_args().model_type:
                            # we have to store the attn mask with the embed for pixart.
                            # smoldit requires the attn mask at inference time 💪🏽
                            prompt_embeds = (prompt_embeds, attention_mask)
                    else:
                        prompt_embeds = self.encode_legacy_prompt(
                            self.text_encoders[0], self.tokenizers[0], [prompt]
                        )
                    if return_concat:
                        if type(prompt_embeds) is tuple:
                            prompt_embeds = (
                                prompt_embeds[0].to(self.accelerator.device),
                                prompt_embeds[1].to(self.accelerator.device),
                            )
                        else:
                            prompt_embeds = prompt_embeds.to(self.accelerator.device)

                    self.save_to_cache(filename, prompt_embeds)

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

                if not return_concat:
                    del prompt_embeds
                    prompt_embeds = None

                if return_concat:
                    prompt_embeds_all.append(prompt_embeds)
                    if attention_mask is not None:
                        attention_masks_all.append(attention_mask)

            while self.write_queue.qsize() > 0:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            logger.debug(
                f"Exiting text cache write busy-loop, {self.write_queue.qsize()} items remaining."
            )

            if self.webhook_handler is not None:
                self.send_progress_update(
                    type="init_cache_text_embeds_status_complete",
                    progress=100,
                    total=len(local_caption_split),
                    current=len(local_caption_split),
                )

            # Close the tqdm progress bar after the loop
            self.write_thread_bar.close()
            self.process_write_batches = False

            if not return_concat:
                del prompt_embeds_all
                gc.collect()
                return

        # logger.debug(f"Returning all prompt embeds: {prompt_embeds_all}")
        if len(attention_masks_all) > 0:
            return prompt_embeds_all, attention_masks_all
        return prompt_embeds_all

    def compute_embeddings_for_sana_prompts(
        self,
        prompts: list = None,
        return_concat: bool = True,
        load_from_cache: bool = True,
        is_negative_prompt: bool = False,
    ):
        logger.debug(
            f"compute_embeddings_for_sana_prompts arguments: prompts={prompts}, return_concat={return_concat}, load_from_cache={load_from_cache}"
        )
        prompt_embeds_all = []
        prompt_embeds_all = []
        should_encode = not load_from_cache
        args = StateTracker.get_args()
        if (
            hasattr(args, "cache_clear_validation_prompts")
            and args.cache_clear_validation_prompts
            and not load_from_cache
        ):
            # If --cache_clear_validation_prompts was provided, we will forcibly overwrite them.
            should_encode = True
            logger.debug("Setting should_encode = True")
        # self.debug_log(
        #     f"compute_embeddings_for_sana_prompts received list of prompts: {list(prompts)[:5]}"
        # )
        if should_encode:
            local_caption_split = self.split_captions_between_processes(
                prompts or self.prompts
            )
        else:
            local_caption_split = prompts or self.prompts

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
            attention_mask = None
            attention_masks_all = []
            last_reported_index = 0
            for prompt in tqdm(
                local_caption_split,
                desc="Processing prompts",
                leave=False,
                ncols=125,
                disable=return_concat,
                position=get_rank() + self.accelerator.num_processes + 1,
            ):
                filename = os.path.join(self.cache_dir, self.hash_prompt(prompt))
                if prompt != "":
                    prompt = PromptHandler.filter_caption(self.data_backend, prompt)
                if prompt is None:
                    continue

                if return_concat and load_from_cache:
                    try:
                        # We attempt to load.
                        logging.debug("Loading embed from cache.")
                        prompt_embeds = self.load_from_cache(filename)
                        if type(prompt_embeds) is tuple and len(prompt_embeds) == 2:
                            # we have an attention mask stored with the embed.
                            prompt_embeds, attention_mask = prompt_embeds
                        logging.debug(f"Loaded embeds: {prompt_embeds.shape}")
                    except Exception as e:
                        # We failed to load. Now encode the prompt.
                        logger.error(
                            f"Failed retrieving prompt from cache:"
                            f"\n-> prompt: {prompt}"
                            f"\n-> filename: {filename}"
                            f"\n-> error: {e}"
                        )
                        should_encode = True
                        raise Exception(
                            "Cache retrieval for text embed file failed. Ensure your dataloader config value for skip_file_discovery does not contain 'text', and that preserve_data_backend_cache is disabled or unset."
                        )

                if should_encode:
                    # self.debug_log(f"Encoding prompt: {prompt}")
                    # Get the current size of the queue.
                    current_size = self.write_queue.qsize()
                    if current_size >= 2048:
                        log_msg = str(
                            f"[WARNING] Write queue size is {current_size}. This is quite large."
                            " Consider increasing the write batch size. Delaying encode so that writes can catch up."
                        )
                        self.write_thread_bar.write(log_msg)
                        while self.write_queue.qsize() > 100:
                            logger.debug("Waiting for write thread to catch up.")
                            time.sleep(5)
                    # TODO: Batch this
                    prompt_embeds, attention_mask = self.compute_gemma_prompt(
                        prompt=prompt, is_negative_prompt=is_negative_prompt
                    )
                    if "deepfloyd" not in StateTracker.get_args().model_type:
                        # we have to store the attn mask with the embed for pixart.
                        # smoldit requires the attn mask at inference time 💪🏽
                        prompt_embeds = (prompt_embeds, attention_mask)
                    if return_concat:
                        if type(prompt_embeds) is tuple:
                            prompt_embeds = (
                                prompt_embeds[0].to(self.accelerator.device),
                                prompt_embeds[1].to(self.accelerator.device),
                            )
                        else:
                            prompt_embeds = prompt_embeds.to(self.accelerator.device)

                    self.save_to_cache(filename, prompt_embeds)

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

                if not return_concat:
                    del prompt_embeds
                    prompt_embeds = None

                if return_concat:
                    prompt_embeds_all.append(prompt_embeds)
                    if attention_mask is not None:
                        attention_masks_all.append(attention_mask)

            while self.write_queue.qsize() > 0:
                time.sleep(0.1)  # Sleep briefly to avoid busy-waiting

            logger.debug(
                f"Exiting text cache write busy-loop, {self.write_queue.qsize()} items remaining."
            )

            if self.webhook_handler is not None:
                self.send_progress_update(
                    type="init_cache_text_embeds_status_complete",
                    progress=100,
                    total=len(local_caption_split),
                    current=len(local_caption_split),
                )

            # Close the tqdm progress bar after the loop
            self.write_thread_bar.close()
            self.process_write_batches = False

            if not return_concat:
                del prompt_embeds_all
                gc.collect()
                return

        # logger.debug(f"Returning all prompt embeds: {prompt_embeds_all}")
        if len(attention_masks_all) > 0:
            return prompt_embeds_all, attention_masks_all
        return prompt_embeds_all

    def compute_embeddings_for_flux_prompts(
        self,
        prompts: list = None,
        return_concat: bool = True,
        is_validation: bool = False,
        load_from_cache: bool = True,
    ):
        prompt_embeds_all = []
        add_text_embeds_all = []
        time_ids_all = []
        masks_all = []
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
                        _flux_embed = self.load_from_cache(filename)
                        if len(_flux_embed) == 3:
                            # legacy flux embed w/o attn mask
                            prompt_embeds, add_text_embeds, time_ids = _flux_embed
                            masks = None
                        elif len(_flux_embed) == 4:
                            # flux embed with attn mask
                            prompt_embeds, add_text_embeds, time_ids, masks = (
                                _flux_embed
                            )
                        del _flux_embed
                        logger.debug(
                            f"Cached Flux text embeds: {prompt_embeds.shape}, {add_text_embeds.shape}, {time_ids.shape}, {masks.shape if masks is not None else None}"
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
                    self.debug_log(f"Encoding prompt: {prompt}")
                    prompt_embeds, pooled_prompt_embeds, time_ids, masks = (
                        self.encode_flux_prompt(
                            self.text_encoders,
                            self.tokenizers,
                            [prompt],
                            is_validation,
                            zero_padding_tokens=StateTracker.get_args().t5_padding
                            == "zero",
                        )
                    )
                    logger.debug(
                        f"Flux prompt embeds: {prompt_embeds.shape}, {pooled_prompt_embeds.shape}, {time_ids.shape}, {masks.shape}"
                    )
                    add_text_embeds = pooled_prompt_embeds
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
                    self.save_to_cache(
                        filename, (prompt_embeds, add_text_embeds, time_ids, masks)
                    )
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
                        prompt_embeds = prompt_embeds.to(self.accelerator.device)
                        add_text_embeds = add_text_embeds.to(self.accelerator.device)
                        time_ids = time_ids.to(self.accelerator.device)
                        masks = masks.to(self.accelerator.device)
                    else:
                        del prompt_embeds
                        del add_text_embeds
                        del pooled_prompt_embeds
                        del masks
                        continue

                if return_concat:
                    prompt_embeds_all.append(prompt_embeds)
                    add_text_embeds_all.append(add_text_embeds)
                    time_ids_all.append(time_ids)
                    masks_all.append(masks)

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
            self.process_write_batches = False

            if not return_concat:
                del prompt_embeds_all
                del add_text_embeds_all
                del time_ids_all
                del masks_all
                return

            logger.debug(f"Returning all prompt embeds: {prompt_embeds_all}")
            prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
            add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)
            time_ids_all = torch.cat(time_ids_all, dim=0)
            # if any masks_all are None, we can't cat
            masks_all = torch.cat(masks_all, dim=0) if None not in masks_all else None

        return prompt_embeds_all, add_text_embeds_all, time_ids_all, masks_all

    def compute_embeddings_for_sd3_prompts(
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
        # self.debug_log(
        #     f"compute_embeddings_for_sdxl_prompts received list of prompts: {list(prompts)[:5]}"
        # )

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
                        prompt_embeds, add_text_embeds = self.load_from_cache(filename)
                        logger.debug(
                            f"Cached SD3 embeds: {prompt_embeds.shape}, {add_text_embeds.shape}"
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
                    prompt_embeds, pooled_prompt_embeds = self.encode_sd3_prompt(
                        self.text_encoders,
                        self.tokenizers,
                        [prompt],
                        is_validation,
                        zero_padding_tokens=(
                            True
                            if StateTracker.get_args().t5_padding == "zero"
                            else False
                        ),
                    )
                    logger.debug(
                        f"Filename {filename} SD3 prompt embeds: {prompt_embeds.shape}, {pooled_prompt_embeds.shape}"
                    )
                    add_text_embeds = pooled_prompt_embeds
                    # StabilityAI say not to zero them out.
                    if prompt == "":
                        if StateTracker.get_args().sd3_clip_uncond_behaviour == "zero":
                            prompt_embeds = torch.zeros_like(prompt_embeds)
                        if StateTracker.get_args().sd3_t5_uncond_behaviour == "zero":
                            add_text_embeds = torch.zeros_like(add_text_embeds)
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
                    self.save_to_cache(filename, (prompt_embeds, add_text_embeds))

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
            self.process_write_batches = False

            if not return_concat:
                del prompt_embeds_all
                del add_text_embeds_all
                return

            logger.debug(f"Returning all prompt embeds: {prompt_embeds_all}")
            prompt_embeds_all = torch.cat(prompt_embeds_all, dim=0)
            add_text_embeds_all = torch.cat(add_text_embeds_all, dim=0)

        return prompt_embeds_all, add_text_embeds_all

    def __del__(self):
        """Ensure that the batch write thread is properly closed."""
        if self.batch_write_thread.is_alive():
            self.batch_write_thread.join()
