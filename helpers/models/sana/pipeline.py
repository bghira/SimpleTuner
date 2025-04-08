# Copyright 2025 Terminus Research and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import inspect
import re
import urllib.parse as ul
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

from diffusers import DiffusionPipeline
from diffusers.loaders import SanaLoraLoaderMixin
from diffusers.models import AutoencoderDC, SanaTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import (
    is_bs4_available,
    is_ftfy_available,
    is_torch_xla_available,
    logging,
)
from diffusers.image_processor import PixArtImageProcessor, PipelineImageInput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import (
    ASPECT_RATIO_512_BIN,
    ASPECT_RATIO_1024_BIN,
)
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import ASPECT_RATIO_2048_BIN
from diffusers.pipelines.pixart_alpha.pipeline_pixart_sigma import (
    retrieve_timesteps,
)
from diffusers.pipelines.sana.pipeline_sana import (
    ASPECT_RATIO_4096_BIN,
    SanaPipelineOutput,
)
import ftfy
from bs4 import BeautifulSoup
from diffusers.utils import is_xformers_available


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class SanaImg2ImgPipeline(DiffusionPipeline, SanaLoraLoaderMixin):
    r"""
    Pipeline for image-to-image generation using [Sana](https://huggingface.co/papers/2410.10629).

    This pipeline extends the [`SanaPipeline`] approach, but starts from an initial image (rather than pure noise)
    and partially denoises it. Similar to the Stable Diffusion Img2Img concept, you can control how strongly the
    original image is preserved vs. how strongly it is altered using the `strength` parameter.

    Args:
        tokenizer ([`GemmaTokenizer`] or [`GemmaTokenizerFast`]):
            Tokenizer to tokenize text for the Gemma text encoder.
        text_encoder ([`Gemma2PreTrainedModel`]):
            The text encoder to encode the prompt into embeddings.
        vae ([`AutoencoderDC`]):
            Variational Autoencoder Model to encode and decode images.
        transformer ([`SanaTransformer2DModel`]):
            The core Sana image transformer model used to do the diffusion-based noising/denoising of latents.
        scheduler ([`DPMSolverMultistepScheduler`]):
            The scheduler used to step the diffusion process forward (or backward).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        tokenizer: Union[GemmaTokenizer, GemmaTokenizerFast],
        text_encoder: Gemma2PreTrainedModel,
        vae: AutoencoderDC,
        transformer: SanaTransformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        # The scale factor from the VAE config helps us understand the shape of latents
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 32
        )
        self.image_processor = PixArtImageProcessor(
            vae_scale_factor=self.vae_scale_factor
        )

        # placeholders for dynamic usage
        self._guidance_scale = 4.5
        self._attention_kwargs: Dict[str, Any] = {}
        self._interrupt = False
        self._num_timesteps = None

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding to save memory.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding to save memory and allow large images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding.
        """
        self.vae.disable_tiling()

    # Copied from pipeline_sana._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        import re
        import html

        if clean_caption and not is_bs4_available():
            logger.warning(
                "clean_caption=True requires beautifulsoup4. Setting clean_caption=False..."
            )
            clean_caption = False

        if clean_caption and not is_ftfy_available():
            logger.warning(
                "clean_caption=True requires ftfy. Setting clean_caption=False..."
            )
            clean_caption = False

        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(txt: str):
            if clean_caption:
                txt = self._clean_caption(txt)
                txt = self._clean_caption(txt)
            else:
                txt = txt.lower().strip()
            return txt

        return [process(t) for t in text]

    # Copied from pipeline_sana._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            self.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    def _get_gemma_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        dtype: torch.dtype,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
    ):
        """
        Copied from pipeline_sana's `_get_gemma_prompt_embeds`. Prepares (tokenizes + encodes) prompt into Gemma text-encoder embeddings.
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt

        # optional cleaning
        prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)

        # handle complex human instruction
        if not complex_human_instruction:
            max_length_all = max_sequence_length
        else:
            chi_prompt = "\n".join(complex_human_instruction)
            prompt = [chi_prompt + p for p in prompt]
            num_chi_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_tokens + max_sequence_length - 2

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length_all,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        prompt_embeds = self.text_encoder(input_ids, attention_mask=attention_mask)
        prompt_embeds = prompt_embeds[0].to(dtype=dtype, device=device)

        return prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 300,
        complex_human_instruction: Optional[List[str]] = None,
        lora_scale: Optional[float] = None,
    ):
        r"""
        Same logic as the main SanaPipeline, but used for the Img2Img pipeline. Encodes the prompt into text-encoder
        hidden states, ready to be fed into the Transformer.
        """
        if device is None:
            device = self._execution_device

        # The model dtype
        if self.transformer is not None:
            dtype = self.transformer.dtype
        elif self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        else:
            dtype = torch.float32

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # handle LoRA scale
        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._lora_scale = lora_scale
            self._maybe_scale_lora(self.text_encoder, lora_scale)

        # If the user didn't supply direct embeddings, we compute them
        if prompt_embeds is None:
            # encode
            prompt_embeds, prompt_attention_mask = self._get_gemma_prompt_embeds(
                prompt=prompt,
                device=device,
                dtype=dtype,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
                complex_human_instruction=complex_human_instruction,
            )
            # clip
            select_index = [0] + list(range(-max_sequence_length + 1, 0))
            prompt_embeds = prompt_embeds[:, select_index]
            prompt_attention_mask = prompt_attention_mask[:, select_index]

        # replicate for batch expansions
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # handle negative prompt
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            # negative prompt
            negative_prompt = (
                [negative_prompt] * batch_size
                if isinstance(negative_prompt, str)
                else negative_prompt
            )
            negative_prompt_embeds, negative_prompt_attention_mask = (
                self._get_gemma_prompt_embeds(
                    prompt=negative_prompt,
                    device=device,
                    dtype=dtype,
                    clean_caption=clean_caption,
                    max_sequence_length=max_sequence_length,
                    complex_human_instruction=False,
                )
            )
            negative_prompt_embeds = negative_prompt_embeds[:, select_index]
            negative_prompt_attention_mask = negative_prompt_attention_mask[
                :, select_index
            ]

            # replicate
            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size, -1
            )
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(
                num_images_per_prompt, 1
            )

        # un-scale the LoRA if used
        if lora_scale is not None and isinstance(self, SanaLoraLoaderMixin):
            self._maybe_unscale_lora(self.text_encoder)

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def check_inputs(
        self,
        prompt,
        height,
        width,
        strength,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if strength < 0 or strength > 1:
            raise ValueError(f"strength must be in [0.0, 1.0], but is {strength}")

        if callback_on_step_end_tensor_inputs is not None:
            for key in callback_on_step_end_tensor_inputs:
                if key not in self._callback_tensor_inputs:
                    raise ValueError(
                        f"callback_on_step_end_tensor_inputs={callback_on_step_end_tensor_inputs} contains invalid "
                        f"tensor key {key}. Allowed are: {self._callback_tensor_inputs}"
                    )

        # height/width must be multiple of 32
        if height % 32 != 0 or width % 32 != 0:
            raise ValueError(
                f"`height` and `width` have to be multiples of 32, but are {height} and {width}."
            )

        # checking that prompt or prompt_embeds is set
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                "Cannot define both `prompt` and `prompt_embeds` at the same time."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError("Must define either `prompt` or `prompt_embeds`.")
        if prompt is not None:
            if not isinstance(prompt, (str, list)):
                raise ValueError(
                    f"`prompt` must be type `str` or `list`, but is {type(prompt)}"
                )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError(
                "Must provide `prompt_attention_mask` when specifying `prompt_embeds` directly."
            )
        if (
            negative_prompt_embeds is not None
            and negative_prompt_attention_mask is None
        ):
            raise ValueError(
                "Must provide `negative_prompt_attention_mask` when specifying `negative_prompt_embeds` directly."
            )

        # also check that negative prompt usage matches
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                "Cannot define both `negative_prompt` and `negative_prompt_embeds` simultaneously."
            )

        # shape check
        if (
            prompt_embeds is not None
            and negative_prompt_embeds is not None
            and prompt_embeds.shape != negative_prompt_embeds.shape
        ):
            raise ValueError(
                f"shape mismatch: `prompt_embeds` = {prompt_embeds.shape} vs "
                f"`negative_prompt_embeds` = {negative_prompt_embeds.shape}"
            )

    def get_timesteps(
        self, num_inference_steps: int, strength: float, device: torch.device
    ):
        """
        Helper to compute which portion of the diffusion timesteps to use for a given strength. We do the same logic
        as typical "img2img" approaches: we skip (1-strength) portion of the beginning of the noise schedule.
        """
        # standard approach: compute how many steps to actually use
        init_timestep = int(num_inference_steps * strength)
        init_timestep = min(init_timestep, num_inference_steps)

        # For a scheduler with .timesteps, the real-time steps might be multiple of order
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        # Because the pipeline only does partial diffusion from t_start onward,
        # the effective number of steps is (num_inference_steps - t_start).
        new_num_inference_steps = num_inference_steps - t_start

        return timesteps, new_num_inference_steps

    def prepare_latents(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        batch_size: int,
        num_images_per_prompt: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator], None] = None,
    ):
        """
        1. Encode the init image into latents using the VAE encoder.
        2. Expand latents for batch size and guidance
        3. Add noise according to the timesteps
        """
        if len(image.shape) == 3:
            image = image.unsqueeze(0)

        # shape = [batch_size, 3, H, W] after pre-processing
        image = image.to(device=device, dtype=dtype)
        repeat_by = batch_size * num_images_per_prompt // image.shape[0]
        if image.shape[0] * repeat_by != batch_size * num_images_per_prompt:
            raise ValueError(
                f"Cannot broadcast image batch (size {image.shape[0]}) to match requested batch size "
                f"{batch_size * num_images_per_prompt}."
            )
        image = image.repeat_interleave(repeat_by, dim=0)

        # encode with the VAE
        with torch.no_grad():
            latents = self.vae.encode(image).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor  # typical scaling for Sana
        # shape = [B, latent_channels, height/8, width/8] or /32 depending on config

        # now we add noise to latents
        # because we want to partially degrade them, so we can do "img2img" rather than starting from pure noise
        shape = latents.shape
        if isinstance(generator, list) and len(generator) != shape[0]:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested a latent batch "
                f"size of {shape[0]}. Make sure the batch size matches the number of generators."
            )

        noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        # scale the latents to the correct variance by the current scheduler step
        latents = self.scheduler.add_noise(latents, noise, timestep)

        return latents

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        image: PipelineImageInput = None,
        strength: float = 0.8,
        num_inference_steps: int = 20,
        timesteps: Optional[List[int]] = None,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = 1024,
        width: Optional[int] = 1024,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        clean_caption: bool = False,
        use_resolution_binning: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 300,
        complex_human_instruction: List[str] = None,
    ) -> Union[SanaPipelineOutput, Tuple]:
        r"""
        The call method to generate an image starting from an initial image.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                Prompt(s) to guide image generation. If not provided, must pass `prompt_embeds`.
            negative_prompt (`str` or `List[str]`, *optional*):
                Negative prompt(s). Only relevant if `guidance_scale > 1.0`.
            image (`PipelineImageInput`, *required*):
                The initial image (PIL, torch.Tensor, etc.) from which to start.
            strength (`float`, defaults to 0.8):
                Proportion of denoising steps to be applied. Value in [0.0, 1.0]. If strength is 1.0, the initial image
                is used solely for shape but the final result may deviate heavily. If it's closer to 0.0, the final
                image will preserve more of the initial image details.
            num_inference_steps (`int`, defaults to 20):
                Number of denoising steps (sometimes called "steps"). The higher, the longer the process but typically
                better results.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to override the default scheduler steps. Must match the scheduler's expectations.
            sigmas (`List[float]`, *optional*):
                Alternative custom schedule for some schedulers. Not used if `timesteps` is set.
            guidance_scale (`float`, defaults to 4.5):
                Classifier-Free guidance scale, i.e. how strongly to push the generation towards the prompt vs. just
                unconditional generation. Usually >1.0 to achieve guided results.
            num_images_per_prompt (`int`, defaults to 1):
                How many images to generate per prompt input.
            height, width (`int`, defaults to 1024):
                The final output resolution in pixels (height, width). Must be multiples of 32 for Sana.
            generator (`torch.Generator` or list, *optional*):
                RNG for deterministic generation. Can pass a list of torch.Generators for batched generation.
            prompt_embeds (`torch.Tensor`, *optional*):
                Already pre-computed embeddings for the prompt. If provided, skip textual encoding from `prompt`.
            prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for `prompt_embeds`.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Already pre-computed embeddings for negative prompt.
            negative_prompt_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for `negative_prompt_embeds`.
            output_type (`str`, defaults to "pil"):
                Format of the returned images. Either `"pil"` or `"np.array"`.
            return_dict (`bool`, defaults to True):
                Whether to return a SanaPipelineOutput or a plain tuple.
            clean_caption (`bool`, defaults to False):
                Whether to run cleaning logic on the input prompt (requires `beautifulsoup4` + `ftfy`).
            use_resolution_binning (`bool`, defaults to True):
                If True, will quantize the requested resolution to the nearest recommended aspect ratio for Sana
                transformations, then scale back after decoding. Helps keep output aligned with model training scales.
            attention_kwargs (`Dict[str, Any]`, *optional*):
                Extra arguments for attention modules.
            callback_on_step_end (`Callable`, *optional*):
                Function called at the end of each step, with signature
                `callback_on_step_end(self, step: int, timestep: int, callback_kwargs: Dict)`.
            callback_on_step_end_tensor_inputs (`List[str]`, defaults to `["latents"]`):
                Which pipeline tensors to pass along to the `callback_on_step_end` function inside `callback_kwargs`.
            max_sequence_length (`int`, defaults to 300):
                Maximum token length for the prompt(s).
            complex_human_instruction (`List[str]`, *optional*):
                If not None, concatenated to the start of the prompt for advanced instruction-based generation.

        Returns:
            [`SanaPipelineOutput`] or `tuple`:
                If `return_dict=True`, returns a SanaPipelineOutput with the `images` field. Otherwise, returns a
                tuple: (images, ).
        """
        if image is None:
            raise ValueError(
                "`image` is required for SanaImg2ImgPipeline (the initial image)."
            )

        # Possibly bin the resolution
        orig_height, orig_width = height, width
        if use_resolution_binning:
            if self.transformer.config.sample_size == 128:
                aspect_ratio_bin = ASPECT_RATIO_4096_BIN
            elif self.transformer.config.sample_size == 64:
                aspect_ratio_bin = ASPECT_RATIO_2048_BIN
            elif self.transformer.config.sample_size == 32:
                aspect_ratio_bin = ASPECT_RATIO_1024_BIN
            elif self.transformer.config.sample_size == 16:
                aspect_ratio_bin = ASPECT_RATIO_512_BIN
            else:
                raise ValueError("Invalid sample_size in transformer config.")
            height, width = self.image_processor.classify_height_width_bin(
                height, width, aspect_ratio_bin
            )

        # check inputs
        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            strength=strength,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = (
            attention_kwargs if attention_kwargs is not None else {}
        )
        self._interrupt = False

        # figure out batch_size
        if prompt is not None:
            if isinstance(prompt, str):
                batch_size = 1
            else:
                batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # handle LoRA scaling from attention_kwargs
        lora_scale = self._attention_kwargs.get("scale", None)

        # encode prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            neg_prompt_embeds,
            neg_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
            complex_human_instruction=complex_human_instruction,
            lora_scale=lora_scale,
        )

        # if we do CF guidance, we cat the negative embeddings
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat(
                [neg_prompt_attention_mask, prompt_attention_mask], dim=0
            )

        # Prepare timesteps
        # If the user passes custom timesteps or sigmas, we can do that too:
        timesteps_, num_inference_steps_ = retrieve_timesteps(
            self.scheduler,
            num_inference_steps=num_inference_steps,
            device=device,
            timesteps=timesteps,
            sigmas=sigmas,
        )
        # But we only do partial usage from the last portion, so we slice them
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps_, strength, device
        )
        self._num_timesteps = len(timesteps)

        # Pre-process the input image to the correct size, then get latents
        # (We downsample the image to VAE resolution, then add noise)
        init_image = self.image_processor.preprocess(image, height=height, width=width)
        init_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        latents = self.prepare_latents(
            init_image,
            init_timestep,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            device,
            generator,
        )

        # Denoising
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    break

                # expand latents if CF guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                # forward pass of the Transformer
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=timestep,
                    return_dict=False,
                    attention_kwargs=self.attention_kwargs,
                )[0]
                noise_pred = noise_pred.float()

                # guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # handle learned sigma if out_channels // 2 == in_channels, etc. (similar to pipeline_sana)
                if (
                    self.transformer.config.out_channels // 2
                    == self.transformer.config.in_channels
                ):
                    noise_pred, _ = noise_pred.chunk(2, dim=1)

                # step
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                # callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    neg_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", neg_prompt_embeds
                    )

                # progress
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # decoding
        if output_type == "latent":
            image = latents
        else:
            latents = latents.to(self.vae.dtype)
            # decode
            try:
                image = self.vae.decode(
                    latents / self.vae.config.scaling_factor, return_dict=False
                )[0]
            except torch.cuda.OutOfMemoryError as e:
                warnings.warn(
                    f"OutOfMemoryError: {e}. Try using `pipe.vae.enable_tiling()` or smaller images to reduce memory usage."
                )
            # if we used binning, resize final
            if use_resolution_binning:
                image = self.image_processor.resize_and_crop_tensor(
                    image, orig_width, orig_height
                )

            # finalize
            image = self.image_processor.postprocess(image, output_type=output_type)

        # cleanup
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)

        return SanaPipelineOutput(images=image)
