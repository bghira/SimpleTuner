from __future__ import annotations

import importlib.util
import math
import warnings

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn

from simpletuner.helpers.models.ideogram.prompting import Ideogram4JSONSchemaConverter
from simpletuner.helpers.models.prompt_upsampling import PromptUpsampler

PROMPT_UPSAMPLE_TEMPERATURE = 1.0


class Ideogram4PromptEnhancerHead(ModelMixin, ConfigMixin):
    """LM head that makes the head-less Qwen3-VL text encoder generative."""

    @register_to_config
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 151936):
        super().__init__()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


def build_prompt_enhancer(text_encoder, prompt_enhancer_head):
    from accelerate import init_empty_weights
    from transformers import Qwen3VLForConditionalGeneration

    with init_empty_weights():
        enhancer = Qwen3VLForConditionalGeneration(text_encoder.config)
    enhancer.model = text_encoder
    enhancer.lm_head = prompt_enhancer_head.lm_head
    return enhancer.eval()


CAPTION_USER_TEMPLATE = """TARGET IMAGE ASPECT RATIO: {aspect_ratio} (width:height).
User idea: {original_prompt}"""


def build_caption_logits_processor(model, tokenizer):
    import outlines

    outlines_model = outlines.from_transformers(model, tokenizer)
    return outlines.Generator(outlines_model, Ideogram4JSONSchemaConverter.CAPTION_MODEL).logits_processor


def generate_captions(
    prompt_enhancer,
    tokenizer,
    logits_processor,
    prompt: str | list[str],
    height: int,
    width: int,
    temperature: float = PROMPT_UPSAMPLE_TEMPERATURE,
    max_new_tokens: int = 1024,
    generator: torch.Generator | list[torch.Generator] | None = None,
    device: torch.device | None = None,
    reference_images=(),
    processor=None,
) -> list[str]:
    device = device or prompt_enhancer.device
    prompts = [prompt] if isinstance(prompt, str) else list(prompt)
    divisor = math.gcd(width, height) or 1
    aspect_ratio = f"{width // divisor}:{height // divisor}"

    sampling_seed = None
    if generator is not None:
        gen = generator[0] if isinstance(generator, list) else generator
        sampling_seed = int(torch.randint(0, 2**63 - 1, (1,), generator=gen, device=gen.device).item())
    fork_devices = [device] if getattr(device, "type", None) == "cuda" else []

    captions = []
    for i, text_prompt in enumerate(prompts):
        system_message = Ideogram4JSONSchemaConverter.GENERATION_INSTRUCTIONS
        content = CAPTION_USER_TEMPLATE.format(aspect_ratio=aspect_ratio, original_prompt=text_prompt)
        if reference_images:
            system_message += (
                "\nThe user supplied reference images for an edit. Describe the requested final image in the same JSON "
                "schema. Preserve the references' subjects and appearance except for changes the user requests."
            )
            content = [{"type": "image", "image": image} for image in reference_images] + [{"type": "text", "text": content}]
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": content,
            },
        ]
        input_processor = processor if reference_images else tokenizer
        inputs = input_processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature,
            "use_cache": True,
        }
        if logits_processor is not None:
            logits_processor.reset()
            generate_kwargs["logits_processor"] = [logits_processor]
        with torch.random.fork_rng(devices=fork_devices, enabled=sampling_seed is not None):
            if sampling_seed is not None:
                torch.manual_seed(sampling_seed + i)
            generated = prompt_enhancer.generate(**inputs, **generate_kwargs)
        new_tokens = generated[:, inputs["input_ids"].shape[1] :]
        captions.append(tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip())
    return captions


class IdeogramPromptUpsampler(PromptUpsampler):
    def __init__(self, text_encoder, tokenizer, head, device):
        self.tokenizer = tokenizer
        self.head = head
        self.device = device
        self.enhancer = build_prompt_enhancer(text_encoder, head)
        self.logits_processor = None
        if importlib.util.find_spec("outlines") is not None:
            self.logits_processor = build_caption_logits_processor(self.enhancer, tokenizer)
        else:
            warnings.warn(
                "`outlines` is not installed; Ideogram prompt upsampling runs unconstrained "
                "and may not return schema-valid JSON.",
                stacklevel=2,
            )
        self.processor = None
        self.temperature = PROMPT_UPSAMPLE_TEMPERATURE
        self.max_new_tokens = 1024
        self.generator = None

    def upsample(self, requests):
        self.head.to(self.device)
        self.enhancer.model.to(self.device)
        captions = []
        for request in requests:
            if request.reference_images and self.processor is None:
                from transformers import AutoProcessor

                self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
                self.processor.tokenizer = self.tokenizer
            captions.extend(
                generate_captions(
                    self.enhancer,
                    self.tokenizer,
                    self.logits_processor,
                    request.prompt,
                    request.height,
                    request.width,
                    temperature=self.temperature,
                    max_new_tokens=self.max_new_tokens,
                    generator=self.generator,
                    device=self.device,
                    reference_images=request.reference_images,
                    processor=self.processor,
                )
            )
        return captions
