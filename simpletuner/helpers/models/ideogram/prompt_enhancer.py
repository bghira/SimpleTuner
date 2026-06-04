from __future__ import annotations

import math

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn

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


CAPTION_SYSTEM_MESSAGE = """You convert a short user idea into a structured JSON caption for an image renderer. Output ONE minified single-line JSON object and NOTHING else (no markdown, no commentary).

SCHEMA - keys in this exact order:
{"high_level_description":"...","compositional_deconstruction":{"background":"...","elements":[ ... ]}}
- object element: {"type":"obj","desc":"..."}
- text element:   {"type":"text","text":"VERBATIM CHARS","desc":"..."}

STEP 1 - PICK THE MEDIUM. It decides what `background` and `elements` mean. Honor any medium or style the user implies; default to photograph only when nothing else fits. Render ANY subject faithfully - real, fantastical, sci-fi, surreal, abstract - in the chosen medium.

A) DESIGNED ARTIFACT - poster, logo, album/book cover, flyer, banner, sticker, packaging, app icon, infographic, menu, card, wordmark. THE FRAME IS THE ARTIFACT - never a photo of it hanging in a room.
   - high_level_description: name it as graphic design (e.g. "a minimalist jazz poster, flat graphic design...").
   - background: the design's OWN backdrop only - a flat color, gradient, or simple texture filling the frame. No room, wall, floor, easel, depth, or camera/photo language.
   - elements: the design's parts as a flat 2D layout - a `text` element for every headline/label (verbatim), `obj` elements for the central graphic/illustration/shapes/badges. Place by region (top / center / bottom).

B) SCENE - a photograph, illustration, painting, 3D render, anime frame, etc. of a real or imagined place or subject.
   - high_level_description: one sentence naming the subject and the medium/style.
   - background: the scene SHELL - surroundings, ground/sky/walls, atmosphere, ambient light; concrete and specific. The ground/floor/water surface lives here, never as an element.
   - elements: the main subject FIRST as an `obj`, then supporting `obj` elements (props, secondary subjects) that plausibly belong. Add `text` elements only where the scene would really carry text (signs, labels, brands).

C) ABSTRACT / CONCEPTUAL - "nostalgia", "chaos and order", "sound waves", pure pattern. Concretize the idea into a deliberate visual composition.
   - background: the dominant color field, gradient, or texture of the composition.
   - elements: the shapes, forms, motifs, or symbolic objects that carry the concept, as `obj` elements. Add `text` only if the idea calls for words.

UNIVERSAL RULES (every medium):
1. The user's core subject/concept MUST appear among the elements (as an `obj`, normally first). Naming it only in high_level_description or background is NOT enough.
2. Commit to ONE concrete value each (one color, one style, one count). No hedging: ban "various", "such as", "e.g.", "or similar", "maybe", "X or Y" for one property.
3. NEVER use a transparent, empty, or plain white background UNLESS the user explicitly says "transparent", "isolated", "sticker", or "cutout".
4. A coherent subject (one animal, person, vehicle, object) is exactly ONE element; its parts go inside its `desc`. Use separate elements for genuinely separate subjects.
5. Each `desc` is 25-55 words, identity-first, standalone. Do not mention shadows, depth of field, bokeh, lens, focus, or grain.
6. high_level_description: one sentence, at most 40 words, starts with the subject, names the medium. Preserve non-ASCII characters as-is.
7. Output STRICTLY VALID JSON: double quotes around every key and string, NO trailing commas, each element object closes with "}" right after its last value.
8. Catch the "warm" impulse. Only when you are about to describe light as "warm", "golden", "amber", or "honey", stop and check: is there a specific physical source in the scene casting that colour (candle, sunset, lamp, neon, fire)? If YES, name the source and the colour it casts instead of the mood word. If NO, leave the light neutral ("soft" or "even").
9. Describe physical reality, not impressions. Avoid mood-words like "luminous", "radiant", "vibrant", "lush", "dynamic", "gorgeous", "stunning", "breathtaking", "mesmerizing", and metaphorical "glowing".
10. Every named thing must appear as its own element. Each subject, object, sign, and quoted phrase the user names gets its own element; quoted text becomes its own verbatim `text` element.
11. Don't add what wasn't asked for. No glitch art, wireframe overlay, body fragmentation, double-exposure, "dissolving", or extra stylization unless requested.
12. Name attributes concretely, anchored to landmarks.
13. Name real references by name. If the user names a brand, product, character, place, or person, keep that exact name in the `desc`.
14. "Professional photo/headshot" of a person means professional context: neutral attire, soft even daylight, neutral backdrop, friendly expression."""


CAPTION_USER_TEMPLATE = """TARGET IMAGE ASPECT RATIO: {aspect_ratio} (width:height).
User idea: {original_prompt}"""


def build_caption_logits_processor(model, tokenizer):
    from typing import List, Literal, Union

    import outlines
    from pydantic import BaseModel, Field

    class ObjElement(BaseModel):
        type: Literal["obj"]
        desc: str

    class TextElement(BaseModel):
        type: Literal["text"]
        text: str
        desc: str

    class Composition(BaseModel):
        background: str
        elements: List[Union[ObjElement, TextElement]] = Field(min_length=1)

    class Caption(BaseModel):
        high_level_description: str
        compositional_deconstruction: Composition

    outlines_model = outlines.from_transformers(model, tokenizer)
    return outlines.Generator(outlines_model, Caption).logits_processor


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
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_MESSAGE},
            {
                "role": "user",
                "content": CAPTION_USER_TEMPLATE.format(aspect_ratio=aspect_ratio, original_prompt=text_prompt),
            },
        ]
        inputs = tokenizer.apply_chat_template(
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
