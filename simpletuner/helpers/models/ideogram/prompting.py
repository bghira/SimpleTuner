from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any, Literal, Union

from pydantic import BaseModel, Field

from simpletuner.helpers.models.json_schema import JSONSchemaConverter

HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{6})\b")
PHOTO_TERMS = {
    "35mm",
    "50mm",
    "85mm",
    "aperture",
    "bokeh",
    "camera",
    "cinematic",
    "depth of field",
    "dslr",
    "film",
    "lens",
    "photo",
    "photograph",
    "photoreal",
    "shot",
}


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _extract_palette(prompt: str, limit: int = 16) -> list[str]:
    colors: list[str] = []
    for match in HEX_COLOR_RE.findall(prompt or ""):
        color = match.upper()
        if color not in colors:
            colors.append(color)
        if len(colors) >= limit:
            break
    return colors


def _is_photo_prompt(prompt: str) -> bool:
    lower = prompt.lower()
    return any(term in lower for term in PHOTO_TERMS)


def _ordered_style(style: dict[str, Any], *, source_prompt: str = "") -> OrderedDict[str, Any]:
    has_photo = "photo" in style or ("art_style" not in style and _is_photo_prompt(source_prompt))
    palette = style.get("color_palette", style.get("colour_palette", None))
    if palette is None:
        palette = _extract_palette(source_prompt)
    palette = [str(color).upper() for color in palette or [] if HEX_COLOR_RE.fullmatch(str(color))]

    ordered: OrderedDict[str, Any] = OrderedDict()
    ordered["aesthetics"] = _clean_text(style.get("aesthetics") or "visually grounded, detailed")
    ordered["lighting"] = _clean_text(style.get("lighting") or "natural, balanced lighting")
    if has_photo:
        ordered["photo"] = _clean_text(style.get("photo") or "natural perspective, sharp focus")
        ordered["medium"] = _clean_text(style.get("medium") or "photograph")
    else:
        ordered["medium"] = _clean_text(style.get("medium") or "illustration")
        ordered["art_style"] = _clean_text(style.get("art_style") or "detailed digital illustration")
    if palette:
        ordered["color_palette"] = palette[:16]
    return ordered


def _ordered_element(element: dict[str, Any]) -> OrderedDict[str, Any]:
    element_type = str(element.get("type") or "obj")
    if element_type not in {"obj", "text"}:
        element_type = "obj"

    ordered: OrderedDict[str, Any] = OrderedDict()
    ordered["type"] = element_type
    bbox = element.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        ordered["bbox"] = [int(float(coord)) for coord in bbox]
    if element_type == "text":
        ordered["text"] = _clean_text(element.get("text"))
    ordered["desc"] = _clean_text(element.get("desc") or element.get("description") or element.get("label"))
    palette = element.get("color_palette", element.get("colour_palette", None))
    if palette:
        colors = [str(color).upper() for color in palette if HEX_COLOR_RE.fullmatch(str(color))]
        if colors:
            ordered["color_palette"] = colors[:5]
    return ordered


def canonicalize_ideogram_json_caption(caption: dict[str, Any], *, source_prompt: str = "") -> OrderedDict[str, Any]:
    high_level = _clean_text(caption.get("high_level_description") or caption.get("description") or source_prompt)
    style = caption.get("style_description")
    composition = caption.get("compositional_deconstruction")
    if not isinstance(style, dict):
        style = {}
    if not isinstance(composition, dict):
        composition = {}

    background = _clean_text(composition.get("background") or high_level)
    raw_elements = composition.get("elements")
    if not isinstance(raw_elements, list):
        raw_elements = []

    ordered: OrderedDict[str, Any] = OrderedDict()
    if high_level:
        ordered["high_level_description"] = high_level
    ordered["style_description"] = _ordered_style(style, source_prompt=source_prompt or high_level)
    ordered["compositional_deconstruction"] = OrderedDict(
        [
            ("background", background),
            ("elements", [_ordered_element(element) for element in raw_elements if isinstance(element, dict)]),
        ]
    )
    if not ordered["compositional_deconstruction"]["elements"]:
        ordered["compositional_deconstruction"]["elements"].append(
            OrderedDict([("type", "obj"), ("desc", high_level or background)])
        )
    return ordered


def prompt_to_ideogram_json_caption(prompt: str) -> OrderedDict[str, Any]:
    prompt = _clean_text(prompt)
    style: dict[str, Any] = {}
    palette = _extract_palette(prompt)
    if palette:
        style["color_palette"] = palette
    if _is_photo_prompt(prompt):
        style["medium"] = "photograph"
        style["photo"] = "natural perspective, sharp focus"
    else:
        style["medium"] = "illustration"
        style["art_style"] = "detailed digital illustration"
    return canonicalize_ideogram_json_caption(
        {
            "high_level_description": prompt,
            "style_description": style,
            "compositional_deconstruction": {
                "background": prompt,
                "elements": [{"type": "obj", "desc": prompt}],
            },
        },
        source_prompt=prompt,
    )


class IdeogramObjectElement(BaseModel):
    type: Literal["obj"]
    desc: str


class IdeogramTextElement(BaseModel):
    type: Literal["text"]
    text: str
    desc: str


class IdeogramComposition(BaseModel):
    background: str
    elements: list[Union[IdeogramObjectElement, IdeogramTextElement]] = Field(min_length=1)


class IdeogramCaption(BaseModel):
    high_level_description: str
    compositional_deconstruction: IdeogramComposition


IDEOGRAM_CAPTION_SYSTEM_MESSAGE = """You convert a short user idea into a structured JSON caption for an image renderer. Output ONE minified single-line JSON object and NOTHING else (no markdown, no commentary).

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


class Ideogram4JSONSchemaConverter(JSONSchemaConverter):
    SCHEMA_ID = "ideogram4-v1"
    GENERATION_INSTRUCTIONS = IDEOGRAM_CAPTION_SYSTEM_MESSAGE
    CAPTION_MODEL = IdeogramCaption
    CONVERT_NEGATIVE_PROMPTS = True

    def convert(self, prompt: str, *, height=None, width=None) -> str:
        candidate = str(prompt or "").strip()
        if candidate.startswith("{"):
            try:
                parsed = json.loads(candidate, object_pairs_hook=OrderedDict)
            except json.JSONDecodeError:
                return self.serialize(prompt_to_ideogram_json_caption(candidate))
            if isinstance(parsed, dict):
                return self.serialize(canonicalize_ideogram_json_caption(parsed, source_prompt=candidate))
        return self.serialize(prompt_to_ideogram_json_caption(candidate))
