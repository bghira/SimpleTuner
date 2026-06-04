from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Any

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


def serialize_ideogram_caption(caption: dict[str, Any]) -> str:
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def maybe_convert_prompt_to_ideogram_json(prompt: str, *, enabled: bool = True) -> str:
    if not enabled:
        return prompt

    candidate = str(prompt or "").strip()
    if not candidate:
        return serialize_ideogram_caption(prompt_to_ideogram_json_caption(""))
    if candidate.startswith("{"):
        try:
            parsed = json.loads(candidate, object_pairs_hook=OrderedDict)
        except json.JSONDecodeError:
            return serialize_ideogram_caption(prompt_to_ideogram_json_caption(candidate))
        if isinstance(parsed, dict):
            return serialize_ideogram_caption(canonicalize_ideogram_json_caption(parsed, source_prompt=candidate))
    return serialize_ideogram_caption(prompt_to_ideogram_json_caption(candidate))
