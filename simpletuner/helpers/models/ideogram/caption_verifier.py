"""Verify Ideogram 4 JSON caption format."""

import json
import re
from pathlib import Path
from typing import Sequence

PathLike = str | Path

NON_ASCII_UNICODE_ESCAPE_RE = re.compile(
  r"\\u(?:00[89a-fA-F][0-9a-fA-F]|0[1-9a-fA-F][0-9a-fA-F]{2}|[1-9a-fA-F][0-9a-fA-F]{3})"
)


class CaptionVerifier:
  """Verify Ideogram 4 JSON caption format.

  All ``verify_*`` methods return a list of warning strings (empty means
  the input passed all checks).
  """

  top_level_known_keys: frozenset[str] = frozenset(
    {
      "high_level_description",
      "style_description",
      "compositional_deconstruction",
    }
  )

  style_description_key_order_photo: Sequence[str] = (
    "aesthetics",
    "lighting",
    "photo",
    "medium",
    "color_palette",
  )

  style_description_key_order_non_photo: Sequence[str] = (
    "aesthetics",
    "lighting",
    "medium",
    "art_style",
    "color_palette",
  )

  compositional_deconstruction_key_order: Sequence[str] = (
    "background",
    "elements",
  )

  element_key_order_obj: Sequence[str] = ("type", "bbox", "desc", "color_palette")
  element_key_order_text: Sequence[str] = (
    "type",
    "bbox",
    "text",
    "desc",
    "color_palette",
  )

  style_description_known_keys: frozenset[str] = frozenset(
    {
      "aesthetics",
      "lighting",
      "photo",
      "art_style",
      "medium",
      "color_palette",
    }
  )

  element_known_keys: frozenset[str] = frozenset(
    {
      "type",
      "bbox",
      "text",
      "desc",
      "color_palette",
    }
  )

  element_types: frozenset[str] = frozenset({"obj", "text"})

  bbox_min: int = 0
  bbox_max: int = 1000

  style_description_palette_max: int = 16
  element_palette_max: int = 5

  def verify(self, caption: dict) -> list[str]:
    """Verify a parsed caption dict. Returns a list of warnings."""
    warnings: list[str] = []

    if not isinstance(caption, dict):
      warnings.append(
        f"root: expected a JSON object (dict), got {type(caption).__name__}"
      )
      return warnings

    self._check_unknown_keys(caption, self.top_level_known_keys, "root", warnings)

    if "high_level_description" in caption:
      self._verify_high_level_description(caption["high_level_description"], warnings)

    if "style_description" in caption:
      self._verify_style_description(caption["style_description"], warnings)

    if "compositional_deconstruction" in caption:
      self._verify_compositional_deconstruction(
        caption["compositional_deconstruction"], warnings
      )
    else:
      warnings.append("root: 'compositional_deconstruction' must exist")

    return warnings

  def verify_raw(self, raw_text: str) -> list[str]:
    """Verify a raw JSON string. Includes ensure_ascii=False check."""
    warnings = self.check_ensure_ascii_false(raw_text)
    try:
      caption = json.loads(raw_text)
    except json.JSONDecodeError as e:
      warnings.append(f"invalid JSON: {e}")
      return warnings
    return warnings + self.verify(caption)

  def verify_file(self, path: PathLike) -> list[str]:
    """Verify a JSON file on disk."""
    raw = Path(path).read_text(encoding="utf-8")
    return self.verify_raw(raw)

  # Warn if raw text contains \uXXXX escapes for non-ASCII chars but no literal non-ASCII;
  # typically means the writer used ensure_ascii=True. Suppressed when both forms coexist
  # (escapes are then assumed intentional).
  @classmethod
  def check_ensure_ascii_false(cls, raw_text: str, max_examples: int = 3) -> list[str]:
    warnings: list[str] = []
    matches = NON_ASCII_UNICODE_ESCAPE_RE.findall(raw_text)
    if not matches:
      return warnings

    has_literal_non_ascii = any(ord(c) > 0x7F for c in raw_text)
    if has_literal_non_ascii:
      return warnings

    examples = ", ".join(sorted(set(matches))[:max_examples])
    extra = "" if len(set(matches)) <= max_examples else ", ..."
    warnings.append(
      f"raw text: found {len(matches)} non-ASCII unicode escape(s) "
      f"(e.g. {examples}{extra}) and no literal non-ASCII characters. "
      f"This usually means the file was saved with ensure_ascii=True; "
      f"please re-save with json.dump(..., ensure_ascii=False) so non-ASCII "
      f"characters are stored as literal UTF-8. "
      f"Note: this is a warning, not an error, because the escapes could "
      f"have been written intentionally — we cannot tell from the file alone."
    )
    return warnings

  def _verify_high_level_description(self, hld, warnings: list[str]) -> None:
    if not isinstance(hld, str):
      warnings.append(
        f"high_level_description: expected a string, got {type(hld).__name__}"
      )

  def _verify_style_description(self, sd, warnings: list[str]) -> None:
    if not isinstance(sd, dict):
      warnings.append("style_description: expected a dict")
      return
    self._check_unknown_keys(
      sd, self.style_description_known_keys, "style_description", warnings
    )
    has_photo = "photo" in sd
    has_art_style = "art_style" in sd
    if has_photo and has_art_style:
      warnings.append(
        "style_description: contains both 'photo' and 'art_style'; expected exactly one"
      )
      return
    elif not has_photo and not has_art_style:
      warnings.append(
        "style_description: expected one of 'photo' (for photo captions) "
        "or 'art_style' (for non-photo captions)"
      )
      return

    self._check_key_order(
      sd,
      self._style_description_key_order(sd),
      "style_description",
      warnings,
    )

    if "color_palette" in sd:
      self._verify_color_palette(
        sd["color_palette"],
        "style_description.color_palette",
        self.style_description_palette_max,
        warnings,
      )

  def _verify_compositional_deconstruction(self, cd, warnings: list[str]) -> None:
    if not isinstance(cd, dict):
      warnings.append("compositional_deconstruction: expected a dict")
      return
    if "background" not in cd:
      warnings.append("compositional_deconstruction: 'background' must exist")
      return
    elif not isinstance(cd["background"], str):
      warnings.append(
        f"compositional_deconstruction.background: expected a string, "
        f"got {type(cd['background']).__name__}"
      )
      return
    if "elements" not in cd:
      warnings.append("compositional_deconstruction: 'elements' must exist")
      return
    self._check_key_order(
      cd,
      self.compositional_deconstruction_key_order,
      "compositional_deconstruction",
      warnings,
    )
    elements = cd["elements"]
    if not isinstance(elements, list):
      warnings.append("compositional_deconstruction.elements: expected a list")
      return
    for i, elem in enumerate(elements):
      self._verify_element(i, elem, warnings)

  def _verify_element(self, i: int, elem, warnings: list[str]) -> None:
    if not isinstance(elem, dict):
      warnings.append(f"elements[{i}]: expected a dict")
      return
    self._check_unknown_keys(elem, self.element_known_keys, f"elements[{i}]", warnings)

    if "type" not in elem:
      warnings.append(f"elements[{i}]: 'type' must exist")
      return
    elif elem.get("type", None) not in self.element_types:
      warnings.append(f"elements[{i}]: 'type' must be one of {self.element_types}")
      return

    self._check_key_order(
      elem, self._element_key_order(elem), f"elements[{i}]", warnings
    )

    if "bbox" in elem:
      self._verify_bbox(i, elem["bbox"], warnings)

    if "color_palette" in elem:
      self._verify_color_palette(
        elem["color_palette"],
        f"elements[{i}].color_palette",
        self.element_palette_max,
        warnings,
      )

  def _verify_color_palette(
    self, palette, path: str, max_colors: int, warnings: list[str]
  ) -> None:
    if not isinstance(palette, list):
      warnings.append(f"{path}: expected a list")
      return
    if len(palette) > max_colors:
      warnings.append(
        f"{path}: too many colors ({len(palette)}), expected at most {max_colors}"
      )
      return

    for i, color in enumerate(palette):
      if (
        not isinstance(color, str)
        or len(color) != 7
        or color[0] != "#"
        or not all(c in "0123456789ABCDEF" for c in color[1:])
      ):
        warnings.append(f"{path}[{i}]: '{color}' is not a valid #RRGGBB hex color")

  def _verify_bbox(self, i: int, bbox, warnings: list[str]) -> None:
    if not isinstance(bbox, list) or len(bbox) != 4:
      warnings.append(f"elements[{i}].bbox: expected [ymin, xmin, ymax, xmax]")
      return
    if not all(isinstance(v, int) for v in bbox):
      warnings.append(f"elements[{i}].bbox: all values must be int")
      return
    ymin, xmin, ymax, xmax = bbox
    if not all(self.bbox_min <= v <= self.bbox_max for v in bbox):
      warnings.append(
        f"elements[{i}].bbox: values must be in "
        f"[{self.bbox_min}, {self.bbox_max}], got {bbox}"
      )
    if ymin > ymax:
      warnings.append(f"elements[{i}].bbox: ymin ({ymin}) > ymax ({ymax})")
    if xmin > xmax:
      warnings.append(f"elements[{i}].bbox: xmin ({xmin}) > xmax ({xmax})")

  def _element_key_order(self, element: dict) -> Sequence[str]:
    elem_type = element.get("type", None)

    if elem_type == "text":
      element_key_order = self.element_key_order_text
    elif elem_type == "obj":
      element_key_order = self.element_key_order_obj
    else:
      raise ValueError

    element_key_order_ = []
    for key in element_key_order:
      if key == "bbox":
        if "bbox" in element:
          element_key_order_.append(key)
      elif key == "color_palette":
        if "color_palette" in element:
          element_key_order_.append(key)
      else:
        element_key_order_.append(key)

    return tuple(element_key_order_)

  def _style_description_key_order(self, sd: dict) -> Sequence[str]:
    has_photo = "photo" in sd
    has_art_style = "art_style" in sd
    if has_art_style and not has_photo:
      key_order = self.style_description_key_order_non_photo
    elif not has_art_style and has_photo:
      key_order = self.style_description_key_order_photo
    else:
      raise ValueError

    key_order_ = []
    for key in key_order:
      if key == "color_palette":
        if "color_palette" in sd:
          key_order_.append(key)
      else:
        key_order_.append(key)

    return tuple(key_order_)

  @staticmethod
  def _check_key_order(
    obj: dict,
    expected_order: Sequence[str],
    path: str,
    warnings: list[str],
  ) -> None:
    present_keys = tuple([k for k in obj if k in expected_order])
    if present_keys != tuple(expected_order):
      warnings.append(f"{path}: key order is {present_keys}, expected {expected_order}")
    extra = [k for k in obj if k not in expected_order]
    if extra:
      warnings.append(f"{path}: keys {extra} are not allowed in this context")

  @staticmethod
  def _check_unknown_keys(
    obj: dict,
    known: frozenset[str],
    path: str,
    warnings: list[str],
  ) -> None:
    unknown = [k for k in obj if k not in known]
    if unknown:
      warnings.append(f"{path}: unknown keys {unknown} (not in schema)")
