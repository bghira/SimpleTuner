"""MOSS-Image native layout captions use centered, absolute pixel coordinates."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from simpletuner.helpers.models.json_schema import JSONSchemaConverter


class MossImageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meta: list[str] | None = None
    canvas: Literal["single", "multi_panel", "collage"] = "single"
    caption: str = Field(min_length=1)
    palette: list[str] | None = None


class MossImageItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: Literal["object", "character", "text"]
    caption: str = Field(min_length=1)
    text: str | None = None
    bbox: list[int] = Field(min_length=4, max_length=4)
    tags: list[str] | None = None
    palette: list[str] | None = None

    @model_validator(mode="after")
    def validate_item(self):
        y_min, x_min, y_max, x_max = self.bbox
        if y_min >= y_max or x_min >= x_max:
            raise ValueError("MOSS-Image bounding boxes must have positive width and height.")
        if self.label == "text" and not self.text:
            raise ValueError("MOSS-Image text items require the literal visible text.")
        return self


class MossImageCaption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: MossImageSummary
    items: list[MossImageItem]
    bbox_coord_space: Literal["abs_centralized"] = "abs_centralized"
    bbox_format: Literal["yxyx"] = "yxyx"


class MossImageJSONSchemaConverter(JSONSchemaConverter):
    SCHEMA_ID = "moss-image-layout-v1"
    USES_PIXEL_COORDINATES = True
    CAPTION_MODEL = MossImageCaption
    GENERATION_INSTRUCTIONS = (
        "Describe the requested image using MOSS-Image's native summary/items layout JSON. "
        "Ground descriptions and bounding boxes in the supplied image when present. "
        "Use summary.caption for the complete scene, composition, medium, lighting and colors. "
        "Use one item per distinct character, object or visible text block. "
        "Preserve named subjects and concept tokens from the original caption, including emoji. "
        "Copy visible non-watermark text exactly into text items. Do not invent lettering or describe watermarks. "
        "Bounding boxes are integer [y_min,x_min,y_max,x_max] pixel coordinates, "
        "relative to the center of the specified output canvas, with x rightwards and y downwards. "
        "Set bbox_coord_space to abs_centralized and bbox_format to yxyx."
    )

    def generation_instructions(self, *, height: int, width: int) -> str:
        return super().generation_instructions(height=height, width=width) + (
            f"All x coordinates must be between {-width // 2} and {width // 2}; "
            f"all y coordinates must be between {-height // 2} and {height // 2}. "
        )

    def convert(self, prompt: str, *, height=None, width=None) -> str:
        if not prompt.strip():
            return prompt
        if prompt.lstrip().startswith("{"):
            caption = MossImageCaption.model_validate(json.loads(prompt))
        else:
            caption = MossImageCaption(summary=MossImageSummary(caption=prompt), items=[])
        for item in caption.items:
            y_min, x_min, y_max, x_max = item.bbox
            if height is not None and not (-height / 2 <= y_min < y_max <= height / 2):
                raise ValueError("MOSS-Image bounding box exceeds the output canvas height.")
            if width is not None and not (-width / 2 <= x_min < x_max <= width / 2):
                raise ValueError("MOSS-Image bounding box exceeds the output canvas width.")
        return self.serialize(caption.model_dump(exclude_none=True))
