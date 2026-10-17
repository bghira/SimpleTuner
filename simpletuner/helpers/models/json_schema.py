"""Model-owned JSON caption formats for text conditioning and prompt rewriting."""

import json
from abc import ABC, abstractmethod

from pydantic import BaseModel


class JSONSchemaConverter(ABC):
    SCHEMA_ID: str
    CAPTION_MODEL: type[BaseModel]
    CONVERT_NEGATIVE_PROMPTS = False
    USES_PIXEL_COORDINATES = False
    GENERATION_INSTRUCTIONS = ""

    @classmethod
    def json_schema(cls) -> dict:
        return cls.CAPTION_MODEL.model_json_schema()

    def generation_instructions(self, *, height: int, width: int) -> str:
        return self.GENERATION_INSTRUCTIONS + f" Output canvas: {width} pixels wide by {height} pixels high. "

    @staticmethod
    def serialize(caption: dict) -> str:
        return json.dumps(caption, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

    @abstractmethod
    def convert(self, prompt: str, *, height: int | None = None, width: int | None = None) -> str:
        """Convert a plain caption or canonicalize a model-native JSON caption."""
