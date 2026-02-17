"""Grounding pipeline: bounding box + mask annotations for spatial grounding."""

from simpletuner.helpers.training.grounding.collate import GroundingCollate
from simpletuner.helpers.training.grounding.metadata import BboxMetadata
from simpletuner.helpers.training.grounding.types import BboxEntity, GroundingBatch

__all__ = ["BboxEntity", "BboxMetadata", "GroundingBatch", "GroundingCollate"]
