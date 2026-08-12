"""MageFlow — standalone text-to-image + image-edit inference."""

from .models.mage_flow import ModelConfig
from .pipeline import MageFlowPipeline, generate_edits, generate_images, load_from_repo

__all__ = [
    "MageFlowPipeline",
    "generate_images",
    "generate_edits",
    "load_from_repo",
    "ModelConfig",
]
__version__ = "0.1.0"
