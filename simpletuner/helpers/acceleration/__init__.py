"""Acceleration backend infrastructure for memory optimization presets."""

from .backends import AccelerationBackend
from .preset import AccelerationPreset, get_sdnq_presets

__all__ = ["AccelerationBackend", "AccelerationPreset", "get_sdnq_presets"]
