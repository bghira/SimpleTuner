"""Compatibility imports for the SimpleTuner RVC transform API."""

from simpletuner.helpers.rvc.runtime import configure_rvc_runtime

configure_rvc_runtime()

from huggingface_hub_rvc._runtime import (
    RVCRecord,
    SimpleRVCArtifact,
    SimpleRVCConverter,
    SimpleRVCTrainer,
    _load_model_payload,
)

__all__ = [
    "RVCRecord",
    "SimpleRVCArtifact",
    "SimpleRVCConverter",
    "SimpleRVCTrainer",
    "_load_model_payload",
]
