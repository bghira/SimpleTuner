"""Prompt2Effect training utilities.

These scripts are intentionally kept out of the main SimpleTuner training path.
The workflow trains a LoRA-generating hypernetwork over existing LoRA files and
exports generated adapters as regular PEFT-compatible LoRA safetensors.
"""
