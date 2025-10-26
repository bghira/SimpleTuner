"""Configuration rules for dataloader validation."""

from typing import List

from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_default_rule,
    make_required_rule,
)
from simpletuner.helpers.data_backend.dataset_types import DatasetType


def register_dataloader_rules():
    """Register validation rules for dataloader configuration."""
    rules = [
        make_required_rule(
            field_name="data_backend_config",
            message="Dataloader configuration file must be specified",
            example='data_backend_config: "config/multidatabackend.json"',
            suggestion="Create a multidatabackend.json file with your dataset configuration",
        ),
        ConfigRule(
            field_name="datasets",
            rule_type=RuleType.COMBINATION,
            value={DatasetType.IMAGE.value: 1, DatasetType.TEXT_EMBEDS.value: 1},
            message="Your dataloader config must contain at least one image dataset AND at least one text_embed dataset",
            example="""[
    {
        "id": "my-images",
        "type": "local",
        "instance_data_dir": "/path/to/images",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embeds",
        "dataset_type": "text_embeds",
        "type": "local",
        "cache_dir": "cache/text",
        "default": true
    }
]""",
            suggestion="See https://github.com/bghira/SimpleTuner/blob/main/documentation/DATALOADER.md#configuration-options",
        ),
        ConfigRule(
            field_name="text_embed_default",
            rule_type=RuleType.CUSTOM,
            value=None,
            message="Multiple text_embed datasets require one marked as default",
            condition=lambda cfg: _has_multiple_text_embeds(cfg),
            example='"default": true  # Add to one text_embed dataset',
            suggestion="Set default: true on one of the text_embed datasets",
        ),
        make_default_rule(
            field_name="caption_dropout_probability",
            default_value=0.1,
            message="Caption dropout helps prevent overfitting and improves CFG performance",
            example="caption_dropout_probability: 0.1",
        ),
        make_default_rule(
            field_name="metadata_update_interval",
            default_value=65,
            message="How often to update dataset metadata (in steps)",
            example="metadata_update_interval: 65",
        ),
        ConfigRule(
            field_name="vae_cache_clear_each_epoch",
            rule_type=RuleType.DEFAULT,
            value=False,
            message="Whether to clear VAE cache after each epoch",
            example="vae_cache_clear_each_epoch: false",
            error_level="info",
        ),
    ]

    ConfigRegistry.register_rules("dataloader", rules)
    ConfigRegistry.register_validator(
        "dataloader",
        _validate_dataloader_specific,
        """Validates dataloader-specific requirements:
- Ensures image and text_embed datasets are present
- Validates default text_embed selection
- Checks caption dropout configuration
- Validates dataset backend types""",
    )


def _has_multiple_text_embeds(config: dict) -> bool:
    """Check if config has multiple text embed datasets."""
    datasets = config.get("datasets", [])
    if not isinstance(datasets, list):
        return False

    text_embed_count = sum(1 for d in datasets if d.get("dataset_type") == "text_embeds")
    return text_embed_count > 1


def _validate_dataloader_specific(config: dict) -> List[ValidationResult]:
    """Custom validation for dataloader configuration."""
    results = []

    # Check if caption dropout is disabled
    if config.get("caption_dropout_probability", 0.1) == 0.0:
        results.append(
            ValidationResult(
                passed=False,
                field="caption_dropout_probability",
                message="Not using caption dropout will potentially lead to overfitting on captions, eg. CFG will not work very well",
                level="warning",
                suggestion="Set --caption_dropout_probability=0.1 as a recommended value",
            )
        )

    # Check for multiple text embeds without default
    datasets = config.get("datasets", [])
    if isinstance(datasets, list):
        text_embeds = [d for d in datasets if d.get("dataset_type") == "text_embeds"]
        if len(text_embeds) > 1:
            defaults = [d for d in text_embeds if d.get("default", False)]
            if len(defaults) == 0:
                results.append(
                    ValidationResult(
                        passed=False,
                        field="datasets",
                        message=f"You have {len(text_embeds)} text_embed datasets, but no default text embed was defined",
                        level="error",
                        suggestion="Please set default: true on one of the text_embed datasets",
                    )
                )
            elif len(defaults) > 1:
                results.append(
                    ValidationResult(
                        passed=False,
                        field="datasets",
                        message="Multiple text_embed datasets are marked as default",
                        level="error",
                        suggestion="Only one text_embed dataset should have default: true",
                    )
                )

    return results


# Register dataloader rules when module is imported
register_dataloader_rules()
