"""Configuration rules for dataloader validation."""

from typing import Any, Dict, List, Optional

from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_default_rule,
    make_required_rule,
)
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.distillation.requirements import (
    EMPTY_PROFILE,
    DistillerRequirementProfile,
    describe_requirement_groups,
    evaluate_requirement_profile,
)


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
- Ensures required training datasets and text_embeds are present (respecting distiller requirements)
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

    datasets = config.get("datasets", [])
    profile = _resolve_distiller_profile(config)
    relax_training_requirement = _relaxes_training_requirement(profile)
    requirement_eval = None

    if profile:
        requirement_eval = evaluate_requirement_profile(profile, datasets)
        if not requirement_eval.fulfilled:
            method_label = (_distillation_method_from_config(config) or "distiller").replace("_", " ")
            results.append(
                ValidationResult(
                    passed=False,
                    field="datasets",
                    message=f"{method_label} requires datasets matching "
                    f"{describe_requirement_groups(requirement_eval.missing_requirements)}",
                    level="error",
                )
            )

    if isinstance(datasets, list):
        dataset_types = [_dataset_type_from_entry(dataset) for dataset in datasets if isinstance(dataset, dict)]
        has_image_dataset = any(dataset_type is DatasetType.IMAGE for dataset_type in dataset_types)
        text_embeds = [d for d in datasets if d.get("dataset_type") == "text_embeds"]

        missing_chunks = []
        if not relax_training_requirement and not has_image_dataset:
            missing_chunks.append("an image dataset")
        if len(text_embeds) == 0:
            missing_chunks.append("a text_embed dataset")

        if missing_chunks:
            if len(missing_chunks) == 1:
                message_detail = missing_chunks[0]
            else:
                message_detail = " AND ".join(missing_chunks)
            results.append(
                ValidationResult(
                    passed=False,
                    field="datasets",
                    message=f"Your dataloader config must contain at least {message_detail}",
                    level="error",
                    suggestion="See documentation/DATALOADER.md for dataset configuration examples.",
                )
            )

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


def _dataset_type_from_entry(dataset: Dict[str, Any]) -> DatasetType:
    raw_value = dataset.get("dataset_type")
    try:
        return DatasetType.from_value(raw_value, default=DatasetType.IMAGE)
    except ValueError:
        return DatasetType.IMAGE


def _distillation_method_from_config(config: dict) -> Optional[str]:
    for key in ("distillation_method", "--distillation_method"):
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_distiller_profile(config: dict) -> DistillerRequirementProfile:
    method = _distillation_method_from_config(config)
    if not method:
        return EMPTY_PROFILE
    return DistillationRegistry.get_requirement_profile(method)


def _relaxes_training_requirement(profile: DistillerRequirementProfile) -> bool:
    if not profile:
        return False
    if not profile.is_data_generator:
        return False
    return not any(profile.requires_dataset_type(dataset_type) for dataset_type in (DatasetType.IMAGE, DatasetType.VIDEO))


# Register dataloader rules when module is imported
register_dataloader_rules()
