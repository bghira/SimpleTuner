from simpletuner.helpers.training.optimizer_param import available_optimizer_keys as _available_optimizer_keys
from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldDependency,
    FieldType,
    ImportanceLevel,
    ValidationRule,
    ValidationRuleType,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="validation_lyrics",
            arg_name="--validation_lyrics",
            ui_label="Validation Lyrics",
            field_type=FieldType.TEXTAREA,
            tab="validation",
            section="prompt_management",
            placeholder="Enter lyrics for audio validation",
            help_text="Lyrics to use for audio validation",
            tooltip="Provide lyrics for music generation validation. Only used by audio models.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
            allow_empty=True,
            model_specific=["ace_step"],
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_audio_duration",
            arg_name="--validation_audio_duration",
            ui_label="Validation Audio Duration",
            field_type=FieldType.NUMBER,
            tab="validation",
            section="validation_schedule",
            default_value=30.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1.0, message="Duration must be at least 1 second"),
                ValidationRule(ValidationRuleType.MAX, value=300.0, message="Duration recommended to be under 300s"),
            ],
            help_text="Duration of generated audio for validation (seconds)",
            tooltip="Length of the audio clip to generate during validation runs.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
            model_specific=["ace_step"],
        )
    )

    optimizer_choices = _available_optimizer_keys()
    if not optimizer_choices:
        raise RuntimeError("No optimizers available for the current environment.")
    lr_scheduler_choices = [
        "linear",
        "sine",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ]

    registry._add_field(
        ConfigField(
            name="lyrics_embedder_train",
            arg_name="--lyrics_embedder_train",
            ui_label="Train Lyrics Embedder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="lyrics_embedder",
            default_value=False,
            help_text="Enable fine-tuning of the ACE-Step lyrics embedder components.",
            tooltip="Unlock lyric embedding layers for training. Recommended for ACE-Step only.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["ace_step"],
            order=1,
        )
    )

    registry._add_field(
        ConfigField(
            name="lyrics_embedder_optimizer",
            arg_name="--lyrics_embedder_optimizer",
            ui_label="Lyrics Embedder Optimizer",
            field_type=FieldType.SELECT,
            tab="training",
            section="lyrics_embedder",
            default_value=None,
            choices=[{"value": opt, "label": opt} for opt in optimizer_choices],
            dynamic_choices=True,
            validation_rules=[ValidationRule(ValidationRuleType.CHOICES, value=optimizer_choices)],
            dependencies=[FieldDependency(field="lyrics_embedder_train", operator="equals", value=True, action="show")],
            help_text="Optional optimizer override for the lyrics embedder (leave empty to reuse the main optimizer).",
            tooltip="Pick a different optimizer just for the lyrics embedder, or leave blank to share the primary one.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=["ace_step"],
            allow_empty=True,
            order=2,
        )
    )

    registry._add_field(
        ConfigField(
            name="lyrics_embedder_lr",
            arg_name="--lyrics_embedder_lr",
            ui_label="Lyrics Embedder Learning Rate",
            field_type=FieldType.NUMBER,
            tab="training",
            section="lyrics_embedder",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="lyrics_embedder_train", operator="equals", value=True, action="show")],
            help_text="Optional learning rate override for the lyrics embedder.",
            tooltip="Leave empty to share the main learning rate. Set a value to use a dedicated rate.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["ace_step"],
            allow_empty=True,
            order=3,
        )
    )

    registry._add_field(
        ConfigField(
            name="lyrics_embedder_lr_scheduler",
            arg_name="--lyrics_embedder_lr_scheduler",
            ui_label="Lyrics Embedder LR Scheduler",
            field_type=FieldType.SELECT,
            tab="training",
            section="lyrics_embedder",
            default_value=None,
            choices=[{"value": s, "label": s.replace("_", " ").title()} for s in lr_scheduler_choices],
            validation_rules=[ValidationRule(ValidationRuleType.CHOICES, value=lr_scheduler_choices)],
            dependencies=[FieldDependency(field="lyrics_embedder_train", operator="equals", value=True, action="show")],
            help_text="Select a scheduler for the lyrics embedder (leave empty to mirror the main scheduler).",
            tooltip="Use a distinct scheduler for lyric embeddings if needed, or leave blank to follow the primary plan.",
            importance=ImportanceLevel.EXPERIMENTAL,
            model_specific=["ace_step"],
            allow_empty=True,
            order=4,
        )
    )

    acestep_targets = [
        "attn_qkv",
        "attn_qkv+linear_qkv",
        "attn_qkv+linear_qkv+speech_embedder",
    ]
    registry._add_field(
        ConfigField(
            name="acestep_lora_target",
            arg_name="--acestep_lora_target",
            ui_label="ACE-Step LoRA Target Layers",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="model_specific",
            default_value="attn_qkv+linear_qkv",
            choices=[{"value": t, "label": t} for t in acestep_targets],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="model_family", value="ace_step"),
            ],
            help_text="Which layers to train in ACE-Step models",
            tooltip="'attn_qkv+linear_qkv' is default. '+speech_embedder' adds speaker embedding. 'attn_qkv' is minimal.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["ace_step"],
            order=11,
        )
    )
