import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_loss_fields(registry: "FieldRegistry") -> None:
    """Add loss function configuration fields."""
    # Loss Type
    registry._add_field(
        ConfigField(
            name="loss_type",
            arg_name="--loss_type",
            ui_label="Loss Function",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="l2",
            choices=[
                {"value": "l2", "label": "L2 (MSE)"},
                {"value": "huber", "label": "Huber"},
                {"value": "smooth_l1", "label": "Smooth L1"},
            ],
            help_text="How the model measures prediction errors. L2 (MSE) squares errors so large mistakes are penalized heavily. Huber/Smooth L1 treat large errors more gently, which can help when your dataset has unusual images.",
            tooltip="L2 is standard and works well for most cases. Huber/Smooth L1 reduce the influence of outliers (unusual samples) during training.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Huber Schedule
    registry._add_field(
        ConfigField(
            name="huber_schedule",
            arg_name="--huber_schedule",
            ui_label="Huber Schedule",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="snr",
            choices=[
                {"value": "snr", "label": "SNR (Default)"},
                {"value": "exponential", "label": "Exponential"},
                {"value": "constant", "label": "Constant"},
            ],
            dependencies=[FieldDependency(field="loss_type", operator="equals", value="huber")],
            help_text="Schedule for Huber loss transition threshold",
            tooltip="Controls how huber_c evolves during training. Only applies when using Huber loss.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Huber C Value
    registry._add_field(
        ConfigField(
            name="huber_c",
            arg_name="--huber_c",
            ui_label="Huber C Threshold",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="loss_type", operator="equals", value="huber")],
            help_text="Transition point between L2 and L1 regions for Huber loss",
            tooltip="Lower values emphasise L1 behaviour sooner; higher values behave more like L2.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # SNR Gamma
    registry._add_field(
        ConfigField(
            name="snr_gamma",
            arg_name="--snr_gamma",
            ui_label="SNR Gamma",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            order=4,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="SNR gamma must be non-negative")],
            help_text="SNR weighting gamma value (0 = disabled). Try 5 when using epsilon/V-prediction models.",
            tooltip="Rebalances loss across timesteps. Recommended value: 5.0 for epsilon and V-Prediction models to curb overemphasis on easy timesteps.",
            importance=ImportanceLevel.ADVANCED,
            documentation="OPTIONS.md#--snr_gamma",
        )
    )

    # Masked Loss Probability
    registry._add_field(
        ConfigField(
            name="masked_loss_probability",
            arg_name="--masked_loss_probability",
            ui_label="Masked Loss Probability",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of applying masked loss weighting per batch",
            tooltip="Lower values reduce how often masked loss is applied, useful for datasets with sparse masks.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # HiDream Load Balancing Loss Toggle
    registry._add_field(
        ConfigField(
            name="hidream_use_load_balancing_loss",
            arg_name="--hidream_use_load_balancing_loss",
            ui_label="Enable HiDream Load Balancing Loss",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="model_family", operator="equals", value="hidream")],
            help_text="Apply experimental load balancing loss when training HiDream models.",
            tooltip="Balances expert contributions during HiDream training. Only available for HiDream model family.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=6,
        )
    )

    # HiDream Load Balancing Weight
    registry._add_field(
        ConfigField(
            name="hidream_load_balancing_loss_weight",
            arg_name="--hidream_load_balancing_loss_weight",
            ui_label="HiDream Load Balancing Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="hidream_use_load_balancing_loss", operator="equals", value=True)],
            help_text="Strength multiplier for HiDream load balancing loss.",
            tooltip="Adjust if you need stronger balancing between experts. Leave blank to use the trainer default.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=7,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_enabled",
            arg_name="--crepa_enabled",
            ui_label="Enable CREPA",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            help_text="Enable Cross-frame Representation Alignment for video models.",
            tooltip="Adds a DINOv2-driven temporal alignment regularizer over intermediate hidden states.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=8,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_block_index",
            arg_name="--crepa_block_index",
            ui_label="CREPA Block Index",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=8,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Transformer block index to tap for hidden states (0-based).",
            tooltip="Use an encoder-side layer; earlier blocks capture spatial/temporal structure better.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=9,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_lambda",
            arg_name="--crepa_lambda",
            ui_label="CREPA Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.5,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Scaling factor applied to the CREPA alignment loss.",
            tooltip="Higher values increase regularisation strength; try 0.5–1.0.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=10,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_adjacent_distance",
            arg_name="--crepa_adjacent_distance",
            ui_label="CREPA Adjacent Distance",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be >= 0")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="How many neighbouring frames to include in cross-frame alignment.",
            tooltip="1 aligns to immediate neighbours; larger values expand the temporal window but add compute.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_adjacent_tau",
            arg_name="--crepa_adjacent_tau",
            ui_label="CREPA Temporal Decay",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.01, message="Must be > 0")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Exponential decay for neighbour weighting (smaller = faster decay).",
            tooltip="Controls how quickly similarity weighting drops for farther frames.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=12,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_encoder",
            arg_name="--crepa_encoder",
            ui_label="CREPA Vision Model",
            field_type=FieldType.TEXT,
            tab="training",
            section="loss_functions",
            default_value="dinov2_vitg14",
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Torch hub identifier for the frozen vision encoder (default: DINOv2 ViT-G/14, per CREPA paper).",
            tooltip='Passes directly to torch.hub.load("facebookresearch/dinov2", <id>); e.g., dinov2_vitg14 or dinov2_vits14.',
            importance=ImportanceLevel.EXPERIMENTAL,
            order=13,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_encoder_image_size",
            arg_name="--crepa_encoder_image_size",
            ui_label="CREPA Encoder Resolution",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=518,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=64, message="Must be >= 64")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Input resolution for the vision encoder preprocessing.",
            tooltip="Set to the pretrained encoder's default (518 for DINOv2-G/14; 224 for ViT-S/14).",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=14,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_drop_vae_encoder",
            arg_name="--crepa_drop_vae_encoder",
            ui_label="CREPA: Drop VAE Encoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Release VAE encoder/quant_conv after load to save memory when only decoding latents.",
            tooltip="Enable only if latents come from caches or elsewhere; encoding new pixels will no longer work.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=15,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_normalize_by_frames",
            arg_name="--crepa_normalize_by_frames",
            ui_label="CREPA: Normalize by Frames",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=True,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Divide alignment similarity by the number of frames to keep loss scale stable.",
            tooltip="Turn off to let longer clips contribute proportionally more alignment signal.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=16,
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_spatial_align",
            arg_name="--crepa_spatial_align",
            ui_label="CREPA: Spatial Token Align",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=True,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Interpolate patch tokens to match encoder/DiT token counts instead of global pooling.",
            tooltip="Disable to pool both sides before similarity if memory is tight.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=17,
        )
    )
