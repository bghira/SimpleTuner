import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from simpletuner.helpers.models.registry import ModelRegistry

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
            help_text="Enable CREPA for transformer-based diffusion models (DiT-style). Use U-REPA for UNet models.",
            tooltip="Adds a DINOv2-driven alignment regularizer over intermediate transformer hidden states.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=8,
            documentation="OPTIONS.md#--crepa_enabled",
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
            documentation="OPTIONS.md#--crepa_block_index",
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
            documentation="OPTIONS.md#--crepa_lambda",
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
            documentation="OPTIONS.md#--crepa_adjacent_distance",
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
            documentation="OPTIONS.md#--crepa_adjacent_tau",
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
            documentation="OPTIONS.md#--crepa_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_encoder_frames_batch_size",
            arg_name="--crepa_encoder_frames_batch_size",
            ui_label="CREPA image encoder frames batch size",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=-1,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="How many frames the external feature encoder processes in parallel. Zero or negative for all frames of the whole batch simultaneously.",
            tooltip="Since DINO-like encoders are image models, they can process frames in sliced batches for lower VRAM usage at cost of speed.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=14,
            documentation="OPTIONS.md#--crepa_encoder_frames_batch_size",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_use_backbone_features",
            arg_name="--crepa_use_backbone_features",
            ui_label="CREPA: Use Backbone Features",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Align student/teacher transformer layers instead of an external vision encoder.",
            tooltip="Skips loading DINO when a stronger semantic layer exists inside the backbone.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=15,
            documentation="OPTIONS.md#--crepa_use_backbone_features",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_teacher_block_index",
            arg_name="--crepa_teacher_block_index",
            ui_label="CREPA Teacher Block",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Teacher block index when using backbone features (defaults to the student block).",
            tooltip="Pick a later/stronger layer to supervise the student when skipping DINO.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=16,
            documentation="OPTIONS.md#--crepa_teacher_block_index",
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
            order=17,
            documentation="OPTIONS.md#--crepa_encoder_image_size",
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
            order=18,
            documentation="OPTIONS.md#--crepa_drop_vae_encoder",
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
            order=19,
            documentation="OPTIONS.md#--crepa_normalize_by_frames",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_normalize_neighbour_sum",
            arg_name="--crepa_normalize_neighbour_sum",
            ui_label="CREPA: Normalize Neighbour Sum",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Normalize the neighbour-sum alignment by the per-frame weight sum (experimental).",
            tooltip="Keeps crepa_alignment_score within [-1,1] and applies the same normalization to the loss.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=19.5,
            documentation="OPTIONS.md#--crepa_normalize_neighbour_sum",
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
            order=20,
            documentation="OPTIONS.md#--crepa_spatial_align",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_use_tae",
            arg_name="--crepa_use_tae",
            ui_label="CREPA: Use Tiny AutoEncoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Use lightweight Tiny AutoEncoder instead of full VAE for frame decoding.",
            tooltip="Faster and uses less VRAM, but lower quality decoded frames. Only works for models with TAE support (e.g., WAN).",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=21,
            documentation="OPTIONS.md#--crepa_use_tae",
        )
    )

    # CREPA Scheduling Options
    registry._add_field(
        ConfigField(
            name="crepa_scheduler",
            arg_name="--crepa_scheduler",
            ui_label="CREPA Scheduler",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="constant",
            choices=[
                {"value": "constant", "label": "Constant"},
                {"value": "linear", "label": "Linear Decay"},
                {"value": "cosine", "label": "Cosine Decay"},
                {"value": "polynomial", "label": "Polynomial Decay"},
            ],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Schedule for CREPA coefficient decay over training. Constant keeps the weight fixed.",
            tooltip="Use decay schedules to reduce CREPA regularization strength as training progresses.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=22,
            documentation="OPTIONS.md#--crepa_scheduler",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_warmup_steps",
            arg_name="--crepa_warmup_steps",
            ui_label="CREPA Warmup Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Number of steps to linearly ramp CREPA weight from 0 to crepa_lambda.",
            tooltip="Gradual warmup can help stabilize early training before CREPA regularization kicks in.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=23,
            documentation="OPTIONS.md#--crepa_warmup_steps",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_decay_steps",
            arg_name="--crepa_decay_steps",
            ui_label="CREPA Decay Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[
                FieldDependency(field="crepa_enabled", operator="equals", value=True),
                FieldDependency(field="crepa_scheduler", operator="not_equals", value="constant"),
            ],
            help_text="Total steps for decay (after warmup). 0 means decay over entire training run.",
            tooltip="Controls the duration of the decay phase. Decay starts after warmup completes.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=24,
            documentation="OPTIONS.md#--crepa_decay_steps",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_lambda_end",
            arg_name="--crepa_lambda_end",
            ui_label="CREPA End Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[
                FieldDependency(field="crepa_enabled", operator="equals", value=True),
                FieldDependency(field="crepa_scheduler", operator="not_equals", value="constant"),
            ],
            help_text="Final CREPA weight after decay completes. 0 effectively disables CREPA at end of training.",
            tooltip="The coefficient decays from crepa_lambda to this value over decay_steps.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=25,
            documentation="OPTIONS.md#--crepa_lambda_end",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_power",
            arg_name="--crepa_power",
            ui_label="CREPA Polynomial Power",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.1, message="Must be > 0")],
            dependencies=[
                FieldDependency(field="crepa_enabled", operator="equals", value=True),
                FieldDependency(field="crepa_scheduler", operator="equals", value="polynomial"),
            ],
            help_text="Power factor for polynomial decay. 1.0 = linear, 2.0 = quadratic, etc.",
            tooltip="Higher values cause faster initial decay that slows down towards the end.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=26,
            documentation="OPTIONS.md#--crepa_power",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_cutoff_step",
            arg_name="--crepa_cutoff_step",
            ui_label="CREPA Cutoff Step",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Hard cutoff step after which CREPA is disabled. 0 means no step-based cutoff.",
            tooltip="Useful for disabling CREPA after model has converged on temporal alignment.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=27,
            documentation="OPTIONS.md#--crepa_cutoff_step",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_similarity_threshold",
            arg_name="--crepa_similarity_threshold",
            ui_label="CREPA Similarity Threshold",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="crepa_enabled", operator="equals", value=True)],
            help_text="Similarity EMA threshold at which CREPA cutoff triggers. Leave empty to disable.",
            tooltip="When the exponential moving average of similarity reaches this value, CREPA is disabled to prevent overfitting.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=28,
            documentation="OPTIONS.md#--crepa_similarity_threshold",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_similarity_ema_decay",
            arg_name="--crepa_similarity_ema_decay",
            ui_label="CREPA Similarity EMA Decay",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.99,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="crepa_similarity_threshold", operator="is_set", value=True)],
            help_text="Exponential moving average decay factor for similarity tracking. Higher = smoother.",
            tooltip="0.99 provides a ~100 step smoothing window. Lower values react faster to changes.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=29,
            documentation="OPTIONS.md#--crepa_similarity_ema_decay",
        )
    )

    registry._add_field(
        ConfigField(
            name="crepa_threshold_mode",
            arg_name="--crepa_threshold_mode",
            ui_label="CREPA Threshold Mode",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="permanent",
            choices=[
                {"value": "permanent", "label": "Permanent"},
                {"value": "recoverable", "label": "Recoverable"},
            ],
            dependencies=[FieldDependency(field="crepa_similarity_threshold", operator="is_set", value=True)],
            help_text="Behavior when similarity threshold is reached: permanent disables forever, recoverable allows re-enabling.",
            tooltip="Permanent: once threshold is hit, CREPA stays off. Recoverable: CREPA re-enables if similarity drops.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=30,
            documentation="OPTIONS.md#--crepa_threshold_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="twinflow_enabled",
            arg_name="--twinflow_enabled",
            ui_label="Enable TwinFlow (RCGM)",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            model_specific=[
                name
                for name, cls in ModelRegistry.model_families().items()
                if hasattr(cls, "PREDICTION_TYPE")
                and getattr(getattr(cls, "PREDICTION_TYPE"), "value", getattr(cls, "PREDICTION_TYPE")) == "flow_matching"
            ],
            dependencies=[
                FieldDependency(field="distillation_method", operator="equals", value=None),
                FieldDependency(field="scheduled_sampling_max_step_offset", operator="equals", value=0),
            ],
            help_text="Enable RCGM-based consistency training for few-step generation. Uses recursive consistency gradient matching to train models for 1-4 step inference with CFG baked in.",
            tooltip="Adds RCGM consistency loss for flow-matching models. Validation uses the target step count with zero CFG. Based on TwinFlow/RCGM paper.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=21,
            documentation="OPTIONS.md#--twinflow_enabled",
        )
    )

    registry._add_field(
        ConfigField(
            name="twinflow_target_step_count",
            arg_name="--twinflow_target_step_count",
            ui_label="TwinFlow Target Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, 1, "Must be at least 1 step"),
                ValidationRule(ValidationRuleType.MAX, 8, "Maximum 8 steps recommended"),
            ],
            model_specific=[
                name
                for name, cls in ModelRegistry.model_families().items()
                if hasattr(cls, "PREDICTION_TYPE")
                and getattr(getattr(cls, "PREDICTION_TYPE"), "value", getattr(cls, "PREDICTION_TYPE")) == "flow_matching"
            ],
            dependencies=[
                FieldDependency(field="twinflow_enabled", operator="equals", value=True),
            ],
            help_text="Target number of inference steps. Validation will run with this many steps using the UCGM sampler. Recommended: 1-4 NFE.",
            tooltip="1 = one-step generation, 2-4 = few-step generation (better quality). Validation uses zero CFG since guidance is baked in during training.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=22,
            documentation="OPTIONS.md#--twinflow_target_step_count",
        )
    )

    registry._add_field(
        ConfigField(
            name="layersync_enabled",
            arg_name="--layersync_enabled",
            ui_label="Enable LayerSync",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            help_text="Enable LayerSync self-alignment between two transformer blocks.",
            tooltip="Adds a cosine-similarity regularizer between student/teacher layers captured from the backbone.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=23,
            documentation="OPTIONS.md#--layersync_enabled",
        )
    )

    registry._add_field(
        ConfigField(
            name="layersync_student_block",
            arg_name="--layersync_student_block",
            ui_label="LayerSync Student Block",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="layersync_enabled", operator="equals", value=True)],
            help_text="Block index to treat as the student for LayerSync (1-based depths accepted).",
            tooltip="Pick an earlier/weaker layer to receive guidance; accepts paper-style 1-based depths.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=24,
            documentation="OPTIONS.md#--layersync_student_block",
        )
    )

    registry._add_field(
        ConfigField(
            name="layersync_teacher_block",
            arg_name="--layersync_teacher_block",
            ui_label="LayerSync Teacher Block",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="layersync_enabled", operator="equals", value=True)],
            help_text="Teacher block index; defaults to the student block when omitted (1-based depths accepted).",
            tooltip="Use a later/stronger layer to supervise the student; accepts paper-style 1-based depths.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=25,
            documentation="OPTIONS.md#--layersync_teacher_block",
        )
    )

    registry._add_field(
        ConfigField(
            name="layersync_lambda",
            arg_name="--layersync_lambda",
            ui_label="LayerSync Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.2,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="layersync_enabled", operator="equals", value=True)],
            help_text="Strength multiplier for LayerSync alignment loss.",
            tooltip="Set >0 to activate LayerSync when enabled.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=26,
            documentation="OPTIONS.md#--layersync_lambda",
        )
    )

    # U-REPA (Universal REPA for UNet models)
    registry._add_field(
        ConfigField(
            name="urepa_enabled",
            arg_name="--urepa_enabled",
            ui_label="Enable U-REPA",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            model_specific=["sdxl", "sd1x", "sd2x", "kolors"],
            help_text="Enable Universal REPA for UNet-based diffusion models (SDXL, SD1.5, Kolors).",
            tooltip="Adds DINOv2-driven representation alignment with manifold loss, optimized for UNet architectures.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=40,
            documentation="OPTIONS.md#--urepa_enabled",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_lambda",
            arg_name="--urepa_lambda",
            ui_label="U-REPA Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.5,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Scaling factor for U-REPA alignment loss. Paper recommends 0.5.",
            tooltip="Higher values increase regularization strength; 0.5 is the paper default.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=41,
            documentation="OPTIONS.md#--urepa_lambda",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_manifold_weight",
            arg_name="--urepa_manifold_weight",
            ui_label="U-REPA Manifold Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=3.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Weight for manifold loss relative to cosine alignment. Paper recommends w=3.0.",
            tooltip="Manifold loss aligns relative similarity structure; higher values prioritize it over direct alignment.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=42,
            documentation="OPTIONS.md#--urepa_manifold_weight",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_model",
            arg_name="--urepa_model",
            ui_label="U-REPA Vision Model",
            field_type=FieldType.TEXT,
            tab="training",
            section="loss_functions",
            default_value="dinov2_vitg14",
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Torch hub identifier for the frozen vision encoder (default: DINOv2 ViT-G/14).",
            tooltip='Passes directly to torch.hub.load("facebookresearch/dinov2", <id>); e.g., dinov2_vitg14 or dinov2_vits14.',
            importance=ImportanceLevel.EXPERIMENTAL,
            order=43,
            documentation="OPTIONS.md#--urepa_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_encoder_image_size",
            arg_name="--urepa_encoder_image_size",
            ui_label="U-REPA Encoder Resolution",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=518,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=64, message="Must be >= 64")],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Input resolution for the vision encoder preprocessing.",
            tooltip="Set to the pretrained encoder's default (518 for DINOv2-G/14; 224 for ViT-S/14).",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=44,
            documentation="OPTIONS.md#--urepa_encoder_image_size",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_use_tae",
            arg_name="--urepa_use_tae",
            ui_label="U-REPA: Use Tiny AutoEncoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            default_value=False,
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Use lightweight Tiny AutoEncoder instead of full VAE for decoding.",
            tooltip="Faster and uses less VRAM, but lower quality decoded images.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=45,
            documentation="OPTIONS.md#--urepa_use_tae",
        )
    )

    # U-REPA Scheduling Options
    registry._add_field(
        ConfigField(
            name="urepa_scheduler",
            arg_name="--urepa_scheduler",
            ui_label="U-REPA Scheduler",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="constant",
            choices=[
                {"value": "constant", "label": "Constant"},
                {"value": "linear", "label": "Linear Decay"},
                {"value": "cosine", "label": "Cosine Decay"},
                {"value": "polynomial", "label": "Polynomial Decay"},
            ],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Schedule for U-REPA coefficient decay over training.",
            tooltip="Use decay schedules to reduce U-REPA regularization as training progresses.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=46,
            documentation="OPTIONS.md#--urepa_scheduler",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_warmup_steps",
            arg_name="--urepa_warmup_steps",
            ui_label="U-REPA Warmup Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Number of steps to linearly ramp U-REPA weight from 0 to urepa_lambda.",
            tooltip="Gradual warmup can help stabilize early training.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=47,
            documentation="OPTIONS.md#--urepa_warmup_steps",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_decay_steps",
            arg_name="--urepa_decay_steps",
            ui_label="U-REPA Decay Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[
                FieldDependency(field="urepa_enabled", operator="equals", value=True),
                FieldDependency(field="urepa_scheduler", operator="not_equals", value="constant"),
            ],
            help_text="Total steps for decay (after warmup). 0 means decay over entire training run.",
            tooltip="Controls the duration of the decay phase. Decay starts after warmup completes.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=48,
            documentation="OPTIONS.md#--urepa_decay_steps",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_lambda_end",
            arg_name="--urepa_lambda_end",
            ui_label="U-REPA End Weight",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[
                FieldDependency(field="urepa_enabled", operator="equals", value=True),
                FieldDependency(field="urepa_scheduler", operator="not_equals", value="constant"),
            ],
            help_text="Final U-REPA weight after decay completes. 0 effectively disables U-REPA at end of training.",
            tooltip="The coefficient decays from urepa_lambda to this value over decay_steps.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=49,
            documentation="OPTIONS.md#--urepa_lambda_end",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_power",
            arg_name="--urepa_power",
            ui_label="U-REPA Polynomial Power",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.1, message="Must be > 0")],
            dependencies=[
                FieldDependency(field="urepa_enabled", operator="equals", value=True),
                FieldDependency(field="urepa_scheduler", operator="equals", value="polynomial"),
            ],
            help_text="Power factor for polynomial decay. 1.0 = linear, 2.0 = quadratic, etc.",
            tooltip="Higher values cause faster initial decay that slows down towards the end.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=50,
            documentation="OPTIONS.md#--urepa_power",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_cutoff_step",
            arg_name="--urepa_cutoff_step",
            ui_label="U-REPA Cutoff Step",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Hard cutoff step after which U-REPA is disabled. 0 = no cutoff.",
            tooltip="Useful for disabling U-REPA after the model has learned good representations.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=51,
            documentation="OPTIONS.md#--urepa_cutoff_step",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_similarity_threshold",
            arg_name="--urepa_similarity_threshold",
            ui_label="U-REPA Similarity Threshold",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="urepa_enabled", operator="equals", value=True)],
            help_text="Similarity EMA threshold at which U-REPA cutoff triggers. Leave empty to disable.",
            tooltip="When the exponential moving average of similarity reaches this value, U-REPA is disabled to prevent overfitting.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=52,
            documentation="OPTIONS.md#--urepa_similarity_threshold",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_similarity_ema_decay",
            arg_name="--urepa_similarity_ema_decay",
            ui_label="U-REPA Similarity EMA Decay",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            default_value=0.99,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            dependencies=[FieldDependency(field="urepa_similarity_threshold", operator="is_set", value=True)],
            help_text="Exponential moving average decay factor for similarity tracking. Higher = smoother.",
            tooltip="0.99 provides a ~100 step smoothing window. Lower values react faster to changes.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=53,
            documentation="OPTIONS.md#--urepa_similarity_ema_decay",
        )
    )

    registry._add_field(
        ConfigField(
            name="urepa_threshold_mode",
            arg_name="--urepa_threshold_mode",
            ui_label="U-REPA Threshold Mode",
            field_type=FieldType.SELECT,
            tab="training",
            section="loss_functions",
            default_value="permanent",
            choices=[
                {"value": "permanent", "label": "Permanent"},
                {"value": "recoverable", "label": "Recoverable"},
            ],
            dependencies=[FieldDependency(field="urepa_similarity_threshold", operator="is_set", value=True)],
            help_text="Behavior when similarity threshold is reached: permanent disables forever, recoverable allows re-enabling.",
            tooltip="Permanent: once threshold is hit, U-REPA stays off. Recoverable: U-REPA re-enables if similarity drops.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=54,
            documentation="OPTIONS.md#--urepa_threshold_mode",
        )
    )
