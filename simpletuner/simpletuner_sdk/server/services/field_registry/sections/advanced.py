import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_advanced_fields(registry: "FieldRegistry") -> None:
    """Add advanced configuration fields."""
    # Danger Mode Toggle
    registry._add_field(
        ConfigField(
            name="i_know_what_i_am_doing",
            arg_name="--i_know_what_i_am_doing",
            ui_label="I Know What I'm Doing",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="project",
            subsection="advanced",
            default_value=False,
            help_text="Unlock experimental overrides and bypass built-in safety limits.",
            tooltip="Only enable if you understand the implications. Required for editing prediction type and other safeguards.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=0,
        )
    )

    # Flow Matching Configuration
    registry._add_field(
        ConfigField(
            name="flow_sigmoid_scale",
            arg_name="--flow_sigmoid_scale",
            ui_label="Flow Sigmoid Scale",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Scale factor for sigmoid timestep sampling for flow-matching models.",
            tooltip="Adjusts the sigmoid curve for timestep sampling. Higher values may affect training dynamics.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_fast_schedule",
            arg_name="--flux_fast_schedule",
            ui_label="Flow Fast Schedule",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            help_text="Use experimental fast schedule for Flux.1S training",
            tooltip="Experimental feature that may improve training speed for Flux.1S models.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_use_uniform_schedule",
            arg_name="--flow_use_uniform_schedule",
            ui_label="Use Uniform Schedule",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=False,
            help_text="Use uniform schedule instead of sigmoid for flow-matching",
            tooltip="May cause bias toward dark images. Use with caution.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_use_beta_schedule",
            arg_name="--flow_use_beta_schedule",
            ui_label="Use Beta Schedule",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=False,
            help_text="Use beta schedule instead of sigmoid for flow-matching",
            tooltip="Alternative to sigmoid scheduling. May affect training dynamics.",
            importance=ImportanceLevel.ADVANCED,
            order=22,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_beta_schedule_alpha",
            arg_name="--flow_beta_schedule_alpha",
            ui_label="Beta Schedule Alpha",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=2.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Alpha value for beta schedule (default: 2.0)",
            tooltip="Controls the shape of the beta distribution curve.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_beta_schedule_beta",
            arg_name="--flow_beta_schedule_beta",
            ui_label="Beta Schedule Beta",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=2.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Beta value for beta schedule (default: 2.0)",
            tooltip="Controls the shape of the beta distribution curve.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_schedule_shift",
            arg_name="--flow_schedule_shift",
            ui_label="Schedule Shift",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=3.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Shift the noise schedule for flow-matching models",
            tooltip="Affects contrast/brightness learning. Higher values focus on composition, lower on fine details.",
            importance=ImportanceLevel.ADVANCED,
            order=25,
        )
    )

    registry._add_field(
        ConfigField(
            name="flow_schedule_auto_shift",
            arg_name="--flow_schedule_auto_shift",
            ui_label="Auto Shift Schedule",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=False,
            help_text="Auto-adjust schedule shift based on image resolution",
            tooltip="Automatically calculates optimal shift for different resolutions. May require learning rate adjustment.",
            importance=ImportanceLevel.ADVANCED,
            order=26,
        )
    )

    # Flux Guidance Configuration
    registry._add_field(
        ConfigField(
            name="flux_guidance_mode",
            arg_name="--flux_guidance_mode",
            ui_label="Flux Guidance Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            default_value="constant",
            choices=[{"value": "constant", "label": "Constant"}, {"value": "random-range", "label": "Random Range"}],
            help_text="Guidance mode for Flux training",
            tooltip="Constant uses same guidance for all samples. Random Range varies guidance per sample.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    # Flux Attention Masked Training
    registry._add_field(
        ConfigField(
            name="flux_attention_masked_training",
            arg_name="--flux_attention_masked_training",
            ui_label="Attention Masked Training",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            model_specific=["flux"],
            help_text="Enable attention masked training for Flux models",
            tooltip="Experimental feature for Flux models that masks certain attention patterns during training.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=10,
        )
    )

    # Flux Fast Schedule
    registry._add_field(
        ConfigField(
            name="flux_fast_schedule",
            arg_name="--flux_fast_schedule",
            ui_label="Fast Training Schedule",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            default_value=False,
            model_specific=["flux"],
            help_text="Use experimental fast schedule for Flux training",
            tooltip="Experimental feature that may speed up Flux.1S training at the cost of quality.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_value",
            arg_name="--flux_guidance_value",
            ui_label="Flux Guidance Value",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Guidance value for constant mode",
            tooltip="1.0 preserves CFG distillation. Higher values require CFG at inference.",
            importance=ImportanceLevel.ADVANCED,
            order=41,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_min",
            arg_name="--flux_guidance_min",
            ui_label="Flux Guidance Min",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=0.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Minimum guidance value for random-range mode",
            tooltip="Lower bound of guidance range when using random-range mode.",
            importance=ImportanceLevel.ADVANCED,
            order=42,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="flux_guidance_max",
            arg_name="--flux_guidance_max",
            ui_label="Flux Guidance Max",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            default_value=4.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Maximum guidance value for random-range mode",
            tooltip="Upper bound of guidance range when using random-range mode.",
            importance=ImportanceLevel.ADVANCED,
            order=43,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
        )
    )

    # T5 Configuration
    registry._add_field(
        ConfigField(
            name="t5_padding",
            arg_name="--t5_padding",
            ui_label="T5 Padding",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value="unmodified",
            choices=[{"value": "zero", "label": "Zero"}, {"value": "unmodified", "label": "Unmodified"}],
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="Padding behavior for T5 text encoder",
            tooltip="Zero pads with zeros. Unmodified leaves original padding. Affects model behavior.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
        )
    )

    registry._add_field(
        ConfigField(
            name="sd3_clip_uncond_behaviour",
            arg_name="--sd3_clip_uncond_behaviour",
            ui_label="SD3 CLIP Unconditional Behavior",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value="empty_string",
            choices=[{"value": "empty_string", "label": "Empty String"}, {"value": "zero", "label": "Zero"}],
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="How SD3 handles unconditional prompts",
            tooltip="Affects how SD3 processes prompts without conditioning. Empty string is default.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    registry._add_field(
        ConfigField(
            name="sd3_t5_uncond_behaviour",
            arg_name="--sd3_t5_uncond_behaviour",
            ui_label="SD3 T5 Unconditional Behavior",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=None,
            choices=[{"value": "empty_string", "label": "Empty String"}, {"value": "zero", "label": "Zero"}],
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="How SD3 T5 handles unconditional prompts",
            tooltip="Overrides CLIP behavior for T5. If not set, follows CLIP setting.",
            importance=ImportanceLevel.ADVANCED,
            order=12,
        )
    )

    # Soft Min SNR Configuration
    registry._add_field(
        ConfigField(
            name="soft_min_snr_sigma_data",
            arg_name="--soft_min_snr_sigma_data",
            ui_label="Soft Min SNR Sigma Data",
            field_type=FieldType.NUMBER,
            tab="training",
            section="loss_functions",
            subsection="advanced",
            default_value=None,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            help_text="Sigma data for soft min SNR weighting",
            tooltip="Required when using soft min SNR. Affects loss calculation.",
            importance=ImportanceLevel.ADVANCED,
            order=27,
        )
    )

    # Mixed Precision
    registry._add_field(
        ConfigField(
            name="mixed_precision",
            arg_name="--mixed_precision",
            ui_label="Mixed Precision",
            field_type=FieldType.SELECT,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value="bf16",
            choices=[
                {"value": "no", "label": "No (FP32)"},
                {"value": "fp16", "label": "FP16"},
                {"value": "bf16", "label": "BF16 (Recommended)"},
                {"value": "fp8", "label": "FP8 (Experimental)"},
            ],
            help_text="Precision for training computations",
            tooltip="BF16 is recommended for stability. FP16 saves memory but less stable. FP8 is experimental.",
            importance=ImportanceLevel.IMPORTANT,
            order=10,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
        )
    )

    # Attention Mechanism
    attention_mechanisms = [
        "diffusers",
        "xformers",
        "sageattention",
        "sageattention-int8-fp16-triton",
        "sageattention-int8-fp16-cuda",
        "sageattention-int8-fp8-cuda",
    ]
    registry._add_field(
        ConfigField(
            name="attention_mechanism",
            arg_name="--attention_mechanism",
            ui_label="Attention Implementation",
            field_type=FieldType.SELECT,
            tab="model",
            section="memory_optimization",
            subsection="advanced",
            default_value="diffusers",
            choices=[{"value": a, "label": a} for a in attention_mechanisms],
            help_text="Attention computation backend",
            tooltip="Xformers saves memory. SageAttention is faster but experimental. Diffusers is default.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
        )
    )

    # SageAttention Usage
    registry._add_field(
        ConfigField(
            name="sageattention_usage",
            arg_name="--sageattention_usage",
            ui_label="SageAttention Usage",
            field_type=FieldType.SELECT,
            tab="model",
            section="memory_optimization",
            subsection="advanced",
            default_value="inference",
            choices=[
                {"value": "training", "label": "Training"},
                {"value": "inference", "label": "Inference"},
                {"value": "training+inference", "label": "Training + Inference"},
            ],
            help_text="When to use SageAttention",
            tooltip="SageAttention breaks gradient tracking. Use only for inference unless you understand the implications.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    # Disable TF32
    registry._add_field(
        ConfigField(
            name="disable_tf32",
            arg_name="--disable_tf32",
            ui_label="Disable TF32",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value=False,
            platform_specific=["cuda"],
            help_text="Disable TF32 precision on Ampere GPUs",
            tooltip="TF32 is enabled by default on RTX 3000/4000 series. Disabling may reduce performance but increase precision.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    # Set Grads to None
    registry._add_field(
        ConfigField(
            name="set_grads_to_none",
            arg_name="--set_grads_to_none",
            ui_label="Set Gradients to None",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value=False,
            help_text="Set gradients to None instead of zero",
            tooltip="Can save memory and improve performance. May cause issues with some optimizers.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=12,
        )
    )

    # Noise Offset
    registry._add_field(
        ConfigField(
            name="noise_offset",
            arg_name="--noise_offset",
            ui_label="Noise Offset",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="noise_settings",
            default_value=0.1,
            help_text="Fine-tuning against a modified noise",
            tooltip="See: https://www.crosslabs.org//blog/diffusion-with-offset-noise for more information.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Noise Offset Probability
    registry._add_field(
        ConfigField(
            name="noise_offset_probability",
            arg_name="--noise_offset_probability",
            ui_label="Noise Offset Probability",
            field_type=FieldType.NUMBER,
            tab="training",
            section="noise_settings",
            default_value=0.25,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of applying noise offset",
            tooltip="Apply noise offset this fraction of the time. Default: 25%",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Input Perturbation
    registry._add_field(
        ConfigField(
            name="input_perturbation",
            arg_name="--input_perturbation",
            ui_label="Input Perturbation",
            field_type=FieldType.NUMBER,
            tab="training",
            section="noise_settings",
            default_value=0.0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Add additional noise only to the inputs fed to the model during training",
            tooltip="Add additional noise only to the inputs fed to the model during training. This will make the training converge faster. A value of 0.1 is suggested if you want to enable this. Input perturbation seems to also work with flow-matching (e.g., SD3 and Flux).",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Input Perturbation Steps
    registry._add_field(
        ConfigField(
            name="input_perturbation_steps",
            arg_name="--input_perturbation_steps",
            ui_label="Input Perturbation Steps",
            field_type=FieldType.NUMBER,
            tab="training",
            section="noise_settings",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Only apply input perturbation over the first N steps with linear decay",
            tooltip="This should prevent artifacts from showing up in longer training runs. This should prevent artifacts from showing up in longer training runs.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # LR End
    registry._add_field(
        ConfigField(
            name="lr_end",
            arg_name="--lr_end",
            ui_label="Learning Rate End",
            field_type=FieldType.TEXT,
            tab="training",
            section="learning_rate",
            default_value="4e-7",
            help_text="A polynomial learning rate will end up at this value after the specified number of warmup steps",
            tooltip="A sine or cosine wave will use this value as its lower bound for the learning rate.",
            importance=ImportanceLevel.ADVANCED,
            order=27,
        )
    )

    # LR Scale
    registry._add_field(
        ConfigField(
            name="lr_scale",
            arg_name="--lr_scale",
            ui_label="Scale Learning Rate",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="learning_rate",
            subsection="advanced",
            default_value=False,
            help_text="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size",
            tooltip="When using multiple GPUs, gradient accumulation steps, and batch size, the learning rate may need to be scaled. This option will automatically scale the learning rate.",
            importance=ImportanceLevel.ADVANCED,
            order=28,
        )
    )

    # LR Scale Square Root
    registry._add_field(
        ConfigField(
            name="lr_scale_sqrt",
            arg_name="--lr_scale_sqrt",
            ui_label="Scale Learning Rate by Square Root",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="learning_rate",
            subsection="advanced",
            default_value=False,
            help_text="If using --lr-scale, use the square root of (number of GPUs * gradient accumulation steps * batch size)",
            tooltip="If using --lr-scale, use the square root of (number of GPUs * gradient accumulation steps * batch size).",
            importance=ImportanceLevel.ADVANCED,
            order=29,
        )
    )

    # Ignore Final Epochs
    registry._add_field(
        ConfigField(
            name="ignore_final_epochs",
            arg_name="--ignore_final_epochs",
            ui_label="Ignore Final Epochs",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="training_schedule",
            default_value=False,
            help_text="When provided, the max epoch counter will not determine the end of the training run",
            tooltip="Instead, it will end when it hits --max_train_steps.",
            importance=ImportanceLevel.ADVANCED,
            order=30,
        )
    )

    # Freeze Encoder Before
    registry._add_field(
        ConfigField(
            name="freeze_encoder_before",
            arg_name="--freeze_encoder_before",
            ui_label="Freeze Encoder Before Layer",
            field_type=FieldType.NUMBER,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=12,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="When using 'before' strategy, we will freeze layers earlier than this",
            tooltip="When freezing the text encoder, we can use the 'before', 'between', or 'after' strategy. The 'before' strategy will freeze layers earlier than this.",
            importance=ImportanceLevel.ADVANCED,
            order=32,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Freeze Encoder After
    registry._add_field(
        ConfigField(
            name="freeze_encoder_after",
            arg_name="--freeze_encoder_after",
            ui_label="Freeze Encoder After Layer",
            field_type=FieldType.NUMBER,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=17,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="When using 'after' strategy, we will freeze layers later than this",
            tooltip="When freezing the text encoder, we can use the 'before', 'between', or 'after' strategy. The 'after' strategy will freeze all layers from 17 up.",
            importance=ImportanceLevel.ADVANCED,
            order=33,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Freeze Encoder Strategy
    registry._add_field(
        ConfigField(
            name="freeze_encoder_strategy",
            arg_name="--freeze_encoder_strategy",
            ui_label="Freeze Encoder Strategy",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value="after",
            choices=[
                {"value": "before", "label": "Before"},
                {"value": "between", "label": "Between"},
                {"value": "after", "label": "After"},
            ],
            help_text="When freezing the text encoder, we can use the 'before', 'between', or 'after' strategy",
            tooltip="The 'before' strategy will freeze layers earlier than this. The 'between' strategy will freeze layers between those two values, leaving the outer layers unfrozen. The 'after' strategy will freeze all layers from 17 up. This can be helpful when fine-tuning Stable Diffusion 2.1 on a new style.",
            importance=ImportanceLevel.ADVANCED,
            order=34,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Layer Freeze Strategy
    registry._add_field(
        ConfigField(
            name="layer_freeze_strategy",
            arg_name="--layer_freeze_strategy",
            ui_label="Layer Freeze Strategy",
            field_type=FieldType.SELECT,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=None,
            choices=[
                {"value": "none", "label": "None"},
                {"value": "bitfit", "label": "BitFit"},
            ],
            help_text="When freezing parameters, we can use the 'none' or 'bitfit' strategy",
            tooltip="The 'bitfit' strategy will freeze all weights, and leave bias in a trainable state. The 'none' strategy will leave all parameters in a trainable state. Freezing the weights can improve convergence for finetuning. Using bitfit only moderately reduces VRAM consumption, but substantially reduces the count of trainable parameters.",
            importance=ImportanceLevel.ADVANCED,
            order=35,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Fully Unload Text Encoder
    registry._add_field(
        ConfigField(
            name="fully_unload_text_encoder",
            arg_name="--fully_unload_text_encoder",
            ui_label="Fully Unload Text Encoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=False,
            help_text="If set, will fully unload the text_encoder from memory when not in use",
            tooltip="This currently has the side effect of crashing validations, but it is useful for initiating VAE caching on GPUs that would otherwise be too small.",
            importance=ImportanceLevel.ADVANCED,
            order=36,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Save Text Encoder
    registry._add_field(
        ConfigField(
            name="save_text_encoder",
            arg_name="--save_text_encoder",
            ui_label="Save Text Encoder",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=False,
            help_text="If set, will save the text encoder after training",
            tooltip="This is useful if you're using --push_to_hub so that the final pipeline contains all necessary components to run.",
            importance=ImportanceLevel.ADVANCED,
            order=37,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Text Encoder Limit
    registry._add_field(
        ConfigField(
            name="text_encoder_limit",
            arg_name="--text_encoder_limit",
            ui_label="Text Encoder Training Limit",
            field_type=FieldType.NUMBER,
            tab="training",
            section="text_encoder",
            subsection="advanced",
            default_value=100,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="When training the text encoder, we want to limit how long it trains for to avoid catastrophic loss",
            tooltip="When training the text encoder, we want to limit how long it trains for to avoid catastrophic loss.",
            importance=ImportanceLevel.ADVANCED,
            order=38,
            dependencies=[FieldDependency(field="train_text_encoder", operator="equals", value=True, action="show")],
        )
    )

    # Prepend Instance Prompt
    registry._add_field(
        ConfigField(
            name="prepend_instance_prompt",
            arg_name="--prepend_instance_prompt",
            ui_label="Prepend Instance Prompt",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caption_processing",
            default_value=False,
            help_text="When determining the captions from the filename, prepend the instance prompt as an enforced keyword",
            tooltip="When determining the captions from the filename, prepend the instance prompt as an enforced keyword.",
            importance=ImportanceLevel.ADVANCED,
            order=39,
        )
    )

    # Only Instance Prompt
    registry._add_field(
        ConfigField(
            name="only_instance_prompt",
            arg_name="--only_instance_prompt",
            ui_label="Only Use Instance Prompt",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caption_processing",
            default_value=False,
            help_text="Use the instance prompt instead of the caption from filename",
            tooltip="Use the instance prompt instead of the caption from filename.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
        )
    )

    # Data Aesthetic Score
    registry._add_field(
        ConfigField(
            name="data_aesthetic_score",
            arg_name="--data_aesthetic_score",
            ui_label="Data Aesthetic Score",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caption_processing",
            default_value=7.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            help_text="Since currently we do not calculate aesthetic scores for data, we will statically set it to one value. This is only used by the SDXL Refiner",
            tooltip="Since currently we do not calculate aesthetic scores for data, we will statically set it to one value. This is only used by the SDXL Refiner.",
            importance=ImportanceLevel.ADVANCED,
            order=41,
        )
    )

    # Caption Dropout Probability
    registry._add_field(
        ConfigField(
            name="caption_dropout_probability",
            arg_name="--caption_dropout_probability",
            ui_label="Caption Dropout Probability",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caption_processing",
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Caption dropout will randomly drop captions and, for SDXL, size conditioning inputs based on this probability",
            tooltip="When set to a value of 0.1, it will drop approximately 10 percent of the inputs. The default is to use zero caption dropout, though for better generalisation, a value of 0.1 is recommended.",
            importance=ImportanceLevel.ADVANCED,
            order=42,
        )
    )

    # Delete Unwanted Images
    registry._add_field(
        ConfigField(
            name="delete_unwanted_images",
            arg_name="--delete_unwanted_images",
            ui_label="Delete Unwanted Images",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="Remove images that fail your minimum resolution requirements instead of just skipping them",
            tooltip="Deletes too-small images (as defined by minimum image size settings) from the source backend instead of only removing them from buckets.",
            importance=ImportanceLevel.ADVANCED,
            order=43,
        )
    )

    # Delete Problematic Images
    registry._add_field(
        ConfigField(
            name="delete_problematic_images",
            arg_name="--delete_problematic_images",
            ui_label="Delete Problematic Images",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="Delete images that cannot be loaded (corrupt, truncated, or unreadable) so caching doesn't retry them",
            tooltip="Removes problematic files from the backing store when they fail to load during caching, avoiding repeated failures.",
            importance=ImportanceLevel.ADVANCED,
            order=44,
        )
    )

    # Disable Bucket Pruning
    registry._add_field(
        ConfigField(
            name="disable_bucket_pruning",
            arg_name="--disable_bucket_pruning",
            ui_label="Disable Bucket Pruning",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="When training on very small datasets, you might not care that the batch sizes will outpace your image count. Setting this option will prevent SimpleTuner from deleting your bucket lists that do not meet the minimum image count requirements. Use at your own risk, it may end up throwing off your statistics or epoch tracking",
            tooltip="When training on very small datasets, you might not care that the batch sizes will outpace your image count. Setting this option will prevent SimpleTuner from deleting your bucket lists that do not meet the minimum image count requirements. Use at your own risk, it may end up throwing off your statistics or epoch tracking.",
            importance=ImportanceLevel.ADVANCED,
            order=45,
        )
    )

    # Disable Segmented Timestep Sampling
    registry._add_field(
        ConfigField(
            name="disable_segmented_timestep_sampling",
            arg_name="--disable_segmented_timestep_sampling",
            ui_label="Disable Segmented Timestep Sampling",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=False,
            help_text="By default, the timestep schedule is divided into roughly `train_batch_size` number of segments, and then each of those are sampled from separately. This improves the selection distribution, but may not be desired in certain training scenarios, eg. when limiting the timestep selection range",
            tooltip="By default, the timestep schedule is divided into roughly `train_batch_size` number of segments, and then each of those are sampled from separately. This improves the selection distribution, but may not be desired in certain training scenarios, eg. when limiting the timestep selection range.",
            importance=ImportanceLevel.ADVANCED,
            order=46,
        )
    )

    # Preserve Data Backend Cache
    registry._add_field(
        ConfigField(
            name="preserve_data_backend_cache",
            arg_name="--preserve_data_backend_cache",
            ui_label="Preserve Data Backend Cache",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caching",
            default_value=False,
            help_text="For very large cloud storage buckets that will never change, enabling this option will prevent the trainer from scanning it at startup, by preserving the cache files that we generate. Be careful when using this, as, switching datasets can result in the preserved cache being used, which would be problematic. Currently, cache is not stored in the dataset itself but rather, locally. This may change in a future release",
            tooltip="For very large cloud storage buckets that will never change, enabling this option will prevent the trainer from scanning it at startup, by preserving the cache files that we generate. Be careful when using this, as, switching datasets can result in the preserved cache being used, which would be problematic. Currently, cache is not stored in the dataset itself but rather, locally. This may change in a future release.",
            importance=ImportanceLevel.ADVANCED,
            order=47,
        )
    )

    # Override Dataset Config
    registry._add_field(
        ConfigField(
            name="override_dataset_config",
            arg_name="--override_dataset_config",
            ui_label="Override Dataset Config",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            default_value=False,
            help_text="When provided, the dataset's config will not be checked against the live backend config",
            tooltip="This is useful if you want to simply update the behaviour of an existing dataset, but the recommendation is to not change the dataset configuration after caching has begun, as most options cannot be changed without unexpected behaviour later on. Additionally, it prevents accidentally loading an SDXL configuration on a SD 2.x model and vice versa.",
            importance=ImportanceLevel.ADVANCED,
            order=48,
        )
    )

    # Cache Directory
    registry._add_field(
        ConfigField(
            name="cache_dir",
            arg_name="--cache_dir",
            ui_label="Cache Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="caching",
            default_value=None,
            help_text="The directory where the downloaded models and datasets will be stored",
            tooltip="The directory where the downloaded models and datasets will be stored.",
            importance=ImportanceLevel.ADVANCED,
            order=49,
        )
    )

    # Cache Directory Text
    registry._add_field(
        ConfigField(
            name="cache_dir_text",
            arg_name="--cache_dir_text",
            ui_label="Text Cache Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="caching",
            subsection="advanced",
            default_value="cache",
            help_text="This is the path to a local directory that will contain your text embed cache",
            tooltip="This is the path to a local directory that will contain your text embed cache.",
            importance=ImportanceLevel.ADVANCED,
            order=50,
        )
    )

    # Cache Directory VAE
    registry._add_field(
        ConfigField(
            name="cache_dir_vae",
            arg_name="--cache_dir_vae",
            ui_label="VAE Cache Directory",
            field_type=FieldType.TEXT,
            tab="basic",
            section="caching",
            subsection="advanced",
            default_value="",
            help_text="This is the path to a local directory that will contain your VAE outputs",
            tooltip="This is the path to a local directory that will contain your VAE outputs. Each backend can have its own value, but if that is not provided, this will be the default value.",
            importance=ImportanceLevel.ADVANCED,
            order=51,
        )
    )

    # Cache Clear Validation Prompts
    registry._add_field(
        ConfigField(
            name="cache_clear_validation_prompts",
            arg_name="--cache_clear_validation_prompts",
            ui_label="Clear Validation Prompts from Cache",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caching",
            subsection="advanced",
            default_value=False,
            help_text="When provided, any validation prompt entries in the text embed cache will be recreated",
            tooltip="This is useful if you've modified any of the existing prompts.",
            importance=ImportanceLevel.ADVANCED,
            order=52,
        )
    )

    # Compress Disk Cache
    registry._add_field(
        ConfigField(
            name="compress_disk_cache",
            arg_name="--compress_disk_cache",
            ui_label="Compress Disk Cache",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caching",
            default_value=True,
            help_text="If set, will gzip-compress the disk cache for Pytorch files. This will save substantial disk space, but may slow down the training process",
            tooltip="If set, will gzip-compress the disk cache for Pytorch files. This will save substantial disk space, but may slow down the training process.",
            importance=ImportanceLevel.ADVANCED,
            order=53,
        )
    )

    # Aspect Bucket Disable Rebuild
    registry._add_field(
        ConfigField(
            name="aspect_bucket_disable_rebuild",
            arg_name="--aspect_bucket_disable_rebuild",
            ui_label="Disable Aspect Bucket Rebuild",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value=False,
            help_text="When using a randomised aspect bucket list, the VAE and aspect cache are rebuilt on each epoch. With a large and diverse enough dataset, rebuilding the aspect list may take a long time, and this may be undesirable. This option will not override vae_cache_clear_each_epoch. If both options are provided, only the VAE cache will be rebuilt",
            tooltip="When using a randomised aspect bucket list, the VAE and aspect cache are rebuilt on each epoch. With a large and diverse enough dataset, rebuilding the aspect list may take a long time, and this may be undesirable. This option will not override vae_cache_clear_each_epoch. If both options are provided, only the VAE cache will be rebuilt.",
            importance=ImportanceLevel.ADVANCED,
            order=54,
        )
    )

    # Keep VAE Loaded
    registry._add_field(
        ConfigField(
            name="keep_vae_loaded",
            arg_name="--keep_vae_loaded",
            ui_label="Keep VAE Loaded",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="vae_config",
            default_value=False,
            help_text="If set, will keep the VAE loaded in memory. This can reduce disk churn, but consumes VRAM during the forward pass",
            tooltip="If set, will keep the VAE loaded in memory. This can reduce disk churn, but consumes VRAM during the forward pass.",
            importance=ImportanceLevel.ADVANCED,
            order=23,
        )
    )

    # Skip File Discovery
    registry._add_field(
        ConfigField(
            name="skip_file_discovery",
            arg_name="--skip_file_discovery",
            ui_label="Skip File Discovery",
            field_type=FieldType.TEXT,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value="",
            help_text="Comma-separated values of which stages to skip discovery for. Skipping any stage will speed up resumption, but will increase the risk of errors, as missing images or incorrectly bucketed images may not be caught. Valid options: aspect, vae, text, metadata",
            tooltip="Comma-separated values of which stages to skip discovery for. Skipping any stage will speed up resumption, but will increase the risk of errors, as missing images or incorrectly bucketed images may not be caught. Valid options: aspect, vae, text, metadata.",
            importance=ImportanceLevel.ADVANCED,
            order=56,
        )
    )

    # Data Backend Sampling
    registry._add_field(
        ConfigField(
            name="data_backend_sampling",
            arg_name="--data_backend_sampling",
            ui_label="Data Backend Sampling",
            field_type=FieldType.SELECT,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value="auto-weighting",
            choices=[
                {"value": "uniform", "label": "Uniform"},
                {"value": "auto-weighting", "label": "Auto-Weighting"},
            ],
            help_text="When using multiple data backends, the sampling weighting can be set to 'uniform' or 'auto-weighting'",
            tooltip="The default value is 'auto-weighting', which will automatically adjust the sampling weights based on the number of images in each backend. 'uniform' will sample from each backend equally.",
            importance=ImportanceLevel.ADVANCED,
            order=57,
        )
    )

    # Image Processing Batch Size
    registry._add_field(
        ConfigField(
            name="image_processing_batch_size",
            arg_name="--image_processing_batch_size",
            ui_label="Image Processing Batch Size",
            field_type=FieldType.NUMBER,
            tab="model",
            section="vae_config",
            default_value=32,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="When resizing and cropping images, we do it in parallel using processes or threads. This defines how many images will be read into the queue before they are processed",
            tooltip="When resizing and cropping images, we do it in parallel using processes or threads. This defines how many images will be read into the queue before they are processed.",
            importance=ImportanceLevel.ADVANCED,
            order=24,
        )
    )

    # Write Batch Size
    registry._add_field(
        ConfigField(
            name="write_batch_size",
            arg_name="--write_batch_size",
            ui_label="Write Batch Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caching",
            subsection="advanced",
            default_value=128,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="When using certain storage backends, it is better to batch smaller writes rather than continuous dispatching. In SimpleTuner, write batching is currently applied during VAE caching, when many small objects are written. This mostly applies to S3, but some shared server filesystems may benefit as well. Default: 64",
            tooltip="When using certain storage backends, it is better to batch smaller writes rather than continuous dispatching. In SimpleTuner, write batching is currently applied during VAE caching, when many small objects are written. This mostly applies to S3, but some shared server filesystems may benefit as well. Default: 64.",
            importance=ImportanceLevel.ADVANCED,
            order=59,
        )
    )

    # Read Batch Size
    registry._add_field(
        ConfigField(
            name="read_batch_size",
            arg_name="--read_batch_size",
            ui_label="Read Batch Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="caching",
            subsection="advanced",
            default_value=25,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Used by the VAE cache to prefetch image data. This is the number of images to read ahead",
            tooltip="Used by the VAE cache to prefetch image data. This is the number of images to read ahead.",
            importance=ImportanceLevel.ADVANCED,
            order=60,
        )
    )

    # Enable Multiprocessing
    registry._add_field(
        ConfigField(
            name="enable_multiprocessing",
            arg_name="--enable_multiprocessing",
            ui_label="Enable Multiprocessing",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="caching",
            default_value=False,
            help_text="If set, will use processes instead of threads during metadata caching operations",
            tooltip="For some systems, multiprocessing may be faster than threading, but will consume a lot more memory. Use this option with caution, and monitor your system's memory usage.",
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
            importance=ImportanceLevel.ADVANCED,
            order=61,
        )
    )

    # Accelerate Config Path
    registry._add_field(
        ConfigField(
            name="accelerate_config",
            arg_name="--accelerate_config",
            ui_label="Accelerate Config Path",
            field_type=FieldType.TEXT,
            tab="hardware",
            section="accelerate",
            default_value=None,
            placeholder="~/.cache/huggingface/accelerate/default_config.yaml",
            validation_rules=[ValidationRule(ValidationRuleType.PATH_EXISTS, message="Accelerate config file not found")],
            help_text="Path to the accelerate `default_config.yaml` used when launching training.",
            tooltip="Overrides the default Hugging Face accelerate config discovery. The YAML controls distributed mode, DeepSpeed, multi-GPU, and multi-node settings.",
            importance=ImportanceLevel.IMPORTANT,
            order=10,
        )
    )

    registry._add_field(
        ConfigField(
            name="deepspeed_config",
            arg_name="--deepspeed_config",
            ui_label="DeepSpeed Config (JSON)",
            field_type=FieldType.TEXTAREA,
            tab="hardware",
            section="accelerate",
            default_value=None,
            placeholder='{"zero_optimization": {"stage": 2}}',
            help_text="Custom DeepSpeed configuration JSON. Leave blank to rely on Accelerate defaults.",
            tooltip="Accepts raw JSON or a path to a JSON file. The training wizard can generate starter configs for common ZeRO stages.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
            dependencies=[FieldDependency(field="model_type", operator="equals", value="full", action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_enable",
            arg_name="--fsdp_enable",
            ui_label="Enable FSDP v2",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            default_value=False,
            help_text="Enable PyTorch Fully Sharded Data Parallel v2 (DTensor-based) via Accelerate.",
            tooltip="Activates FSDP2 for single-node multi-GPU runs. Mutually exclusive with DeepSpeed.",
            importance=ImportanceLevel.ADVANCED,
            order=12,
            dependencies=[FieldDependency(field="model_type", operator="equals", value="full", action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_version",
            arg_name="--fsdp_version",
            ui_label="FSDP Version",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=2,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="Valid FSDP versions are 1 or 2"),
                ValidationRule(ValidationRuleType.MAX, value=2, message="Only FSDP v2 is supported"),
            ],
            help_text="Select FSDP API version. Version 2 enables the DTensor-backed implementation.",
            tooltip="SimpleTuner defaults to FSDP2. FSDP1 is deprecated and may be removed in a future release.",
            importance=ImportanceLevel.ADVANCED,
            order=13,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_reshard_after_forward",
            arg_name="--fsdp_reshard_after_forward",
            ui_label="Reshard After Forward",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            default_value=True,
            help_text="Release parameter shards after each forward pass (equivalent to FULL_SHARD in FSDP1).",
            tooltip="Keeps memory usage low at the cost of extra communication. Disable to keep params sharded across backward only.",
            importance=ImportanceLevel.ADVANCED,
            order=14,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_state_dict_type",
            arg_name="--fsdp_state_dict_type",
            ui_label="Checkpoint Format",
            field_type=FieldType.SELECT,
            tab="hardware",
            section="accelerate",
            default_value="SHARDED_STATE_DICT",
            choices=[
                {"value": "SHARDED_STATE_DICT", "label": "Sharded (recommended)"},
                {"value": "FULL_STATE_DICT", "label": "Full state dict"},
            ],
            help_text="Controls how checkpoints are saved when using FSDP2.",
            tooltip="Sharded checkpoints are fast and memory efficient. Full checkpoints gather weights on rank 0.",
            importance=ImportanceLevel.ADVANCED,
            order=15,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_cpu_ram_efficient_loading",
            arg_name="--fsdp_cpu_ram_efficient_loading",
            ui_label="CPU RAM Efficient Loading",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            default_value=False,
            help_text="Load checkpoints on rank 0 before broadcasting shards when resuming with FSDP2.",
            tooltip="Reduces peak host memory usage at the cost of a synchronization barrier.",
            importance=ImportanceLevel.ADVANCED,
            order=16,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_auto_wrap_policy",
            arg_name="--fsdp_auto_wrap_policy",
            ui_label="Auto Wrap Policy",
            field_type=FieldType.SELECT,
            tab="hardware",
            section="accelerate",
            default_value="TRANSFORMER_BASED_WRAP",
            choices=[
                {"value": "TRANSFORMER_BASED_WRAP", "label": "Transformer based (recommended)"},
                {"value": "SIZE_BASED_WRAP", "label": "Size based"},
                {"value": "NO_WRAP", "label": "No wrap"},
            ],
            help_text="Configure how FSDP decides which modules to shard.",
            tooltip="Transformer based works for most HF transformer models. Size based lets you control wrapping via parameter counts.",
            importance=ImportanceLevel.ADVANCED,
            order=17,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="fsdp_transformer_layer_cls_to_wrap",
            arg_name="--fsdp_transformer_layer_cls_to_wrap",
            ui_label="Transformer Classes to Wrap",
            field_type=FieldType.TEXT,
            tab="hardware",
            section="accelerate",
            default_value=None,
            placeholder="TransformerBlock,DiffusionTransformerLayer",
            help_text="Comma separated list of layer class names to wrap when using transformer auto wrap.",
            tooltip="Overrides Accelerate's automatic class detection. Leave blank unless validation errors request a specific layer class.",
            importance=ImportanceLevel.ADVANCED,
            order=18,
            dependencies=[FieldDependency(field="fsdp_enable", operator="equals", value=True, action="show")],
        )
    )

    registry._add_field(
        ConfigField(
            name="context_parallel_size",
            arg_name="--context_parallel_size",
            ui_label="Context Parallel Size",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Number of ranks used for Accelerate's context parallel sharding across the sequence dimension.",
            tooltip="Set to >1 to shard attention across GPUs when using FSDP2. Leave at 1 to disable context parallelism.",
            importance=ImportanceLevel.ADVANCED,
            order=19,
            platform_specific=["cuda"],
        )
    )

    registry._add_field(
        ConfigField(
            name="context_parallel_comm_strategy",
            arg_name="--context_parallel_comm_strategy",
            ui_label="Context Parallel Rotation",
            field_type=FieldType.SELECT,
            tab="hardware",
            section="accelerate",
            default_value="allgather",
            choices=[
                {"value": "allgather", "label": "All-Gather (recommended)"},
                {"value": "alltoall", "label": "All-to-All"},
            ],
            help_text="Communication primitive used to rotate K/V shards during context parallel attention.",
            tooltip="All-gather generally offers better overlap; all-to-all may help niche workloads. Requires FSDP2 on CUDA GPUs.",
            importance=ImportanceLevel.ADVANCED,
            order=19,
            platform_specific=["cuda"],
        )
    )

    # Training Num Processes
    registry._add_field(
        ConfigField(
            name="num_processes",
            arg_name="--num_processes",
            ui_label="Processes per Node",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must launch at least one process")],
            help_text="Process count passed to accelerate launch when no config file is supplied.",
            tooltip="For single-node multi-GPU starts, set this to the number of visible GPUs. Ignored when an accelerate config file is provided.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
        )
    )

    # Training Num Machines
    registry._add_field(
        ConfigField(
            name="num_machines",
            arg_name="--num_machines",
            ui_label="Machine Count",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=1,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Total machines participating when using accelerate launch without a config file.",
            tooltip="Only used when fallback CLI flags are required. Provide full multi-node details via accelerate config for production runs.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
        )
    )

    # Accelerate Extra Args
    registry._add_field(
        ConfigField(
            name="accelerate_extra_args",
            arg_name="--accelerate_extra_args",
            ui_label="Extra Accelerate Flags",
            field_type=FieldType.TEXT,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value=None,
            placeholder="--machine_rank=0 --main_process_ip=10.0.0.100",
            help_text="Additional arguments inserted between `accelerate launch` and `simpletuner/train.py`.",
            tooltip="Use for rare cases where you cannot express a setting in accelerate config. Space-separated flags are appended verbatim.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
        )
    )

    # Main Process IP
    registry._add_field(
        ConfigField(
            name="main_process_ip",
            arg_name="--main_process_ip",
            ui_label="Main Process IP",
            field_type=FieldType.TEXT,
            tab="hardware",
            section="accelerate",
            default_value="127.0.0.1",
            help_text="Hostname or IP of the coordination node (usually rank 0). Required when running across multiple machines without an accelerate config file.",
            tooltip="Set to the address reachable by every worker. When using an accelerate YAML this value is usually supplied there instead.",
            dependencies=[FieldDependency(field="num_machines", operator="greater_than", value=1, action="show")],
            importance=ImportanceLevel.IMPORTANT,
            order=45,
        )
    )

    # Main Process Port
    registry._add_field(
        ConfigField(
            name="main_process_port",
            arg_name="--main_process_port",
            ui_label="Main Process Port",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=29500,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Port must be positive")],
            help_text="TCP port used for rendezvous when launching across machines.",
            tooltip="All nodes must be able to reach this port on the main process host. Pick an open port and ensure firewalls allow access.",
            dependencies=[FieldDependency(field="num_machines", operator="greater_than", value=1, action="show")],
            importance=ImportanceLevel.IMPORTANT,
            order=46,
        )
    )

    # Machine Rank
    registry._add_field(
        ConfigField(
            name="machine_rank",
            arg_name="--machine_rank",
            ui_label="Machine Rank",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="accelerate",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Rank must be >= 0")],
            help_text="Unique rank for this machine within the cluster (0-based).",
            tooltip="Set to 0 on the primary node, 1 on the second, and so on.",
            dependencies=[FieldDependency(field="num_machines", operator="greater_than", value=1, action="show")],
            importance=ImportanceLevel.IMPORTANT,
            order=47,
        )
    )

    # Same Network
    registry._add_field(
        ConfigField(
            name="same_network",
            arg_name="--same_network",
            ui_label="Machines Share Network",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            default_value=True,
            help_text="Indicates that all machines can reach each other without SSH tunnelling.",
            tooltip="Disable if workers require SSH tunnelling to communicate.",
            dependencies=[FieldDependency(field="num_machines", operator="greater_than", value=1, action="show")],
            importance=ImportanceLevel.ADVANCED,
            order=48,
        )
    )

    # Torch Dynamo Backend
    registry._add_field(
        ConfigField(
            name="dynamo_backend",
            arg_name="--dynamo_backend",
            ui_label="Torch Dynamo Backend",
            field_type=FieldType.SELECT,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value="no",
            choices=[
                {"value": "no", "label": "Disabled"},
                {"value": "inductor", "label": "Inductor"},
                {"value": "eager", "label": "Eager"},
                {"value": "aot_eager", "label": "AOT Eager"},
                {"value": "aot_ts_nvfuser", "label": "AOT TS NVFuser"},
                {"value": "nvprims_nvfuser", "label": "NVPrims NVFuser"},
                {"value": "cudagraphs", "label": "CUDAGraphs"},
                {"value": "ofi", "label": "OFI"},
                {"value": "fx2trt", "label": "FX2TRT"},
                {"value": "onnxrt", "label": "ONNX Runtime"},
                {"value": "tensorrt", "label": "TensorRT"},
                {"value": "ipex", "label": "Intel IPEX"},
                {"value": "tvm", "label": "TVM"},
                {"value": "hpu_backend", "label": "Habana (HPU)"},
            ],
            help_text="Selects the torch.compile backend Accelerate should use for TorchDynamo compilation.",
            tooltip="Enable only after confirming your hardware supports the selected backend. Prefer Accelerate YAML overrides for multi-node DeepSpeed setups.",
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show")],
            importance=ImportanceLevel.EXPERIMENTAL,
            order=50,
        )
    )

    registry._add_field(
        ConfigField(
            name="dynamo_mode",
            arg_name="--dynamo_mode",
            ui_label="Torch Dynamo Mode",
            field_type=FieldType.SELECT,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value="",
            choices=[
                {"value": "", "label": "Auto (Accelerate default)"},
                {"value": "default", "label": "default"},
                {"value": "reduce-overhead", "label": "reduce-overhead"},
                {"value": "max-autotune", "label": "max-autotune"},
                {"value": "max-autotune-no-cudagraphs", "label": "max-autotune-no-cudagraphs"},
            ],
            help_text="Optional torch.compile optimisation profile passed to TorchDynamo.",
            tooltip="Auto lets Accelerate pick the best mode. Reduce overhead minimises compile time, max-autotune searches for the fastest kernels.",
            dependencies=[
                FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show"),
                FieldDependency(field="dynamo_backend", operator="not_equals", value="no", action="show"),
            ],
            importance=ImportanceLevel.EXPERIMENTAL,
            order=51,
        )
    )

    registry._add_field(
        ConfigField(
            name="dynamo_fullgraph",
            arg_name="--dynamo_fullgraph",
            ui_label="Full-Graph Compilation",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value=False,
            help_text="Request full graph compilation from torch.compile instead of per-region lowering.",
            tooltip="May improve steady-state throughput at the cost of longer initial compile times.",
            dependencies=[
                FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show"),
                FieldDependency(field="dynamo_backend", operator="not_equals", value="no", action="show"),
            ],
            importance=ImportanceLevel.EXPERIMENTAL,
            order=52,
        )
    )

    registry._add_field(
        ConfigField(
            name="dynamo_dynamic",
            arg_name="--dynamo_dynamic",
            ui_label="Dynamic Shapes Support",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value=False,
            help_text="Enable dynamic shape guards when compiling with torch.compile.",
            tooltip="Turn on only if your model receives tensors with varying shapes between steps.",
            dependencies=[
                FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show"),
                FieldDependency(field="dynamo_backend", operator="not_equals", value="no", action="show"),
            ],
            importance=ImportanceLevel.EXPERIMENTAL,
            order=53,
        )
    )

    registry._add_field(
        ConfigField(
            name="dynamo_use_regional_compilation",
            arg_name="--dynamo_use_regional_compilation",
            ui_label="Use Regional Compilation",
            field_type=FieldType.CHECKBOX,
            tab="hardware",
            section="accelerate",
            subsection="advanced",
            default_value=False,
            help_text="Compile repeated model blocks individually to reduce cold-start time.",
            tooltip="Recommended for large transformer style models. Keeps runtime close to full graph with significantly faster compilation.",
            dependencies=[
                FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True, action="show"),
                FieldDependency(field="dynamo_backend", operator="not_equals", value="no", action="show"),
            ],
            importance=ImportanceLevel.EXPERIMENTAL,
            order=54,
        )
    )

    # Max Workers
    registry._add_field(
        ConfigField(
            name="max_workers",
            arg_name="--max_workers",
            ui_label="Max Workers",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="hardware",
            default_value=32,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="How many active threads or processes to run during VAE caching",
            tooltip="How many active threads or processes to run during VAE caching.",
            importance=ImportanceLevel.ADVANCED,
            order=62,
        )
    )

    # AWS Max Pool Connections
    registry._add_field(
        ConfigField(
            name="aws_max_pool_connections",
            arg_name="--aws_max_pool_connections",
            ui_label="AWS Max Pool Connections",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value=128,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="When using AWS backends, the maximum number of connections to keep open to the S3 bucket at a single time",
            tooltip="When using AWS backends, the maximum number of connections to keep open to the S3 bucket at a single time. This should be greater or equal to the max_workers and aspect bucket worker count values.",
            importance=ImportanceLevel.ADVANCED,
            order=63,
        )
    )

    # Torch Num Threads
    registry._add_field(
        ConfigField(
            name="torch_num_threads",
            arg_name="--torch_num_threads",
            ui_label="Torch Num Threads",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="hardware",
            subsection="advanced",
            default_value=8,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="The number of threads to use for PyTorch operations. This is not the same as the number of workers",
            tooltip="The number of threads to use for PyTorch operations. This is not the same as the number of workers. Default: 8.",
            importance=ImportanceLevel.ADVANCED,
            order=64,
        )
    )

    # Dataloader Prefetch
    registry._add_field(
        ConfigField(
            name="dataloader_prefetch",
            arg_name="--dataloader_prefetch",
            ui_label="Dataloader Prefetch",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="data_config",
            default_value=False,
            help_text="When provided, the dataloader will read-ahead and attempt to retrieve latents, text embeds, and other metadata ahead of the time when the batch is required, so that it can be immediately available",
            tooltip="When provided, the dataloader will read-ahead and attempt to retrieve latents, text embeds, and other metadata ahead of the time when the batch is required, so that it can be immediately available.",
            importance=ImportanceLevel.ADVANCED,
            order=65,
        )
    )

    # Dataloader Prefetch Queue Length
    registry._add_field(
        ConfigField(
            name="dataloader_prefetch_qlen",
            arg_name="--dataloader_prefetch_qlen",
            ui_label="Dataloader Prefetch Queue Length",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value=10,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="Set the number of prefetched batches",
            tooltip="Set the number of prefetched batches.",
            importance=ImportanceLevel.ADVANCED,
            order=66,
        )
    )

    # Aspect Bucket Worker Count
    registry._add_field(
        ConfigField(
            name="aspect_bucket_worker_count",
            arg_name="--aspect_bucket_worker_count",
            ui_label="Aspect Bucket Worker Count",
            field_type=FieldType.NUMBER,
            tab="hardware",
            section="hardware",
            default_value=12,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
            help_text="The number of workers to use for aspect bucketing. This is a CPU-bound task, so the number of workers should be set to the number of CPU threads available. If you use an I/O bound backend, an even higher value may make sense. Default: 12",
            tooltip="The number of workers to use for aspect bucketing. This is a CPU-bound task, so the number of workers should be set to the number of CPU threads available. If you use an I/O bound backend, an even higher value may make sense. Default: 12.",
            importance=ImportanceLevel.ADVANCED,
            order=67,
        )
    )

    # Aspect Bucket Alignment
    registry._add_field(
        ConfigField(
            name="aspect_bucket_alignment",
            arg_name="--aspect_bucket_alignment",
            ui_label="Aspect Bucket Alignment",
            field_type=FieldType.SELECT,
            tab="basic",
            section="image_processing",
            subsection="advanced",
            choices=[
                {"value": 8, "label": "8"},
                {"value": 16, "label": "16"},
                {"value": 24, "label": "24"},
                {"value": 32, "label": "32"},
                {"value": 64, "label": "64"},
            ],
            default_value=64,
            help_text="When training diffusion models, the image sizes generally must align to a 64 pixel interval",
            tooltip="When training diffusion models, the image sizes generally must align to a 64 pixel interval. This is an exception when training models like DeepFloyd that use a base resolution of 64 pixels, as aligning to 64 pixels would result in a 1:1 or 2:1 aspect ratio, overly distorting images. For DeepFloyd, this value is set to 32, but all other training defaults to 64. You may experiment with this value, but it is not recommended.",
            importance=ImportanceLevel.ADVANCED,
            order=68,
        )
    )

    # Minimum Image Size
    registry._add_field(
        ConfigField(
            name="minimum_image_size",
            arg_name="--minimum_image_size",
            ui_label="Minimum Image Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="image_processing",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=128, message="Must be at least 128")],
            help_text="The minimum resolution for both sides of input images",
            tooltip="If --delete_unwanted_images is set, images smaller than this will be DELETED. The default value is None, which means no minimum resolution is enforced. If this option is not provided, it is possible that images will be destructively upsampled, harming model performance.",
            importance=ImportanceLevel.ADVANCED,
            order=69,
        )
    )

    # Maximum Image Size
    registry._add_field(
        ConfigField(
            name="maximum_image_size",
            arg_name="--maximum_image_size",
            ui_label="Maximum Image Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="image_processing",
            default_value=None,
            help_text="When cropping images that are excessively large, the entire scene context may be lost, eg. the crop might just end up being a portion of the background. To avoid this, a maximum image size may be provided, which will result in very-large images being downsampled before cropping them. This value uses --resolution_type to determine whether it is a pixel edge or megapixel value",
            tooltip="When cropping images that are excessively large, the entire scene context may be lost, eg. the crop might just end up being a portion of the background. To avoid this, a maximum image size may be provided, which will result in very-large images being downsampled before cropping them. This value uses --resolution_type to determine whether it is a pixel edge or megapixel value.",
            importance=ImportanceLevel.ADVANCED,
            order=70,
        )
    )

    # Target Downsample Size
    registry._add_field(
        ConfigField(
            name="target_downsample_size",
            arg_name="--target_downsample_size",
            ui_label="Target Downsample Size",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="image_processing",
            default_value=None,
            help_text="When using --maximum_image_size, very-large images exceeding that value will be downsampled to this target size before cropping",
            tooltip="When using --maximum_image_size, very-large images exceeding that value will be downsampled to this target size before cropping. If --resolution_type=area and --maximum_image_size=4.0, --target_downsample_size=2.0 would result in a 4 megapixel image being resized to 2 megapixel before cropping to 1 megapixel.",
            importance=ImportanceLevel.ADVANCED,
            order=71,
        )
    )

    # Metadata Update Interval
    registry._add_field(
        ConfigField(
            name="metadata_update_interval",
            arg_name="--metadata_update_interval",
            ui_label="Metadata Update Interval",
            field_type=FieldType.NUMBER,
            tab="basic",
            section="data_config",
            subsection="advanced",
            default_value=3600,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=60, message="Must be at least 60")],
            help_text="When generating the aspect bucket indicies, we want to save it every X seconds",
            tooltip="When generating the aspect bucket indicies, we want to save it every X seconds. The default is to save it every 1 hour, such that progress is not lost on clusters where runtime is limited to 6-hour increments (e.g. the JUWELS Supercomputer). The minimum value is 60 seconds.",
            importance=ImportanceLevel.ADVANCED,
            order=72,
        )
    )

    # Debug Aspect Buckets
    registry._add_field(
        ConfigField(
            name="debug_aspect_buckets",
            arg_name="--debug_aspect_buckets",
            ui_label="Debug Aspect Buckets",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="If set, will print excessive debugging for aspect bucket operations",
            tooltip="If set, will print excessive debugging for aspect bucket operations.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=73,
        )
    )

    # Debug Dataset Loader
    registry._add_field(
        ConfigField(
            name="debug_dataset_loader",
            arg_name="--debug_dataset_loader",
            ui_label="Debug Dataset Loader",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="If set, will print excessive debugging for data loader operations",
            tooltip="If set, will print excessive debugging for data loader operations.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=74,
        )
    )

    # Print Filenames
    registry._add_field(
        ConfigField(
            name="print_filenames",
            arg_name="--print_filenames",
            ui_label="Print Filenames",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="If any image files are stopping the process eg. due to corruption or truncation, this will help identify which is at fault",
            tooltip="If any image files are stopping the process eg. due to corruption or truncation, this will help identify which is at fault.",
            importance=ImportanceLevel.ADVANCED,
            order=75,
        )
    )

    # Print Sampler Statistics
    registry._add_field(
        ConfigField(
            name="print_sampler_statistics",
            arg_name="--print_sampler_statistics",
            ui_label="Print Sampler Statistics",
            field_type=FieldType.CHECKBOX,
            tab="basic",
            section="training_data",
            subsection="advanced",
            default_value=False,
            help_text="If provided, will print statistics about the dataset sampler. This is useful for debugging",
            tooltip="If provided, will print statistics about the dataset sampler. This is useful for debugging.",
            importance=ImportanceLevel.ADVANCED,
            order=76,
        )
    )

    # Mixed Precision
    registry._add_field(
        ConfigField(
            name="mixed_precision",
            arg_name="--mixed_precision",
            ui_label="Mixed Precision",
            field_type=FieldType.SELECT,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value="bf16",
            choices=[
                {"value": "no", "label": "No (FP32)"},
                {"value": "fp16", "label": "FP16"},
                {"value": "bf16", "label": "BF16 (Recommended)"},
                {"value": "fp8", "label": "FP8 (Experimental)"},
            ],
            help_text="Precision for training computations",
            tooltip="BF16 is recommended for stability. FP16 saves memory but less stable. FP8 is experimental.",
            importance=ImportanceLevel.IMPORTANT,
            order=10,
            dependencies=[FieldDependency(field="i_know_what_i_am_doing", operator="equals", value=True)],
        )
    )

    # Attention Mechanism
    attention_mechanisms = [
        "diffusers",
        "xformers",
        "sageattention",
        "sageattention-int8-fp16-triton",
        "sageattention-int8-fp16-cuda",
        "sageattention-int8-fp8-cuda",
    ]
    registry._add_field(
        ConfigField(
            name="attention_mechanism",
            arg_name="--attention_mechanism",
            ui_label="Attention Implementation",
            field_type=FieldType.SELECT,
            tab="model",
            section="memory_optimization",
            subsection="advanced",
            default_value="diffusers",
            choices=[{"value": a, "label": a} for a in attention_mechanisms],
            help_text="Attention computation backend",
            tooltip="Xformers saves memory. SageAttention is faster but experimental. Diffusers is default.",
            importance=ImportanceLevel.ADVANCED,
            order=10,
        )
    )

    # Disable TF32
    registry._add_field(
        ConfigField(
            name="disable_tf32",
            arg_name="--disable_tf32",
            ui_label="Disable TF32",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value=False,
            platform_specific=["cuda"],
            help_text="Disable TF32 precision on Ampere GPUs",
            tooltip="TF32 is enabled by default on RTX 3000/4000 series. Disabling may reduce performance but increase precision.",
            importance=ImportanceLevel.ADVANCED,
            order=11,
        )
    )

    # Set Grads to None
    registry._add_field(
        ConfigField(
            name="set_grads_to_none",
            arg_name="--set_grads_to_none",
            ui_label="Set Gradients to None",
            field_type=FieldType.CHECKBOX,
            tab="training",
            section="memory_optimization",
            subsection="advanced",
            default_value=False,
            help_text="Set gradients to None instead of zero",
            tooltip="Can save memory and improve performance. May cause issues with some optimizers.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=12,
        )
    )

    # Noise Offset
    registry._add_field(
        ConfigField(
            name="noise_offset",
            arg_name="--noise_offset",
            ui_label="Noise Offset",
            field_type=FieldType.NUMBER,
            tab="training",
            section="noise_settings",
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Values above 1.0 are extreme"),
            ],
            help_text="Add noise offset to training",
            tooltip="Helps generate darker/lighter images. Common values: 0.05-0.1. 0 = disabled.",
            importance=ImportanceLevel.ADVANCED,
            order=1,
        )
    )

    # Noise Offset Probability
    registry._add_field(
        ConfigField(
            name="noise_offset_probability",
            arg_name="--noise_offset_probability",
            ui_label="Noise Offset Probability",
            field_type=FieldType.NUMBER,
            tab="training",
            section="noise_settings",
            default_value=0.25,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            help_text="Probability of applying noise offset",
            tooltip="Apply noise offset this fraction of the time. Default: 25%",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # Timestep Bias Strategy
    registry._add_field(
        ConfigField(
            name="timestep_bias_strategy",
            arg_name="--timestep_bias_strategy",
            ui_label="Timestep Bias Strategy",
            field_type=FieldType.SELECT,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=None,
            choices=[
                {"value": "earlier", "label": "Bias Earlier"},
                {"value": "later", "label": "Bias Later"},
                {"value": "range", "label": "Bias Range"},
                {"value": "none", "label": "No Bias"},
            ],
            help_text="Strategy for biasing timestep sampling",
            tooltip="'earlier'/'later' emphasise different regions of the schedule, 'range' targets a custom window, 'none' disables the bias.",
            importance=ImportanceLevel.ADVANCED,
            order=41,
        )
    )

    # Timestep Bias Begin
    registry._add_field(
        ConfigField(
            name="timestep_bias_begin",
            arg_name="--timestep_bias_begin",
            ui_label="Timestep Bias Begin",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=0,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1000, message="Must be <= 1000"),
            ],
            dependencies=[
                FieldDependency(field="timestep_bias_strategy", operator="not_equals", value="none", action="show")
            ],
            help_text="Beginning of timestep bias range",
            tooltip="Start of timestep range to bias towards. Only used with timestep bias strategies.",
            importance=ImportanceLevel.ADVANCED,
            order=42,
        )
    )

    # Timestep Bias End
    registry._add_field(
        ConfigField(
            name="timestep_bias_end",
            arg_name="--timestep_bias_end",
            ui_label="Timestep Bias End",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=1000,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1000, message="Must be <= 1000"),
            ],
            dependencies=[
                FieldDependency(field="timestep_bias_strategy", operator="not_equals", value="none", action="show")
            ],
            help_text="End of timestep bias range",
            tooltip="End of timestep range to bias towards. Only used with timestep bias strategies.",
            importance=ImportanceLevel.ADVANCED,
            order=43,
        )
    )

    # Timestep Bias Multiplier
    registry._add_field(
        ConfigField(
            name="timestep_bias_multiplier",
            arg_name="--timestep_bias_multiplier",
            ui_label="Timestep Bias Multiplier",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=1.0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be non-negative")],
            dependencies=[
                FieldDependency(field="timestep_bias_strategy", operator="not_equals", value="none", action="show")
            ],
            help_text="Multiplier for timestep bias probability",
            tooltip="Strength of timestep bias. Higher values = stronger bias to selected timesteps.",
            importance=ImportanceLevel.ADVANCED,
            order=44,
        )
    )

    # Timestep Bias Portion
    registry._add_field(
        ConfigField(
            name="timestep_bias_portion",
            arg_name="--timestep_bias_portion",
            ui_label="Timestep Bias Portion",
            field_type=FieldType.NUMBER,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value=0.25,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
            ],
            dependencies=[
                FieldDependency(field="timestep_bias_strategy", operator="not_equals", value="none", action="show")
            ],
            help_text="Portion of training steps to apply timestep bias",
            tooltip="Fraction of training where timestep bias is applied. 0.0 = entire training, 1.0 = only at end.",
            importance=ImportanceLevel.ADVANCED,
            order=45,
        )
    )

    # Training Scheduler Timestep Spacing
    registry._add_field(
        ConfigField(
            name="training_scheduler_timestep_spacing",
            arg_name="--training_scheduler_timestep_spacing",
            ui_label="Training Scheduler Timestep Spacing",
            field_type=FieldType.SELECT,
            tab="training",
            section="training_schedule",
            subsection="advanced",
            default_value="trailing",
            choices=[
                {"value": "leading", "label": "Leading"},
                {"value": "linspace", "label": "Linear"},
                {"value": "trailing", "label": "Trailing"},
            ],
            help_text="Timestep spacing for training scheduler",
            tooltip="How timesteps are spaced during training. 'trailing' is most common for diffusion models.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
        )
    )

    # Inference Scheduler Timestep Spacing
    registry._add_field(
        ConfigField(
            name="inference_scheduler_timestep_spacing",
            arg_name="--inference_scheduler_timestep_spacing",
            ui_label="Inference Scheduler Timestep Spacing",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_schedule",
            subsection="advanced",
            default_value="trailing",
            choices=[
                {"value": "leading", "label": "Leading"},
                {"value": "linspace", "label": "Linear"},
                {"value": "trailing", "label": "Trailing"},
            ],
            help_text="Timestep spacing for inference scheduler",
            tooltip="How timesteps are spaced during validation/inference. Should match training for consistency.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
        )
    )
