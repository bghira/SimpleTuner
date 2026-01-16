import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from ..types import ConfigField, FieldDependency, FieldType, ImportanceLevel, ValidationRule, ValidationRuleType

if TYPE_CHECKING:
    from ..registry import FieldRegistry


logger = logging.getLogger(__name__)


def register_lora_fields(registry: "FieldRegistry") -> None:
    """Add LoRA configuration fields."""
    # LoRA Rank
    registry._add_field(
        ConfigField(
            name="lora_rank",
            arg_name="--lora_rank",
            ui_label="LoRA Rank",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value=16,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=1, message="LoRA rank must be at least 1"),
                ValidationRule(ValidationRuleType.MAX, value=256, message="LoRA rank >256 is very large"),
            ],
            dependencies=[FieldDependency(field="model_type", value="lora")],
            help_text="Dimension of LoRA update matrices",
            tooltip="Higher rank = more capacity but larger files. Common values: 4-32 for style, 64-128 for complex concepts",
            importance=ImportanceLevel.IMPORTANT,
            order=1,
        )
    )

    # LoRA Alpha
    registry._add_field(
        ConfigField(
            name="lora_alpha",
            arg_name="--lora_alpha",
            ui_label="LoRA Alpha",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.1, message="LoRA alpha should be positive")],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
            ],
            help_text="Scales the LoRA effect (usually equal to rank). Note: Diffusers stores alpha in safetensors metadata rather than as a tensor, which may cause issues in some tools. Use --lora_format=comfyui for scalar alpha tensors if needed.",
            tooltip="Controls the strength of LoRA's influence. Effective scale = alpha / rank. ComfyUI format stores alpha as a proper tensor for better compatibility.",
            importance=ImportanceLevel.ADVANCED,
            order=2,
        )
    )

    # LoRA Type
    registry._add_field(
        ConfigField(
            name="lora_type",
            arg_name="--lora_type",
            ui_label="LoRA Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value="standard",
            choices=[
                {"value": "standard", "label": "Standard PEFT LoRA"},
                {"value": "lycoris", "label": "LyCORIS (LoKr/LoHa/etc)"},
            ],
            dependencies=[FieldDependency(field="model_type", value="lora")],
            help_text="Standard (PEFT) has broad ecosystem compatibility. LyCORIS offers LoKr, LoHa, and other advanced methods that may train more easily but have less tool support.",
            tooltip="Standard LoRA uses the PEFT library. LyCORIS provides alternative decomposition algorithms with different training characteristics.",
            importance=ImportanceLevel.ADVANCED,
            order=3,
        )
    )

    # Slider-specific target selection
    registry._add_field(
        ConfigField(
            name="slider_lora_target",
            arg_name="--slider_lora_target",
            ui_label="Use slider LoRA targets",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=False,
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="Route LoRA training to slider-friendly modules (self-attn + conv/time embeddings).",
            tooltip="Enable this to match the Concept Sliders paper: focus on attn1/convolution/time-embedding layers.",
            importance=ImportanceLevel.ADVANCED,
            order=3.5,
        )
    )

    registry._add_field(
        ConfigField(
            name="peft_lora_target_modules",
            arg_name="--peft_lora_target_modules",
            ui_label="PEFT LoRA Target Modules",
            field_type=FieldType.TEXT,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=None,
            placeholder='["to_k", "to_q", "to_v", "to_out.0"]',
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="Override PEFT LoRA target modules with a JSON list or JSON file path.",
            tooltip="Provide a JSON array (or path to a JSON file) listing module names. Overrides preset targets.",
            importance=ImportanceLevel.ADVANCED,
            order=3.6,
        )
    )

    # LoRA Dropout
    registry._add_field(
        ConfigField(
            name="lora_dropout",
            arg_name="--lora_dropout",
            ui_label="LoRA Dropout",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=0.1,
            validation_rules=[
                ValidationRule(ValidationRuleType.MIN, value=0.0, message="LoRA dropout must be non-negative"),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="LoRA dropout must be <= 1.0"),
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="LoRA dropout randomly ignores neurons during training. This can help prevent overfitting.",
            tooltip="Randomly drops LoRA neurons during training to prevent overfitting. Typical values: 0.0-0.2.",
            importance=ImportanceLevel.ADVANCED,
            order=4,
        )
    )

    # LoRA Init Type
    registry._add_field(
        ConfigField(
            name="lora_init_type",
            arg_name="--lora_init_type",
            ui_label="LoRA Initialization Type",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value="default",
            choices=[
                {"value": "default", "label": "Default (Microsoft)"},
                {"value": "gaussian", "label": "Gaussian"},
                {"value": "loftq", "label": "LoftQ"},
                {"value": "olora", "label": "OLORA"},
                {"value": "pissa", "label": "PISSA"},
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="The initialization type for the LoRA model",
            tooltip="'default' uses Microsoft's method. 'gaussian' uses scaled distribution. 'loftq' quantization-aware init.",
            importance=ImportanceLevel.ADVANCED,
            order=5,
        )
    )

    # PEFT LoRA Mode
    registry._add_field(
        ConfigField(
            name="peft_lora_mode",
            arg_name="--peft_lora_mode",
            ui_label="PEFT LoRA Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value="standard",
            choices=[
                {"value": "standard", "label": "Standard LoRA"},
                {"value": "singlora", "label": "SingLoRA"},
            ],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="PEFT LoRA training mode",
            tooltip="Standard LoRA vs SingLoRA (more efficient representation). SingLoRA may require ramp-up steps.",
            importance=ImportanceLevel.ADVANCED,
            order=6,
        )
    )

    # SingLoRA Ramp Up Steps
    registry._add_field(
        ConfigField(
            name="singlora_ramp_up_steps",
            arg_name="--singlora_ramp_up_steps",
            ui_label="SingLoRA Ramp Up Steps",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Ramp up steps must be non-negative")],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
                FieldDependency(field="peft_lora_mode", value="singlora"),
            ],
            help_text="Number of ramp-up steps for SingLoRA",
            tooltip="For diffusion models, ramp-up steps may be harmful to training. Default: 0",
            importance=ImportanceLevel.ADVANCED,
            order=7,
        )
    )

    # Initialize LoRA
    registry._add_field(
        ConfigField(
            name="init_lora",
            arg_name="--init_lora",
            ui_label="Initialize from LoRA",
            field_type=FieldType.TEXT,
            tab="model",
            section="lora_config",
            subsection="advanced",
            placeholder="path/to/existing_lora.safetensors",
            dependencies=[FieldDependency(field="model_type", value="lora")],
            help_text="Specify an existing LoRA or LyCORIS safetensors file to initialize the adapter",
            tooltip="Load existing LoRA weights to continue training. Useful for resuming or fine-tuning existing models.",
            importance=ImportanceLevel.ADVANCED,
            order=8,
        )
    )

    assistant_dependencies = [
        FieldDependency(field="model_type", value="lora"),
        FieldDependency(field="model_family", operator="in", values=["flux", "z-image"]),
        FieldDependency(field="model_flavour", operator="in", values=["schnell", "turbo"]),
    ]

    registry._add_field(
        ConfigField(
            name="assistant_lora_path",
            arg_name="--assistant_lora_path",
            ui_label="Assistant LoRA Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=None,
            dependencies=assistant_dependencies,
            help_text="Optional frozen assistant LoRA applied during training (Flux schnell / Z-Image turbo).",
            tooltip="Provide a safetensors path or Hub repo for the assistant adapter. Disabled when empty.",
            importance=ImportanceLevel.ADVANCED,
            order=8.5,
        )
    )

    registry._add_field(
        ConfigField(
            name="assistant_lora_strength",
            arg_name="--assistant_lora_strength",
            ui_label="Assistant LoRA Strength",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=1.0,
            dependencies=assistant_dependencies,
            help_text="Scale applied to the assistant adapter during training.",
            tooltip="Set to 0 to ignore the assistant adapter during training.",
            importance=ImportanceLevel.ADVANCED,
            order=8.6,
        )
    )

    registry._add_field(
        ConfigField(
            name="assistant_lora_inference_strength",
            arg_name="--assistant_lora_inference_strength",
            ui_label="Assistant LoRA Inference Strength",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=0.0,
            dependencies=assistant_dependencies,
            help_text="Scale for the assistant adapter during validation/inference (default disables it).",
            tooltip="Keep at 0.0 to strip assistant influence from validation outputs.",
            importance=ImportanceLevel.ADVANCED,
            order=8.7,
        )
    )

    registry._add_field(
        ConfigField(
            name="disable_assistant_lora",
            arg_name="--disable_assistant_lora",
            ui_label="Disable Assistant LoRA",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=False,
            dependencies=assistant_dependencies,
            checkbox_label="Disable Assistant LoRA",
            help_text="Completely disable assistant LoRA auto-configuration.",
            tooltip="When enabled, assistant LoRA will not be loaded or applied even for supported flavours.",
            importance=ImportanceLevel.ADVANCED,
            order=8.8,
        )
    )

    registry._add_field(
        ConfigField(
            name="lora_format",
            arg_name="--lora_format",
            ui_label="LoRA Format",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value="diffusers",
            choices=[
                {"value": "diffusers", "label": "Diffusers / PEFT"},
                {"value": "comfyui", "label": "ComfyUI (diffusion_model.*)"},
            ],
            dependencies=[FieldDependency(field="model_type", value="lora")],
            help_text="Select the LoRA checkpoint key format for load/save.",
            tooltip="Choose 'comfyui' to read/write diffusion_model.* keys with lora_A/lora_B weights and .alpha tensors.",
            importance=ImportanceLevel.ADVANCED,
            order=8.9,
            documentation="OPTIONS.md#--lora_format",
        )
    )

    # LyCORIS Config
    registry._add_field(
        ConfigField(
            name="lycoris_config",
            arg_name="--lycoris_config",
            ui_label="LyCORIS Config",
            field_type=FieldType.TEXT,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value="config/lycoris_config.json",
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="lycoris"),
            ],
            help_text="Path to LyCORIS configuration JSON file",
            tooltip="Configuration file for LyCORIS training parameters and network architecture.",
            importance=ImportanceLevel.ADVANCED,
            order=9,
        )
    )

    # Init LoKR Norm
    registry._add_field(
        ConfigField(
            name="init_lokr_norm",
            arg_name="--init_lokr_norm",
            ui_label="LoKR Normalization Init",
            field_type=FieldType.NUMBER,
            tab="model",
            section="lora_config",
            subsection="basic",
            default_value=None,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be positive")],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="lycoris"),
            ],
            help_text="Perturbed normal initialization for LyCORIS LoKr layers",
            tooltip="Good values: 1e-4 to 1e-2. Enables perturbed normal initialization for better training.",
            importance=ImportanceLevel.ADVANCED,
            placeholder="1e-3",
            order=10,
        )
    )

    # Flux LoRA Target
    flux_targets = [
        "mmdit",
        "context",
        "context+ffs",
        "all",
        "all+ffs",
        "ai-toolkit",
        "tiny",
        "nano",
        "controlnet",
        "all+ffs+embedder",
        "all+ffs+embedder+controlnet",
    ]
    registry._add_field(
        ConfigField(
            name="flux_lora_target",
            arg_name="--flux_lora_target",
            ui_label="Flux LoRA Target Layers",
            field_type=FieldType.SELECT,
            tab="model",
            section="lora_config",
            subsection="model_specific",
            default_value="all",
            choices=[{"value": t, "label": t} for t in flux_targets],
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="model_family", value="flux"),
            ],
            help_text="Which layers to train in Flux models",
            tooltip="'all' trains all attention layers. 'context' only trains text layers. '+ffs' includes feed-forward layers.",
            importance=ImportanceLevel.ADVANCED,
            model_specific=["flux"],
            order=10,
        )
    )

    # ACE-Step LoRA Target
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

    # Use DoRA
    registry._add_field(
        ConfigField(
            name="use_dora",
            arg_name="--use_dora",
            ui_label="Use DoRA",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="lora_config",
            subsection="advanced",
            default_value=False,
            dependencies=[
                FieldDependency(field="model_type", value="lora"),
                FieldDependency(field="lora_type", value="standard"),
            ],
            help_text="Weight-Decomposed LoRA separates weight magnitude from direction for potentially better quality. Only works with Standard lora_type; for LyCORIS, enable DoRA in your lycoris_config.json instead.",
            tooltip="DoRA can improve training results but adds computational overhead. Must use Standard LoRA; LyCORIS users should configure DoRA directly in their config file.",
            warning="Only compatible with Standard lora_type - not LyCORIS",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=20,
        )
    )
