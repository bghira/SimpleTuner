from simpletuner.simpletuner_sdk.server.services.field_registry.types import (
    ConfigField,
    FieldType,
    ImportanceLevel,
    ValidationRule,
    ValidationRuleType,
)


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="ltx2_validation_pipeline_mode",
            arg_name="--ltx2_validation_pipeline_mode",
            ui_label="LTX-2 Validation Pipeline Mode",
            field_type=FieldType.SELECT,
            tab="validation",
            section="validation_options",
            model_specific=["ltxvideo2"],
            default_value="trained-stage",
            choices=[
                {"value": "trained-stage", "label": "Trained Stage Only"},
                {"value": "spatial-upscale", "label": "Spatial Upscale Pipeline"},
            ],
            help_text="Choose whether LTX-2 validation runs only the trained model or runs a half-resolution generation plus latent spatial upscaler refinement.",
            tooltip="Spatial upscale runs stage 1 at half resolution, applies the LTX-2 spatial latent upscaler, then re-denoises at the requested validation resolution.",
            importance=ImportanceLevel.ADVANCED,
            order=19,
            documentation="OPTIONS.md#--ltx2_validation_pipeline_mode",
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx2_validation_spatial_upsampler_model",
            arg_name="--ltx2_validation_spatial_upsampler_model",
            ui_label="LTX-2 Validation Spatial Upsampler Model",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            model_specific=["ltxvideo2"],
            default_value="Lightricks/LTX-2.3",
            placeholder="Lightricks/LTX-2.3 or /path/to/upsampler.safetensors",
            help_text="Hugging Face repo, local directory, or local safetensors file for the LTX-2 spatial latent upsampler.",
            tooltip="Used only when ltx2_validation_pipeline_mode is spatial-upscale.",
            importance=ImportanceLevel.ADVANCED,
            order=20,
            documentation="OPTIONS.md#--ltx2_validation_spatial_upsampler_model",
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx2_validation_spatial_upsampler_filename",
            arg_name="--ltx2_validation_spatial_upsampler_filename",
            ui_label="LTX-2 Validation Spatial Upsampler Filename",
            field_type=FieldType.TEXT,
            tab="validation",
            section="validation_options",
            model_specific=["ltxvideo2"],
            default_value="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            placeholder="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            help_text="Upsampler safetensors filename when the spatial upsampler model is a repo or directory.",
            tooltip="Ignored when ltx2_validation_spatial_upsampler_model points directly to a safetensors file.",
            importance=ImportanceLevel.ADVANCED,
            order=21,
            documentation="OPTIONS.md#--ltx2_validation_spatial_upsampler_filename",
        )
    )

    registry._add_field(
        ConfigField(
            name="ltx2_intrinsic_conditioning",
            arg_name="--ltx2_intrinsic_conditioning",
            ui_label="LTX-2 Intrinsic Conditioning",
            field_type=FieldType.TEXT_JSON,
            tab="model",
            section="model_specific",
            model_specific=["ltxvideo2"],
            default_value=None,
            help_text="JSON array of LTX-2 intrinsic clean-token conditioning objects",
            tooltip="Advanced LTX-2 training config. Supported condition types: first_frame, prefix, suffix, spatial_crop, mask.",
            importance=ImportanceLevel.ADVANCED,
            order=26,
            documentation="OPTIONS.md#ltx-2-intrinsic-and-reference-conditioning",
        )
    )

    for offset, field_name, label, help_text in (
        (
            0,
            "ltx2_first_frame_conditioning_probability",
            "LTX-2 First Frame Conditioning Probability",
            "Probability of replacing first-frame target tokens with clean latents and removing them from video loss",
        ),
        (
            1,
            "ltx2_prefix_conditioning_probability",
            "LTX-2 Prefix Conditioning Probability",
            "Probability of replacing prefix target tokens with clean latents and removing them from video loss",
        ),
        (
            2,
            "ltx2_suffix_conditioning_probability",
            "LTX-2 Suffix Conditioning Probability",
            "Probability of replacing suffix target tokens with clean latents and removing them from video loss",
        ),
        (
            3,
            "ltx2_mask_conditioning_probability",
            "LTX-2 Mask Conditioning Probability",
            "Probability of using mask=1 target tokens as clean conditioning with no video loss",
        ),
    ):
        registry._add_field(
            ConfigField(
                name=field_name,
                arg_name=f"--{field_name}",
                ui_label=label,
                field_type=FieldType.NUMBER,
                tab="model",
                section="model_specific",
                model_specific=["ltxvideo2"],
                default_value=0.0,
                validation_rules=[
                    ValidationRule(ValidationRuleType.MIN, value=0.0, message="Must be between 0 and 1"),
                    ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be between 0 and 1"),
                ],
                help_text=help_text,
                tooltip=help_text,
                importance=ImportanceLevel.ADVANCED,
                order=27 + offset,
                documentation="OPTIONS.md#ltx-2-intrinsic-and-reference-conditioning",
            )
        )

    for offset, field_name, label, default_value in (
        (0, "ltx2_prefix_conditioning_frames", "LTX-2 Prefix Conditioning Frames", 1),
        (1, "ltx2_suffix_conditioning_frames", "LTX-2 Suffix Conditioning Frames", 1),
        (2, "ltx2_reference_spatial_scale_factor", "LTX-2 Reference Spatial Scale Factor", None),
        (3, "ltx2_reference_temporal_scale_factor", "LTX-2 Reference Temporal Scale Factor", 1),
    ):
        registry._add_field(
            ConfigField(
                name=field_name,
                arg_name=f"--{field_name}",
                ui_label=label,
                field_type=FieldType.NUMBER,
                tab="model",
                section="model_specific",
                model_specific=["ltxvideo2"],
                default_value=default_value,
                validation_rules=[ValidationRule(ValidationRuleType.MIN, value=1, message="Must be at least 1")],
                help_text=f"Advanced LTX-2 conditioning setting: {label.lower()}",
                tooltip=f"Advanced LTX-2 conditioning setting: {label.lower()}",
                importance=ImportanceLevel.ADVANCED,
                order=31 + offset,
                documentation="OPTIONS.md#ltx-2-intrinsic-and-reference-conditioning",
            )
        )

    registry._add_field(
        ConfigField(
            name="validation_audio_only",
            arg_name="--validation_audio_only",
            ui_label="Validation Audio Only",
            field_type=FieldType.CHECKBOX,
            tab="validation",
            section="validation_options",
            default_value=False,
            help_text="Disable video generation during validation and emit audio only.",
            tooltip="LTX-2 only: skips video generation during validation so only audio outputs are produced.",
            importance=ImportanceLevel.ADVANCED,
            order=17,
            model_specific=["ltxvideo2"],
        )
    )

    registry._add_field(
        ConfigField(
            name="validation_ltx2_video_conditioning",
            arg_name="--validation_ltx2_video_conditioning",
            ui_label="LTX-2 Validation Video Conditioning",
            field_type=FieldType.TEXT_JSON,
            tab="validation",
            section="validation_options",
            default_value=None,
            help_text="JSON list of IC-LoRA reference videos for LTX-2 validation",
            tooltip="LTX-2 only: pass reference videos as paths, [path, strength] pairs, or objects with path/video_path and optional strength.",
            importance=ImportanceLevel.ADVANCED,
            order=18,
            model_specific=["ltxvideo2"],
            documentation="OPTIONS.md#ltx-2-conditioning-options",
        )
    )
