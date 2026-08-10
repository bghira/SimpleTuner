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
            name="minimax_h3_reference_mode",
            arg_name="--minimax_h3_reference_mode",
            ui_label="MiniMax H3 Reference Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value="vanilla",
            choices=[
                {"value": "vanilla", "label": "Vanilla"},
                {"value": "cached_kv", "label": "Cached KV"},
            ],
            help_text="Choose MiniMax-H3 reference handling. cached_kv is an experimental no-grad inference mode that treats text plus conditioning rows as static memory and reuses their K/V projections.",
            tooltip="Vanilla preserves the stock full packed self-attention path. Cached KV is experimental and intended for reference-conditioning inference comparisons.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=37,
            documentation="OPTIONS.md#--minimax_h3_reference_mode",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_target_mode",
            arg_name="--minimax_h3_target_mode",
            ui_label="MiniMax H3 Target Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value="auto",
            choices=[
                {"value": "auto", "label": "Auto"},
                {"value": "video", "label": "Video only"},
                {"value": "av", "label": "Audio + video"},
            ],
            help_text=(
                "Controls whether MiniMax-H3 includes target audio rows. auto resolves to video-only unless the "
                "global config or data backend sets minimax_h3_target_mode/h3_target_mode to av."
            ),
            tooltip=(
                "Use video to ignore auto-split or explicit audio backends and save VRAM; use av for joint "
                "audio-video training or sampling."
            ),
            importance=ImportanceLevel.ADVANCED,
            order=38,
            documentation="OPTIONS.md#--minimax_h3_target_mode",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_sparse_attention",
            arg_name="--minimax_h3_sparse_attention",
            ui_label="MiniMax H3 Sparse Attention",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value="disabled",
            choices=[
                {"value": "disabled", "label": "Disabled"},
                {"value": "moba3d", "label": "Experimental 3D MoBA"},
            ],
            help_text=(
                "Enable experimental train-aware 3D block sparse attention for target-video tokens. Text, audio, "
                "references, and other packed context remain dense."
            ),
            tooltip="The exact MiniMax production routing configuration has not been released.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=39,
            documentation="quickstart/MINIMAX_H3.md#experimental-sparse-attention",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_sparse_block_shape",
            arg_name="--minimax_h3_sparse_block_shape",
            ui_label="MiniMax H3 Sparse Block Shape",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value="1,8,16",
            help_text="Temporal, height, and width dimensions of each 128-token sparse video block.",
            tooltip="Examples: 1,8,16; 2,8,8; 4,4,8. The dimensions must multiply to 128.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=40,
            documentation="quickstart/MINIMAX_H3.md#experimental-sparse-attention",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_sparse_video_kv_fraction",
            arg_name="--minimax_h3_sparse_video_kv_fraction",
            ui_label="MiniMax H3 Sparse Video KV Fraction",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value=0.5,
            validation_rules=[
                ValidationRule(
                    ValidationRuleType.MIN,
                    value=0.001,
                    message="Must be greater than zero",
                ),
                ValidationRule(ValidationRuleType.MAX, value=1.0, message="Must be at most 1"),
            ],
            help_text="Fraction of target-video key/value blocks selected for every target-video query block.",
            tooltip="A value of 1.0 is the dense numerical control through the sparse kernel.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=41,
            step=0.05,
            documentation="quickstart/MINIMAX_H3.md#experimental-sparse-attention",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_sparse_share_heads",
            arg_name="--minimax_h3_sparse_share_heads",
            ui_label="MiniMax H3 Share Sparse Routes Across Heads",
            field_type=FieldType.CHECKBOX,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value=False,
            help_text="Average block probes across attention heads and use one selected route per sample.",
            tooltip="Disabled selects video blocks independently for each attention head.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=42,
            documentation="quickstart/MINIMAX_H3.md#experimental-sparse-attention",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_h3_sparse_start_layer",
            arg_name="--minimax_h3_sparse_start_layer",
            ui_label="MiniMax H3 Sparse Start Layer",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["minimaxh3"],
            default_value=0,
            validation_rules=[ValidationRule(ValidationRuleType.MIN, value=0, message="Must be non-negative")],
            help_text="Keep transformer layers below this zero-based index on dense attention.",
            tooltip="Use 0 to apply sparse routing to every transformer layer.",
            importance=ImportanceLevel.EXPERIMENTAL,
            order=43,
            step=1,
            documentation="quickstart/MINIMAX_H3.md#experimental-sparse-attention",
        )
    )
