from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


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
