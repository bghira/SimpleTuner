from simpletuner.simpletuner_sdk.server.services.field_registry.types import ConfigField, FieldType, ImportanceLevel


def register_fields(registry) -> None:
    registry._add_field(
        ConfigField(
            name="minimax_music_train_component",
            arg_name="--minimax_music_train_component",
            ui_label="MiniMax Music Trained Component",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="transformer",
            choices=[
                {"value": "transformer", "label": "Music transformer (DiT)"},
                {"value": "language_model", "label": "Qwen3 language model (AR stage)"},
            ],
            help_text=(
                "Which MiniMax Music 3 component receives training. transformer trains the flow-matching music "
                "DiT on cached Flow-VAE latents. language_model trains the Qwen3 autoregressive stage with "
                "next-token cross-entropy on RVQ semantic codes; datasets must provide precomputed audio tokens "
                "(audio_tokens_path metadata) alongside caption and lyrics."
            ),
            tooltip=(
                "language_model mode is for style/keyword adaptation of the AR planner (dreambooth-style trigger "
                "words). Validation audio generation is disabled in this mode; render from checkpoints instead."
            ),
            importance=ImportanceLevel.ADVANCED,
            order=39,
            documentation="OPTIONS.md#--minimax_music_train_component",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_lm_max_frames",
            arg_name="--minimax_music_lm_max_frames",
            ui_label="MiniMax Music LM Max Audio Frames",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value=0,
            help_text=(
                "language_model training only: truncate each track's audio token sequence to this many 25Hz frames, "
                "taken from the start so lyrics stay aligned. 0 trains on full tracks. Truncated samples do not "
                "receive an end-of-audio target."
            ),
            tooltip="One frame is 40ms. 7500 frames = 5 minutes. Lower this if long tracks exhaust VRAM.",
            importance=ImportanceLevel.ADVANCED,
            order=40,
            documentation="OPTIONS.md#--minimax_music_lm_max_frames",
        )
    )
