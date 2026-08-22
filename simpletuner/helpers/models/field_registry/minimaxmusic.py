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
                "next-token cross-entropy on RVQ semantic codes; raw audio datasets are encoded through the "
                "MiniMax Music RVQ cache encoder unless the dataset provides audio_tokens or audio_tokens_path "
                "metadata alongside caption and lyrics."
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
            name="minimax_music_rvq_encoder_model_name_or_path",
            arg_name="--minimax_music_rvq_encoder_model_name_or_path",
            ui_label="MiniMax Music RVQ Encoder",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="SimpleTuner/open-rvq-encoder-minimax-music3-169m-v4",
            help_text=(
                "MiniMax Music language_model training only: model repository or local directory containing the "
                "RVQ encoder config and weights used to convert cached DAV audio latents into per-codebook codes."
            ),
            tooltip="The default is SimpleTuner's v4 open RVQ encoder for MiniMax Music 3.",
            importance=ImportanceLevel.ADVANCED,
            order=44,
            documentation="OPTIONS.md#--minimax_music_rvq_encoder_model_name_or_path",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_rvq_encoder_subfolder",
            arg_name="--minimax_music_rvq_encoder_subfolder",
            ui_label="MiniMax Music RVQ Encoder Subfolder",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="final",
            help_text=(
                "Subfolder within the RVQ encoder repository or local directory that contains "
                "rvq_encoder_config.json, rvq_encoder.safetensors, and any muP base-shape metadata."
            ),
            tooltip="Leave at final for the default SimpleTuner v4 RVQ encoder package.",
            importance=ImportanceLevel.ADVANCED,
            order=45,
            documentation="OPTIONS.md#--minimax_music_rvq_encoder_subfolder",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_rvq_encoder_revision",
            arg_name="--minimax_music_rvq_encoder_revision",
            ui_label="MiniMax Music RVQ Encoder Revision",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="",
            help_text=(
                "Optional Hub revision for the RVQ encoder. If unset, SimpleTuner uses the main training revision "
                "setting when one is provided."
            ),
            tooltip="Use this only when pinning a specific RVQ encoder checkpoint revision.",
            importance=ImportanceLevel.ADVANCED,
            order=46,
            documentation="OPTIONS.md#--minimax_music_rvq_encoder_revision",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_rvq_vae_model_name_or_path",
            arg_name="--minimax_music_rvq_vae_model_name_or_path",
            ui_label="MiniMax Music RVQ Audio VAE",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="",
            help_text=(
                "MiniMax Music language_model training only: DAV/audio VAE repository, local directory, or dav.pth "
                "file used before the RVQ encoder. If unset in config, the standard pretrained VAE path is used "
                "before this default."
            ),
            tooltip="This is the audio encoder stage, not the RVQ code predictor.",
            importance=ImportanceLevel.ADVANCED,
            order=47,
            documentation="OPTIONS.md#--minimax_music_rvq_vae_model_name_or_path",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_lm_adapter",
            arg_name="--minimax_music_lm_adapter",
            ui_label="MiniMax Music LM Adapter Path",
            field_type=FieldType.TEXT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="",
            help_text=(
                "Path to a language-model LoRA (pytorch_lora_weights.safetensors with language_model.-prefixed "
                "keys) to apply to the Qwen3 planner while pre-caching conditioning for DiT training, so the "
                "cached hidden states reflect the adapted planner."
            ),
            tooltip="Produced by --minimax_music_train_component=language_model runs.",
            importance=ImportanceLevel.ADVANCED,
            order=41,
            documentation="OPTIONS.md#--minimax_music_lm_adapter",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_lm_adapter_strength",
            arg_name="--minimax_music_lm_adapter_strength",
            ui_label="MiniMax Music LM Adapter Strength",
            field_type=FieldType.NUMBER,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value=1.0,
            help_text="Scale applied to the LM adapter's delta while pre-caching (lora_B weights are scaled).",
            tooltip="1.0 is full strength.",
            importance=ImportanceLevel.ADVANCED,
            order=42,
            documentation="OPTIONS.md#--minimax_music_lm_adapter",
        )
    )
    registry._add_field(
        ConfigField(
            name="minimax_music_lm_precache_mode",
            arg_name="--minimax_music_lm_precache_mode",
            ui_label="MiniMax Music LM Precache Mode",
            field_type=FieldType.SELECT,
            tab="model",
            section="model_specific",
            model_specific=["minimaxmusic"],
            default_value="text-only",
            choices=[
                {"value": "text-only", "label": "Text-only rollout (default)"},
                {"value": "audio-only", "label": "Teacher-forced audio codes"},
                {"value": "audio+text", "label": "Text prefix + teacher-forced audio codes"},
            ],
            help_text=(
                "How the Qwen3 planner produces the cached DiT conditioning. text-only samples an autoregressive "
                "rollout from the caption and lyrics (the stock behaviour). The teacher-forced modes feed the "
                "sample's ground-truth RVQ codes (audio_tokens_path metadata) through the planner instead, so the "
                "per-frame hidden states line up one-to-one with the DAV latents the DiT is trained against."
            ),
            tooltip=(
                "Teacher-forced modes require raw per-codebook audio tokens in the dataset metadata and truncate "
                "them to the same audio window the VAE cache covers."
            ),
            importance=ImportanceLevel.ADVANCED,
            order=43,
            documentation="OPTIONS.md#--minimax_music_lm_precache_mode",
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
