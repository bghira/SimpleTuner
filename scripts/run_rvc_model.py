#!/usr/bin/env python3
"""Train or reuse a SimpleTuner RVC identity model and convert an audio directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from simpletuner.helpers.data_transforms import process_data_transforms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SimpleTuner's RVC identity-transfer transform outside a training dataloader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory of audio files to convert.")
    parser.add_argument(
        "--identity-dir",
        type=Path,
        default=None,
        help="Directory of target identity audio used to train an RVC artifact when no compatible artifact exists.",
    )
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory where converted audio is written.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="SimpleTuner output directory used for RVC logs and default cache paths.",
    )
    parser.add_argument("--transform-id", default="rvc-identity-transfer", help="Generated backend/transform id.")
    parser.add_argument("--source-id", default="rvc-source", help="Synthetic source backend id.")
    parser.add_argument(
        "--model-cache-dir", type=Path, default=None, help="Directory containing or receiving the RVC artifact."
    )
    parser.add_argument("--model-name", default=None, help="Human-readable name stored in the RVC artifact metadata.")

    parser.add_argument("--sample-rate", type=int, default=48000, help="RVC sample rate. Only 48000 is supported today.")
    parser.add_argument("--channels", type=int, default=2, help="Generated dataset channel count metadata.")
    parser.add_argument(
        "--asset-hub-model-id", default="lj1995/VoiceConversionWebUI", help="HF repo with default RVC assets."
    )
    parser.add_argument("--asset-hub-token", default=None, help="Optional token for the RVC asset repo.")
    parser.add_argument(
        "--hub-model-id", default=None, help="Optional HF model repo for a reusable SimpleTuner RVC artifact."
    )
    parser.add_argument("--hub-token", default=None, help="Optional token for artifact hub reuse or upload.")
    parser.add_argument("--reuse-from-hub", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-to-hub", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--public",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Publish the reusable RVC artifact as a public Hub repo.",
    )
    parser.add_argument("--train-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--build-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--flat-index-threshold", type=int, default=100000, help="Use a flat FAISS index below this frame count."
    )

    parser.add_argument(
        "--identity-audio-mode",
        choices=("separate", "vocal_only"),
        default="separate",
        help="How identity clips are prepared before RVC feature extraction.",
    )
    parser.add_argument("--training-steps", type=int, default=1000, help="RVC generator/discriminator training steps.")
    parser.add_argument("--batch-size", type=int, default=4, help="RVC training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="RVC AdamW learning rate.")
    parser.add_argument("--max-seconds-per-file", type=float, default=180.0, help="Maximum identity audio seconds per file.")
    parser.add_argument(
        "--identity-stem-debug-dir",
        type=Path,
        default=None,
        help="Optional directory for identity vocal/accompaniment stem previews.",
    )
    parser.add_argument("--pretrained-generator-path", type=Path, default=None)
    parser.add_argument("--pretrained-discriminator-path", type=Path, default=None)
    parser.add_argument("--rmvpe-model-path", type=Path, default=None)
    parser.add_argument("--hubert-model-path", type=Path, default=None)

    parser.add_argument(
        "--audio-mode",
        choices=("separate_convert_remix", "vocal_only", "full_mix_convert"),
        default="separate_convert_remix",
        help="How source audio is converted.",
    )
    parser.add_argument("--separation-method", choices=("demucs",), default="demucs")
    parser.add_argument("--demucs-model", default="htdemucs", help="Demucs model used for two-stem vocal separation.")
    parser.add_argument("--device", default=None, help="RVC train/convert device, e.g. cpu, cuda, mps.")
    parser.add_argument("--demucs-device", default=None, help="Demucs device, e.g. cpu, cuda, mps.")
    parser.add_argument("--retrieval-strength", type=float, default=0.75, help="RVC retrieval/index blend strength.")
    parser.add_argument(
        "--timbre-strength", type=float, default=1.0, help="Blend between source waveform and converted waveform."
    )
    parser.add_argument("--torch-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--is-half", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--dry-run", action="store_true", help="Print the generated transform config without running it.")
    parser.add_argument("--print-config", action="store_true", help="Print the generated transform config before running.")
    return parser


def _set_if_present(config: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        config[key] = str(value) if isinstance(value, Path) else value


def build_data_backend_config(args: argparse.Namespace) -> list[dict[str, Any]]:
    model_cache_dir = args.model_cache_dir or args.output_dir / "cache" / "data_transforms" / args.transform_id / "rvc_model"
    model: dict[str, Any] = {
        "cache_dir": str(model_cache_dir),
        "train_if_missing": args.train_if_missing,
        "force_retrain": args.force_retrain,
        "build_index": args.build_index,
        "reuse_from_hub": args.reuse_from_hub,
        "push_to_hub": args.push_to_hub,
        "public": args.public,
        "asset_hub_model_id": args.asset_hub_model_id,
        "model_name": args.model_name
        or (args.hub_model_id.rstrip("/").rsplit("/", 1)[-1] if args.hub_model_id else args.transform_id),
        "sample_rate": args.sample_rate,
        "identity_audio_mode": args.identity_audio_mode,
        "separation_method": args.separation_method,
        "training_steps": args.training_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_seconds_per_file": args.max_seconds_per_file,
        "flat_index_threshold": args.flat_index_threshold,
    }
    _set_if_present(model, "identity_data_dir", args.identity_dir)
    _set_if_present(model, "asset_hub_token", args.asset_hub_token)
    _set_if_present(model, "hub_model_id", args.hub_model_id)
    _set_if_present(model, "hub_token", args.hub_token)
    _set_if_present(model, "device", args.device)
    _set_if_present(model, "demucs_device", args.demucs_device)
    _set_if_present(model, "demucs_model", args.demucs_model)
    _set_if_present(model, "identity_stem_debug_dir", args.identity_stem_debug_dir)
    _set_if_present(model, "pretrained_generator_path", args.pretrained_generator_path)
    _set_if_present(model, "pretrained_discriminator_path", args.pretrained_discriminator_path)
    _set_if_present(model, "rmvpe_model_path", args.rmvpe_model_path)
    _set_if_present(model, "hubert_model_path", args.hubert_model_path)
    _set_if_present(model, "is_half", args.is_half)

    conversion: dict[str, Any] = {
        "audio_mode": args.audio_mode,
        "separation_method": args.separation_method,
        "demucs_model": args.demucs_model,
        "retrieval_strength": args.retrieval_strength,
        "timbre_strength": args.timbre_strength,
        "torch_retrieval": args.torch_retrieval,
    }
    _set_if_present(conversion, "device", args.device)
    _set_if_present(conversion, "demucs_device", args.demucs_device)
    _set_if_present(conversion, "is_half", args.is_half)

    return [
        {
            "id": args.source_id,
            "type": "local",
            "dataset_type": "audio",
            "metadata_backend": "discovery",
            "caption_strategy": "textfile",
            "instance_data_dir": str(args.source_dir),
            "audio": {"sample_rate": args.sample_rate, "channels": args.channels, "audio_only": True},
            "data_transforms": [
                {
                    "id": args.transform_id,
                    "task": "identity_transfer",
                    "method": "rvc",
                    "model": model,
                    "conversion": conversion,
                    "target": {
                        "id": args.transform_id,
                        "type": "local",
                        "dataset_type": "audio",
                        "metadata_backend": "discovery",
                        "caption_strategy": "textfile",
                        "instance_data_dir": str(args.generated_dir),
                        "audio": {"sample_rate": args.sample_rate, "channels": args.channels, "audio_only": True},
                    },
                }
            ],
        }
    ]


def validate_args(args: argparse.Namespace) -> None:
    if not args.source_dir.exists():
        raise FileNotFoundError(f"--source-dir does not exist: {args.source_dir}")
    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"--source-dir is not a directory: {args.source_dir}")
    if args.sample_rate != 48000:
        raise ValueError("SimpleTuner RVC currently supports --sample-rate 48000 only.")
    if args.train_if_missing and args.identity_dir is None:
        raise ValueError("--identity-dir is required when --train-if-missing is enabled.")
    if args.identity_dir is not None and not args.identity_dir.exists():
        raise FileNotFoundError(f"--identity-dir does not exist: {args.identity_dir}")


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    validate_args(args)
    config = build_data_backend_config(args)
    if args.print_config or args.dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
    if args.dry_run:
        return config
    result = process_data_transforms(SimpleNamespace(output_dir=str(args.output_dir)), config)
    generated = sorted(args.generated_dir.rglob("*.wav"))
    print(json.dumps({"backend_count": len(result), "generated_dir": str(args.generated_dir)}, indent=2))
    for path in generated:
        print(path.relative_to(args.generated_dir))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
