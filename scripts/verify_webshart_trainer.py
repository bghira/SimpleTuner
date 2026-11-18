"""Exercise native dataset mixing through the production trainer on two CUDA ranks."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def make_model(path, clip):
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    torch.manual_seed(1729)
    name = clip
    text = CLIPTextModel.from_pretrained(name, local_files_only=True)
    tok = CLIPTokenizer.from_pretrained(name, local_files_only=True)
    unet = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        cross_attention_dim=text.config.hidden_size,
    )
    vae = AutoencoderKL(
        block_out_channels=(32, 32, 32, 32),
        layers_per_block=1,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
    )
    StableDiffusionPipeline(
        vae=vae,
        text_encoder=text,
        tokenizer=tok,
        unet=unet,
        scheduler=DDIMScheduler(),
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).save_pretrained(path)


def train(root, resume, prefetch, case, checkpoint_step, conditioning):
    from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
    from simpletuner.helpers.training.state_tracker import StateTracker
    from simpletuner.train import Trainer, _run_training

    run = root / case
    from contextlib import ExitStack
    from unittest.mock import patch

    import torch

    if case == "deterministic":
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)

    acknowledge = MultiAspectSampler.acknowledge_batch
    traverse = MultiAspectSampler._iterate_native
    prepare = Trainer.prepare_batch
    compute_loss = Trainer._compute_model_prediction_loss
    losses = []
    diagnostics = {}
    occurrences = {}
    inputs = {}

    def traverse_with_trace(sampler):
        for batch in traverse(sampler):
            token = batch[0]["dataset_continuation"]
            choices = [
                (entry["image_path"], entry.get("instance_prompt_text"), entry.get("caption_dropout_seed"))
                for entry in batch
                if isinstance(entry, dict)
            ]
            references = [entry for entry in batch if not isinstance(entry, dict)]
            if conditioning and not references:
                raise AssertionError("A conditioned batch must include a selected reference")
            choices.extend(
                (entry.data_backend_id, entry.image_path(), entry.caption, entry.training_sample_path(sampler.id))
                for entry in references
            )
            occurrences[sampler.id, token["layout"]["epoch"], token["position"]] = hashlib.sha256(
                json.dumps(choices).encode()
            ).hexdigest()
            yield batch

    def prepare_with_trace(trainer, raw):
        result = prepare(trainer, raw)
        inputs.clear()
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, torch.Tensor):
                    data = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
                    inputs[key] = hashlib.sha256(data).hexdigest()
        return result

    def loss_with_trace(trainer, prepared):
        def digest(tensors):
            result = hashlib.sha256()
            for tensor in tensors:
                result.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy())
            return result.hexdigest()

        model = trainer.model.get_trained_component(unwrap_model=False)
        diagnostics["parameters"] = digest(v for v in model.parameters() if v.requires_grad)
        diagnostics["frozen_parameters"] = digest(v for v in model.parameters() if not v.requires_grad)
        diagnostics["moments"] = digest(
            v for state in trainer.optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)
        )
        diagnostics["gpu_rng"] = digest([torch.cuda.get_rng_state()])
        diagnostics["lr"] = trainer.optimizer.param_groups[0]["lr"]
        diagnostics["lora_scales"] = sorted(
            {float(scale) for layer in model.modules() if hasattr(layer, "lora_A") for scale in layer.scaling.values()}
        )
        result = compute_loss(trainer, prepared)
        losses.append(float(result[0].detach()))
        return result

    trace = run / f"consumed-{resume}-{os.environ['RANK']}.jsonl"

    def record(sampler, token):
        acknowledge(sampler, token)
        with trace.open("a") as out:
            out.write(
                json.dumps(
                    [
                        token,
                        StateTracker.consumed_dataset_mixer,
                        occurrences.pop((sampler.id, token["layout"]["epoch"], token["position"])),
                        dict(inputs),
                        list(losses),
                        dict(diagnostics),
                    ]
                )
                + "\n"
            )

        losses.clear()

    output = run / ("resume" if resume else "baseline")
    config = dict(
        __skip_config_fallback__=True,
        model_family="sd1x",
        model_flavour="1.5",
        model_type="lora",
        pretrained_model_name_or_path=str(root / "model"),
        data_backend_config=str(root / "data.json"),
        output_dir=str(output),
        cache_dir=str(root / "cache"),
        report_to="none",
        push_to_hub=False,
        mixed_precision="bf16" if case == "bf16" else "no",
        optimizer="adamw_bf16" if case == "bf16" else "torch-adamw",
        learning_rate=0.0001,
        lr_scheduler="constant",
        lora_rank=4,
        checkpoint_step_interval=checkpoint_step,
        max_train_steps=4,
        num_train_epochs=0,
        seed=81,
        train_batch_size=1,
        gradient_accumulation_steps=1,
        validation_disable=True,
        resolution=64,
        resolution_type="pixel",
        disable_bucket_pruning=True,
        dataloader_prefetch=prefetch,
        dataloader_prefetch_qlen=2,
        aspect_bucket_alignment=8,
        caption_dropout_probability=0.1,
        conditioning_multidataset_sampling="random",
        vae_batch_size=2,
        write_batch_size=2,
    )
    if resume:
        config["resume_from_checkpoint"] = f"checkpoint-{checkpoint_step}"
    with ExitStack() as stack:
        stack.enter_context(patch.object(MultiAspectSampler, "_iterate_native", traverse_with_trace))
        stack.enter_context(patch.object(MultiAspectSampler, "acknowledge_batch", record))
        stack.enter_context(patch.object(Trainer, "prepare_batch", prepare_with_trace))
        stack.enter_context(patch.object(Trainer, "_compute_model_prediction_loss", loss_with_trace))
        trainer = Trainer(config=config, exit_on_error=True)
        try:
            _run_training(trainer)
        finally:
            trainer.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--case", choices=("bf16", "fp32", "sync", "deterministic"), default="fp32")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-step", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument(
        "--conditioning", action="store_true", help="Choose one of two native reference datasets per occurrence"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--clip", default="openai/clip-vit-base-patch32")
    parser.add_argument("--rank", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.rank:
        train(root, args.resume, not args.no_prefetch, args.case, args.checkpoint_step, args.conditioning)
    else:
        import shutil

        run = root / args.case
        run.mkdir(exist_ok=True, parents=True)
        if list(run.glob(f"consumed-{args.resume}-*.jsonl")):
            parser.error("This case already contains a consumption trace. Select a fresh run directory.")
        if not (root / "model/model_index.json").exists():
            make_model(root / "model", args.clip)
        credentials = json.load(sys.stdin)
        os.environ.update(
            AWS_ACCESS_KEY_ID=credentials["access_id"],
            AWS_SECRET_ACCESS_KEY=credentials["secret_key"],
            AWS_ENDPOINT=credentials["url"],
            AWS_ENDPOINT_URL=credentials["url"],
            AWS_REGION="auto",
        )
        datasets = []
        for name, policy, strength, count in [("small", "global", 1, 8), ("large", "shard_local", -1, 16)]:
            datasets.append(
                dict(
                    id=name,
                    type="webshart",
                    source=args.source,
                    aws_region_name="auto",
                    webshart=dict(
                        format="parquet",
                        manifest=args.manifest,
                        sampling_policy=policy,
                        shard_cache_gb=0.5,
                        cache_dir=str(root / name),
                    ),
                    metadata_backend="webshart",
                    caption_strategy="webshart",
                    max_num_samples=count,
                    resolution=64,
                    resolution_type="pixel",
                    crop=True,
                    crop_style="center",
                    crop_aspect="square",
                    cache_dir_vae=str(root / ("vae-" + name)),
                    slider_strength=strength,
                )
            )
        if args.conditioning:
            from copy import deepcopy

            references = []
            for ref_id in ("reference-a", "reference-b"):
                ref = deepcopy(datasets[-1])
                ref.update(
                    id=ref_id,
                    dataset_type="conditioning",
                    conditioning_type="reference_loose",
                    caption_strategy="instanceprompt",
                    instance_prompt=ref_id,
                )
                ref["webshart"]["cache_dir"] = str(root / ref_id)
                ref["cache_dir_vae"] = str(root / ("vae-" + ref_id))
                references.append(ref)
            for source in datasets:
                source["conditioning_data"] = [ref["id"] for ref in references]
            datasets.extend(references)
        datasets.append(
            dict(
                id="text",
                dataset_type="text_embeds",
                default=True,
                type="local",
                cache_dir=str(root / "text"),
                text_cache_encode_only=True,
            )
        )
        (root / "data.json").write_text(json.dumps(datasets))
        if args.resume:
            (run / "resume").mkdir(exist_ok=True)
            checkpoint = f"checkpoint-{args.checkpoint_step}"
            shutil.copytree(run / "baseline" / checkpoint, run / "resume" / checkpoint, dirs_exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            str(root),
            "--rank",
            "--checkpoint-step",
            str(args.checkpoint_step),
            "--case",
            args.case,
            "--source",
            args.source,
            "--manifest",
            args.manifest,
        ]
        if args.resume:
            command.append("--resume")
        if args.conditioning:
            command.append("--conditioning")
        if args.no_prefetch:
            command.append("--no-prefetch")
        if args.case == "deterministic":
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(command, check=True)
        if args.resume:
            import torch
            from safetensors.torch import load_file

            for rank in range(2):
                baseline, restarted = (
                    [json.loads(line) for line in (run / f"consumed-{flag}-{rank}.jsonl").read_text().splitlines()]
                    for flag in (False, True)
                )
                if (
                    len(baseline) != 4
                    or len(restarted) != 4 - args.checkpoint_step
                    or [x[:3] for x in baseline[args.checkpoint_step :]] != [x[:3] for x in restarted]
                ):
                    raise AssertionError(f"Consumed occurrences or mixer state differ on rank {rank}")
                changed = {
                    key
                    for before, after in zip(baseline[args.checkpoint_step :], restarted)
                    for key in before[3]
                    if before[3][key] != after[3].get(key)
                }
                if changed:
                    raise AssertionError(f"Prepared batch differs on rank {rank}: {sorted(changed)}")
                loss_deltas = []
                for before, after in zip(baseline[args.checkpoint_step :], restarted, strict=True):
                    if not before[4]:
                        raise AssertionError("Training loss was not captured")
                    if args.case == "deterministic" and before[4:] != after[4:]:
                        raise AssertionError(f"Deterministic loss or optimizer state differs on rank {rank}")
                    torch.testing.assert_close(torch.tensor(before[4]), torch.tensor(after[4]))
                    loss_deltas.extend(abs(x - y) for x, y in zip(before[4], after[4], strict=True))
                print(
                    json.dumps(
                        dict(
                            rank=rank,
                            consumed_occurrences_match=True,
                            inputs_match=True,
                            max_loss_difference=max(loss_deltas),
                        )
                    )
                )
            a, b = (load_file(str(run / d / "pytorch_lora_weights.safetensors")) for d in ("baseline", "resume"))
            if a.keys() != b.keys():
                raise AssertionError("Adapter keys differ after restart")
            for key in a:
                if args.case == "deterministic" and not torch.equal(a[key], b[key]):
                    raise AssertionError(f"Deterministic adapter differs after restart: {key}")
                torch.testing.assert_close(a[key], b[key])
            print(
                json.dumps(
                    dict(
                        tensor_count=len(a),
                        identical=sum(bool((a[k] == b[k]).all()) for k in a),
                        max_abs_difference=max(float((a[k] - b[k]).abs().max()) for k in a),
                    )
                )
            )
