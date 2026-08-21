# MiniMax H3 Quickstart

MiniMax H3 is a 33B flow-matching video/audio model. SimpleTuner supports adapter training through the MiniMax H3 model family, including FL2VA-style first/last-frame conditioning and the ConvRot quantized flavours.

## Starting Configs

The checked-in examples are the recommended starting points:

- `simpletuner/examples/minimaxh3-fl2va-convrot-int8.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-24g.peft-lora+ramtorch`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-32g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-48g.peft-lora`
- `simpletuner/examples/minimaxh3-fl2va-convrot-int8-80g.peft-lora`

Use the closest VRAM preset first, then adjust resolution, frame count, attention backend, and checkpointing after a smoke test.

## Core Settings

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "mixed_precision": "bf16",
  "base_model_precision": "no_change",
  "text_encoder_1_precision": "int8-quanto",
  "flow_schedule_shift": 12.0,
  "audio_flow_schedule_shift": 3.0,
  "validation_disable_unconditional": true,
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0
}
```

`convrot-int8` is the default example flavour. `convrot-int4` can be used with the same model family when the lower precision checkpoint is desired.

## Maintaining Distillation

MiniMax H3 is CFG-distilled. The base checkpoint is intended to run without an unconditional branch, so the example configs keep validation at guidance `1.0` and set `validation_disable_unconditional: true`.

Adapter training can still drift away from the frozen distilled behavior. The examples enable the H3 drift distiller by default:

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

This runs a no-grad frozen-base reference pass with the adapter disabled and penalizes the adapter when its video/audio prediction moves too far from that reference. Keep it enabled for normal H3 LoRAs. Lower `loss_weight` if the adapter is not learning the concept; raise it if validation starts losing the base model's distilled behavior. For the full explanation, see [MiniMax H3 Drift Distillation](../distillation/MINIMAX_H3_DRIFT.md).

Negative prompting is not part of the base H3 contract. SimpleTuner keeps real CFG and negative prompt plumbing available for de-distilled community checkpoints, but `h3_drift` deliberately preserves the original distilled conditional behavior.

## Audio Target Mode

By default, `minimax_h3_target_mode: "auto"` resolves to `av` when enabled audio data is detected and to `video` otherwise. Validation uses the same detected-data default, so audio-only and joint audio/video runs render audio without a separate validation override. Set `video` explicitly to avoid audio VAE work:

```json
{
  "minimax_h3_target_mode": "video"
}
```

Use `"av"` explicitly when the dataset has target audio latents and you want joint audio/video training. You can set `h3_target_mode` or `minimax_h3_target_mode` inside a data backend entry to opt only selected backends into audio.

For audio-only training, `dataset_type: "audio"` is sufficient. Because H3 advertises fake-video support, SimpleTuner
records `audio.audio_only: true` in the normalized backend config, builds the placeholder video stream, and masks video
loss. The explicit `audio_only` setting remains accepted but is not required.

## Experimental Sparse Attention

MiniMax reports that H3 used train-aware, MoBA-style 3D sparse attention for video tokens during its final training stage. The initial public release uses dense attention, and MiniMax has not yet published the exact H3 block shape, retention budget, layer schedule, or production kernel. SimpleTuner therefore keeps its experimental approximation disabled by default.

```json
{
  "minimax_h3_sparse_attention": "moba3d",
  "minimax_h3_sparse_block_shape": "1,8,16",
  "minimax_h3_sparse_video_kv_fraction": 0.25,
  "minimax_h3_sparse_share_heads": false,
  "minimax_h3_sparse_start_layer": 0
}
```

The implementation mean-pools 3D query/key video blocks for parameter-free top-k routing. Target-video queries retain dense access to text, audio, and reference context; non-target queries remain dense. The block dimensions must multiply to 128. A video KV fraction of `1.0` is the dense-connectivity numerical control through FlexAttention.

This mode currently requires CUDA and introduces a Dynamo graph boundary around FlexAttention. Ulysses context parallelism is supported with `context_parallel_strategy=alltoall`; ring context parallelism and TREAD are not supported. At 480px, sparse routing may use more memory than FlashAttention because the target lattice and packed context must be padded and reordered. Treat it as a fine-tuning ablation until MiniMax publishes its reference implementation, not as a guaranteed speedup.

## Memory Knobs

- Use the 24G RamTorch example when VRAM is tight.
- Use `musubi_blocks_to_swap` when block swap is faster than heavier checkpointing on your GPU.
- Keep video `flow_schedule_shift` at `12.0` and audio `audio_flow_schedule_shift` at `3.0`. The H3 helper corrects the inherited global `3.0` video default because it does not match the MiniMax H3 schedule.
- SimpleTuner forces H3 video VAE tiling and temporal roll/chunking on. The tiling geometry is the upstream `256` tile size with `64` overlap; setting those options false is ignored because untiled decode can produce severe colour shifts and halftone artifacts.
- Benchmark `attention_mechanism` values on the target GPU; H3 shapes may prefer a different backend than Wan or LTX Video.
- Re-test after changing `torch.compile` mode because compile caches can change peak VRAM.

## Run

From the SimpleTuner directory:

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Run a short smoke test before committing to a long run. Check that `h3_drift_loss`, normal training loss, and validation samples move together instead of one term dominating the run.
