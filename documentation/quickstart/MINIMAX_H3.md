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

By default, `minimax_h3_target_mode: "auto"` resolves to video-only and avoids audio VAE work:

```json
{
  "minimax_h3_target_mode": "video"
}
```

Use `"av"` only when the dataset has target audio latents and you want joint audio/video training. You can set `h3_target_mode` or `minimax_h3_target_mode` inside a data backend entry to opt only selected backends into audio.

## Memory Knobs

- Use the 24G RamTorch example when VRAM is tight.
- Use `musubi_blocks_to_swap` when block swap is faster than heavier checkpointing on your GPU.
- Keep VAE tiling, slicing, and temporal roll enabled for the video VAE.
- Benchmark `attention_mechanism` values on the target GPU; H3 shapes may prefer a different backend than Wan or LTX Video.
- Re-test after changing `torch.compile` mode because compile caches can change peak VRAM.

## Run

From the SimpleTuner directory:

```bash
simpletuner train example=minimaxh3-fl2va-convrot-int8.peft-lora
```

Run a short smoke test before committing to a long run. Check that `h3_drift_loss`, normal training loss, and validation samples move together instead of one term dominating the run.
