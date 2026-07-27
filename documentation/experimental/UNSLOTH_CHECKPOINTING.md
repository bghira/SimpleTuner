# Unsloth-Style Checkpointing

Short version: use this when a job almost fits, and try FFN-only first when the model supports it.

The `unsloth` checkpointing backend offloads saved activation tensors to CPU while PyTorch checkpointing recomputes them during backward. That can buy the last few GiB needed for a larger batch, resolution, or video clip. It is not free speed. If the run already fits with the normal `torch` backend, `torch` is usually the better default.

## Controls

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` has four useful values:

| Value | Scope | Storage path | Use it when |
| --- | --- | --- | --- |
| `torch` | whole block | recompute | You need the biggest built-in memory cut before CPU offload. |
| `torch-ffn` | feed-forward side | recompute | You want the cheap win after Flash Attention already handled attention memory. |
| `unsloth` | whole block | CPU offload | Whole-block torch checkpointing still does not quite fit. |
| `unsloth-ffn` | feed-forward side | CPU offload | FFN-only torch checkpointing almost fits and CPU offload can buy the last bit. |

For supported model families, you can also checkpoint fewer blocks:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` checkpoints every other supported block. Higher values spend less time checkpointing and keep more activations in VRAM.

`torch-ffn` and `unsloth-ffn` currently support Flux.1-style blocks and MageFlow. Other model families fail clearly until their block internals expose the same safe boundary.

## What It Trades

- `torch`: drops intermediate activations and recomputes them in backward.
- `unsloth`: saves some of those tensors on CPU and copies them back for backward.
- `*-ffn`: checkpoints only the feed-forward side on models with a clean FFN boundary.
- Flash Attention already avoids materializing the big attention matrix. That "free checkpointing" claim is mostly about attention, not the whole transformer block.
- CPU offload helps most when activation tensors are large and parameter/optimizer memory is not the peak.

This backend needs CUDA and enough CPU RAM. PCIe bandwidth matters. If CPU-GPU copies are exposed instead of hidden behind other work, the step gets slower.

## Our Sweep

Synthetic transformer block, bf16, flash SDPA, frozen base weights, batch 1. These numbers are not model guarantees; they show the shape of the trade.

### Packed Image Latents

For 2x2 packed latents, `64x64`, `128x128`, and `256x256` become `1024`, `4096`, and `16384` transformer tokens.

| GPU | Tokens | No checkpoint | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

At `1024` tokens, the extra offload is noise unless you are already out of room. At `16384` tokens, `torch-ffn` is the best cheap step and whole-layer checkpointing is the big fit lever. `unsloth` buys about another `1.3 GiB` beyond torch layer checkpointing.

### Larger Transformer Shape

Frozen `32` layer, width `4096`, `3072` tokens:

| GPU | No checkpoint | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

Trainable full weights changed the picture: gradients and optimizer state dominated the peak, so `unsloth` did not save more than `torch` in that toy run. PEFT runs are closer to the frozen-weight case.

## Decision Rule

Start here:

1. If the job fits without checkpointing, leave it off.
2. If it does not fit, try `gradient_checkpointing_backend: torch-ffn`.
3. If that is still too tight, try `torch`.
4. If torch layer checkpointing still does not fit, try `unsloth-ffn`, then `unsloth`.
5. If the model supports `gradient_checkpointing_interval`, use `2` or higher only after the run fits and you want speed back.

The backend is worth it when it lets you run the batch, resolution, frame count, or rank you actually wanted. It is not worth it for small token counts or jobs where the memory peak is mostly trainable weights, gradients, optimizer state, VAE cache, or validation.

## Notes

- With FSDP activation checkpointing enabled, SimpleTuner disables model-level gradient checkpointing to avoid conflicting checkpoint systems.
- `torch-ffn` and `unsloth-ffn` currently require model support. SimpleTuner errors out instead of silently running a different scope.
- `gradient_checkpointing_interval: 1` is treated like normal every-block checkpointing.
- Some model families do not expose interval checkpointing. SimpleTuner warns and ignores the interval there.
- `torch.compile` did not rescue the offload path in our sweep. It may still help a full model for unrelated reasons, but do not count on it as the reason to pick this backend.
