# ConvRot / Hadamard SDNQ

SimpleTuner exposes ConvRot-style rotation through SDNQ's Hadamard path. This is useful for large PEFT jobs where the frozen base model should run as int8 while LoRA or LyCORIS adapters stay trainable in bf16.

SimpleTuner does not consume arbitrary ConvRot sidecar buffers as a separate feature. For the common path, load the original model weights and let SimpleTuner quantize the trained component with SDNQ after model load. Model loaders that support single-file quantized transformer weights can also load compatible INT8 ConvRot transformer safetensors and run them through SDNQ Hadamard.

## Quick Setup

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

For large models, keep `quantize_via` on `cpu` unless the model guide says otherwise. CPU quantization reduces peak accelerator memory during setup.

## What The Options Do

- `base_model_precision: int8-sdnq` selects SDNQ int8 post-load quantization for the trained base component.
- `sdnq_use_hadamard: true` enables the Hadamard rotation path.
- `sdnq_hadamard_group_size: 256` sets the rotation block size used by SDNQ. Use `256` for ConvRot; smaller blocks select a QuaRot-style path.
- `sdnq_group_size: -1` uses static full-row weight scales. This avoids the dynamic grouped path that is mainly targeted at full fine-tuning and can requantize weights during training.
- `sdnq_use_quantized_matmul: true` keeps the SDNQ int8 matmul path active.
- `sdnq_compile_mode: compile` compiles the SDNQ quantization helpers and kernels where SDNQ supports it.
- `gradient_checkpointing: true` lets SDNQ use the lower-overhead training path for PEFT workloads. SimpleTuner passes this to SDNQ as `use_grad_ckpt=True`; with gradient checkpointing enabled, setting that SDNQ flag to false only adds work to save quantized backward inputs that checkpointing immediately discards.

## PEFT Behavior

The base transformer is quantized by SDNQ. The adapter weights remain trainable and use the normal mixed-precision dtype, usually bf16.

Some models load fixed helper adapters before training. Z-Image Turbo, for example, has an assistant LoRA. SimpleTuner defers that assistant adapter until after SDNQ quantization so SDNQ sees the original transformer modules instead of PEFT wrapper proxy weights.

## Requirements And Limits

- SimpleTuner installs and configures the SDNQ training dependency for supported install targets.
- This preset is intended for LoRA and LyCORIS training of large models. Full fine-tuning with SDNQ Hadamard needs separate validation.
- First steps can be slow because SDNQ and Torch compile kernels during setup and early training.
- Validation and inference use the quantized base model plus the active adapter, the same as training.
- ConvRot can reduce quantization damage, but it is not a guarantee that INT8 will match BF16 or FP8 for every model. Validate both loss curves and generated samples before committing to a long run.
- Standalone inference with SDNQ ConvRot is outside this training guide. For direct SDNQ inference APIs, follow the [upstream SDNQ documentation](https://github.com/Disty0/sdnq) because that API changes more often than SimpleTuner's training configuration.

## Measured Results

These are model-specific SimpleTuner trainer measurements, not synthetic GEMM-only results. `Loop s/step` is the wrapper's train-loop wall time per step. `Mean step` excludes the first five warmup steps.

| Model | GPU | Steps | Weight path | Loop s/step | Mean step | p50 | p95 | Peak allocated VRAM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | SDNQ Hadamard post-load quantization | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | SDNQ Hadamard post-load quantization | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard path | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights, diffusers attention | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer weights, cuDNN attention | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ INT8 vanilla post-load quantization | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA, square crop | H100 80GB | 100 | SDNQ ConvRot 256 post-load quantization | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

On the warmed L40S Z-Image comparison, the current path was 10.3% faster by train-loop wall time and 5.2% faster by measured train-step mean than the baseline SDNQ Hadamard path. The Krea 2 rows verify the Hugging Face INT8 ConvRot transformer-weight path in real 100-step training runs. The Mage-Flow rows show why model-specific validation matters: square crop removed most shape-compile churn, ConvRot reduced total train-loop time versus vanilla INT8, but the warmed measured step was slightly slower than vanilla INT8.

## Example Models

SimpleTuner includes SDNQ Hadamard examples for Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3, and LTXVideo 2.3. These examples use `sdnq_group_size: -1` because that setting matched the PEFT workload better than the dynamic grouped training default.
