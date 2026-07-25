# ConvRot / Hadamard SDNQ

SimpleTuner exposes ConvRot-style rotation through SDNQ's Hadamard path. This is useful for large PEFT jobs where the frozen base model should run as int8 while LoRA or LyCORIS adapters stay trainable in bf16.

This does not load external ConvRot checkpoint buffers directly. Load the original model weights, then let SimpleTuner quantize the trained component with SDNQ after model load.

## Quick Setup

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

For large models, keep `quantize_via` on `cpu` unless the model guide says otherwise. CPU quantization reduces peak accelerator memory during setup.

## What The Options Do

- `base_model_precision: int8-sdnq` selects SDNQ int8 post-load quantization for the trained base component.
- `sdnq_use_hadamard: true` enables the Hadamard rotation path.
- `sdnq_hadamard_group_size: 128` sets the rotation block size used by SDNQ.
- `sdnq_group_size: -1` uses static full-row weight scales. This avoids the dynamic grouped path that is mainly targeted at full fine-tuning and can requantize weights during training.
- `sdnq_use_quantized_matmul: true` keeps the SDNQ int8 matmul path active.
- `sdnq_compile_mode: compile` compiles the SDNQ quantization helpers and kernels where SDNQ supports it.
- `gradient_checkpointing: true` lets SDNQ use the lower-overhead training path for PEFT workloads. SimpleTuner passes this to SDNQ as `use_grad_ckpt=True`; with gradient checkpointing enabled, setting that SDNQ flag to false only adds work to save quantized backward inputs that checkpointing immediately discards.

## PEFT Behavior

The base transformer is quantized by SDNQ. The adapter weights remain trainable and use the normal mixed-precision dtype, usually bf16.

Some models load fixed helper adapters before training. Z-Image Turbo, for example, has an assistant LoRA. SimpleTuner defers that assistant adapter until after SDNQ quantization so SDNQ sees the original transformer modules instead of PEFT wrapper proxy weights.

## Requirements And Limits

- Use an SDNQ build with Hadamard support. The H100 verification used upstream SDNQ `0.2.3`; PyPI `0.2.2` does not include the same bf16 Hadamard fix.
- This preset is intended for LoRA and LyCORIS training of large models. Full fine-tuning with SDNQ Hadamard needs separate validation.
- First steps can be slow because SDNQ and Torch compile kernels during setup and early training.
- Validation and inference use the quantized base model plus the active adapter, the same as training.

## Example Models

SimpleTuner includes SDNQ Hadamard examples for Z-Image Turbo, Krea 2, FLUX.2, Cosmos 3, and LTXVideo 2.3. These examples use `sdnq_group_size: -1` because that setting matched the PEFT workload better than the dynamic grouped training default.
