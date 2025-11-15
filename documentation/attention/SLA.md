# Sparse–Linear Attention (SLA) in SimpleTuner

Sparse–Linear Attention (SLA) fuses sparse FlashAttention and a linear attention compensator inside a single CUDA kernel. Critical query/key blocks take the expensive sparse path, while marginal blocks use lightweight linear attention plus a learnable projection. This keeps quality close to full attention while dramatically reducing FLOPs.

SimpleTuner exposes SLA through the regular `--attention_mechanism` flag, so you can fine-tune models with SLA and later run inference with the same kernel.

## Requirements

1. Install the reference implementation:

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. Use a CUDA build of PyTorch (SLA kernels are CUDA-only today).

## Enabling SLA

- Pass `--attention_mechanism=sla` (or set `attention_mechanism: "sla"` in configs).
- No extra flags are required; SimpleTuner injects SLA by wrapping PyTorch’s SDPA entrypoint.
- Override SLA settings (top-k ratio, block sizes, feature map type, whether query/key feature maps are tied) via `--sla_config` / `sla_config` in JSON/Python dict form. Example: `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`. Defaults: top 20 %, block size 64, tied feature maps.

## Training Behaviour

- SLA is trainable. The controller keeps the linear projection head (`proj_l`) in `float32` even when the rest of SLA executes in BF16/FP16 so AMP/GradScaler remain stable.
- Because the backbone is fine-tuned to expect SLA’s mixed sparse/linear behaviour, you should continue to use SLA during inference. Switching back to Diffusers SDPA/XFormers after training will likely hurt quality.
- During checkpoint saves, SimpleTuner writes `sla_attention.pt` alongside the normal accelerator state. This file contains the SLA projection weights and related buffers for every unique head dimension/dtype pair that was materialised. Keep this file with the rest of your checkpoint; removing it means the next resume/inference run will reinitialise SLA’s projection layer.

## Inference

- Keep `--attention_mechanism=sla` enabled whenever you resume training or rerun validation steps so the checkpoint continues to use the SLA kernel it was fine-tuned with.
- The loader automatically replays `sla_attention.pt` if it exists inside the checkpoint directory, so no extra flags are needed.
- If you intentionally want to compare SLA-trained weights with standard SDPA, expect a quality drop. The SLA paper shows that a few thousand tuning steps are required to adapt the backbone, so inference without SLA should be treated as unsupported.

## Troubleshooting & Notes

- **Missing `sla_attention.pt`:** This means the checkpoint was created before SLA state saving existed or the file was removed. Re-run a short training session (even a single step) with SLA enabled to regenerate the file.
- **AMP/GradScaler errors:** Ensure you are not manually casting SLA modules back to BF16/FP16. SimpleTuner forces the projection head to FP32 automatically; further casts can destabilise training.
- **Hub uploads:** When pushing checkpoints to the Hugging Face Hub (or any artifact store), include `sla_attention.pt`. Consumers who download your checkpoint will then inherit the trained SLA weights without extra steps.

For more background on SLA’s design and the full algorithm, see [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://www.arxiv.org/abs/2509.24006).
