# Internal Guidance

Internal Guidance 在 diffusion transformer 的早期 block 上增加辅助 denoising head。该 head 与最终 head 预测相同目标。采样时，两者之差可在不增加其他模型或 unconditional pass 的情况下提供 guidance。

## 配置

```json
{
  "internal_guidance_enabled": true,
  "internal_guidance_loss_weight": 0.5,
  "internal_guidance_block_index": 7,
  "validation_internal_guidance_scale": 1.4
}
```

- `internal_guidance_loss_weight`：intermediate loss 权重；参考实现使用 `0.5`。
- `internal_guidance_block_index`：从 0 开始的捕获 block。省略时选择 transformer 深度的四分之一。
- `validation_internal_guidance_scale`：采样 scale。`1.0` 禁用 extrapolation；参考实现使用 `1.4`。

```text
loss = final_loss + weight * intermediate_loss
guided = intermediate + scale * (final - intermediate)
```

Projection 使用零初始化，并随标准 PEFT LoRA 或完整 checkpoint 保存。不支持 LyCORIS、autoregressive model、UNet 和 ACE-Step v1.5 decoder。

## Diffusers inference

```python
from safetensors.torch import load_file
from simpletuner.helpers.training.internal_guidance import attach_internal_guidance_head_from_state_dict, internal_guidance_inference, internal_guidance_lora_state_dict

state_dict = load_file("pytorch_lora_weights.safetensors")
attach_internal_guidance_head_from_state_dict(pipe.transformer, state_dict)
pipe.transformer.load_lora_adapter(internal_guidance_lora_state_dict(state_dict), prefix="transformer")
with internal_guidance_inference(pipe.transformer, scale=1.4):
    image = pipe(prompt).images[0]
```

Transformer 必须接受 `hidden_states_buffer` 并存储 `layer_{index}`。监控 `internal_guidance_loss`，并测试 `1.0` 到 `1.8` 的采样 scale。
