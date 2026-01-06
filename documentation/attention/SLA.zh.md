# SimpleTuner 中的 Sparse–Linear Attention（SLA）

Sparse–Linear Attention（SLA）在单个 CUDA 内核中融合稀疏 FlashAttention 与线性注意力补偿器。关键的 query/key 块走高开销的稀疏路径，边缘块使用轻量线性注意力并加上可学习投影。在保持接近完整注意力质量的同时，显著减少 FLOPs。

SimpleTuner 通过常规的 `--attention_mechanism` 标志暴露 SLA，因此你可以用 SLA 微调模型，并在推理时继续使用同一内核。

## 要求

1. 安装参考实现：

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. 使用 CUDA 版 PyTorch（SLA 内核目前仅支持 CUDA）。

## 启用 SLA

- 传入 `--attention_mechanism=sla`（或在配置中设置 `attention_mechanism: "sla"`）。
- 不需要额外标志；SimpleTuner 通过包装 PyTorch 的 SDPA 入口注入 SLA。
- 可通过 `--sla_config` / `sla_config` 以 JSON/Python dict 形式覆盖 SLA 设置（top-k 比例、块尺寸、特征图类型、query/key 特征图是否绑定）。示例：`--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`。默认：top 20%、块大小 64、特征图绑定。

## 训练行为

- SLA 可训练。控制器会将线性投影头（`proj_l`）保持在 `float32`，即使 SLA 其余部分以 BF16/FP16 执行也能保证 AMP/GradScaler 稳定。
- 因为主干会针对 SLA 的稀疏/线性混合行为进行微调，推理时也应继续使用 SLA。训练后切回 Diffusers SDPA/XFormers 可能会显著降质。
- 保存检查点时，SimpleTuner 会在常规 accelerator 状态旁写入 `sla_attention.pt`。该文件包含所有已实例化的头维度/dtype 对应的 SLA 投影权重与相关缓冲。请与检查点一起保留；删除会导致下次恢复/推理时重新初始化 SLA 投影层。

## 推理

- 继续保持 `--attention_mechanism=sla`，用于恢复训练或再次运行验证，以确保检查点仍使用其微调过的 SLA 内核。
- 若检查点目录中存在 `sla_attention.pt`，加载器会自动重放，无需额外标志。
- 如果你刻意想将 SLA 训练的权重与标准 SDPA 对比，预期会降质。SLA 论文显示需要数千步适配主干，因此不启用 SLA 的推理应视为不受支持。

## 排障与注意事项

- **缺少 `sla_attention.pt`:** 说明检查点在 SLA 状态保存引入前生成，或该文件被移除。用 SLA 启用的短训练（哪怕 1 步）重生成。
- **AMP/GradScaler 报错:** 请勿手动将 SLA 模块转换回 BF16/FP16。SimpleTuner 会自动将投影头固定为 FP32；额外的转换会破坏稳定性。
- **Hub 上传:** 将检查点推送到 Hugging Face Hub（或任何工件存储）时，务必包含 `sla_attention.pt`，这样下载者无需额外步骤即可继承训练后的 SLA 权重。

有关 SLA 的设计与完整算法，请参阅 [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://www.arxiv.org/abs/2509.24006)。
