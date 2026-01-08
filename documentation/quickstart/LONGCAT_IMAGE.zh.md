# LongCat‑Image 快速入门

LongCat‑Image 是一个 6B 双语（zh/en）文生图模型，使用流匹配和 Qwen‑2.5‑VL 文本编码器。本指南带你完成 SimpleTuner 的设置、数据准备，以及首次训练/验证。

---

## 1) 硬件要求（参考）

- VRAM：使用 `int8-quanto` 或 `fp8-torchao` 时，16–24GB 可覆盖 1024px LoRA。完整 bf16 运行可能需要约 24GB。
- 系统内存：通常约 32GB 足够。
- Apple MPS：支持推理/预览；为避免 dtype 问题，MPS 上已将 pos‑ids 降为 float32。

---

## 2) 前提条件（逐步）

1. 确认 Python 3.10–3.12：
   ```bash
   python --version
   ```
2. （Linux/CUDA）在全新镜像上安装常见构建工具链：
   ```bash
   apt -y update
   apt -y install build-essential nvidia-cuda-toolkit
   ```
3. 按你的硬件选择 extras 安装 SimpleTuner：
   ```bash
   pip install "simpletuner[cuda]"   # CUDA
   pip install "simpletuner[mps]"    # Apple Silicon
   pip install "simpletuner[cpu]"    # CPU-only
   ```
4. 量化已内置（`int8-quanto`, `int4-quanto`, `fp8-torchao`），通常无需额外手动安装。

---

## 3) 环境设置

### Web UI（最省心）
```bash
simpletuner server
```
访问 http://localhost:8001 并选择模型家族 `longcat_image`。

### CLI 基础配置（config/config.json）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "final",                // options: final, dev
  "pretrained_model_name_or_path": null,   // auto-selected from flavour; override with a local path if needed
  "base_model_precision": "int8-quanto",   // good default; fp8-torchao also works
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 16,
  "learning_rate": 1e-4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 30
}
```

**需保留的关键默认值**
- 流匹配调度器是自动的；无需额外调度参数。
- 宽高比桶保持 64px 对齐；不要降低 `aspect_bucket_alignment`。
- 最大 token 长度 512（Qwen‑2.5‑VL）。

可选内存节省项（按硬件选择）：
- `--enable_group_offload --group_offload_type block_level --group_offload_blocks_per_group 1`
- 降低 `lora_rank`（4–8）并/或使用 `int8-quanto` 作为基础精度
- 如验证 OOM，先降低 `validation_resolution` 或步数

### 快速创建配置（仅一次）
```bash
cp config/config.json.example config/config.json
```
编辑上面提到的字段（model_family、flavour、precision、paths），并将 `output_dir` 与数据集路径指向你的存储位置。

### 开始训练（CLI）
```bash
simpletuner train --config config/config.json
```
也可以启动 WebUI，选择同样的配置后在 Jobs 页面发起任务。

---

## 4) 数据加载器要点（需要准备什么）

- 常规的带字幕图片文件夹（textfile/JSON/CSV）即可；若要保持双语能力，请同时包含 zh/en。
- 桶边界保持 64px 网格。若训练多宽高比，列出多个分辨率（如 `1024x1024,1344x768`）。
- VAE 为带 shift+scale 的 KL；缓存会自动使用内置缩放系数。

---

## 5) 验证与推理

- Guidance：4–6 是良好起点；负向提示词可保持为空。
- 步数：速度检查约 30；最佳质量 40–50。
- 验证预览可直接使用；为避免通道不匹配，解码前会先解包 latent。

示例（CLI 验证）：
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour final \
  --validation_resolution 1024x1024 \
  --validation_num_inference_steps 30 \
  --validation_guidance 4.5
```

---

## 6) 故障排查

- **MPS float64 报错**：已在内部处理；配置保持 float32/bf16。
- **预览通道不匹配**：通过解码前 unpack latent 修复（本指南代码已包含）。
- **OOM**：降低 `validation_resolution`，减小 `lora_rank`，启用 group offload，或切换到 `int8-quanto`/`fp8-torchao`。
- **分词变慢**：Qwen‑2.5‑VL 上限 512 token，避免过长提示词。

---

## 7) 版本选择
- `final`：主发布（最佳质量）。
- `dev`：中期检查点，适合实验/微调。
