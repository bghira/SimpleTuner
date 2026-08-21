# Self-Transcendence

Self-Transcendence 使用模型内部目标训练扩散 Transformer 的浅层，不依赖外部视觉编码器。实现基于 [Sun 等人](https://arxiv.org/abs/2601.07773)的两阶段方法。

支持能够输出潜变量 token 中间状态的图像、视频和音频扩散 Transformer。不支持 UNet、自回归模型和 LyCORIS。支持完整模型训练和标准 PEFT LoRA。

## 阶段 1：VAE 结构引导

阶段 1 将浅层状态投影到 VAE 潜空间中的模型扩散目标：流速度、epsilon、v-prediction 或干净样本。目标会在不丢弃数值的情况下按模型的 token 网格分块。

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "vae", "student_block": 8, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "projector_hidden_dim": 2048
  }}
}
```

保存此阶段的适配器或检查点，供阶段 2 作为固定教师使用。

## 阶段 2：自引导表示

阶段 2 在同一个噪声输入上分别使用正常提示和缓存的空提示运行固定教师。深层状态通过特征空间 CFG 组合后，用于监督新学生的浅层。

PEFT LoRA 运行应创建新的学生适配器，并将 `teacher_adapter_path` 指向阶段 1 的 safetensors 文件：

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "self", "student_block": 8, "teacher_block": 16,
    "teacher_adapter_path": "output/stage1/pytorch_lora_weights.safetensors",
    "cfg_scale": 30.0, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "stop_step": 5000, "projector_hidden_dim": 2048
  }}
}
```

教师和学生必须使用相同的基础模型、PEFT rank 和目标模块。未设置 `teacher_adapter_path` 时，阶段 2 会快照恢复后已有的可训练参数；这支持完整模型和单阶段实验，但不是论文中的新学生设置。

层索引从 0 开始。可从约三分之一深度的学生层和三分之二深度的教师层开始。`stop_step` 之后停止教师前向，但保留零权重投影路径以兼容 DDP。阶段 2 会自动缓存空提示嵌入。

记录 `self_transcendence/loss`、`self_transcendence/weight` 和阶段 2 的 `self_transcendence/teacher_cfg_scale`。此方法不能与其他蒸馏器或文本编码器训练同时使用。
