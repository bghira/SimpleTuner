# MixFlow 训练

MixFlow 是用于流匹配模型的后训练方法。模型在时间步 $t$ 接收噪声更高的真实插值，从而减小训练时精确插值与采样时不完美 latent 之间的差异。

## 配置

```json
{
  "mixflow_enabled": true,
  "mixflow_gamma": 0.8
}
```

`mixflow_gamma` 控制减速插值范围。`0.8` 是论文默认值。`0.0` 保留标准插值，但仍使用 MixFlow 时间步采样。

MixFlow 从 $Beta(2,1)$ 采样面向数据的模型时间步。SimpleTuner 使用方向相反、面向噪声的 flow sigma，因此实现为 $sigma = 1 - sqrt(U)$，随后应用模型配置的 flow schedule shift。模型接收原始时间步，latent 输入使用：

$$
sigma_{input} = sigma + U' gamma (1 - sigma)
$$

线性 flow 路径的速度目标不变。推理过程不变。

## 支持范围

所有预测类型为 `flow_matching` 的 SimpleTuner 模型系列都使用共享 MixFlow 路径。模型封装会处理面向数据的时间步约定、非线性 sigma 变换以及联合音频/视频输入。

MixFlow 不能与其他训练轨迹替换方法组合：自定义、uniform、Beta 或 fast flow schedule、Self-Flow、TwinFlow、scheduled sampling 或蒸馏。仍支持 schedule shift。

MixFlow 用于现有 flow 模型的后训练。先采用短期常规续训使用的学习率和优化器，再用固定 seed 将验证样本与起始 checkpoint 比较。

## 参考

- [MixFlow 论文](https://arxiv.org/abs/2512.19311)
- [参考实现](https://github.com/fudan-generative-vision/MixFlow)
