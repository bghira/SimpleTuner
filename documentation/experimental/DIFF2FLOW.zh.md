# Diff2Flow（Diffusion-to-Flow 桥接）

## 背景

传统扩散模型按预测目标分类：
*   **Epsilon（$\epsilon$）：** 预测添加到图像的噪声（SD 1.5、SDXL）
*   **V-Prediction（$v$）：** 预测结合噪声与数据的速度（SD 2.0、SDXL Refiner）

较新的 SOTA 模型如 **Flux**、**Stable Diffusion 3**、**AuraFlow** 使用 **Flow Matching**（具体为 Rectified Flow）。Flow Matching 将生成过程视为常微分方程（ODE），沿直线路径把粒子从噪声分布移动到数据分布。

直线轨迹通常更易于求解器处理，从而实现更少步数与更稳定的生成。

## 桥接

**Diff2Flow** 是一个轻量适配器，使“传统”模型（Epsilon 或 V-pred）无需更改底层架构就能用 Flow Matching 目标进行训练。

它通过将模型原生输出（例如 epsilon 预测）数学上转换为 flow 向量场 $u_t(x|1)$，再与 flow 目标（$x_1 - x_0$，或 `noise - latents`）计算损失。

> 🟡 **实验性状态：** 该功能会改变模型看到的损失地形。虽然理论上成立，但会显著改变训练动态，主要用于研究与实验。

## 配置

使用 Diff2Flow 需要启用桥接，并可选切换损失函数。

### 基础设置

在 `config.json` 中添加：

```json
{
  "diff2flow_enabled": true,
  "diff2flow_loss": true
}
```

### 选项参考

#### `--diff2flow_enabled`（Boolean）
**默认：** `false`
初始化数学桥接。会为时间步计算分配小缓冲区，但除非同时设置 `diff2flow_loss`，否则不会改变训练行为。
*   **必需：** `diff2flow_loss`。
*   **支持模型：** 使用 `epsilon` 或 `v_prediction` 的模型（SD1.5、SD2.x、SDXL、DeepFloyd IF、PixArt Alpha）。

#### `--diff2flow_loss`（Boolean）
**默认：** `false`
切换训练目标。
*   **False:** 最小化预测与标准目标之间的误差（如 `MSE(pred_noise, real_noise)`）。
*   **True:** 最小化 *flow 转换后* 的预测与 flow 目标（`noise - latents`）之间的误差。

### 协同效果

Diff2Flow 与 **Scheduled Sampling** 配合极佳。

当你结合：
1.  **Diff2Flow**（拉直轨迹）
2.  **Scheduled Sampling**（在自生成 rollout 上训练）

相当于逼近 **Reflow** 或 **Rectified Flow** 模型的训练配方，可能让 SDXL 等旧架构获得更现代的稳定性与质量特性。
