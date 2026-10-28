# 训练示例

示例按 `model_family/model_flavour/vram/recipe` 组织。先选择模型系列，再选择变体和 `24g`、`48g` 或 `80g` 等显存档位。`default` 显存档位没有特定显存目标；`default` 变体使用模型配置的默认值。`8x80g` 等多 GPU 档位需要八张显存各为 80 GB 的 GPU。

命名配方文件夹区分 LoRA、LoKr、ControlNet、AnyFlow 和条件输入变体。共享数据集配置、提示词库和验证脚本位于 `_shared/`。

## 列出并运行示例

安装 SimpleTuner 后，列出可用配方，并将完整相对路径传给 `example=`：

```bash
simpletuner examples list
simpletuner train example=flux3_image/klein/24g/peft-lora
```

## 复制并自定义

复制命令在目标目录下保留分类文件夹，并将引用的 JSON 文件复制到配方文件夹中。根据自己的数据编辑配置和数据集计划，然后从该环境开始训练：

```bash
simpletuner examples copy flux3_image/klein/24g/peft-lora --dest config
# config/flux3_image/klein/24g/peft-lora/config.json
cd config/flux3_image/klein/24g/peft-lora
simpletuner train
```

在 WebUI 环境创建器和设置向导中，先选择模型系列。示例选择器只显示该系列的配方。切换系列会清除先前选择的示例。

这些配方是起点。请参阅模型指南，根据数据集和硬件调整批量大小、分辨率和训练步数。
