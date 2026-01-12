SimpleTuner 的实验性功能实现了 ["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) 的思想，用于提供稳定的评估损失值。

由于处于实验阶段，它可能会引入问题，或缺少完整功能/集成。

可以在生产中使用，但请注意未来版本可能出现的 bug 或行为变化。

数据加载器示例：

```json
[
    {
        "id": "something-special-to-remember-by",
        "crop": false,
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 512,
        "maximum_image_size": 1536,
        "target_downsample_size": 512,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/sana",
        "vae_cache_clear_each_epoch": false,
        "skip_file_discovery": ""
    },
    {
        "id": "sana-eval",
        "type": "local",
        "dataset_type": "eval",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/sana-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/sana"
    }
]
```

- 评估图像数据集可以像普通图像数据集一样配置。
- 评估数据集**不会**用于训练。
- 建议使用代表训练集之外概念的图像。

配置并启用评估损失计算：

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb"
}
```

现在评估可以按步或按 epoch 调度。`--eval_epoch_interval` 接受小数值，因此 `0.5` 会在每个 epoch 执行两次评估。如果同时设置 `--eval_steps_interval` 和 `--eval_epoch_interval`，训练器会记录警告并同时按两种节奏执行评估。

若要保留评估数据集配置但禁用评估损失计算（例如仅用于 CLIP 评分）：

```json
{
    "--eval_loss_disable": true
}
```

这在你想用评估数据集做 CLIP 分数（`--evaluation_type clip`）但不希望每个时间步都计算验证损失时很有用。

> **注记**：目前完整的评估图表功能需要 Weights & Biases（wandb）。其他跟踪器只会收到单一均值。
