An experimental feature in SimpleTuner implements the ideas behind ["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) to provide a stable loss value for evaluation.

Due to its experimental nature, it may cause problems or lack functionality / integration that a fully finalised feature might have.

It is fine to use this feature in production, but beware of the potential for bugs or changes in future versions.

Example dataloader:

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

- Eval image datasets can be configured exactly like a normal image dataset.
- The evaluation dataset is **not** used for training.
- It's recommended to use images that represent concepts outside of your training set.

To configure and enable evaluation loss calculations:

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb",
}
```

Evaluations can now be scheduled by step or by epoch. `--eval_epoch_interval` accepts decimal values, so `0.5`
will run evaluation twice per epoch. If you set both `--eval_steps_interval` and `--eval_epoch_interval`, the
trainer will log a warning and run evaluations on both schedules.

> **Note**: Weights & Biases (wandb) is currently required for the full evaluation charting functionality. Other trackers only receive the single mean value.
