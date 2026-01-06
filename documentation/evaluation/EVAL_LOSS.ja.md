SimpleTuner の実験的機能として、["Demystifying SD fine-tuning"](https://github.com/spacepxl/demystifying-sd-finetuning) の考え方を実装し、評価用に安定した損失値を提供します。

実験的機能のため、完全版で備わるはずの機能/統合が不足していたり、問題を引き起こす可能性があります。

本番利用は可能ですが、将来のバージョンでのバグや変更の可能性に注意してください。

データローダ例:

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

- 評価画像データセットは通常の画像データセットと同様に設定できます。
- 評価データセットは学習には**使用されません**。
- 学習セット外の概念を表す画像の使用を推奨します。

評価損失計算を設定・有効化するには:

```json
{
    "--eval_steps_interval": 10,
    "--eval_epoch_interval": 0.5,
    "--num_eval_images": 1,
    "--report_to": "wandb"
}
```

評価はステップまたはエポックでスケジュールできます。`--eval_epoch_interval` は小数を受け付けるため、`0.5` なら 1 エポックあたり 2 回評価が実行されます。`--eval_steps_interval` と `--eval_epoch_interval` の両方を設定すると、トレーナーは警告をログし、両方のスケジュールで評価を実行します。

評価データセットを維持したまま評価損失計算を無効化するには（例: CLIP スコアのみ）:

```json
{
    "--eval_loss_disable": true
}
```

これは、各タイムステップでの検証損失計算のオーバーヘッドなしに、評価データセットで CLIP スコア（`--evaluation_type clip`）を使いたい場合に便利です。

> **注記**: 現時点で完全な評価チャート機能には Weights & Biases（wandb）が必要です。他のトラッカーには平均値のみ送信されます。
