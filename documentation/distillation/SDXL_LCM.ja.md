# SDXL LCM 蒸留クイックスタート（SimpleTuner）

この例では、事前学習済み SDXL 教師モデルから **LCM（Latent Consistency Model）蒸留**を使い、**4〜8 ステップの SDXL 学生** を学習します。

> **注記**: 他のモデルもベースとして利用できます。ここでは LCM の設定概念を示すために SDXL を使用しています。

LCM が実現すること:
* 超高速推論（4〜8 ステップ、通常は 25〜50）
* タイムステップ間の整合性
* 最小ステップでの高品質出力

## 📦 インストール

標準の SimpleTuner インストール [ガイド](../INSTALL.md) に従ってください:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.12 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**注記:** setup.py はプラットフォーム（CUDA/ROCm/Apple）を自動判別し、適切な依存関係をインストールします。

コンテナ環境（Vast、RunPod など）の場合:
```bash
apt -y install nvidia-cuda-toolkit
```

---

## 📁 設定

SDXL LCM 用の `config/config.json` を作成します:

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "output_dir": "/home/user/output/sdxl-lcm",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",

  "distillation_method": "lcm",
  "distillation_config": {
    "lcm": {
      "num_ddim_timesteps": 50,
      "w_min": 1.0,
      "w_max": 12.0,
      "loss_type": "l2",
      "huber_c": 0.001,
      "timestep_scaling_factor": 10.0
    }
  },

  "resolution": 1024,
  "resolution_type": "pixel",
  "validation_resolution": "1024x1024,1280x768,768x1280",
  "aspect_bucket_rounding": 64,
  "minimum_image_size": 0.5,
  "maximum_image_size": 1.0,

  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 1000,
  "max_train_steps": 10000,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",

  "lora_type": "standard",
  "lora_rank": 64,
  "lora_alpha": 64,
  "lora_dropout": 0.0,

  "validation_step_interval": 250,
  "validation_num_inference_steps": 4,
  "validation_guidance": 0.0,
  "validation_prompt": "A portrait of a woman with flowers in her hair, highly detailed, professional photography",
  "validation_negative_prompt": "blurry, low quality, distorted, amateur",

  "checkpoint_step_interval": 500,
  "checkpoints_total_limit": 5,
  "resume_from_checkpoint": "latest",

  "optimizer": "adamw_bf16",
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_weight_decay": 1e-2,
  "adam_epsilon": 1e-8,
  "max_grad_norm": 1.0,

  "seed": 42,
  "push_to_hub": true,
  "hub_model_id": "your-username/sdxl-lcm-distilled",
  "report_to": "wandb",
  "tracker_project_name": "sdxl-lcm-distillation",
  "tracker_run_name": "sdxl-lcm-4step"
}
```

### 主要な LCM 設定オプション:

- **`num_ddim_timesteps`**: DDIM ソルバのタイムステップ数（一般的に 50〜100）
- **`w_min/w_max`**: 学習時のガイダンススケール範囲（SDXL は 1.0〜12.0）
- **`loss_type`**: "l2" または "huber"（huber は外れ値に強い）
- **`timestep_scaling_factor`**: 境界条件のスケーリング（既定 10.0）
- **`validation_num_inference_steps`**: 目標ステップ数で検証（4〜8）
- **`validation_guidance`**: LCM では 0.0 を推奨（推論で CFG なし）

### 量子化学習（VRAM 低減）向け:

メモリ使用量を減らすには次を追加:
```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```

---

## 🎬 データセット設定

出力ディレクトリに `multidatabackend.json` を作成します:

```json
[
  {
    "id": "your-dataset-name",
    "type": "local",
    "crop": false,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1.0,
    "minimum_image_size": 0.5,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/your-dataset",
    "instance_data_dir": "/path/to/your/dataset",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sdxl/your-dataset",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> **重要**: LCM 蒸留には多様で高品質なデータが必要です。良好な結果には最低 1 万枚以上の画像を推奨します。

---

## 🚀 学習

1. **各サービスにログイン**（Hub 機能を使う場合）:
   ```bash
   huggingface-cli login
   wandb login
   ```

2. **学習開始**:
   ```bash
   bash train.sh
   ```

3. **進捗監視**:
   - LCM loss が減少していることを確認
   - 4〜8 ステップで検証画像の品質が維持されていること
   - 学習は通常 5k〜10k ステップ

---

## 📊 期待される結果

| 指標 | 期待値 | 備考 |
| ------ | -------------- | ----- |
| LCM Loss | < 0.1 | 継続的に減少するはず |
| 検証品質 | 4 ステップで良好 | guidance=0 が必要な場合あり |
| 学習時間 | 5〜10 時間 | 単一 A100 の場合 |
| 最終推論 | 4〜8 ステップ | ベース SDXL は 25〜50 |

---

## 🧩 トラブルシューティング

| 問題 | 解決策 |
| ------- | -------- |
| **OOM エラー** | バッチサイズを下げる、勾配チェックポイントを有効化、int8 量子化を使用 |
| **ぼやけた出力** | `num_ddim_timesteps` を増やす、データ品質を確認、学習率を下げる |
| **収束が遅い** | 学習率を 2e-4 に上げる、データセットの多様性を確保 |
| **検証が悪い** | `validation_guidance: 0.0` を使用、正しいスケジューラか確認 |
| **低ステップでアーティファクト** | 4 ステップ未満では通常。学習を長くするか `w_min/w_max` を調整 |

---

## 🔧 高度なヒント

1. **マルチ解像度学習**: SDXL は複数アスペクトの学習が有効です:
   ```json
   "validation_resolution": "1024x1024,1280x768,768x1280,1152x896,896x1152"
   ```

2. **段階的学習**: まず多いステップで学習し、後で減らす:
   - Week 1: `validation_num_inference_steps: 8` で学習
   - Week 2: `validation_num_inference_steps: 4` で微調整

3. **推論用スケジューラ**: 学習後は LCM スケジューラを使用:
   ```python
   from diffusers import LCMScheduler
   scheduler = LCMScheduler.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       subfolder="scheduler"
   )
   ```

4. **ControlNet と併用**: LCM は低ステップでのガイド付き生成に ControlNet と相性が良いです。

---

## 📚 追加リソース

- [LCM 論文](https://arxiv.org/abs/2310.04378)
- [Diffusers LCM ドキュメント](https://huggingface.co/docs/diffusers/using-diffusers/inference_with_lcm)
- [SimpleTuner 追加ドキュメント](../quickstart/SDXL.md)

---

## 🎯 次のステップ

LCM 蒸留が成功したら:
1. 4〜8 ステップでさまざまなプロンプトをテスト
2. 別のベースモデルで LCM-LoRA を試す
3. さらに少ないステップ（2〜3）での用途を試す
4. ドメイン特化データで追加微調整を検討
