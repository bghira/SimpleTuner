# DMD 蒸留クイックスタート（SimpleTuner）

この例では、大規模な flow-matching 教師モデル（例: [Wan 2.1 T2V](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)）から **DMD（Distribution Matching Distillation）** を使って **3 ステップ学生** を学習します。

DMD の特徴:

* **Generator（Student）**: 教師に近づくよう少ないステップで学習
* **Fake Score Transformer**: 教師/学生の出力を判別
* **Multi-step simulation**: 学習と推論の整合性モード（任意）

---

## ✅ ハードウェア要件


⚠️ DMD は Fake Score Transformer のため、ベースモデルの完全な 2 つ目のコピーをメモリに保持する必要があり、メモリ負荷が高いです。

必要な VRAM がない場合、14B の Wan モデルでは DMD より LCM や DCM の蒸留手法を試すことを推奨します。

疎 Attention サポートなしで 14B モデルを蒸留する場合、NVIDIA B200 が必要になる可能性があります。

LoRA 学生学習を使うと要件を大幅に下げられますが、それでもかなり重いです。

---

## 📦 インストール

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv && source .venv/bin/activate

# Install with automatic platform detection
pip install -e .
```

**注記:** setup.py はプラットフォーム（CUDA/ROCm/Apple）を自動判別し、適切な依存関係をインストールします。

---

## 📁 設定

`config/config.json` を編集します:

```json
{
    "aspect_bucket_rounding": 2,
    "attention_mechanism": "diffusers",
    "base_model_precision": "int8-quanto",
    "caption_dropout_probability": 0.1,
    "checkpoint_step_interval": 200,
    "checkpoints_total_limit": 3,
    "compress_disk_cache": true,
    "data_backend_config": "config/wan/multidatabackend.json",
    "delete_problematic_images": false,
    "disable_benchmark": false,
    "disable_bucket_pruning": true,
    "distillation_method": "dmd",
    "distillation_config": {
        "dmd_denoising_steps": "1000,757,522",
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": [0.9, 0.999],
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "fake_score_guidance_scale": 0.0,
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "num_frame_per_block": 3,
        "independent_first_frame": false,
        "same_step_across_blocks": false,
        "last_step_only": false,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": true,
        "ts_schedule_max": false,
        "min_score_timestep": 0,
        "timestep_shift": 1.0
    },
    "ema_update_interval": 5,
    "ema_validation": "ema_only",
    "flow_schedule_shift": 5,
    "grad_clip_method": "value",
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": true,
    "hub_model_id": "wan-disney-DMD-3step",
    "ignore_final_epochs": true,
    "learning_rate": 2e-5,
    "lora_alpha": 128,
    "lora_rank": 128,
    "lora_type": "standard",
    "lr_scheduler": "cosine_with_min_lr",
    "lr_warmup_steps": 100,
    "max_grad_norm": 1.0,
    "max_train_steps": 4000,
    "minimum_image_size": 0,
    "mixed_precision": "bf16",
    "model_family": "wan",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/wan-dmd",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "resolution": 480,
    "resolution_type": "pixel_area",
    "resume_from_checkpoint": "latest",
    "seed": 1000,
    "text_encoder_1_precision": "int8-quanto",
    "tracker_project_name": "dmd-training",
    "tracker_run_name": "wan-DMD-3step",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_guidance": 1.0,
    "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    "validation_num_inference_steps": 3,
    "validation_num_video_frames": 121,
    "validation_prompt": "A black and white animated scene unfolds featuring a distressed upright cow with prominent horns and expressive eyes, suspended by its legs from a hook on a static background wall. A smaller Mickey Mouse-like character enters, standing near a wooden bench, initiating interaction between the two. The cow's posture changes as it leans, stretches, and falls, while the mouse watches with a concerned expression, its face a mixture of curiosity and worry, in a world devoid of color.",
    "validation_prompt_library": "config/wan/validation_prompts_dmd.json",
    "validation_resolution": "1280x704",
    "validation_seed": 42,
    "validation_step_interval": 200,
    "webhook_config": "config/wan/webhook.json"
}
```

### 主要な DMD 設定:

* **`dmd_denoising_steps`** – 逆方向シミュレーションの対象タイムステップ（3 ステップ学生の既定は `1000,757,522`）。
* **`generator_update_interval`** – 高コストな generator リプレイを _N_ ステップごとに実行。速度を上げたい場合に増やします（品質低下の可能性）。
* **`fake_score_lr` / `fake_score_weight_decay` / `fake_score_betas`** – Fake Score Transformer の最適化ハイパーパラメータ。
* **`fake_score_guidance_scale`** – Fake Score ネットワークの classifier-free guidance（既定は無効）。
* **`num_frame_per_block`, `same_step_across_blocks`, `last_step_only`** – Self-forcing rollout 時の時間ブロックのスケジューリング制御。
* **`num_training_frames`** – 逆シミュレーション中に生成される最大フレーム数（大きいほど忠実度が上がるがメモリコスト増）。
* **`min_timestep_ratio`, `max_timestep_ratio`, `timestep_shift`** – KL サンプリング窓の形状。デフォルトから外す場合は教師の flow スケジュールに合わせてください。

---

## 🎬 データセット & データローダ

DMD を効果的に動かすには、**多様で高品質なデータ**が必要です:

```json
{
  "dataset_type": "video",
  "cache_dir": "cache/wan-dmd",
  "resolution_type": "pixel_area",
  "crop": false,
  "num_frames": 121,
  "frame_interval": 1,
  "resolution": 480,
  "minimum_image_size": 0,
  "repeats": 0
}
```

> **注記**: Disney データセットは DMD に **不適** です。**使わないでください！** あくまで例示用です。

必要条件:
> - 高いデータ量（最低 1 万本以上の動画）
> - 多様な内容（スタイル、動き、被写体の違い）
> - 高品質（圧縮アーティファクトなし）

これらは親モデルから生成できます。

---

## 🚀 学習のヒント

1. **generator 間隔は小さく**: `"generator_update_interval": 1` から開始。スループットが必要でノイズの増加を許容できる場合のみ増やします。
2. **両方の損失を監視**: wandb で `dmd_loss` と `fake_score_loss` を確認
3. **検証頻度**: DMD は収束が速いので頻繁に検証
4. **メモリ管理**:
   - `gradient_checkpointing` を使用
   - `train_batch_size` を 1 に下げる
   - `base_model_precision: "int8-quanto"` を検討

---

## 📌 DMD vs DCM

| 特徴 | DCM | DMD |
|---------|-----|-----|
| メモリ使用量 | 低い | 高い（fake score モデル） |
| 学習時間 | 長い | 短い（通常 4k ステップ） |
| 品質 | 良い | 優秀 |
| 推論ステップ | 4-8+ | 3-8 |
| 安定性 | 安定 | 要チューニング |

---

## 🧩 トラブルシューティング

| 問題 | 対処 |
|---------|-----|
| **OOM エラー** | `num_training_frames` を減らす、`generator_update_interval` を下げる、またはバッチサイズを下げる |
| **Fake score が学習しない** | `fake_score_lr` を上げるか別のスケジューラを使用 |
| **Generator の過学習** | `generator_update_interval` を 10 に増やす |
| **3 ステップの品質が悪い** | まず 2 ステップの "1000,500" を試す |
| **学習が不安定** | 学習率を下げ、データ品質を確認 |

---

## 🔬 高度なオプション

試したい方向け:

```json
"distillation_config": {
    "dmd_denoising_steps": "1000,666,333",
    "generator_update_interval": 4,
    "fake_score_guidance_scale": 1.2,
    "num_training_frames": 28,
    "same_step_across_blocks": true,
    "timestep_shift": 7.0
}
```

> ⚠️ リソースが限られたプロジェクトでは、sequence-parallel と video-sparse attention（VSA）に対応し、より効率的なランタイムを提供する FastVideo の DMD 実装を使うことを推奨します。
