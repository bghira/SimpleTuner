# Kandinsky 5.0 Video クイックスタート

この例では、HunyuanVideo VAE とデュアルテキストエンコーダを使用して Kandinsky 5.0 Video LoRA (Lite または Pro) をトレーニングします。

## ハードウェア要件

Kandinsky 5.0 Video は重量級モデルです。以下を組み合わせます:
1.  **Qwen2.5-VL (7B)**: 巨大な視覚言語テキストエンコーダ。
2.  **HunyuanVideo VAE**: 高品質な 3D VAE。
3.  **Video Transformer**: 複雑な DiT アーキテクチャ。

この構成は VRAM を大量に消費しますが、「Lite」と「Pro」で必要量が異なります。

- **Lite モデル学習**: 驚くほど効率的で、**~13GB VRAM** でも学習可能。
  - **注意**: 初期の **VAE 事前キャッシュ**では巨大な HunyuanVideo VAE のため VRAM が大きく必要になります。キャッシュ時だけ CPU オフロードやより大きな GPU が必要になる可能性があります。
  - **ヒント**: `config.json` に `"offload_during_startup": true` を設定し、VAE とテキストエンコーダが同時に GPU に載らないようにすると、事前キャッシュ時のメモリ圧力を大きく下げられます。
  - **VAE が OOM する場合**: `--vae_enable_patch_conv=true` を設定して HunyuanVideo VAE の 3D Conv を分割します。少し遅くなりますがピーク VRAM が下がります。
- **Pro モデル学習**: **FSDP2** (マルチ GPU) か、LoRA と強力な **Group Offload** が必要です。具体的な VRAM/RAM 要件は未確定ですが、「多ければ多いほど良い」です。
- **システム RAM**: Lite モデルで **45GB** RAM が快適でした。安全のため 64GB+ を推奨します。

### メモリオフロード（必須級）

単一 GPU で **Pro** を学習する場合、ほぼ必ずグループオフロードを有効にする必要があります。**Lite** でも VRAM を節約してバッチや解像度を上げたい場合に推奨です。

`config.json` に追加:

<details>
<summary>設定例を表示</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## 前提条件

Python 3.12 がインストールされていることを確認してください。

```bash
python --version
```

## インストール

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

詳細は [INSTALL.md](../INSTALL.md) を参照してください。

## 環境のセットアップ

### Web インターフェース

```bash
simpletuner server
```

http://localhost:8001 でアクセスできます。

### 手動設定

ヘルパースクリプトを実行:

```bash
simpletuner configure
```

またはサンプルをコピーして手動編集:

```bash
cp config/config.json.example config/config.json
```

#### 設定パラメータ

Kandinsky 5 Video の主要設定:

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Lite モデル、約 5 秒出力。(デフォルト)
  - `t2v-lite-sft-10s`: Lite モデル、約 10 秒出力。
  - `t2v-pro-sft-5s-hd`: Pro モデル、約 5 秒、高解像度学習。
  - `t2v-pro-sft-10s-hd`: Pro モデル、約 10 秒、高解像度学習。
  - `i2v-lite-5s`: 画像から動画の Lite、5 秒出力 (コンディショニング画像が必要)。
  - `i2v-pro-sft-5s`: 画像から動画の Pro SFT、5 秒出力 (コンディショニング画像が必要)。
  - *(上記すべてに pretrain バリアントあり)*
- `train_batch_size`: `1`。A100/H100 以外では増やさないでください。
- `validation_resolution`:
  - `512x768` がテスト向けの安全なデフォルト。
  - `720x1280` (720p) は可能ですが重いです。
- `validation_num_video_frames`: **VAE の圧縮 (4x) に一致する必要があります。**
  - 5 秒 (約 12-24fps) の場合は `61` か `49`。
  - 公式: `(frames - 1) % 4 == 0`。
- `validation_guidance`: `5.0`。
- `frame_rate`: デフォルトは 24。

### 任意: CREPA 時間方向正則化

ちらつきを抑え、被写体をフレーム間で安定させるには:
- **Training → Loss functions** で **CREPA** を有効化。
- 推奨初期値: **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**。
- 既定のビジョンエンコーダ (`dinov2_vitg14`, サイズ `518`) を使い、必要なら `dinov2_vits14` + `224` へ。
- 初回は DINOv2 重み取得のためネットワーク (またはキャッシュ済み torch hub) が必要。
- **Drop VAE Encoder** はキャッシュ済み latent だけで学習する場合のみ有効にし、それ以外はオフ。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### データセットの考慮事項

動画データセットは慎重に設定する必要があります。`config/multidatabackend.json` を作成します:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

`video` セクション内:
- `num_frames`: 学習の目標フレーム数。
- `min_frames`: 最短の動画長 (これより短い動画は破棄)。
- `max_frames`: 最長の動画長フィルタ。
- `bucket_strategy`: バケット分け方法:
  - `aspect_ratio` (デフォルト): 空間アスペクト比のみでグループ化。
  - `resolution_frames`: `WxH@F` 形式 (例: `1920x1080@61`) で解像度/長さを併せてグループ化。
- `frame_interval`: `resolution_frames` 使用時にフレーム数を丸める間隔。

> caption_strategy のオプションと要件は [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

#### ディレクトリ設定

```bash
mkdir -p datasets/videos
</details>

# .mp4 / .mov をここに配置
# キャプション用の .txt を同名で配置
```

#### ログイン

```bash
wandb login
huggingface-cli login
```

### トレーニングの実行

```bash
simpletuner train
```

## メモとトラブルシューティングのヒント

### Out of Memory (OOM)

動画学習は非常に負荷が高いです。OOM の場合:

1.  **解像度を下げる**: 480p (`480x854` など) を試す。
2.  **フレーム数を減らす**: `validation_num_video_frames` とデータセットの `num_frames` を `33` または `49` に。
3.  **オフロードを確認**: `--enable_group_offload` が有効か確認。

### 検証動画の品質

- **黒/ノイズ動画**: `validation_guidance` が高すぎる (> 6.0) または低すぎる (< 2.0) ことが原因の場合が多いです。`5.0` に合わせてください。
- **モーションのガタつき**: データセットのフレームレートがモデルの学習フレームレート (多くは 24fps) と一致しているか確認。
- **静止/ほぼ動かない**: 学習不足か、プロンプトが動きを記述していない可能性。"camera pans right"、"zoom in"、"running" などを使ってください。

### TREAD トレーニング

TREAD は動画にも有効で、計算を節約するため強く推奨されます。

`config.json` に追加:

<details>
<summary>設定例を表示</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

比率によっては 25-40% 程度高速化できます。

### I2V (Image-to-Video) トレーニング

`i2v` フレーバーを使う場合:
- SimpleTuner は学習動画の最初のフレームを自動でコンディショニング画像として抽出します。
- 学習中は最初のフレームが自動的にマスクされます。

#### I2V検証オプション

i2vモデルの検証には、2つのオプションがあります：

1. **自動抽出された最初のフレーム**: デフォルトでは、検証は動画サンプルの最初のフレームを使用します。

2. **別の画像データセット**（よりシンプルな設定）: `--validation_using_datasets=true`を使用し、`--eval_dataset_id`を画像データセットに指定します：

```json
{
  "validation_using_datasets": true,
  "eval_dataset_id": "my-image-dataset"
}
```

これにより、トレーニング時に使用される複雑なコンディショニングデータセットのペアリング設定なしで、任意の画像データセットを検証動画の最初のフレームコンディショニング入力として使用できます。
