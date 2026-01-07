# LTX Video 2 クイックスタート

この例では、LTX-2 の Video/Audio VAE と Gemma3 テキストエンコーダを使って LTX Video 2 LoRA をトレーニングします。

## ハードウェア要件

LTX Video 2 は重量級の **19B** モデルです。以下を組み合わせます:
1.  **Gemma3**: テキストエンコーダ。
2.  **LTX-2 Video VAE**（音声条件付きの場合は Audio VAE も使用）。
3.  **19B Video Transformer**: 大規模な DiT バックボーン。

この構成は VRAM を大量に消費し、VAE の事前キャッシュでメモリ使用量が跳ね上がることがあります。

- **単一 GPU 学習**: `train_batch_size: 1` から始め、group offload を有効にしてください。
  - **注意**: 初期の **VAE 事前キャッシュ**でより多くの VRAM が必要になる場合があります。キャッシュ時だけ CPU オフロードやより大きな GPU が必要になる可能性があります。
  - **ヒント**: `config.json` に `"offload_during_startup": true` を設定し、VAE とテキストエンコーダが同時に GPU に載らないようにすると、事前キャッシュ時のメモリ圧力を大きく下げられます。
- **マルチ GPU 学習**: 余裕が必要なら **FSDP2** か強力な **Group Offload** を推奨します。
- **システム RAM**: 大きめの実行では 64GB+ を推奨します。RAM が多いほどキャッシュが安定します。

### メモリオフロード（必須級）

単一 GPU で LTX Video 2 を学習する場合、グループオフロードの有効化を推奨します。バッチや解像度の余裕を確保するためにも有効です。

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
pip install simpletuner[cuda]
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

LTX Video 2 の主要設定:

- `model_family`: `ltxvideo2`
- `model_flavour`: `2.0` (デフォルト)
- `pretrained_model_name_or_path`: `Lightricks/LTX-2`（combined checkpoint の repo）またはローカル `.safetensors` ファイル。
- `ltx2_checkpoint_filename`: 任意。ディレクトリを指していて combined checkpoint のファイル名が `ltx-2-19b-dev.safetensors` でない場合に指定。
- `train_batch_size`: `1`。A100/H100 以外では増やさないでください。
- `validation_resolution`:
  - `512x768` がテスト向けの安全なデフォルト。
  - `720x1280` (720p) は可能ですが重いです。
- `validation_num_video_frames`: **VAE の圧縮 (4x) に一致する必要があります。**
  - 5 秒 (約 12-24fps) の場合は `61` か `49`。
  - 公式: `(frames - 1) % 4 == 0`。
- `validation_guidance`: `5.0`。
- `frame_rate`: デフォルトは 25。

LTX-2 は transformer / video VAE / audio VAE / vocoder を含む約 43GB の `.safetensors` 単体チェックポイントで配布されます。
SimpleTuner はこの combined ファイルから直接読み込みます。

### 任意: VRAM 最適化

VRAM 余裕が必要なら:
- **Musubi ブロックスワップ**: `musubi_blocks_to_swap`（`4-8` を試す）と必要なら `musubi_block_swap_device`（既定 `cpu`）で、最後の Transformer ブロックを CPU からストリーミング。スループットは下がるがピーク VRAM が減る。
- **VAE パッチ畳み込み**: `--vae_enable_patch_conv=true` で LTX-2 VAE の時間方向チャンクを有効化。小さな速度低下と引き換えにピーク VRAM を削減。
- **VAE temporal roll**: `--vae_enable_temporal_roll=true` でより積極的な時間分割（速度低下は大きめ）。
- **VAE タイリング**: `--vae_enable_tiling=true` で高解像度の VAE encode/decode をタイル分割。

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
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1,
        "duration_interval": 3.0
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
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

音声 auto-split は video dataset でデフォルト有効です。sample rate/channels を調整する場合は `audio` block を
追加し、無効化したい場合は `audio.auto_split: false` を設定します。別の audio dataset を用意して
`s2v_datasets` で紐付けることもできます。SimpleTuner は audio latents を video latents と一緒にキャッシュします。

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
- **モーションのガタつき**: データセットのフレームレートがモデルの学習フレームレート (多くは 25fps) と一致しているか確認。
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

### 検証ワークフロー (T2V vs I2V)

- **T2V (text-to-video)**: `validation_using_datasets: false` のまま、`validation_prompt` または `validation_prompt_library` を使います。
- **I2V (image-to-video)**: `validation_using_datasets: true` を設定し、`eval_dataset_id` を参照画像を含む検証スプリットに指定します。検証は image-to-video パイプラインに切り替わり、画像を条件として使用します。
- **S2V (audio-conditioned)**: `validation_using_datasets: true` のとき、`eval_dataset_id` が `s2v_datasets`（またはデフォルトの `audio.auto_split`）を持つデータセットを指すようにします。検証はキャッシュ済み audio latents を自動で読み込みます。
