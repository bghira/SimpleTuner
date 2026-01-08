## Lumina2 クイックスタート

この例では、Lumina2 の LoRA もしくはフルモデルのファインチューニングを行います。

### ハードウェア要件

Lumina2 は 2B パラメータのモデルで、Flux や SD3 のような大型モデルよりもはるかに扱いやすいです。小型であることにより:

Rank-16 LoRA の場合:
- LoRA 学習で約 12〜14GB の VRAM
- フルモデルのファインチューニングで約 16〜20GB の VRAM
- 起動時に約 20〜30GB のシステム RAM

必要なもの:
- **最小**: RTX 3060 12GB または RTX 4060 Ti 16GB
- **推奨**: RTX 3090、RTX 4090、A100（より高速）
- **システム RAM**: 32GB 以上推奨

### 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.12 python3.12-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock（など）の場合、CUDA 12.2-12.8 イメージで以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pip で SimpleTuner をインストールします:

```bash
pip install simpletuner[cuda]
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### 環境のセットアップ

SimpleTuner を実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

- `model_type` - LoRA トレーニングなら `lora`、フルファインチューニングなら `full` に設定します。
- `model_family` - `lumina2` に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - GPU メモリとデータセットサイズに応じて 1〜4。
- `validation_resolution` - Lumina2 は複数解像度をサポートします。よく使われるのは `1024x1024`、`512x512`、`768x768`。
- `validation_guidance` - Lumina2 は classifier-free guidance を使用します。3.5〜7.0 が有効です。
- `validation_num_inference_steps` - 20〜30 ステップが適切です。
- `gradient_accumulation_steps` - 大きなバッチサイズを擬似的に再現できます。2〜4 が有効です。
- `optimizer` - `adamw_bf16` を推奨します。`lion` や `optimi-stableadamw` も動作します。
- `mixed_precision` - 最良の結果のため `bf16` を維持します。
- `gradient_checkpointing` - VRAM を節約するため `true` に設定します。
- `learning_rate` - LoRA: `1e-4`〜`5e-5`。フルファインチューニング: `1e-5`〜`1e-6`。

#### Lumina2 の設定例

これは `config.json` に入れます。

<details>
<summary>設定例を表示</summary>

```json
{
    "base_model_precision": "int8-torchao",
    "checkpoint_step_interval": 50,
    "data_backend_config": "config/lumina2/multidatabackend.json",
    "disable_bucket_pruning": true,
    "eval_steps_interval": 50,
    "evaluation_type": "clip",
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "lumina2-lora",
    "learning_rate": 1e-4,
    "lora_alpha": 16,
    "lora_rank": 16,
    "lora_type": "standard",
    "lr_scheduler": "constant",
    "max_train_steps": 400000,
    "model_family": "lumina2",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/lumina2",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "seed": 42,
    "tracker_project_name": "lumina2-training",
    "tracker_run_name": "lumina2-lora",
    "train_batch_size": 4,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 40,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 50
}
```
</details>

Lycoris トレーニングの場合は `lora_type` を `lycoris` に切り替えてください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json` には「プライマリ検証プロンプト」があります。さらに、プロンプトライブラリを作成します:

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

設定に追加:
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### データセットの考慮事項

Lumina2 は高品質な学習データから恩恵を受けます。`--data_backend_config`（`config/multidatabackend.json`）を作成します:

> 💡 **ヒント:** ディスク容量が問題になる大規模データセットの場合、`--vae_cache_disable` を使ってオンライン VAE エンコードを行い、結果をディスクにキャッシュしないようにできます。

```json
[
  {
    "id": "lumina2-training",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 2048,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/lumina2/training",
    "instance_data_dir": "/datasets/training",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/lumina2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。

データセットディレクトリを作成し、実際の場所に合わせてパスを更新してください。

```bash
mkdir -p /datasets/training
</details>

# /datasets/training/ に画像とキャプションファイルを配置します
```

キャプションファイルは画像と同名で、拡張子が `.txt` である必要があります。

#### WandB へのログイン

SimpleTuner には **任意** のトラッカー対応があり、主に Weights & Biases に焦点を当てています。`report_to=none` で無効化できます。

wandb を有効にするには、以下を実行します:

```bash
wandb login
```

#### Huggingface Hub へのログイン

チェックポイントを Huggingface Hub にプッシュするには、以下を実行してください:
```bash
huggingface-cli login
```

### トレーニングの実行

SimpleTuner ディレクトリから、トレーニングを開始するいくつかのオプションがあります:

**オプション 1（推奨 - pip install）:**
```bash
pip install simpletuner[cuda]
simpletuner train
```

**オプション 2（Git clone 方法）:**
```bash
simpletuner train
```

**オプション 3（レガシー方法 - まだ動作します）:**
```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

## Lumina2 のトレーニングのヒント

### 学習率

#### LoRA トレーニング
- `1e-4` から始め、結果に応じて調整します
- Lumina2 は学習が速いので、初期イテレーションをよく監視してください
- rank 8〜32 は多くの用途で有効です。64〜128 は監視が必要で、256〜512 は新しいタスクをモデルに学習させる用途で有用です

#### フルファインチューニング
- 学習率は低めに設定（`1e-5`〜`5e-6`）
- 安定性のため EMA（指数移動平均）を検討
- 勾配クリッピング（`max_grad_norm`）1.0 を推奨

### 解像度の考慮

Lumina2 は柔軟な解像度をサポートします:
- 1024x1024 での学習が最良の品質
- 混合解像度（512px、768px、1024px）の品質影響は未検証
- アスペクト比バケッティングは Lumina2 で良好に動作します

### 学習期間

2B パラメータの効率性により:
- LoRA 学習は 500〜2000 ステップで収束することが多い
- フルファインチューニングは 2000〜5000 ステップが必要な場合がある
- モデルが速く学習するため、検証画像を頻繁に確認してください

### よくある問題と解決策

1. **収束が速すぎる**: 学習率を下げ、Lion から AdamW に切り替える
2. **生成画像のアーティファクト**: 高品質データを使用し、学習率を下げることを検討
3. **メモリ不足**: 勾配チェックポイントを有効にし、バッチサイズを下げる
4. **過学習しやすい**: 正則化データセットを使用する

## 推論のヒント

### 学習済みモデルの利用

Lumina2 モデルは以下で使用できます:
- Diffusers ライブラリを直接使用
- 適切なノードを備えた ComfyUI
- Gemma2 ベースモデルをサポートする他の推論フレームワーク

### 推奨推論設定

- Guidance scale: 4.0〜6.0
- Inference steps: 20〜50
- 最高の結果のため、学習時と同じ解像度を使用

## メモ

### Lumina2 の利点

- 2B パラメータによる高速学習
- サイズに対する高品質
- 多様なトレーニングモード（LoRA、LyCORIS、フル）をサポート
- 効率的なメモリ使用

### 現在の制限

- ControlNet には未対応
- text-to-image 生成のみ
- 最良の結果には高品質なキャプションが必要

### メモリ最適化

大規模モデルと異なり、Lumina2 では通常以下は不要です:
- モデル量子化
- 極端なメモリ最適化手法
- 複雑な混合精度戦略
