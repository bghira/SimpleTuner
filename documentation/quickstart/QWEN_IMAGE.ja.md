## Qwen Image 2.1

Qwen Image 2.1 がデフォルトです（`model_flavour: "v2.1"`）。`Qwen/Qwen-Image-2.1` を使用し、32 ブロックの Transformer、Qwen3-VL テキストエンコーダー、空間圧縮率 16 倍の 64 チャンネル VAE を備えています。

`qwen_image.peft-lora` の例は `RareConcepts/Domokun` を 512px で学習し、トリガーに `🟫` を使います。まず BF16（`base_model_precision: "no_change"`）を使用し、メモリが不足する場合は勾配チェックポイントを有効にしてください。2.1 専用の潜在表現とテキストのキャッシュを使用するため、旧バージョンのキャッシュを再利用しないでください。

```bash
simpletuner train example=qwen_image.peft-lora
```

検証には `validation_guidance: 1.0`、`validation_guidance_real: 1.0`、`validation_num_inference_steps: 40` を使用します。検証プロンプトにもトリガーを含め、対象を学習できたか確認してください。

Qwen Image 2.1 は、使われない時間方向の特徴キャッシュを保持せずに単一画像をデコードします。H200 上で BF16 を用いて 2048×2048 の画像を 1 枚単独でデコードした測定では、出力を完全に維持しながら、割り当てメモリのピークが 26.87 GiB から 15.28 GiB に減りました。タイルデコードはさらにメモリを節約しますが、空間的な文脈が制限されるため色の継ぎ目が生じる場合があります。未使用キャッシュの削除では、この継ぎ目は解消しません。

旧バージョンも引き続き利用できます。`v1.0` は Qwen-Image、`v2.0` は Qwen-Image-2512 を選択し、`edit-*` は従来のチェックポイントを使用します。これらのアダプターと潜在表現キャッシュは 2.1 と互換性がありません。

250 step の Domokun 設定はスループット測定用であり、安定した収束を保証するレシピではありません。以前の checkpoint は再ロード後に認識可能な Domokun を生成しましたが、新規の 250 step 学習では再現できませんでした。padding mask の保持、コンパイルの無効化、以前の RoPE 式への復元でも改善しませんでした。キャッシュ latent のデコードでは正しい被写体を確認しています。学習品質低下の原因は未解決で、時間の表は attention backend 間の画質の同等性を示すものではありません。

### VRAM プリセット

これらの例は 512px、BF16、rank-32 LoRA、Optimi Lion、リージョナルコンパイルを使用し、勾配チェックポイントは無効です。初回はコンパイル時間が必要なので、ウォームアップ後の学習ステップを比較してください。24 GB と 32 GB のメモリ予算は L40S で確認しており、各容量の別 GPU での測定ではありません。

| VRAM 予算 | 例 | データセットのバッチサイズ | ピーク VRAM (GiB) | ウォーム後のステップ (秒) |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 1 | 20.6 | 0.238 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 2 | 26.5 | 0.390 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 2 | 26.5 | 0.390 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 10 | 71.8 | 0.639 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 20 | 128.4 | 1.223 |

L40S（24/32/48 GB プリセット）、H100（80 GB）、H200（144 GB）で 20 ステップ測定し、最初の 5 ステップを計時から除外しました。ピーク VRAM は初期化を含みます。512px と各バッチサイズでの結果であり、大きな画像や長いプロンプトで同じ使用量を保証するものではありません。

48 GB プリセットもバッチサイズ 2 を使用します。L40S では画像あたりのスループットがバッチ 3、4、5 より良好でした。バッチ 5 は 43.3 GiB に収まり 0.991 秒/ステップ、バッチ 2 は 0.390 秒/ステップでした。

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

各例に付属する、バッチサイズが明示されたデータセットファイルを使用してください。バッチサイズやデータセット設定を変更する場合は新しい学習を開始し、互換性のない学習状態チェックポイントを再利用しないでください。

VRAM を減らすには `gradient_checkpointing: true` と `gradient_checkpointing_interval: 2` を設定します。現在は連続する2つの block を1組として checkpoint します。実測の比較は [Qwen Image 2.1 の checkpoint と attention 測定](../experimental/SEGMENTED_CHECKPOINTING.ja.md#qwen-image-21)を参照してください。以前の1つおきの block を対象にした測定は旧方式の結果です。これらのプリセットは int8 checkpoint を使わず BF16 で収まります。

文から画像への経路では、テンソルの値に依存するシーケンス構築を避け、グラフブレークなしでキャプチャできます。実数 RoPE により Inductor が正規化と回転を融合し、変調・残差・MLP の後処理もコンパイルします。既存の Hopper CuTe ConvRot GEMM と推論専用 LTX RoPE カーネルは、これらの学習例では使用しません。


### 実験的な AnyFlow パイロット

3 つの `qwen_image-2.1-anyflow-stage*.peft-lora` サンプルは、区間蒸留によってベースモデルの挙動を保ちながら `🟫` を導入できるか検証します。実験用の設定です。Qwen のモデルカードは、以前の劣化が guidance distillation に起因するとは示していません。

全段階で 1024px、BF16、AdamW、rank 32、batch 4 を使用します。段階 1 は H200 で勾配チェックポイントを無効にし、段階 2 と 3 は連続 2 ブロック単位で適用します。上記の 512px スループット設定とは負荷が異なります。実行前に `webshart` をインストールしてください。

これらの AnyFlow 例は `grad_clip_method: "value"` と `max_grad_norm: 0.01` を明示的に指定し、各勾配要素を ±0.01 に制限します。これは全体の勾配ノルムの上限ではありません。ノルムを 1.0 でクリップする実験では、`grad_clip_method: "norm"` と `max_grad_norm: 1.0` の両方を設定してください。

1. **段階 1：** 学習率 `1e-5` で forward AnyFlow を 10,000 回更新します。CC12M は `webshart/cc12m-structured-captions` の `caption_key: "long_caption"`、e621 は `webshart/e621-2024-webp-4Mpixel-webshart-indices` を使用します。各事前知識データセットは条件を満たす 4,096 枚まで、サンプリング重みは各 0.49 です。`RareConcepts/Domokun` はトリガー `🟫`、重み 0.02、`repeats: 0` を使用します。
2. **段階 2：** 学習率 `2e-6` で on-policy DMD を 2,000 回更新し、同じデータ混合で forward 目的も学習します。これは AnyFlow DMD であり、選好ペアの DPO ではありません。両段階にキャラクター画像を含めます。凍結教師だけでは新しいキャラクターを教えられません。
3. **段階 3：** 学習率 `5e-7` で 100 回更新します。Domokun の重みは 0.5、正則化用の 2 データセットは各 0.25、各 64 枚までです。全区間を `r=t` として生の flow 目標を使い、区間埋め込みを保持して短い教師あり微調整を行います。正則化バッチは adapter を無効にしたベース予測を使います。

段階 1 を 20,000 回まで延長する前に、2,000、5,000、10,000 回時点のチェックポイントを確認します。段階 2 の上限は 2,000 回とし、同条件の検証画像が悪化した場合は早めに停止します。更新回数だけで完成モデルとは判断できません。

可変のキャプション長に対応するため動的コンパイルを有効にしています。`schedule_shift: 2.000802574061872` は公開された scheduler の 1024px（4,096 latent tokens）での設定に対応します。解像度を変更する場合は再計算してください。

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

段階 2 と 3 は `init_lora` で直前の最終 adapter を読み込み、optimizer と dataloader の状態を初期化します。重みは未消費のデータセット間の選択を制御し、最終的なサンプル比率を保証しません。この限定パイロットは事前知識コーパス全体を使用しません。文字列の `long_caption` は既存の Webshart caption selector で扱えるため、JSON オブジェクト caption の対応は不要です。

段階 1、2 は `diffusion_target: "base_prediction"` と `fuse_guidance_scale: 1.0` を使用します。diffusion 分岐は凍結された条件付き予測場を保ち、他の分岐はデータから学習します。これは保持の仮説であり、キャラクター習得を保証しません。付属のキャラクターと事前知識プロンプトを 4、16、40 推論ステップで比較し、ベースモデルの 40 ステップ出力を基準にしてください。自動検証は 4 ステップです。目的関数とチェックポイントの要件は [AnyFlow](../experimental/ANYFLOW.ja.md) を参照してください。

完了した試験では、ステージ 1 は 10,000 更新、ステージ 2 は 2,000 更新に到達しました。適応器を再ロードした 40 ステップのキツネ画像とキャラクタープロンプトの画像は一貫性を保ちましたが、後者は Domokun ではなく人物を生成しました。4 ステップ画像にはぼけやノイズが残りました。別の L40S 比較では両チェックポイントを 1024px、シード 42、CFG 1/2/4/6、空のネガティブプロンプトで検証しました。前向き呼び出しのアサーションにより、CFG が 1 より大きい場合は各ステップで 2 回実行されることを確認しました。確認したキツネと海辺の画像は改善せず、高い CFG は彩度とアーティファクトを増やしました。原因はまだ特定できておらず、ステージ 3 は未検証です。

同じチェックポイントからステージ 1 を各 1,000 更新継続し、要素ごとのクリッピング閾値 0.01 と 1.0 を比較しました。再読み込み後の 4 ステップのキツネ画像は両方ともノイズが残り、40 ステップの海辺の画像も人物を描写しました。クリッピングが適用された更新は 0.01 で 65/1,000、1.0 で 0/1,000 でした。両群とも未修正オプティマイザを使用し、クリッピング適用前から初期勾配が異なったため、微小な差を閾値だけに帰することはできません。

<a id="qwen21-optimizer-correction"></a>

ここで紹介したステージ 1 とアシスタントの試験は、AdamW BF16 の確率的加算ヘルパーの修正前に実行されました。この関数は `input + alpha * other` ではなく `other + alpha * input` を計算していました。β₁ = 0.9 では、一次モーメントの更新が `m = 0.9 * m + 0.1 * g` ではなく `m = 0.09 * m + g` になっていました。厳密に表現できる値を使う回帰テストを CPU、MPS、CUDA で実施済みです。画像の観察結果はこれらのチェックポイントには有効ですが、アーキテクチャや蒸留目的だけを評価する試験にはなりません。修正済みオプティマイザで学習を再検証する必要があります。

修正済みオプティマイザ、同じ開始チェックポイント、要素ごとのクリッピング値 1.0 で 1,000 更新の継続試験を繰り返しても、確認した 4 ステップのキツネと海辺の画像は改善しませんでした。40 ステップではキツネは整合性を保ち、海辺のプロンプトは依然として人物を生成しました。これは既存チェックポイントの回復を検証する試験であり、修正済みオプティマイザで最初から学習する試験ではありません。全体的な失敗原因は未解明です。

### 旧版 Qwen Image の設定（v1.0 / v2.0）

> 🆕 edit チェックポイントを探していますか？ 参照ペア学習の手順は [Qwen Image Edit quickstart](./QWEN_EDIT.md) を参照してください。

この例では、20B パラメータの Vision-Language モデルである Qwen Image の LoRA をトレーニングします。サイズが大きいため、積極的なメモリ最適化が必要です。

24GB GPU は最低ラインで、さらに強い量子化と慎重な設定が必要です。スムーズな運用には 40GB+ を強く推奨します。

24G で学習する場合、検証はメモリ不足になりやすいため、低解像度や int8 を超える強い量子化が必要です。

### ハードウェア要件

Qwen Image は 20B パラメータのモデルで、洗練されたテキストエンコーダだけでも量子化前で ~16GB VRAM を消費します。16 チャンネルの独自 VAE を使用します。

**重要な制限:**
- **AMD ROCm と MacOS は未対応**（効率的な Flash Attention がないため）
- バッチサイズ > 1 は現在正しく動作しないため、gradient accumulation を使用してください
- TREAD（Text-Representation Enhanced Adversarial Diffusion）は未対応

### 前提条件

Python がインストールされていることを確認してください。SimpleTuner は 3.10 から 3.12 でうまく動作します。

以下を実行して確認できます:

```bash
python --version
```

Ubuntu に Python 3.12 がインストールされていない場合は、以下を試してください:

```bash
apt -y install python3.13 python3.13-venv
```

#### コンテナイメージの依存関係

Vast、RunPod、TensorDock（など）の場合、CUDA 12.2-12.8 イメージで CUDA 拡張のコンパイルを有効にするには以下が機能します:

```bash
apt -y install nvidia-cuda-toolkit
```

### インストール

pip で SimpleTuner をインストールします:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](../INSTALL.md)を参照してください。

### 環境のセットアップ

SimpleTuner を実行するには、設定ファイル、データセットとモデルのディレクトリ、およびデータローダー設定ファイルをセットアップする必要があります。

#### 設定ファイル

実験的なスクリプト `configure.py` を使用すると、インタラクティブなステップバイステップの設定でこのセクションを完全にスキップできる可能性があります。一般的な落とし穴を避けるための安全機能が含まれています。

**注意:** これはデータローダーを設定しません。後で手動で設定する必要があります。

実行するには:

```bash
simpletuner configure
```

> ⚠️ Hugging Face Hub にアクセスしにくい国にいるユーザーは、システムが使用する `$SHELL` に応じて `~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加してください。

手動で設定したい場合:

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

そこで、以下の変数を変更する必要があります:

- `model_type` - `lora` に設定します。
- `lora_type` - PEFT LoRA は `standard`、LoKr は `lycoris` を使用します。
- `model_family` - `qwen_image` に設定します。
- `model_flavour` - `v1.0` に設定します。
- `output_dir` - チェックポイントと検証画像を保存するディレクトリに設定します。フルパスの使用を推奨します。
- `train_batch_size` - 利用可能な VRAM に合わせて設定します。現在の SimpleTuner の Qwen override ではバッチサイズ 1 超も利用できます。
- `gradient_accumulation_steps` - 1 ステップあたりの VRAM を増やさず実効バッチを大きくしたい場合は 2〜8 を設定します。
- `validation_resolution` - `1024x1024` もしくはメモリ制約のためより低い値に設定します。
  - 24G は現状 1024x1024 検証に対応できません。サイズを下げてください。
  - 他の解像度はカンマ区切りで指定できます: `1024x1024,768x768,512x512`
- `validation_guidance` - 3.0〜4.0 前後が良好です。
- `validation_num_inference_steps` - 30 前後を使用します。
- `use_ema` - `true` に設定すると滑らかな結果が得られますがメモリを追加で消費します。

- `optimizer` - 良好な結果のため `optimi-lion` を使用するか、余裕があれば `adamw-bf16`。
- `mixed_precision` - Qwen Image は `bf16` 必須です。
- `gradient_checkpointing` - **必須**（`true`）。妥当なメモリ使用量のため必要です。
- `base_model_precision` - **強く推奨** `int8-quanto` または `nf4-bnb`（24GB では必須）。
- `quantize_via` - 小型 GPU の量子化 OOM を避けるため `cpu` に設定します。
- `quantize_activations` - 学習品質維持のため `false` にします。

24GB GPU 向けのメモリ最適化設定:
- `lora_rank` - 8 以下を使用。
- `lora_alpha` - `lora_rank` と同じ値にする。
- `flow_schedule_shift` - 1.73 に設定（1.0〜3.0 で調整）。

最小構成の `config.json` 例:

<details>
<summary>設定例を表示</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ マルチ GPU ユーザーは、使用する GPU 数の設定については [このドキュメント](../OPTIONS.md#environment-configuration-variables) を参照してください。

> ⚠️ **24GB GPU で重要:** テキストエンコーダ単体で ~16GB VRAM を消費します。`int2-quanto` または `nf4-bnb` を使うことで大幅に削減できます。

動作確認用の既知構成:

**オプション 1（推奨 - pip install）:**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**オプション 2（Git clone 方法）:**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**オプション 3（レガシー方法 - まだ動作します）:**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTuner には、トレーニングの安定性とパフォーマンスを大幅に向上させる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが自身の入力を生成することで露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

#### 検証プロンプト

`config/config.json` 内には「プライマリ検証プロンプト」があり、これは通常、単一の被写体やスタイルでトレーニングしているメインの instance_prompt です。さらに、検証中に実行する追加のプロンプトを含む JSON ファイルを作成できます。

設定ファイル例 `config/user_prompt_library.json.example` には以下の形式が含まれています:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

ニックネームは検証のファイル名になるため、短くファイルシステムと互換性のあるものにしてください。

このプロンプトライブラリを使用するには、`config.json` に以下を追加します:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

多様なプロンプトのセットは、モデルが正しく学習しているかを判断する助けになります:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### CLIP スコアトラッキング

モデルのパフォーマンスをスコアリングするための評価を有効にしたい場合は、CLIP スコアの設定と解釈に関する情報について [このドキュメント](../evaluation/CLIP_SCORES.md) を参照してください。

#### 安定した評価損失

モデルのパフォーマンスをスコアリングするために安定した MSE 損失を使用したい場合は、評価損失の設定と解釈に関する情報について [このドキュメント](../evaluation/EVAL_LOSS.md) を参照してください。

#### 検証プレビュー

SimpleTuner は Tiny AutoEncoder モデルを使用して生成中の中間検証プレビューのストリーミングをサポートしています。これにより、webhook コールバックを介してリアルタイムで検証画像が生成されるのを段階的に確認できます。

有効にするには:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**要件:**
- Webhook 設定
- 検証が有効

Tiny AutoEncoder のオーバーヘッドを削減するには、`validation_preview_steps` をより高い値（例: 3 または 5）に設定してください。`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、ステップ 5、10、15、20 でプレビュー画像を受け取ります。

#### Flow スケジュールシフト

Qwen Image はフローマッチングモデルとして、生成過程のどの部分を学習するかを制御するタイムステップシフトに対応しています。

`flow_schedule_shift` の目安:
- 低い値（0.1〜1.0）: 細部重視
- 中程度（1.0〜3.0）: バランス（推奨）
- 高い値（3.0〜6.0）: 大域的構図重視

##### 自動シフト
`--flow_schedule_auto_shift` を有効にすると、解像度依存のタイムステップシフトが適用されます。大きな画像には高いシフト値、小さな画像には低いシフト値が使用され、安定する一方で平凡になる可能性があります。

##### 手動指定
Qwen Image では `--flow_schedule_shift` を 1.73 にするのが出発点として推奨されます。データセットや目的に応じて調整してください。

#### データセットの考慮事項

モデルをトレーニングするには十分なデータセットが不可欠です。データセットサイズには制限があり、モデルを効果的にトレーニングできる十分な大きさのデータセットであることを確認する必要があります。

> ℹ️ 画像が少なすぎる場合、**no images detected in dataset** というメッセージが表示されることがあります。`repeats` 値を増やすことでこの制限を克服できます。

> ⚠️ **重要**: 現在の制約により `train_batch_size` は 1 に固定し、代わりに `gradient_accumulation_steps` で実効バッチを増やしてください。

以下を含む `--data_backend_config`（`config/multidatabackend.json`）ドキュメントを作成します:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ `.txt` のキャプションがある場合は `caption_strategy=textfile` を使用してください。
> caption_strategy のオプションと要件については [DATALOADER.md](../DATALOADER.md#caption_strategy) を参照してください。
> ℹ️ OOM を避けるため、テキスト埋め込みの `write_batch_size` は小さくしています。

次に、`datasets` ディレクトリを作成します:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

これにより、約 10k の写真サンプルが `datasets/pseudo-camera-10k` ディレクトリにダウンロードされ、自動的に作成されます。

Dreambooth の画像は `datasets/dreambooth-subject` ディレクトリに入れてください。

#### WandB と Huggingface Hub へのログイン

特に `--push_to_hub` と `--report_to=wandb` を使う場合は、トレーニング開始前に WandB と HF Hub にログインしておく必要があります。

Git LFS リポジトリに手動でアイテムをプッシュする場合は、`git config --global credential.helper store` も実行してください。

以下のコマンドを実行します:

```bash
wandb login
```

および

```bash
huggingface-cli login
```

指示に従って両方のサービスにログインしてください。

</details>

### トレーニングの実行

SimpleTuner ディレクトリから、以下を実行するだけです:

```bash
./train.sh
```

これにより、テキスト埋め込みと VAE 出力のディスクへのキャッシングが開始されます。

詳細については、[データローダー](../DATALOADER.md) と [チュートリアル](../TUTORIAL.md) のドキュメントを参照してください。

### メモリ最適化のヒント

#### 最低 VRAM 構成（24GB 最低）

Qwen Image の最低 VRAM 構成は約 24GB 必要です:

- OS: Ubuntu Linux 24
- GPU: 単一の NVIDIA CUDA デバイス（24GB 最低）
- システムメモリ: 64GB+ 推奨
- ベースモデル精度:
  - NVIDIA: `int2-quanto` または `nf4-bnb`（24GB 必須）
  - `int4-quanto` でも動作するが品質低下の可能性
- オプティマイザ: `optimi-lion` または `bnb-lion8bit-paged` でメモリ効率重視
- 解像度: まず 512px または 768px、余裕があれば 1024px
- バッチサイズ: 1（制約のため必須）
- 勾配蓄積: 2〜8 で実効バッチを稼ぐ
- `--gradient_checkpointing` を有効化（必須）
- `--quantize_via=cpu` を使用して起動時 OOM を回避
- 小さな LoRA rank（1〜8）を使用
- 環境変数 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定すると VRAM 使用を最小化できます

**注**: VAE 埋め込みやテキストエンコーダ出力の事前キャッシュはメモリを多く使います。OOM が出る場合は `offload_during_startup=true` を有効にしてください。

### LoRA の推論

Qwen Image は新しいモデルのため、以下に動作する推論例を示します:

<details>
<summary>Python 推論例を表示</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### 注意事項とトラブルシューティングのヒント

#### バッチサイズの制限

以前の diffusers の Qwen 実装では、テキスト埋め込みのパディングと attention mask 処理の問題でバッチサイズ > 1 が壊れていました。現在の SimpleTuner の Qwen override はこの 2 点を修正するため、VRAM が足りればより大きいバッチも動作します。
- `train_batch_size` はメモリ余裕を確認してから増やしてください。
- 古い環境でまだアーティファクトが出る場合は、更新して古い text embed を再生成してください。

#### 量子化

- `int2-quanto` は最も強い省メモリだが品質に影響する可能性
- `nf4-bnb` はメモリと品質のバランスが良い
- `int4-quanto` は中間的
- 40GB+ の VRAM がある場合を除き `int8` は避ける

#### 学習率

LoRA トレーニングの場合:
- 小さな LoRA（rank 1〜8）: 1e-4 前後
- 大きな LoRA（rank 16〜32）: 5e-5 前後
- Prodigy オプティマイザ: 1.0 から開始し自動適応

#### 画像アーティファクト

アーティファクトが出る場合:
- 学習率を下げる
- 勾配蓄積を増やす
- 画像品質と前処理を確認
- 初期は低解像度から開始する

#### 複数解像度トレーニング

まず 512px または 768px で学習し、その後 1024px で微調整します。異なる解像度で学習する場合は `--flow_schedule_auto_shift` を有効にしてください。

### プラットフォームの制限

**未対応:**
- AMD ROCm（効率的な Flash Attention がない）
- Apple Silicon/MacOS（メモリと注意機構の制限）
- 24GB 未満のコンシューマ GPU

### 既知の問題

1. バッチサイズ > 1 は正しく動作しない（勾配蓄積を使用）
2. TREAD は未対応
3. テキストエンコーダのメモリ消費が大きい（量子化前 ~16GB）
4. シーケンス長処理の問題（[上流 issue](https://github.com/huggingface/diffusers/issues/12075)）

追加のヘルプとトラブルシューティングは [SimpleTuner documentation](/documentation) を参照するか、コミュニティ Discord に参加してください。
