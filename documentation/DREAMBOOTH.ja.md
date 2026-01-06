# Dreambooth（単一被写体の学習）

## 背景

Dreambooth は、少量の高品質画像で被写体をモデルに注入するために Google が開発した手法を指します（[paper](https://dreambooth.github.io)）。

ファインチューニングの文脈では、Dreambooth は過学習やアーティファクトによるモデル崩壊を防ぐための新しいテクニックを加えます。

### 正則化画像

正則化画像は通常、学習中のモデルが「クラス」を表すトークンを使って生成した画像です。

モデルが生成した合成画像である必要は**必ずしも**ありませんが、実データ（例: 実在人物の写真）を使うより性能が良い場合があります。

例: 男性被写体を学習する場合、正則化データはランダムな男性の写真または合成サンプルになります。

> 🟢 正則化画像は別データセットとして設定でき、学習データと均等に混ぜられます。

### レアトークン学習

元論文には、モデルのトークナイザ語彙を逆引きして学習関連が少ない「レア」な文字列を探すという概念がありました。

その後、この考えは発展と議論が進み、十分に似ている有名人名を使う派も登場しました。こちらのほうが計算負荷が少ないためです。

> 🟡 レアトークン学習は SimpleTuner でサポートされていますが、レアトークンを探すツールはありません。

### 事前保持損失

モデルには理論上「prior（事前）」があり、Dreambooth 学習中に保持できるとされています。しかし Stable Diffusion の実験では効果が見られず、モデルは自身の知識に過学習してしまいました。

> 🟢 ([#1031](https://github.com/bghira/SimpleTuner/issues/1031)) 事前保持損失は、LyCORIS アダプタ学習時に対象データセットへ `is_regularisation_data` を設定することで SimpleTuner で利用できます。

### マスク付き損失

画像マスクを画像データとペアで定義できます。マスクの暗い部分は損失計算から無視されます。

`input_dir` と `output_dir` を指定してマスクを生成する [スクリプト](/scripts/toolkit/datasets/masked_loss/generate_dataset_masks.py) が用意されています:

```bash
python generate_dataset_masks.py --input_dir /images/input \
                      --output_dir /images/output \
                      --text_input "person"
```

ただし、マスクのパディングやぼかしなどの高度な機能はありません。

マスク用データセットを定義する際は:

- すべての画像にマスクが必要です。マスクを使わない場合は真っ白の画像を使います。
- マスクデータフォルダに `dataset_type=conditioning` を設定
- マスクデータセットに `conditioning_type=mask` を設定
- 画像データセットに `conditioning_data=` としてマスクデータセットの `id` を設定

```json
[
    {
        "id": "dreambooth-data",
        "type": "local",
        "dataset_type": "image",
        "conditioning_data": "dreambooth-conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth",
        "cache_dir_vae": "/training/cache/vae/sdxl/dreambooth-data",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "an dreambooth",
        "metadata_backend": "discovery",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area"
    },
    {
        "id": "dreambooth-conditioning",
        "type": "local",
        "dataset_type": "conditioning",
        "instance_data_dir": "/training/datasets/test_datasets/dreambooth_mask",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution_type": "pixel_area",
        "conditioning_type": "mask"
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "/training/cache/text/sdxl-base/masked_loss"
    }
]
```

## セットアップ

Dreambooth 固有の設定に進む前に、[チュートリアル](TUTORIAL.md) に従う必要があります。

DeepFloyd のチューニングについては、モデル固有の注意点があるため [このページ](DEEPFLOYD.md) を参照してください。

### 量子化モデルの学習（LoRA/LyCORIS のみ）

Apple および NVIDIA 環境で検証済みで、Hugging Face Optimum-Quanto を使うと精度と VRAM 要件を下げられます。

SimpleTuner の venv 内で実行します:

```bash
pip install optimum-quanto
```

利用可能な精度レベルはハードウェアとその能力に依存します。

- int2-quanto, int4-quanto, **int8-quanto**（推奨）
- fp8-quanto, fp8-torchao（CUDA >= 8.9 のみ、例: 4090 または H100）
- nf4-bnb（低 VRAM ユーザー向けに必要）

config.json では、次の値を変更または追加します:
```json
{
    "base_model_precision": "int8-quanto",
    "text_encoder_1_precision": "no_change",
    "text_encoder_2_precision": "no_change",
    "text_encoder_3_precision": "no_change"
}
```

データローダ設定 `multidatabackend-dreambooth.json` は次のようになります:

```json
[
    {
        "id": "subjectname-data-512px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname",
        "repeats": 100,
        "crop": false,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192
    },
    {
        "id": "subjectname-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/subjectname",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "subjectname",
        "cache_dir_vae": "/training/vae_cache/subjectname-1024px",
        "repeats": 100,
        "crop": false,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768
    },
    {
        "id": "regularisation-data",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation",
        "repeats": 0,
        "resolution": 512,
        "resolution_type": "pixel_area",
        "minimum_image_size": 192,
        "is_regularisation_data": true
    },
    {
        "id": "regularisation-data-1024px",
        "type": "local",
        "instance_data_dir": "/training/datasets/regularisation",
        "caption_strategy": "instanceprompt",
        "instance_prompt": "a picture of a man",
        "cache_dir_vae": "/training/vae_cache/regularisation-1024px",
        "repeats": 0,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "minimum_image_size": 768,
        "is_regularisation_data": true
    },
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_base"
    }
]
```

単一被写体の学習を容易にするために、いくつかの値を調整しています:

- 2 つのデータセットを 2 回ずつ設定し、合計 4 つのデータセットにしています。正則化データは任意で、ないほうがうまくいく場合もあります。必要に応じてリストから削除してください。
- 解像度は 512px と 1024px の混合バケットを使い、学習速度と収束の改善に役立ちます。
- 最小画像サイズを 192px または 768px に設定し、低解像度ながら重要な画像を含むデータセットでは、ある程度のアップスケールを許容しています。
- `caption_strategy` を `instanceprompt` にしたため、すべての画像のキャプションは `instance_prompt` になります。
  - **注記:** インスタンスプロンプトは Dreambooth の伝統的な方法ですが、短いキャプションのほうが良い場合もあります。モデルが汎化しない場合はキャプションの使用を検討してください。

### 正則化データセットの考慮事項

正則化データセットでは:

- Dreambooth 被写体の `repeats` を高く設定し、Dreambooth データの枚数（`repeats` 倍）が正則化データより多くなるようにします
  - 正則化セットが 1000 枚で、学習セットが 10 枚なら、`repeats` は少なくとも 100 が望ましいです
- `minimum_image_size` を上げ、低品質アーティファクトを増やしすぎないようにします
- より説明的なキャプションを使うと忘却を防げることがあります。`instanceprompt` から `textfile` などに切り替える場合は、各画像に `.txt` ファイルが必要です。
- `is_regularisation_data`（米国綴りは `is_regularization_data`）を設定すると、そのデータはベースモデルに入力され、学生の LyCORIS モデルの損失ターゲットになる予測を得ます。
  - なお、現在これは LyCORIS アダプタでのみ機能します。

## インスタンスプロンプトの選択

前述のとおり、Dreambooth の元々の焦点はレアトークンの選定でした。

代替として、被写体の実名や「似ている」有名人名を使う方法もあります。

いくつかの学習実験では、被写体の実名で生成した結果が似ていない場合、「似ている」有名人名のほうが良い結果になる傾向があります。

# スケジュールドサンプリング（ロールアウト）

Dreambooth のような小規模データセットで学習すると、モデルは学習中に加える「理想的な」ノイズに素早く過適合します。これにより **露出バイアス** が生じ、完全な入力には強い一方、推論時に自分の不完全な出力に直面すると失敗します。

**スケジュールドサンプリング（ロールアウト）** は、学習ループ中にモデル自身のノイズ付き潜在を数ステップ生成させます。純粋なガウスノイズ + 信号だけで学習する代わりに、モデル自身の誤りを含む「ロールアウト」サンプルで学習し、自己修正を学ばせることで被写体生成を安定させます。

> 🟢 この機能は実験的ですが、小規模データセットで過学習や「frying」が起きやすい場合に強く推奨されます。
> ⚠️ ロールアウトを有効にすると、学習ループ中に追加の推論ステップが必要となり、計算負荷が増えます。

有効化するには、`config.json` に次を追加します:

```json
{
  "scheduled_sampling_max_step_offset": 10,
  "scheduled_sampling_probability": 1.0,
  "scheduled_sampling_ramp_steps": 1000,
  "scheduled_sampling_sampler": "unipc"
}
```

*   `scheduled_sampling_max_step_offset`: 生成ステップ数。小さな値（5〜10）で十分なことが多いです。
*   `scheduled_sampling_probability`: 適用頻度（0.0〜1.0）。
*   `scheduled_sampling_ramp_steps`: 学習初期の不安定化を避けるため、最初の N ステップで確率を徐々に上げます。

# 指数移動平均（EMA）

チェックポイントと並行して 2 つ目のモデルをほぼ無料で学習できます。追加で必要なのは（既定では）システムメモリで、VRAM は増えません。

`config` に `use_ema=true` を設定すると、この機能が有効になります。

# CLIP スコアの追跡

評価でモデル性能をスコア化したい場合は、CLIP スコアの設定と解釈について [このドキュメント](evaluation/CLIP_SCORES.md) を参照してください。

# 安定評価損失

安定 MSE 損失でモデル性能を評価したい場合は、評価損失の設定と解釈について [このドキュメント](evaluation/EVAL_LOSS.md) を参照してください。

# 検証プレビュー

SimpleTuner は Tiny AutoEncoder モデルを使い、生成中の検証プレビューをストリーミングできます。この機能により、完全な生成が終わるのを待たずに、Webhook コールバックでリアルタイムに検証画像の生成過程を確認できます。

## 検証プレビューの有効化

`config.json` に次を追加します:

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

## 要件

- Tiny AutoEncoder をサポートするモデルファミリー（Flux、SDXL、SD3 など）
- プレビュー画像を受け取る Webhook 設定
- 検証が有効であること（`validation_disable` を true にしない）

## 設定オプション

- `--validation_preview`: プレビュー機能の有効/無効（既定: false）
- `--validation_preview_steps`: サンプリング中にプレビューをデコードする頻度（既定: 1）
  - 1 に設定すると各ステップでプレビューを受け取ります
  - 3 や 5 など高い値にすると Tiny AutoEncoder デコードのオーバーヘッドを減らせます

## 例

`validation_num_inference_steps=20` と `validation_preview_steps=5` の場合、検証生成の各ステップ 5、10、15、20 でプレビュー画像を受け取ります。

# Refiner のチューニング

SDXL の refiner が好みの場合、Dreambooth で生成した結果を「台無し」にすることがあります。

SimpleTuner は SDXL refiner の LoRA およびフルランク学習をサポートしています。

これにはいくつかの考慮事項があります:
- 画像は純粋に高品質であること
- テキスト埋め込みはベースモデルと共有できないこと
- VAE 埋め込みはベースモデルと共有**可能**

`multidatabackend.json` の `cache_dir` を更新してください:

```json
[
    {
        "id": "textembeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/training/text_cache/sdxl_refiner"
    }
]
```

特定の美的スコアを狙う場合は、`config/config.json` に次を追加します:

```bash
"--data_aesthetic_score": 5.6,
```

**5.6** を目標スコアに置き換えてください。既定値は **7.0** です。

> ⚠️ SDXL refiner の学習では検証プロンプトは無視されます。代わりにデータセットからランダムな画像がリファインされます。
