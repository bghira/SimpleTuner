# DeepFloyd IF

> DeepFloyd の学習には、LoRA でも少なくとも 24G の VRAM が必要です。このガイドは 4 億パラメータのベースモデルに焦点を当てていますが、4.3B の XL 版も同じ指針で学習できます。

## 背景

2023 年春、StabilityAI は DeepFloyd と呼ばれるカスケード型のピクセル拡散モデルを公開しました。
![](https://tripleback.net/public/deepfloyd.png)

Stable Diffusion XL と簡単に比較すると:
- テキストエンコーダ
  - SDXL は 2 つの CLIP エンコーダ（"OpenCLIP G/14" と "OpenAI CLIP-L/14"）を使用
  - DeepFloyd は Google の T5 XXL という自己教師あり Transformer を 1 つ使用
- パラメータ数
  - DeepFloyd は 400M、900M、4.3B の複数構成があり、より大きいほど学習コストが高くなります。
  - SDXL は約 3B の 1 構成のみです。
  - DeepFloyd のテキストエンコーダは単体で 11B パラメータがあり、最大構成は約 15.3B パラメータになります。
- モデル数
  - DeepFloyd は **3** 段階: 64px -> 256px -> 1024px
    - 各段階でデノイジング目標を完了します
  - SDXL はリファイナを含む **2** 段階で 1024px -> 1024px
    - 各段階でデノイジング目標を部分的に完了します
- 設計
  - DeepFloyd の 3 モデルは解像度と細部を段階的に強化します
  - SDXL の 2 モデルは細部と構図を調整します

両モデルとも、最初の段階が画像の構図（大きな要素や影の位置）をほとんど決めます。

## モデル評価

DeepFloyd を学習または推論に使うときの期待値は以下のとおりです。

### 美的傾向

SDXL または Stable Diffusion 1.x/2.x と比較すると、DeepFloyd の美的傾向は Stable Diffusion 2.x と SDXL の中間に位置します。


### デメリット

このモデルが一般的ではない理由は次のとおりです:

- 推論時の VRAM 要件が他モデルより重い
- 学習時の VRAM 要件が他モデルを大きく上回る
  - フル U-Net チューニングには 48G を超える VRAM が必要
  - rank-32、batch-4 の LoRA でも約 24G VRAM が必要
  - テキスト埋め込みキャッシュが非常に大きい（SDXL の二重 CLIP 埋め込みが数百 KB に対し、数 MB）
  - テキスト埋め込みキャッシュの生成が遅い（A6000 non-Ada で現在 1 秒あたり約 9〜10 個）
- 既定の美的品質が他モデルより劣る（素の SD 1.5 を学習するような感覚）
- 推論時に調整またはロードすべきモデルが **3** つある（テキストエンコーダを含めると 4 つ）
- StabilityAI の主張が実際の使用感と一致しなかった（期待が過度だった）
- DeepFloyd-IF のライセンスは商用利用に制限がある
  - これは NovelAI の重みに影響しませんでしたが、実際には不正に流出したものです。商用ライセンスの制限は、他のより重大な問題を考えると、都合の良い理由に見える面もあります。

### メリット

一方で、DeepFloyd には見落とされがちな利点もあります:

- 推論時の T5 テキストエンコーダは世界理解が強い
- 非常に長いキャプションでの学習が可能
- 第 1 段階は約 64x64 のピクセル面積で、複数アスペクト解像度で学習できる
  - 低解像度のため、LAION-A のほぼすべて（64x64 未満が少ない）で学習できる _唯一のモデル_ だった
- 各段階を個別にチューニングでき、異なる目的に集中できる
  - 第 1 段階は構図、第 2/3 段階は拡大後の細部に集中して調整できる
- より大きなメモリフットプリントにもかかわらず学習が速い
  - スループットが高く、stage 1 の学習ではサンプル/時間が高い
  - CLIP 系モデルより学習が速く、CLIP に慣れた人には違和感があるかもしれません
    - つまり学習率やスケジュールの期待値を調整する必要があります
- VAE がなく、学習サンプルは目標サイズに直接ダウンサンプルされ、U-Net がピクセルを直接処理します
- ControlNet LoRA や他の多くのテクニック（典型的な線形 CLIP U-Net 向けのもの）が利用できます

## LoRA のファインチューニング

> ⚠️ DeepFloyd の最小 400M モデルでもフル U-Net の逆伝播は計算要件が高いため、未検証です。本ドキュメントでは LoRA を使用しますが、フル U-Net チューニングも動作するはずです。

これらの手順は SimpleTuner の基本的な使い方を想定しています。初めての方は、[Kwai Kolors](quickstart/KOLORS.md) のようなサポートが充実したモデルから始めることを推奨します。

それでも DeepFloyd を学習する場合は、`model_flavour` 設定で対象モデルを明示する必要があります。

### config.json

```bash
"model_family": "deepfloyd",

# Possible values:
# - i-medium-400m
# - i-large-900m
# - i-xlarge-4.3b
# - ii-medium-450m
# - ii-large-1.2b
"model_flavour": "i-medium-400m",

# DoRA isn't tested a whole lot yet. It's still new and experimental.
"use_dora": false,
# Bitfit hasn't been tested for efficacy on DeepFloyd.
# It will probably work, but no idea what the outcome is.
"use_bitfit": false,

# Highest learning rate to use.
"learning_rate": 4e-5,
# For schedules that decay or oscillate, this will be the end LR or the bottom of the valley.
"lr_end": 4e-6,
```

- `model_family` は deepfloyd
- `model_flavour` は Stage I または II を指す
- `resolution` は `64`、`resolution_type` は `pixel`
- `attention_mechanism` は `xformers` に設定できますが、AMD や Apple 環境では設定できず、より多くの VRAM が必要になります。
  - **注記** ~~Apple MPS には DeepFloyd チューニングを完全に妨げるバグがありました。~~ Pytorch 2.6 以降（またはそれ以前の時点）で、Apple MPS でも stage I と II が学習できるようになりました。

より詳細な検証のために、`validation_resolution` は次のように設定できます:

- `validation_resolution=64` は 64x64 の正方形画像を生成
- `validation_resolution=96x64` は 3:2 のワイド画像を生成
- `validation_resolution=64,96,64x96,96x64` は検証ごとに 4 枚の画像を生成:
  - 64x64
  - 96x96
  - 64x96
  - 96x64

### multidatabackend_deepfloyd.json

続いて DeepFloyd 学習用のデータローダを設定します。SDXL や旧来モデルのデータセット設定とほぼ同じですが、解像度パラメータに重点があります。

```json
[
    {
        "id": "primary-dataset",
        "type": "local",
        "instance_data_dir": "/training/data/primary-dataset",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "random",
        "resolution": 64,
        "resolution_type": "pixel",
        "minimum_image_size": 64,
        "maximum_image_size": 256,
        "target_downsample_size": 128,
        "prepend_instance_prompt": false,
        "instance_prompt": "Your Subject Trigger Phrase or Word",
        "caption_strategy": "instanceprompt",
        "repeats": 1
    },
    {
        "id": "an example backend for text embeds.",
        "dataset_type": "text_embeds",
        "default": true,
        "disable": false,
        "type": "local",
        "cache_dir": "/training/cache/deepfloyd/text/dreambooth"
    }
]
```

上記は DeepFloyd の基本的な Dreambooth 設定例です:

- `resolution` と `resolution_type` はそれぞれ `64` と `pixel`
- `minimum_image_size` は 64 ピクセルに下げ、より小さい画像の誤ったアップサンプルを避けます
- `maximum_image_size` は 256 ピクセルに設定し、大きな画像が 4:1 を超える比率でクロップされて、致命的なコンテキスト損失を招くのを防ぎます
- `target_downsample_size` は 128 ピクセルに設定し、`maximum_image_size` の 256 ピクセルを超える画像をクロップ前に 128 ピクセルへリサイズします

注記: 画像は 25% ずつダウンサンプルされるため、極端なサイズ変化でシーンの詳細が不正に平均化されるのを避けられます。

## 推論の実行

現在、DeepFloyd 専用の推論スクリプトは SimpleTuner ツールキットにはありません。

内蔵の検証プロセス以外では、推論後の実行例が掲載された [Hugging Face のドキュメント](https://huggingface.co/docs/diffusers/v0.23.1/en/training/dreambooth#if) を参照してください:

```py
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0", use_safetensors=True)
pipe.load_lora_weights("<lora weights path>")
pipe.scheduler = pipe.scheduler.__class__.from_config(pipe.scheduler.config, variance_type="fixed_small")
```

> ⚠️ `DiffusionPipeline.from_pretrained(...)` の最初の値は `IF-I-M-v1.0` に設定されていますが、学習した LoRA のベースモデルパスに合わせて変更する必要があります。

> ⚠️ Hugging Face の推奨事項すべてが SimpleTuner に当てはまるわけではありません。例えば、効率的な事前キャッシュと純粋な bf16 オプティマイザ状態により、DeepFloyd stage I の LoRA を Diffusers の Dreambooth 例（28G）より少ない 22G の VRAM で調整できます。

## 超解像 stage II モデルのファインチューニング

DeepFloyd の stage II モデルは 64x64（または 96x64）程度の入力を受け取り、`VALIDATION_RESOLUTION` 設定に応じたアップスケール画像を生成します。

評価画像はデータセットから自動的に収集され、`--num_eval_images` は各データセットから選ぶアップスケール画像の数を指定します。画像は現在ランダムに選ばれますが、各セッションで同じ画像が維持されます。

誤ったサイズ設定で実行しないように、追加のチェックも行われます。

stage II を学習するには、上記の手順に従い、`MODEL_TYPE` に `deepfloyd-lora` の代わりに `deepfloyd-stage2-lora` を指定します:

```bash
export MODEL_TYPE="deepfloyd-stage2-lora"
```
