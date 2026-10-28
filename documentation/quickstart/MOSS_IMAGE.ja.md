# MOSS-Image クイックスタート

自分の画像で MOSS-Image の LoRA を学習しましょう。LoRA は、元のチェックポイントを保ちながら被写体や画風を学習する小さなアダプターです。

`model_family: "moss_image"`、`model_flavour: "sft"`、`model_type: "lora"` を選び、選別されたデータで調整された既定の [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) から始めます。[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) を使う場合は `model_flavour: "base"` にします。両モデルは 5B の画像 Transformer、Qwen3.5-9B テキストエンコーダー、FLUX.2 VAE を共有し、[Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE) で公開されています。学習中はテキストエンコーダーを固定します。

## VRAM プリセットの選択

GPU メモリーに合ったレシピを選びます。全レベルで 512×512 相当の面積、縦横比の維持、rank 32 BF16 LoRA、REPA を使用します。

| GPU メモリー | Base の例 | SFT の例 | バッチ内画像数 | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image/base/24g/peft-lora` | `moss_image/sft/24g/peft-lora` | 1 | 1 |
| 32 GB | `moss_image/base/32g/peft-lora` | `moss_image/sft/32g/peft-lora` | 1 | 2 |
| 48 GB | `moss_image/base/48g/peft-lora` | `moss_image/sft/48g/peft-lora` | 2 | 2 |
| 80 GB | `moss_image/base/80g/peft-lora` | `moss_image/sft/80g/peft-lora` | 4 | 2 |
| 144 GB | `moss_image/base/144g/peft-lora` | `moss_image/sft/144g/peft-lora` | 8 | 2 |

既定の `moss_image/sft/default/default.peft-lora` と `moss_image/sft/default/peft-lora` は 48 GB 設定です。バッチサイズは開始時の目安です。長いキャプション、追加機能、または縦横比バケットの画像数がバッチサイズより少ない場合は `train_batch_size` を下げてください。144 GB レベルは H200 クラスの GPU 向けです。

## SimpleTuner のインストール

以下の CUDA レシピには Python 3.13 と NVIDIA GPU を使います。他の環境や CUDA バージョンは[インストールガイド](../INSTALL.md)を参照してください。

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
```

## 画像とキャプションの準備

`datasets/moss-image` を作成し、画像と同名の `.txt` キャプションを入れます。512px での学習に十分な細部を持つ鮮明な画像を用意してください。正方形なら 512×512 に相当し、縦長・横長画像は縦横比を維持します。

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

`TOK` などの被写体用の単語を決め、その被写体の各キャプションに含めます。被写体、構図、照明、背景、画風を説明してください。プロジェクトは英語での開始を推奨しています。画像内の文字と改行も正確に記述します。例: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## 学習設定

48 GB の SFT レシピとサンプルプロンプトをコピーします。`48g` は選んだレベルに置き換えてください。Base の場合はコマンド中の `moss_image/sft/48g/peft-lora` を `moss_image/base/48g/peft-lora` に変更します。

```bash
mkdir -p config/examples/_shared datasets/moss-image
mkdir -p config/examples/moss_image/sft/48g
cp -R simpletuner/examples/moss_image/sft/48g/peft-lora config/examples/moss_image/sft/48g/
mkdir -p config/examples/_shared
cp simpletuner/examples/_shared/moss_image-prompts.json config/examples/_shared/
cp config/examples/moss_image/sft/48g/peft-lora/config.json config/config.json
```

レシピは rank 32 の BF16 LoRA、学習率 `1e-4`、gradient checkpointing、REPA、4,000 オプティマイザーステップ、512 相当の解像度を使用します。`config/config.json` を開き、既存設定内の次の項目を編集します。

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

コピーした `dataloader.json` は `datasets/moss-image` のテキストファイルを読み、潜在表現とテキスト埋め込みをキャッシュします。`resolution_type: "area"` と `resolution: 0.262144` は 512×512 相当の面積です。寸法は 16 の倍数に揃えます。データセット追加や正則化画像は[データローダーガイド](../DATALOADER.md)を参照してください。

## 学習とプレビューの比較

仮想環境を有効にし、SimpleTuner ディレクトリーで `simpletuner train` を実行します。初回はモデルと REPA エンコーダーをダウンロードし、キャッシュを作成します。250 ステップごとにチェックポイントと検証画像を保存します。被写体の再現、構図、細部を比べ、データに応じて `max_train_steps` を調整してください。

```bash
simpletuner train
```

## サンプリング設定

まず **50 ステップ**、**CFG 3**、**1024×1024** で検証します。サンプリング用 `moss_image_validation_shift` の既定値は **5** で、学習には元のシフトなしのグリッドを使います。`moss_image_uncond_skip_layers` は既定で **[22]** となり、負の CFG 分岐でレイヤー 22 をスキップします。通常の CFG には `[]` を指定します。正の分岐での追加 SLG は `validation_guidance_skip_layers`、`validation_guidance_skip_scale` と開始・終了設定で制御します。[OPTIONS](../OPTIONS.md)を参照してください。Python では `use_cfg_zero_star=True` で CFG-Zero* を使えます。

学習では 1% の一様分布を混合した MOSS-Image 本来のノイズ分布を自動使用します。検証はサンプリングシフトを一度だけ適用し、最後のゼロノイズ端点を含みます。AnyFlow では検証と学習ロールアウトに同じシフトを指定してください。

## サンプリング中のプレビュー

WebUI でノイズ除去中の進捗画像を見るには、次の設定を追加します。プレビューは TAEF2、最終検証画像は完全な VAE を使います。CLI では設定した webhook にプレビューを配信できます。

```json
{
  "validation_preview": true,
  "validation_preview_steps": 5
}
```

## メモリーと学習オプション

レシピの `train_batch_size` は選択した VRAM レベルに合わせてあります。小規模データセットや、長いキャプション・追加機能でメモリーが増える場合は下げてください。 `gradient_checkpointing_backend` は `torch`、`unsloth`、`torch-ffn`、`unsloth-ffn` を選べます。適用範囲は interval と `gradient_checkpointing_segment_stride` で調整します。Musubi は `musubi_blocks_to_swap` でブロックを GPU と CPU 間で移動します。RamTorch は `ramtorch: true` と `ramtorch_transformer_percent` で選択的にオフロードします。システム RAM と転送時間を使うため、一度に一つの方式から設定してください。[OPTIONS](../OPTIONS.md)に詳細があります。

CUDA のインストールには CuTE attention が含まれます。学習時の既定値 `attention_mechanism: "kohaku-fa-auto"` は Ada・Hopper・RTX Blackwell GPU で CuTE、それ以外のアーキテクチャで標準の SDPA を選びます。逆伝播用に保存する attention テンソルをオフロードするには `gradient_checkpointing_offload_attention: true`、転送の先読みには `gradient_checkpointing_offload_prefetch: true` を指定します。他の attention バックエンドは [OPTIONS](../OPTIONS.md) を参照してください。

レシピは [REPA](../experimental/IMAGE_REPA.md) を有効にします。後段の画像特徴を前段と揃えるには、[LayerSync](../experimental/LAYERSYNC.md) を追加します。

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 20,
  "layersync_teacher_block": 8,
  "layersync_lambda": 0.2
}
```

ポストトレーニングは [AnyFlow](../quickstart/ANYFLOW.md) または [TwinFlow](../distillation/TWINFLOW.md) を参照してください。AnyFlow では `lora_dropout: 0.0` を維持します。Python で AnyFlow LoRA を読む際は `load_lora_weights` に学習設定を渡します。例: `flowmap_deltatime_type="r", flowmap_gate_value=0.25`。

## プロンプトの改善とレイアウト

詳細なキャプション、タグ列、JSON レイアウトを使えます。プロジェクトの[プロンプト改善スキル](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)はエージェントによる作成を助けます。完成したテキストをキャプションや生成プロンプトに使ってください。[公式ガイド](https://github.com/OpenMOSS/MOSS-Image/blob/main/USERGUIDE.md)にキャプションの推奨事項があります。

レイアウトは場面、文字、物体の位置を記述します。座標はキャンバス中心を原点とするピクセル単位の `[y_min, x_min, y_max, x_max]` です。1024×1024 では −512 から +512 になります。`json.dumps(layout, ensure_ascii=False)` で文字列化してプロンプトに渡し、レイアウトと同じ生成寸法を使います。

前処理でキャプションをネイティブ JSON に拡張するには、学習設定に次を追加します。

```json
{
  "moss_image_auto_json": true,
  "moss_image_prompt_upsample": true
}
```

既定のキャプションモデルは、プロジェクト推奨の VLM の一つである `Qwen/Qwen3.5-27B` です。80 GB GPU と追加のシステム RAM を用意してください。各学習画像とバケット寸法を使って説明と中心原点のピクセル座標ボックスを生成し、テキスト埋め込みをキャッシュします。検証プロンプトも拡張し、空とネガティブのプロンプトは変更しません。別モデルを読み込まず既存キャプションを JSON に包むには、`moss_image_auto_json` だけを有効にします。

```json
{
  "summary": {"caption": "A cream travel poster with the title EXPLORE above a blue mountain."},
  "items": [
    {"label": "text", "text": "EXPLORE", "caption": "Large navy title", "bbox": [-416, -384, -288, 384]},
    {"label": "object", "caption": "Blue mountain", "bbox": [-192, -288, 384, 288]}
  ],
  "bbox_coord_space": "abs_centralized",
  "bbox_format": "yxyx"
}
```

## LoRA を使った画像生成

出力ディレクトリーの `pytorch_lora_weights.safetensors` が最終アダプターです。学習時と同じ Base または SFT チェックポイントで読み込み、被写体用の単語と希望する場面をプロンプトに含めます。

```python
import torch
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
)
pipe.load_lora_weights("output/my-moss-lora")
pipe.to("cuda")
image = pipe(
    "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
    height=1024, width=1024, num_inference_steps=50, guidance_scale=3.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("my-moss-image.png")
```

## 編集・参照画像LoRAを学習する

参照画像付きで学習するには `model_flavour: "sft-edit"` または `"base-edit"` を選びます。対応するSFTまたはBaseチェックポイントから開始し、アダプターで参照画像による条件付けを学習します。`simpletuner/examples/moss_image/sft-edit/default/peft-lora` または `moss_image/base-edit/default/peft-lora` を使用してください。いずれもバッチあたりターゲット1枚、512相当の学習、REPA、データセットによる検証を設定しています。

```bash
mkdir -p config/examples/_shared datasets/moss-edits datasets/moss-references
mkdir -p config/examples/moss_image/sft-edit/default
cp -R simpletuner/examples/moss_image/sft-edit/default/peft-lora config/examples/moss_image/sft-edit/default/
cp config/examples/moss_image/sft-edit/default/peft-lora/config.json config/config.json
```

目的の出力を `datasets/moss-edits/sample.png`、編集指示を `datasets/moss-edits/sample.txt`、入力参照画像を `datasets/moss-references/sample.png` に配置します。両フォルダーの画像ファイル名は拡張子も含めて一致させます。指示の例は `Change the blue mug to red while keeping the background unchanged.` です。編集用の例には次のデータローダー設定が含まれます。

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/moss-edits",
    "caption_strategy": "textfile",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "conditioning_data": [
      "references"
    ],
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/targets",
    "text_embeds": "text",
    "repeats": 10
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/moss-references",
    "conditioning_type": "reference_loose",
    "resolution": 0.262144,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/moss-image/sft-edit/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/moss-image/sft-edit"
  }
]
```

`reference_loose` は各参照画像のアスペクト比を保持し、参照画像とターゲットのサイズが異なっていても使用できます。位置が対応する画像ペアには `reference_strict` を使います。複数の参照画像を同時に渡すには、`conditioning_data` に参照データセットを追加し、`conditioning_multidataset_sampling: "combined"` を維持します。`"random"` はサンプルごとに参照データセットを1つ選びます。指示が異なる場合は各参照データセットに対応する指示キャプションを用意してください。ターゲット、参照画像、テキストのキャッシュは分けます。参照画像は追加メモリを必要とするため、必要に応じてバッチサイズを減らすか、上記のオフロード機能を使ってください。

検証にはペアデータセットの参照画像と指示を使用します。参照ピクセルは固定されたQwen3.5の視覚タワーを通り、ノイズのないVAE潜在表現が画像の詳細を補います。学習損失はターゲット画像に対して計算します。キャプションドロップアウトは参照画像を保持し、指示だけを除去します。Qwen画像プロセッサーは `Qwen/Qwen3.5-9B` から自動でダウンロードされます。

学習後は編集アダプターを読み込み、参照画像1枚または参照画像のリストを `image` に渡します。プロンプトのバッチでは、各プロンプトに対応する参照画像のリストを入れた入れ子リストを使います。サンプリングでは参照K/Vキャッシュが有効です。毎ステップ参照トークンを再計算する場合は `cache_reference_kv=False` を指定します。

```python
import torch
from diffusers.utils import load_image
from simpletuner.helpers.models.moss_image.pipeline import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained(
    "OpenMOSS-Team/MOSS-Image-SFT", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/examples/moss_image-sft-edit.peft-lora")
image = pipe(
    "Change the blue mug to red while keeping the background unchanged.",
    image=[load_image("reference.png")],
    height=512, width=512, num_inference_steps=50, guidance_scale=3.0,
).images[0]
image.save("edited.png")
```
