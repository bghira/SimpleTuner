# MOSS-Image クイックスタート

自分の画像で MOSS-Image の LoRA を学習しましょう。LoRA は、元のチェックポイントを保ちながら被写体や画風を学習する小さなアダプターです。

[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) を使うには `model_family: "moss_image"`、`model_flavour: "base"`、`model_type: "lora"` を選びます。選別されたデータで調整された [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) を使う場合は `model_flavour: "sft"` にします。両モデルは 5B の画像 Transformer、Qwen3.5-9B テキストエンコーダー、FLUX.2 VAE を共有し、[Apache-2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/LICENSE) で公開されています。学習中はテキストエンコーダーを固定します。

## VRAM プリセットの選択

GPU メモリーに合ったレシピを選びます。全レベルで 1024×1024 相当の面積、縦横比の維持、rank 32 BF16 LoRA、REPA を使用します。

| GPU メモリー | Base の例 | SFT の例 | バッチ内画像数 | `gradient_checkpointing_interval` |
| --- | --- | --- | ---: | ---: |
| 24 GB | `moss_image-24g.peft-lora` | `moss_image-sft-24g.peft-lora` | 1 | 1 |
| 32 GB | `moss_image-32g.peft-lora` | `moss_image-sft-32g.peft-lora` | 1 | 2 |
| 48 GB | `moss_image-48g.peft-lora` | `moss_image-sft-48g.peft-lora` | 2 | 2 |
| 80 GB | `moss_image-80g.peft-lora` | `moss_image-sft-80g.peft-lora` | 4 | 2 |
| 144 GB | `moss_image-144g.peft-lora` | `moss_image-sft-144g.peft-lora` | 8 | 2 |

既定の `moss_image.peft-lora` と `moss_image-sft.peft-lora` は 48 GB 設定です。バッチサイズは開始時の目安です。長いキャプション、追加機能、または縦横比バケットの画像数がバッチサイズより少ない場合は `train_batch_size` を下げてください。144 GB レベルは H200 クラスの GPU 向けです。

## SimpleTuner のインストール

以下の CUDA レシピには Python 3.13 と NVIDIA GPU を使います。他の環境や CUDA バージョンは[インストールガイド](../INSTALL.md)を参照してください。

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda,attention-cute]'
```

## 画像とキャプションの準備

`datasets/moss-image` を作成し、画像と同名の `.txt` キャプションを入れます。約 1 メガピクセルでの学習に十分な細部を持つ鮮明な画像を用意してください。正方形なら 1024×1024 に相当し、縦長・横長画像は縦横比を維持します。

```text
datasets/moss-image/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

`TOK` などの被写体用の単語を決め、その被写体の各キャプションに含めます。被写体、構図、照明、背景、画風を説明してください。プロジェクトは英語での開始を推奨しています。画像内の文字と改行も正確に記述します。例: `A photograph of TOK, a small blue ceramic owl on a wooden table beside a window.`

## 学習設定

48 GB の Base レシピとサンプルプロンプトをコピーします。`48g` は選んだレベルに置き換えてください。SFT の場合はコマンド中の `moss_image-48g.peft-lora` を `moss_image-sft-48g.peft-lora` に変更します。

```bash
mkdir -p config/examples datasets/moss-image
cp -R simpletuner/examples/moss_image-48g.peft-lora config/examples/
cp simpletuner/examples/moss_image-prompts.json config/examples/
cp config/examples/moss_image-48g.peft-lora/config.json config/config.json
```

レシピは rank 32 の BF16 LoRA、学習率 `1e-4`、gradient checkpointing、REPA、4,000 オプティマイザーステップ、1024 相当の解像度を使用します。`config/config.json` を開き、既存設定内の次の項目を編集します。

```json
{
  "output_dir": "output/my-moss-lora",
  "validation_prompt": "A studio photograph of TOK on a wooden table, soft window light, a pale gray background.",
  "user_prompt_library": null
}
```

コピーした `dataloader.json` は `datasets/moss-image` のテキストファイルを読み、潜在表現とテキスト埋め込みをキャッシュします。`resolution_type: "area"` と `resolution: 1.048576` は 1024×1024 相当の面積です。寸法は 16 の倍数に揃えます。データセット追加や正則化画像は[データローダーガイド](../DATALOADER.md)を参照してください。

## 学習とプレビューの比較

仮想環境を有効にし、SimpleTuner ディレクトリーで `simpletuner train` を実行します。初回はモデルと REPA エンコーダーをダウンロードし、キャッシュを作成します。250 ステップごとにチェックポイントと検証画像を保存します。被写体の再現、構図、細部を比べ、データに応じて `max_train_steps` を調整してください。

```bash
simpletuner train
```

## サンプリング設定

まず **50 ステップ**、**CFG 3**、**1024×1024** で検証します。サンプリング用 `moss_image_validation_shift` の既定値は **5** で、学習には元のシフトなしのグリッドを使います。`moss_image_uncond_skip_layers` は既定で **[22]** となり、負の CFG 分岐でレイヤー 22 をスキップします。通常の CFG には `[]` を指定します。正の分岐での追加 SLG は `validation_guidance_skip_layers`、`validation_guidance_skip_scale` と開始・終了設定で制御します。[OPTIONS](../OPTIONS.md)を参照してください。Python では `use_cfg_zero_star=True` で CFG-Zero* を使えます。

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

CUDA のインストールには CuTE attention が含まれます。学習時の既定値 `attention_mechanism: "kohaku-fa-auto"` は Ada・Hopper GPU で CuTE、それ以外のアーキテクチャで標準の SDPA を選びます。逆伝播用に保存する attention テンソルをオフロードするには `gradient_checkpointing_offload_attention: true`、転送の先読みには `gradient_checkpointing_offload_prefetch: true` を指定します。他の attention バックエンドは [OPTIONS](../OPTIONS.md) を参照してください。

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
    "OpenMOSS-Team/MOSS-Image-Base", torch_dtype=torch.bfloat16
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
