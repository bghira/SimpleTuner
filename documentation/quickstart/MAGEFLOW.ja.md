# Mage-Flow クイックスタート

このガイドでは SimpleTuner で Mage-Flow LoRA を学習する手順を扱います。Mage-Flow は Microsoft の 4B rectified-flow 画像生成・編集ファミリーで、ネイティブ解像度 MMDiT、Qwen3-VL テキスト条件、128 チャンネルで 16x ダウンサンプルの Mage-VAE を使います。

## ハードウェア

Mage-Flow は Flux.1 や Qwen-Image より小さいものの、大きな transformer と凍結 Qwen3-VL text encoder を使います。

開始点:

- `bf16`、512px、batch 1 でスモークテスト
- `bf16`、1024px、batch 1 で通常の LoRA
- VRAM が足りない場合は `int8-torchao` または NF4
- Turbo flavours は検証 4 steps

24GB は低解像度または量子化実験向け、48GB は 1024px の実用ライン、80GB は edit training や大きめの batch に向いています。

## 設定

SimpleTuner をインストールします:

```bash
pip install 'simpletuner[cuda]'
```

Mage-Flow は packed variable-length attention を使います。ローカルで `flash-attn` パッケージをビルドせずに FlashAttention 2 を使うには、`"attention_mechanism": "flash-attn-varlen-hub"` を設定し、SimpleTuner に Hugging Face Hub kernel を読み込ませてください。PyTorch SDPA を使う場合は既定の `diffusers` のままで構いません。

text-to-image の開始設定:

```json
{
  "model_family": "mageflow",
  "model_flavour": "base",
  "model_type": "lora",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Base",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "optimizer": "optimi-lion",
  "learning_rate": 1e-4,
  "lora_rank": 32,
  "train_batch_size": 1,
  "resolution": 1024,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 30,
  "validation_guidance": 5.0
}
```

利用できる flavours:

- `base` - `microsoft/Mage-Flow-Base`
- `default` - `microsoft/Mage-Flow`
- `turbo` - `microsoft/Mage-Flow-Turbo`
- `edit-base` - `microsoft/Mage-Flow-Edit-Base`
- `edit` - `microsoft/Mage-Flow-Edit`
- `edit-turbo` - `microsoft/Mage-Flow-Edit-Turbo`

編集 LoRA:

```json
{
  "model_family": "mageflow",
  "model_flavour": "edit-turbo",
  "pretrained_model_name_or_path": "microsoft/Mage-Flow-Edit-Turbo",
  "validation_num_inference_steps": 4
}
```

## Mage Flow (Edit) Considerations

Mage-Flow edit checkpoint は conditioning/reference dataset を必須としません。Microsoft は edit model を生成タスクと編集タスクで joint training しているため、生成 prior は保持されています。SimpleTuner では `model_flavour` が `edit-base`、`edit`、`edit-turbo` の場合でも、subject、style、concept LoRA 用の通常の画像 dataset をそのまま使えます。

編集挙動を明示的に学習したい場合だけ source/target のペアデータを使ってください。SimpleTuner は edit flavour に対して編集対応 pipeline を自動的に使いますが、conditioning image がない場合は validation と prompt encoding が text-to-image path を使います。

## Dataloader

通常の subject/style LoRA では標準の画像 dataloader を使います:

```json
[
  {
    "id": "dreambooth-1024",
    "type": "local",
    "instance_data_dir": "/path/to/images",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "resolution": 1024,
    "resolution_type": "pixel",
    "metadata_backend": "discovery",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "cache_dir_vae": "cache/vae/mageflow/dreambooth-1024"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/mageflow"
  }
]
```

任意で編集挙動を学習する場合は source/target のペアデータを使います。caption は最終画像の説明ではなく、編集指示として書いてください。

## メモリプリセット

Mage-Flow はメモリ最適化メニューに RAMTorch と Musubi block swap のプリセットを用意しています。Transformer weight を CPU RAM に置きたい場合は RAMTorch、forward/backward 中に最後の Transformer block だけをストリームしたい場合は Musubi block swap を使います。これらは configurator では相互排他です。

## 検証と量子化

`default` は約 20 steps、`base` は約 30 steps、`turbo` / `edit-turbo` は 4 steps が目安です。

```json
{
  "base_model_precision": "int8-torchao",
  "quantize_via": "cpu"
}
```

SimpleTuner は MIT ライセンスの Mage-Flow コードを vendored し、検証と save hook の一貫性のため Diffusers pipeline で包んでいます。
