# Ideogram 4 クイックスタート

このガイドでは、SimpleTuner で Ideogram 4 LoRA を学習する方法を説明します。Ideogram 4 は約9Bパラメータの flow-matching 画像モデルで、文字表現、レイアウト、複雑なプロンプト追従に強いモデルです。公開チェックポイントは FP8 重みで配布されており、SimpleTuner でも FP8 をデフォルトとして使います。

スターター設定:

```bash
simpletuner/examples/ideogram-fp8.peft-lora/config.json
```

## ハードウェアと精度

推奨開始点:

- **標準:** FP8 ベース重み、bf16 の LoRA 学習重み、rank 16-32。
- **低VRAM:** ベースモデルに NF4 を使う。
- **高VRAM:** 十分なVRAMがある場合は bf16-upcast 重みで量子化ロードを避ける。

H100 80GB での実測値です。native FP8（`base_model_precision=fp8-torchao`、`quantize_via=pipeline`）、rank 32 LoRA、bf16 mixed precision、gradient checkpointing 有効、1024px square、validation 無効のトレーニングピーク:

| Batch size | Peak VRAM |
| --- | ---: |
| 1 | 15,999 MiB / 15.6 GiB |
| 2 | 20,095 MiB / 19.6 GiB |
| 4 | 28,603 MiB / 27.9 GiB |

Validation には別の生成ピークがあるため、`ideogram_validation=true` を使う場合は余裕を見てください。小さいGPUでは FP8 または NF4、rank 8-16、gradient checkpointing、offload から始めてください。Apple GPU は Ideogram 4 学習には推奨しません。

### Torch compile

`torch.compile` では、native FP8 重みに regional compilation を使う設定を推奨します:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

通常の `dynamo_backend="inductor"` も動作しますが、モデル全体の初回 step コンパイルは遅くなります。現時点では Ideogram 4 LoRA で `dynamo_mode="reduce-overhead"` や `dynamo_fullgraph=true` は避けてください。PEFT LoRA レイヤーが 2 回目の compiled invocation で CUDA graph output reuse を踏むことがあります。

## 設定

設定と dataloader をコピーします:

```bash
mkdir -p config/examples
cp simpletuner/examples/ideogram-fp8.peft-lora/config.json config/config.json
cp simpletuner/examples/multidatabackend-ideogram-dreambooth-1024px.json config/examples/multidatabackend-ideogram-dreambooth-1024px.json
```

重要な項目:

```json
{
  "model_type": "lora",
  "model_family": "ideogram",
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu",
  "mixed_precision": "bf16",
  "train_batch_size": 1,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "gradient_checkpointing": true,
  "ideogram_auto_json": true,
  "ideogram_validation": true,
  "ideogram_schedule_mu": 0.0,
  "ideogram_schedule_std": 1.5
}
```

FP8 が最初の推奨です:

```json
{
  "model_flavour": "fp8",
  "base_model_precision": "no_change",
  "quantize_via": "cpu"
}
```

低VRAMでは NF4 を使えます:

```json
{
  "base_model_precision": "nf4-bnb",
  "base_model_default_dtype": "bf16",
  "quantize_via": "cpu"
}
```

## Text Embed Cache

Ideogram 4 の text encoder 出力は、13 層の Qwen hidden-state を連結したものです。デフォルトでは、SimpleTuner は text embed cache ファイルへ書き込む前に、transformer の凍結された `llm_cond_norm` と `llm_cond_proj` で raw features を投影します。これにより cache file は大幅に小さくなり、transformer が実際に消費する conditioning tensor を保持できます。

この projection layer は LoRA と full transformer training の両方で凍結されます。text encoder training、標準以外の LoRA、または `llm_cond_norm` / `llm_cond_proj` を明示的に含む LoRA target では、SimpleTuner は raw text encoder output を cache に保持します。

cache の大きなコストは保存された padding ではなく feature 幅です。text embed precompute は prompt ごとの実際の token 長で 1 ファイルを書きます。batch padding は後でメモリ上だけで行われます。raw 13-layer tensor は `13 * 4096 = 53,248` 個の float32 値/token、serialization overhead 前で約 0.203 MiB/token です。512-token caption は raw で約 104 MiB、projected bf16 cache では約 4.5 MiB になります。

この経路を使って Ideogram に似たモデルを scratch から学習し、text projection が固定済み pretrained component ではない場合は、projected cache を無効化し、raw text embed storage がかなり大きくなる前提で計画してください。

raw text encoder features が必要な場合、または cache 互換性をデバッグする場合だけ full cache を使ってください:

```json
{
  "text_embed_full_cache": true
}
```

これは Ideogram 4 の projected text embed cache 最適化を無効化し、13 層すべての text encoder output を保存します。

## 検証

Ideogram の検証は明示的に有効化するまで無効です:

```json
{
  "ideogram_validation": true
}
```

これは一時的なフラグです。上流の Ideogram CFG 推論は別の unconditional transformer を想定していますが、SimpleTuner は現在デフォルトで conditional transformer のみを学習します。有効化すると、検証では conditional transformer を negative/unconditional pass にも使うため、プロンプトと negative prompt の挙動を確認できます。

## Caption 形式

Ideogram 4 では構造化 JSON caption が望ましいです。推奨フィールド:

- `high_level_description`
- `style_description`
- `style_description.color_palette`、hex color を使う
- `compositional_deconstruction.background`
- `compositional_deconstruction.elements`
- 任意の `bbox`、形式は `[x1, y1, x2, y2]`

通常テキストと JSON caption が混在する場合は、次を有効のままにします:

```json
{
  "ideogram_auto_json": true
}
```

通常テキストは Ideogram JSON schema に包まれ、既存 JSON は正規化されて保持されます。ただし、構図、背景、要素、色まで書いた手動 JSON caption のほうが安定します。

## Prompt upsampling

任意で prompt upsampling を有効化できます:

```json
{
  "ideogram_prompt_upsample": true,
  "ideogram_prompt_enhancer_head_id": "diffusers/qwen3-vl-8b-instruct-lm-head"
}
```

JSON 変換前に Ideogram prompt upsampler でプロンプトを書き換えます。遅くなるため、まず通常の学習経路が動くことを確認してください。

## LoRA と LyCORIS

標準 PEFT LoRA は attention projection を対象にします:

```json
{
  "lora_type": "standard",
  "lora_rank": 32
}
```

LyCORIS/LoKr は Ideogram が公開している `Attention` と `FeedForward` モジュールクラスを対象にできます。full-matrix LoKr は非常に大きくなる場合があるため、素早い反復では標準 LoRA から始めてください。

## Loss の目安

Ideogram の loss は他のモデルより高く見えることがあります。`1.0` 付近またはそれ以上でも、モデルが壊れているとは限らず、検証画像の一貫性が失われるとは限りません。

テストでは、step loss が約 `0.3-1.3` の範囲で揺れ、時々より高い spike が出ても、Ideogram は一貫した検証画像を生成しました。低い scalar loss だけを期待せず、検証画像、プロンプト追従、loss が継続的に発散していないかを見て判断してください。

## 学習

```bash
simpletuner train
```

開発環境:

```bash
CONFIG_BACKEND=json CONFIG_PATH=config/config.json .venv/bin/python simpletuner/train.py
```
