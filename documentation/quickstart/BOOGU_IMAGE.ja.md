# Boogu-Image 0.1 クイックスタート

このガイドでは、SimpleTuner で Boogu-Image 0.1 の LoRA と LyCORIS LoKr を学習する手順を説明します。Boogu-Image は text-to-image、turbo、edit の flavour を持つ flow matching 画像モデルです。SimpleTuner 統合ではローカルの pipeline / transformer 実装を使い、エクスポート済み pipeline checkpoint は Hugging Face の `SimpleTuner` namespace にあります。

同梱のスターター config:

```bash
simpletuner/examples/boogu-image-v0.1.peft-lora/config.json
simpletuner/examples/boogu-image-v0.1.lycoris-lokr/config.json
```

## ハードウェア要件

Boogu-Image は大型の transformer 画像モデルとして扱ってください。最初は 1024px、batch size 1、bf16 mixed precision、gradient checkpointing 有効で始めるのが無難です。

推奨の開始点:

- **標準:** `v0.1-base`、bf16 LoRA、rank 16。
- **低 VRAM:** `v0.1-base-fp8`、`v0.1-turbo-fp8`、`v0.1-edit-fp8` などの FP8 flavour。
- **高速な検証/推論:** turbo flavour。ただし assistant LoRA の状態に注意してください。
- **編集:** paired conditioning data と `v0.1-edit` または `v0.1-edit-fp8`。

メモリ使用量は rank、optimizer、validation 解像度、offload、compile、FP8 の有無で変わります。単一 H100 では、同梱 PEFT LoRA 例を 1024px、benchmark と validation 有効で 1000 steps 学習できます。

小さい GPU では FP8 weights、rank 8-16、`train_batch_size=1`、gradient checkpointing、model/group offload から始めてください。

### メモリ offload

transformer weight がボトルネックの場合、group offload で VRAM を下げられます:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream
```

任意の disk offload:

```bash
--group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- stream は CUDA でのみ有効です。SimpleTuner は ROCm、MPS、CPU では無効化します。
- group offload を他の CPU offload と組み合わせないでください。
- disk offload には高速なローカル NVMe を推奨します。

### Torch compile と attention

NVIDIA GPU では、利用可能なら Hugging Face Hub kernel の attention alias を使います:

```json
{
  "attention_mechanism": "flash-attn-3-hub",
  "dynamo_backend": "inductor",
  "dynamo_use_regional_compilation": true
}
```

特定の GPU/driver で compile された validation が黒画像になる場合は、学習レシピを変える前に torch compile を無効にして再テストしてください。

## インストール

pip で SimpleTuner をインストールします:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell ユーザー
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

手動インストールや開発セットアップは [installation documentation](../INSTALL.md) を参照してください。

## 環境設定

### Web UI

SimpleTuner WebUI で Boogu-Image の training config を作成できます:

```bash
simpletuner server
```

http://localhost:8001 を開き、model family に `boogu_image` を選びます。

### 手動 / コマンドライン

`config/config.json.example` を `config/config.json` にコピーします:

```bash
cp config/config.json.example config/config.json
```

主な設定:

- `model_type` - `lora`。
- `lora_type` - PEFT LoRA は `standard`、LyCORIS LoKr は `lycoris`。
- `model_family` - `boogu_image`。
- `model_flavour` - `v0.1-base`、`v0.1-base-fp8`、`v0.1-turbo`、`v0.1-turbo-fp8`、`v0.1-edit`、`v0.1-edit-fp8`。
- `pretrained_model_name_or_path` - 通常は未設定にし、flavour から `SimpleTuner/Boogu-Image-0.1-*` pipeline を選ばせます。
- `output_dir` - checkpoint と validation 画像の保存先。
- `train_batch_size` - `1` から始めます。
- `resolution` - `1024` から始めます。
- `resolution_type` - multi-aspect bucket には `pixel_area`。
- `validation_resolution` - 通常は `1024x1024`。複数指定は comma 区切り。
- `validation_guidance` - base/edit では `4.0` 前後。
- `validation_num_inference_steps` - base/edit では `30` 前後。turbo は少ない step で動きます。
- `mixed_precision` - 最近の NVIDIA GPU では `bf16`。
- `gradient_checkpointing` - 有効にします。
- `flow_schedule_shift` - examples は `3` を使います。

最小 PEFT LoRA config:

```json
{
  "model_type": "lora",
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base",
  "lora_type": "standard",
  "lora_rank": 16,
  "lora_alpha": 16,
  "output_dir": "output/models-boogu-image-v0.1",
  "train_batch_size": 1,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "validation_prompt": "a polished product photo of a ceramic mug on a walnut desk",
  "validation_steps": 50,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "flow_schedule_shift": 3,
  "optimizer": "adamw_bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant_with_warmup",
  "lr_warmup_steps": 10,
  "max_train_steps": 1000,
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "data_backend_config": "config/examples/multidatabackend-small-dreambooth-1024px.json"
}
```

## examples の実行

```bash
simpletuner train example=boogu-image-v0.1.peft-lora
simpletuner train example=boogu-image-v0.1.lycoris-lokr
```

開発 checkout 形式:

```bash
simpletuner train env=examples/boogu-image-v0.1.peft-lora
simpletuner train env=examples/boogu-image-v0.1.lycoris-lokr
```

## FP8 flavour

エクスポート済み FP8 pipeline weights を使う場合は `-fp8` flavour を選びます:

```json
{
  "model_family": "boogu_image",
  "model_flavour": "v0.1-base-fp8"
}
```

同じパターンで `v0.1-turbo-fp8` と `v0.1-edit-fp8` も使えます。Boogu の `.bin` ファイルを直接指定する必要はありません。

## Turbo assistant LoRA

SimpleTuner は `v0.1-turbo` と `v0.1-turbo-fp8` で assistant LoRA のコードパスを有効にします。現時点では、この統合用の別 adapter が公開されていないため、adapter path は `None` placeholder です。

adapter が用意されるまでは turbo をエクスポート済み pipeline target として扱い、品質は validation で確認してください。最も予測しやすい baseline は `v0.1-base` です。

## edit training

Boogu edit flavour には paired conditioning data が必要です。[Qwen Image Edit quickstart](./QWEN_EDIT.md) と同じ paired-reference dataset 構造を使ってください。

text-to-image LoRA では base または turbo flavour を使います。

## validation prompt

`validation_prompt` は主 validation prompt です。広く確認するには prompt library を追加します:

```json
{
  "product": "a polished product photo of <token> on a walnut desk",
  "studio": "a clean studio portrait of <token> with softbox lighting",
  "cinematic": "a cinematic scene featuring <token>, detailed lighting, shallow depth of field"
}
```

config から指定します:

```json
{
  "validation_prompt_library": "config/user_prompt_library.json"
}
```

過学習、prompt collapse、style drift を見つけやすいように、十分に異なる prompt を使ってください。

## 推論

学習後は、学習に使った flavour と同じ Boogu-Image pipeline で adapter を読み込みます。通常の主要ファイル:

```bash
output/models-boogu-image-v0.1/pytorch_lora_weights.safetensors
```
