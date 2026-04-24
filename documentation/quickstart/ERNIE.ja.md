# ERNIE-Image [base / turbo] クイックスタート

このガイドでは ERNIE-Image の LoRA 学習を扱います。ERNIE-Image は Baidu の single-stream flow-matching transformer 系列で、SimpleTuner では `base` と `turbo` をサポートします。

## ハードウェア要件

ERNIE は軽量モデルではありません。大きめの single-stream transformer と同じ感覚で考えてください。

- int8 量子化 + bf16 LoRA でも、現実的には 24GB 以上の GPU が望ましいです
- 16GB でも RamTorch と強めの offload で動く可能性はありますが、かなり遅くなります
- マルチ GPU、FSDP2、CPU/RAM offload も有効です

Apple GPU での学習は推奨しません。

## インストール

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

詳細は [インストール文書](../INSTALL.md) を参照してください。

## 環境のセットアップ

### WebUI

```bash
simpletuner server
```

その後、トレーニングウィザードで ERNIE ファミリーを選択します。

### コマンドライン

まずは同梱のサンプルから始めるのが簡単です。

- サンプル設定: `simpletuner/examples/ernie.peft-lora/config.json`
- 実行用ローカル環境: `config/ernie-example/config.json`

実行:

```bash
simpletuner train --env ernie-example
```

手動設定で重要なのは次の項目です。

- `model_type`: `lora`
- `model_family`: `ernie`
- `model_flavour`: `base` または `turbo`
- `pretrained_model_name_or_path`:
  - `base`: `baidu/ERNIE-Image`
  - `turbo`: `baidu/ERNIE-Image-Turbo`
- `resolution`: まずは `512`
- `train_batch_size`: `1`
- `ramtorch`: `true`
- `ramtorch_text_encoder`: `true`
- `gradient_checkpointing`: `true`

サンプル設定では以下を使っています。

- `max_train_steps: 100`
- `optimizer: optimi-lion`
- `learning_rate: 1e-4`
- `validation_guidance: 4.0`
- `validation_num_inference_steps: 20`

### Turbo の Assistant LoRA

ERNIE Turbo には assistant LoRA の仕組みがありますが、現在はデフォルトのアダプターパスはありません。

- 対応 flavour: `turbo`
- デフォルト重み名: `pytorch_lora_weights.safetensors`
- 必要なもの: ユーザー指定の `assistant_lora_path`

独自の assistant adapter がある場合:

```json
{
  "assistant_lora_path": "your-org/your-ernie-turbo-assistant-lora",
  "assistant_lora_weight_name": "pytorch_lora_weights.safetensors"
}
```

使わない場合:

```json
{
  "disable_assistant_lora": true
}
```

### データセットとキャプション

サンプル環境では次を使います。

- `dataset_name`: `RareConcepts/Domokun`
- `caption_strategy`: `instanceprompt`
- `instance_prompt`: `🟫`

これは smoke test には十分ですが、ERNIE は 1 トークンのトリガーより、普通の文章キャプションの方が安定しやすいです。実運用では `a studio photo of <token>` のような、少し説明的なキャプションを勧めます。

### 追加機能

ERNIE でも次の機能が使えます。

- TREAD
- LayerSync
- REPA / CREPA 系の hidden state capture
- Turbo の assistant LoRA 読み込み

まずは基本の学習を通してから有効化するのが安全です。
