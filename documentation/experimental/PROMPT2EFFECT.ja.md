# Prompt2Effect

Prompt2Effect は、エフェクトプロンプトから PEFT LoRA 重みを生成するハイパーネットワークを学習するための実験的な CLI 専用ワークフローです。SimpleTuner の通常の画像/動画 denoising トレーナーとは別です。

重要なのは、Prompt2Effect がハイパーネットワークの学習そのものを 3.3 秒にするわけではない点です。重い処理は、既存のエフェクト LoRA ライブラリに対する一度きりの学習段階へ移ります。そのハイパーネットワークができた後は、新しいプロンプトから LoRA を生成する処理が 1 回の forward pass になります。

## 学習対象

学習サンプルはメディアファイルではなく、既存の LoRA チェックポイントです。

- エフェクトプロンプト
- そのエフェクト用の PEFT LoRA チェックポイント
- 固定されたベースモデルと固定されたターゲット層スキーマ

prepare ステップは各 LoRA update を SVD 正準化された因子に変換します。学習損失は、それらの正準 LoRA 因子に対する normalized MSE であり、latent に対する diffusion loss ではありません。

## 対応ファミリー

現在のスクリプトは次をサポートします。

- `ltxvideo2`
- `wan` I2V フレーバー
- `hunyuanvideo`

生成物は、PEFT の `lora_A`、`lora_B`、`alpha` キーを持つ通常の `pytorch_lora_weights.safetensors` ファイルです。

## ファイル

Prompt2Effect は `scripts/prompt2effect/` にあります。

- `prepare.py`: LoRA manifest を検証し、SVD 正準ターゲットを書き出します。
- `train.py`: Prompt2Effect ハイパーネットワークを学習します。
- `generate.py`: 学習済みハイパーネットワークとエフェクトプロンプトから PEFT LoRA を出力します。

これは WebUI には公開されていません。

## Manifest

1 行に 1 つのエフェクト LoRA を持つ JSONL ファイルを作成します。

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

同じ Prompt2Effect 実行内のすべての LoRA は、同じターゲットモジュールスキーマと同じ入出力次元を使う必要があります。prepare 時の `--rank` で正準/生成 LoRA rank を選びます。省略すると最初の LoRA の rank が使われます。

## ターゲット準備

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

主なオプション:

- `--model_family`: `ltxvideo2`、`wan`、または `hunyuanvideo`。
- `--base_model`: ベースモデルの repo またはローカルパスを上書きします。
- `--model_flavour`: `--base_model` がない場合に、既知のファミリー既定値を使います。
- `--target_modules`: カンマ区切りの PEFT ターゲット suffix、`default`、または `all-linear`。
- `--rank`: 生成 LoRA rank。既定では最初のソース LoRA の rank。
- `--component_subfolder`: ベースモデルの component subfolder。既定ではファミリーの transformer subfolder。

`prepare.py` は次を書き出します。

- `schema.json`
- `targets.safetensors`

必要なモジュールがない、想定外のモジュールがある、またはベースモデル tensor shape と一致しない LoRA がある場合は失敗します。

## 学習

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

テキストエンコーダーは凍結され、エフェクトプロンプトの encode にのみ使われます。ベースモデル重みも凍結され、ハイパーネットワークへの構造的 conditioning として使われます。

既定ではベース重みは CPU に置かれます。選択したターゲット層が accelerator に収まる場合だけ `--base_weights_device training` を使ってください。

## LoRA 生成

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

出力ディレクトリには `pytorch_lora_weights.safetensors` が作成されます。通常の SimpleTuner/Diffusers PEFT LoRA と同じように読み込めます。

## 制限

- PEFT の linear LoRA のみ対応しています。LyCORIS、convolution LoRA、DoRA magnitude vector、任意の sidecar tensor はこのワークフローでは未対応です。
- ハイパーネットワークは、1 つの model family、base model shape、target module schema、rank に結びつきます。
- スクリプトは Accelerate、WebUI、SimpleTuner のメイン checkpoint manager と統合されていません。
- 学習品質はソースとなるエフェクト LoRA の数と多様性に依存します。少数の LoRA は経路のテストには十分ですが、汎化を期待するには不十分です。
- 生成された LoRA は、公開や本番ワークフローでの利用前に通常どおり検証してください。
