# FLUX.3 Image Klein クイックスタート

このガイドでは、自分の画像で FLUX.3 Image Klein の LoRA を学習します。LoRA は、元のモデルの重みを分けて保持しながら、被写体や画風を追加学習する小さなアダプターです。

`model_family: "flux3_image"`、`model_flavour: "klein-base"`、`model_type: "lora"` から始めます。[black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) が読み込まれます。ダウンロード前にモデルページでアクセス条件に同意し、[FLUX [Kom]munity ライセンス](https://huggingface.co/black-forest-labs/flux-3-image-klein-base/blob/main/LICENSE)を確認してください。

## インストール

Python 3.13、Git、モデル・画像・キャッシュを保存する空きディスク容量が必要です。以下は NVIDIA CUDA GPU 向けの手順です。他のプラットフォームや CUDA 13 は[インストールガイド](../INSTALL.ja.md)を参照してください。

```bash
git clone https://github.com/bghira/SimpleTuner.git
cd SimpleTuner
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e '.[cuda]'
hf auth login
```

以降のコマンドは仮想環境を有効にしたまま `SimpleTuner` ディレクトリで実行します。`hf auth login` には、モデルの読み取り権限がある Hugging Face トークンを入力してください。

## GPU に合う設定を選ぶ

GPU メモリ（VRAM）によって同時に処理できる画像数が決まります。容量に合う例を選んでください。24、32 GB の Base 設定は解像度 512、それ以上の設定は 1024 のデータセットも含みます。

| GPU メモリ | Base 設定例 | 512 / 1024 のバッチ内画像数 |
| --- | --- | --- |
| 24 GB | `flux3_image-24g.peft-lora` | 3 / — |
| 32 GB | `flux3_image-32g.peft-lora` | 4 / — |
| 48 GB | `flux3_image-48g.peft-lora` | 9 / 2 |
| 80 GB | `flux3_image-80g.peft-lora` | 18 / 5 |
| 144 GB | `flux3_image-144g.peft-lora` | 36 / 10 |

蒸留版の `klein` には次のレシピを選びます。後述の設定コピーコマンドで、例の名前全体を置き換えてください。学習スケールは正方形に換算した面積を表し、画像の縦横比は維持されます。

| GPU メモリ | 蒸留版の設定例 | 学習スケール | バッチ内画像数 |
| --- | --- | --- | ---: |
| 24 GB | `flux3_image-klein-24g.peft-lora` | 768px | 1 |
| 80 GB | `flux3_image-klein-80g.peft-lora` | 1024px | 4 |

表のバッチ数を出発点にしてください。長いキャプションや参照画像はメモリを多く使うため、不足時は `train_batch_size` を下げます。設定例は rank 32 LoRA、学習率 `1e-4`、画像特徴を比較する REPA を使います。付属 dataloader は被写体画像と正則化画像を組み合わせ、モデルの幅広い画像知識の保持を助けます。

## 画像とキャプションを準備する

SimpleTuner 内に `datasets/my-concept` を作り、学習画像を置きます。各画像と同名のテキストファイルにキャプションを書きます。

```text
datasets/my-concept/
  image01.png
  image01.txt
  image02.jpg
  image02.txt
```

被写体 LoRA では `TOK` のような固有の単語を選び、キャプションで統一して使います。被写体、背景、姿勢などを記述してください。`image01.txt` の例です。

```text
木のテーブルに置かれた、青い羽の小さな陶器のフクロウ TOK の写真。
```

## 学習設定を作る

選んだレシピをコピーします。以下は `flux3_image-48g.peft-lora` の例です。名前全体を、選択した Base または蒸留版のレシピに置き換えてください。

```bash
mkdir -p config/examples
cp -R simpletuner/examples/flux3_image-48g.peft-lora config/examples/
cp simpletuner/examples/flux3_image-prompts.json config/examples/
cp config/examples/flux3_image-48g.peft-lora/config.json config/config.json
```

テキストエディターで `config/config.json` を開きます。最初に上記のローカル画像で実行する場合、既存の設定内で次の値を変更します。

```json
{
  "data_backend_config": "config/dataloader.json",
  "output_dir": "output/my-flux3-lora",
  "train_batch_size": 1
}
```

バッチ数 1 なら少数の画像で開始できます。最初の実行が成功し、画像数とメモリに余裕があれば増やしてください。`max_train_steps` で学習の長さを調整します。`config/examples/flux3_image-prompts.json` に被写体の単語を含むプロンプトを書き、学習中の進捗画像に使います。

以下を `config/dataloader.json` に保存します。画像の場所、キャプションの読み方、キャッシュの保存先を指定するファイルです。

```json
[
  {
    "id": "my-concept",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my-concept",
    "caption_strategy": "textfile",
    "resolution": 0.589824,
    "resolution_type": "area",
    "crop": false,
    "cache_dir_vae": "cache/vae/flux3-image/my-concept",
    "text_embeds": "text"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image/my-concept"
  }
]
```

このローカルデータセットでは、`resolution: 0.589824` と `resolution_type: "area"` で 768×768 相当の面積になります。80 GB の蒸留版レシピの 1024 スケールには `1.048576` を使います。

`resolution` は学習画像のサイズです。画像寸法は 16 の倍数にそろえられます。キャッシュは画像の潜在表現とテキスト埋め込みを再利用し、次回の起動を速めます。モデルファミリーごとに保存先を分けてください。追加データや正則化画像は[データローダーガイド](../DATALOADER.ja.md)を参照してください。

## 任意：プロンプトアップサンプラーで説明を詳しくする

短い学習キャプションや検証プロンプトを [FLUX.3 Image Klein アップサンプラー](https://huggingface.co/black-forest-labs/flux-3-image-klein-pup)で詳しくするには、`config/config.json` に次の設定を追加します。

```json
{
  "flux3_prompt_upsample": true
}
```

既定では無効です。有効にすると別の Qwen3.5-9B モデル（重み約 19 GB）をダウンロードし、テキスト埋め込みのキャッシュ前に正のプロンプトを書き直します。編集では対応する参照画像を使います。負のプロンプト、dropout の空キャプション、元のキャプションファイルは保持されます。

追加のシステム RAM と前処理時間を確保してください。アップサンプラーとテキストエンコーダーが交互に GPU を使い、生成した埋め込みを学習で再利用します。展開した埋め込みと元の埋め込みは分けて保存され、同じ参照画像を再び選んだ場合はキャッシュを再利用します。

## 学習を開始する

SimpleTuner ディレクトリで実行します。

```bash
simpletuner train
```

初回はモデルのダウンロード、画像の読み込み、テキストと画像のキャッシュ作成後に学習が始まります。コンパイルで起動時間が増える場合もあるため、処理を実行したまま待ってください。進捗画像とチェックポイントは `output/my-flux3-lora` に保存されます。進捗画像で被写体の学習状況を確認します。

`klein-base` の検証は 50 ステップ、guidance 3.5 を使います。蒸留モデルを学習する場合は `model_flavour: "klein"` を選びます。`black-forest-labs/flux-3-image-klein` を読み込み、16 ステップ、guidance 1 を使います。事前にそのリポジトリのアクセスを取得してください。

## 編集・参照画像の学習

両フレーバーで参照画像は任意です。ターゲットには編集後の画像と指示キャプション、条件データセットには元画像や参照画像を入れます。以下を別の JSON に保存し、`data_backend_config` を設定して `train_batch_size: 1` から開始します。

```json
[
  {
    "id": "edits",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/edits",
    "caption_strategy": "textfile",
    "resolution": 512,
    "resolution_type": "pixel",
    "conditioning_data": ["references"],
    "cache_dir_vae": "cache/vae/flux3-image/edits",
    "text_embeds": "text"
  },
  {
    "id": "references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "datasets/references",
    "conditioning_type": "reference_loose",
    "resolution": 512,
    "resolution_type": "pixel",
    "cache_dir_vae": "cache/vae/flux3-image/references"
  },
  {
    "id": "text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux3-image-edit"
  }
]
```

ファイル名を対応させます。例は `edits/sample.png`、`edits/sample.txt`、`references/sample.png` です。`reference_loose` は異なるサイズ・縦横比を許可し、`reference_strict` は整列したクロップ用です。`conditioning_data` に ID を追加すると最大 10 枚の参照を使えます。全部を同時に渡すには `conditioning_multidataset_sampling: "combined"` を指定します。`conditioning_data` を省くと文生図学習です。参照、ターゲット、他のモデルファミリーのキャッシュを分離してください。

`conditioning_multidataset_sampling: "random"` では、各学習サンプルが1つの参照データセットとそのキャプションを使用します。各参照データセットに `caption_strategy: "textfile"` を設定し、`references/sample.txt` のような対応する `.txt` ファイルに編集指示を記述してください。

## LoRA で画像を生成する

`pytorch_lora_weights.safetensors` が入ったチェックポイントのディレクトリを探し、下の `output/my-flux3-lora/checkpoint-1000` を置き換えます。学習時と同じモデルを使ってください。この例は base と CFG-Zero-Star を使います。

```python
import torch
from simpletuner.helpers.models.flux3_image.pipeline import Flux3ImagePipeline

pipe = Flux3ImagePipeline.from_pretrained(
    "black-forest-labs/flux-3-image-klein-base", torch_dtype=torch.bfloat16
).to("cuda")
pipe.load_lora_weights("output/my-flux3-lora/checkpoint-1000")
image = pipe(
    "A photo of TOK, a small ceramic owl, sitting on a wooden shelf",
    num_inference_steps=50,
    guidance_scale=3.5,
    use_cfg_zero_star=True,
).images[0]
image.save("my-flux3-image.png")
```

## アシスタント LoRA

`model_flavour: "klein"` の LoRA 学習では、[公開済みの学習用アシスタント](https://huggingface.co/SimpleTuner/flux-3-klein-image-training-assistant)を凍結したアダプターとして自動で読み込みます。学習時の `assistant_lora_strength` は既定で `1.0`、検証時の `assistant_lora_inference_strength` は `0.0` です。無効にするには `disable_assistant_lora: true` を設定します。別のアダプターを使う場合は `assistant_lora_path` と、必要に応じて `assistant_lora_weight_name` を指定します。

独自のアシスタントを学習するには `flux3_image-assistant-lora.peft-lora` を選びます。このレシピは既存のアシスタントを無効にし、モデルが生成したサンプルから学習します。完了後、蒸留版の LoRA 設定で `assistant_lora_path` に保存先を指定してください。

追加設定は[オプションリファレンス](../OPTIONS.ja.md)を参照してください。TwinFlow は `twinflow_enabled: true` と `use_ema: true` を同時に設定します。EMA はこの学習方法で使う教師モデルを維持します。
