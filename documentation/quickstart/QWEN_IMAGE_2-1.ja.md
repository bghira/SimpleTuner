# Qwen Image 2.1 クイックスタート

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) は 7B・32 ブロックの画像 Transformer、Qwen3-VL テキストエンコーダー、空間圧縮率 16 倍の 64 チャンネル VAE を使います。SimpleTuner のデフォルトは `model_family: "qwen_image"`、`model_flavour: "v2.1"` です。このガイドはテキストから画像を生成する LoRA の学習を扱います。

20B の `v1.0` / `v2.0` は[旧版 Qwen Image ガイド](QWEN_IMAGE.ja.md)を参照してください。旧 `edit-*` の参照画像ペア学習は [Qwen Edit](QWEN_EDIT.ja.md)にあります。これらのアダプター、テキスト埋め込み、潜在キャッシュは 2.1 と互換性がありません。

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## インストール

Python 3.12–3.14 を使用し、環境に合った[インストールガイド](../INSTALL.ja.md)に従ってください。以下のプリセットは NVIDIA GPU で検証しています。`webshart` は依存関係に含まれます。正則化データにはネットワークとローカルキャッシュ容量が必要です。

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## VRAM に合うプリセットを選ぶ

標準例は[補助アダプター v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2)、合成データ正則化、REPA、自動フローシフトを組み合わせます。デフォルトの `qwen_image.peft-lora` は 48 GB と同じ 512px + 1024px の設定です。24/32 GB は検証も含めて 512px のみです。

| VRAM 予算 | 例 | 基準解像度 | 更新数 | 勾配チェックポイント間隔 |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

全プリセットは BF16、rank/alpha 32、バッチ 1、AdamW BF16、学習率 `1e-4`、ウォームアップ 25 更新、ノルムクリップ 1.0 を使用します。REPA は `dinov2_vitg14`、ブロック 8、重み 0.5、画像サイズ 518、空間位置合わせ、時間距離 0 です。自動シフトを有効にし、固定シフトは 0 にします。補助アダプターは学習中に固定し、検証時に無効化します。正則化の教師は両方のアダプターを無効にしたベースモデルの予測です。

通常の確率サンプラーは重みの半分を `RareConcepts/Domokun`（トリガー `🟫`）、残り半分を `is_regularisation_data: true` の `webshart/qwen-image-2.1-generated-images` に割り当てます。多解像度では各半分を 512px と 1024px に等分します。面積バケットは縦横比を保持し、0.262144 と 1.048576 メガピクセルを使います。合成データの正方形・縦長・横長は各解像度の正則化重みを等分します。厳密な交互選択ではありません。検証と保存は 250 更新ごとです。

各合成データのアスペクト比サブセットは最大 1,024 枚で、解像度とデータソースごとに潜在キャッシュを分けます。VAE タイリングは無効です。更新数は開始時の目安であり、延長前にチェックポイント画像を確認してください。

<details markdown="1">
<summary>メモリ測定結果</summary>

L40S のメモリ確認では各バックエンド 4 枚、16 更新、検証と保存を実行しました。512px・間隔 1 は 24 GiB の割り当て上限で成功し、PyTorch のピーク割り当て／予約量は 20.06 / 21.10 GiB でした。多解像度・間隔 2 は L40S で 32.49 / 41.21 GiB で成功しましたが、32 GiB 制限ではメモリ不足になりました。少数画像でのメモリ確認であり、速度や収束の測定ではありません。小さいメモリ予算は別のカードではなく L40S 上で模擬しています。 最終的な 32 GB プリセット（512px・間隔 2）も成功し、ピーク割り当て／予約量は 22.12 / 23.68 GiB でした。

</details>

## 自分のデータを使う

選んだ例の `config.json` と `dataloader.json` を学習環境へコピーし、独自のプロンプト集を用意します。`data_backend_config`、`user_prompt_library`、`output_dir` を対応するファイルに変更してください。配置は[学習チュートリアル](../TUTORIAL.ja.md)と[データローダー資料](../DATALOADER.ja.md)を参照します。

Domokun バックエンドを対象の概念または写真データに置き換えます。マルチスケールでは 512px と 1024px のバックエンドを分け、異なる ID と VAE キャッシュパスを使います。`resolution_type: "area"` はメガピクセル単位で **0.262144** と **1.048576** を設定します。`crop: false` は縦横比を保ちます。学習設定の `aspect_bucket_alignment: 32` は維持してください。

組み合わせ設定では合成バックエンドの `is_regularisation_data: true` を維持し、学習用は false にします。学習データと正則化に半分ずつの重みを割り当て、各半分を解像度間で分けます。確率サンプラーを使い、厳密な交互選択ではありません。repeats はデータが利用可能な期間を調整し、選択確率を決めません。データ、解像度、バッチ設定、分散構成を変更するときは新規実行にしてください。

## 学習用アシスタントを選ぶ

main と標準例は現在 **v2** を使います。学習中は固定して強度 1、検証では無効です。正則化バッチの親モデル目標は、学習 LoRA とアシスタントを両方無効にした**素のベースモデル予測**です。

[アシスタント v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) は 512/1024/1536/2048 で 30k 更新した rank-64 の別公開アダプターです。明示的に使う場合は下記の `assistant_lora_path` を指定します。プリセットのメモリ確認は v2 で行ったため、v3 へ変更後は VRAM の余裕を再確認してください。

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v3",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

アシスタントなしで学習する場合は `disable_assistant_lora: true` を設定します。新しいアシスタントを作る場合も同じです。下流推論では学習した LoRA だけを読み込みます。アシスタントと REPA プロジェクターは学習用です。

## 学習と品質を検証する

例では 250 更新ごとに検証と保存を行い、推論 40 ステップ、true CFG 1、シード 42 を使います。24/32 GB は 512px、それ以上では 512px と 1024px を検証します。独自設定では以下を維持してください。

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

概念プロンプトにはトリガーを含め、無関係な被写体も使って整合性と品質を確認します。比較ではプロンプト、シード、デコーダー、解像度、ガイダンスを揃えてください。学習と出力の解像度は別です。512px 学習のアダプターも 1024px で試します。この設定は 2048px の概念データを必須としません。

[実験集](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments)にはアシスタント、正則化、REPA、自動シフト、長期学習の比較があります。悪化してから回復する実行もあり、一つの弱いチェックポイントだけでは判断できません。[写真美学 v3 比較](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3)では、512px の 50k 更新モデルが 1024px の 10k モデルより 1MP 出力でも豊かなシーンの細部を示します。ただし更新数が異なります。

## VAE と推論

Qwen Image 2.1 は、検証やその他の VAE デコードに [Ollin のテクスチャ修正版 VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) をデフォルトで使用します。微調整されたのはデコーダーのみで、エンコーダーは変更されていないため、既存の 2.1 学習用潜在表現と互換性があります。VAE のリビジョンはベースモデルとは独立して固定されています。元の VAE で出力を再現するには、`pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"` を設定してください。明示的な VAE 指定と旧フレーバーの動作は変わりません。

標準 PEFT アダプターには SimpleTuner 内蔵パイプラインを使えるため、Diffusers の Git インストールは不要です。下記は LoRA を読み込み、学習専用 REPA プロジェクターのテンソルを除外します。全フレームをデコードし、アシスタントは読み込みません。

モデルの CPU オフロードにより、テキストエンコーダー、Transformer、VAE が同時に GPU メモリを占有するのを避けます。VRAM に余裕がある場合は `pipe.enable_model_cpu_offload()` を `pipe.to("cuda")` に置き換えると、繰り返しの推論を高速化できます。

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## メモリと画像のアーティファクト

まず BF16（`base_model_precision: "no_change"`）を使います。量子化の前にバッチを小さくするか小さいプリセットを選びます。量子化は省メモリになりますが高速化を保証しません。例は領域単位のコンパイルを有効にします。長いキャプション、追加解像度、検証、REPA は VRAM ピークに影響するため、バッチ拡大前に実測してください。

余裕があれば `vae_enable_tiling: false` を維持します。修正デコーダーはキャンバス状の質感に対応します。タイルの色の継ぎ目は空間文脈に関する別の問題です。ノルムクリップ 1.0 は `grad_clip_method: "norm"` と `max_grad_norm: 1.0` で、旧 AnyFlow の要素ごとの 0.01 クリップとは異なります。

<details markdown="1">
<summary>メモリ測定結果: VAE</summary>

Qwen Image 2.1 は、使われない時間方向の特徴キャッシュを保持せずに単一画像をデコードします。H200 上で BF16 を用いて 2048×2048 の画像を 1 枚単独でデコードした測定では、出力を完全に維持しながら、割り当てメモリのピークが 26.87 GiB から 15.28 GiB に減りました。タイルデコードはさらにメモリを節約しますが、空間的な文脈が制限されるため色の継ぎ目が生じる場合があります。未使用キャッシュの削除では、この継ぎ目は解消しません。

</details>

## 過去の設定と実験

以下はアシスタント作成や蒸留の実験で、上記の推奨下流学習とは別です。過去の試験設定と結果は参考として保存しています。

<details markdown="1">
<summary>AnyFlow 試験とオプティマイザの履歴</summary>

<a id="experimental-anyflow-pilot"></a>

### 実験的な AnyFlow パイロット

3 つの `qwen_image-2.1-anyflow-stage*.peft-lora` サンプルは、区間蒸留によってベースモデルの挙動を保ちながら `🟫` を導入できるか検証します。実験用の設定です。Qwen のモデルカードは、以前の劣化が guidance distillation に起因するとは示していません。

全段階で 1024px、BF16、AdamW、rank 32、batch 4 を使用します。段階 1 は H200 で勾配チェックポイントを無効にし、段階 2 と 3 は連続 2 ブロック単位で適用します。上記の補助 REPA プリセットとは別の蒸留学習です。

これらの AnyFlow 例は `grad_clip_method: "value"` と `max_grad_norm: 0.01` を明示的に指定し、各勾配要素を ±0.01 に制限します。これは全体の勾配ノルムの上限ではありません。ノルムを 1.0 でクリップする実験では、`grad_clip_method: "norm"` と `max_grad_norm: 1.0` の両方を設定してください。

1. **段階 1：** 学習率 `1e-5` で forward AnyFlow を 10,000 回更新します。CC12M は `webshart/cc12m-structured-captions` の `caption_key: "long_caption"`、e621 は `webshart/e621-2024-webp-4Mpixel-webshart-indices` を使用します。各事前知識データセットは条件を満たす 4,096 枚まで、サンプリング重みは各 0.49 です。`RareConcepts/Domokun` はトリガー `🟫`、重み 0.02、`repeats: 0` を使用します。
2. **段階 2：** 学習率 `2e-6` で on-policy DMD を 2,000 回更新し、同じデータ混合で forward 目的も学習します。これは AnyFlow DMD であり、選好ペアの DPO ではありません。両段階にキャラクター画像を含めます。凍結教師だけでは新しいキャラクターを教えられません。
3. **段階 3：** 学習率 `5e-7` で 100 回更新します。Domokun の重みは 0.5、正則化用の 2 データセットは各 0.25、各 64 枚までです。全区間を `r=t` として生の flow 目標を使い、区間埋め込みを保持して短い教師あり微調整を行います。正則化バッチは adapter を無効にしたベース予測を使います。

段階 1 を 20,000 回まで延長する前に、2,000、5,000、10,000 回時点のチェックポイントを確認します。段階 2 の上限は 2,000 回とし、同条件の検証画像が悪化した場合は早めに停止します。更新回数だけで完成モデルとは判断できません。

可変のキャプション長に対応するため動的コンパイルを有効にしています。`schedule_shift: 2.000802574061872` は公開された scheduler の 1024px（4,096 latent tokens）での設定に対応します。解像度を変更する場合は再計算してください。

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

段階 2 と 3 は `init_lora` で直前の最終 adapter を読み込み、optimizer と dataloader の状態を初期化します。重みは未消費のデータセット間の選択を制御し、最終的なサンプル比率を保証しません。この限定パイロットは事前知識コーパス全体を使用しません。文字列の `long_caption` は既存の Webshart caption selector で扱えるため、JSON オブジェクト caption の対応は不要です。

段階 1、2 は `diffusion_target: "base_prediction"` と `fuse_guidance_scale: 1.0` を使用します。diffusion 分岐は凍結された条件付き予測場を保ち、他の分岐はデータから学習します。これは保持の仮説であり、キャラクター習得を保証しません。付属のキャラクターと事前知識プロンプトを 4、16、40 推論ステップで比較し、ベースモデルの 40 ステップ出力を基準にしてください。自動検証は 4 ステップです。目的関数とチェックポイントの要件は [AnyFlow](../experimental/ANYFLOW.ja.md) を参照してください。

完了した試験では、ステージ 1 は 10,000 更新、ステージ 2 は 2,000 更新に到達しました。適応器を再ロードした 40 ステップのキツネ画像とキャラクタープロンプトの画像は一貫性を保ちましたが、後者は Domokun ではなく人物を生成しました。4 ステップ画像にはぼけやノイズが残りました。別の L40S 比較では両チェックポイントを 1024px、シード 42、CFG 1/2/4/6、空のネガティブプロンプトで検証しました。前向き呼び出しのアサーションにより、CFG が 1 より大きい場合は各ステップで 2 回実行されることを確認しました。確認したキツネと海辺の画像は改善せず、高い CFG は彩度とアーティファクトを増やしました。原因はまだ特定できておらず、ステージ 3 は未検証です。

同じチェックポイントからステージ 1 を各 1,000 更新継続し、要素ごとのクリッピング閾値 0.01 と 1.0 を比較しました。再読み込み後の 4 ステップのキツネ画像は両方ともノイズが残り、40 ステップの海辺の画像も人物を描写しました。クリッピングが適用された更新は 0.01 で 65/1,000、1.0 で 0/1,000 でした。両群とも未修正オプティマイザを使用し、クリッピング適用前から初期勾配が異なったため、微小な差を閾値だけに帰することはできません。

<a id="qwen21-optimizer-correction"></a>

ここで紹介したステージ 1 とアシスタントの試験は、AdamW BF16 の確率的加算ヘルパーの修正前に実行されました。この関数は `input + alpha * other` ではなく `other + alpha * input` を計算していました。β₁ = 0.9 では、一次モーメントの更新が `m = 0.9 * m + 0.1 * g` ではなく `m = 0.09 * m + g` になっていました。厳密に表現できる値を使う回帰テストを CPU、MPS、CUDA で実施済みです。画像の観察結果はこれらのチェックポイントには有効ですが、アーキテクチャや蒸留目的だけを評価する試験にはなりません。修正済みオプティマイザで学習を再検証する必要があります。

修正済みオプティマイザ、同じ開始チェックポイント、要素ごとのクリッピング値 1.0 で 1,000 更新の継続試験を繰り返しても、確認した 4 ステップのキツネと海辺の画像は改善しませんでした。40 ステップではキツネは整合性を保ち、海辺のプロンプトは依然として人物を生成しました。これは既存チェックポイントの回復を検証する試験であり、修正済みオプティマイザで最初から学習する試験ではありません。全体的な失敗原因は未解明です。

</details>

<details markdown="1">
<summary>アシスタント作成：オンライン、多解像度、オフライン</summary>

<a id="assistant-lora"></a>

### キャプションから補助 LoRA を学習する

`qwen_image-2.1-assistant-lora.peft-lora` は L40S 向けの実験用設定です。BF16、バッチ 1、間隔 2 の勾配チェックポイント、rank 32、学習率 `1e-4` の AdamW BF16 を使います。1,000 更新を予定し、50 更新ごとに検証・保存します。本格的な学習前に十二件のスモークテスト用キャプションを多様な文章へ置き換えてください。収束を確認した設定ではありません。

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

`distillation_method: assistant_lora` を指定します。キャプションのテキスト埋め込みを事前計算し、各バッチでアダプターを無効化して、40 推論ステップ、CFG 1 でベースモデルの新しい潜在変数を生成します。専用パイプラインは transformer を共有し、VAE、processor、テキストエンコーダーを読み込みません。アダプターを復元した後、通常のノイズ除去学習を行います。最終潜在変数はキャッシュしません。現在は Qwen Image 2.1 のテキストから画像への生成のみ対応します。

`distillation_config.assistant_lora` は `num_inference_steps`（既定 40）、`resolutions`（空でない `[幅, 高さ]` のリスト、既定 `[[1024, 1024]]`）、`seed`（既定 42）を受け取ります。寸法は 32 の倍数です。解像度はバッチごとに循環し、シードは端数バッチを含めサンプルごとに進みます。カウンターはチェックポイントに保存します。再開時はデータセット、バッチサイズ、勾配累積、分散構成を変更しないでください。テキストキャッシュのオンデマンド処理には対応しません。

動作確認には 8 更新・教師 2 ステップを使い、画像評価前に教師 40 ステップへ戻します。100、250、500、1,000 更新で、同じプロンプトとシードのベース画像と比較してください。実用性は別の概念学習でも検証が必要です。

Qwen Image 2.1 の LoRA 学習では、[学習補助アダプター v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2) を既定で読み込みます。学習中は固定し、検証時は無効にします。無効化には `disable_assistant_lora: true`、別のアダプターには `assistant_lora_path` を指定します。旧フレーバーの既定値は変わりません。新しい補助アダプターを学習する際は、専用の例と同様に `disable_assistant_lora: true` を維持してください。

蒸留方式はこの方法のみを指定してください。他の蒸留器との併用は対応しません。

キャプションデータセットでは `dataloader_prefetch: false` が必要です。チェックポイントの位置を消費済みキャプションと一致させるためです。再開時にキャプション ID・本文、バッチサイズ、繰り返し、シャッフル、シード、勾配累積、分散構成が変わるとエラーになります。 補助 LoRA のチェックポイントでは、生成シード、解像度リスト、教師の推論ステップ数の変更も拒否します。

<a id="assistant-lora-multires"></a>

#### 4 つの基準解像度とアスペクト比バケットを使うアシスタント実験

`qwen_image-2.1-assistant-lora-multires.peft-lora` は、基準解像度 512、1024、1536、2048 に対応する計 12 個のアスペクト比バケットを巡回します。各基準解像度に正方形、縦長 4:7、横長 7:4 のバケットがあり、1 バッチにつき 1 個を使用します。1 周期では各基準解像度と各アスペクト比を均等に学習します。基本例の教師 40 ステップ、BF16・バッチ 1、2 ブロック間隔のチェックポイント、ランク 32、AdamW BF16、1,000 更新を維持し、キャプションバックエンドと検証プロンプトを共有します。動作確認用キャプションは、基準実験と同じ多様なキャプション集合に置き換えてください。新しい出力ディレクトリでアダプターとオプティマイザーを新規作成します。`resume_from_checkpoint: ""` で再開を無効にします。比較のため、最初の実験の重みを保存してください。

複数の画像サイズがアシスタントを改善するか調べる実験であり、ネイティブ解像度の要件や品質向上を実証するものではありません。L40S の 1,000 更新は全 12 バケットを完了し、1024×1024 と 2048×2048 で検証しました。最終的なキツネと肖像画像は一貫性を保ちましたが、タイル VAE デコードによる既知の色の継ぎ目が残っています。教師は VAE デコードなしで潜在ターゲットを生成します。後続の概念学習への効果は未検証です。

Assistant LoRA は AnyFlow と同じく、実行中に限って Dynamo キャッシュを最低 32 エントリにし、教師・学生・検証の各バリアントを保持します。ユーザー設定がこれより大きければ維持し、終了時に元の値を復元します。解像度や長いキャプションを追加するときは再コンパイルを監視してください。

その後の Domokun 比較試験では、2048px、バッチ 1、学習率 `1e-5`、要素ごとのクリッピング値 1.0 で、新しいアダプターを各 250 更新学習しました。学習時のアシスタント強度は対照群で 0、アシスタント群で 1 とし、推論時は両方で無効化しました。初期アダプター重みと初期検証画像は完全一致しました。1024px・40 推論ステップで確認した最終結果は、両群ともキャラクタープロンプトから人物を生成し、キツネと肖像の事前知識は維持しました。アシスタントの利点は実証されていません。両群の学習とアシスタントの事前学習は、[上記](#qwen21-optimizer-correction)の未修正オプティマイザを使用しています。

<a id="assistant-lora-offline"></a>

#### 再利用できる生成画像による補助 LoRA

`qwen_image-2.1-assistant-lora-offline.peft-lora` は Webshart 経由で [10,000 枚の生成画像](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images)を学習します。データセットは CC12M の `long_caption`、教師のネイティブ 40 ステップ、CFG 1、画像全体の VAE デコードを使います。12 個の画像バックエンドが、512、1024、1536、2048 の基本解像度ごとに正方形・縦長・横長のバケットを扱い、同じ重みでサンプリングします。繰り返しは設定しません。

新規学習用の設定は、修正済み `adamw_bf16`、学習率 `1e-4`、`grad_clip_method: "norm"`、`max_grad_norm: 1.0`、BF16、バッチ 1、ランク 32、2 ブロック単位のチェックポイントです。1,000 更新を予定し、50 更新ごとに保存、100 更新ごとに 1024 で検証します。公開前に別途 2048px のプレビューを生成してください。VAE のタイル処理を無効にし、エンコードのバッチは 1 にします。`vae_cache_ondemand: true` はサンプリング時に画像をエンコードしてキャッシュするため、1,000 更新の前に全 10,000 枚をエンコードする必要がありません。通常の画像学習でオンライン教師生成を置き換えるため、補助アダプターの作成時は `distillation_method` を省略し、`disable_assistant_lora: true` を維持してください。

画像は複数の実験で再利用できます。PNG は VAE で再エンコードするため、教師の最終潜在表現を直接使う場合とは同一ではありません。公開前に検証画像を確認し、別の概念学習で補助効果を評価してください。キャプションのみの設定から切り替える場合は、元のオプティマイザーやデータセット状態を再開せず、新規に学習します。

[置き換え後の v1 アシスタント](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) は、この設定で L40S 上の 1,000 更新を完了し、最終画像を 1024 と 2048 で確認しました。その後の Domokun 比較では、2048、LR `1e-4`、ノルムクリッピング 1.0 で各 1,000 更新を実行し、アシスタントは学習中に凍結、検証中に無効化しました。対照群とアシスタント群の両方で、二つのキャラクタープロンプトは依然として人物を生成しました。対照群では無関係なキツネのプロンプトに強い Domokun の特徴が混入しましたが、アシスタント群ではキツネとして認識できる画像を維持しました。両群のポートレートは整合性を保ちました。これは概念の混入を減らした限定的な証拠であり、キャラクター学習の成功や全般的な品質向上を示すものではありません。評価は一つの学習シードと四つのプロンプトに限られます。

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```

</details>
