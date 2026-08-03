# クイックスタートガイド

**注意**: より高度な設定については、[チュートリアル](TUTORIAL.md)および[オプションリファレンス](OPTIONS.md)を参照してください。

## 機能互換性

完全かつ最も正確な機能マトリックスについては、[メインREADME](https://github.com/bghira/SimpleTuner#model-architecture-support)を参照してください。

## モデルクイックスタートガイド

| モデル | パラメータ数 | PEFT LoRA | Lycoris | Full-Rank | 量子化 | 混合精度 | Grad Checkpoint | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | ライセンス | 商用利用 | ガイド |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 オプション | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 条件付き<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | 条件付き<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | 条件付き<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 条件付き<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 条件付き<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | いいえ<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✗ | ✓* | int8 オプション | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | 条件付き<sup>6</sup> | [KREA2.md](quickstart/KREA2.ja.md) |
| Mage-Flow | 4B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [MAGEFLOW.md](quickstart/MAGEFLOW.ja.md) |
| Boogu-Image 0.1 | - | ✓ | ✓ | ✓* | fp8 オプション | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.ja.md) |
| zlab i1 | 3B | ✓ | ✓ | ✓ | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | 条件付き<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.ja.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 デフォルト、nf4 オプション | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | いいえ<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.ja.md) |
| ERNIE-Image | - | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [ERNIE.md](quickstart/ERNIE.ja.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | はい | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | 条件付き<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 オプション | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 条件付き<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | 非推奨 | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | はい<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓ | ✓* | no_change first | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | はい | [COSMOS3.md](quickstart/COSMOS3.ja.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | 条件付き<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | 条件付き<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | 条件付き<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [SANAVIDEO.md](quickstart/SANAVIDEO.ja.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **必須** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **必須** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | 非対応 | fp32 (必須) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | いいえ<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | はい | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 オプション | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = サポート、✓* = Full-RankにはDeepSpeed/FSDP2が必要、✗ = 非サポート、`✓+`はVRAMプレッシャーによりチェックポイントが推奨されることを示します。Ref Inputs は既存の参照/編集/I2V 条件パスのみを示し、`opt` は任意、`req` は編集/I2V flavour で必須であることを示します。TwinFlow ✓は`twinflow_enabled=true`のときのネイティブサポートを意味します（拡散モデルには`diff2flow_enabled+twinflow_allow_diff2flow`が必要）。Self-Flow ✓は`crepa_enabled=true`、`crepa_feature_source=self_flow`、`use_ema=true`、および`crepa_teacher_block_index`設定時のネイティブサポートを意味します。LayerSync ✓はバックボーンがセルフアライメント用のトランスフォーマー隠れ状態を公開していることを意味し、✗はそのバッファを持たないUNetスタイルのバックボーンを示します。†SlidersはLoRAおよびLyCORIS（Full-Rank LyCORIS "full"を含む）に適用されます。*

**ライセンス注記:** 商用利用の表示はモデル重み、派生チェックポイント、fine-tune、ホスト型モデル利用を対象にしています。生成出力の権利は異なる場合があります。商用展開前にリンク先のライセンス本文を確認してください。

<sup>1</sup> OpenRAIL 系ライセンスは通常、商用利用を許可しますが、モデルと派生物には利用制限が残ります。

<sup>2</sup> Stability AI Community License は収益しきい値未満の対象ユーザー向けです。より大きな商用利用には Stability のエンタープライズ条件が必要です。

<sup>3</sup> Flux.1 は flavour により異なります。Schnell と LibreFlux は Apache-2.0、Dev、Krea、Kontext は BFL の非商用条件です。FluxBooru は商用利用前に upstream metadata を確認してください。

<sup>4</sup> Flux.2 は flavour により異なります。Klein 4B は Apache-2.0、Dev と Klein 9B は BFL の非商用条件です。

<sup>5</sup> 公開されている非商用モデル条件では、別ライセンスなしに重み、派生チェックポイント、ホスト型モデルサービスを商用利用できません。

<sup>6</sup> Krea 2 Community License は収益および安全性/フィルタリング要件を満たす場合のみ商用利用を許可します。それ以外はエンタープライズライセンスが必要です。

<sup>7</sup> Kolors のモデルまたは派生物の商用利用には、ライセンサーへの申請と明示的な許可が必要です。

<sup>8</sup> AuraFlow は Apache-2.0 の upstream flavour と、別のカスタムライセンスを持つ Pony flavour をサポートします。選択した flavour を確認してください。

<sup>9</sup> NVIDIA Open Model License は商用利用を許可しますが、契約、利用許諾ポリシー、輸出管理条件を含みます。

<sup>10</sup> LTX Video 0.9.5 は OpenRAIL-M、LTX Video 2 は商用利用に収益しきい値がある LTX community terms を使用します。

<sup>11</sup> Tencent Hunyuan Community License には地域除外と、非常に大規模なサービス向けの商用しきい値があります。

<sup>12</sup> この mirror は標準ライセンス本文なしで `license: other` を公開しています。商用利用前に upstream terms を確認してください。

> ℹ️ Wanクイックスタートには2.1 + 2.2ステージプリセットと時間埋め込みトグルが含まれます。Flux Kontextは、Flux.1をベースに構築された編集ワークフローをカバーします。

> ⚠️ これらのクイックスタートは生きたドキュメントです。新しいモデルの登場やトレーニングレシピの改善に伴い、時折更新されることがあります。

### 高速パス: Z-Image TurboとFlux Schnell

- **Z-Image Turbo**: TREADを使用した完全サポートのLoRA。量子化なし（int8も動作）でもNVIDIAとmacOSで高速に動作します。多くの場合、ボトルネックはトレーナーのセットアップだけです。
- **Flux Schnell**: クイックスタート設定が高速ノイズスケジュールとアシスタントLoRAスタックを自動的に処理します。Schnell LoRAをトレーニングするための追加フラグは不要です。

### 高度な実験的機能

- **Diff2Flow**: 標準的なepsilon/v-predictionモデル（SD1.5、SDXL、DeepFloydなど）をFlow Matching損失目的関数を使用してトレーニングできます。これにより、古いアーキテクチャと最新のフローベースのトレーニング間のギャップを埋めます。
- **Scheduled Sampling**: トレーニング中にモデル自身に中間ノイズ潜在変数を生成させる（「ロールアウト」）ことで露出バイアスを軽減します。これにより、モデルが自身の生成エラーから回復する方法を学習できます。

## よくある問題

### データセットのサンプル数が予想より少ない

データセットの使用可能なサンプル数が予想より少ない場合、処理中にファイルがフィルタされた可能性があります。一般的な理由：

- **ファイルが小さすぎる**: `minimum_image_size` 未満の画像はフィルタされます
- **アスペクト比が範囲外**: `minimum_aspect_ratio`/`maximum_aspect_ratio` の範囲外の画像は除外されます
- **時間制限**: 時間制限を超えるオーディオ/ビデオファイルはスキップされます

**フィルタリング統計の確認:**
- WebUI でデータセットディレクトリを参照し、選択するとフィルタリング統計が表示されます
- データセット処理中のログで次のような統計を確認: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

詳細なトラブルシューティングについては、データローダードキュメントの[フィルタされたデータセットのトラブルシューティング](DATALOADER.ja.md)を参照してください。
