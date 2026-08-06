# モデルガイド

対応する各モデルアーキテクチャのトレーニング手順ガイドです。

## 画像モデル

### Flow Matching

| モデル | パラメータ | ライセンス | 商用利用 | ガイド |
| ------- | ------------ | --- | :---: | ------- |
| **Flux.1** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 条件付き<sup>3</sup> | [Flux.1 ガイド](FLUX.md) |
| **Flux.2** | 32B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 条件付き<sup>4</sup> | [Flux.2 ガイド](FLUX2.md) |
| **Flux Kontext** | 12B | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | いいえ<sup>5</sup> | [Kontext ガイド](FLUX_KONTEXT.md) |
| **Chroma** | 8.9B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Chroma ガイド](CHROMA.md) |
| **Stable Diffusion 3** | 2-8B | [Stability AI Community](https://stability.ai/license) | 条件付き<sup>2</sup> | [SD3 ガイド](SD3.md) |
| **Auraflow** | 6.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | 条件付き<sup>8</sup> | [Auraflow ガイド](AURAFLOW.md) |
| **Sana** | 0.6-4.8B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Sana ガイド](SANA.md) |
| **Lumina2** | 2B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Lumina2 ガイド](LUMINA2.md) |
| **HiDream** | 17B MoE | [MIT](https://opensource.org/license/mit) | はい | [HiDream ガイド](HIDREAM.md) |
| **Z-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Z-Image ガイド](ZIMAGE.md) |
| **Krea2** | - | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | はい<sup>6</sup> | [Krea2 ガイド](KREA2.ja.md) |
| **Mage-Flow** | 4B | [MIT](https://opensource.org/license/mit) | はい | [Mage-Flow ガイド](MAGEFLOW.ja.md) |
| **Boogu-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Boogu-Image ガイド](BOOGU_IMAGE.ja.md) |
| **zlab i1** | 3B | [MIT](https://opensource.org/license/mit) | はい | [zlab i1 ガイド](ZLAB_i1.ja.md) |
| **Ideogram 4** | 9B | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | いいえ<sup>5</sup> | [Ideogram 4 ガイド](IDEOGRAM4.ja.md) |
| **ERNIE-Image** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [ERNIE ガイド](ERNIE.md) |

### DiT / Transformer

| モデル | パラメータ | ライセンス | 商用利用 | ガイド |
| ------- | ------------ | --- | :---: | ------- |
| **PixArt Sigma** | 0.6-0.9B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 条件付き<sup>1</sup> | [Sigma ガイド](SIGMA.md) |
| **Cosmos2** | 2-14B | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | はい<sup>9</sup> | [Cosmos2 ガイド](COSMOS2IMAGE.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | はい | [Cosmos3 ガイド](COSMOS3.ja.md) |
| **OmniGen** | 3.8B | [MIT](https://opensource.org/license/mit) | はい | [OmniGen ガイド](OMNIGEN.md) |
| **Qwen Image** | 20B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Qwen ガイド](QWEN_IMAGE.md) |
| **LongCat Image** | 6B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [LongCat ガイド](LONGCAT_IMAGE.md) |
| **Kandinsky 5** | - | [MIT](https://opensource.org/license/mit) | はい | [Kandinsky ガイド](KANDINSKY5_IMAGE.md) |

### U-Net

| モデル | パラメータ | ライセンス | 商用利用 | ガイド |
| ------- | ------------ | --- | :---: | ------- |
| **Stable Diffusion XL** | 3.5B | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | 条件付き<sup>1</sup> | [SDXL ガイド](SDXL.md) |
| **Kolors** | 5B | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Abandonware<sup>7</sup> | [Kolors ガイド](KOLORS.md) |
| **Stable Cascade** | - | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | Abandonware<sup>7</sup> | [Cascade ガイド](STABLE_CASCADE_C.md) |

### 画像編集

| モデル | ライセンス | 商用利用 | ガイド |
| ------- | --- | :---: | ------- |
| **Qwen Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Qwen Edit ガイド](QWEN_EDIT.md) |
| **LongCat Edit** | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [LongCat Edit ガイド](LONGCAT_EDIT.md) |

## 動画モデル

| モデル | パラメータ | ライセンス | 商用利用 | ガイド |
| ------- | ------------ | --- | :---: | ------- |
| **Wan Video** | 1.3-14B | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Wan ガイド](WAN.md) |
| **LTX Video** | 5B | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | 条件付き<sup>10</sup> | [LTX ガイド](LTXVIDEO.md) |
| **LTX Video 2** | 19B | [LTX-2 Community](https://ltx.io/model/license) | 条件付き<sup>10</sup> | [LTX Video 2 ガイド](LTXVIDEO2.md) |
| **Cosmos3** | 4-65B | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | はい | [Cosmos3 ガイド](COSMOS3.ja.md) |
| **Hunyuan Video** | 8.3B | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | 条件付き<sup>11</sup> | [Hunyuan ガイド](HUNYUANVIDEO.md) |
| **MiniMax H3** | 33B | [MiniMax H3 Community](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | 条件付き<sup>12</sup> | [MiniMax H3 ガイド](MINIMAX_H3.ja.md) |
| **Sana Video** | - | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | はい | [Sana Video ガイド](SANAVIDEO.md) |
| **Kandinsky 5 Video** | - | [MIT](https://opensource.org/license/mit) | はい | [Kandinsky Video ガイド](KANDINSKY5_VIDEO.md) |
| **LongCat Video** | - | [MIT](https://opensource.org/license/mit) | はい | [LongCat Video ガイド](LONGCAT_VIDEO.md) |
| **LongCat Video Edit** | - | [MIT](https://opensource.org/license/mit) | はい | [LongCat Video Edit ガイド](LONGCAT_VIDEO_EDIT.md) |

**ライセンス注記:** 商用利用の表示はモデル重み、派生チェックポイント、fine-tune、ホスト型モデル利用を対象にしています。生成出力の権利は異なる場合があります。商用展開前にリンク先のライセンス本文を確認してください。

<sup>1</sup> OpenRAIL 系ライセンスは通常、商用利用を許可しますが、モデルと派生物には利用制限が残ります。

<sup>2</sup> Stability AI Community License は収益しきい値未満の対象ユーザー向けです。より大きな商用利用には Stability のエンタープライズ条件が必要です。

<sup>3</sup> Flux.1 は flavour により異なります。Schnell と LibreFlux は Apache-2.0、Dev、Krea、Kontext は BFL の非商用条件です。FluxBooru は商用利用前に upstream metadata を確認してください。

<sup>4</sup> Flux.2 は flavour により異なります。Klein 4B は Apache-2.0、Dev と Klein 9B は BFL の非商用条件です。

<sup>5</sup> 公開されている非商用モデル条件では、別ライセンスなしに重み、派生チェックポイント、ホスト型モデルサービスを商用利用できません。

<sup>6</sup> Krea 2 Community License は、収益上限（年間 $1M 未満）と安全性/フィルタリング要件を満たす場合に商用利用を許可します。それ以外はエンタープライズライセンスが必要です。

<sup>7</sup> Abandonware は、元のベンダーが実質的にモデルを放置しており、許可を得る信頼できる経路がないことを意味します。そのリスクを受け入れるかはエンドユーザーの判断です。

<sup>8</sup> AuraFlow は Apache-2.0 の upstream flavour と、別のカスタムライセンスを持つ Pony flavour をサポートします。選択した flavour を確認してください。

<sup>9</sup> NVIDIA Open Model License は商用利用を許可しますが、契約、利用許諾ポリシー、輸出管理条件を含みます。

<sup>10</sup> LTX Video 0.9.5 は OpenRAIL-M、LTX Video 2 は商用利用に収益しきい値がある LTX community terms を使用します。

<sup>11</sup> Tencent Hunyuan Community License には地域除外と、非常に大規模なサービス向けの商用しきい値があります。

<sup>12</sup> MiniMax H3 Community License は標準の適用地域から米国、欧州連合、英国、韓国を除外しており、これらの地域では別途認可が必要です。


## 音声モデル

| モデル | サイズ / バージョン | ガイド |
|-------|------------------------|-------|
| **ACE-Step** | 3.5B / 1.5 | [ACE-Step ガイド](ACE_STEP.md) |
| **HeartMuLa** | 3B | [HeartMuLa ガイド](HEARTMULA.md) |

## モデルの選び方

**初心者向け:**

- 高品質な画像生成には **Flux.1** から始めてください
- VRAM 要件を下げるには **LoRA** トレーニングを使ってください

**プロダクション向け:**

- 幅広い互換性には **SD3** または **SDXL**
- 最高品質には **Flux.2**（より多くの VRAM が必要）

**動画向け:**

- 品質とリソースのバランスには **Wan Video**
- I2V と超解像には **Hunyuan Video**

**用途別:**

- 画像編集/コンディショニングには **Flux Kontext**
- テキストから音楽の LoRA 学習には **ACE-Step**（v1 / v1.5）
- 自己回帰のテキストから音声には **HeartMuLa**
