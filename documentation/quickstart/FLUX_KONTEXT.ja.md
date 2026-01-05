# Kontext \[dev] ミニクイックスタート

> 📝  Kontextはトレーニングワークフローの90%をFluxと共有しているため、このファイルでは*異なる点*のみを記載しています。ここで言及されていないステップについては、元の[手順](../quickstart/FLUX.md)に従ってください


---

## 1. モデル概要

|                                                  | Flux-dev               | Kontext-dev                                 |
| ------------------------------------------------ | -------------------    | ------------------------------------------- |
| ライセンス                                          | 非商用         | 非商用                              |
| ガイダンス                                         | ディスティル (CFG ≈ 1)     | ディスティル (CFG ≈ 1)                         |
| 利用可能なバリアント                               | *dev*, schnell,\[pro]   | *dev*, \[pro, max]                          |
| T5シーケンス長                               | 512 dev, 256 schnell   | 512 dev                                     |
| 一般的な1024 px推論時間<br>(4090 @ CFG 1)  | ≈ 20 s                  | **≈ 80 s**                                  |
| 1024 px LoRA @ int8-quantoのVRAM               | 18 G                   | **24 G**                                    |

KontextはFluxトランスフォーマーバックボーンを維持しながら、**ペア参照コンディショニング**を導入しています。

Kontextでは2つの`conditioning_type`モードが利用可能です:

* `conditioning_type=reference_loose` (✅ 安定) – 参照は編集とアスペクト比/サイズが異なっていても可能。
  - 両方のデータセットがメタデータスキャンされ、アスペクトバケット化され、互いに独立してクロップされるため、起動時間が大幅に増加する可能性があります。
  - 編集画像と参照画像のアライメントを確保したい場合(1つのファイル名につき1つの画像を使用するデータローダーなど)、これは問題になる可能性があります。
* `conditioning_type=reference_strict` (✅ 安定) – 参照は編集クロップとまったく同じように事前変換されます。
  - 編集画像と参照画像間のクロップ/アスペクトバケット化を完璧にアライメントする必要がある場合は、このようにデータセットを設定する必要があります。
  - 元々`--vae_cache_ondemand`といくらかのVRAM使用量の増加が必要でしたが、現在は必要ありません。
  - 起動時にソースデータセットからクロップ/アスペクトバケットメタデータを複製するため、手動で行う必要はありません。

フィールド定義については、[`conditioning_type`](../DATALOADER.md#conditioning_type)と[`conditioning_data`](../DATALOADER.md#conditioning_data)を参照してください。複数のコンディショニングセットのサンプリング方法を制御するには、[OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling)で説明されている`conditioning_multidataset_sampling`を使用してください。


---

## 2. ハードウェア要件

* **システムRAM**: 量子化にはまだ50 GBが必要です。
* **GPU**: 1024 pxトレーニングには、**int8-quanto**で3090(24 G)が現実的な最小要件です。
  * Flash Attention 3を備えたHopper H100/H200システムでは、`--fuse_qkv_projections`を有効にしてトレーニングを大幅に高速化できます。
  * 512 pxでトレーニングする場合は12 Gカードにも収まりますが、バッチが遅くなることを覚悟してください(シーケンス長は大きいままです)。


---

## 3. クイック設定差分

以下は、通常のFluxトレーニング設定と比較して、`config/config.json`で必要な*最小限*の変更セットです。

<details>
<summary>設定例を表示</summary>

```jsonc
{
  "model_family":   "flux",
  "model_flavour": "kontext",                       // <-- これを"dev"から"kontext"に変更
  "base_model_precision": "int8-quanto",            // 1024 pxで24 Gに収まる
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <-- Hopper H100/H200システムでトレーニングを高速化するために使用。警告: flash-attnの手動インストールが必要。
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                       // <-- より速い結果にはLionを使用し、より遅いが安定する可能性のある結果にはadamw_bf16を使用。
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <-- kontextは本当にガイダンス値2.5で最良の結果を出す
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"    // <-- 2つのコンディショニングデータセットが定義されている場合、これを"combined"に設定すると、切り替える代わりに同時に表示される
}
```
</details>

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

### データローダースニペット(マルチデータバックエンド)

画像ペアデータセットを手動でキュレーションした場合、2つの別々のディレクトリを使用して設定できます: 1つは編集画像用、もう1つは参照画像用。

編集データセットの`conditioning_data`フィールドは参照データセットの`id`を指す必要があります。

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/edited-images",   // <-- VAE出力が保存される場所
    "instance_data_dir": "/datasets/edited-images",             // <-- 絶対パスを使用
    "conditioning_data": [
      "my-reference-images"                                     // <-- これは参照セットの"id"である必要があります
                                                                // 2つ目のセットを指定して交互にまたは組み合わせることもできます。例: ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                              // <-- これらのキャプションには編集指示が含まれている必要があります
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/ref-images",      // <-- VAE出力が保存される場所。他のデータセットのVAEパスと異なる必要があります。
    "instance_data_dir": "/datasets/reference-images",          // <-- 絶対パスを使用
    "conditioning_type": "reference_strict",                    // <-- これをreference_looseに設定すると、画像は編集画像とは独立してクロップされます
    "resolution": 1024,
    "caption_strategy": null,                                   // <-- 参照にはキャプションは必要ありませんが、利用可能な場合は編集キャプションの代わりに使用されます
                                                                // 注意: conditioning_multidataset_sampling=combinedを使用する場合、別のコンディショニングキャプションを定義することはできません。
                                                                // 編集データセットのキャプションのみが使用されます。
  }
]
```
</details>

> caption_strategyのオプションと要件については、[DATALOADER.md](../DATALOADER.md#caption_strategy)を参照してください。

*すべての編集画像は、両方のデータセットフォルダで1対1で一致するファイル名と拡張子を持つ必要があります。SimpleTunerは参照エンベッディングを編集のコンディショニングに自動的にステープルします。

準備された例として、[Kontext Max派生デモデータセット](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol)があり、参照画像と編集画像、およびそのキャプションテキストファイルが含まれています。セットアップ方法をより良く理解するために閲覧できます。

### 専用の検証スプリットのセットアップ

以下は、200,000サンプルのトレーニングセットと少数の検証セットを使用する設定例です。

`config.json`に以下を追加する必要があります:

<details>
<summary>設定例を表示</summary>

```json
{
  "eval_dataset_id": "edited-images",
}
```
</details>

`multidatabackend.json`では、`edited-images`と`reference-images`は通常のトレーニングスプリットと同じレイアウトの検証データを含む必要があります。

<details>
<summary>設定例を表示</summary>

```json
[
    {
        "id": "edited-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/edited-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "textfile",
        "cache_dir_vae": "cache/vae/flux-edit",
        "vae_cache_clear_each_epoch": false,
        "conditioning_data": ["reference-images"]
    },
    {
        "id": "reference-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/reference-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": null,
        "cache_dir_vae": "cache/vae/flux-ref",
        "vae_cache_clear_each_epoch": false,
        "conditioning_type": "reference_strict"
    },
    {
        "id": "subjects200k-left",
        "disabled": false,
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "conditioning_data": ["subjects200k-right"],
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            }
        }
    },
    {
        "id": "subjects200k-right",
        "disabled": false,
        "type": "huggingface",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "subjects200k-left",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            }
        }
    },

    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```
</details>

### 自動参照-編集ペア生成

既存の参照-編集ペアがない場合、SimpleTunerは単一のデータセットから自動的にそれらを生成できます。これは特に以下のモデルのトレーニングに便利です:
- 画像強調/超解像
- JPEG圧縮アーティファクト除去
- ぼかし除去
- その他の修復タスク

#### 例: ぼかし除去トレーニングデータセット

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

この設定は以下を行います:
1. 高品質のシャープな画像からぼかしバージョンを作成(これらが「参照」画像になります)
2. 元の高品質画像をトレーニング損失のターゲットとして使用
3. 低品質の参照画像を強調/ぼかし除去するようにKontextをトレーニング

> **注意**: `conditioning_multidataset_sampling=combined`を使用する場合、コンディショニングデータセットに`captions`を定義することはできません。代わりに編集データセットのキャプションが使用されます。

#### 例: JPEG圧縮アーティファクト除去

<details>
<summary>設定例を表示</summary>

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

#### 重要な注意事項

1. **生成は起動時に行われます**: 劣化バージョンはトレーニング開始時に自動的に作成されます
2. **キャッシング**: 生成された画像は保存されるため、後続の実行では再生成されません
3. **キャプション戦略**: コンディショニング設定の`captions`フィールドは、一般的な画像説明よりもうまく機能するタスク固有のプロンプトを提供します
4. **パフォーマンス**: これらのCPUベースのジェネレーター(ぼかし、JPEG)は高速で、複数のプロセスを使用します
5. **ディスク容量**: 生成された画像は大きくなる可能性があるため、十分なディスク容量を確保してください!残念ながら、オンデマンドで作成する機能はまだありません。

その他のコンディショニングタイプと高度な設定については、[ControlNetドキュメント](../CONTROLNET.md)を参照してください。

---

## 4. Kontext固有のトレーニングのヒント

1. **より長いシーケンス → より遅いステップ。**  1024 px、rank-1 LoRA、bf16 + int8で単一の4090で\~0.4 it/sを期待してください。
2. **正しい設定を探索してください。**  Kontextのファインチューニングについてはあまり知られていません。安全のため、`1e-5`(Lion)または`5e-4`(AdamW)にとどまってください。
3. **VAEキャッシング中のVRAMスパイクに注意してください。**  OOMになった場合は、`--offload_during_startup=true`を追加するか、`resolution`を下げるか、`config.json`経由でVAEタイリングを有効にしてください。
4. **参照画像なしでトレーニングできますが、現在SimpleTuner経由では不可能です。**  現在、条件付き画像の提供が必要なようにハードコードされていますが、編集ペアと一緒に通常のデータセットを提供することで、被写体や類似性を学習させることができます。
5. **ガイダンス再ディスティレーション。**  Flux-devと同様に、Kontext-devはCFGディスティルされています。多様性が必要な場合は、`validation_guidance_real > 1`で再トレーニングし、推論時にAdaptive-Guidanceノードを使用してください。ただし、これは収束にはるかに長い時間がかかり、大きなrank LoRAまたはLycoris LoKrが成功するために必要です。
6. **フルランクトレーニングはおそらく時間の無駄です。** Kontextはローランクでトレーニングされるように設計されており、フルランクトレーニングはLycoris LoKrよりも良い結果をもたらす可能性は低いです。LoKrは通常、最良のパラメータを追求する作業が少なくて済み、標準LoRAよりも優れた性能を発揮します。それでも試したい場合は、DeepSpeedを使用する必要があります。
7. **トレーニングに2つ以上の参照画像を使用できます。** 例として、2つの被写体を1つのシーンに挿入するための被写体-被写体-シーン画像がある場合、すべての関連画像を参照入力として提供できます。フォルダ間でファイル名がすべて一致していることを確認してください。

---

## 5. 推論の落とし穴

- トレーニングと推論の精度レベルを一致させてください。int8トレーニングはint8推論で最良の結果を出し、同様に続きます。
- 2つの画像が同時にシステムを通過するため、非常に遅くなります。4090で1024 pxの編集あたり80秒を予想してください。

---

## 6. トラブルシューティングチートシート

| 症状                                 | 可能性のある原因               | 簡単な修正                                              |
| --------------------------------------- | -------------------------- | ------------------------------------------------------ |
| 量子化中のOOM                 | **システム**RAMが不足  | `quantize_via=cpu`を使用                                 |
| 参照画像が無視される/編集が適用されない     | データローダーのミスペアリング     | 同一のファイル名と`conditioning_data`フィールドを確認 |
| 正方形グリッドアーティファクト                   | 低品質編集が支配的  | より高品質なデータセットを作成、LRを下げる、Lionを避ける      |

---

## 7. 参考資料

高度なチューニングオプション(LoKr、NF4量子化、DeepSpeedなど)については、[Fluxの元のクイックスタート](../quickstart/FLUX.md)を参照してください。上記で特に記載がない限り、すべてのフラグは同じように機能します。
