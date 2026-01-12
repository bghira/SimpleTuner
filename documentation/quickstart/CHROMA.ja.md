# Chroma 1 クイックスタート

![image](https://github.com/user-attachments/assets/3c8a12c6-9d45-4dd4-9fc8-6b7cd3ed51dd)

Chroma 1は8.9BパラメータのFlux.1 Schnellのトリミングバリアントで、Lodestone Labsによってリリースされました。このガイドでは、SimpleTunerでLoRAトレーニングを設定する方法を説明します。

## ハードウェア要件

パラメータ数は少ないですが、メモリ使用量はFlux Schnellに近いです:

- ベーストランスフォーマーの量子化には、依然として**約40〜50 GB**のシステムRAMを使用する可能性があります。
- Rank-16 LoRAトレーニングは通常以下を消費します:
  - ベース量子化なしで約28 GB VRAM
  - int8 + bf16で約16 GB VRAM
  - int4 + bf16で約11 GB VRAM
  - NF4 + bf16で約8 GB VRAM
- 現実的なGPU最小要件: **RTX 3090 / RTX 4090 / L40S**クラスのカード以上。
- **Apple M-series（MPS）**でLoRAトレーニング、AMD ROCmでも良好に動作します。
- フルランクファインチューニングには80 GBクラスのアクセラレータまたはマルチGPUセットアップを推奨します。

## 前提条件

ChromaはFluxガイドと同じランタイム要件を共有しています:

- Python **3.10 – 3.12**
- サポートされているアクセラレータバックエンド（CUDA、ROCm、またはMPS）

Pythonバージョンを確認:

```bash
python3 --version
```

SimpleTunerをインストール（CUDAの例）:

```bash
pip install 'simpletuner[cuda]'
```

バックエンド固有のセットアップの詳細（CUDA、ROCm、Apple）については、[インストールガイド](../INSTALL.md)を参照してください。

## Web UIの起動

```bash
simpletuner server
```

UIはhttp://localhost:8001で利用可能になります。

## CLI経由の設定

`simpletuner configure`でコア設定を順を追って説明します。Chromaのキー値は:

- `model_type`: `lora`
- `model_family`: `chroma`
- `model_flavour`: 以下のいずれか
  - `base`（デフォルト、バランスの取れた品質）
  - `hd`（より高い忠実度、より多くの計算を消費）
  - `flash`（高速だが不安定 - 本番環境には推奨されません）
- `pretrained_model_name_or_path`: 上記のフレーバーマッピングを使用するために空のままにしてください
- `model_precision`: デフォルトの`bf16`を維持
- `flux_fast_schedule`: **無効**のまま; Chromaには独自の適応サンプリングがあります

### 手動設定スニペットの例

<details>
<summary>設定例を表示</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "chroma",
  "model_flavour": "base",
  "output_dir": "/workspace/chroma-output",
  "network_rank": 16,
  "learning_rate": 2.0e-4,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "pretrained_model_name_or_path": null
}
```
</details>

> ⚠️ 地域でHugging Faceアクセスが遅い場合は、起動前に`HF_ENDPOINT=https://hf-mirror.com`をエクスポートしてください。

### 高度な実験的機能

<details>
<summary>高度な実験的詳細を表示</summary>


SimpleTunerには、トレーニングの安定性とパフォーマンスを大幅に向上させることができる実験的機能が含まれています。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** トレーニング中にモデルが独自の入力を生成できるようにすることで、露出バイアスを減らし、出力品質を向上させます。

> ⚠️ これらの機能はトレーニングの計算オーバーヘッドを増加させます。

</details>

## データセットとデータローダー

ChromaはFluxと同じデータローダー形式を使用します。データセットの準備とプロンプトライブラリについては、[一般チュートリアル](../TUTORIAL.md)または[Web UIチュートリアル](../webui/TUTORIAL.md)を参照してください。

## Chroma固有のトレーニングオプション

- `flux_lora_target`: LoRAアダプターを受け取るトランスフォーマーモジュールを制御します（`all`、`all+ffs`、`context`、`tiny`など）。デフォルトはFluxをミラーしており、ほとんどの場合に適しています。
- `flux_guidance_mode`: `constant`がうまく機能します。Chromaはガイダンス範囲を公開していません。
- アテンションマスキングは常に有効です - テキスト埋め込みキャッシュがパディングマスク付きで生成されていることを確認してください（現在のSimpleTunerリリースのデフォルト動作）。
- スケジュールシフトオプション（`flow_schedule_shift` / `flow_schedule_auto_shift`）はChromaには必要ありません - ヘルパーが既にテールタイムステップを自動的にブーストします。
- `flux_t5_padding`: マスキング前にパディングされたトークンをゼロにしたい場合は`zero`に設定します。

## 自動テールタイムステップサンプリング

Fluxは、高ノイズ/低ノイズの極端値をアンダーサンプリングするlog-normalスケジュールを使用していました。Chromaのトレーニングヘルパーは、サンプリングされたシグマにquadratic（`σ ↦ σ²` / `1-(1-σ)²`）リマッピングを適用し、テール領域がより頻繁に訪問されるようにします。これには**追加の設定は必要ありません** - `chroma`モデルファミリに組み込まれています。

## 検証とサンプリングのヒント

- `validation_guidance_real`はパイプラインの`guidance_scale`に直接マッピングされます。シングルパスサンプリングの場合は`1.0`のままにするか、検証レンダリング中にclassifier-free guidanceが必要な場合は`2.0`〜`3.0`に上げます。
- クイックプレビューには20推論ステップを使用し、より高品質には28〜32を使用します。
- ネガティブプロンプトはオプションのまま; ベースモデルは既にde-distilledされています。
- 現時点ではtext-to-imageのみをサポートしています。img2imgサポートは今後のアップデートで追加される予定です。

## トラブルシューティング

- **起動時のOOM**: `offload_during_startup`を有効にするか、ベースモデルを量子化します（`base_model_precision: int8-quanto`）。
- **トレーニングが早期に発散**: Gradient Checkpointingがオンになっていることを確認し、`learning_rate`を`1e-4`に下げ、キャプションが多様であることを確認します。
- **検証が同じポーズを繰り返す**: プロンプトを長くしてください。フローマッチングモデルはプロンプトの多様性が低いと崩壊します。

高度なトピック（DeepSpeed、FSDP2、評価メトリクス）については、README全体にリンクされている共有ガイドを参照してください。
