# SimpleTuner 💹

> ℹ️ オプトイン設定の `report_to`、`push_to_hub`、または手動で設定されたwebhook以外、いかなる第三者にもデータは送信されません。

**SimpleTuner**は、シンプルさに重点を置いており、コードを理解しやすくすることに焦点を当てています。このコードベースは共有の学術的な取り組みとして機能し、貢献を歓迎します。

コミュニティに参加したい場合は、Terminus Research Groupを通じて[Discord](https://discord.gg/JGkSwEbjRb)で見つけることができます。
ご質問がある場合は、お気軽にそちらでお問い合わせください。

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## 目次

- [設計思想](#設計思想)
- [チュートリアル](#チュートリアル)
- [機能](#機能)
  - [コアトレーニング機能](#コアトレーニング機能)
  - [モデルアーキテクチャサポート](#モデルアーキテクチャサポート)
  - [高度なトレーニング技術](#高度なトレーニング技術)
  - [モデル固有の機能](#モデル固有の機能)
  - [クイックスタートガイド](#クイックスタートガイド)
- [ハードウェア要件](#ハードウェア要件)
- [ツールキット](#ツールキット)
- [セットアップ](#セットアップ)
- [トラブルシューティング](#トラブルシューティング)

## 設計思想

- **シンプルさ**: ほとんどのユースケースに適した優れたデフォルト設定を目指し、調整の手間を減らします。
- **汎用性**: 小規模なデータセットから大規模なコレクションまで、幅広い画像数量を処理できるように設計されています。
- **最先端の機能**: 実証済みの効果がある機能のみを組み込み、未検証のオプションの追加を避けます。

## チュートリアル

[新しいWeb UIチュートリアル](/documentation/webui/TUTORIAL.md)または[クラシックなコマンドラインチュートリアル](/documentation/TUTORIAL.md)を始める前に、このREADMEを完全に確認してください。このドキュメントには、最初に知っておく必要がある重要な情報が含まれています。

完全なドキュメントを読んだりWebインターフェースを使用したりせずに手動で設定するクイックスタートについては、[クイックスタート](/documentation/QUICKSTART.md)ガイドを使用できます。

メモリに制約のあるシステムの場合は、[DeepSpeedドキュメント](/documentation/DEEPSPEED.md)を参照してください。これは、Microsoftの🤗AccelerateとDeepSpeedを使用してオプティマイザー状態のオフロードを設定する方法を説明しています。DTensorベースのシャーディングとコンテキスト並列化については、SimpleTuner内の新しいFullyShardedDataParallel v2ワークフローをカバーする[FSDP2ガイド](/documentation/FSDP2.md)をお読みください。

マルチノード分散トレーニングの場合、[このガイド](/documentation/DISTRIBUTED.md)は、INSTALLおよびクイックスタートガイドの設定をマルチノードトレーニングに適したものに調整し、数十億のサンプルを含む画像データセットに最適化するのに役立ちます。

---

## 機能

SimpleTunerは、一貫した機能可用性を持つ複数の拡散モデルアーキテクチャにわたる包括的なトレーニングサポートを提供します:

### コアトレーニング機能

- **ユーザーフレンドリーなWeb UI** - 洗練されたダッシュボードを通じてトレーニングライフサイクル全体を管理
- **マルチモーダルトレーニング** - **画像、動画、音声**生成モデルのための統合パイプライン
- **マルチGPUトレーニング** - 自動最適化による複数GPU間の分散トレーニング
- **高度なキャッシング** - より高速なトレーニングのために、画像、動画、音声、キャプション埋め込みをディスクにキャッシュ
- **アスペクト比バケッティング** - さまざまな画像/動画サイズとアスペクト比のサポート
- **コンセプトスライダー** - LoRA/LyCORIS/full（LyCORIS `full`経由）のスライダー対応ターゲティング、ポジティブ/ネガティブ/ニュートラルサンプリング、プロンプトごとの強度；[Slider LoRAガイド](/documentation/SLIDER_LORA.md)を参照
- **メモリ最適化** - ほとんどのモデルが24G GPUでトレーニング可能、多くは最適化により16Gで可能
- **DeepSpeed & FSDP2統合** - optim/grad/parameterシャーディング、コンテキスト並列アテンション、勾配チェックポイント、オプティマイザー状態オフロードにより、より小さなGPUで大規模モデルをトレーニング
- **S3トレーニング** - クラウドストレージ（Cloudflare R2、Wasabi S3）から直接トレーニング
- **EMAサポート** - 安定性と品質向上のための指数移動平均重み
- **カスタム実験トラッカー** - `accelerate.GeneralTracker`を`simpletuner/custom-trackers`にドロップし、`--report_to=custom-tracker --custom_tracker=<name>`を使用

### マルチユーザー&エンタープライズ機能

SimpleTunerには、エンタープライズグレードの機能を備えた完全なマルチユーザートレーニングプラットフォームが含まれています—**永久に無料でオープンソース**。

- **ワーカーオーケストレーション** - 中央パネルに自動接続し、SSE経由でジョブディスパッチを受信する分散GPUワーカーを登録；エフェメラル（クラウド起動）および永続的（常時稼働）ワーカーをサポート；[ワーカーオーケストレーションガイド](/documentation/experimental/server/WORKERS.md)を参照
- **SSO統合** - LDAP/Active DirectoryまたはOIDCプロバイダー（Okta、Azure AD、Keycloak、Google）で認証；[外部認証ガイド](/documentation/experimental/server/EXTERNAL_AUTH.md)を参照
- **ロールベースアクセス制御** - 4つのデフォルトロール（Viewer、Researcher、Lead、Admin）と17以上の詳細な権限；globパターンでリソースルールを定義し、チームごとに設定、ハードウェア、またはプロバイダーを制限
- **組織&チーム** - 上限ベースのクォータを持つ階層的マルチテナント構造；組織の制限は絶対最大値を強制し、チームの制限は組織の境界内で動作
- **クォータ&支出制限** - 組織、チーム、またはユーザースコープでコスト上限（日次/月次）、ジョブ同時実行数制限、送信レート制限を強制；アクションにはブロック、警告、または承認要求が含まれる
- **優先度付きジョブキュー** - チーム間の公平な共有スケジューリング、長時間待機ジョブの飢餓防止、管理者の優先度オーバーライドを備えた5つの優先度レベル（Low → Critical）
- **承認ワークフロー** - コストしきい値を超えるジョブ、初回ユーザー、または特定のハードウェアリクエストに対して承認をトリガーする設定可能なルール；UI、API、またはメール返信経由で承認
- **メール通知** - ジョブステータス、承認リクエスト、クォータ警告、完了アラートのためのSMTP/IMAP統合
- **APIキー&スコープ権限** - CI/CDパイプライン用に有効期限と制限されたスコープを持つAPIキーを生成
- **監査ログ** - コンプライアンスのためのチェーン検証ですべてのユーザーアクションを追跡；[監査ガイド](/documentation/experimental/server/AUDIT.md)を参照

デプロイメントの詳細については、[エンタープライズガイド](/documentation/experimental/server/ENTERPRISE.md)を参照してください。

### モデルアーキテクチャサポート

| モデル | パラメータ数 | PEFT LoRA | Lycoris | Full-Rank | ControlNet | 量子化 | Flow Matching | テキストエンコーダー |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **LTX Video 2** | 19B | ✓ | ✓ | ✓* | ✗ | int8/fp8 | ✓ | Gemma3 |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = サポート, ✗ = 非サポート, * = Full-rankトレーニングにDeepSpeedが必要*

### 高度なトレーニング技術

- **TREAD** - Kontextトレーニングを含む、transformerモデル用のトークンワイズドロップアウト
- **マスクロストレーニング** - セグメンテーション/深度ガイダンスによる優れた収束
- **事前正則化** - キャラクター一貫性のためのトレーニング安定性の向上
- **勾配チェックポイント** - メモリ/速度最適化のための設定可能な間隔
- **損失関数** - スケジューリングサポート付きのL2、Huber、Smooth L1
- **SNR重み付け** - トレーニングダイナミクスを改善するためのMin-SNRガンマ重み付け
- **グループオフロード** - Diffusers v0.33+モジュールグループCPU/ディスクステージング（オプションのCUDAストリーム付き）
- **検証アダプタースイープ** - 検証中に一時的にLoRAアダプター（単一またはJSONプリセット）を接続して、トレーニングループに触れることなくアダプターのみまたは比較レンダリングを測定
- **外部検証フック** - 組み込みの検証パイプラインまたはアップロード後のステップを独自のスクリプトに交換して、別のGPUでチェックを実行したり、任意のクラウドプロバイダーにアーティファクトを転送したりできます（[詳細](/documentation/OPTIONS.md#validation_method)）
- **CREPA正則化** - ビデオDiTのためのフレーム間表現アライメント（[ガイド](/documentation/experimental/VIDEO_CREPA.md)）
- **LoRA I/Oフォーマット** - 標準のDiffusersレイアウトまたはComfyUIスタイルの`diffusion_model.*`キーでPEFT LoRAをロード/保存（Flux/Flux2/Lumina2/Z-ImageはComfyUI入力を自動検出）

### モデル固有の機能

- **Flux Kontext** - Fluxモデルの編集条件付けとimage-to-imageトレーニング
- **PixArt two-stage** - PixArt Sigma用のeDiffトレーニングパイプラインサポート
- **Flow matchingモデル** - beta/uniform分布を使用した高度なスケジューリング
- **HiDream MoE** - Mixture of Expertsゲートロス増強
- **T5マスクトレーニング** - FluxおよびコンパチブルモデルのディテールアップEnhanced
- **QKVフュージョン** - メモリと速度の最適化（Flux、Lumina2）
- **TREAD統合** - ほとんどのモデルの選択的トークンルーティング
- **Wan 2.x I2V** - 高/低ステージプリセットと2.1時間埋め込みフォールバック（Wanクイックスタートを参照）
- **Classifier-free guidance** - 蒸留モデルのオプションのCFG再導入

### クイックスタートガイド

サポートされているすべてのモデルの詳細なクイックスタートガイドが利用可能です:

- **[TwinFlow Few-Step (RCGM)ガイド](/documentation/distillation/TWINFLOW.md)** - Few-step/one-step生成のためのRCGM補助損失を有効化（フローモデルまたはdiff2flow経由の拡散）
- **[Flux.1ガイド](/documentation/quickstart/FLUX.md)** - Kontext編集サポートとQKVフュージョンを含む
- **[Flux.2ガイド](/documentation/quickstart/FLUX2.md)** - **NEW!** Mistral-3テキストエンコーダーを搭載した最新の巨大なFluxモデル
- **[Z-Imageガイド](/documentation/quickstart/ZIMAGE.md)** - アシスタントアダプター + TREAD高速化を備えたBase/Turbo LoRA
- **[ACE-Stepガイド](/documentation/quickstart/ACE_STEP.md)** - **NEW!** 音声生成モデルトレーニング（text-to-music）
- **[Chromaガイド](/documentation/quickstart/CHROMA.md)** - ChromaSpecificスケジュールを持つLodestoneのflow-matching transformer
- **[Stable Diffusion 3ガイド](/documentation/quickstart/SD3.md)** - ControlNet付きのFullおよびLoRAトレーニング
- **[Stable Diffusion XLガイド](/documentation/quickstart/SDXL.md)** - 完全なSDXLトレーニングパイプライン
- **[Auraflowガイド](/documentation/quickstart/AURAFLOW.md)** - Flow-matchingモデルトレーニング
- **[PixArt Sigmaガイド](/documentation/quickstart/SIGMA.md)** - two-stageサポート付きのDiTモデル
- **[Sanaガイド](/documentation/quickstart/SANA.md)** - 軽量flow-matchingモデル
- **[Lumina2ガイド](/documentation/quickstart/LUMINA2.md)** - 2Bパラメータflow-matchingモデル
- **[Kwai Kolorsガイド](/documentation/quickstart/KOLORS.md)** - ChatGLMエンコーダー付きSDXLベース
- **[LongCat-Videoガイド](/documentation/quickstart/LONGCAT_VIDEO.md)** - Qwen-2.5-VLを使用したflow-matching text-to-videoおよびimage-to-video
- **[LongCat-Video Editガイド](/documentation/quickstart/LONGCAT_VIDEO_EDIT.md)** - Conditioning-firstフレーバー（image-to-video）
- **[LongCat-Imageガイド](/documentation/quickstart/LONGCAT_IMAGE.md)** - Qwen-2.5-VLエンコーダーを備えた6Bバイリンガルflow-matchingモデル
- **[LongCat-Image Editガイド](/documentation/quickstart/LONGCAT_EDIT.md)** - 参照潜在変数を必要とする画像編集フレーバー
- **[LTX Videoガイド](/documentation/quickstart/LTXVIDEO.md)** - ビデオ拡散トレーニング
- **[Hunyuan Video 1.5ガイド](/documentation/quickstart/HUNYUANVIDEO.md)** - SRステージを備えた8.3B flow-matching T2V/I2V
- **[Wan Videoガイド](/documentation/quickstart/WAN.md)** - TREADサポート付きビデオflow-matching
- **[HiDreamガイド](/documentation/quickstart/HIDREAM.md)** - 高度な機能を備えたMoEモデル
- **[Cosmos2ガイド](/documentation/quickstart/COSMOS2IMAGE.md)** - マルチモーダル画像生成
- **[OmniGenガイド](/documentation/quickstart/OMNIGEN.md)** - 統合画像生成モデル
- **[Qwen Imageガイド](/documentation/quickstart/QWEN_IMAGE.md)** - 20Bパラメータ大規模トレーニング
- **[Stable Cascade Stage Cガイド](/quickstart/STABLE_CASCADE_C.md)** - 結合されたprior+decoder検証を備えたPrior LoRA
- **[Kandinsky 5.0 Imageガイド](/documentation/quickstart/KANDINSKY5_IMAGE.md)** - Qwen2.5-VL + Flux VAEを使用した画像生成
- **[Kandinsky 5.0 Videoガイド](/documentation/quickstart/KANDINSKY5_VIDEO.md)** - HunyuanVideo VAEを使用したビデオ生成

---

## ハードウェア要件

### 一般要件

- **NVIDIA**: RTX 3080+推奨（H200まで検証済み）
- **AMD**: 7900 XTX 24GBおよびMI300X検証済み（NVIDIAに比べて高いメモリ使用量）
- **Apple**: LoRAトレーニング用にM3 Max+、24GB+のユニファイドメモリ

### モデルサイズ別のメモリガイドライン

- **大規模モデル（12B+）**: Full-rankにはA100-80G、LoRA/LycorisにはFFFF24G+
- **中規模モデル（2B-8B）**: LoRAには16G+、Full-rankトレーニングには40G+
- **小規模モデル（<2B）**: ほとんどのトレーニングタイプに12G+で十分

**注**: 量子化（int8/fp8/nf4）により、メモリ要件が大幅に削減されます。モデル固有の要件については、個別の[クイックスタートガイド](#クイックスタートガイド)を参照してください。

## セットアップ

SimpleTunerは、ほとんどのユーザーがpip経由でインストールできます:

```bash
# 基本インストール（CPU専用PyTorch）
pip install simpletuner

# CUDAユーザー（NVIDIA GPU）
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwellユーザー（NVIDIA Bシリーズ GPU）
pip install 'simpletuner[cuda13]'

# ROCmユーザー（AMD GPU）
pip install 'simpletuner[rocm]'

# Apple Siliconユーザー（M1/M2/M3/M4 Mac）
pip install 'simpletuner[apple]'
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](/documentation/INSTALL.md)を参照してください。

## トラブルシューティング

環境（`config/config.env`）ファイルに`export SIMPLETUNER_LOG_LEVEL=DEBUG`を追加することで、より詳細なインサイトを得るためにデバッグログを有効にします。

トレーニングループのパフォーマンス分析には、`SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`を設定すると、設定の問題を強調表示するタイムスタンプが付きます。

利用可能なオプションの包括的なリストについては、[このドキュメント](/documentation/OPTIONS.md)を参照してください。
