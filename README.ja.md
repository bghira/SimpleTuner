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
- **CaptionFlow連携** - [bghira/CaptionFlow](https://github.com/bghira/CaptionFlow)を使い、Web UIのジョブキューからローカルGPUでdataset captionsを生成できます。詳しくは[CaptionFlow連携ガイド](/documentation/CAPTIONFLOW.ja.md)を参照してください
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

SimpleTunerは以下のモデルファミリーをサポートしています。詳細なトレーニング機能の対応状況は[Quickstartガイド](/documentation/QUICKSTART.ja.md)を参照してください。

| モデル | パラメータ数 | ライセンス | 商用利用 |
| --- | --- | --- | --- |
| **ACE-Step** | 3.5B | Apache-2.0 | 可 |
| **Anima** | 未指定 | CircleStone Labs Non-Commercial License v1.2 | 不可（モデル）；出力は可 |
| **Auraflow** | 6B | Apache-2.0 | 可 |
| **Boogu-Image** | 未指定 | Apache-2.0 | 可 |
| **Chroma 1** | 8.9B | Apache-2.0 | 可 |
| **Cosmos2** | 2B-14B | NVIDIA Open Model License | 可 |
| **Cosmos3** | 16B-65B | OpenMDW-1.1 | 可 |
| **DeepFloyd IF** | 0.4B-4.3B stages | DeepFloyd IF License | Abandonware |
| **ERNIE-Image** | 未指定 | Apache-2.0 | 可 |
| **Flux.1** | 8B-12B | Apache-2.0 (schnell); FLUX.1 [dev] Non-Commercial License (dev/Kontext) | checkpointごとに異なる |
| **Flux.2** | 4B-32B | Apache-2.0 (klein 4B); FLUX Non-Commercial License (dev/klein 9B) | checkpointごとに異なる |
| **HeartMuLa** | 3B | SimpleTunerでは未指定 | 上流の条件を確認 |
| **HiDream** | 17B (8.5B MoE) | MIT | 可 |
| **Hunyuan Video** | 8.3B | AGPL-3.0 | 可（copyleft） |
| **Ideogram 4** | 9B | Ideogram 4 Non-Commercial | 不可 |
| **Kandinsky 5.0 Image** | 6B (lite) | MIT | 可 |
| **Kandinsky 5.0 Video** | 2B lite, 19B pro | MIT | 可 |
| **Kwai Kolors** | 2.7B | Apache-2.0 | Abandonware |
| **Krea2** | 未指定 | Krea 2 Community License | 可（年収100万米ドル未満；安全対策必須） |
| **LongCat Image** | 6B | Apache-2.0 | 可 |
| **LongCat Video** | 13.6B | MIT | 可 |
| **LTX Video** | ~2.5B | Apache-2.0 | 可 |
| **LTX Video 2** | 19B | Apache-2.0 | 可 |
| **Lumina2** | 2B | Apache-2.0 | 可 |
| **Mage-Flow** | 4B | MIT | 可 |
| **MiniMax H3** | 33B | MiniMax H3 Community License | 条件付き（地域除外あり；米国/EU/英国/韓国は認可が必要） |
| **OmniGen** | 3.8B | MIT | 可 |
| **PixArt Sigma** | 0.6B-0.9B | OpenRAIL++ | 可（制限あり） |
| **Qwen Image** | 20B | Apache-2.0 | 可 |
| **Sana** | 0.6B-4.8B | Apache-2.0 | 可 |
| **Sana Video** | 2B | Apache-2.0 | 可 |
| **SD 1.x/2.x (Legacy)** | 0.9B | OpenRAIL++ | 可（制限あり） |
| **Stable Diffusion 3** | 2B-8B | Stability AI Community License | 可（年収100万米ドル未満） |
| **Stable Diffusion XL** | 3.5B | CreativeML OpenRAIL-M | 可（制限あり） |
| **Stable Cascade (Stage C)** | 1B, 3.6B prior | SimpleTunerでは未指定 | Abandonware |
| **Wan Video** | 1.3B-14B | Apache-2.0 | 可 |
| **Wan S2V** | 14B | Apache-2.0 | 可 |
| **Z-Image** | 6B | Apache-2.0 | 可 |
| **Z-Image Omni** | 6B | Apache-2.0 | 可 |
| **ZLab I1** | 3B | MIT | 可 |

*ライセンス値は、利用可能な場合はSimpleTunerのモデルヘルパーから、以前未指定だった項目は上流のモデルカード/ライセンスから取得しています。`SimpleTunerでは未指定`は、ヘルパーがライセンス名を持たず、ここでも上流条件を要約していないことを意味します。使用前に上流モデルカードを確認してください。*

### 高度なトレーニング技術

- **TREAD** - Kontextトレーニングを含む、transformerモデル用のトークンワイズドロップアウト
- **マスクロストレーニング** - セグメンテーション/深度ガイダンスによる優れた収束
- **事前正則化** - キャラクター一貫性のためのトレーニング安定性の向上
- **勾配チェックポイント** - メモリ/速度最適化のための設定可能な間隔
- **損失関数** - スケジューリングサポート付きのL2、Huber、Smooth L1
- **SNR重み付け** - トレーニングダイナミクスを改善するためのMin-SNRガンマ重み付け
- **検証アダプタースイープ** - 検証中に一時的にLoRAアダプター（単一またはJSONプリセット）を接続して、トレーニングループに触れることなくアダプターのみまたは比較レンダリングを測定
- **外部検証フック** - 組み込みの検証パイプラインまたはアップロード後のステップを独自のスクリプトに交換して、別のGPUでチェックを実行したり、任意のクラウドプロバイダーにアーティファクトを転送したりできます（[詳細](/documentation/OPTIONS.md#validation_method)）
- **AnyFlow distillation** - online teacher target を使う flow-matching モデル向けの FlowMap interval conditioning（[ガイド](/documentation/experimental/ANYFLOW.ja.md)）
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
- **[Ideogram 4ガイド](/documentation/quickstart/IDEOGRAM4.ja.md)** - **NEW!** 構造化JSON captionを使うFP8優先のLoRAトレーニング
- **[ACE-Stepガイド](/documentation/quickstart/ACE_STEP.md)** - **NEW!** 音声生成モデルトレーニング（text-to-music）
- **[HeartMuLaガイド](/documentation/quickstart/HEARTMULA.md)** - **NEW!** 自己回帰の音声生成モデルトレーニング（text-to-audio）
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
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130

# ROCmユーザー（AMD GPU）
pip install 'simpletuner[rocm]' --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Apple Siliconユーザー（M1/M2/M3/M4 Mac）
pip install 'simpletuner[apple]'
```

手動インストールまたは開発セットアップについては、[インストールドキュメント](/documentation/INSTALL.md)を参照してください。

## トラブルシューティング

環境（`config/config.env`）ファイルに`export SIMPLETUNER_LOG_LEVEL=DEBUG`を追加することで、より詳細なインサイトを得るためにデバッグログを有効にします。

トレーニングループのパフォーマンス分析には、`SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG`を設定すると、設定の問題を強調表示するタイムスタンプが付きます。

利用可能なオプションの包括的なリストについては、[このドキュメント](/documentation/OPTIONS.md)を参照してください。
