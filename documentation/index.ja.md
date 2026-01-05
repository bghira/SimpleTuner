# SimpleTuner

**SimpleTuner** は、シンプルさと理解しやすさに焦点を当てたマルチモーダル拡散モデルファインチューニングツールキットです。

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __はじめに__

    ---

    SimpleTuner をインストールして、数分で最初のモデルをトレーニング

    [:octicons-arrow-right-24: インストール](INSTALL.ja.md)

-   :material-cog:{ .lg .middle } __Web UI__

    ---

    洗練された Web インターフェースでトレーニングを設定・実行

    [:octicons-arrow-right-24: Web UI チュートリアル](webui/TUTORIAL.md)

-   :material-api:{ .lg .middle } __REST API__

    ---

    HTTP API でトレーニングワークフローを自動化

    [:octicons-arrow-right-24: API チュートリアル](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __クラウドトレーニング__

    ---

    Replicate または分散ワーカーでトレーニングを実行

    [:octicons-arrow-right-24: クラウドトレーニング](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __マルチユーザー__

    ---

    エンタープライズ機能：SSO、クォータ、RBAC、ワーカーオーケストレーション

    [:octicons-arrow-right-24: エンタープライズガイド](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __モデルガイド__

    ---

    Flux、SD3、SDXL、動画モデルなどのステップバイステップガイド

    [:octicons-arrow-right-24: モデルガイド](quickstart/index.md)

</div>

## 機能

- **マルチモーダルトレーニング** - 画像、動画、音声生成モデル
- **Web UI & API** - ブラウザまたは REST で自動化
- **ワーカーオーケストレーション** - 複数の GPU マシンにジョブを分散
- **エンタープライズ対応** - LDAP/OIDC SSO、RBAC、クォータ、監査ログ
- **クラウド統合** - Replicate、セルフホストワーカー
- **メモリ最適化** - DeepSpeed、FSDP2、量子化

## 対応モデル

| タイプ | モデル |
|--------|--------|
| **画像** | Flux.1/2、SD3、SDXL、Chroma、Auraflow、PixArt、Sana、Lumina2、HiDream など |
| **動画** | Wan、LTX Video、Hunyuan Video、Kandinsky 5、LongCat |
| **音声** | ACE-Step |

詳細は[モデルガイド](quickstart/index.md)をご覧ください。

## コミュニティ

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - バグ報告・機能リクエスト

## ライセンス

SimpleTuner はオープンソースソフトウェアです。
