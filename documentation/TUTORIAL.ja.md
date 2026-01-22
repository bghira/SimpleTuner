# チュートリアル

このチュートリアルは [インストールガイド](INSTALL.md) に統合されました。

## クイックスタート

1. **SimpleTunerのインストール**: `pip install 'simpletuner[cuda]'` (他のプラットフォームについてはREADMEを参照)
   - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]'`
2. **設定**: `simpletuner configure` (対話型セットアップ)
3. **トレーニング**: `simpletuner train`

## 詳細ガイド

- **[インストールガイド](INSTALL.md)** - トレーニングデータの準備を含む完全なセットアップ
- **[クイックスタートガイド](QUICKSTART.md)** - モデル固有のトレーニングガイド
- **[ハードウェア要件](https://github.com/bghira/SimpleTuner#hardware-requirements)** - VRAMとシステム要件

詳細については以下を参照してください:

- **[インストールガイド](INSTALL.md)** - トレーニングデータの準備を含む完全なセットアップ
- **[オプションリファレンス](OPTIONS.md)** - 完全なパラメータリスト
- **[データローダー設定](DATALOADER.md)** - データセットのセットアップ
