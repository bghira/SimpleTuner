# SimpleTuner の Docker

この Docker 構成は、Runpod、Vast.ai、その他 Docker 互換ホストを含む様々な環境で SimpleTuner を実行するための包括的な環境を提供します。使いやすさと堅牢性を重視して最適化されており、機械学習プロジェクトに必要なツールとライブラリを統合しています。

## コンテナの特徴

- **CUDA 有効のベースイメージ**: GPU 対応のため `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` をベースに構築しています。
- **開発ツール**: Git、SSH、`tmux`、`vim`、`htop` などの各種ユーティリティを含みます。
- **Python とライブラリ**: Python 3.10 を同梱し、SimpleTuner を pip で事前インストールしています。
- **Huggingface と WandB の統合**: Huggingface Hub と WandB への連携を事前設定済みで、モデル共有と実験追跡を円滑にします。

## はじめに

### WSL による Windows OS 対応（実験的）

以下のガイドは Dockerengine をインストールした WSL2 ディストリビューションでテストされています。


### 1. コンテナのビルド

リポジトリをクローンし、Dockerfile のあるディレクトリに移動して Docker イメージをビルドします:

```bash
docker build -t simpletuner .
```

### 2. コンテナの実行

GPU 対応でコンテナを実行するには、次を実行します:

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

このコマンドは GPU アクセスを有効にし、外部接続用に SSH ポートをマッピングします。

### 3. 環境変数

外部ツールとの連携のため、Huggingface と WandB のトークン用環境変数をサポートしています。実行時に次のように渡します:

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. データボリューム

永続ストレージやホストとコンテナ間のデータ共有のために、データボリュームをマウントします:

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. SSH アクセス

SSH はデフォルトで設定されています。環境変数（Vast.ai は `SSH_PUBLIC_KEY`、Runpod は `PUBLIC_KEY`）を通じて SSH 公開鍵を渡してください。

### 6. SimpleTuner の利用

SimpleTuner は事前インストール済みです。以下のように学習コマンドを実行できます:

```bash
simpletuner configure
simpletuner train
```

設定とセットアップは [インストールドキュメント](INSTALL.md) と [クイックスタートガイド](QUICKSTART.md) を参照してください。

## 追加設定

### カスタムスクリプトと設定

カスタムの起動スクリプトを追加したり設定を変更したい場合は、エントリスクリプト（`docker-start.sh`）を拡張してください。

このセットアップで実現できない要件がある場合は、新しい issue を作成してください。

### Docker Compose

`docker-compose.yaml` を使いたい方向けに、拡張可能なテンプレートを用意しています。

スタックをデプロイしたら、上記の手順どおりにコンテナへ接続して操作できます。

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ WandB と Hugging Face のトークンの扱いには十分注意してください。プライベートなリポジトリでもコミットしないことを推奨します。本番用途ではキー管理ストレージの利用が望ましいですが、本ガイドの範囲外です。
---

## トラブルシューティング

### CUDA バージョンの不一致

**症状**: アプリケーションが GPU を利用できない、または GPU アクセラレーション時に CUDA ライブラリエラーが出る。

**原因**: コンテナ内の CUDA バージョンとホストの CUDA ドライババージョンが一致していない場合に発生します。

**解決策**:
1. **ホストの CUDA ドライババージョンを確認**: ホストで次を実行します。
   ```bash
   nvidia-smi
   ```
   出力右上に CUDA バージョンが表示されます。

2. **コンテナの CUDA バージョンを合わせる**: Docker イメージの CUDA ツールキットがホストの CUDA ドライバと互換であることを確認してください。NVIDIA は前方互換を提供していますが、公式互換表を確認してください。

3. **イメージを再ビルド**: 必要に応じて Dockerfile のベースイメージをホストの CUDA に合わせます。たとえばホストが CUDA 11.2 で、コンテナが 11.8 の場合は次のように変更します:
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   変更後に Docker イメージを再ビルドします。

### SSH 接続の問題

**症状**: SSH でコンテナに接続できない。

**原因**: SSH キー設定の不備、または SSH サービスが正しく起動していない。

**解決策**:
1. **SSH 設定の確認**: 公開鍵がコンテナ内の `~/.ssh/authorized_keys` に正しく追加されていることを確認します。さらに、コンテナに入って SSH サービスが起動しているかを確認します:
   ```bash
   service ssh status
   ```
2. **ポートの公開**: SSH ポート（22）が正しく公開・マッピングされていることを確認します。実行コマンドは次のとおりです:
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### 一般的なアドバイス

- **ログと出力**: コンテナログや出力を確認し、問題の手がかりになるエラーメッセージや警告を探してください。
- **ドキュメントとフォーラム**: Docker と NVIDIA CUDA のドキュメントを参照してください。使用しているソフトウェアや依存関係のコミュニティフォーラムや issue tracker も有用です。
