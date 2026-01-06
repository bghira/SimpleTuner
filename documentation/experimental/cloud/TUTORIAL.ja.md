# クラウド学習チュートリアル

このガイドでは、クラウド GPU インフラ上で SimpleTuner の学習ジョブを実行する方法を説明します。Web UI と REST API の両方の手順を含みます。

## 前提条件

- SimpleTuner をインストール済みでサーバーが起動していること（[ローカル API チュートリアル](../../api/TUTORIAL.md#start-the-server)参照）
- キャプション付きデータセットをローカルに用意済み（ローカル学習と同じ[データセット要件](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends)）
- クラウドプロバイダーのアカウント（[対応プロバイダー](#provider-setup)参照）
- API を使う場合：`curl` と `jq` が使えるシェル

## プロバイダー設定 {#provider-setup}

クラウド学習では、選択したプロバイダーの資格情報が必要です。以下のガイドに従って設定してください。

| プロバイダー | セットアップガイド |
|----------|-------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

設定が完了したら、このチュートリアルに戻ってジョブを送信します。

## クイックスタート

プロバイダーの設定ができたら、次の手順を実行します。

1. `http://localhost:8001` を開き **Cloud** タブへ
2. **Settings**（歯車）→ **Validate** で資格情報を確認
3. Model/Training/Dataloader タブで学習を構成
4. **Train in Cloud** をクリック
5. アップロード概要を確認して **Submit**

## 学習済みモデルの受け取り

学習完了後にモデルをどこへ送るかを事前に設定します。初回ジョブの前にいずれかを設定してください。

### オプション 1: HuggingFace Hub（推奨）

HuggingFace アカウントへ直接プッシュします。

1. 書き込み権限付きの [HuggingFace トークン](https://huggingface.co/settings/tokens) を取得
2. 環境変数を設定：
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. **Publishing** タブで「Push to Hub」を有効化し、リポジトリ名を設定

### オプション 2: Webhook によるローカルダウンロード

モデルを自分のマシンに戻す方法です。サーバーをインターネットに公開する必要があります。

1. トンネルを起動：
   ```bash
   ngrok http 8001   # または: cloudflared tunnel --url http://localhost:8001
   ```
2. 公開 URL（例：`https://abc123.ngrok.io`）をコピー
3. Cloud タブ → Settings → Webhook URL に貼り付け
4. モデルは `~/.simpletuner/cloud_outputs/` に保存されます

### オプション 3: 外部 S3

任意の S3 互換エンドポイント（AWS S3、MinIO、Backblaze B2、Cloudflare R2）にアップロードします。

1. **Publishing** タブで S3 設定を行う
2. エンドポイント、バケット、アクセスキー、シークレットキーを入力

## Web UI ワークフロー

### ジョブの送信

1. Model/Training/Dataloader タブで **学習設定**
2. **Cloud** タブへ移動してプロバイダーを選択
3. **Train in Cloud** をクリックして送信前ダイアログを開く
4. **アップロード概要を確認** — ローカルデータセットがパッケージされてアップロードされます
5. 必要に応じて **実行名** を設定
6. **Submit** をクリック

### ジョブの監視

ジョブ一覧には、クラウド/ローカルのすべてのジョブが表示されます：

- **ステータス表示**: Queued → Running → Completed/Failed
- **ライブ進行状況**: 学習ステップ、loss 値（利用可能な場合）
- **コスト追跡**: GPU 時間に基づく推定コスト

ジョブをクリックすると詳細を確認できます：
- ジョブ設定スナップショット
- リアルタイムログ（**View Logs** をクリック）
- 操作: Cancel、Delete（完了後）

### Settings パネル

歯車アイコンから以下にアクセスできます：

- **API キー検証** とアカウント状態
- **Webhook URL**（ローカル配信用）
- **コスト制限**（支出の暴走防止）
- **ハードウェア情報**（GPU 種類、時間単価）

## API ワークフロー

### ジョブの送信

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

`PROVIDER` をプロバイダー名（例：`replicate`）に置き換えてください。

またはインライン設定で送信：

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "--model_family": "flux",
      "--model_type": "lora",
      "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
      "--output_dir": "/outputs/flux-lora",
      "--max_train_steps": 1000,
      "--lora_rank": 16
    },
    "dataloader_config": [
      {
        "id": "training-images",
        "type": "local",
        "dataset_type": "image",
        "instance_data_dir": "/data/datasets/my-dataset",
        "caption_strategy": "textfile",
        "resolution": 1024
      }
    ]
  }' | jq
```

### ジョブ状態の監視

```bash
# ジョブ詳細
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# ジョブ一覧
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# アクティブジョブの状態を同期
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### ジョブログの取得

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### 実行中ジョブのキャンセル

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### 完了ジョブの削除

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## CI/CD 連携

### 冪等なジョブ送信

冪等キーで重複ジョブを防止できます：

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

同じキーが 24 時間以内に再送されると、新規作成せず既存ジョブが返されます。

### GitHub Actions 例

```yaml
name: Cloud Training

on:
  push:
    branches: [main]
    paths:
      - 'training-configs/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Submit Training Job
        env:
          SIMPLETUNER_URL: ${{ secrets.SIMPLETUNER_URL }}
        run: |
          RESPONSE=$(curl -s -X POST "$SIMPLETUNER_URL/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d '{
              "config_name_to_load": "production-lora",
              "idempotency_key": "gh-${{ github.sha }}",
              "tracker_run_name": "gh-run-${{ github.run_number }}"
            }')

          JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
          echo "Submitted job: $JOB_ID"
          echo "JOB_ID=$JOB_ID" >> $GITHUB_ENV

      - name: Wait for Completion
        run: |
          while true; do
            STATUS=$(curl -s "$SIMPLETUNER_URL/api/cloud/jobs/$JOB_ID" | jq -r '.job.status')
            echo "Job status: $STATUS"

            case $STATUS in
              completed) exit 0 ;;
              failed|cancelled) exit 1 ;;
              *) sleep 60 ;;
            esac
          done
```

### API キー認証

自動化パイプラインでは、セッション認証ではなく API キーの作成を推奨します。

**UI から:** Cloud タブ → Settings → API Keys → Create New Key

**API から:**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/auth/api-keys' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_SESSION_TOKEN' \
  -d '{
    "name": "ci-pipeline",
    "expires_days": 90,
    "scoped_permissions": ["job.submit", "job.view.own"]
  }'
```

完全なキーは一度だけ返されます。安全に保管してください。

**API キーの使用：**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**スコープ付き権限：**

| 権限 | 説明 |
|------------|-------------|
| `job.submit` | 新規ジョブの送信 |
| `job.view.own` | 自分のジョブを閲覧 |
| `job.cancel.own` | 自分のジョブをキャンセル |
| `job.view.all` | すべてのジョブを閲覧（管理者） |

## トラブルシューティング

プロバイダー固有の問題（認証、キュー、ハードウェア）については、各プロバイダーのドキュメントを参照してください：

- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### 一般的な問題

**データアップロードが失敗する**
- データセットのパスが存在し、読み取り可能か確認
- zip 作成用のディスク空き容量を確認
- ブラウザのコンソールまたは API レスポンスのエラーを確認

**Webhook がイベントを受信しない**
- ローカルインスタンスが外部公開されているか確認（トンネル稼働）
- Webhook URL が正しいか確認（`https://` を含む）
- SimpleTuner のターミナル出力で webhook 処理エラーを確認

## API リファレンス

### プロバイダー非依存のエンドポイント

| エンドポイント | メソッド | 説明 |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | フィルター付きジョブ一覧 |
| `/api/cloud/jobs/submit` | POST | 新規学習ジョブ送信 |
| `/api/cloud/jobs/sync` | POST | プロバイダーからジョブ同期 |
| `/api/cloud/jobs/{id}` | GET | ジョブ詳細取得 |
| `/api/cloud/jobs/{id}/logs` | GET | ジョブログ取得 |
| `/api/cloud/jobs/{id}/cancel` | POST | 実行中ジョブのキャンセル |
| `/api/cloud/jobs/{id}` | DELETE | 完了ジョブの削除 |
| `/api/metrics` | GET | ジョブとコストのメトリクス取得 |
| `/api/cloud/metrics/cost-limit` | GET | コスト上限の状態取得 |
| `/api/cloud/providers/{provider}` | PUT | プロバイダー設定更新 |
| `/api/cloud/storage/{bucket}/{key}` | PUT | S3 互換アップロードエンドポイント |

プロバイダー固有のエンドポイントは以下を参照：
- [Replicate API Reference](REPLICATE.md#api-reference)

OpenAPI の詳細は `http://localhost:8001/docs` を参照してください。

## 関連資料

- [README.md](README.md) – アーキテクチャ概要とプロバイダー状況
- [REPLICATE.md](REPLICATE.md) – Replicate プロバイダーの設定と詳細
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO、承認、ガバナンス
- [エンドツーエンドのクラウド運用チュートリアル](OPERATIONS_TUTORIAL.md) – 本番運用と監視
- [エンドツーエンドのローカル API チュートリアル](../../api/TUTORIAL.md) – API によるローカル学習
- [Dataloader Configuration](../../DATALOADER.md) – データセット設定リファレンス
