# 構造化ログとバックグラウンドタスク

このドキュメントでは、SimpleTuner のクラウド学習機能における構造化ログシステムとバックグラウンドタスクワーカーを説明します。

## 目次

- [構造化ログ](#構造化ログ)
  - [設定](#設定)
  - [JSON ログ形式](#json-ログ形式)
  - [LogContext によるフィールド注入](#logcontext-によるフィールド注入)
  - [Correlation ID](#correlation-id)
- [バックグラウンドタスク](#バックグラウンドタスク)
  - [ジョブ状態ポーリングワーカー](#ジョブ状態ポーリングワーカー)
  - [キュー処理ワーカー](#キュー処理ワーカー)
  - [承認期限ワーカー](#承認期限ワーカー)
  - [設定オプション](#設定オプション)
- [ログによるデバッグ](#ログによるデバッグ)

---

## 構造化ログ

SimpleTuner のクラウド学習では、分散トレーシングのための Correlation ID を自動追跡し、解析可能な JSON ログを提供する構造化ログシステムを使用します。

### 設定

環境変数でログを設定します:

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) or "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: Log to file in addition to stdout
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```

<details>
<summary>プログラムによる設定</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    configure_structured_logging,
    init_from_env,
)

# Configure with explicit options
configure_structured_logging(
    level="INFO",
    json_output=True,
    log_file="/var/log/simpletuner/cloud.log",
    include_stack_info=False,  # Include stack traces for errors
)

# Or initialize from environment variables
init_from_env()
```

</details>

### JSON ログ形式

JSON 出力が有効な場合、各ログエントリには以下が含まれます:

<details>
<summary>JSON ログ例</summary>

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Job submitted successfully",
  "correlation_id": "abc123def456",
  "source": {
    "file": "jobs.py",
    "line": 350,
    "function": "submit_job"
  },
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate",
    "cost_estimate": 2.50
  }
}
```

</details>

| フィールド | 説明 |
|-------|-------------|
| `timestamp` | UTC の ISO 8601 タイムスタンプ |
| `level` | ログレベル（DEBUG, INFO, WARNING, ERROR, CRITICAL） |
| `logger` | ロガー名の階層 |
| `message` | 人間が読めるログメッセージ |
| `correlation_id` | リクエスト追跡 ID（自動生成または伝播） |
| `source` | ファイル、行番号、関数名 |
| `extra` | LogContext からの追加フィールド |

### LogContext によるフィールド注入

`LogContext` を使うと、スコープ内のログに構造化フィールドを自動追加できます:

<details>
<summary>LogContext の使用例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.structured_logging import (
    get_logger,
    LogContext,
)

logger = get_logger("simpletuner.cloud.jobs")

async def process_job(job_id: str, provider: str):
    # All logs within this block include job_id and provider
    with LogContext(job_id=job_id, provider=provider):
        logger.info("Starting job processing")

        # Nested context adds more fields
        with LogContext(step="validation"):
            logger.info("Validating configuration")

        with LogContext(step="submission"):
            logger.info("Submitting to provider")

        logger.info("Job processing complete")
```

出力ログにはコンテキストフィールドが含まれます:

```json
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "INFO",
  "logger": "simpletuner.cloud.jobs",
  "message": "Starting job processing",
  "correlation_id": "abc123",
  "extra": {
    "job_id": "xyz789",
    "provider": "replicate"
  }
}
```

</details>

よく使う注入フィールド:

| フィールド | 用途 |
|-------|---------|
| `job_id` | 学習ジョブ識別子 |
| `provider` | クラウドプロバイダ（replicate など） |
| `user_id` | 認証ユーザー |
| `step` | 処理フェーズ（validation, upload, submission） |
| `attempt` | リトライ回数 |

### Correlation ID

Correlation ID はサービス境界を跨ぐリクエスト追跡に使います。以下の特性があります:

1. **自動生成**: リクエストに ID がない場合、各スレッドで自動生成
2. **伝播**: `X-Correlation-ID` HTTP ヘッダで伝播
3. **保存**: スレッドローカルに保存されログへ自動注入
4. **送信**: クラウドプロバイダへの HTTP リクエストに含める

<details>
<summary>Correlation ID フロー図</summary>

```
User Request
     |
     v
[X-Correlation-ID: abc123]  <-- Incoming header (or auto-generated)
     |
     v
[Thread-local storage]  <-- set_correlation_id("abc123")
     |
     +---> Log entry: {"correlation_id": "abc123", ...}
     |
     +---> Outbound HTTP: X-Correlation-ID: abc123
           (to Replicate API)
```

</details>

<details>
<summary>Correlation ID の手動管理</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_correlation_id,
    set_correlation_id,
    clear_correlation_id,
)

# Get current ID (auto-generates if none exists)
current_id = get_correlation_id()

# Set a specific ID (e.g., from incoming request header)
set_correlation_id("request-abc-123")

# Clear when request completes
clear_correlation_id()
```

</details>

<details>
<summary>HTTP クライアントの Correlation ID</summary>

HTTP クライアントファクトリは、Correlation ID を自動的にアウトバウンドリクエストに含めます:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.http_client import (
    get_async_client,
)

# Correlation ID is automatically added to X-Correlation-ID header
async with get_async_client() as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
    # Request includes: X-Correlation-ID: <current-id>
```

</details>

---

## バックグラウンドタスク

クラウド学習システムは非同期処理のために複数のバックグラウンドワーカーを実行します。

### Background Task Manager

すべてのバックグラウンドタスクは `BackgroundTaskManager` シングルトンで管理されます:

<details>
<summary>タスクマネージャの使用</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Start all configured tasks (typically in app lifespan)
await start_background_tasks()

# Stop gracefully on shutdown
await stop_background_tasks()
```

</details>

### ジョブ状態ポーリングワーカー

ポーリングワーカーはクラウドプロバイダからジョブ状態を同期します。Webhook が使えない場合（例: ファイアウォール内）に有効です。

**目的:**
- アクティブジョブ（pending, uploading, queued, running）をポーリング
- ローカルジョブストアを最新状態に更新
- ステータス変更時に SSE を発行
- 終端状態のキューエントリを更新

<details>
<summary>ポーリングフロー図</summary>

```
[Every 30 seconds]
     |
     v
List active jobs from local store
     |
     v
Group by provider
     |
     +---> [replicate] --> Get status from API --> Update local job
     |
     v
Emit SSE events for status changes
     |
     v
Update queue on terminal statuses (completed, failed, cancelled)
```

</details>

<details>
<summary>自動有効化ロジック</summary>

Webhook URL が設定されていない場合、ポーリングワーカーは自動で開始します:

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### キュー処理ワーカー

キューの優先度と同時実行制限に基づいてジョブをスケジューリング・ディスパッチします。

**目的:**
- 5 秒ごとにジョブキューを処理
- 優先度に応じてジョブをディスパッチ
- ユーザー/組織単位の同時実行制限を尊重
- キューエントリの状態遷移を処理

**キュー処理間隔:** 5 秒（固定）

### 承認期限ワーカー

期限切れの承認リクエストを自動的に却下します。

**目的:**
- 5 分ごとに期限切れの承認を確認
- 期限切れ承認のジョブを自動却下
- キューエントリを失敗状態に更新
- 期限切れ通知の SSE を発行

<details>
<summary>処理フロー図</summary>

```
[Every 5 minutes]
     |
     v
List pending approval requests
     |
     v
Filter expired requests (past deadline)
     |
     v
Mark approval requests as expired
     |
     +---> Update queue entries to "failed"
     |
     +---> Update job status to "cancelled"
     |
     +---> Emit SSE "approval_expired" events
```

</details>

### 設定オプション

#### 環境変数

```bash
# Set custom polling interval (seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL="60"
```

<details>
<summary>Enterprise 設定ファイル</summary>

`simpletuner-enterprise.yaml` を作成します:

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
  queue_processing:
    enabled: true
    interval_seconds: 5
```

</details>

#### 設定プロパティ

| プロパティ | 既定 | 説明 |
|----------|---------|-------------|
| `job_polling_enabled` | false（Webhook なしなら自動） | 明示的ポーリングの有効化 |
| `job_polling_interval` | 30 seconds | ポーリング間隔 |
| Queue processing | 常時有効 | 無効化できない |
| Approval expiration | 常時有効 | 5 分ごとに確認 |

<details>
<summary>設定のプログラム参照</summary>

```python
from simpletuner.simpletuner_sdk.server.config.enterprise import get_enterprise_config

config = get_enterprise_config()

if config.job_polling_enabled:
    interval = config.job_polling_interval
    print(f"Polling enabled with {interval}s interval")
```

</details>

---

## ログによるデバッグ

### 関連ログの検索

Correlation ID を使ってリクエストを全コンポーネントで追跡します:

<details>
<summary>ログフィルタコマンド</summary>

```bash
# Find all logs for a specific request
grep '"correlation_id": "abc123"' /var/log/simpletuner/cloud.log

# Or with jq for JSON parsing
cat /var/log/simpletuner/cloud.log | jq 'select(.correlation_id == "abc123")'
```

</details>

<details>
<summary>ジョブによるフィルタ</summary>

```bash
# Find all logs for a specific job
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.extra.job_id == "xyz789")'
```

</details>

<details>
<summary>バックグラウンドタスクの監視</summary>

```bash
# Watch polling activity
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("polling")) | {timestamp, message}'

# Monitor approval expirations
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("expired"))'
```

</details>

### ログレベルの推奨

| 環境 | レベル | 理由 |
|-------------|-------|-----------|
| Development | DEBUG | トラブルシュートのための全可視化 |
| Staging | INFO | 主要イベントを含む通常運用 |
| Production | INFO or WARNING | 可視性とログ量のバランス |

### よくあるログメッセージ

| メッセージ | レベル | 意味 |
|---------|-------|---------|
| "Starting job status polling" | INFO | ポーリングワーカー開始 |
| "Synced N active jobs" | DEBUG | ポーリングサイクル完了 |
| "Queue scheduler started" | INFO | キュー処理が稼働 |
| "Expired N approval requests" | INFO | 承認が自動却下 |
| "Failed to sync job X" | DEBUG | 単一ジョブ同期失敗（暫定） |
| "Error in job polling" | ERROR | ポーリングループでエラー |

### ログ集約基盤との連携

JSON ログ形式は以下と互換です:

- **Elasticsearch/Kibana**: JSON ログを直接取り込み
- **Splunk**: JSON 解析とフィールド抽出
- **Datadog**: JSON 解析パイプライン
- **Loki/Grafana**: `json` パーサを使用

<details>
<summary>Loki/Promtail 設定例</summary>

```yaml
scrape_configs:
  - job_name: simpletuner
    static_configs:
      - targets: [localhost]
        labels:
          job: simpletuner
          __path__: /var/log/simpletuner/cloud.log
    pipeline_stages:
      - json:
          expressions:
            level: level
            correlation_id: correlation_id
            job_id: extra.job_id
      - labels:
          level:
          correlation_id:
```

</details>

### トラブルシューティングチェックリスト

1. **リクエストが追跡されない?**
   - `X-Correlation-ID` ヘッダが設定されているか確認
   - `CorrelationIDFilter` がロガーに付与されているか確認

2. **コンテキストフィールドが表示されない?**
   - コードが `LogContext` ブロック内にあるか確認
   - JSON 出力が有効か確認

3. **ポーリングが動かない?**
   - Webhook URL が設定されているか（自動ポーリング無効になる）
   - 明示ポーリングを使う場合は enterprise 設定を確認
   - "Starting job status polling" ログがあるか確認

4. **キューが処理されない?**
   - "Queue scheduler started" メッセージがあるか確認
   - "Failed to start queue processing" のエラーを確認
