# ジョブ進捗と同期 API

このドキュメントでは、クラウド学習ジョブの進捗監視と、クラウドプロバイダとのローカル状態同期の仕組みを説明します。

## 概要

SimpleTuner には複数のステータス追跡方法があります:

| 方法 | 用途 | レイテンシ | リソース使用量 |
|--------|----------|---------|----------------|
| Inline Progress API | 実行中ジョブの UI ポーリング | 低（既定 5 秒） | ジョブごとの API 呼び出し |
| Job Sync（pull） | プロバイダからジョブを発見 | 中（オンデマンド） | バッチ API 呼び出し |
| `sync_active` パラメータ | アクティブジョブのステータス更新 | 中（オンデマンド） | アクティブジョブごとの呼び出し |
| Background Poller | 自動ステータス更新 | 設定可能（既定 30 秒） | 継続的ポーリング |
| Webhooks | リアルタイムのプッシュ通知 | 即時 | ポーリング不要 |

## Inline Progress API

### エンドポイント

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### 目的

実行中の単一ジョブの簡易進捗情報を返し、完全なログを取得せずにジョブ一覧でインライン表示するのに適しています。

### レスポンス

```json
{
  "job_id": "abc123",
  "stage": "Training",
  "last_log": "Step 1500/5000 - loss: 0.0234",
  "progress": 30.0
}
```

| フィールド | 型 | 説明 |
|-------|------|-------------|
| `job_id` | string | ジョブ識別子 |
| `stage` | string or null | 現在の学習ステージ: `Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string or null | 最新のログ行（80 文字に切り詰め） |
| `progress` | float or null | ステップ/エポック解析による進捗率（0-100） |

### ステージ判定

API は直近のログを解析してステージを判定します:

- **Preprocessing**: ログに "preprocessing" または "loading" が含まれる
- **Warmup**: "warming up" または "warmup" が含まれる
- **Training**: "step" または "epoch" パターンが含まれる
- **Validation**: "validat" が含まれる
- **Saving checkpoint**: "checkpoint" が含まれる

### 進捗計算

進捗率は次のようなログパターンから抽出します:
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### 利用タイミング

次の用途で Inline Progress API を使います:
- ジョブ一覧カードの簡易ステータス表示
- 実行中ジョブのみを高頻度（5 秒ごと）でポーリング
- 1 リクエストあたりの転送量を最小化したい場合

<details>
<summary>クライアント例（JavaScript）</summary>

```javascript
async function fetchInlineProgress() {
    const runningJobs = jobs.filter(j => j.status === 'running');

    for (const job of runningJobs) {
        try {
            const response = await fetch(
                `/api/cloud/jobs/${job.job_id}/inline-progress`
            );
            if (response.ok) {
                const data = await response.json();
                // Update job card with progress info
                job.inline_stage = data.stage;
                job.inline_log = data.last_log;
                job.inline_progress = data.progress;
            }
        } catch (error) {
            // Silently ignore - job may have completed
        }
    }
}

// Poll every 5 seconds
setInterval(fetchInlineProgress, 5000);
```

</details>

## ジョブ同期メカニズム

SimpleTuner には、クラウドプロバイダとローカル状態を同期する 2 つの方法があります。

### 1. プロバイダ全体同期

#### エンドポイント

```
POST /api/cloud/jobs/sync
```

#### 目的

クラウドプロバイダ上にあるがローカルストアに存在しないジョブを発見します。以下に有効です:
- SimpleTuner 以外から直接送信されたジョブ（Replicate API など）
- ローカルジョブストアのリセット/破損
- 過去ジョブのインポート

#### レスポンス

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### 挙動

1. Replicate から最大 100 件の最近ジョブを取得
2. 各ジョブについて:
   - ローカルに無ければ `UnifiedJob` を新規作成
   - 既に存在する場合はステータス/コスト/タイムスタンプを更新
3. 新規発見ジョブの件数を返す

<details>
<summary>クライアント例</summary>

```bash
# Sync jobs from Replicate
curl -X POST http://localhost:8001/api/cloud/jobs/sync

# Response
{"synced": 2, "message": "Discovered 2 new jobs from Replicate"}
```

</details>

#### Web UI の同期ボタン

Cloud ダッシュボードには、孤立したジョブを発見するための同期ボタンがあります:

1. ジョブ一覧ツールバーの **Sync** ボタンをクリック
2. 同期中はスピナーが表示
3. 成功時にトースト通知 *"Discovered N jobs from Replicate"* が表示
4. ジョブ一覧とメトリクスが自動更新

**用途:**
- Replicate API または Web コンソールから直接送信したジョブの検出
- DB リセット後の復旧
- 共有チームの Replicate アカウントからのインポート

同期ボタンは内部的に `POST /api/cloud/jobs/sync` を呼び出し、その後ジョブ一覧とダッシュボード指標を再読み込みします。

### 2. アクティブジョブ状態同期（`sync_active`）

#### エンドポイント

```
GET /api/cloud/jobs?sync_active=true
```

#### 目的

ジョブ一覧を返す前に、アクティブ（終端でない）クラウドジョブの状態を更新します。バックグラウンドポーリングを待たずに最新状態を取得できます。

#### アクティブ状態

以下の状態が「アクティブ」とみなされ同期されます:
- `pending` - 送信済みでまだ開始していない
- `uploading` - データアップロード中
- `queued` - プロバイダのキュー待ち
- `running` - 学習実行中

#### 挙動

1. ジョブ一覧の前に、アクティブなクラウドジョブごとに最新状態を取得
2. ローカルストアを更新:
   - 現在のステータス
   - `started_at` / `completed_at`
   - `cost_usd`（累積コスト）
   - `error_message`（失敗時）
3. 更新済みジョブ一覧を返す

<details>
<summary>クライアント例（JavaScript）</summary>

```javascript
// Load jobs with active status sync
async function loadJobs(syncActive = false) {
    const params = new URLSearchParams({
        limit: '50',
        provider: 'replicate',
    });

    if (syncActive) {
        params.set('sync_active', 'true');
    }

    const response = await fetch(`/api/cloud/jobs?${params}`);
    const data = await response.json();
    return data.jobs;
}

// Use sync_active during periodic refresh
setInterval(() => loadJobs(true), 30000);
```

</details>

### 比較: Sync vs sync_active

| 機能 | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| 新規ジョブ発見 | Yes | No |
| 既存ジョブ更新 | Yes | Yes（アクティブのみ） |
| 範囲 | プロバイダ全体 | アクティブなローカルジョブのみ |
| 用途 | 初期インポート、復旧 | 通常の状態更新 |
| パフォーマンス | 重い（バッチ） | 軽い（選択的） |

## Background Poller 設定

バックグラウンドポーラーは、クライアント操作なしでアクティブジョブのステータスを自動同期します。

### 既定の挙動

- **自動有効化**: Webhook URL が設定されていない場合に自動で有効
- **既定の間隔**: 30 秒
- **対象**: すべてのアクティブクラウドジョブ

<details>
<summary>Enterprise 設定</summary>

本番環境では `simpletuner-enterprise.yaml` で設定します:

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

<details>
<summary>環境変数</summary>

```bash
# Set custom polling interval (in seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL=60
```

</details>

### 仕組み

1. サーバ起動時に `BackgroundTaskManager` が確認:
   - enterprise 設定で明示的に有効ならその間隔を使用
   - それ以外で webhook が無ければ 30 秒で自動有効化
2. 各間隔でポーラーが実行:
   - アクティブ状態の全ジョブを列挙
   - プロバイダ別にグループ化
   - 各プロバイダから最新状態を取得
   - ローカルストアを更新
   - ステータス変更の SSE を発行
   - 終端状態のキューエントリを更新

<details>
<summary>SSE イベント</summary>

バックグラウンドポーラーがステータス変更を検知すると SSE を配信します:

```javascript
// Subscribe to SSE events
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'job_status_changed') {
        console.log(`Job ${data.job_id} is now ${data.status}`);
        // Refresh job list
        loadJobs();
    }
});
```

</details>

<details>
<summary>プログラムからのアクセス</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.background_tasks import (
    get_task_manager,
    start_background_tasks,
    stop_background_tasks,
)

# Get the manager
manager = get_task_manager()

# Check if running
if manager._running:
    print("Background tasks are active")

# Manual start/stop (usually handled by app lifespan)
await start_background_tasks()
await stop_background_tasks()
```

</details>

## ベストプラクティス

### 1. 適切な同期戦略を選ぶ

| シナリオ | 推奨アプローチ |
|----------|---------------------|
| 初回ページロード | `GET /jobs`（同期なしで高速） |
| 定期更新（30 秒） | `GET /jobs?sync_active=true` |
| ユーザーの "Refresh" クリック | `POST /jobs/sync` で発見 |
| 実行中ジョブ詳細 | Inline Progress API（5 秒） |
| 本番環境 | Background poller + webhooks |

### 2. 過度なポーリングを避ける

<details>
<summary>例</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. SSE でリアルタイム更新

<details>
<summary>例</summary>

強いポーリングの代わりに SSE に購読します:

```javascript
// Combine SSE with conservative polling
const eventSource = new EventSource('/api/events');

eventSource.addEventListener('message', (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'job_status_changed') {
        loadJobs();  // Refresh on status change
    }
});

// Fallback: poll every 30 seconds
setInterval(() => loadJobs(true), 30000);
```

</details>

### 4. 終端状態の扱い

<details>
<summary>例</summary>

終端状態になったジョブはポーリングを止めます:

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. 本番環境では Webhook を設定

<details>
<summary>例</summary>

Webhooks を使うとポーリングが不要になります:

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

Webhook が設定されると:
- バックグラウンドポーリングは無効化（明示的に有効化しない限り）
- ステータス更新はプロバイダのコールバックでリアルタイムに到達
- プロバイダ API 呼び出しを削減

</details>

## トラブルシューティング

### ジョブが更新されない

<details>
<summary>デバッグ手順</summary>

1. バックグラウンドポーラーが動作しているか確認:
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. プロバイダ接続を確認:
   ```bash
   curl http://localhost:8001/api/cloud/providers/replicate/validate
   ```

3. 強制同期:
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### SSE イベントが届かない

<details>
<summary>デバッグ手順</summary>

1. SSE の接続上限を確認（既定は IP ごとに 5）
2. EventSource の接続を確認:
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### プロバイダ API 使用量が高い

<details>
<summary>対処</summary>

レート制限に達している場合:
1. enterprise 設定の `job_polling_interval` を増やす
2. inline 進捗のポーリング頻度を下げる
3. Webhook を設定してポーリングを排除

</details>
