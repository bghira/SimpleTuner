# ジョブキュー

キューシステムは、ローカルおよびクラウド学習ジョブのスケジューリング、同時実行制限、GPU 割り当てを管理します。夜間スケジューリング、GPU リソース管理、リソース使用の制御を可能にします。

## 概要

クラウド学習ジョブを送信すると、次に基づいてキューに追加され処理されます:

- **優先度** - 優先度の高いジョブが先に実行
- **同時実行制限** - グローバルおよびユーザー単位の制限でリソース枯渇を防止
- **同一優先度内 FIFO** - 同じ優先度のジョブは送信順に実行

## キューの状態

Cloud タブのアクションバーで **キューアイコン** をクリックするとキューパネルが開きます。表示される内容:

| 指標 | 説明 |
|--------|-------------|
| **Queued** | 実行待ちのジョブ |
| **Running** | 実行中のジョブ |
| **Max Concurrent** | 同時実行のグローバル上限 |
| **Avg Wait** | キュー待機の平均時間 |

## 優先度レベル

ジョブの優先度はユーザーレベルに基づきます:

| ユーザーレベル | 優先度 | 値 |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

値が大きいほど優先度が高く、先に処理されます。

### 優先度の上書き

Lead と Admin は特定の状況（例: 緊急実験）でジョブ優先度を上書きできます。

## 同時実行制限

同時に実行できるジョブ数は 2 つの制限で制御します:

### グローバル制限（`max_concurrent`）

全ユーザーで同時実行できる最大ジョブ数。既定: **5 ジョブ**。

### ユーザー単位制限（`user_max_concurrent`）

1 ユーザーが同時に実行できる最大ジョブ数。既定: **2 ジョブ**。

これにより、特定ユーザーが全スロットを消費するのを防ぎます。

### 制限の更新

Admin はキューパネルまたは API から制限を更新できます。

<details>
<summary>例</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## キュー内のジョブライフサイクル

1. **Submitted** - ジョブ作成、`pending` 状態でキューに追加
2. **Pending** - スロット待ち（同時実行制限）
3. **Running** - クラウド GPU で学習中
4. **Completed/Failed** - 終端状態。アクティブキューから除外

## API エンドポイント

### キューエントリ一覧

```
GET /api/queue
```

パラメータ:
- `status` - 状態でフィルタ（pending, running, blocked）
- `limit` - 返却数上限（既定: 50）
- `include_completed` - 完了済みジョブを含める

### キュー統計

```
GET /api/queue/stats
```

<details>
<summary>レスポンス例</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "user_max_concurrent": 2,
  "avg_wait_seconds": 45.2,
  "by_status": {"pending": 3, "running": 2},
  "by_user": {"1": 2, "2": 3}
}
```

</details>

### 自分のキュー状態

```
GET /api/queue/me
```

現在のユーザーのキュー位置、待機中ジョブ、実行中ジョブを返します。

<details>
<summary>レスポンス例</summary>

```json
{
  "pending_count": 2,
  "running_count": 1,
  "blocked_count": 0,
  "best_position": 3,
  "pending_jobs": [...],
  "running_jobs": [...]
}
```

| フィールド | 型 | 説明 |
|-------|------|-------------|
| `pending_count` | int | 待機中ジョブ数 |
| `running_count` | int | 実行中ジョブ数 |
| `blocked_count` | int | 承認待ちジョブ数 |
| `best_position` | int または null | 最優先（または最も早い提出）の待機ジョブの位置 |
| `pending_jobs` | array | 待機ジョブの詳細一覧 |
| `running_jobs` | array | 実行中ジョブの詳細一覧 |

`best_position` はユーザーの最優先（または最も早い）待機ジョブのキュー位置を示します。次のジョブがいつ開始されるかの目安になります。`null` は待機ジョブがないことを意味します。

</details>

### ジョブ位置

```
GET /api/queue/position/{job_id}
```

指定ジョブのキュー位置を返します。

### 待機ジョブのキャンセル

```
POST /api/queue/{job_id}/cancel
```

まだ開始していないジョブをキャンセルします。

### ブロックされたジョブの承認

```
POST /api/queue/{job_id}/approve
```

Admin 専用。承認が必要なジョブ（例: コスト閾値超過）を承認します。

### ブロックされたジョブの却下

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

Admin 専用。ブロックされたジョブを理由付きで却下します。

### 同時実行の更新

```
POST /api/queue/concurrency
```

<details>
<summary>リクエスト body</summary>

```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

</details>

### 処理のトリガー

```
POST /api/queue/process
```

Admin 専用。キュー処理を手動で起動します（通常は自動）。

### 古いエントリのクリーンアップ

```
POST /api/queue/cleanup?days=30
```

Admin 専用。指定日数より古い完了エントリを削除します。

**パラメータ:**

| パラメータ | 型 | 既定値 | 範囲 | 説明 |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | 保持日数 |

**挙動:**

次の条件に該当するキューエントリを削除します:
- 終端状態（`completed`、`failed`、`cancelled`）
- `completed_at` が指定日数より古い

アクティブジョブ（pending, running, blocked）は削除されません。

<details>
<summary>レスポンスと例</summary>

**Response:**

```json
{
  "success": true,
  "deleted": 42,
  "days": 30
}
```

**Example Usage:**

```bash
# 7 日より古いエントリを削除
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 90 日より古いエントリを削除（四半期クリーンアップ）
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

</details>

## アーキテクチャ

<details>
<summary>システム図</summary>

```
┌─────────────────────────────────────────────────────────────┐
│                     Job Submission                          │
│              (routes/cloud/jobs.py:submit_job)              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                  JobSubmissionService                       │
│              Uploads data, submits to provider              │
│                    Enqueues job                             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    QueueScheduler                           │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │
│   │ Queue Store │  │   Policy     │  │ Background Task │    │
│   │  (SQLite)   │  │  (Priority)  │  │    (5s loop)    │    │
│   └─────────────┘  └──────────────┘  └─────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   QueueDispatcher                           │
│            Updates job status, syncs with provider          │
└─────────────────────────────────────────────────────────────┘
```

</details>

### コンポーネント

| コンポーネント | 位置 | 説明 |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | ジョブとキューの統合 SQLite 永続化 |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | スケジューラ互換のアダプタ |
| `QueueScheduler` | `queue/scheduler.py` | スケジューリングロジック |
| `SchedulingPolicy` | `queue/scheduler.py` | 優先度/公平性アルゴリズム |
| `QueueDispatcher` | `queue/dispatcher.py` | ジョブ配送処理 |
| `QueueEntry` | `queue/models.py` | キューエントリのデータモデル |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | ローカルジョブの GPU 割り当て |

### データベーススキーマ

キューとジョブエントリは統合 SQLite データベース（`~/.simpletuner/cloud/jobs.db`）に保存されます。

<details>
<summary>スキーマ定義</summary>

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    job_id TEXT UNIQUE NOT NULL,
    user_id INTEGER,
    team_id TEXT,
    provider TEXT DEFAULT 'replicate',
    config_name TEXT,
    priority INTEGER DEFAULT 10,
    priority_override INTEGER,
    status TEXT DEFAULT 'pending',
    position INTEGER DEFAULT 0,
    queued_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    estimated_cost REAL DEFAULT 0.0,
    requires_approval INTEGER DEFAULT 0,
    approval_id INTEGER,
    attempt INTEGER DEFAULT 1,
    max_attempts INTEGER DEFAULT 3,
    error_message TEXT,
    metadata TEXT,
    -- GPU allocation (schema v3)
    allocated_gpus TEXT,          -- JSON array of device indices, e.g., "[0,1]"
    job_type TEXT DEFAULT 'cloud', -- "local" or "cloud"
    num_processes INTEGER DEFAULT 1 -- Number of GPUs required
);
```

起動時にマイグレーションが自動実行されます。

</details>

## ローカル GPU の同時実行

ローカル学習ジョブを送信する場合、キューシステムは GPU 割り当てを追跡して競合を防ぎます。必要な GPU が空いていない場合、ジョブはキューに入ります。

### GPU 割り当ての追跡

各ローカルジョブは以下を指定します:

- **num_processes** - 必要な GPU 数（`--num_processes`）
- **device_ids** - 優先 GPU インデックス（`--accelerate_visible_devices`）

アロケータは実行中ジョブに割り当てられた GPU を追跡し、リソースが空いたときにのみ新しいジョブを開始します。

### CLI オプション

#### ジョブ送信

<details>
<summary>例</summary>

```bash
# GPU が空いていなければキューに入れる（既定）
simpletuner jobs submit my-config

# GPU が空いていなければ即時拒否
simpletuner jobs submit my-config --no-wait

# 設定済みの device ID ではなく、空いている GPU を使用
simpletuner jobs submit my-config --any-gpu

# GPU の空き状況をドライランで確認
simpletuner jobs submit my-config --dry-run
```

</details>

#### ジョブ一覧

<details>
<summary>例</summary>

```bash
# 最近のジョブ一覧
simpletuner jobs list

# 特定フィールドのみ表示
simpletuner jobs list -o job_id,status,config_name

# JSON 出力でカスタムフィールド
simpletuner jobs list --format json -o job_id,status

# ドット記法でネストされたフィールドを取得
simpletuner jobs list --format json -o job_id,metadata.run_name

# ステータスでフィルタ
simpletuner jobs list --status running
simpletuner jobs list --status queued

# 件数制限
simpletuner jobs list -l 10
```

`-o`（出力）オプションはドット記法でジョブメタデータ内のネストされたフィールドにアクセスできます。例: `metadata.run_name` は `run_name` を取り出します。

</details>

### GPU ステータス API

GPU 割り当ての状態はシステム状態エンドポイントで確認できます:

```
GET /api/system/status?include_allocation=true
```

<details>
<summary>レスポンス例</summary>

```json
{
  "timestamp": 1704067200.0,
  "load_avg_5min": 2.5,
  "memory_percent": 45.2,
  "gpus": [...],
  "gpu_inventory": {
    "backend": "cuda",
    "count": 4,
    "capabilities": {...}
  },
  "gpu_allocation": {
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "running_local_jobs": 1,
    "devices": [
      {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
      {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
      {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
    ]
  }
}
```

</details>

キュー統計にもローカル GPU 情報が含まれます:

```
GET /api/queue/stats
```

<details>
<summary>レスポンス例</summary>

```json
{
  "queue_depth": 3,
  "running": 2,
  "max_concurrent": 5,
  "local_gpu_max_concurrent": 6,
  "local_job_max_concurrent": 2,
  "local": {
    "running_jobs": 1,
    "pending_jobs": 0,
    "allocated_gpus": [0, 1],
    "available_gpus": [2, 3],
    "total_gpus": 4,
    "max_concurrent_gpus": 6,
    "max_concurrent_jobs": 2
  }
}
```

</details>

### ローカル同時実行の制限

既存の同時実行エンドポイントで、ローカルジョブと GPU 使用数を制御できます:

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

同時実行エンドポイントは、クラウド制限に加えてローカル GPU 制限も受け付けます:

| フィールド | 型 | 説明 |
|-------|------|-------------|
| `max_concurrent` | int | クラウド同時実行ジョブの最大数（既定: 5） |
| `user_max_concurrent` | int | ユーザーあたりのクラウド最大ジョブ数（既定: 2） |
| `local_gpu_max_concurrent` | int または null | ローカルジョブに使える最大 GPU 数（null = 無制限） |
| `local_job_max_concurrent` | int | ローカル同時実行ジョブ数の最大（既定: 1） |

<details>
<summary>例</summary>

```bash
# ローカルジョブを最大 2 件、合計 6 GPU まで許可
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

</details>

### ローカルジョブ送信 API

```
POST /api/queue/submit
```

<details>
<summary>リクエストとレスポンス</summary>

**Request body:**

```json
{
  "config_name": "my-training-config",
  "no_wait": false,
  "any_gpu": false
}
```

**Response:**

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

</details>

ステータス値:

| ステータス | 説明 |
|--------|-------------|
| `running` | GPU が割り当てられ即時開始 |
| `queued` | GPU 待ちでキューに入る |
| `rejected` | GPU 不足かつ `no_wait` が true |

### 自動ジョブ処理

ジョブが完了または失敗すると GPU が解放され、キューが処理されて待機中ジョブが開始されます。これはプロセスキーパーのライフサイクルフックにより自動的に行われます。

**キャンセル時の挙動**: ジョブがキャンセルされると GPU は解放されますが、待機ジョブは自動的に開始されません。これは大量キャンセル（`simpletuner jobs cancel --all`）時に待機ジョブが先に開始される競合を避けるためです。キャンセル後は `POST /api/queue/process` を使うか、サーバーを再起動して手動で処理をトリガーしてください。

## ワーカーへのディスパッチ

ジョブはオーケストレーターのローカル GPU ではなくリモートワーカーに送ることもできます。詳しくは [Worker Orchestration](experimental/server/WORKERS.md) を参照してください。

### ジョブターゲット

ジョブ送信時に実行場所を指定できます:

| ターゲット | 挙動 |
|--------|-------------|
| `auto`（既定） | まずリモートワーカーを試し、なければローカルへフォールバック |
| `worker` | リモートワーカーのみに送信。空きがなければキューへ |
| `local` | オーケストレーターのローカル GPU のみで実行 |

<details>
<summary>例</summary>

```bash
# CLI
simpletuner jobs submit my-config --target=worker

# API
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"config_name": "my-config", "target": "worker"}'
```

</details>

### ワーカー選択

ジョブはワーカーのラベル要件を指定できます:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

ラベルは glob パターンに対応します。スケジューラは以下の順でマッチします:

1. ラベル要件（すべて一致）
2. GPU 数要件
3. ワーカーの可用性（IDLE ステータス）
4. 一致するワーカー内の FIFO

### 起動時の挙動

サーバー起動時に、キューシステムは保留中のローカルジョブを自動処理します。GPU に空きがあれば、キュー内ジョブは手動操作なしで即時開始されます。サーバー再起動前に送信されたジョブが、再起動後に継続されることを保証します。

起動シーケンス:
1. サーバーが GPU アロケータを初期化
2. 保留中のローカルジョブをキューから取得
3. 各ジョブについて、GPU が空いていれば開始
4. GPU 不足のジョブはキューに残る

注記: クラウドジョブは別のクラウドキュー・スケジューラで処理され、起動時に再開されます。

## 設定

キューの同時実行制限は API で設定し、キューデータベースに永続化されます。

**Web UI:** Cloud タブ → Queue Panel → Settings

<details>
<summary>API 設定例</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 5,
    "user_max_concurrent": 2,
    "team_max_concurrent": 10,
    "enable_fair_share": false
  }'
```

</details>

| 設定 | 既定値 | 説明 |
|---------|---------|-------------|
| `max_concurrent` | 5 | グローバル同時実行ジョブの最大数 |
| `user_max_concurrent` | 2 | ユーザーあたりの最大実行ジョブ数 |
| `team_max_concurrent` | 10 | チームあたりの最大実行ジョブ数 |
| `enable_fair_share` | false | チーム単位の公平性制限を有効化 |

### フェアシェア・スケジューリング

`enable_fair_share: true` の場合、スケジューラはチーム所属を考慮し、特定チームがリソースを独占するのを防ぎます。

#### 仕組み

フェアシェアでは 3 層の同時実行制御を追加します:

| 層 | 制限 | 目的 |
|-------|-------|---------|
| グローバル | `max_concurrent` | 全ユーザー/チームの総ジョブ数 |
| ユーザー単位 | `user_max_concurrent` | 1 ユーザーの独占防止 |
| チーム単位 | `team_max_concurrent` | 1 チームの独占防止 |

ジョブをディスパッチする際の判定:

1. グローバル制限を確認 → いっぱいならスキップ
2. ユーザー制限を確認 → 上限ならスキップ
3. フェアシェアが有効かつ `team_id` がある場合:
   - チーム制限を確認 → 上限ならスキップ

`team_id` のないジョブはチーム制限の対象外です。

#### フェアシェアの有効化

**UI:** Cloud タブ → Queue Panel → 「Fair-Share Scheduling」をオン

<details>
<summary>API 例</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{
    "max_concurrent": 10,
    "user_max_concurrent": 3,
    "team_max_concurrent": 5,
    "enable_fair_share": true
  }'
```

</details>

#### チーム割り当て

チームは管理パネルでユーザーに割り当てます。ユーザーがジョブを送信すると、そのチーム ID がキューエントリに付与されます。スケジューラはチームごとの実行数を追跡し、上限を適用します。

<details>
<summary>例</summary>

`max_concurrent=6`、`user_max_concurrent=2`、`team_max_concurrent=3` の場合:

| チーム | ユーザー | 提出 | 実行中 | ブロック |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3（チーム上限） | 1 |
| Beta | Carol | 3 | 2 | 1（グローバル枠待ち） |

- Alpha は 3 件実行中（`team_max_concurrent` 上限）
- 総実行数は 5（`max_concurrent=6` 未満）
- Carol のジョブは 5+1=6 のためグローバル上限でブロック
- Alice の 4 件目はチーム上限 3/3 のためブロック

これにより、どのチームも大量提出でキューを独占できません。

</details>

### 飢餓防止

`starvation_threshold_minutes` を超えて待機しているジョブは優先度ブーストを受け、無期限待機を防ぎます。

## 承認ワークフロー

ジョブは承認が必要な状態としてマークできます（例: 予測コストが閾値超過）:

1. `requires_approval: true` でジョブ送信
2. ジョブが `blocked` 状態に入る
3. Admin がキューパネルまたは API で確認
4. Admin が承認または却下
5. 承認されると `pending` に移動し、通常通りスケジュール

承認ルールの設定は [Enterprise Guide](experimental/server/ENTERPRISE.md) を参照してください。

## トラブルシューティング

### ジョブがキューに停滞する

<details>
<summary>確認手順</summary>

同時実行制限を確認:
```bash
curl http://localhost:8000/api/queue/stats
```

`running` が `max_concurrent` に等しい場合、ジョブはスロット待ちです。

</details>

### キューが処理されない

<details>
<summary>確認手順</summary>

バックグラウンド処理は 5 秒ごとに動作します。サーバーログでエラーがないか確認:
```
Queue scheduler started with 5s processing interval
```

出力がない場合、スケジューラが起動していない可能性があります。

</details>

### ジョブがキューから消えた

<details>
<summary>確認手順</summary>

完了または失敗していないか確認:
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### ローカルジョブが「実行中」だが学習が始まらない

<details>
<summary>確認手順</summary>

`jobs list` でローカルジョブが "running" なのに学習が進まない場合:

1. GPU 割り当て状況を確認:
   ```bash
   simpletuner jobs status --format json
   ```
   `local.allocated_gpus` に使用中 GPU が表示されるはずです。

2. allocated GPUs が空なのに running 数が非ゼロなら、キュー状態の不整合の可能性があります。サーバーを再起動して自動再整合をトリガーしてください。

3. GPU 割り当てエラーをサーバーログで確認:
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### キュー深度の表示が誤っている

<details>
<summary>説明</summary>

キュー深度と実行中ジョブ数はローカルとクラウドで別々に計算されます:

- **ローカルジョブ**: `LocalGPUAllocator` が GPU 割り当て状態に基づき追跡
- **クラウドジョブ**: `QueueScheduler` がプロバイダ状態に基づき追跡

`simpletuner jobs status --format json` で内訳を確認できます:
- `local.running_jobs` - 実行中のローカル学習ジョブ
- `local.pending_jobs` - GPU 待機中のローカルジョブ
- `running` - 実行中の総ジョブ数（クラウドキュー）
- `queue_depth` - 待機中のクラウドジョブ数

</details>

## 関連情報

- [Worker Orchestration](experimental/server/WORKERS.md) - 分散ワーカー登録とジョブディスパッチ
- [Cloud Training Tutorial](experimental/cloud/TUTORIAL.md) - クラウド学習の始め方
- [Enterprise Guide](experimental/server/ENTERPRISE.md) - マルチユーザー構成、承認、ガバナンス
- [Operations Guide](experimental/cloud/OPERATIONS_TUTORIAL.md) - 本番デプロイ
