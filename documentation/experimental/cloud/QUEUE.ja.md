# キューシステム

> **ステータス:** Experimental
> **利用可能:** Web UI（Cloud タブ）

キューシステムは、クラウド学習ジョブのスケジューリング、同時実行制限、公平な割り当てを管理します。単一ユーザーモードでも常に有効で、夜間のジョブ予約やリソース管理が可能です。

## 概要

クラウド学習ジョブを送信すると、キューに追加され、以下に基づいて処理されます:

- **優先度** - 優先度が高いジョブが先に実行
- **同時実行制限** - グローバルおよびユーザー単位の上限でリソース枯渇を防止
- **同一優先度内 FIFO** - 同じ優先度のジョブは投入順で実行

## キュー状態

Cloud タブのアクションバーで **キューアイコン** をクリックすると、キューパネルが表示されます:

| 指標 | 説明 |
|--------|-------------|
| **Queued** | 実行待ちジョブ |
| **Running** | 実行中ジョブ |
| **Max Concurrent** | 同時実行のグローバル上限 |
| **Avg Wait** | キュー待ち平均時間 |

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

Lead と Admin は特定状況（例: 緊急実験）で優先度を上書きできます。

## 同時実行制限

同時実行ジョブ数は 2 つの制限で制御されます:

### グローバル上限（`max_concurrent`）

全ユーザー合計の同時実行上限。既定: **5 ジョブ**。

### ユーザー上限（`user_max_concurrent`）

単一ユーザーが同時に実行できる最大ジョブ数。既定: **2 ジョブ**。

これにより、特定ユーザーが全スロットを消費するのを防ぎます。

### 上限の更新

管理者はキューパネルまたは API で上限を更新できます:

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

## キュー内のジョブライフサイクル

1. **Submitted** - ジョブ作成、`pending` 状態でキューに追加
2. **Pending** - スロット待ち（同時実行制限）
3. **Running** - クラウド GPU で学習中
4. **Completed/Failed** - 終端状態、アクティブキューから削除

## API エンドポイント

### キューエントリ一覧

```http
GET /api/queue
```

パラメータ:
- `status` - 状態でフィルタ（pending, running, blocked）
- `limit` - 返す最大件数（既定 50）
- `include_completed` - 完了ジョブを含める

### キュー統計

```http
GET /api/queue/stats
```

返却値:
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

### 自分のキュー状態

```http
GET /api/queue/me
```

現在ユーザーのキュー位置、待機ジョブ、実行中ジョブを返します。

**レスポンス:**

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
| `pending_count` | int | キュー待ちジョブ数 |
| `running_count` | int | 実行中ジョブ数 |
| `blocked_count` | int | 承認待ちジョブ数 |
| `best_position` | int or null | 最優先の待機ジョブの位置 |
| `pending_jobs` | array | 待機ジョブの詳細リスト |
| `running_jobs` | array | 実行中ジョブの詳細リスト |

`best_position` はユーザーの最優先（優先度が高い、または最も早く投入された）待機ジョブの位置を示します。次に自分のジョブが開始されるタイミングの目安になります。`null` は待機ジョブがないことを示します。

### ジョブ位置

```http
GET /api/queue/position/{job_id}
```

特定ジョブのキュー位置を返します。

### 待機ジョブのキャンセル

```http
POST /api/queue/{job_id}/cancel
```

未開始ジョブをキャンセルします。

### ブロックジョブの承認

```http
POST /api/queue/{job_id}/approve
```

管理者専用。承認が必要なジョブを承認します（例: コスト上限超過）。

### ブロックジョブの却下

```http
POST /api/queue/{job_id}/reject?reason=<reason>
```

管理者専用。理由付きでブロックされたジョブを却下します。

### 同時実行更新

```http
POST /api/queue/concurrency
```

Body:
```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

### 処理のトリガー

```http
POST /api/queue/process
```

管理者専用。キュー処理を手動で実行します（通常は自動）。

### 古いエントリのクリーンアップ

```http
POST /api/queue/cleanup?days=30
```

管理者専用。指定日数より古い完了エントリを削除します。
