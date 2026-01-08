# ジョブの冪等性とクォータ予約

このドキュメントでは、SimpleTuner のクラウド学習における安全で信頼性の高いジョブ送信を支える 2 つの仕組みを説明します。

1. **Idempotency Keys** - 重複ジョブ送信を防止
2. **Quota Reservations** - 同時ジョブ上限のためのアトミックなスロット確保

どちらも分散環境、CI/CD パイプライン、ネットワーク不安定な状況での耐障害性を前提に設計されています。

## Idempotency Keys

### 概要

Idempotency Key を使うと、重複ジョブを作らずに安全にリトライできます。リクエストが中断（ネットワークタイムアウト、プロセスクラッシュなど）された場合、同じキーで再送すると元のジョブの応答が返ります。

<details>
<summary>冪等性の動作（シーケンス図）</summary>

```
Client                         Server
  |                              |
  |  POST /jobs/submit           |
  |  idempotency_key: "abc123"   |
  |----------------------------->|
  |                              |  Check idempotency_keys table
  |                              |  Key not found - proceed
  |                              |
  |                              |  Create job, store key->job_id
  |                              |
  |  { job_id: "xyz789" }        |
  |<-----------------------------|
  |                              |
  |  [Connection lost before     |
  |   client receives response]  |
  |                              |
  |  POST /jobs/submit (retry)   |
  |  idempotency_key: "abc123"   |
  |----------------------------->|
  |                              |  Key found - return existing job
  |                              |
  |  { job_id: "xyz789",         |
  |    idempotent_hit: true }    |
  |<-----------------------------|
```

</details>

### キー形式と推奨

Idempotency Key はクライアント側で生成する文字列です。推奨形式:

<details>
<summary>キー形式の例</summary>

```bash
# CI/CD builds - use commit SHA or build ID
idempotency_key="ci-build-${GITHUB_SHA}"
idempotency_key="jenkins-${BUILD_NUMBER}"

# Scheduled jobs - include timestamp or run identifier
idempotency_key="nightly-train-$(date +%Y%m%d)"

# User-triggered - combine user ID with action context
idempotency_key="user-42-config-flux-lora-$(date +%s)"

# UUID for guaranteed uniqueness (when duplicates are never desired)
idempotency_key="$(uuidgen)"
```

</details>

**ベストプラクティス:**
- キーは 256 文字以内にする
- URL セーフ文字（英数字、ハイフン、アンダースコア）を使う
- 論理的な操作を特定できる文脈を含める
- CI/CD ではコミット/ビルド/デプロイ ID に紐付ける

### TTL と有効期限

Idempotency Key は作成から **24 時間** で期限切れになります。つまり:

- 24 時間以内のリトライは元のジョブを返す
- 24 時間を過ぎると同じキーでも新規ジョブになる
- TTL はキー単位で設定可能だが既定は 24 時間

<details>
<summary>TTL 設定例</summary>

```python
# Default: 24-hour TTL
await async_store.store_idempotency_key(
    key="ci-build-abc123",
    job_id="job-xyz789",
    user_id=42,
    ttl_hours=24  # Default
)
```

</details>

### API 利用

#### Idempotency Key 付き送信

```bash
curl -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=replicate' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "idempotency_key": "ci-build-abc123"
  }'
```

<details>
<summary>レスポンス例</summary>

**新規ジョブのレスポンス:**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "uploading",
  "data_uploaded": true,
  "idempotent_hit": false
}
```

**重複（既に使用されたキー）のレスポンス:**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "running",
  "idempotent_hit": true
}
```

`idempotent_hit: true` は、Idempotency Key で既存ジョブが一致したことを示します。

</details>

<details>
<summary>データベーススキーマ</summary>

```sql
CREATE TABLE idempotency_keys (
    idempotency_key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    user_id INTEGER,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX idx_idempotency_expires ON idempotency_keys(expires_at);
```

キーは検索時に期限切れなら自動クリーンアップされます。

</details>

## クォータ予約システム

### 問題: 競合状態

予約がないと、同時リクエストがクォータ制限をすり抜ける可能性があります:

<details>
<summary>競合状態の例</summary>

```
Request A                    Request B
    |                            |
    |  Check quota: 4/5 used     |
    |  (OK to proceed)           |  Check quota: 4/5 used
    |                            |  (OK to proceed)
    |                            |
    |  Create job (now 5/5)      |  Create job (now 6/5!)
    |                            |
```

両方がクォータチェックを通過し、合計で上限を超えてしまいます。

</details>

### 解決策: アトミック予約

予約システムは「作成前に確保」するアトミックな意味論を提供します:

<details>
<summary>予約フロー</summary>

```
Request A                    Request B
    |                            |
    |  Reserve slot (4/5 -> 5/5) |
    |  Got reservation R1        |  Reserve slot (5/5)
    |                            |  DENIED - quota exceeded
    |                            |
    |  Create job successfully   |
    |  Consume reservation R1    |
    |                            |
```

</details>

### 仕組み

1. **事前予約**: ジョブ作成前に予約を要求
2. **アトミックな確認と確保**: クォータが許可する場合のみ予約成功
3. **TTL 保護**: 予約は 5 分で期限切れ（孤立ロック防止）
4. **消費または解放**: ジョブ作成後に予約を消費。失敗時は解放

<details>
<summary>予約コード例</summary>

```python
# The reservation flow
reservation_id = await async_store.reserve_job_slot(
    user_id=42,
    max_concurrent=5,  # User's quota limit
    ttl_seconds=300    # 5-minute expiration
)

if reservation_id is None:
    # Quota exceeded - reject immediately
    return SubmitJobResponse(
        success=False,
        error="Quota exceeded: Maximum 5 concurrent jobs allowed"
    )

try:
    # Proceed with job creation
    result = await submission_service.submit(ctx)

    # Job created successfully - consume the reservation
    await async_store.consume_reservation(reservation_id)

except Exception:
    # Job creation failed - release the reservation
    await async_store.release_reservation(reservation_id)
    raise
```

</details>

### 予約状態

| 状態 | 説明 | アクション |
|-------|-------------|--------|
| Active | スロット確保済み、ジョブ未作成 | 他の予約をブロック |
| Consumed | ジョブ作成成功 | クォータにカウントしない |
| Expired | TTL 経過で未消費 | 自動的に無視 |
| Released | 失敗時に明示解放 | 即座にスロット解放 |

### TTL と自動クリーンアップ

予約は **5 分の TTL**（300 秒）を持ちます。以下に対応します:

- **クライアントクラッシュ**: 送信途中で死んでも予約が期限切れで解放
- **大きなアップロード**: 5 分あれば大規模データセットのアップロードに対応
- **ネットワーク問題**: 一時的な断線でスロットが永久ロックされない

<details>
<summary>TTL 強制のクエリ</summary>

TTL はクエリ時に適用され、期限切れ予約は自動的に無視されます:

```python
# During slot counting, expired reservations are ignored
cursor = await conn.execute("""
    SELECT COUNT(*) FROM job_reservations
    WHERE user_id = ? AND expires_at > ? AND consumed = 0
""", (user_id, now.isoformat()))
```

</details>

<details>
<summary>データベーススキーマ</summary>

```sql
CREATE TABLE job_reservations (
    reservation_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    consumed INTEGER DEFAULT 0
);

CREATE INDEX idx_reservations_user ON job_reservations(user_id);
CREATE INDEX idx_reservations_expires ON job_reservations(expires_at);
```

</details>

## API クライアントのベストプラクティス

### リトライロジックの実装

ジョブ送信クライアントでは、冪等性を保った指数バックオフを実装してください:

<details>
<summary>Python リトライ実装</summary>

```python
import time
import uuid
import requests

def submit_job_with_retry(config_name: str, max_retries: int = 3) -> dict:
    """Submit a job with automatic retry and idempotency protection."""

    # Generate idempotency key once, reuse across retries
    idempotency_key = f"client-{uuid.uuid4()}"

    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8001/api/cloud/jobs/submit",
                params={"provider": "replicate"},
                json={
                    "config_name_to_load": config_name,
                    "idempotency_key": idempotency_key,
                },
                timeout=120,  # Allow time for data upload
            )

            result = response.json()

            if result.get("success"):
                if result.get("idempotent_hit"):
                    print(f"Retry matched existing job: {result['job_id']}")
                return result

            # Check for quota errors (don't retry these)
            error = result.get("error", "")
            if "Quota exceeded" in error:
                raise QuotaExceededError(error)

            # Other errors might be transient
            raise TransientError(error)

        except requests.exceptions.Timeout:
            # Network timeout - retry with same idempotency key
            pass
        except requests.exceptions.ConnectionError:
            # Connection failed - retry
            pass

        # Exponential backoff: 1s, 2s, 4s, ...
        sleep_time = 2 ** attempt
        print(f"Retry {attempt + 1}/{max_retries} in {sleep_time}s")
        time.sleep(sleep_time)

    raise MaxRetriesExceeded(f"Failed after {max_retries} attempts")
```

</details>

### CI/CD 統合パターン

<details>
<summary>GitHub Actions の例</summary>

```yaml
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Submit Training Job
        id: submit
        run: |
          # Use commit SHA for idempotency - reruns of same commit are safe
          IDEMPOTENCY_KEY="gh-${{ github.repository }}-${{ github.sha }}"

          RESPONSE=$(curl -sf -X POST \
            "${SIMPLETUNER_URL}/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d "{
              \"config_name_to_load\": \"production\",
              \"idempotency_key\": \"${IDEMPOTENCY_KEY}\",
              \"tracker_run_name\": \"gh-${{ github.run_number }}\"
            }" || echo '{"success":false,"error":"Request failed"}')

          SUCCESS=$(echo "$RESPONSE" | jq -r '.success')
          JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id // empty')
          IDEMPOTENT=$(echo "$RESPONSE" | jq -r '.idempotent_hit // false')

          if [ "$SUCCESS" != "true" ]; then
            echo "::error::Job submission failed: $(echo "$RESPONSE" | jq -r '.error')"
            exit 1
          fi

          if [ "$IDEMPOTENT" = "true" ]; then
            echo "::notice::Matched existing job from previous run"
          fi

          echo "job_id=$JOB_ID" >> $GITHUB_OUTPUT
```

</details>

### クォータエラーの扱い

クォータエラーはリトライしても解決しないため、再試行しないでください:

<details>
<summary>エラーハンドリング例</summary>

```python
def handle_submission_response(response: dict) -> None:
    """Handle job submission response with appropriate error handling."""

    if not response.get("success"):
        error = response.get("error", "Unknown error")

        # Quota errors - inform user, don't retry
        if "Quota exceeded" in error:
            print(f"Cannot submit: {error}")
            print("Wait for existing jobs to complete or contact administrator")
            return

        # Cost limit errors - similar handling
        if "Cost limit" in error:
            print(f"Spending limit reached: {error}")
            return

        # Other errors might be transient
        raise TransientError(error)

    # Check for warnings even on success
    for warning in response.get("quota_warnings", []):
        print(f"Warning: {warning}")

    if response.get("cost_limit_warning"):
        print(f"Cost warning: {response['cost_limit_warning']}")
```

</details>

## 例

### Python: エラーハンドリング付き送信

<details>
<summary>完全な async クライアント例</summary>

```python
import asyncio
import aiohttp
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass
class SubmitResult:
    success: bool
    job_id: Optional[str] = None
    error: Optional[str] = None
    was_duplicate: bool = False


async def submit_cloud_job(
    base_url: str,
    config_name: str,
    idempotency_key: Optional[str] = None,
    tracker_run_name: Optional[str] = None,
) -> SubmitResult:
    """Submit a cloud training job with idempotency protection.

    Args:
        base_url: SimpleTuner server URL (e.g., "http://localhost:8001")
        config_name: Name of the training configuration to use
        idempotency_key: Optional key for deduplication (auto-generated if None)
        tracker_run_name: Optional name for experiment tracking

    Returns:
        SubmitResult with job_id on success, error message on failure
    """
    # Generate idempotency key if not provided
    if idempotency_key is None:
        idempotency_key = f"py-client-{uuid.uuid4()}"

    payload = {
        "config_name_to_load": config_name,
        "idempotency_key": idempotency_key,
    }

    if tracker_run_name:
        payload["tracker_run_name"] = tracker_run_name

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{base_url}/api/cloud/jobs/submit",
                params={"provider": "replicate"},
                json=payload,
                timeout=aiohttp.ClientTimeout(total=300),  # 5 min for uploads
            ) as response:
                result = await response.json()

                if result.get("success"):
                    return SubmitResult(
                        success=True,
                        job_id=result["job_id"],
                        was_duplicate=result.get("idempotent_hit", False),
                    )
                else:
                    return SubmitResult(
                        success=False,
                        error=result.get("error", "Unknown error"),
                    )

        except asyncio.TimeoutError:
            return SubmitResult(
                success=False,
                error="Request timed out - job may have been created, check with same idempotency key",
            )
        except aiohttp.ClientError as e:
            return SubmitResult(
                success=False,
                error=f"Connection error: {e}",
            )


# Usage
async def main():
    result = await submit_cloud_job(
        base_url="http://localhost:8001",
        config_name="flux-lora-v1",
        idempotency_key="training-batch-2024-01-15",
        tracker_run_name="flux-lora-experiment-42",
    )

    if result.success:
        if result.was_duplicate:
            print(f"Matched existing job: {result.job_id}")
        else:
            print(f"Created new job: {result.job_id}")
    else:
        print(f"Submission failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
```

</details>

### Bash: 冪等な CI スクリプト

<details>
<summary>完全な bash スクリプト</summary>

```bash
#!/bin/bash
set -euo pipefail

# Configuration
SIMPLETUNER_URL="${SIMPLETUNER_URL:-http://localhost:8001}"
CONFIG_NAME="${1:-production-lora}"
MAX_RETRIES=3

# Generate idempotency key from git context if available
if [ -n "${GITHUB_SHA:-}" ]; then
    IDEMPOTENCY_KEY="github-${GITHUB_REPOSITORY}-${GITHUB_SHA}"
elif [ -n "${CI_COMMIT_SHA:-}" ]; then
    IDEMPOTENCY_KEY="gitlab-${CI_PROJECT_PATH}-${CI_COMMIT_SHA}"
else
    IDEMPOTENCY_KEY="manual-$(date +%Y%m%d-%H%M%S)-$$"
fi

echo "Using idempotency key: ${IDEMPOTENCY_KEY}"

# Submit with retry
for i in $(seq 1 $MAX_RETRIES); do
    echo "Attempt $i/$MAX_RETRIES..."

    RESPONSE=$(curl -sf -X POST \
        "${SIMPLETUNER_URL}/api/cloud/jobs/submit?provider=replicate" \
        -H 'Content-Type: application/json' \
        -d "{
            \"config_name_to_load\": \"${CONFIG_NAME}\",
            \"idempotency_key\": \"${IDEMPOTENCY_KEY}\"
        }" 2>&1) || {
        if [ $i -lt $MAX_RETRIES ]; then
            SLEEP_TIME=$((2 ** i))
            echo "Request failed, retrying in ${SLEEP_TIME}s..."
            sleep $SLEEP_TIME
            continue
        else
            echo "Failed after $MAX_RETRIES attempts"
            exit 1
        fi
    }

    SUCCESS=$(echo "$RESPONSE" | jq -r '.success')

    if [ "$SUCCESS" = "true" ]; then
        JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')
        IDEMPOTENT=$(echo "$RESPONSE" | jq -r '.idempotent_hit')

        if [ "$IDEMPOTENT" = "true" ]; then
            echo "Matched existing job: $JOB_ID"
        else
            echo "Created new job: $JOB_ID"
        fi

        # Output for CI systems
        echo "JOB_ID=$JOB_ID"
        exit 0
    else
        ERROR=$(echo "$RESPONSE" | jq -r '.error')

        # Don't retry quota errors
        if echo "$ERROR" | grep -q "Quota exceeded"; then
            echo "Quota exceeded: $ERROR"
            exit 1
        fi

        if [ $i -lt $MAX_RETRIES ]; then
            echo "Error: $ERROR, retrying..."
            sleep $((2 ** i))
        else
            echo "Failed: $ERROR"
            exit 1
        fi
    fi
done
```

</details>

## 監視とデバッグ

### Idempotency Key のステータス確認

<details>
<summary>デバッグ用 SQL クエリ</summary>

現在、Idempotency Key はデータベース内部にあります。デバッグには以下を使用します:

```sql
-- Check if a key exists (connect to jobs.db)
SELECT idempotency_key, job_id, created_at, expires_at
FROM idempotency_keys
WHERE idempotency_key = 'your-key-here';

-- List all active keys
SELECT * FROM idempotency_keys
WHERE expires_at > datetime('now')
ORDER BY created_at DESC;
```

</details>

### 予約ステータスの確認

<details>
<summary>予約の SQL クエリ</summary>

```sql
-- Active (unconsumed, unexpired) reservations
SELECT reservation_id, user_id, created_at, expires_at
FROM job_reservations
WHERE consumed = 0 AND expires_at > datetime('now');

-- Count active slots per user
SELECT user_id, COUNT(*) as active_reservations
FROM job_reservations
WHERE consumed = 0 AND expires_at > datetime('now')
GROUP BY user_id;
```

</details>

### よくある問題

| 症状 | 原因 | 解決策 |
|---------|-------|----------|
| 重複ジョブが作られる | Idempotency Key を使っていない | リクエストに `idempotency_key` を追加 |
| 初回ジョブで “Quota exceeded” | クラッシュしたリクエストの孤立予約 | TTL 期限切れ（5 分）まで待つ |
| Idempotency Key が一致しない | キーの期限切れ（>24h） | 新しいキーを使うか TTL を延長 |
| 想定外のクォータ数 | 予約 + 実行中ジョブを合算している | 正しい挙動です |

## 参照

- [TUTORIAL.md](TUTORIAL.md) - クラウド学習の完全なウォークスルー
- [ENTERPRISE.md](../server/ENTERPRISE.md) - マルチテナントのクォータ管理
- [OPERATIONS_TUTORIAL.md](OPERATIONS_TUTORIAL.md) - 本番デプロイのガイダンス
