# 任务幂等性与配额预留

本文描述 SimpleTuner 云训练中用于安全、可靠提交任务的两个相关系统：

1. **Idempotency Keys（幂等键）** - 防止重复提交任务
2. **Quota Reservations（配额预留）** - 并发配额的原子占位

这两个系统面向分布式环境、CI/CD 管道以及网络不稳定场景的韧性设计。

## Idempotency Keys

### 概述

幂等键允许客户端安全重试任务提交而不会创建重复任务。如果请求被中断（网络超时、进程崩溃等），客户端可使用相同键重发并获得原始任务响应。

<details>
<summary>幂等性如何工作（时序图）</summary>

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

### Key 格式与建议

幂等键是客户端生成的字符串。推荐格式：

<details>
<summary>Key 格式示例</summary>

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

**最佳实践：**
- 保持键长度小于 256 字符
- 使用 URL 安全字符（字母数字、连字符、下划线）
- 包含足够上下文以识别逻辑操作
- 对 CI/CD，绑定提交/构建/部署标识符

### TTL 与过期

幂等键在创建后 **24 小时** 过期。意味着：

- 24 小时内的重试返回原任务
- 24 小时后同一键将创建新任务
- TTL 可按键配置，但默认 24 小时

<details>
<summary>TTL 配置示例</summary>

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

### API 用法

#### 带幂等键提交

```bash
curl -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=replicate' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "idempotency_key": "ci-build-abc123"
  }'
```

<details>
<summary>响应示例</summary>

**新任务响应：**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "uploading",
  "data_uploaded": true,
  "idempotent_hit": false
}
```

**重复提交（键已使用）：**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "running",
  "idempotent_hit": true
}
```

`idempotent_hit: true` 表示响应来自与幂等键匹配的现有任务。

</details>

<details>
<summary>数据库结构</summary>

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

过期键会在查询时自动清理。

</details>

## 配额预留系统

### 问题：竞态条件

没有预留时，可能出现并发请求绕过配额限制：

<details>
<summary>竞态示例</summary>

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

两次请求都通过配额检查，但合并后超过限制。

</details>

### 解决方案：原子预留

预留系统提供“先占位再创建”的原子语义：

<details>
<summary>预留流程</summary>

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

### 工作方式

1. **预检查预留**：创建任务前先申请预留
2. **原子检查与占位**：仅在配额允许时成功
3. **TTL 保护**：预留 5 分钟过期（防止孤儿锁）
4. **消耗或释放**：创建成功则消耗；失败则释放

<details>
<summary>预留代码示例</summary>

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

### 预留状态

| 状态 | 描述 | 动作 |
|-------|-------------|--------|
| Active | 预留已占用、任务未创建 | 阻止其他预留 |
| Consumed | 任务创建成功 | 不再计入配额 |
| Expired | TTL 到期未消耗 | 自动忽略 |
| Released | 失败时显式释放 | 立即释放配额 |

### TTL 与自动清理

预留具有 **5 分钟 TTL**（300 秒）。用于处理：

- **客户端崩溃**：中途崩溃则自动过期
- **慢上传**：5 分钟足够上传大数据集
- **网络问题**：临时断线不会永久锁定

<details>
<summary>TTL 执行查询</summary>

TTL 在查询时执行——过期预留自动清理：

```python
# During slot counting, expired reservations are ignored
cursor = await conn.execute("""
    SELECT COUNT(*) FROM job_reservations
    WHERE user_id = ? AND expires_at > ? AND consumed = 0
""", (user_id, now.isoformat()))
```

</details>

<details>
<summary>数据库结构</summary>

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

## API 客户端最佳实践

### 实现重试逻辑

提交任务的客户端应使用带幂等键的指数退避重试：

<details>
<summary>Python 重试实现</summary>

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

### CI/CD 集成模式

<details>
<summary>GitHub Actions 示例</summary>

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

### 处理配额错误

配额错误不应重试——它表示限制不会立即改变：

<details>
<summary>错误处理示例</summary>

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

## 示例

### Python：带错误处理的完整提交

<details>
<summary>完整 async 客户端示例</summary>

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

### Bash：幂等 CI 脚本

<details>
<summary>完整 bash 脚本</summary>

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

## 监控与调试

### 查询幂等键状态

<details>
<summary>调试用 SQL</summary>

目前幂等键仅存在数据库中。调试可用：

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

### 查询预留状态

<details>
<summary>预留 SQL 查询</summary>

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

### 常见问题

| 现象 | 原因 | 解决方案 |
|---------|-------|----------|
| 产生重复任务 | 未使用幂等键 | 在请求中添加 `idempotency_key` |
| 第一个任务就提示 “Quota exceeded” | 崩溃请求留下孤儿预留 | 等待 5 分钟 TTL 过期 |
| 幂等键未匹配 | 键已过期（>24h） | 使用新键或延长 TTL |
| 配额计数异常 | 计入预留 + 活跃任务 | 这是正确行为 |

## 参见

- [TUTORIAL.md](TUTORIAL.md) - 云训练完整流程
- [ENTERPRISE.md](../server/ENTERPRISE.md) - 多租户配额管理
- [OPERATIONS_TUTORIAL.md](OPERATIONS_TUTORIAL.md) - 生产部署指南
