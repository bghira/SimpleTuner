# Job Idempotency and Quota Reservation

This document describes two related systems that ensure safe, reliable job submission in SimpleTuner's cloud training:

1. **Idempotency Keys** - Prevent duplicate job submissions
2. **Quota Reservations** - Atomic slot consumption for concurrent job limits

Both systems are designed for resilience in distributed environments, CI/CD pipelines, and scenarios with network instability.

## Idempotency Keys

### Overview

Idempotency keys allow clients to safely retry job submissions without creating duplicate jobs. If a request is interrupted (network timeout, process crash, etc.), the client can resend with the same key and receive the original job's response.

<details>
<summary>How idempotency works (sequence diagram)</summary>

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

### Key Format and Recommendations

The idempotency key is a client-generated string. Recommended formats:

<details>
<summary>Key format examples</summary>

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

**Best practices:**
- Keep keys under 256 characters
- Use URL-safe characters (alphanumeric, hyphens, underscores)
- Include enough context to identify the logical operation
- For CI/CD, tie to commit/build/deployment identifiers

### TTL and Expiration

Idempotency keys expire after **24 hours** from creation. This means:

- Retries within 24 hours return the original job
- After 24 hours, the same key creates a new job
- The TTL is configurable per-key but defaults to 24 hours

<details>
<summary>TTL configuration example</summary>

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

### API Usage

#### Submit with Idempotency Key

```bash
curl -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=replicate' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "idempotency_key": "ci-build-abc123"
  }'
```

<details>
<summary>Response examples</summary>

**Response for New Job:**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "uploading",
  "data_uploaded": true,
  "idempotent_hit": false
}
```

**Response for Duplicate (Key Already Used):**

```json
{
  "success": true,
  "job_id": "xyz789abc",
  "status": "running",
  "idempotent_hit": true
}
```

The `idempotent_hit: true` field indicates the response is for an existing job matched by the idempotency key.

</details>

<details>
<summary>Database schema</summary>

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

Keys are automatically cleaned up during lookup operations when they've expired.

</details>

## Quota Reservation System

### The Problem: Race Conditions

Without reservations, concurrent requests can bypass quota limits:

<details>
<summary>Race condition example</summary>

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

Both requests passed the quota check, but together they exceeded the limit.

</details>

### The Solution: Atomic Reservations

The reservation system provides atomic "claim before create" semantics:

<details>
<summary>Reservation flow</summary>

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

### How It Works

1. **Pre-flight Reservation**: Before creating a job, request a reservation
2. **Atomic Check-and-Claim**: Reservation succeeds only if quota allows
3. **TTL Protection**: Reservations expire after 5 minutes (prevents orphan locks)
4. **Consume or Release**: After job creation, consume the reservation; on failure, release it

<details>
<summary>Reservation code example</summary>

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

### Reservation States

| State | Description | Action |
|-------|-------------|--------|
| Active | Slot reserved, job not yet created | Blocks other reservations |
| Consumed | Job was created successfully | No longer counts against quota |
| Expired | TTL elapsed without consumption | Automatically ignored |
| Released | Explicitly released on failure | Slot freed immediately |

### TTL and Automatic Cleanup

Reservations have a **5-minute TTL** (300 seconds). This handles:

- **Crashed clients**: If a client dies mid-submission, the reservation expires
- **Slow uploads**: 5 minutes allows for large dataset uploads
- **Network issues**: Temporary disconnects don't permanently lock slots

<details>
<summary>TTL enforcement query</summary>

The TTL is enforced at query time - expired reservations are cleaned up automatically:

```python
# During slot counting, expired reservations are ignored
cursor = await conn.execute("""
    SELECT COUNT(*) FROM job_reservations
    WHERE user_id = ? AND expires_at > ? AND consumed = 0
""", (user_id, now.isoformat()))
```

</details>

<details>
<summary>Database schema</summary>

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

## Best Practices for API Clients

### Implementing Retry Logic

When building clients that submit jobs, implement exponential backoff with idempotency:

<details>
<summary>Python retry implementation</summary>

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

### CI/CD Integration Pattern

<details>
<summary>GitHub Actions example</summary>

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

### Handling Quota Errors

Quota errors should not trigger retries - they indicate a limit that won't change:

<details>
<summary>Error handling example</summary>

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

## Examples

### Python: Complete Submission with Error Handling

<details>
<summary>Full async client example</summary>

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

### Bash: Idempotent CI Script

<details>
<summary>Full bash script</summary>

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

## Monitoring and Debugging

### Checking Idempotency Key Status

<details>
<summary>SQL queries for debugging</summary>

Currently, idempotency keys are internal to the database. To debug:

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

### Checking Reservation Status

<details>
<summary>SQL queries for reservations</summary>

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

### Common Issues

| Symptom | Cause | Solution |
|---------|-------|----------|
| Duplicate jobs created | Not using idempotency keys | Add `idempotency_key` to requests |
| "Quota exceeded" on first job | Orphan reservations from crashed requests | Wait 5 minutes for TTL expiration |
| Idempotency key not matching | Key expired (>24h) | Use fresh key or extend TTL |
| Unexpected quota counts | Counting reserved + active jobs | This is correct behavior |

## See Also

- [TUTORIAL.md](TUTORIAL.md) - Complete cloud training walkthrough
- [ENTERPRISE.md](../server/ENTERPRISE.md) - Multi-tenant quota management
- [OPERATIONS_TUTORIAL.md](OPERATIONS_TUTORIAL.md) - Production deployment guidance
