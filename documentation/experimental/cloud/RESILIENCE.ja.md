# レジリエンス基盤

SimpleTuner のクラウド学習システムは、外部サービスの障害時に安全に処理するため、サーキットブレーカとリトライロジックを使用します。

## 概要

主なレジリエンスパターン:

1. **サーキットブレーカ** - 失敗サービスへのリクエストを遮断して連鎖障害を防止
2. **指数バックオフ付きリトライ** - 一時的な失敗を自動的に再試行

## サーキットブレーカパターン

サーキットブレーカは外部サービスへの呼び出しを監視し、失敗が閾値を超えると「開いて」一定期間リクエストを遮断します。

### 状態

| 状態 | 説明 | 挙動 |
|-------|-------------|----------|
| **CLOSED** | 正常動作 | リクエストは通過、失敗はカウント |
| **OPEN** | サービス障害 | リクエストは即時ブロック |
| **HALF_OPEN** | 回復テスト | 限定リクエストで回復確認 |

<details>
<summary>状態遷移図</summary>

```
                                    Success threshold met
                                   +------------------------+
                                   |                        |
                                   v                        |
+----------+   Failure threshold    +----------+  Timeout    +-------------+
|  CLOSED  | ---------------------->|   OPEN   | ----------->|  HALF_OPEN  |
+----------+                        +----------+             +-------------+
     ^                                   ^                        |
     |                                   |                        |
     |         Success resets            |     Any failure        |
     |          failure count            +------------------------+
     |
     +--------------------------------------------------------------------+
                            Success in CLOSED state
```

</details>

### 設定

| パラメータ | 既定 | 説明 |
|-----------|---------|-------------|
| `failure_threshold` | 5 | 連続失敗で OPEN になる閾値 |
| `success_threshold` | 2 | HALF_OPEN から CLOSED に戻る成功回数 |
| `timeout_seconds` | 60.0 | OPEN から HALF_OPEN に移行するまでの秒数 |
| `excluded_exceptions` | `()` | 失敗として数えない例外型 |

<details>
<summary>Python 設定例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
)

config = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout_seconds=60.0,
    excluded_exceptions=(),
)

breaker = CircuitBreaker("replicate-api", config)
```

Replicate では事前構成済みブレーカを使用:

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_replicate_circuit_breaker,
)

breaker = get_replicate_circuit_breaker()
# Uses: failure_threshold=5, success_threshold=2, timeout_seconds=30.0
```

</details>

<details>
<summary>使用例</summary>

**コンテキストマネージャとして:**

```python
breaker = CircuitBreaker("replicate-api")

async def submit_job():
    try:
        async with breaker:
            response = await client.post("/api/submit", data=job_data)
            return response.json()
    except CircuitBreakerError as e:
        print(f"Service unavailable. Retry after {e.retry_after:.1f} seconds")
        return None
```

**デコレータとして:**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**HTTP クライアントファクトリと併用:**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>ジョブ送信におけるサーキットブレーカ</summary>

```python
# From job_submission.py (simplified)
async def submit(self, ctx: SubmissionContext) -> SubmissionResult:
    circuit = await get_circuit_breaker(ctx.provider)

    if not await circuit.can_execute():
        return SubmissionResult(
            success=False,
            error=f"Provider '{ctx.provider}' is temporarily unavailable.",
        )

    try:
        cloud_job = await client.run_job(config=config, ...)
        await circuit.record_success()
    except Exception as provider_exc:
        await circuit.record_failure(provider_exc)
        return SubmissionResult(success=False, error=str(provider_exc))
```

サーキットが開いている場合（連続 5 回失敗後）、ジョブ送信は即座にブロックされます。

</details>

## リトライパターン

一時的なエラーの場合、指数バックオフで再試行します:

1. 短い待機
2. 再試行
3. 失敗したらより長く待機
4. 最大試行回数まで繰り返し

### 設定

| パラメータ | 既定 | 説明 |
|-----------|---------|-------------|
| `max_attempts` | 3 | 最大試行回数（初回含む） |
| `base_delay` | 1.0 | 初期待機秒数 |
| `max_delay` | 30.0 | 最大待機秒数 |
| `exponential_base` | 2.0 | 試行ごとの倍率 |
| `jitter` | True | 0-25% のランダムジッタ |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | 再試行対象の HTTP コード |

### 待機時間計算

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay += delay * random(0, 0.25)
```

| 試行 | 基本待機 | ジッタあり |
|---------|------------|-------------|
| 1 | 1.0s | 1.0-1.25s |
| 2 | 2.0s | 2.0-2.5s |
| 3 | 4.0s | 4.0-5.0s |
| 4 | 8.0s | 8.0-10.0s |
| 5 | 16.0s | 16.0-20.0s |
| 6+ | 30.0s（上限） | 30.0-37.5s |

<details>
<summary>使用例</summary>

**直接関数呼び出し:**

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    retry_async,
    RetryConfig,
)

async def fetch_predictions():
    async def _call():
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.replicate.com/v1/predictions")
            response.raise_for_status()
            return response.json()

    config = RetryConfig(max_attempts=5, base_delay=2.0)
    return await retry_async(_call, config=config)
```

**デコレータとして:**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**サーキットブレーカと併用:**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

順序が重要です。retry が breaker を包むため、失敗は再試行のたびに累積します。

</details>

## 監視

### ヘルスチェック統合

`/api/cloud/health` はサーキットブレーカの状態を含みます:

```bash
curl http://localhost:8080/api/cloud/health
```

| サーキット状態 | ヘルス状態 | メッセージ |
|--------------|---------------|---------|
| `closed` | `healthy` | "Circuit closed - normal operation" |
| `half_open` | `degraded` | "Circuit half-open - testing recovery" |
| `open` | `unhealthy` | "Circuit open - blocking requests" |

<details>
<summary>ヘルスレスポンス例</summary>

```json
{
  "status": "degraded",
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 1.2
    },
    {
      "name": "circuit_breaker_replicate-api",
      "status": "unhealthy",
      "message": "Circuit open - blocking requests (failures: 5)"
    }
  ]
}
```

</details>

<details>
<summary>プログラムによるヘルス確認</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_all_circuit_breaker_health,
    get_circuit_breaker,
)

# All breakers
health = get_all_circuit_breaker_health()

# Single breaker
breaker = get_circuit_breaker("replicate-api")
health = breaker.get_health()
```

</details>

### ログ

サーキットブレーカとリトライは構造化ログを出力します:

```
WARNING - Circuit breaker 'replicate-api' opening after 5 failures: ConnectionError
INFO - Circuit breaker 'replicate-api' transitioning from OPEN to HALF_OPEN
INFO - Circuit breaker 'replicate-api' closing after 2 successful calls

WARNING - Attempt 1/3 failed, retrying in 1.15s: TimeoutError
ERROR - All 3 attempts failed: TimeoutError
```

## 運用設定

### プロバイダ設定

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{"http_timeout": 60.0}'
```

タイムアウトを長くすると、遅いが成功するリクエストの誤判定を減らせます。

### 手動リセット

<details>
<summary>サーキットブレーカのリセット</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_circuit_breaker,
    reset_all_circuit_breakers,
)

# Reset a specific breaker
breaker = get_circuit_breaker("replicate-api")
breaker.reset()

# Reset all breakers
reset_all_circuit_breakers()
```

</details>

## プロバイダ障害時の挙動

| フェーズ | 挙動 |
|-------|----------|
| **初期失敗（1-4）** | リクエストを試行し、リトライが処理 |
| **サーキット OPEN（5+）** | すべて即時拒否（"Provider temporarily unavailable"） |
| **回復テスト** | タイムアウト後に限定リクエストを許可 |
| **完全回復** | サーキットが閉じて通常運用へ復帰 |

## トラブルシューティング

**サーキットが開いたまま:**
- プロバイダが実際にダウンしていないか確認
- API 資格情報の有効性を確認
- ネットワーク接続とプロキシ設定を確認
- 必要なら手動でリセット

**偽陽性が多い:**
- `failure_threshold` を増やす（例: 5 → 10）
- `timeout_seconds` を増やして回復判定を遅らせる
- `excluded_exceptions` で特定エラーを除外

**期待したエラーで再試行しない:**
- 例外型が `retryable_exceptions` に含まれるか確認
- HTTP ステータスが `retryable_status_codes` に含まれるか確認

## GPU サーキットブレーカ

外部サービス用サーキットブレーカに加え、SimpleTuner には **GPU サーキットブレーカ** が含まれており、GPU ハードウェアの健全性を監視し、トレーニング中の CUDA 障害を検出します。これはクラウドトレーニングで特に重要で、GPU ハードウェア障害を早期に検出しないと費用が無駄になります。

### 動作原理

GPU サーキットブレーカは NVIDIA GPU でのトレーニング時に **常に有効**（設定不要）です：

1. **バックグラウンドヘルスモニタリング** - PyNVML 経由で 5 秒ごとに GPU メトリクスをポーリング
2. **CUDA エラー検出** - トレーニング中の CUDA ランタイムエラーをキャッチ
3. **Webhook 送信** - オーケストレータに通知する `gpu.fault` イベントを送信

### 監視メトリクス

| メトリクス | 検出 | 重大度 |
|------------|------|--------|
| **ECC エラー** | 閾値を超えた訂正不可（ダブルビット）エラー | 重大 |
| **温度** | シャットダウン閾値まで 5°C 以内 | 重大 |
| **スロットリング** | ハードウェアスローダウン、サーマルスロットリング、パワーブレーキ | 重大 |
| **CUDA エラー** | トレーニング中の CUDA ランタイムエラー | 重大 |

### Webhook ペイロード

サーキットが開くと `gpu.fault` webhook が送信されます：

```json
{
  "type": "gpu.fault",
  "severity": "critical",
  "job_id": "training-job-123",
  "title": "GPU Fault: cuda_error",
  "message": "CUDA driver error: unknown error",
  "fault": {
    "type": "cuda_error",
    "gpu": {
      "index": 0,
      "name": "NVIDIA RTX 5090",
      "temperature_celsius": 75.5,
      "ecc_errors_double": 2,
      "throttle_reasons": ["hw_thermal_slowdown"],
      "memory_used_percent": 85.5
    },
    "action_taken": "circuit_opened",
    "exception_type": "RuntimeError"
  },
  "timestamp": "2025-01-25T12:34:56.789Z"
}
```

### 障害タイプ

| タイプ | トリガー |
|--------|----------|
| `cuda_error` | トレーニングステップ中の CUDA ランタイムエラー |
| `ecc_error` | 閾値を超えた訂正不可 ECC エラー |
| `health_warning` | 温度またはスロットリング問題を検出 |
| `circuit_open` | 以前の障害でサーキットが既に開いている |

### オーケストレータ統合

クラウドオーケストレータ（RunPod、Lambda Labs など）は `gpu.fault` webhook を使用して：

- 課金を避けるためコンテナを自動終了
- オペレータにハードウェア問題を通知
- 正常なインスタンスへのフェイルオーバーをトリガー
- フリートヘルストラッキング用に GPU 障害をログ

## 参照

- [Operations Guide](OPERATIONS_TUTORIAL.md) - 本番デプロイと監視
- [Cloud Training Tutorial](TUTORIAL.md) - 入門ガイド
- [Replicate Integration](REPLICATE.md) - プロバイダ別設定
- [分散トレーニング](../../DISTRIBUTED.md) - マルチ GPU およびマルチノード設定
