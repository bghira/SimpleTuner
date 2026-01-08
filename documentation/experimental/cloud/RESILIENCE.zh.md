# 韧性基础设施

SimpleTuner 的云训练系统使用断路器与重试逻辑，在外部服务出现问题时优雅处理失败。

## 概述

两种主要韧性模式：

1. **断路器** - 阻止对失败服务的请求，避免级联故障
2. **指数退避重试** - 对临时性失败进行自动重试

## 断路器模式

断路器监控外部服务调用。当失败超过阈值时，断路器“打开”，在冷却期内阻断请求。

### 状态

| 状态 | 说明 | 行为 |
|-------|-------------|----------|
| **CLOSED** | 正常运行 | 请求正常通过，统计失败 |
| **OPEN** | 服务异常 | 请求立即被阻断 |
| **HALF_OPEN** | 测试恢复 | 允许少量请求测试恢复情况 |

<details>
<summary>状态迁移图</summary>

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

### 配置

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `failure_threshold` | 5 | 连续失败次数达到后打开断路器 |
| `success_threshold` | 2 | HALF_OPEN 状态下成功次数达到后关闭 |
| `timeout_seconds` | 60.0 | OPEN 到 HALF_OPEN 的等待秒数 |
| `excluded_exceptions` | `()` | 不计入失败的异常类型 |

<details>
<summary>Python 配置示例</summary>

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

Replicate 可使用预配置断路器：

```python
from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    get_replicate_circuit_breaker,
)

breaker = get_replicate_circuit_breaker()
# Uses: failure_threshold=5, success_threshold=2, timeout_seconds=30.0
```

</details>

<details>
<summary>使用示例</summary>

**作为上下文管理器：**

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

**作为装饰器：**

```python
@breaker
async def call_replicate_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

**与 HTTP 客户端工厂配合：**

```python
async with get_async_client(circuit_breaker_name="replicate-api") as client:
    response = await client.get("https://api.replicate.com/v1/predictions")
```

</details>

<details>
<summary>任务提交如何使用断路器</summary>

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

当断路器打开（连续 5 次失败后），任务提交会立即被阻断。

</details>

## 重试模式

当请求遇到临时错误时，使用指数退避进行重试：

1. 等待短暂延迟
2. 重试请求
3. 若再次失败，等待更久
4. 重复直到达到最大次数

### 配置

| 参数 | 默认值 | 说明 |
|-----------|---------|-------------|
| `max_attempts` | 3 | 最大尝试次数（含首次） |
| `base_delay` | 1.0 | 初始延迟秒数 |
| `max_delay` | 30.0 | 最大延迟上限 |
| `exponential_base` | 2.0 | 每次尝试的倍增系数 |
| `jitter` | True | 添加 0-25% 随机抖动 |
| `retryable_status_codes` | `(429, 500, 502, 503, 504)` | 需要重试的 HTTP 码 |

### 延迟计算

```
delay = min(base_delay * (exponential_base ^ attempt), max_delay)
if jitter:
    delay += delay * random(0, 0.25)
```

| 尝试次数 | 基础延迟 | 含抖动 |
|---------|------------|-------------|
| 1 | 1.0s | 1.0-1.25s |
| 2 | 2.0s | 2.0-2.5s |
| 3 | 4.0s | 4.0-5.0s |
| 4 | 8.0s | 8.0-10.0s |
| 5 | 16.0s | 16.0-20.0s |
| 6+ | 30.0s（封顶） | 30.0-37.5s |

<details>
<summary>使用示例</summary>

**直接函数调用：**

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

**作为装饰器：**

```python
@retry(config=RetryConfig(max_attempts=5))
async def call_api():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        response.raise_for_status()
        return response.json()
```

**结合断路器与重试：**

```python
@retry(config=RetryConfig(max_attempts=3))
@breaker
async def resilient_api_call():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.replicate.com/v1/predictions")
        return response.json()
```

顺序很重要：retry 包裹断路器，因此失败会在重试中累积。

</details>

## 监控

### 健康检查集成

`/api/cloud/health` 端点包含断路器状态：

```bash
curl http://localhost:8080/api/cloud/health
```

| 断路器状态 | 健康状态 | 信息 |
|--------------|---------------|---------|
| `closed` | `healthy` | "Circuit closed - normal operation" |
| `half_open` | `degraded` | "Circuit half-open - testing recovery" |
| `open` | `unhealthy` | "Circuit open - blocking requests" |

<details>
<summary>健康检查响应示例</summary>

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
<summary>程序化健康检查</summary>

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

### 日志

断路器与重试会输出结构化日志：

```
WARNING - Circuit breaker 'replicate-api' opening after 5 failures: ConnectionError
INFO - Circuit breaker 'replicate-api' transitioning from OPEN to HALF_OPEN
INFO - Circuit breaker 'replicate-api' closing after 2 successful calls

WARNING - Attempt 1/3 failed, retrying in 1.15s: TimeoutError
ERROR - All 3 attempts failed: TimeoutError
```

## 运维配置

### 提供商设置

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/replicate \
  -H "Content-Type: application/json" \
  -d '{"http_timeout": 60.0}'
```

更长的超时可减少慢请求被误判失败的情况。

### 手动重置

<details>
<summary>重置断路器</summary>

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

## 提供商故障期间的行为

| 阶段 | 行为 |
|-------|----------|
| **初始失败（1-4）** | 仍尝试请求，重试处理临时错误 |
| **断路器开启（5+）** | 所有请求立即拒绝并提示“Provider temporarily unavailable” |
| **恢复测试** | 超时后允许少量测试请求 |
| **完全恢复** | 断路器关闭，恢复正常操作 |

## 排障

**断路器一直处于开启：**
- 确认提供商是否真的故障
- 验证 API 凭据有效
- 检查网络连接与代理设置
- 必要时手动重置断路器

**误报过多：**
- 提高 `failure_threshold`（如 5 → 10）
- 增加 `timeout_seconds` 以延长恢复窗口
- 配置 `excluded_exceptions` 忽略某些错误类型

**未重试预期错误：**
- 检查异常类型是否在 `retryable_exceptions`
- 检查 HTTP 状态码是否在 `retryable_status_codes`

## 参见

- [Operations Guide](OPERATIONS_TUTORIAL.md) - 生产部署与监控
- [Cloud Training Tutorial](TUTORIAL.md) - 入门指南
- [Replicate Integration](REPLICATE.md) - 提供商配置
