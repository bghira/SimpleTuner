# 结构化日志与后台任务

本文介绍 SimpleTuner 云训练功能中的结构化日志系统与后台任务工作器。

## 目录

- [结构化日志](#结构化日志)
  - [配置](#配置)
  - [JSON 日志格式](#json-日志格式)
  - [LogContext 字段注入](#logcontext-字段注入)
  - [Correlation ID](#correlation-id)
- [后台任务](#后台任务)
  - [任务状态轮询工作器](#任务状态轮询工作器)
  - [队列处理工作器](#队列处理工作器)
  - [审批过期工作器](#审批过期工作器)
  - [配置选项](#配置选项)
- [使用日志进行调试](#使用日志进行调试)

---

## 结构化日志

SimpleTuner 云训练使用结构化 JSON 日志系统，提供一致、可解析的日志输出，并自动跟踪 Correlation ID 以便分布式追踪。

### 配置

使用环境变量配置日志：

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
export SIMPLETUNER_LOG_LEVEL="INFO"

# Format: "json" (structured) or "text" (traditional)
export SIMPLETUNER_LOG_FORMAT="json"

# Optional: Log to file in addition to stdout
export SIMPLETUNER_LOG_FILE="/var/log/simpletuner/cloud.log"
```

<details>
<summary>程序化配置</summary>

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

### JSON 日志格式

启用 JSON 输出时，每条日志包含：

<details>
<summary>JSON 日志示例</summary>

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

| 字段 | 说明 |
|-------|-------------|
| `timestamp` | UTC 的 ISO 8601 时间戳 |
| `level` | 日志级别（DEBUG, INFO, WARNING, ERROR, CRITICAL） |
| `logger` | Logger 名称层级 |
| `message` | 可读的日志信息 |
| `correlation_id` | 请求追踪 ID（自动生成或传递） |
| `source` | 文件、行号、函数名 |
| `extra` | 来自 LogContext 的额外结构化字段 |

### LogContext 字段注入

使用 `LogContext` 可以在作用域内自动为日志添加结构化字段：

<details>
<summary>LogContext 使用示例</summary>

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

输出日志将包含上下文字段：

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

常用注入字段：

| 字段 | 用途 |
|-------|---------|
| `job_id` | 训练任务标识符 |
| `provider` | 云提供商（replicate 等） |
| `user_id` | 已认证用户 |
| `step` | 处理阶段（validation, upload, submission） |
| `attempt` | 重试次数 |

### Correlation ID

Correlation ID 用于跨服务边界的请求追踪：

1. **自动生成**：请求线程未携带时自动生成
2. **传播**：通过 `X-Correlation-ID` HTTP 头传播
3. **存储**：存入线程本地存储并自动注入日志
4. **外发**：加入对云提供商的 HTTP 请求头

<details>
<summary>Correlation ID 流程图</summary>

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
<summary>Correlation ID 手动管理</summary>

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
<summary>HTTP 客户端中的 Correlation ID</summary>

HTTP 客户端工厂会自动把 Correlation ID 加入外部请求：

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

## 后台任务

云训练系统会运行多个后台工作器来处理异步操作。

### Background Task Manager

所有后台任务由 `BackgroundTaskManager` 单例管理：

<details>
<summary>任务管理器用法</summary>

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

### 任务状态轮询工作器

当 Webhook 不可用（例如位于防火墙后）时，轮询工作器会同步云提供商的任务状态。

**目的：**
- 轮询活跃任务（pending、uploading、queued、running）
- 更新本地任务存储状态
- 状态变化时发送 SSE
- 更新终态任务的队列条目

<details>
<summary>轮询流程图</summary>

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
<summary>自动启用逻辑</summary>

若未配置 webhook URL，轮询会自动启动：

```python
# In background_tasks.py
async def _should_auto_enable_polling(self) -> bool:
    config = await store.get_config("replicate")
    return not config.get("webhook_url")  # Enable if no webhook
```

</details>

### 队列处理工作器

根据队列优先级和并发限制调度与分发任务。

**目的：**
- 每 5 秒处理任务队列
- 按优先级分发任务
- 尊重用户/组织级并发限制
- 处理队列条目状态迁移

**队列处理间隔：** 5 秒（固定）

### 审批过期工作器

自动过期并拒绝超时的审批请求。

**目的：**
- 每 5 分钟检查过期审批
- 自动拒绝过期审批的任务
- 将队列条目标记为失败
- 发送审批过期 SSE 通知

<details>
<summary>处理流程图</summary>

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

### 配置选项

#### 环境变量

```bash
# Set custom polling interval (seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL="60"
```

<details>
<summary>Enterprise 配置文件</summary>

创建 `simpletuner-enterprise.yaml`：

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

#### 配置属性

| 属性 | 默认值 | 说明 |
|----------|---------|-------------|
| `job_polling_enabled` | false（若无 webhook 则自动） | 显式启用轮询 |
| `job_polling_interval` | 30 seconds | 轮询间隔 |
| Queue processing | 始终启用 | 不可禁用 |
| Approval expiration | 始终启用 | 每 5 分钟检查 |

<details>
<summary>以编程方式访问配置</summary>

```python
from simpletuner.simpletuner_sdk.server.config.enterprise import get_enterprise_config

config = get_enterprise_config()

if config.job_polling_enabled:
    interval = config.job_polling_interval
    print(f"Polling enabled with {interval}s interval")
```

</details>

---

## 使用日志进行调试

### 查找相关日志

使用 Correlation ID 追踪请求在各组件中的日志：

<details>
<summary>日志过滤命令</summary>

```bash
# Find all logs for a specific request
grep '"correlation_id": "abc123"' /var/log/simpletuner/cloud.log

# Or with jq for JSON parsing
cat /var/log/simpletuner/cloud.log | jq 'select(.correlation_id == "abc123")'
```

</details>

<details>
<summary>按任务过滤</summary>

```bash
# Find all logs for a specific job
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.extra.job_id == "xyz789")'
```

</details>

<details>
<summary>监控后台任务</summary>

```bash
# Watch polling activity
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("polling")) | {timestamp, message}'

# Monitor approval expirations
cat /var/log/simpletuner/cloud.log | \
  jq 'select(.message | contains("expired"))'
```

</details>

### 日志级别建议

| 环境 | 级别 | 理由 |
|-------------|-------|-----------|
| Development | DEBUG | 全量信息用于排障 |
| Staging | INFO | 正常运行与关键事件 |
| Production | INFO 或 WARNING | 可视性与日志量平衡 |

### 常见日志消息

| 消息 | 级别 | 含义 |
|---------|-------|---------|
| "Starting job status polling" | INFO | 轮询工作器启动 |
| "Synced N active jobs" | DEBUG | 轮询周期完成 |
| "Queue scheduler started" | INFO | 队列处理启动 |
| "Expired N approval requests" | INFO | 审批自动拒绝 |
| "Failed to sync job X" | DEBUG | 单任务同步失败（可能暂时） |
| "Error in job polling" | ERROR | 轮询循环出错 |

### 与日志聚合系统集成

JSON 日志格式兼容：

- **Elasticsearch/Kibana**：直接摄取 JSON 日志
- **Splunk**：JSON 解析并提取字段
- **Datadog**：JSON 解析管线
- **Loki/Grafana**：使用 `json` 解析器

<details>
<summary>Loki/Promtail 配置示例</summary>

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

### 排障清单

1. **请求未被追踪？**
   - 检查 `X-Correlation-ID` 是否设置
   - 验证 `CorrelationIDFilter` 是否绑定到 logger

2. **上下文字段未出现？**
   - 确保代码位于 `LogContext` 块内
   - 检查 JSON 输出是否启用

3. **轮询不工作？**
   - 检查是否配置 webhook URL（会禁用自动轮询）
   - 若使用显式轮询，检查 enterprise 配置
   - 查看是否有 “Starting job status polling” 日志

4. **队列未处理？**
   - 检查 “Queue scheduler started” 消息
   - 查找 “Failed to start queue processing” 错误
