# 任务进度与同步 API

本文介绍 SimpleTuner 用于监控云训练任务进度并与云提供商同步本地状态的机制。

## 概述

SimpleTuner 提供多种跟踪任务状态的方法：

| 方法 | 用途 | 延迟 | 资源消耗 |
|--------|----------|---------|----------------|
| Inline Progress API | 运行中任务的 UI 轮询 | 低（默认 5s） | 按任务 API 调用 |
| Job Sync（pull） | 从提供商发现任务 | 中（按需） | 批量 API 调用 |
| `sync_active` 参数 | 刷新活跃任务状态 | 中（按需） | 按活跃任务调用 |
| Background Poller | 自动状态更新 | 可配置（默认 30s） | 持续轮询 |
| Webhooks | 实时推送通知 | 即时 | 无需轮询 |

## Inline Progress API

### Endpoint

```
GET /api/cloud/jobs/{job_id}/inline-progress
```

### 目的

返回单个运行任务的紧凑进度信息，适合在任务列表中显示内联状态而无需获取完整日志。

### 响应

```json
{
  "job_id": "abc123",
  "stage": "Training",
  "last_log": "Step 1500/5000 - loss: 0.0234",
  "progress": 30.0
}
```

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `job_id` | string | 任务标识符 |
| `stage` | string or null | 当前训练阶段：`Preprocessing`, `Warmup`, `Training`, `Validation`, `Saving checkpoint` |
| `last_log` | string or null | 最近一行日志（截断为 80 字符） |
| `progress` | float or null | 基于 step/epoch 解析的进度百分比（0-100） |

### 阶段判定

API 会解析最近的日志以确定训练阶段：

- **Preprocessing**：日志包含 “preprocessing” 或 “loading”
- **Warmup**：日志包含 “warming up” 或 “warmup”
- **Training**：日志包含 “step” 或 “epoch”
- **Validation**：日志包含 “validat”
- **Saving checkpoint**：日志包含 “checkpoint”

### 进度计算

进度百分比从以下日志模式提取：
- `step 1500/5000` -> 30%
- `epoch 3/10` -> 30%

### 适用场景

使用 inline progress API：
- 在任务列表卡片中显示紧凑状态
- 仅对运行中的任务高频轮询（每 5 秒）
- 每次请求希望最小传输量

<details>
<summary>客户端示例（JavaScript）</summary>

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

## 任务同步机制

SimpleTuner 提供两种同步方式，以保持本地任务状态与云提供商一致。

### 1. 全量提供商同步

#### Endpoint

```
POST /api/cloud/jobs/sync
```

#### 目的

发现云提供商中的任务，这些任务可能不在本地存储中。适用于：
- 任务在 SimpleTuner 之外提交（直接使用 Replicate API）
- 本地任务存储被重置或损坏
- 导入历史任务

#### 响应

```json
{
  "synced": 3,
  "message": "Discovered 3 new jobs from Replicate"
}
```

#### 行为

1. 从 Replicate 拉取最近 100 条任务
2. 对每个任务：
   - 本地不存在：创建新的 `UnifiedJob` 记录
   - 本地存在：更新状态、成本与时间戳
3. 返回新发现任务数量

<details>
<summary>客户端示例</summary>

```bash
# Sync jobs from Replicate
curl -X POST http://localhost:8001/api/cloud/jobs/sync

# Response
{"synced": 2, "message": "Discovered 2 new jobs from Replicate"}
```

</details>

#### Web UI 同步按钮

Cloud 面板包含同步按钮，用于发现孤儿任务：

1. 点击任务列表工具栏中的 **Sync** 按钮
2. 同步期间按钮显示加载动画
3. 成功后弹出通知：*"Discovered N jobs from Replicate"*
4. 任务列表与指标自动刷新

**使用场景：**
- 发现通过 Replicate API 或 Web 控制台直接提交的任务
- 数据库重置后的恢复
- 从团队共享 Replicate 账号导入任务

同步按钮内部调用 `POST /api/cloud/jobs/sync`，并随后刷新任务列表与仪表盘指标。

### 2. 活跃任务状态同步（`sync_active`）

#### Endpoint

```
GET /api/cloud/jobs?sync_active=true
```

#### 目的

在返回任务列表前，刷新所有活跃（非终态）云任务的状态。无需等待后台轮询即可获得最新状态。

#### 活跃状态

以下状态视为“活跃”，将会被同步：
- `pending` - 已提交但未开始
- `uploading` - 数据上传中
- `queued` - 处于提供商队列
- `running` - 训练进行中

#### 行为

1. 列表返回前，逐个获取活跃云任务的最新状态
2. 更新本地存储：
   - 当前状态
   - `started_at` / `completed_at` 时间戳
   - `cost_usd`（累计成本）
   - `error_message`（失败时）
3. 返回更新后的任务列表

<details>
<summary>客户端示例（JavaScript）</summary>

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

### 对比：Sync vs sync_active

| 功能 | `POST /jobs/sync` | `GET /jobs?sync_active=true` |
|---------|-------------------|------------------------------|
| 发现新任务 | Yes | No |
| 更新已有任务 | Yes | Yes（仅活跃） |
| 范围 | 提供商全部任务 | 仅本地活跃任务 |
| 用途 | 初次导入、恢复 | 常规状态刷新 |
| 性能 | 较重（批量查询） | 较轻（选择性） |

## Background Poller 配置

后台轮询器可自动同步活跃任务状态，无需客户端介入。

### 默认行为

- **自动启用**：若未配置 webhook URL，将自动启用轮询
- **默认间隔**：30 秒
- **范围**：所有活跃云任务

<details>
<summary>Enterprise 配置</summary>

在生产部署中，可通过 `simpletuner-enterprise.yaml` 配置：

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
<summary>环境变量</summary>

```bash
# Set custom polling interval (in seconds)
export SIMPLETUNER_JOB_POLL_INTERVAL=60
```

</details>

### 工作方式

1. 服务器启动时，`BackgroundTaskManager` 检查：
   - 若 enterprise 配置明确启用，则按该间隔运行
   - 否则若未配置 webhook，则自动启用 30s
2. 每个间隔周期，轮询器：
   - 列出所有活跃状态任务
   - 按提供商分组
   - 从各提供商拉取当前状态
   - 更新本地存储
   - 对状态变化发送 SSE
   - 更新终态任务的队列条目

<details>
<summary>SSE 事件</summary>

后台轮询器检测到状态变化时，会广播 SSE 事件：

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
<summary>程序化访问</summary>

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

## 最佳实践

### 1. 选择合适的同步策略

| 场景 | 推荐方式 |
|----------|---------------------|
| 初次页面加载 | `GET /jobs` 不同步（更快） |
| 定期刷新（30s） | `GET /jobs?sync_active=true` |
| 用户点击“刷新” | `POST /jobs/sync`（用于发现） |
| 运行任务详情 | Inline Progress API（5s） |
| 生产部署 | Background poller + webhooks |

### 2. 避免过度轮询

<details>
<summary>示例</summary>

```javascript
// Good: Poll inline progress only for running jobs
const runningJobs = jobs.filter(j => j.status === 'running');

// Bad: Poll all jobs regardless of status
for (const job of jobs) { /* ... */ }
```

</details>

### 3. 使用 SSE 做实时更新

<details>
<summary>示例</summary>

与其高频轮询，不如订阅 SSE：

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

### 4. 处理终态

<details>
<summary>示例</summary>

停止对终态任务的轮询：

```javascript
const terminalStates = ['completed', 'failed', 'cancelled'];

function shouldPollJob(job) {
    return !terminalStates.includes(job.status);
}
```

</details>

### 5. 生产环境配置 Webhooks

<details>
<summary>示例</summary>

Webhooks 可完全消除轮询需求：

```yaml
# In provider config
webhook_url: "https://your-server.com/api/webhooks/replicate"
```

配置 webhook 后：
- 后台轮询被禁用（除非显式开启）
- 状态更新通过提供商回调实时到达
- 降低对提供商 API 的调用

</details>

## 排障

### 任务未更新

<details>
<summary>调试步骤</summary>

1. 检查后台轮询器是否运行：
   ```bash
   # Look for log line on startup
   grep "job status polling" server.log
   ```

2. 验证提供商连接：
   ```bash
   curl http://localhost:8001/api/cloud/providers/replicate/validate
   ```

3. 强制同步：
   ```bash
   curl -X POST http://localhost:8001/api/cloud/jobs/sync
   ```

</details>

### SSE 事件未收到

<details>
<summary>调试步骤</summary>

1. 检查 SSE 连接上限（默认每 IP 5 个）
2. 确认 EventSource 已连接：
   ```javascript
   eventSource.addEventListener('open', () => {
       console.log('SSE connected');
   });
   ```

</details>

### 提供商 API 使用过高

<details>
<summary>解决方案</summary>

若触及速率限制：
1. 增加 enterprise 配置中的 `job_polling_interval`
2. 降低 inline progress 的轮询频率
3. 配置 webhook 以消除轮询

</details>
