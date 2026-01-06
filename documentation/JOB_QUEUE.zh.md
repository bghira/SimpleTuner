# 任务队列

队列系统管理本地与云端训练任务的调度、并发限制与 GPU 分配。它支持夜间任务排队、GPU 资源管理与资源使用控制。

## 概览

当你提交云端训练任务时，它会进入队列，并按以下规则处理：

- **优先级** - 优先级高的任务先运行
- **并发限制** - 全局与用户级限制避免资源耗尽
- **同优先级 FIFO** - 同一优先级按提交顺序执行

## 队列状态

在 Cloud 标签页操作栏点击 **队列图标** 可打开队列面板，包含：

| 指标 | 说明 |
|--------|-------------|
| **Queued** | 等待运行的任务 |
| **Running** | 当前执行的任务 |
| **Max Concurrent** | 全局并发上限 |
| **Avg Wait** | 平均等待时间 |

## 优先级等级

任务优先级基于用户等级：

| 用户等级 | 优先级 | 值 |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

值越高优先级越高，越先处理。

### 优先级覆盖

Lead 和 Admin 可在特定场景（例如紧急实验）覆盖任务优先级。

## 并发限制

同时运行的任务数由两类限制控制：

### 全局限制（`max_concurrent`）

全用户同时运行的最大任务数。默认：**5 个任务**。

### 单用户限制（`user_max_concurrent`）

单个用户可同时运行的最大任务数。默认：**2 个任务**。

防止单个用户占满所有槽位。

### 更新限制

Admin 可通过队列面板或 API 更新限制。

<details>
<summary>示例</summary>

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

</details>

## 队列中的任务生命周期

1. **Submitted** - 任务创建，加入队列并处于 `pending`
2. **Pending** - 等待空位（并发限制）
3. **Running** - 在云端 GPU 上训练
4. **Completed/Failed** - 终止状态，从活动队列移除

## API 端点

### 列出队列条目

```
GET /api/queue
```

参数：
- `status` - 按状态过滤（pending, running, blocked）
- `limit` - 返回条目上限（默认 50）
- `include_completed` - 包含已完成任务

### 队列统计

```
GET /api/queue/stats
```

<details>
<summary>响应示例</summary>

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

### 我的队列状态

```
GET /api/queue/me
```

返回当前用户的队列位置、待运行任务和运行中任务。

<details>
<summary>响应示例</summary>

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

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `pending_count` | int | 队列中等待的任务数 |
| `running_count` | int | 当前运行的任务数 |
| `blocked_count` | int | 等待审批的任务数 |
| `best_position` | int 或 null | 用户最高优先级（或最早提交）待运行任务的位置 |
| `pending_jobs` | array | 待运行任务详情列表 |
| `running_jobs` | array | 运行中任务详情列表 |

`best_position` 表示用户最高优先级待运行任务在队列中的位置，便于判断下一个任务何时开始。值为 `null` 表示没有待运行任务。

</details>

### 任务位置

```
GET /api/queue/position/{job_id}
```

返回指定任务的队列位置。

### 取消排队任务

```
POST /api/queue/{job_id}/cancel
```

取消尚未开始的任务。

### 审批被阻止的任务

```
POST /api/queue/{job_id}/approve
```

仅管理员。批准需要审批的任务（例如超过成本阈值）。

### 拒绝被阻止的任务

```
POST /api/queue/{job_id}/reject?reason=<reason>
```

仅管理员。拒绝被阻止的任务并附带原因。

### 更新并发设置

```
POST /api/queue/concurrency
```

<details>
<summary>请求体</summary>

```json
{
  "max_concurrent": 10,
  "user_max_concurrent": 3
}
```

</details>

### 触发处理

```
POST /api/queue/process
```

仅管理员。手动触发队列处理（通常自动）。

### 清理旧条目

```
POST /api/queue/cleanup?days=30
```

仅管理员。删除指定天数前完成的条目。

**参数：**

| 参数 | 类型 | 默认 | 范围 | 说明 |
|-----------|------|---------|-------|-------------|
| `days` | int | 30 | 1-365 | 保留天数 |

**行为：**

会删除满足以下条件的队列条目：
- 终态（`completed`、`failed`、`cancelled`）
- `completed_at` 早于指定天数

活动任务（pending, running, blocked）不会被清理。

<details>
<summary>响应与示例</summary>

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
# 清理 7 天前的条目
curl -X POST "http://localhost:8000/api/queue/cleanup?days=7" \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 清理 90 天前的条目（季度清理）
curl -X POST "http://localhost:8000/api/queue/cleanup?days=90" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

</details>

## 架构

<details>
<summary>系统示意图</summary>

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

### 组件

| 组件 | 位置 | 说明 |
|-----------|----------|-------------|
| `JobRepository` | `storage/job_repository.py` | Jobs 与队列的统一 SQLite 持久化 |
| `JobRepoQueueAdapter` | `queue/job_repo_adapter.py` | 调度器兼容适配器 |
| `QueueScheduler` | `queue/scheduler.py` | 调度逻辑 |
| `SchedulingPolicy` | `queue/scheduler.py` | 优先级/公平算法 |
| `QueueDispatcher` | `queue/dispatcher.py` | 任务派发处理 |
| `QueueEntry` | `queue/models.py` | 队列条目数据模型 |
| `LocalGPUAllocator` | `services/local_gpu_allocator.py` | 本地任务 GPU 分配 |

### 数据库结构

队列和任务条目存储在统一 SQLite 数据库（`~/.simpletuner/cloud/jobs.db`）中。

<details>
<summary>Schema 定义</summary>

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

迁移在启动时自动运行。

</details>

## 本地 GPU 并发

提交本地训练任务时，队列系统会跟踪 GPU 分配以防资源冲突。当所需 GPU 不可用时任务会排队。

### GPU 分配跟踪

每个本地任务指定：

- **num_processes** - 需要的 GPU 数量（来自 `--num_processes`）
- **device_ids** - 优选 GPU 索引（来自 `--accelerate_visible_devices`）

分配器会跟踪正在运行任务占用的 GPU，仅在资源可用时启动新任务。

### CLI 选项

#### 提交任务

<details>
<summary>示例</summary>

```bash
# 提交任务，GPU 不可用则排队（默认）
simpletuner jobs submit my-config

# GPU 不可用则立即拒绝
simpletuner jobs submit my-config --no-wait

# 使用任意可用 GPU 而非配置设备 ID
simpletuner jobs submit my-config --any-gpu

# 仅检查 GPU 可用性（dry-run）
simpletuner jobs submit my-config --dry-run
```

</details>

#### 列出任务

<details>
<summary>示例</summary>

```bash
# 列出最近任务
simpletuner jobs list

# 列出指定字段
simpletuner jobs list -o job_id,status,config_name

# JSON 输出指定字段
simpletuner jobs list --format json -o job_id,status

# 使用点号访问嵌套字段
simpletuner jobs list --format json -o job_id,metadata.run_name

# 按状态过滤
simpletuner jobs list --status running
simpletuner jobs list --status queued

# 限制结果数量
simpletuner jobs list -l 10
```

`-o`（输出）选项支持点号访问嵌套字段。例如 `metadata.run_name` 可提取 `run_name`。

</details>

### GPU 状态 API

GPU 分配状态可通过系统状态端点获取：

```
GET /api/system/status?include_allocation=true
```

<details>
<summary>响应示例</summary>

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

队列统计也包含本地 GPU 信息：

```
GET /api/queue/stats
```

<details>
<summary>响应示例</summary>

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

### 本地并发限制

通过已有并发端点控制本地任务与 GPU 使用上限：

```
GET /api/queue/concurrency
POST /api/queue/concurrency
```

并发端点现在可同时接收云端限制与本地 GPU 限制：

| 字段 | 类型 | 说明 |
|-------|------|-------------|
| `max_concurrent` | int | 云端最大并发任务数（默认: 5） |
| `user_max_concurrent` | int | 单用户云端最大任务数（默认: 2） |
| `local_gpu_max_concurrent` | int 或 null | 本地任务可用的最大 GPU 数（null=无限制） |
| `local_job_max_concurrent` | int | 本地最大并发任务数（默认: 1） |

<details>
<summary>示例</summary>

```bash
# 允许最多 2 个本地任务，总计最多 6 块 GPU
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"local_gpu_max_concurrent": 6, "local_job_max_concurrent": 2}'
```

</details>

### 本地任务提交 API

```
POST /api/queue/submit
```

<details>
<summary>请求与响应</summary>

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

状态值：

| 状态 | 说明 |
|--------|-------------|
| `running` | 立即启动并分配 GPU |
| `queued` | 进入队列等待 GPU |
| `rejected` | GPU 不可用且 `no_wait` 为 true |

### 自动任务处理

任务完成或失败后，GPU 会释放并触发队列处理，启动待运行任务。该流程由进程守护生命周期钩子自动执行。

**取消行为**：任务取消时会释放 GPU，但不会自动启动待运行任务。这是为避免批量取消（`simpletuner jobs cancel --all`）时发生竞态条件，导致待运行任务先启动再被取消。取消后请使用 `POST /api/queue/process` 或重启服务器手动触发处理。

## 工作节点派发

任务可以派发到远程工作节点，而不是在编排器的本地 GPU 上运行。完整配置见 [Worker Orchestration](experimental/server/WORKERS.md)。

### 任务目标

提交任务时指定运行位置：

| 目标 | 行为 |
|--------|-------------|
| `auto`（默认） | 先尝试远程工作节点，失败则回退本地 GPU |
| `worker` | 只派发到远程工作节点；无可用则排队 |
| `local` | 仅在编排器本地 GPU 上运行 |

<details>
<summary>示例</summary>

```bash
# CLI
simpletuner jobs submit my-config --target=worker

# API
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{"config_name": "my-config", "target": "worker"}'
```

</details>

### 工作节点选择

任务可指定匹配工作节点的标签要求：

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "location": "us-*"}
  }'
```

标签支持 glob 模式。调度器按以下顺序匹配：

1. 标签要求（必须全部满足）
2. GPU 数量要求
3. 工作节点可用性（IDLE 状态）
4. 匹配工作节点内的 FIFO 顺序

### 启动行为

服务器启动时，队列系统会自动处理所有待运行的本地任务。若 GPU 可用，排队任务会立即启动，无需人工介入。这确保在服务器重启前提交的任务能在重启后继续处理。

启动流程：
1. 服务器初始化 GPU 分配器
2. 从队列中获取待运行本地任务
3. 对每个任务检查 GPU 可用性并启动
4. 无法启动的任务继续排队

注：云端任务由独立的云队列调度器处理，也会在启动时恢复。

## 配置

队列并发限制通过 API 配置并持久化到队列数据库。

**Web UI：**Cloud 标签 → Queue Panel → Settings

<details>
<summary>API 配置示例</summary>

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

| 设置 | 默认 | 说明 |
|---------|---------|-------------|
| `max_concurrent` | 5 | 全局最大运行任务数 |
| `user_max_concurrent` | 2 | 单用户最大运行任务数 |
| `team_max_concurrent` | 10 | 单团队最大运行任务数 |
| `enable_fair_share` | false | 启用团队公平份额限制 |

### 公平份额调度

当 `enable_fair_share: true` 时，调度器会考虑团队归属，防止单一团队垄断资源。

#### 工作机制

公平份额在现有并发控制上增加第三层限制：

| 层级 | 限制 | 目的 |
|-------|-------|---------|
| 全局 | `max_concurrent` | 所有用户/团队的总任务数 |
| 用户 | `user_max_concurrent` | 防止单个用户占满资源 |
| 团队 | `team_max_concurrent` | 防止单个团队占满资源 |

调度器在考虑派发任务时：

1. 检查全局限制 → 达到上限则跳过
2. 检查用户限制 → 达到上限则跳过
3. 若启用公平份额且任务有 `team_id`：
   - 检查团队限制 → 达到上限则跳过

没有 `team_id` 的任务不受团队限制。

#### 启用公平份额

**UI：**Cloud 标签 → Queue Panel → 切换 “Fair-Share Scheduling”

<details>
<summary>API 示例</summary>

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

#### 团队分配

团队由管理员在后台分配给用户。用户提交任务时，其团队 ID 会附加到队列条目中。调度器跟踪每个团队的运行任务并执行限制。

<details>
<summary>示例场景</summary>

当 `max_concurrent=6`、`user_max_concurrent=2`、`team_max_concurrent=3`：

| 团队 | 用户 | 提交 | 运行 | 阻塞 |
|------|-------|-----------|---------|---------|
| Alpha | Alice, Bob | 4 | 3（团队上限） | 1 |
| Beta | Carol | 3 | 2 | 1（等待全局槽位） |

- Alpha 运行 3（达到 `team_max_concurrent`）
- 总运行数 5（低于 `max_concurrent=6`）
- Carol 的任务被阻塞，因为 5+1=6 达到全局上限
- Alice 的第 4 个任务被阻塞，因为团队已达 3/3

这确保任何团队都无法通过大量提交垄断队列。

</details>

### 防止饥饿

等待超过 `starvation_threshold_minutes` 的任务会获得优先级提升，避免长期等待。

## 审批流程

当预计成本超过阈值等情况时，可将任务标记为需要审批：

1. 提交任务时 `requires_approval: true`
2. 任务进入 `blocked` 状态
3. 管理员在队列面板或 API 中审核
4. 管理员批准或拒绝
5. 批准后任务转为 `pending`，按正常流程调度

审批规则配置见 [Enterprise Guide](experimental/server/ENTERPRISE.md)。

## 故障排查

### 任务卡在队列中

<details>
<summary>排查步骤</summary>

检查并发限制：
```bash
curl http://localhost:8000/api/queue/stats
```

若 `running` 等于 `max_concurrent`，说明任务在等待空位。

</details>

### 队列未处理

<details>
<summary>排查步骤</summary>

后台处理器每 5 秒运行一次。检查服务器日志是否有以下信息：
```
Queue scheduler started with 5s processing interval
```

若没有，可能是调度器未启动。

</details>

### 任务从队列中消失

<details>
<summary>排查步骤</summary>

检查是否已完成或失败：
```bash
curl "http://localhost:8000/api/queue?include_completed=true"
```

</details>

### 本地任务显示运行但未训练

<details>
<summary>排查步骤</summary>

若 `jobs list` 显示本地任务为 "running" 但未训练：

1. 检查 GPU 分配状态：
   ```bash
   simpletuner jobs status --format json
   ```
   查看 `local.allocated_gpus` 字段应显示已分配 GPU。

2. 若 allocated GPUs 为空但 running 数非零，队列状态可能不一致。重启服务器触发自动队列修正。

3. 查看服务器日志中的 GPU 分配错误：
   ```
   Failed to allocate GPUs [0] to job <job_id>
   ```

</details>

### 队列深度显示错误

<details>
<summary>说明</summary>

队列深度与运行任务数分别针对本地与云任务计算：

- **本地任务**：由 `LocalGPUAllocator` 按 GPU 分配状态跟踪
- **云任务**：由 `QueueScheduler` 根据提供商状态跟踪

使用 `simpletuner jobs status --format json` 查看细分：
- `local.running_jobs` - 运行中的本地训练任务
- `local.pending_jobs` - 等待 GPU 的本地任务
- `running` - 云队列总运行任务数
- `queue_depth` - 云队列待运行任务数

</details>

## 另见

- [Worker Orchestration](experimental/server/WORKERS.md) - 分布式 worker 注册与任务派发
- [Cloud Training Tutorial](experimental/cloud/TUTORIAL.md) - 云端训练入门
- [Enterprise Guide](experimental/server/ENTERPRISE.md) - 多用户配置、审批、治理
- [Operations Guide](experimental/cloud/OPERATIONS_TUTORIAL.md) - 生产部署
