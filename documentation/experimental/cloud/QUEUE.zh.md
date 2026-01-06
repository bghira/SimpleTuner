# 队列系统

> **状态：** Experimental
> **可用：** Web UI（Cloud 选项卡）

队列系统负责云训练任务的调度、并发限制与公平分配。即使在单用户模式下也始终启用，支持夜间排队与受控资源使用。

## 概述

提交云训练任务后，会被加入队列并按以下规则处理：

- **优先级** - 优先级高的任务先执行
- **并发限制** - 全局与用户级限制防止资源耗尽
- **同优先级 FIFO** - 同一优先级按提交顺序执行

## 队列状态

在 Cloud 选项卡的操作栏点击 **队列图标** 可打开队列面板，显示：

| 指标 | 说明 |
|--------|-------------|
| **Queued** | 等待执行的任务 |
| **Running** | 正在执行的任务 |
| **Max Concurrent** | 全局并发上限 |
| **Avg Wait** | 平均等待时间 |

## 优先级等级

任务优先级由用户级别决定：

| 用户级别 | 优先级 | 值 |
|------------|----------|-------|
| Admin | Urgent | 30 |
| Lead | High | 20 |
| Researcher | Normal | 10 |
| Viewer | Low | 0 |

数值越高优先级越高，越早处理。

### 优先级覆盖

Lead 与 Admin 可以在特殊情况下覆盖任务优先级（例如紧急实验）。

## 并发限制

同时运行的任务数量由两个限制控制：

### 全局限制（`max_concurrent`）

所有用户的同时运行任务上限。默认：**5 个任务**。

### 单用户限制（`user_max_concurrent`）

单个用户同时运行任务上限。默认：**2 个任务**。

防止单个用户占用全部资源。

### 更新限制

管理员可通过队列面板或 API 更新限制：

```bash
curl -X POST http://localhost:8000/api/queue/concurrency \
  -H "Content-Type: application/json" \
  -d '{"max_concurrent": 10, "user_max_concurrent": 3}'
```

## 队列中的任务生命周期

1. **Submitted** - 任务创建，`pending` 状态加入队列
2. **Pending** - 等待空闲槽位（并发限制）
3. **Running** - 在云 GPU 上训练
4. **Completed/Failed** - 终态，从活跃队列移除

## API 端点

### 列出队列条目

```http
GET /api/queue
```

参数：
- `status` - 按状态过滤（pending, running, blocked）
- `limit` - 返回最大数量（默认 50）
- `include_completed` - 包含已完成任务

### 队列统计

```http
GET /api/queue/stats
```

返回：
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

### 我的队列状态

```http
GET /api/queue/me
```

返回当前用户的队列位置、待处理任务与运行任务。

**响应：**

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
| `pending_count` | int | 队列中等待任务数量 |
| `running_count` | int | 当前运行任务数量 |
| `blocked_count` | int | 需要审批的任务数量 |
| `best_position` | int or null | 最高优先级等待任务的位置 |
| `pending_jobs` | array | 等待任务详情列表 |
| `running_jobs` | array | 运行任务详情列表 |

`best_position` 表示用户最高优先级（或最早提交）的等待任务在队列中的位置，便于了解下一个任务启动时间。`null` 表示没有等待任务。

### 任务位置

```http
GET /api/queue/position/{job_id}
```

返回指定任务的队列位置。

### 取消队列任务

```http
POST /api/queue/{job_id}/cancel
```

取消尚未开始的任务。

### 批准阻塞任务

```http
POST /api/queue/{job_id}/approve
```

管理员专用。批准需要审批的任务（例如超出成本阈值）。

### 拒绝阻塞任务

```http
POST /api/queue/{job_id}/reject?reason=<reason>
```

管理员专用。拒绝阻塞任务并给出原因。

### 更新并发限制

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

### 触发处理

```http
POST /api/queue/process
```

管理员专用。手动触发队列处理（通常自动）。

### 清理旧条目

```http
POST /api/queue/cleanup?days=30
```

管理员专用。删除指定天数之前的已完成条目。
