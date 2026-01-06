# Worker 编排

SimpleTuner 的 Worker 编排允许将训练任务分发到多台 GPU 机器。Worker 会注册到中央编排器，实时接收任务并回报状态。

## 概览

编排器/Worker 架构支持：

- **分布式训练** - 在任意 GPU 机器上运行任务
- **自动发现** - Worker 自动上报 GPU 能力
- **实时分发** - 通过 SSE（Server-Sent Events）派发任务
- **混合资源池** - 云端短生命周期 Worker 与本地常驻机器混用
- **容错** - 孤立任务会自动重新入队

## Worker 类型

| 类型 | 生命周期 | 使用场景 |
|------|-----------|----------|
| **Ephemeral** | 任务完成后关闭 | 云端抢占式实例（RunPod、Vast.ai） |
| **Persistent** | 任务之间保持在线 | 本地 GPU、预留实例 |

## 快速开始

### 1. 启动编排器

在中央机器上启动 SimpleTuner 服务器：

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

生产环境启用 SSL：

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. 创建 Worker Token

**Web UI：** Administration → Workers → Create Worker

**API：**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-worker-1",
    "worker_type": "persistent",
    "labels": {"location": "datacenter-a", "gpu_type": "a100"}
  }'
```

响应中包含 token（仅显示一次）：

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. 启动 Worker

在 GPU 机器上执行：

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

或者使用环境变量：

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

Worker 将执行：

1. 连接编排器
2. 上报 GPU 能力（自动检测）
3. 进入任务分发循环
4. 每 30 秒发送心跳

### 4. 提交任务给 Worker

**Web UI：** 配置训练后点击 **Train in Cloud** → 选择 **Worker**

**API：**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

目标选项：

| 目标 | 行为 |
|--------|----------|
| `worker` | 仅派发到远程 Worker |
| `local` | 在编排器本机 GPU 上运行 |
| `auto` | 优先 Worker，无则本地 |

## CLI 参考

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   编排器面板 URL（或 SIMPLETUNER_ORCHESTRATOR_URL）
  --worker-token TOKEN     认证 token（或 SIMPLETUNER_WORKER_TOKEN）
  --name NAME              Worker 名称（默认：主机名）
  --persistent             任务之间保持在线（默认：ephemeral）
  -v, --verbose            启用调试日志
```

### Ephemeral 与 Persistent

**Ephemeral（默认）：**
- 完成一个任务后关闭
- 适合按分钟计费的云端实例
- 编排器会在 1 小时后清理离线的 ephemeral Worker

**Persistent（`--persistent`）：**
- 持续在线等待新任务
- 断线后自动重连
- 适合本地 GPU 或预留实例

## Worker 生命周期

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECTING │ ──▶ │    IDLE     │ ──▶ │    BUSY     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  DRAINING   │     │   OFFLINE   │
                    └─────────────┘     └─────────────┘
```

| 状态 | 描述 |
|--------|-------------|
| `CONNECTING` | Worker 正在建立连接 |
| `IDLE` | 已就绪等待任务 |
| `BUSY` | 正在运行任务 |
| `DRAINING` | 完成当前任务后关闭 |
| `OFFLINE` | 断开连接（心跳超时） |

## 健康监控

编排器监控 Worker 状态：

- **心跳间隔：** 30 秒（Worker → 编排器）
- **超时阈值：** 120 秒未心跳即标记离线
- **健康检查循环：** 编排器每 60 秒执行一次

### 故障处理

**Worker 在任务中掉线：**

1. 心跳超时后任务标记失败
2. 如仍有重试次数（默认 3 次）则重新入队
3. 下一个可用 Worker 接管任务

**编排器重启：**

1. Worker 自动重连
2. Worker 报告进行中的任务
3. 编排器协调状态并继续

## GPU 匹配

Worker 在注册时上报 GPU 能力：

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

任务可指定 GPU 需求：

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

调度器按以下规则匹配：

1. GPU 数量需求
2. 标签匹配（支持 glob 模式）
3. Worker 可用性（IDLE）

## 标签

标签用于灵活选择 Worker：

**创建 Worker 时设置标签：**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "worker-1",
    "labels": {
      "location": "us-west",
      "gpu_type": "a100",
      "team": "nlp"
    }
  }'
```

**按标签选择 Worker：**

```bash
# 匹配 team=nlp 的 Worker
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# 匹配 gpu_type 以 "a100" 开头的 Worker
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## 管理操作

### 列出 Worker

```bash
curl -s http://localhost:8001/api/admin/workers | jq
```

响应：

```json
{
  "workers": [
    {
      "id": "w_abc123",
      "name": "gpu-worker-1",
      "status": "idle",
      "worker_type": "persistent",
      "gpu_count": 2,
      "gpu_name": "A100",
      "labels": {"location": "datacenter-a"},
      "last_heartbeat": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### Drain Worker

优雅完成当前任务并停止接收新任务：

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

Worker 将：

1. 完成当前任务
2. 进入 DRAINING 状态
3. 拒绝新任务
4. 任务完成后断开连接（ephemeral）或保持 draining（persistent）

### 轮换 Token

重新生成 Worker 认证 token：

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

旧 token 会立即失效，请在 Worker 端更新新 token。

### 删除 Worker

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

仅在 Worker 离线时可删除。

## 安全性

### Token 认证

- Worker 通过 `X-Worker-Token` 头认证
- Token 存储前会进行 SHA-256 哈希
- Token 创建后不会离开编排器
- 建议定期轮换 token

### 网络安全

生产环境建议：

1. 使用 `--ssl` 或由反向代理终止 TLS
2. 限制 Worker 注册仅来自可信网络
3. 防火墙限制 `/api/workers/*` 端点访问

### 审计日志

所有 Worker 操作都会记录：

- 注册尝试（成功/失败）
- 任务分发事件
- 状态变化
- Token 轮换
- 管理操作

日志访问参见 [Audit Guide](AUDIT.md)。

## 故障排查

### Worker 无法连接

**“Connection refused”**
- 检查编排器 URL 与端口
- 确认防火墙允许入站连接
- 确认编排器使用 `--host 0.0.0.0` 启动

**“Invalid token”**
- Token 可能已轮换
- 检查 Token 字符串是否有空白

**“SSL certificate verify failed”**
- 自签名证书可使用 `--ssl-no-verify`（仅开发环境）
- 或将 CA 证书加入系统信任

### Worker 意外离线

**心跳超时（120 秒）**
- 检查 Worker 与编排器之间的网络稳定性
- 检查 Worker 机器资源耗尽（CPU/内存）
- 网络不稳定时可增大 `SIMPLETUNER_HEARTBEAT_TIMEOUT`

**进程崩溃**
- 查看 Worker 日志中的 Python 异常
- 确认 GPU 驱动正常（`nvidia-smi`）
- 确保训练所需磁盘空间充足

### 任务未派发到 Worker

**无空闲 Worker**
- 在管理面板检查 Worker 状态
- 确认 Worker 已连接且处于 IDLE
- 检查任务与 Worker 标签是否匹配

**GPU 要求不满足**
- 任务需要的 GPU 数量超过任何 Worker
- 调整训练配置中的 `--num_processes`

## API 参考

### Worker 端点（Worker → 编排器）

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/workers/register` | POST | 注册并上报能力 |
| `/api/workers/stream` | GET | 任务分发 SSE 流 |
| `/api/workers/heartbeat` | POST | 周期性心跳 |
| `/api/workers/job/{id}/status` | POST | 上报任务进度 |
| `/api/workers/disconnect` | POST | 优雅断开通知 |

### 管理端点（需要 `admin.workers` 权限）

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/admin/workers` | GET | 列出所有 Worker |
| `/api/admin/workers` | POST | 创建 Worker Token |
| `/api/admin/workers/{id}` | DELETE | 删除 Worker |
| `/api/admin/workers/{id}/drain` | POST | Drain Worker |
| `/api/admin/workers/{id}/token` | POST | 轮换 Token |

## 另请参阅

- [Enterprise Guide](ENTERPRISE.md) - SSO、配额、审批流程
- [Job Queue](../../JOB_QUEUE.md) - 队列调度与优先级
- [Cloud Training](../cloud/README.md) - 云提供方集成
- [API Tutorial](../../api/TUTORIAL.md) - REST API 本地训练
