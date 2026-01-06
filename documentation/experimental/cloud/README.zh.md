# 云训练系统

> **状态：** Experimental
>
> **可用：** Web UI（Cloud 选项卡）

SimpleTuner 的云训练系统支持在云 GPU 提供商上运行训练作业，无需自建基础设施。系统采用可插拔设计，可随着时间增加更多提供商。

## 概述

云训练系统提供：

- **统一任务追踪** - 在一个地方追踪本地与云端训练任务
- **自动数据打包** - 本地数据集自动打包并上传
- **结果交付** - 训练模型可发送到 HuggingFace、S3 或本地下载
- **成本追踪** - 按提供商监控支出与任务成本，并设置花费上限
- **配置快照** - 可选用 git 对训练配置进行版本控制

## 关键概念

使用云训练前，请先理解以下三点：

### 1. 你的数据会发生什么

提交云任务时：

1. **数据集打包** - 本地数据集（`type: "local"`）会被压缩并显示摘要
2. **上传到提供商** - 同意后压缩包直接上传到云提供商
3. **训练执行** - 模型可能需要先下载，再在云 GPU 上训练
4. **数据删除** - 训练后上传数据会从提供商服务器删除，并交付模型

**安全说明：**
- API token 不会离开你的机器
- 敏感文件（.env、.git、credentials）会自动排除
- 每次提交前你都会审阅并确认上传

### 2. 如何接收训练模型

训练产生的模型需要目标位置。配置以下之一：

| 目标 | 设置 | 最佳用途 |
|-------------|-------|----------|
| **HuggingFace Hub** | 设置 `HF_TOKEN` 环境变量，并在 Publishing 选项卡启用 | 共享模型、易访问 |
| **本地下载** | 设置 webhook URL，并用 ngrok 暴露服务器 | 隐私、本地工作流 |
| **S3 存储** | 在 Publishing 选项卡配置端点 | 团队共享、归档 |

详细步骤请参见 [Receiving Trained Models](TUTORIAL.md#receiving-trained-models)。

### 3. 成本模型

Replicate 按 GPU 时间（秒）计费：

| 硬件 | VRAM | 成本 | 典型 LoRA（2000 steps） |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**计费开始** 于训练开始，**计费结束** 于完成或失败。

**保护自己：**
- 在 Cloud 设置中设置支出上限
- 每次提交前显示成本估算
- 可随时取消运行中的任务（只为已用时间付费）

价格与上限详见 [Costs](REPLICATE.md#costs)。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Cloud Tab)                       │
├─────────────────────────────────────────────────────────────────┤
│  Job List  │  Metrics/Charts  │  Actions/Config  │  Job Details │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Cloud API Routes  │
                    │   /api/cloud/*      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│    JobStore     │   │ Upload Service  │   │    Provider     │
│  (Persistence)  │   │ (Data Packaging)│   │   Clients       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                     │
                             ┌───────────────────────┤
                             │                       │
                   ┌─────────▼─────────┐   ┌─────────▼─────────┐
                   │     Replicate     │   │  SimpleTuner.io   │
                   │    Cog Client     │   │   (Coming Soon)   │
                   └───────────────────┘   └───────────────────┘
```

## 支持的提供商

| 提供商 | 状态 | 特性 |
|----------|--------|----------|
| [Replicate](REPLICATE.md) | Supported | 成本追踪、实时日志、Webhook |
| [Worker Orchestration](../server/WORKERS.md) | Supported | 自托管分布式 worker、任意 GPU |
| SimpleTuner.io | Coming Soon | SimpleTuner 团队的托管训练服务 |

### Worker Orchestration

自托管多机分布式训练请参考 [Worker Orchestration Guide](../server/WORKERS.md)。Worker 可运行在：

- 本地 GPU 服务器
- 云 VM（任意提供商）
- 抢占式实例（RunPod、Vast.ai、Lambda Labs）

Worker 注册到 SimpleTuner orchestrator 后会自动接收任务。

## 数据流

### 提交任务

1. **配置准备** - 序列化训练配置
2. **数据打包** - 本地数据集（`type: "local"`）被压缩
3. **上传** - 压缩包上传至 Replicate 文件托管
4. **提交** - 将任务提交到云提供商
5. **追踪** - 轮询任务状态并实时更新

### 接收结果

结果可通过以下方式交付：

1. **HuggingFace Hub** - 推送训练模型到你的 HuggingFace 账户
2. **S3 兼容存储** - 上传到任意 S3 端点（AWS、MinIO 等）
3. **本地下载** - SimpleTuner 提供内置 S3 兼容端点用于本地接收

## 数据隐私与同意

提交云任务时，SimpleTuner 可能上传：

- **训练数据集** - `type: "local"` 的图片/文件
- **配置** - 训练参数（学习率、模型设置等）
- **字幕/元数据** - 与数据集关联的文本文件

数据会直接上传到云提供商（如 Replicate 的文件托管），不会经过 SimpleTuner 服务器。

### 同意设置

在 Cloud 选项卡中可配置数据上传行为：

| 设置 | 行为 |
|---------|----------|
| **Always Ask** | 每次上传前弹出确认对话框列出数据集 |
| **Always Allow** | 跳过确认（可信工作流） |
| **Never Upload** | 禁用云训练（仅本地） |

## 本地 S3 端点

SimpleTuner 内置 S3 兼容端点用于接收训练模型：

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

文件默认保存到 `~/.simpletuner/cloud_outputs/`。

你可以手动配置凭证；若不配置，会为每个训练任务自动生成临时凭证，这是推荐方式。

这使 “仅下载” 模式成为可能：
1. 设置 webhook URL 指向本地 SimpleTuner 实例
2. SimpleTuner 自动配置 S3 发布设置
3. 训练模型回传到本机

**注记：** 需要通过 ngrok、cloudflared 等将本地 SimpleTuner 暴露给云提供商访问。

## 添加新提供商

云系统可扩展。新增提供商需：

1. 创建实现 `CloudTrainerService` 的客户端类：

```python
from .base import CloudTrainerService, CloudJobInfo, CloudJobStatus

class NewProviderClient(CloudTrainerService):
    @property
    def provider_name(self) -> str:
        return "new_provider"

    @property
    def supports_cost_tracking(self) -> bool:
        return True  # or False

    @property
    def supports_live_logs(self) -> bool:
        return True  # or False

    async def validate_credentials(self) -> Dict[str, Any]:
        # Validate API key and return user info
        ...

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        # List recent jobs from the provider
        ...

    async def run_job(self, config, dataloader, ...) -> CloudJobInfo:
        # Submit a new training job
        ...

    async def cancel_job(self, job_id: str) -> bool:
        # Cancel a running job
        ...

    async def get_job_logs(self, job_id: str) -> str:
        # Fetch logs for a job
        ...

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        # Get current status of a job
        ...
```

2. 在 cloud routes 中注册提供商
3. 为新提供商添加 UI 选项卡

## 文件与位置

| 路径 | 说明 |
|------|-------------|
| `~/.simpletuner/cloud/` | 云相关状态与任务历史 |
| `~/.simpletuner/cloud/job_history.json` | 统一任务追踪数据库 |
| `~/.simpletuner/cloud/provider_configs/` | 提供商配置 |
| `~/.simpletuner/cloud_outputs/` | 本地 S3 端点存储 |

## 排障

### "REPLICATE_API_TOKEN not set"

启动 SimpleTuner 前设置环境变量：

```bash
export REPLICATE_API_TOKEN="r8_..."
simpletuner --webui
```

### 数据上传失败

- 检查网络连接
- 确认数据集路径存在
- 查看浏览器控制台错误
- 确认云提供商余额与权限

### Webhook 未接收到结果

- 确保本地实例可公开访问
- 检查 webhook URL 是否正确
- 确认防火墙允许入站连接

## 当前限制

云训练系统面向 **一次性训练任务**。以下特性暂不支持：

### 工作流/管道任务（DAG）

SimpleTuner 不支持任务依赖或多步工作流（一个任务输出作为另一个任务输入）。每个任务独立自包含。

**若需要工作流：**
- 使用外部编排工具（Airflow、Prefect、Dagster）
- 通过 REST API 在管道中串联任务
- Airflow 集成示例见 [ENTERPRISE.md](../server/ENTERPRISE.md#external-orchestration-airflow)

### 恢复训练

目前不支持中断/失败/提前停止的训练恢复。任务失败或取消后：
- 必须从头重新提交
- 不会从云存储自动恢复检查点

**变通方式：**
- 配置频繁推送至 HuggingFace Hub（`--push_checkpoints_to_hub`）保存中间检查点
- 下载输出并重新上传作为新任务的起点以实现自定义管理
- 对关键长时间任务考虑分段训练

这些限制可能在未来版本中改进。

## 参见

### 云训练

- [Cloud Training Tutorial](TUTORIAL.md) - 入门指南
- [Replicate Integration](REPLICATE.md) - Replicate 设置
- [Job Queue](../../JOB_QUEUE.md) - 任务调度与并发
- [Operations Guide](OPERATIONS_TUTORIAL.md) - 生产部署

### 多用户功能（本地与云通用）

- [Enterprise Guide](../server/ENTERPRISE.md) - SSO、审批与治理
- [External Authentication](../server/EXTERNAL_AUTH.md) - OIDC 与 LDAP 设置
- [Audit Logging](../server/AUDIT.md) - 安全事件审计

### 通用

- [Local API Tutorial](../../api/TUTORIAL.md) - 通过 REST API 进行本地训练
- [Datasets Documentation](../../DATALOADER.md) - 理解 dataloader 配置
