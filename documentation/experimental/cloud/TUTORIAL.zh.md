# 云端训练教程

本指南讲解如何在云 GPU 基础设施上运行 SimpleTuner 训练任务，涵盖 Web UI 与 REST API 两种流程。

## 前置条件

- 已安装 SimpleTuner 并运行服务器（参见[本地 API 教程](../../api/TUTORIAL.md#start-the-server)）
- 本地已准备带字幕的数据集（与本地训练相同的[数据集要求](../../api/TUTORIAL.md#optional-upload-datasets-over-the-api-local-backends)）
- 云服务商账号（参见[支持的提供方](#provider-setup)）
- 使用 API 时：安装 `curl` 和 `jq`

## 提供方设置 {#provider-setup}

云端训练需要你选择的提供方凭据。按以下指南完成配置：

| 提供方 | 设置指南 |
|----------|-------------|
| Replicate | [REPLICATE.md](REPLICATE.md#quick-start) |

配置完成后返回本教程提交任务。

## 快速开始

配置好提供方后：

1. 打开 `http://localhost:8001`，进入 **Cloud** 选项卡
2. 在 **Settings**（齿轮图标）中点击 **Validate** 验证凭据
3. 在 Model/Training/Dataloader 选项卡配置训练
4. 点击 **Train in Cloud**
5. 查看上传摘要后点击 **Submit**

## 训练模型交付

训练完成后需要指定模型的交付位置。请在首次任务前配置以下任一方式。

### 选项 1：HuggingFace Hub（推荐）

直接推送到你的 HuggingFace 账号：

1. 获取有写权限的 [HuggingFace Token](https://huggingface.co/settings/tokens)
2. 设置环境变量：
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```
3. 在 **Publishing** 选项卡启用 “Push to Hub”，并设置仓库名

### 选项 2：通过 Webhook 本地下载

将模型回传到本机，需要对外暴露服务器。

1. 启动隧道：
   ```bash
   ngrok http 8001   # 或：cloudflared tunnel --url http://localhost:8001
   ```
2. 复制公网 URL（如 `https://abc123.ngrok.io`）
3. Cloud 选项卡 → Settings → Webhook URL 粘贴该 URL
4. 模型会保存到 `~/.simpletuner/cloud_outputs/`

### 选项 3：外部 S3

上传到任意 S3 兼容端点（AWS S3、MinIO、Backblaze B2、Cloudflare R2）：

1. 在 **Publishing** 选项卡中配置 S3 设置
2. 填写端点、桶、访问密钥、密钥

## Web UI 流程

### 提交任务

1. 在 Model/Training/Dataloader 选项卡 **配置训练**
2. 切换到 **Cloud** 选项卡并选择提供方
3. 点击 **Train in Cloud** 打开预提交对话框
4. **查看上传摘要** — 本地数据集会被打包并上传
5. 可选：设置运行名称用于追踪
6. 点击 **Submit**

### 监控任务

任务列表会显示所有云端与本地任务：

- **状态指示**：Queued → Running → Completed/Failed
- **实时进度**：训练步骤、loss 值（可用时）
- **成本跟踪**：基于 GPU 时间的预估成本

点击任务查看详情：
- 任务配置快照
- 实时日志（点击 **View Logs**）
- 操作：Cancel、Delete（完成后）

### Settings 面板

点击齿轮图标可访问：

- **API Key 验证**与账户状态
- **Webhook URL**（本地模型交付）
- **成本上限**（避免超额支出）
- **硬件信息**（GPU 类型、小时费用）

## API 流程

### 提交任务

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-training-config",
    "tracker_run_name": "api-test-run"
  }' | jq
```

将 `PROVIDER` 替换为你的提供方名称（如 `replicate`）。

或使用内联配置提交：

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config": {
      "--model_family": "flux",
      "--model_type": "lora",
      "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
      "--output_dir": "/outputs/flux-lora",
      "--max_train_steps": 1000,
      "--lora_rank": 16
    },
    "dataloader_config": [
      {
        "id": "training-images",
        "type": "local",
        "dataset_type": "image",
        "instance_data_dir": "/data/datasets/my-dataset",
        "caption_strategy": "textfile",
        "resolution": 1024
      }
    ]
  }' | jq
```

### 监控任务状态

```bash
# 获取任务详情
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID | jq

# 列出所有任务
curl -s 'http://localhost:8001/api/cloud/jobs?limit=10' | jq

# 同步活跃任务状态
curl -s 'http://localhost:8001/api/cloud/jobs?sync_active=true' | jq
```

### 获取任务日志

```bash
curl -s http://localhost:8001/api/cloud/jobs/JOB_ID/logs | jq '.logs'
```

### 取消运行中的任务

```bash
curl -s -X POST http://localhost:8001/api/cloud/jobs/JOB_ID/cancel | jq
```

### 删除已完成的任务

```bash
curl -s -X DELETE http://localhost:8001/api/cloud/jobs/JOB_ID | jq
```

## CI/CD 集成

### 幂等任务提交

通过幂等键防止重复任务：

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name_to_load": "my-config",
    "idempotency_key": "ci-build-12345"
  }' | jq
```

在 24 小时内重复提交相同 key 会返回已有任务，而不是新建。

### GitHub Actions 示例

```yaml
name: Cloud Training

on:
  push:
    branches: [main]
    paths:
      - 'training-configs/**'

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Submit Training Job
        env:
          SIMPLETUNER_URL: ${{ secrets.SIMPLETUNER_URL }}
        run: |
          RESPONSE=$(curl -s -X POST "$SIMPLETUNER_URL/api/cloud/jobs/submit?provider=replicate" \
            -H 'Content-Type: application/json' \
            -d '{
              "config_name_to_load": "production-lora",
              "idempotency_key": "gh-${{ github.sha }}",
              "tracker_run_name": "gh-run-${{ github.run_number }}"
            }')

          JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
          echo "Submitted job: $JOB_ID"
          echo "JOB_ID=$JOB_ID" >> $GITHUB_ENV

      - name: Wait for Completion
        run: |
          while true; do
            STATUS=$(curl -s "$SIMPLETUNER_URL/api/cloud/jobs/$JOB_ID" | jq -r '.job.status')
            echo "Job status: $STATUS"

            case $STATUS in
              completed) exit 0 ;;
              failed|cancelled) exit 1 ;;
              *) sleep 60 ;;
            esac
          done
```

### API Key 认证

自动化流水线建议使用 API Key，而不是会话认证。

**通过 UI：** Cloud 选项卡 → Settings → API Keys → Create New Key

**通过 API：**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/auth/api-keys' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer YOUR_SESSION_TOKEN' \
  -d '{
    "name": "ci-pipeline",
    "expires_days": 90,
    "scoped_permissions": ["job.submit", "job.view.own"]
  }'
```

完整密钥只会返回一次，请妥善保管。

**使用 API Key：**

```bash
curl -s -X POST 'http://localhost:8001/api/cloud/jobs/submit?provider=PROVIDER' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer stk_abc123...' \
  -d '{...}'
```

**权限范围：**

| 权限 | 说明 |
|------------|-------------|
| `job.submit` | 提交新任务 |
| `job.view.own` | 查看自己的任务 |
| `job.cancel.own` | 取消自己的任务 |
| `job.view.all` | 查看所有任务（管理员） |

## 故障排查

提供方相关问题（凭据、排队、硬件）请查看提供方文档：

- [Replicate Troubleshooting](REPLICATE.md#troubleshooting)

### 常见问题

**数据上传失败**
- 确认数据集路径存在且可读
- 检查打包 zip 的磁盘空间
- 查看浏览器控制台或 API 响应中的错误

**Webhook 未接收事件**
- 确保本地实例可被公网访问（隧道运行中）
- 确认 Webhook URL 正确（包含 `https://`）
- 查看 SimpleTuner 终端输出中的 webhook 处理错误

## API 参考

### 与提供方无关的端点

| 端点 | 方法 | 描述 |
|----------|--------|-------------|
| `/api/cloud/jobs` | GET | 列出任务（支持过滤） |
| `/api/cloud/jobs/submit` | POST | 提交新训练任务 |
| `/api/cloud/jobs/sync` | POST | 从提供方同步任务 |
| `/api/cloud/jobs/{id}` | GET | 获取任务详情 |
| `/api/cloud/jobs/{id}/logs` | GET | 获取任务日志 |
| `/api/cloud/jobs/{id}/cancel` | POST | 取消运行中任务 |
| `/api/cloud/jobs/{id}` | DELETE | 删除已完成任务 |
| `/api/metrics` | GET | 获取任务和成本指标 |
| `/api/cloud/metrics/cost-limit` | GET | 获取成本上限状态 |
| `/api/cloud/providers/{provider}` | PUT | 更新提供方设置 |
| `/api/cloud/storage/{bucket}/{key}` | PUT | S3 兼容上传端点 |

提供方特定端点请参见：
- [Replicate API Reference](REPLICATE.md#api-reference)

完整 schema 详见 `http://localhost:8001/docs` 的 OpenAPI 文档。

## 另请参阅

- [README.md](README.md) – 架构概览与提供方状态
- [REPLICATE.md](REPLICATE.md) – Replicate 提供方设置与细节
- [ENTERPRISE.md](../server/ENTERPRISE.md) – SSO、审批与治理
- [端到端云端运维教程](OPERATIONS_TUTORIAL.md) – 生产部署与监控
- [端到端本地 API 教程](../../api/TUTORIAL.md) – 通过 API 进行本地训练
- [Dataloader Configuration](../../DATALOADER.md) – 数据集配置参考
