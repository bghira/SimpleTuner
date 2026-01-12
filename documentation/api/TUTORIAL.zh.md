# API 训练教程

## 简介

本指南介绍如何**完全通过 HTTP API** 运行 SimpleTuner 训练任务，同时将安装和数据集管理保留在命令行中。本教程的结构与其他教程相似，但跳过了 WebUI 的初始配置流程。您将学习：

- 安装并启动统一服务器
- 发现并下载 OpenAPI schema
- 通过 REST 调用创建和更新环境
- 通过 `/api/training` 验证、启动和监控训练任务
- 深入两种经过验证的配置：PixArt Sigma 900M 全量微调和 Flux Kontext LyCORIS LoRA 运行

## 前提条件

- Python 3.10–3.12、Git 和 `pip`
- 在虚拟环境中安装 SimpleTuner（`pip install 'simpletuner[cuda]'` 或与您平台匹配的变体）
- 访问所需的 Hugging Face 仓库（在拉取受限模型之前运行 `huggingface-cli login`）
- 本地已准备好带有标注的数据集（PixArt 使用标注文本文件，Kontext 使用配对的编辑/参考文件夹）
- 带有 `curl` 和 `jq` 的 shell 环境

## 启动服务器 {#start-the-server}

从您的 SimpleTuner 检出目录（或安装了该包的环境）：

```bash
simpletuner server --port 8001
```

API 位于 `http://localhost:8001`。在另一个终端中执行以下命令时，请保持服务器运行。

> **提示：** 如果您已有准备好训练的配置环境，可以使用 `--env` 启动服务器，在服务器完全加载后自动开始训练：
>
> ```bash
> simpletuner server --port 8001 --env my-training-config
> ```
>
> 这会在启动时验证您的配置，并在服务器就绪后立即启动训练——非常适合无人值守或脚本化部署。`--env` 选项的工作方式与 `simpletuner train --env` 完全相同。

### 配置与部署

对于生产环境使用，您可以配置绑定地址和端口：

| 选项 | 环境变量 | 默认值 | 描述 |
|--------|---------------------|---------|-------------|
| `--host` | `SIMPLETUNER_HOST` | `0.0.0.0` | 服务器绑定的地址（在反向代理后使用 `127.0.0.1`） |
| `--port` | `SIMPLETUNER_PORT` | `8001` | 服务器绑定的端口 |

<details>
<summary><b>生产部署选项（TLS、反向代理、Systemd、Docker）</b></summary>

对于生产部署，建议使用反向代理进行 TLS 终止。

#### Nginx 配置

```nginx
server {
    listen 443 ssl http2;
    server_name training.example.com;

    # TLS 配置（示例使用 Let's Encrypt 路径）
    ssl_certificate /etc/letsencrypt/live/training.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/training.example.com/privkey.pem;

    # WebSocket 支持 SSE 流（对实时日志至关重要）
    location /api/training/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        # SSE 特定设置
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }

    # 主应用
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        # 数据集的大文件上传
        client_max_body_size 10G;
        proxy_request_buffering off;
    }
}
```

#### Caddy 配置

```caddyfile
training.example.com {
    reverse_proxy 127.0.0.1:8001 {
        # SSE 流支持
        flush_interval -1
    }
    # 大文件上传
    request_body {
        max_size 10GB
    }
}
```

#### systemd 服务

```ini
[Unit]
Description=SimpleTuner Training Server
After=network.target

[Service]
Type=simple
User=trainer
WorkingDirectory=/home/trainer/simpletuner-workspace
Environment="SIMPLETUNER_HOST=127.0.0.1"
Environment="SIMPLETUNER_PORT=8001"
ExecStart=/home/trainer/simpletuner-workspace/.venv/bin/simpletuner server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### Docker Compose 与 Traefik

```yaml
version: '3.8'
services:
  simpletuner:
    image: ghcr.io/bghira/simpletuner:latest
    command: simpletuner server --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.example.com`)"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8001"
```
</details>

## 身份验证

SimpleTuner 支持多用户身份验证。首次启动时，您需要创建一个管理员账户。

### 首次设置

检查是否需要设置：

```bash
curl -s http://localhost:8001/api/cloud/auth/setup/status | jq
```

如果 `needs_setup` 为 `true`，创建第一个管理员：

```bash
curl -s -X POST http://localhost:8001/api/cloud/auth/setup/first-admin \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "your-secure-password"
  }'
```

### API 密钥

对于脚本化访问，登录后生成 API 密钥：

```bash
# 首先登录（存储会话 cookie）
curl -s -X POST http://localhost:8001/api/cloud/auth/login \
  -H 'Content-Type: application/json' \
  -c cookies.txt \
  -d '{"username": "admin", "password": "your-secure-password"}'

# 创建 API 密钥
curl -s -X POST http://localhost:8001/api/cloud/auth/api-keys \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"name": "automation-key"}' | jq
```

在后续请求中使用返回的密钥（以 `st_` 为前缀）：

```bash
curl -s http://localhost:8001/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

### 用户管理

管理员可以通过 API 或 WebUI 的**用户管理**页面创建其他用户：

```bash
# 创建新用户（需要管理员会话）
curl -s -X POST http://localhost:8001/api/users \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
    "email": "researcher@example.com",
    "username": "researcher",
    "password": "their-password",
    "level_names": ["researcher"]
  }'
```

> **注意：** 默认情况下禁用公开注册。管理员可以在**用户管理 → 注册**选项卡中启用它，但建议私有部署保持禁用状态。

## 发现 API

FastAPI 提供交互式文档和 OpenAPI schema：

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDoc 视图
python -m webbrowser http://localhost:8001/redoc

# 下载 schema 用于本地检查
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

本教程中使用的每个端点都在 `configurations` 和 `training` 标签下有文档说明。

## 快速路径：无需环境即可运行

如果您希望**完全跳过配置/环境管理**，可以通过将完整的 CLI 风格载荷直接发送到训练端点来执行一次性训练运行：

1. 编写或重用描述数据集的 dataloader JSON。训练器只需要 `--data_backend_config` 引用的路径。

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. 验证内联配置。提供每个必需的 CLI 参数（`--model_family`、`--model_type`、`--pretrained_model_name_or_path`、`--output_dir`、`--data_backend_config`，以及 `--num_train_epochs` 或 `--max_train_steps` 之一）：

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   绿色的"Configuration Valid"片段确认训练器将接受该载荷。

3. 使用**相同**的表单字段启动训练（您可以添加覆盖项，如 `--seed` 或 `--validation_prompt`）：

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

服务器会自动将提交的设置与其默认值合并，将解析的配置写入活动文件，然后开始训练。您可以对任何模型系列使用相同的方法——当您需要可重用环境时，后续章节将介绍更完整的工作流程。

### 监控临时运行

您可以通过本指南后面使用的相同状态端点跟踪进度：

- 轮询 `GET /api/training/status` 获取高级状态、活动任务 ID 和启动阶段信息。
- 使用 `GET /api/training/events?since_index=N` 获取增量日志，或通过 `/api/training/events/stream` 的 WebSocket 进行流式传输。

对于推送式更新，请在表单字段中提供 webhook 设置：

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

载荷必须是 JSON 序列化的字符串；服务器会将任务生命周期更新发送到 `callback_url`。有关支持的字段，请参阅 `documentation/OPTIONS.md` 中的 `--webhook_config` 描述或 `config/webhooks.json` 示例模板。

<details>
<summary><b>反向代理的 Webhook 配置</b></summary>

使用带 HTTPS 的反向代理时，您的 webhook URL 必须是公共端点：

1.  **公共服务器：** 使用 `https://training.example.com/webhook/callback`
2.  **隧道：** 使用 ngrok 或 cloudflared 进行本地开发。

**实时日志（SSE）故障排除：**
如果 `GET /api/training/events` 工作正常但流挂起：
*   **Nginx：** 确保 `proxy_buffering off;` 且 `proxy_read_timeout` 设置较高（例如 86400s）。
*   **CloudFlare：** 会终止长连接；使用 CloudFlare Tunnel 或为流端点绕过代理。
</details>

### 触发手动验证

如果您想在计划的验证间隔**之间**强制执行评估，请调用新端点：

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- 服务器响应返回活动的 `job_id`。
- 训练器会在下一次梯度同步后立即排队执行验证运行（不会中断当前的 micro-batch）。
- 该运行重用您配置的验证提示词/设置，因此生成的图像会出现在常规事件/日志流中。
- 要将验证卸载到外部可执行文件而不是内置管道，请在配置（或载荷）中设置 `--validation_method=external-script`，并将 `--validation_external_script` 指向您的脚本。您可以使用占位符将训练上下文传递给脚本：`{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}`、`{remote_checkpoint_path}`（验证时为空），以及任何 `validation_*` 配置值（例如 `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`）。如果您希望脚本"发后即忘"而不阻塞训练，请启用 `--validation_external_background`。
- 想要在每个检查点本地写入后立即触发自动化（即使上传在后台运行）？配置 `--post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'`。它使用与验证钩子相同的占位符；`{remote_checkpoint_path}` 对于此钩子解析为空。
- 更喜欢保留 SimpleTuner 的内置上传并将结果远程 URL 传递给您自己的工具？配置 `--post_upload_script`；它在每个发布提供商/Hugging Face Hub 上传后触发一次，带有 `{remote_checkpoint_path}`（如果后端提供）和相同的上下文占位符。SimpleTuner 不会从您的脚本中获取结果，因此请自行将工件/指标记录到您的 tracker。
  - 示例：`--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'`，其中 `notify.sh` 调用您的 tracker API。
  - 工作示例：
    - `simpletuner/examples/external-validation/replicate_post_upload.py` 使用 `{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}` 和 `{huggingface_path}` 触发 Replicate 推理。
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py` 使用相同的占位符触发 WaveSpeed 推理并轮询完成状态。
    - `simpletuner/examples/external-validation/fal_post_upload.py` 触发 fal.ai Flux LoRA 推理（需要 `FAL_KEY` 且 `model_family` 包含 `flux`）。
    - `simpletuner/examples/external-validation/use_second_gpu.py` 在另一个 GPU 上运行 Flux LoRA 推理，无需上传。

如果没有活动任务，端点返回 HTTP 400，因此在脚本重试时请先检查 `/api/training/status`。

### 触发手动检查点

要立即保存当前模型状态（无需等待下一个计划的检查点），请调用：

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- 服务器响应返回活动的 `job_id`。
- 训练器在下一次梯度同步后保存检查点，使用与计划检查点相同的设置（上传规则、滚动保留等）。
- 滚动清理和 webhook 通知的行为与计划检查点完全相同。

与验证一样，如果没有训练任务正在运行，端点返回 HTTP 400。

### 流式验证预览

暴露 Tiny AutoEncoder（或等效）钩子的模型可以在图像/视频采样过程中发出**逐步验证预览**。通过在载荷中添加 CLI 标志启用此功能：

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …other fields…
```

- `--validation_preview`（默认为 `false`）解锁预览解码器。
- `--validation_preview_steps` 决定发出中间帧的频率。使用上面的示例，您将在步骤 1,5,9,13,17,20 接收事件（第一步始终发出，然后每 4 步一次）。

每个预览作为 `validation.image` 事件发布（参见 `simpletuner/helpers/training/validation.py:899-929`）。您可以通过原始 webhooks、`GET /api/training/events` 或 `/api/training/events/stream` 的 SSE 流来消费它们。典型的载荷如下：

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

支持视频的模型会附加 `videos` 数组（带有 `mime_type: image/gif` 的 GIF data URI）。由于这些事件近乎实时流式传输，您可以直接在仪表板中显示它们，或通过原始 webhook 后端发送到 Slack/Discord。

## 常见 API 工作流程

1. **创建环境** – `POST /api/configs/environments`
2. **填充 dataloader 文件** – 编辑生成的 `config/<env>/multidatabackend.json`
3. **更新训练超参数** – `PUT /api/configs/<env>`
4. **激活环境** – `POST /api/configs/<env>/activate`
5. **验证训练参数** – `POST /api/training/validate`
6. **启动训练** – `POST /api/training/start`
7. **监控或停止任务** – `/api/training/status`、`/api/training/events`、`/api/training/stop`、`/api/training/cancel`

下面的每个示例都遵循此流程。

## 可选：通过 API 上传数据集（本地后端） {#optional-upload-datasets-over-the-api-local-backends}

如果数据集尚未在运行 SimpleTuner 的机器上，您可以在连接 dataloader 之前通过 HTTP 推送它。上传端点遵循配置的 `datasets_dir`（在 WebUI 初始配置期间设置），专用于本地文件系统：

1. **在数据集根目录下创建目标文件夹**：

   ```bash
   DATASETS_DIR=${DATASETS_DIR:-/workspace/simpletuner/datasets}
   curl -s -X POST http://localhost:8001/api/datasets/folders \
     -F parent_path="$DATASETS_DIR" \
     -F folder_name="pixart-upload"
   ```

2. **上传文件或 ZIP**（接受图像以及可选的 `.txt/.jsonl/.csv` 元数据）：

   ```bash
   # 上传 zip（在服务器上自动解压）
   curl -s -X POST http://localhost:8001/api/datasets/upload/zip \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F file=@/path/to/dataset.zip

   # 或上传单个文件
   curl -s -X POST http://localhost:8001/api/datasets/upload \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F files[]=@image001.png \
     -F files[]=@image001.txt
   ```

> **上传故障排除：** 如果使用反向代理时大文件上传失败并显示"Entity Too Large"错误，请确保已增加请求体大小限制（例如，在 Nginx 中使用 `client_max_body_size 10G;` 或在 Caddy 中使用 `request_body { max_size 10GB }`）。

上传完成后，将您的 `multidatabackend.json` 条目指向新文件夹（例如 `"/data/datasets/pixart-upload"`）。

## 示例：PixArt Sigma 900M 全量微调

### 1. 通过 REST 创建环境

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```
```json
{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }
```

这会创建 `config/pixart-api-demo/` 和一个初始的 `multidatabackend.json`。

### 2. 连接数据集

编辑 dataloader 文件（用您实际的数据集/缓存位置替换路径）：

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. 通过 API 更新超参数

获取当前配置，合并覆盖项，然后推送结果：

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. 激活环境

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. 启动前验证

`validate` 使用表单编码数据。至少确保 `num_train_epochs` 或 `max_train_steps` 之一为 0：

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

成功块（`Configuration Valid`）表示训练器接受合并后的配置。

### 6. 开始训练

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

响应包含任务 ID。训练使用步骤 3 中保存的参数运行。

### 7. 监控和停止

```bash
# 查询粗略状态
curl -s http://localhost:8001/api/training/status | jq

# 流式获取增量日志事件
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# 取消或停止
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

PixArt 注意事项：

- 确保数据集足够大，以适应所选的 `train_batch_size * gradient_accumulation_steps`
- 如果需要镜像，请设置 `HF_ENDPOINT`，并在下载 `terminusresearch/pixart-900m-1024-ft-v0.6` 之前进行身份验证
- 根据您的提示词，在 3.6 到 4.4 之间调整 `--validation_guidance`

## 示例：Flux Kontext LyCORIS LoRA

Kontext 与 Flux Dev 共享大部分管道，但需要配对的编辑/参考图像。

### 1. 配置环境

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```
```json
{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }
```

### 2. 描述配对的 dataloader

Kontext 需要编辑/参考对以及文本嵌入缓存：

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

确保编辑和参考文件夹之间的文件名匹配；SimpleTuner 根据名称拼接嵌入。

### 3. 应用 Kontext 特定的超参数

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. 激活、验证和启动

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Kontext 提示：

- `conditioning_type=reference_strict` 保持裁剪对齐；如果您的数据集宽高比不同，请切换到 `reference_loose`
- 量化为 `int8-quanto` 以在 1024 px 时保持在 24 GB VRAM 内；全精度需要 Hopper/Blackwell 级别的 GPU
- 对于多节点运行，在启动服务器之前设置 `--accelerate_config` 或 `CUDA_VISIBLE_DEVICES`

## 使用 GPU 感知队列提交本地任务

在多 GPU 机器上运行时，您可以通过队列 API 提交具有 GPU 分配感知的本地训练任务。如果所需 GPU 不可用，任务将排队。

### 检查 GPU 可用性

```bash
curl -s "http://localhost:8001/api/system/status?include_allocation=true" | jq '.gpu_allocation'
```

响应显示哪些 GPU 可用：

```json
{
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
```

您还可以获取包括本地 GPU 信息的队列统计数据：

```bash
curl -s http://localhost:8001/api/queue/stats | jq '.local'
```

### 提交本地任务

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "no_wait": false,
    "any_gpu": false
  }'
```

选项：

| 选项 | 默认值 | 描述 |
|--------|---------|-------------|
| `config_name` | 必需 | 要运行的训练环境名称 |
| `no_wait` | false | 如果为 true，当 GPU 不可用时立即拒绝 |
| `any_gpu` | false | 如果为 true，使用任何可用 GPU 而不是配置的设备 ID |

响应：

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

`status` 字段指示结果：

- `running` - 任务立即启动，分配了 GPU
- `queued` - 任务已排队，GPU 可用时将启动
- `rejected` - GPU 不可用且 `no_wait` 为 true

### 配置本地并发限制

管理员可以通过队列并发端点限制可使用的本地任务数和 GPU 数：

```bash
# 获取当前限制
curl -s http://localhost:8001/api/queue/stats | jq '{local_gpu_max_concurrent, local_job_max_concurrent}'

# 更新限制（与云端限制一起）
curl -s -X POST http://localhost:8001/api/queue/concurrency \
  -H 'Content-Type: application/json' \
  -d '{
    "local_gpu_max_concurrent": 6,
    "local_job_max_concurrent": 2
  }'
```

将 `local_gpu_max_concurrent` 设置为 `null` 可无限制使用 GPU。

### CLI 替代方案

相同的功能可通过 CLI 使用：

```bash
# 使用默认排队行为提交
simpletuner jobs submit my-config

# GPU 不可用时拒绝
simpletuner jobs submit my-config --no-wait

# 使用任何可用 GPU
simpletuner jobs submit my-config --any-gpu

# 预览会发生什么（模拟运行）
simpletuner jobs submit my-config --dry-run
```

## 将任务分发到远程 worker

如果您有注册为 worker 的远程 GPU 机器（参见 [Worker 编排](../experimental/server/WORKERS.md)），您可以通过队列 API 将任务分发给它们。

### 检查可用的 worker

```bash
curl -s http://localhost:8001/api/admin/workers | jq '.workers[] | {name, status, gpu_name, gpu_count}'
```

### 提交到特定目标

```bash
# 优先远程 worker，回退到本地 GPU（默认）
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "auto"
  }'

# 强制只分发到远程 worker
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'

# 仅在编排器的本地 GPU 上运行
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "local"
  }'
```

### 按标签选择 worker

Worker 可以有用于过滤的标签（例如 GPU 类型、位置、团队）：

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "team": "nlp"}
  }'
```

标签支持 glob 模式（`*` 匹配任意字符）。

## 常用端点一览

- `GET /api/configs/` – 列出环境（传递 `?config_type=model` 获取训练配置）
- `GET /api/configs/examples` – 枚举内置模板
- `POST /api/configs/{name}/dataloader` – 如果您需要默认值，重新生成 dataloader 文件
- `GET /api/training/status` – 高级状态、活动 `job_id` 和启动阶段信息
- `GET /api/training/events?since_index=N` – 增量训练器日志流
- `POST /api/training/checkpoints` – 列出活动任务输出目录的检查点
- `GET /api/system/status?include_allocation=true` – 带有 GPU 分配信息的系统指标
- `GET /api/queue/stats` – 包括本地 GPU 分配的队列统计数据
- `POST /api/queue/submit` – 使用 GPU 感知队列提交本地或 worker 任务
- `POST /api/queue/concurrency` – 更新云端和本地并发限制
- `GET /api/admin/workers` – 列出注册的 worker 及其状态

## 下一步

- 在 `documentation/OPTIONS.md` 中探索具体的选项定义
- 将这些 REST 调用与 `jq`/`yq` 或 Python 客户端结合用于自动化
- 在 `/api/training/events/stream` 连接 WebSocket 用于实时仪表板
- 重用导出的配置（`GET /api/configs/<env>/export`）来版本控制工作设置
- **在云端 GPU 上运行训练**，通过 Replicate——参见[云端训练教程](../experimental/cloud/TUTORIAL.md)

通过这些模式，您可以完全脚本化 SimpleTuner 训练而无需接触 WebUI，同时仍然依赖久经考验的 CLI 设置流程。
