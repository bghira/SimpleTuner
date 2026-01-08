# 发布（Publishing）提供方

SimpleTuner 通过 `--publishing_config` 将训练输出发布到多个目标。Hugging Face 上传仍由 `--push_to_hub` 控制；`publishing_config` 是对其他提供方的补充，会在主进程完成验证后执行。

## 配置格式
- 支持内联 JSON（`--publishing_config='[{"provider": "s3", ...}]'`）、通过 SDK 传入的 Python dict，或 JSON 文件路径。
- 值会规范化为列表，行为与 `--webhook_config` 一致。
- 每个条目都需要 `provider` 键。可选 `base_path` 用于在远端路径前添加前缀。如果配置无法返回 URI，提供方在被查询时会记录一次性警告。

## 默认产物

发布会上传运行的 `output_dir`（目录与文件），使用该目录的基准名称。元数据包含当前任务 ID 与验证类型，便于下游将 URI 关联到具体运行。

## 提供方

使用某个提供方前，请在项目 `.venv` 中安装对应可选依赖。

### S3 兼容与 Backblaze B2（S3 API）
- 提供方：`s3` 或 `backblaze_b2`
- 依赖：`pip install boto3`
- 示例：
```json
[
  {
    "provider": "s3",
    "bucket": "simpletuner-models",
    "region": "us-east-1",
    "access_key": "AKIA...",
    "secret_key": "SECRET",
    "base_path": "runs/2024",
    "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
    "public_base_url": "https://cdn.example.com/models"
  }
]
```

⚠️ **安全提示**：不要将凭据提交到版本控制。生产环境请使用环境变量替换或密钥管理。

### Azure Blob Storage
- 提供方：`azure_blob`（别名 `azure`）
- 依赖：`pip install azure-storage-blob`
- 示例：
```json
[
  {
    "provider": "azure_blob",
    "connection_string": "DefaultEndpointsProtocol=....",
    "container": "simpletuner",
    "base_path": "models/latest"
  }
]
```

### Dropbox
- 提供方：`dropbox`
- 依赖：`pip install dropbox`
- 示例：
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
大文件会自动使用上传会话进行流式上传；若允许，会创建共享链接，否则记录 `dropbox://` 路径。

## CLI 用法
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```

若以编程方式调用 SimpleTuner，可向 `publishing_config` 传入 list/dict，系统会自动规范化。
