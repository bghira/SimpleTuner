# SimpleTuner 的 Docker

此 Docker 配置为 SimpleTuner 提供了在 Runpod、Vast.ai 以及其他 Docker 兼容主机上运行的完整环境。它针对易用性与稳定性进行了优化，集成了机器学习项目所需的工具与库。

## 容器特性

- **CUDA 基础镜像**：基于 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` 构建，支持 GPU 加速。
- **开发工具**：包含 Git、SSH，以及 `tmux`、`vim`、`htop` 等常用工具。
- **Python 与库**：内置 Python 3.10，并通过 pip 预装 SimpleTuner。
- **Huggingface 与 WandB 集成**：预配置 Huggingface Hub 与 WandB，便于模型共享与实验跟踪。

## 快速开始

### 通过 WSL 支持 Windows（实验性）

以下指南在安装了 Dockerengine 的 WSL2 发行版中测试。


### 1. 构建容器

克隆仓库并进入包含 Dockerfile 的目录，使用以下命令构建镜像：

```bash
docker build -t simpletuner .
```

### 2. 运行容器

以 GPU 支持运行容器：

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

该命令启用 GPU 并映射 SSH 端口以便外部访问。

### 3. 环境变量

为便于与外部工具集成，容器支持 Huggingface 与 WandB 的 token 环境变量，运行时传入：

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. 数据卷

为实现持久化存储与宿主机/容器数据共享，可挂载数据卷：

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. SSH 访问

容器已默认配置 SSH。请通过环境变量提供 SSH 公钥（Vast.ai 使用 `SSH_PUBLIC_KEY`，Runpod 使用 `PUBLIC_KEY`）。

### 6. 使用 SimpleTuner

SimpleTuner 已预装，可直接运行训练命令：

```bash
simpletuner configure
simpletuner train
```

配置与设置请参考 [安装文档](INSTALL.md) 和 [quickstart 指南](QUICKSTART.md)。

## 额外配置

### 自定义脚本与配置

如需添加自定义启动脚本或修改配置，可扩展入口脚本（`docker-start.sh`）以满足需求。

如果此配置无法满足某些能力，请提交新的 issue。

### Docker Compose

若偏好 `docker-compose.yaml`，此处提供可扩展模板。

部署完成后，可按上面的步骤连接容器并开始操作。

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ 请谨慎处理 WandB 与 Hugging Face token！建议不要将其提交到版本库，即便是私有库也不安全。生产环境应使用密钥管理存储，但超出本指南范围。
---

## 故障排查

### CUDA 版本不匹配

**症状**：应用无法使用 GPU，或在运行 GPU 加速任务时出现 CUDA 库错误。

**原因**：容器内 CUDA 版本与宿主机 CUDA 驱动版本不匹配。

**解决方案**：
1. **检查宿主机 CUDA 驱动版本**：在宿主机运行：
   ```bash
   nvidia-smi
   ```
   输出右上角会显示 CUDA 版本。

2. **匹配容器 CUDA 版本**：确保 Docker 镜像中的 CUDA 工具包版本与宿主机驱动兼容。NVIDIA 通常支持前向兼容，但应查看 NVIDIA 官方兼容矩阵。

3. **重新构建镜像**：必要时修改 Dockerfile 的基础镜像，使其与宿主机 CUDA 版本匹配。例如宿主机为 CUDA 11.2，而容器为 CUDA 11.8，可更换基础镜像：
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   修改后重新构建镜像。

### SSH 连接问题

**症状**：无法通过 SSH 连接容器。

**原因**：SSH key 配置错误或 SSH 服务未正确启动。

**解决方案**：
1. **检查 SSH 配置**：确认 SSH 公钥已正确添加到容器内 `~/.ssh/authorized_keys`。进入容器后检查 SSH 服务状态：
   ```bash
   service ssh status
   ```
2. **端口映射**：确认 SSH 端口（22）已正确暴露并映射，启动命令如：
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### 通用建议

- **日志与输出**：查看容器日志与输出中的错误或警告信息，以获取更多上下文。
- **文档与论坛**：查阅 Docker 与 NVIDIA CUDA 文档。与你使用的软件或依赖相关的社区论坛和 issue tracker 也很有帮助。
