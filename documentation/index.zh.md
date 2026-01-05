# SimpleTuner

**SimpleTuner** 是一个专注于简洁易用的多模态扩散模型微调工具包。

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __快速开始__

    ---

    安装 SimpleTuner 并在几分钟内训练您的第一个模型

    [:octicons-arrow-right-24: 安装指南](INSTALL.zh.md)

-   :material-cog:{ .lg .middle } __Web 界面__

    ---

    通过简洁的 Web 界面配置和运行训练

    [:octicons-arrow-right-24: Web 界面教程](webui/TUTORIAL.md)

-   :material-api:{ .lg .middle } __REST API__

    ---

    使用 HTTP API 自动化训练工作流

    [:octicons-arrow-right-24: API 教程](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __云端训练__

    ---

    在 Replicate 或分布式工作节点上运行训练

    [:octicons-arrow-right-24: 云端训练](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __多用户支持__

    ---

    企业功能：SSO、配额、RBAC、工作节点编排

    [:octicons-arrow-right-24: 企业指南](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __模型指南__

    ---

    Flux、SD3、SDXL、视频模型等的详细指南

    [:octicons-arrow-right-24: 模型指南](quickstart/index.md)

</div>

## 功能特性

- **多模态训练** - 图像、视频和音频生成模型
- **Web 界面和 API** - 通过浏览器或 REST 自动化训练
- **工作节点编排** - 在多个 GPU 机器上分布任务
- **企业就绪** - LDAP/OIDC SSO、RBAC、配额、审计日志
- **云端集成** - Replicate、自托管工作节点
- **内存优化** - DeepSpeed、FSDP2、量化

## 支持的模型

| 类型 | 模型 |
|------|------|
| **图像** | Flux.1/2、SD3、SDXL、Chroma、Auraflow、PixArt、Sana、Lumina2、HiDream 等 |
| **视频** | Wan、LTX Video、Hunyuan Video、Kandinsky 5、LongCat |
| **音频** | ACE-Step |

查看[模型指南](quickstart/index.md)了解完整文档。

## 社区

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - Bug 报告和功能请求

## 许可证

SimpleTuner 是开源软件。
