# SimpleTuner

**SimpleTuner** is a multi-modal diffusion model fine-tuning toolkit focused on simplicity and ease of understanding.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Get Started__

    ---

    Install SimpleTuner and train your first model in minutes

    [:octicons-arrow-right-24: Installation](INSTALL.md)

-   :material-cog:{ .lg .middle } __Web UI__

    ---

    Configure and run training through a sleek web interface

    [:octicons-arrow-right-24: Web UI Tutorial](webui/TUTORIAL.md)

-   :material-api:{ .lg .middle } __REST API__

    ---

    Automate training workflows with the HTTP API

    [:octicons-arrow-right-24: API Tutorial](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __Cloud Training__

    ---

    Run training on Replicate or distributed workers

    [:octicons-arrow-right-24: Cloud Training](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __Multi-User__

    ---

    Enterprise features: SSO, quotas, RBAC, worker orchestration

    [:octicons-arrow-right-24: Enterprise Guide](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __Model Guides__

    ---

    Step-by-step guides for Flux, SD3, SDXL, video models, and more

    [:octicons-arrow-right-24: Model Guides](quickstart/index.md)

</div>

## Features

- **Multi-modal training** - Image, video, and audio generation models
- **Web UI & API** - Train via browser or automate with REST
- **Worker orchestration** - Distribute jobs across GPU machines
- **Enterprise-ready** - LDAP/OIDC SSO, RBAC, quotas, audit logging
- **Cloud integration** - Replicate, self-hosted workers
- **Memory optimization** - DeepSpeed, FSDP2, quantization

## Supported Models

| Type | Models |
|------|--------|
| **Image** | Flux.1/2, SD3, SDXL, Chroma, Auraflow, PixArt, Sana, Lumina2, HiDream, and more |
| **Video** | Wan, LTX Video, Hunyuan Video, Kandinsky 5, LongCat |
| **Audio** | ACE-Step |

See [Model Guides](quickstart/index.md) for complete documentation.

## Community

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - Bug reports & feature requests

## License

SimpleTuner is open source software.
