# SimpleTuner

**SimpleTuner** é um toolkit de fine-tuning de modelos de difusão multimodais focado em simplicidade e facilidade de entendimento.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Comece Agora__

    ---

    Instale o SimpleTuner e treine seu primeiro modelo em minutos

    [:octicons-arrow-right-24: Instalação](INSTALL.md)

-   :material-cog:{ .lg .middle } __Interface Web__

    ---

    Configure e execute o treinamento por uma interface web elegante

    [:octicons-arrow-right-24: Tutorial da Web UI](webui/TUTORIAL.md)

-   :material-chart-line:{ .lg .middle } __Métricas locais__

    ---

    Salve gráficos e comparações de validação em cada treinamento

    [:octicons-arrow-right-24: Métricas locais](webui/LOCAL_METRICS.md)

-   :material-closed-caption:{ .lg .middle } __CaptionFlow__

    ---

    Gere captions de datasets com workers GPU locais

    [:octicons-arrow-right-24: CaptionFlow captioning](CAPTIONFLOW.pt-BR.md)

-   :material-shield-check:{ .lg .middle } __Verificações NSFW__

    ---

    Filtre samples do cache VAE com políticas locais de classificador

    [:octicons-arrow-right-24: Verificações do classificador NSFW](NSFW.pt-BR.md)

-   :material-api:{ .lg .middle } __API REST__

    ---

    Automatize fluxos de treinamento com a API HTTP

    [:octicons-arrow-right-24: Tutorial da API](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __Treinamento na Nuvem__

    ---

    Execute o treinamento no Replicate ou em workers distribuídos

    [:octicons-arrow-right-24: Treinamento na Nuvem](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __Multiusuário__

    ---

    Recursos corporativos: SSO, cotas, RBAC, orquestração de workers

    [:octicons-arrow-right-24: Guia Enterprise](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __Guias de Modelos__

    ---

    Guias passo a passo para Flux, SD3, SDXL, modelos de vídeo e mais

    [:octicons-arrow-right-24: Guias de Modelos](quickstart/index.md)

-   :material-flask-outline:{ .lg .middle } __Métodos experimentais__

    ---

    Recursos de pesquisa como AnyFlow, quantização SDNQ Hadamard no estilo ConvRot, checkpointing segmentado, checkpointing estilo Unsloth, Prompt2Effect, Self-Flow, Flow-DPO, iREPA, LayerSync, Diff2Flow, Metal Flash Attention e Video CREPA

    [:octicons-arrow-right-24: AnyFlow](experimental/ANYFLOW.md) · [:octicons-arrow-right-24: iREPA](experimental/IREPA.pt-BR.md) · [:octicons-arrow-right-24: ConvRot / Hadamard SDNQ](experimental/CONVROT.md) · [:octicons-arrow-right-24: Segmented Checkpointing](experimental/SEGMENTED_CHECKPOINTING.md) · [:octicons-arrow-right-24: Unsloth Checkpointing](experimental/UNSLOTH_CHECKPOINTING.md) · [:octicons-arrow-right-24: Metal Flash Attention](experimental/METAL_FLASH_ATTENTION.md)

</div>

## Recursos

- **Treinamento multimodal** - Modelos de geração de imagem, vídeo e áudio
- **Web UI e API** - Treine pelo navegador ou automatize com REST
- **Captioning com CaptionFlow** - Gere captions com GPUs locais pela fila de jobs da Web UI
- **Orquestração de workers** - Distribua jobs entre máquinas com GPU
- **Pronto para empresa** - SSO LDAP/OIDC, RBAC, cotas, logs de auditoria
- **Integração com nuvem** - Replicate, workers auto-hospedados
- **Otimização de memória** - DeepSpeed, FSDP2, quantização

## Modelos Suportados

| Tipo | Modelos |
|------|--------|
| **Imagem** | Flux.1/2, SD3, SDXL, Chroma, Auraflow, PixArt, Sana, Lumina2, HiDream e mais |
| **Vídeo** | Wan, LTX Video, Hunyuan Video, Kandinsky 5, LongCat |
| **Áudio** | ACE-Step |

Veja [Guias de Modelos](quickstart/index.md) para documentação completa.

## Comunidade

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - Relatos de bugs e solicitações de recursos

## Licença

SimpleTuner é software de código aberto.
