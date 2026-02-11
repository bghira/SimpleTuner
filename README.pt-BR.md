# SimpleTuner 💹

> ℹ️ Nenhum dado e enviado a terceiros, exceto por flags opt-in `report_to`, `push_to_hub` ou webhooks que devem ser configurados manualmente.

**SimpleTuner** e voltado para simplicidade, com foco em deixar o codigo facil de entender. Este codebase serve como um exercicio academico compartilhado, e contribuicoes sao bem-vindas.

Se voce quiser se juntar a comunidade, estamos [no Discord](https://discord.gg/JGkSwEbjRb) via Terminus Research Group.
Se tiver perguntas, fique a vontade para falar conosco la.

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## Indice

- [Filosofia de design](#filosofia-de-design)
- [Tutorial](#tutorial)
- [Recursos](#recursos)
  - [Recursos centrais de treinamento](#recursos-centrais-de-treinamento)
  - [Suporte a arquitetura de modelos](#suporte-a-arquitetura-de-modelos)
  - [Tecnicas avancadas de treinamento](#tecnicas-avancadas-de-treinamento)
  - [Recursos especificos do modelo](#recursos-especificos-do-modelo)
  - [Guias de inicio rapido](#guias-de-inicio-rapido)
- [Requisitos de hardware](#requisitos-de-hardware)
- [Toolkit](#toolkit)
- [Configuracao](#configuracao)
- [Solucao de problemas](#solucao-de-problemas)

## Filosofia de design

- **Simplicidade**: Objetivo de ter bons defaults para a maioria dos casos, reduzindo ajustes manuais.
- **Versatilidade**: Projetado para lidar com uma ampla faixa de quantidade de imagens - de datasets pequenos a colecoes extensas.
- **Recursos de ponta**: So incorpora recursos que provaram eficacia, evitando adicionar opcoes nao testadas.

## Tutorial

Explore este README por completo antes de iniciar o [novo tutorial da web UI](/documentation/webui/TUTORIAL.pt-BR.md) ou [o tutorial classico via linha de comando](/documentation/TUTORIAL.pt-BR.md), pois este documento contem informacoes vitais que voce precisa saber primeiro.

Para um quick start configurado manualmente sem ler toda a documentacao ou usar interfaces web, voce pode usar o guia [Quick Start](/documentation/QUICKSTART.pt-BR.md).

Para sistemas com memoria limitada, veja o [documento DeepSpeed](/documentation/DEEPSPEED.pt-BR.md), que explica como usar o 🤗Accelerate para configurar o DeepSpeed da Microsoft para offload do estado do otimizador. Para sharding com DTensor e paralelismo de contexto, leia o [guia FSDP2](/documentation/FSDP2.pt-BR.md), que cobre o fluxo do FullyShardedDataParallel v2 dentro do SimpleTuner.

Para treinamento distribuido multi-node, [este guia](/documentation/DISTRIBUTED.pt-BR.md) ajuda a ajustar as configuracoes dos guias INSTALL e Quickstart para multi-node, otimizando para datasets de imagens com bilhoes de samples.

---

## Recursos

SimpleTuner oferece suporte de treinamento abrangente em varias arquiteturas de modelos de difusao com disponibilidade consistente de recursos:

### Recursos centrais de treinamento

- **Web UI amigavel** - Gerencie todo o ciclo de treinamento por um painel elegante
- **Treinamento multi-modal** - Pipeline unificado para modelos generativos de **Imagem, Video e Audio**
- **Treinamento multi-GPU** - Treino distribuido em varias GPUs com otimizacao automatica
- **Cache avancado** - Embeddings de imagem, video, audio e legenda em disco para treino mais rapido
- **Aspect bucketing** - Suporte a tamanhos e proporcoes variadas de imagem/video
- **Sliders de conceito** - Targeting amigavel para sliders em LoRA/LyCORIS/full (via LyCORIS `full`) com amostragem positiva/negativa/neutra e forca por prompt; veja o [guia Slider LoRA](/documentation/SLIDER_LORA.pt-BR.md)
- **Otimizacao de memoria** - A maioria dos modelos treinaveis em GPU 24G, muitos em 16G com otimizacoes
- **Integracao DeepSpeed & FSDP2** - Treine modelos grandes em GPUs menores com sharding de otimizador/gradiente/parametros, atencao paralela por contexto, gradient checkpointing e offload do estado do otimizador
- **Treinamento em S3** - Treine direto do armazenamento cloud (Cloudflare R2, Wasabi S3)
- **Suporte a EMA** - Pesos de media movel exponencial para melhorar estabilidade e qualidade
- **Trackers de experimento customizados** - Coloque um `accelerate.GeneralTracker` em `simpletuner/custom-trackers` e use `--report_to=custom-tracker --custom_tracker=<name>`

### Recursos multi-usuario e enterprise

SimpleTuner inclui uma plataforma completa de treinamento multi-usuario com recursos enterprise — **gratis e open source, para sempre**.

- **Orquestracao de workers** - Registre workers GPU distribuidos que se conectam automaticamente a um painel central e recebem jobs via SSE; suporta workers efemeros (cloud) e persistentes (sempre ligados); veja o [guia de orquestracao de workers](/documentation/experimental/server/WORKERS.pt-BR.md)
- **Integracao SSO** - Autentique com LDAP/Active Directory ou provedores OIDC (Okta, Azure AD, Keycloak, Google); veja o [guia de autenticacao externa](/documentation/experimental/server/EXTERNAL_AUTH.pt-BR.md)
- **Controle de acesso por funcoes** - Quatro funcoes padrao (Viewer, Researcher, Lead, Admin) com 17+ permissoes granulares; defina regras por glob para restringir configs, hardware ou provedores por time
- **Organizacoes e times** - Estrutura hierarquica multi-tenant com quotas por teto; limites da org impõem maximos absolutos, limites de time operam dentro desses limites
- **Quotas e limites de gasto** - Impoem tetos de custo (diario/mensal), limites de concorrencia e taxa de submissao por org/time/usuario; acoes incluem bloquear, alertar ou exigir aprovacao
- **Fila de jobs com prioridades** - Cinco niveis (Low → Critical) com agendamento fair-share entre times, protecao contra starvation para jobs longos e override de prioridade por admin
- **Fluxos de aprovacao** - Regras configuraveis disparam aprovacao para jobs acima de custos, novos usuarios ou pedidos de hardware; aprove via UI, API ou resposta por email
- **Notificacoes por email** - Integracao SMTP/IMAP para status de job, solicitacoes de aprovacao, alertas de quota e conclusao
- **Chaves de API e permissoes com escopo** - Gere chaves com expiracao e escopo limitado para pipelines CI/CD
- **Audit logging** - Registre todas as acoes de usuario com verificacao de cadeia para compliance; veja o [guia de auditoria](/documentation/experimental/server/AUDIT.pt-BR.md)

Para detalhes de deploy, veja o [guia enterprise](/documentation/experimental/server/ENTERPRISE.pt-BR.md).

### Suporte a arquitetura de modelos

| Modelo | Parametros | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Quantizacao | Flow Matching | Text Encoders |
|-------|------------|-----------|---------|-----------|------------|--------------|---------------|---------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HeartMuLa** | 3B | ✓ | ✓ | ✓* | ✗ | int8 | ✗ | Nenhum |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **LTX Video 2** | 19B | ✓ | ✓ | ✓* | ✗ | int8/fp8 | ✓ | Gemma3 |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = Suportado, ✗ = Nao suportado, * = Requer DeepSpeed para treino full-rank*

### Tecnicas avancadas de treinamento

- **TREAD** - Dropout token-wise para modelos transformer, incluindo treino Kontext
- **Masked loss training** - Convergencia superior com orientacao de segmentacao/profundidade
- **Prior regularization** - Estabilidade melhor para consistencia de personagens
- **Gradient checkpointing** - Intervalos configuraveis para otimizacao de memoria/velocidade
- **Funcoes de loss** - L2, Huber, Smooth L1 com suporte a scheduling
- **SNR weighting** - Min-SNR gamma weighting para dinamica de treino melhor
- **Group offloading** - Diffusers v0.33+ com staging de modulos em CPU/disco e streams CUDA opcionais
- **Varreduras de validation adapter** - Anexe LoRAs temporariamente (single ou presets JSON) durante validacao para comparar renders sem tocar o loop de treino
- **External validation hooks** - Troque o pipeline de validacao embutido ou passos pos-upload pelos seus scripts, rodando checks em outra GPU ou enviando artefatos para qualquer provedor cloud ([detalhes](/documentation/OPTIONS.pt-BR.md#validation_method))
- **Regularizacao CREPA** - Alinhamento de representacao entre frames para video DiTs ([guia](/documentation/experimental/VIDEO_CREPA.pt-BR.md))
- **Formatos de I/O LoRA** - Load/save de LoRAs PEFT no layout Diffusers ou no estilo ComfyUI `diffusion_model.*` (Flux/Flux2/Lumina2/Z-Image detectam ComfyUI automaticamente)

### Recursos especificos do modelo

- **Flux Kontext** - Edicao de condicionamento e treino image-to-image para modelos Flux
- **PixArt two-stage** - Suporte ao pipeline eDiff para PixArt Sigma
- **Flow matching models** - Scheduling avancado com distribuicoes beta/uniform
- **HiDream MoE** - Aumento de loss para gate Mixture of Experts
- **T5 masked training** - Mais detalhes finos para Flux e modelos compativeis
- **QKV fusion** - Otimizacoes de memoria e velocidade (Flux, Lumina2)
- **Integracao TREAD** - Roteamento seletivo de tokens para a maioria dos modelos
- **Wan 2.x I2V** - Presets de estagio alto/baixo + fallback time-embedding 2.1 (veja quickstart do Wan)
- **Classifier-free guidance** - Reintroducao opcional de CFG para modelos destilados

### Guias de inicio rapido

Guias detalhados estao disponiveis para todos os modelos suportados:

- **[Guia TwinFlow Few-Step (RCGM)](/documentation/distillation/TWINFLOW.pt-BR.md)** - Habilite perda auxiliar RCGM para geracao few-step/one-step (modelos flow ou difusao via diff2flow)
- **[Guia Flux.1](/documentation/quickstart/FLUX.pt-BR.md)** - Inclui suporte a edicao Kontext e fusao QKV
- **[Guia Flux.2](/documentation/quickstart/FLUX2.pt-BR.md)** - **NOVO!** Modelo Flux enorme com text encoder Mistral-3
- **[Guia Z-Image](/documentation/quickstart/ZIMAGE.pt-BR.md)** - Base/Turbo LoRA com adaptador assistente + aceleracao TREAD
- **[Guia ACE-Step](/documentation/quickstart/ACE_STEP.pt-BR.md)** - **NOVO!** Treinamento de modelo de geracao de audio (texto-para-musica)
- **[Guia HeartMuLa](/documentation/quickstart/HEARTMULA.pt-BR.md)** - **NOVO!** Treinamento de modelo de audio autoregressivo (texto-para-audio)
- **[Guia Chroma](/documentation/quickstart/CHROMA.pt-BR.md)** - Transformer flow-matching da Lodestone com schedules especificos
- **[Guia Stable Diffusion 3](/documentation/quickstart/SD3.pt-BR.md)** - Treino full e LoRA com ControlNet
- **[Guia Stable Diffusion XL](/documentation/quickstart/SDXL.pt-BR.md)** - Pipeline completo de treino SDXL
- **[Guia Auraflow](/documentation/quickstart/AURAFLOW.pt-BR.md)** - Treinamento de modelo flow-matching
- **[Guia PixArt Sigma](/documentation/quickstart/SIGMA.pt-BR.md)** - Modelo DiT com suporte two-stage
- **[Guia Sana](/documentation/quickstart/SANA.pt-BR.md)** - Modelo flow-matching leve
- **[Guia Lumina2](/documentation/quickstart/LUMINA2.pt-BR.md)** - Modelo flow-matching de 2B parametros
- **[Guia Kwai Kolors](/documentation/quickstart/KOLORS.pt-BR.md)** - Baseado em SDXL com encoder ChatGLM
- **[Guia LongCat-Video](/documentation/quickstart/LONGCAT_VIDEO.pt-BR.md)** - Flow-matching text-to-video e image-to-video com Qwen-2.5-VL
- **[Guia LongCat-Video Edit](/documentation/quickstart/LONGCAT_VIDEO_EDIT.pt-BR.md)** - Flavor conditioning-first (image-to-video)
- **[Guia LongCat-Image](/documentation/quickstart/LONGCAT_IMAGE.pt-BR.md)** - Modelo flow-matching bilingue 6B com encoder Qwen-2.5-VL
- **[Guia LongCat-Image Edit](/documentation/quickstart/LONGCAT_EDIT.pt-BR.md)** - Flavor de edicao de imagem que requer latentes de referencia
- **[Guia LTX Video](/documentation/quickstart/LTXVIDEO.pt-BR.md)** - Treino de difusao de video
- **[Guia Hunyuan Video 1.5](/documentation/quickstart/HUNYUANVIDEO.pt-BR.md)** - Flow-matching T2V/I2V 8.3B com estagios SR
- **[Guia Wan Video](/documentation/quickstart/WAN.pt-BR.md)** - Flow-matching de video com suporte TREAD
- **[Guia HiDream](/documentation/quickstart/HIDREAM.pt-BR.md)** - Modelo MoE com recursos avancados
- **[Guia Cosmos2](/documentation/quickstart/COSMOS2IMAGE.pt-BR.md)** - Geracao de imagem multi-modal
- **[Guia OmniGen](/documentation/quickstart/OMNIGEN.pt-BR.md)** - Modelo unificado de geracao de imagens
- **[Guia Qwen Image](/documentation/quickstart/QWEN_IMAGE.pt-BR.md)** - Treinamento de larga escala com 20B parametros
- **[Guia Stable Cascade Stage C](/documentation/quickstart/STABLE_CASCADE_C.pt-BR.md)** - LoRAs de prior com validacao prior+decoder combinada
- **[Guia Kandinsky 5.0 Image](/documentation/quickstart/KANDINSKY5_IMAGE.pt-BR.md)** - Geracao de imagem com Qwen2.5-VL + Flux VAE
- **[Guia Kandinsky 5.0 Video](/documentation/quickstart/KANDINSKY5_VIDEO.pt-BR.md)** - Geracao de video com HunyuanVideo VAE

---

## Requisitos de hardware

### Requisitos gerais

- **NVIDIA**: RTX 3080+ recomendado (testado ate H200)
- **AMD**: 7900 XTX 24GB e MI300X verificados (uso de memoria maior vs NVIDIA)
- **Apple**: M3 Max+ com 24GB+ de memoria unificada para treino LoRA

### Diretrizes de memoria por tamanho de modelo

- **Modelos grandes (12B+)**: A100-80G para full-rank, 24G+ para LoRA/Lycoris
- **Modelos medios (2B-8B)**: 16G+ para LoRA, 40G+ para treino full-rank
- **Modelos pequenos (<2B)**: 12G+ suficiente para a maioria dos treinos

**Nota**: Quantizacao (int8/fp8/nf4) reduz significativamente requisitos de memoria. Veja os [guias de inicio rapido](#guias-de-inicio-rapido) para requisitos especificos.

## Configuracao

SimpleTuner pode ser instalado via pip para a maioria dos usuarios:

```bash
# Instalacao base (PyTorch apenas CPU)
pip install simpletuner

# Usuarios CUDA (GPUs NVIDIA)
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell (GPUs NVIDIA serie B)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130

# Usuarios ROCm (GPUs AMD)
pip install 'simpletuner[rocm]' --extra-index-url https://download.pytorch.org/whl/rocm7.1

# Usuarios Apple Silicon (M1/M2/M3/M4)
pip install 'simpletuner[apple]'
```

Para instalacao manual ou setup de desenvolvimento, veja a [documentacao de instalacao](/documentation/INSTALL.pt-BR.md).

## Solucao de problemas

Ative logs de debug adicionando `export SIMPLETUNER_LOG_LEVEL=DEBUG` no seu ambiente (arquivo `config/config.env`).

Para analise de performance do loop de treino, definir `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` tera timestamps que destacam problemas na configuracao.

Para uma lista completa de opcoes disponiveis, consulte [esta documentacao](/documentation/OPTIONS.pt-BR.md).
