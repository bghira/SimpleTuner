# Guia de Início Rápido

**Nota**: Para configurações mais avançadas, veja o [tutorial](TUTORIAL.md) e a [referência de opções](OPTIONS.md).

## Compatibilidade de recursos

Para a matriz completa e mais precisa de recursos, consulte o [README principal](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Guias de início rápido por modelo

| Modelo | Parâmetros | LoRA PEFT | Lycoris | Full-Rank | Quantização | Precisão mista | Checkpoint de gradiente | Flow Shift | TwinFlow | Self-Flow | LayerSync | ControlNet | Sliders† | Guia |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | ✓ | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 padrão, nf4 opcional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ | [IDEOGRAM4.md](quickstart/IDEOGRAM4.pt-BR.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✓ | ✓ | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [WAN.md](quickstart/WAN.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **obrigatório** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **obrigatório** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | não suportado | fp32 (obrigatório) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = suportado, ✓* = requer DeepSpeed/FSDP2 para full-rank, ✗ = não suportado, `✓+` indica que o checkpointing é recomendado devido à pressão de VRAM. TwinFlow ✓ significa suporte nativo quando `twinflow_enabled=true` (modelos de difusão precisam de `diff2flow_enabled+twinflow_allow_diff2flow`). Self-Flow ✓ significa suporte nativo para `crepa_enabled=true` com `crepa_feature_source=self_flow`, `use_ema=true` e `crepa_teacher_block_index` definido. LayerSync ✓ significa que o backbone expõe estados ocultos do transformer para autoalinhamento; ✗ marca backbones estilo UNet sem esse buffer. †Sliders se aplicam a LoRA e LyCORIS (incluindo LyCORIS full-rank “full”).*

> ℹ️ O quickstart do Wan inclui presets das etapas 2.1 + 2.2 e o toggle de time-embedding. Flux Kontext cobre fluxos de edição construídos sobre o Flux.1.

> ⚠️ Estes quickstarts são documentos vivos. Espere atualizações ocasionais conforme novos modelos chegam ou as receitas de treinamento melhoram.

### Caminhos rápidos: Z-Image Turbo e Flux Schnell

- **Z-Image Turbo**: LoRA totalmente suportado com TREAD; roda rápido em NVIDIA e macOS mesmo sem quantização (int8 também funciona). Muitas vezes o gargalo é apenas a configuração do trainer.
- **Flux Schnell**: A configuração do quickstart lida automaticamente com o agendamento rápido de ruído e o stack de LoRA assistente; não são necessários flags extras para treinar LoRAs Schnell.

### Recursos experimentais avançados

- **Diff2Flow**: Permite treinar modelos padrão de epsilon/v-prediction (SD1.5, SDXL, DeepFloyd etc.) usando uma loss de Flow Matching. Isso reduz a lacuna entre arquiteturas antigas e treinamento moderno baseado em fluxo.
- **Scheduled Sampling**: Reduz o viés de exposição ao permitir que o modelo gere seus próprios latentes ruidosos intermediários durante o treinamento ("rollout"). Isso ajuda o modelo a aprender a se recuperar de seus próprios erros de geração.

## Problemas Comuns

### Dataset tem menos amostras do que esperado

Se seu dataset acaba com menos amostras utilizáveis do que você esperava, arquivos podem ter sido filtrados durante o processamento. Razões comuns incluem:

- **Arquivos muito pequenos**: Imagens abaixo de `minimum_image_size` são filtradas
- **Proporção fora do intervalo**: Imagens fora dos limites de `minimum_aspect_ratio`/`maximum_aspect_ratio` são excluídas
- **Limites de duração**: Arquivos de áudio/vídeo que excedem limites de duração são ignorados

**Visualizando estatísticas de filtragem:**
- Na WebUI, navegue até o diretório do seu dataset e selecione-o para ver estatísticas de filtragem
- Verifique os logs durante o processamento do dataset por estatísticas como: `Sample processing statistics: {'total_processed': 100, 'skipped': {'too_small': 15, ...}}`

Para solução de problemas detalhada, consulte [Solucionando problemas de datasets filtrados](DATALOADER.pt-BR.md#solucionando-problemas-de-datasets-filtrados) na documentação do dataloader.
