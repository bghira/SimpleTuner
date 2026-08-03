# Guia de Início Rápido

**Nota**: Para configurações mais avançadas, veja o [tutorial](TUTORIAL.md) e a [referência de opções](OPTIONS.md).

## Compatibilidade de recursos

Para a matriz completa e mais precisa de recursos, consulte o [README principal](https://github.com/bghira/SimpleTuner#model-architecture-support).

## Guias de início rápido por modelo

| Modelo | Parâmetros | LoRA PEFT | Lycoris | Full-Rank | Quantização | Precisão mista | Checkpoint de gradiente | Flow Shift | TwinFlow | Self-Flow | LayerSync | Ref Inputs | ControlNet | Sliders† | Licença | Permite uso comercial | Guia |
| --- | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- | :---: | --- |
| PixArt Sigma | 0.6B–0.9B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Condições aplicáveis<sup>1</sup> | [SIGMA.md](quickstart/SIGMA.md) |
| NVLabs Sana | 1.6B–4.8B | ✗ | ✓ | ✓ | int8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [SANA.md](quickstart/SANA.md) |
| Kwai Kolors | 2.7B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Kwai Kolors License](https://huggingface.co/terminusresearch/kwai-kolors-1.0/blob/main/MODEL_LICENSE) | Condições aplicáveis<sup>7</sup> | [KOLORS.md](quickstart/KOLORS.md) |
| Stable Diffusion 3 | 2B–8B | ✓ | ✓ | ✓ | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Stability AI Community](https://stability.ai/license) | Condições aplicáveis<sup>2</sup> | [SD3.md](quickstart/SD3.md) |
| Flux.1 | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Condições aplicáveis<sup>3</sup> | [FLUX.md](quickstart/FLUX.md) |
| Flux.2 | 32B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) / [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Condições aplicáveis<sup>4</sup> | [FLUX2.md](quickstart/FLUX2.md) |
| Flux Kontext | 8B–12B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✓ | ✓ | [BFL Non-Commercial](https://bfl.ai/legal/non-commercial-license-terms) | Não<sup>5</sup> | [FLUX_KONTEXT.md](quickstart/FLUX_KONTEXT.md) |
| Z-Image Turbo | 6B | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [ZIMAGE.md](quickstart/ZIMAGE.md) |
| Krea2 | - | ✓ | ✗ | ✓* | int8 opcional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✓ opt | ✗ | ✓ | [Krea 2 Community](https://www.krea.ai/krea-2-licensing) | Condições aplicáveis<sup>6</sup> | [KREA2.md](quickstart/KREA2.pt-BR.md) |
| Mage-Flow | 4B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ edit | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [MAGEFLOW.md](quickstart/MAGEFLOW.pt-BR.md) |
| Boogu-Image 0.1 | - | ✓ | ✓ | ✓* | fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ edit | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [BOOGU_IMAGE.md](quickstart/BOOGU_IMAGE.pt-BR.md) |
| zlab i1 | 3B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Unspecified](https://huggingface.co/bghira/zlab-i1-diffusers) | Condições aplicáveis<sup>12</sup> | [ZLAB_i1.md](quickstart/ZLAB_i1.pt-BR.md) |
| Ideogram 4 | 9B | ✓ | ✓ | ✓* | fp8 padrão, nf4 opcional | bf16 | ✓+ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Ideogram 4 Non-Commercial](https://huggingface.co/ideogram-ai/ideogram-4-nf4/blob/main/LICENSE.md) | Não<sup>5</sup> | [IDEOGRAM4.md](quickstart/IDEOGRAM4.pt-BR.md) |
| ERNIE-Image | - | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [ERNIE.md](quickstart/ERNIE.pt-BR.md) |
| ACE-Step | 3.5B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B) / [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5) | Sim | [ACE_STEP.md](quickstart/ACE_STEP.md) |
| Chroma 1 | 8.9B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [CHROMA.md](quickstart/CHROMA.md) |
| Auraflow | 6B | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓+ | ✓ (SLG) | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) / [Pony License](https://huggingface.co/purplesmartai/pony-v7-base/blob/main/LICENSE) | Condições aplicáveis<sup>8</sup> | [AURAFLOW.md](quickstart/AURAFLOW.md) |
| HiDream I1 | 17B (8.5B MoE) | ✓ | ✓ | ✓* | int8/fp8/nf4 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [HIDREAM.md](quickstart/HIDREAM.md) |
| OmniGen | 3.8B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [OMNIGEN.md](quickstart/OMNIGEN.md) |
| Stable Diffusion XL | 2.6B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md) | Condições aplicáveis<sup>1</sup> | [SDXL.md](quickstart/SDXL.md) |
| Lumina2 | 2B | ✓ | ✓ | ✓ | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [LUMINA2.md](quickstart/LUMINA2.md) |
| Cosmos2 | 2B | ✓ | ✓ | ✓ | não recomendado | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) | Sim<sup>9</sup> | [COSMOS2IMAGE.md](quickstart/COSMOS2IMAGE.md) |
| Cosmos3 | 16B-65B | ✓ | ✓ | ✓* | no_change primeiro | bf16 | ✓ | ✓ | ✗ | ✗ | ✗ | audio opt | ✗ | ✓ | [OpenMDW 1.1](https://github.com/OpenMDW/openmdw/blob/main/1.1/LICENSE.OpenMDW-1.1) | Sim | [COSMOS3.md](quickstart/COSMOS3.pt-BR.md) |
| LTX Video | ~2.5B | ✓ | ✓ | ✓ | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [LTX Video OpenRAIL-M](https://huggingface.co/Lightricks/LTX-Video-0.9.5/blob/main/ltx-video-2b-v0.9.5.license.txt) | Condições aplicáveis<sup>10</sup> | [LTXVIDEO.md](quickstart/LTXVIDEO.md) |
| LTX Video 2 | 19B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [LTX-2 Community](https://ltx.io/model/license) | Condições aplicáveis<sup>10</sup> | [LTXVIDEO2.md](quickstart/LTXVIDEO2.md) |
| Hunyuan Video 1.5 | 8.3B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [Tencent Hunyuan Community](https://huggingface.co/tencent/HunyuanVideo-1.5/blob/main/LICENSE) | Condições aplicáveis<sup>11</sup> | [HUNYUANVIDEO.md](quickstart/HUNYUANVIDEO.md) |
| SanaVideo | 2B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [SANAVIDEO.md](quickstart/SANAVIDEO.pt-BR.md) |
| Wan 2.x | 1.3B–14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [WAN.md](quickstart/WAN.md) |
| Wan 2.2 S2V | 14B | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [WAN_S2V.md](quickstart/WAN_S2V.md) |
| Qwen Image | 20B | ✓ | ✓ | ✓* | **obrigatório** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [QWEN_IMAGE.md](quickstart/QWEN_IMAGE.md) |
| Qwen Image Edit | 20B | ✓ | ✓ | ✓* | **obrigatório** (int8/nf4) | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [QWEN_EDIT.md](quickstart/QWEN_EDIT.md) |
| Stable Cascade (C) | 1B, 3.6B prior | ✓ | ✓ | ✓* | não suportado | fp32 (obrigatório) | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | [Stable Cascade NC Community](https://huggingface.co/stabilityai/stable-cascade/blob/main/LICENSE) | Não<sup>5</sup> | [STABLE_CASCADE_C.md](quickstart/STABLE_CASCADE_C.md) |
| Kandinsky 5.0 Image | 6B (lite) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ I2I | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [KANDINSKY5_IMAGE.md](quickstart/KANDINSKY5_IMAGE.md) |
| Kandinsky 5.0 Video | 2B (lite), 19B (pro) | ✓ | ✓ | ✓* | int8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ I2V | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [KANDINSKY5_VIDEO.md](quickstart/KANDINSKY5_VIDEO.md) |
| LongCat-Video | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ opt | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [LONGCAT_VIDEO.md](quickstart/LONGCAT_VIDEO.md) |
| LongCat-Video Edit | 13.6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓+ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [MIT](https://opensource.org/license/mit) | Sim | [LONGCAT_VIDEO_EDIT.md](quickstart/LONGCAT_VIDEO_EDIT.md) |
| LongCat-Image | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [LONGCAT_IMAGE.md](quickstart/LONGCAT_IMAGE.md) |
| LongCat-Image Edit | 6B | ✓ | ✓ | ✓* | int8/fp8 opcional | bf16 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ req | ✗ | ✓ | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | Sim | [LONGCAT_EDIT.md](quickstart/LONGCAT_EDIT.md) |

*✓ = suportado, ✓* = requer DeepSpeed/FSDP2 para full-rank, ✗ = não suportado, `✓+` indica que o checkpointing é recomendado devido à pressão de VRAM. Ref Inputs marca caminhos existentes de condicionamento por referência/edição/I2V; `opt` significa opcional e `req` significa obrigatório para o flavour de edição/I2V. TwinFlow ✓ significa suporte nativo quando `twinflow_enabled=true` (modelos de difusão precisam de `diff2flow_enabled+twinflow_allow_diff2flow`). Self-Flow ✓ significa suporte nativo para `crepa_enabled=true` com `crepa_feature_source=self_flow`, `use_ema=true` e `crepa_teacher_block_index` definido. LayerSync ✓ significa que o backbone expõe estados ocultos do transformer para autoalinhamento; ✗ marca backbones estilo UNet sem esse buffer. †Sliders se aplicam a LoRA e LyCORIS (incluindo LyCORIS full-rank “full”).*

**Notas de licença:** O status de uso comercial cobre pesos do modelo, checkpoints derivados, fine-tunes e uso de modelo hospedado. Os direitos sobre saídas geradas podem diferir; leia o texto da licença vinculada antes de uma implantação comercial.

<sup>1</sup> Licenças estilo OpenRAIL geralmente permitem uso comercial com restrições de uso que continuam aplicáveis ao modelo e seus derivados.

<sup>2</sup> A Stability AI Community License está disponível para usuários qualificados abaixo do limite de receita; uso comercial maior exige termos empresariais da Stability.

<sup>3</sup> Flux.1 varia por flavour: Schnell e LibreFlux são Apache-2.0, enquanto Dev, Krea e Kontext usam termos não comerciais da BFL; revise os metadados upstream do FluxBooru antes de uso comercial.

<sup>4</sup> Flux.2 varia por flavour: Klein 4B é Apache-2.0, enquanto Dev e Klein 9B usam termos não comerciais da BFL.

<sup>5</sup> Termos públicos de modelo não comercial não permitem uso comercial de pesos, checkpoints derivados ou serviços hospedados do modelo sem uma licença separada.

<sup>6</sup> A Krea 2 Community License permite uso comercial apenas sob seus requisitos de receita e segurança/filtragem; caso contrário, é necessária uma licença empresarial.

<sup>7</sup> O uso comercial do modelo Kolors ou de seus derivados exige solicitar e receber permissão explícita do licenciante.

<sup>8</sup> AuraFlow aceita flavours upstream Apache-2.0 e um flavour Pony com uma licença personalizada separada; confira o flavour selecionado.

<sup>9</sup> A NVIDIA Open Model License permite uso comercial, mas inclui termos de contrato, uso aceitável e controle de exportação.

<sup>10</sup> LTX Video 0.9.5 usa OpenRAIL-M; LTX Video 2 usa termos comunitários da LTX com limite de receita para uso comercial.

<sup>11</sup> A Tencent Hunyuan Community License inclui exclusões territoriais e um limite comercial para serviços muito grandes.

<sup>12</sup> Este mirror publica `license: other` sem texto de licença padrão; revise os termos upstream antes de uso comercial.

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

Para solução de problemas detalhada, consulte [Solucionando problemas de datasets filtrados](DATALOADER.pt-BR.md) na documentação do dataloader.
