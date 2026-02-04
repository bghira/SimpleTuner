# REPA & U-REPA (regularização de imagens)

O Alinhamento de Representação (REPA) é uma técnica de regularização que alinha os estados ocultos do modelo de difusão com características de um codificador de visão congelado (tipicamente DINOv2). Isso melhora a qualidade da geração e a eficiência do treinamento aproveitando representações visuais pré-treinadas.

O SimpleTuner suporta duas variantes:

- **REPA** para modelos de imagem baseados em DiT (Flux, SD3, Chroma, Sana, PixArt, etc.) - PR #2562
- **U-REPA** para modelos de imagem baseados em UNet (SDXL, SD1.5, Kolors) - PR #2563

> **Procurando modelos de vídeo?** Veja [VIDEO_CREPA.pt-BR.md](VIDEO_CREPA.pt-BR.md) para suporte CREPA em modelos de vídeo com alinhamento temporal.

## Quando usar

### REPA (modelos DiT)
- Você está treinando modelos de imagem baseados em DiT e deseja convergência mais rápida
- Você nota problemas de qualidade ou deseja um fundamento semântico mais forte
- Famílias de modelos suportadas: `flux`, `flux2`, `sd3`, `chroma`, `sana`, `pixart`, `hidream`, `auraflow`, `lumina2` e outros

### U-REPA (modelos UNet)
- Você está treinando modelos de imagem baseados em UNet (SDXL, SD1.5, Kolors)
- Você deseja aproveitar o alinhamento de representação otimizado para arquiteturas UNet
- U-REPA usa alinhamento de **bloco intermediário** (não camadas iniciais) e adiciona **perda de variedade** para melhor estrutura de similaridade relativa

## Configuração rápida (WebUI)

### Para modelos DiT (REPA)

1. Abra **Treinamento → Funções de perda**.
2. Habilite **CREPA** (a mesma opção habilita REPA para modelos de imagem).
3. Defina **CREPA Block Index** para uma camada inicial do codificador:
   - Flux / Flux2: `8`
   - SD3: `8`
   - Chroma: `8`
   - Sana / PixArt: `10`
4. Defina **Peso** como `0.5` para começar.
5. Mantenha os padrões para o codificador de visão (`dinov2_vitg14`, resolução `518`).

### Para modelos UNet (U-REPA)

1. Abra **Treinamento → Funções de perda**.
2. Habilite **U-REPA**.
3. Defina **U-REPA Weight** como `0.5` (padrão do artigo).
4. Defina **U-REPA Manifold Weight** como `3.0` (padrão do artigo).
5. Mantenha os padrões para o codificador de visão.

## Configuração rápida (config JSON / CLI)

### Para modelos DiT (REPA)

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

### Para modelos UNet (U-REPA)

```json
{
  "urepa_enabled": true,
  "urepa_lambda": 0.5,
  "urepa_manifold_weight": 3.0,
  "urepa_model": "dinov2_vitg14",
  "urepa_encoder_image_size": 518
}
```

## Diferenças principais: REPA vs U-REPA

| Aspecto | REPA (DiT) | U-REPA (UNet) |
|---------|-----------|---------------|
| Arquitetura | Blocos Transformer | UNet com bloco intermediário |
| Ponto de alinhamento | Camadas transformer iniciais | Bloco intermediário (gargalo) |
| Formato do estado oculto | `(B, S, D)` sequência | `(B, C, H, W)` convolucional |
| Componentes de perda | Alinhamento cosseno | Cosseno + Perda de variedade |
| Peso padrão | 0.5 | 0.5 |
| Prefixo de config | `crepa_*` | `urepa_*` |

## Especificidades do U-REPA

O U-REPA adapta o REPA para arquiteturas UNet com duas inovações principais:

### Alinhamento do bloco intermediário
Diferentemente do REPA baseado em DiT que usa camadas transformer iniciais, o U-REPA extrai características do **bloco intermediário** (gargalo) do UNet. Este é o local onde o UNet tem mais informação semântica comprimida.

- **SDXL/Kolors**: O bloco intermediário produz `(B, 1280, 16, 16)` para imagens 1024x1024
- **SD1.5**: O bloco intermediário produz `(B, 1280, 8, 8)` para imagens 512x512

### Perda de variedade
Além do alinhamento cosseno, o U-REPA adiciona uma **perda de variedade** que alinha a estrutura de similaridade relativa:

```
L_manifold = ||sim(y[i],y[j]) - sim(h[i],h[j])||^2_F
```

Isso garante que se dois patches do codificador são similares, os patches projetados correspondentes também devem ser similares. O parâmetro `urepa_manifold_weight` (padrão 3.0) controla o equilíbrio entre alinhamento direto e alinhamento de variedade.

## Parâmetros de ajuste

### REPA (modelos DiT)
- `crepa_lambda`: Peso da perda de alinhamento (padrão 0.5)
- `crepa_block_index`: Qual bloco transformer usar (indexado em 0)
- `crepa_spatial_align`: Interpolar tokens para corresponder (padrão true)
- `crepa_encoder`: Modelo do codificador de visão (padrão `dinov2_vitg14`)
- `crepa_encoder_image_size`: Resolução de entrada (padrão 518)

### U-REPA (modelos UNet)
- `urepa_lambda`: Peso da perda de alinhamento (padrão 0.5)
- `urepa_manifold_weight`: Peso da perda de variedade (padrão 3.0)
- `urepa_model`: Modelo do codificador de visão (padrão `dinov2_vitg14`)
- `urepa_encoder_image_size`: Resolução de entrada (padrão 518)
- `urepa_use_tae`: Usar Tiny AutoEncoder para decodificação mais rápida

## Agendamento de coeficientes

Tanto REPA quanto U-REPA suportam agendamento para decair a regularização durante o treinamento:

```json
{
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

Para U-REPA, use o prefixo `urepa_`:

```json
{
  "urepa_scheduler": "cosine",
  "urepa_warmup_steps": 100,
  "urepa_cutoff_step": 5000
}
```

<details>
<summary>Como funciona (praticante)</summary>

### REPA (DiT)
- Captura estados ocultos de um bloco transformer escolhido
- Projeta através de LayerNorm + Linear para dimensão do codificador
- Calcula similaridade cosseno com características DINOv2 congeladas
- Interpola tokens espaciais para corresponder se as contagens diferirem

### U-REPA (UNet)
- Registra um hook forward no mid_block do UNet
- Captura características convolucionais `(B, C, H, W)`
- Reformata para sequência `(B, H*W, C)` e projeta para dimensão do codificador
- Calcula tanto alinhamento cosseno quanto perda de variedade
- Perda de variedade alinha a estrutura de similaridade par a par

</details>

<details>
<summary>Técnico (internos do SimpleTuner)</summary>

### REPA
- Implementação: `simpletuner/helpers/training/crepa.py` (classe `CrepaRegularizer`)
- Detecção de modo: `CrepaMode.IMAGE` para modelos de imagem, definido automaticamente via propriedade `crepa_mode`
- Estados ocultos armazenados na chave `crepa_hidden_states` da saída do modelo

### U-REPA
- Implementação: `simpletuner/helpers/training/crepa.py` (classe `UrepaRegularizer`)
- Captura do bloco intermediário: `simpletuner/helpers/utils/hidden_state_buffer.py` (`UNetMidBlockCapture`)
- Tamanho oculto inferido de `block_out_channels[-1]` (1280 para SDXL/SD1.5/Kolors)
- Habilitado apenas para `MODEL_TYPE == ModelTypes.UNET`
- Estados ocultos armazenados na chave `urepa_hidden_states` da saída do modelo

</details>

## Problemas comuns

- **Tipo de modelo errado**: REPA (`crepa_*`) é para modelos DiT; U-REPA (`urepa_*`) é para modelos UNet. Usar o errado não terá efeito.
- **Índice de bloco muito alto** (REPA): Diminua o índice se você receber erros "hidden states not returned".
- **Picos de VRAM**: Tente um codificador menor (`dinov2_vits14` + tamanho de imagem `224`) ou habilite `use_tae` para decodificação.
- **Peso de variedade muito alto** (U-REPA): Se o treinamento ficar instável, reduza `urepa_manifold_weight` de 3.0 para 1.0.

## Referências

- [Artigo REPA](https://arxiv.org/abs/2402.17750) - Alinhamento de Representação para Geração
- [Artigo U-REPA](https://arxiv.org/abs/2410.xxxxx) - REPA Universal para arquiteturas UNet (NeurIPS 2025)
- [DINOv2](https://github.com/facebookresearch/dinov2) - Codificador de visão auto-supervisionado
