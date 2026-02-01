# CREPA (regularizacao de video)

Cross-frame Representation Alignment (CREPA) e um regularizador leve para modelos de video. Ele empurra os hidden states de cada frame em direcao aos recursos de um encoder de visao congelado do frame atual **e de seus vizinhos**, melhorando a consistencia temporal sem mudar sua loss principal.

> **Procurando modelos de imagem?** Veja [IMAGE_REPA.pt-BR.md](IMAGE_REPA.pt-BR.md) para suporte REPA em modelos DiT de imagem (Flux, SD3, etc.) e U-REPA para modelos UNet (SDXL, SD1.5, Kolors).

## Quando usar

- Voce treina videos com movimento complexo, mudancas de cena ou oclusoes.
- Voce esta fazendo fine-tuning de um video DiT (LoRA ou full) e ve flicker/deriva de identidade entre frames.
- Familias suportadas: `kandinsky5_video`, `ltxvideo`, `sanavideo` e `wan` (outras familias nao expoem hooks CREPA).
- Voce tem VRAM extra (CREPA adiciona ~1-2GB dependendo das configuracoes) para o encoder DINO e o VAE, que precisam ficar em memoria durante o treino para decodificar latentes em pixels.

## Config rapida (WebUI)

1. Abra **Training -> Loss functions**.
2. Habilite **CREPA**.
3. Defina **CREPA Block Index** para uma camada do lado do encoder. Comece com:
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. Deixe **Weight** em `0.5` para comecar.
5. Mantenha **Adjacent Distance** em `1` e **Temporal Decay** em `1.0` para um setup que corresponde ao paper original do CREPA.
6. Use os defaults para o encoder de visao (`dinov2_vitg14`, resolucao `518`). Mude apenas se precisar de um encoder menor (ex.: `dinov2_vits14` + tamanho de imagem `224` para economizar VRAM).
7. Treine normalmente. O CREPA adiciona uma loss auxiliar e registra `crepa_loss` / `crepa_similarity`.

## Config rapida (config JSON / CLI)

Adicione o seguinte ao seu `config.json` ou args da CLI:

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## Ajustes

- `crepa_spatial_align`: mantem estrutura em nivel de patch (padrao). Defina `false` para fazer pool se memoria estiver apertada.
- `crepa_normalize_by_frames`: mantem a escala da loss estavel entre comprimentos de clipe (padrao). Desligue se quiser que clipes longos contribuam mais.
- `crepa_drop_vae_encoder`: libera memoria se voce so **decodificar** latentes (inseguro se voce precisar codificar pixels).
- `crepa_adjacent_distance=0`: se comporta como REPA* por frame (sem ajuda de vizinhos); combine com `crepa_adjacent_tau` para decaimento por distancia.
- `crepa_cumulative_neighbors=true` (apenas config): usa todos os offsets `1..d` em vez de apenas os vizinhos mais proximos.
- `crepa_use_backbone_features=true`: pula o encoder externo e alinha com um bloco transformer mais profundo; defina `crepa_teacher_block_index` para escolher o teacher.
- Tamanho do encoder: reduza para `dinov2_vits14` + `224` se VRAM estiver apertada; mantenha `dinov2_vitg14` + `518` para melhor qualidade.

## Agendamento de coeficiente

O CREPA suporta agendamento do coeficiente (`crepa_lambda`) ao longo do treinamento com warmup, decaimento e corte automatico baseado em limiar de similaridade. Isso e particularmente util para treinamento text2video onde o CREPA pode causar listras horizontais/verticais ou uma aparencia desbotada se aplicado muito forte por muito tempo.

### Agendamento basico

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

Esta configuracao:
1. Aumenta o peso do CREPA de 0 para 0.5 nos primeiros 100 steps
2. Decai de 0.5 para 0.0 usando um agendamento cosseno em 5000 steps
3. Apos o step 5100, o CREPA esta efetivamente desabilitado

### Tipos de agendamento

- `constant`: Sem decaimento, o peso permanece em `crepa_lambda` (padrao)
- `linear`: Interpolacao linear de `crepa_lambda` ate `crepa_lambda_end`
- `cosine`: Anelamento cosseno suave (recomendado para a maioria dos casos)
- `polynomial`: Decaimento polinomial com potencia configuravel via `crepa_power`

### Corte baseado em steps

Para um corte rigido apos um step especifico:

```json
{
  "crepa_cutoff_step": 3000
}
```

O CREPA e completamente desabilitado apos o step 3000.

### Corte baseado em similaridade

Esta e a abordagem mais flexivel: o CREPA desabilita automaticamente quando a metrica de similaridade estabiliza, indicando que o modelo aprendeu alinhamento temporal suficiente:

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`: Quando a media movel exponencial da similaridade atinge este valor, o CREPA e cortado
- `crepa_similarity_ema_decay`: Fator de suavizacao (0.99 ≈ janela de 100 steps)
- `crepa_threshold_mode`: `permanent` (permanece desligado) ou `recoverable` (pode reabilitar se a similaridade cair)

### Configuracoes recomendadas

**Para image2video (i2v)**:
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
O CREPA padrao funciona bem para i2v ja que o frame de referencia ancora a consistencia.

**Para text2video (t2v)**:
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
Decai o CREPA ao longo do treinamento e corta quando a similaridade satura para prevenir artefatos.

**Para fundos solidos (t2v)**:
```json
{
  "crepa_cutoff_step": 2000
}
```
Corte antecipado previne artefatos de listras em fundos uniformes.

<details>
<summary>Como funciona (pratico)</summary>

- Captura hidden states de um bloco DiT escolhido, projeta via LayerNorm+Linear head e alinha com recursos de visao congelados.
- Por padrao, codifica frames em pixels com DINOv2; modo backbone reutiliza um bloco transformer mais profundo.
- Alinha cada frame aos seus vizinhos com decaimento exponencial por distancia (`crepa_adjacent_tau`); modo cumulativo soma opcionalmente todos os offsets ate `d`.
- Alinhamento espacial/temporal reamostra tokens para que patches do DiT e do encoder se alinhem antes da similaridade de cosseno; a loss e media sobre patches e frames.

</details>

<details>
<summary>Tecnico (internals do SimpleTuner)</summary>

- Implementacao: `simpletuner/helpers/training/crepa.py`; registrada em `ModelFoundation._init_crepa_regularizer` e anexada ao modelo treinavel (projetor fica no modelo para cobertura do otimizador).
- Captura de hidden state: transformers de video guardam `crepa_hidden_states` (e opcionalmente `crepa_frame_features`) quando `crepa_enabled` e true; modo backbone tambem pode puxar `layer_{idx}` do buffer de hidden states compartilhado.
- Caminho de loss: decodifica latentes com o VAE para pixels a menos que `crepa_use_backbone_features` esteja ligado; normaliza hidden states projetados e recursos do encoder, aplica similaridade de cosseno ponderada por distancia, registra `crepa_loss` / `crepa_similarity` e adiciona a loss escalada.
- Interacao: roda antes de LayerSync para que ambos reutilizem o buffer de hidden states; limpa o buffer depois. Exige um indice de bloco valido e um hidden size inferido do config do transformer.

</details>

## Armadilhas comuns

- Habilitar CREPA em familias nao suportadas leva a hidden states ausentes; use `kandinsky5_video`, `ltxvideo`, `sanavideo` ou `wan`.
- **Indice de bloco alto demais** -> “hidden states not returned”. Reduza o indice; ele e baseado em zero nos blocos transformer.
- **Picos de VRAM** -> tente `crepa_spatial_align=false`, um encoder menor (`dinov2_vits14` + `224`), ou um indice de bloco menor.
- **Erros no modo backbone** -> defina `crepa_block_index` (student) e `crepa_teacher_block_index` (teacher) para camadas que existam.
- **Falta de memoria** -> Se o RamTorch nao ajudar, sua unica solucao pode ser uma GPU maior; se H200 ou B200 nao funcionarem, abra um issue report.
