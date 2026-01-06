# CREPA (regularizacao de video)

Cross-frame Representation Alignment (CREPA) e um regularizador leve para modelos de video. Ele empurra os hidden states de cada frame em direcao aos recursos de um encoder de visao congelado do frame atual **e de seus vizinhos**, melhorando a consistencia temporal sem mudar sua loss principal.

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
