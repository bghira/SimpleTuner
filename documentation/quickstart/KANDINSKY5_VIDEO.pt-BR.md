# Guia de Início Rápido do Kandinsky 5.0 Video

Neste exemplo, vamos treinar um LoRA de Kandinsky 5.0 Video (Lite ou Pro) usando o VAE do HunyuanVideo e encoders de texto duplos.

## Requisitos de hardware

Kandinsky 5.0 Video é um modelo pesado. Ele combina:
1.  **Qwen2.5-VL (7B)**: Um enorme encoder de texto visão-linguagem.
2.  **HunyuanVideo VAE**: Um VAE 3D de alta qualidade.
3.  **Video Transformer**: Uma arquitetura DiT complexa.

Este setup é intensivo em VRAM, embora as variantes "Lite" e "Pro" tenham requisitos diferentes.

- **Treino do modelo Lite**: Surpreendentemente eficiente, capaz de treinar com **~13GB de VRAM**.
  - **Nota**: O **passo inicial de pré-cache do VAE** exige significativamente mais VRAM devido ao grande VAE do HunyuanVideo. Talvez seja necessário usar offload para CPU ou uma GPU maior apenas para a fase de cache.
  - **Dica**: Defina `"offload_during_startup": true` no seu `config.json` para garantir que o VAE e o encoder de texto não sejam carregados na GPU ao mesmo tempo, o que reduz bastante a pressão de memória do pré-cache.
  - **Se o VAE der OOM**: Defina `--vae_enable_patch_conv=true` para fatiar as convs 3D do VAE do HunyuanVideo; espere uma pequena perda de velocidade, mas menor pico de VRAM.
- **Treino do modelo Pro**: Requer **FSDP2** (multi-GPU) ou **Group Offload** agressivo com LoRA para caber em hardware de consumidor. Requisitos específicos de VRAM/RAM não foram estabelecidos, mas "quanto mais, melhor" se aplica.
- **RAM do sistema**: Testes foram confortáveis em um sistema com **45GB** de RAM para o modelo Lite. 64GB+ é recomendado para garantir.

### Offload de memória (Crítico)

Para praticamente qualquer setup de GPU única treinando o modelo **Pro**, você **deve** habilitar offload em grupo. É opcional, mas recomendado para **Lite** para economizar VRAM em batches/resoluções maiores.

Adicione isto ao seu `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "group_offload_blocks_per_group": 1,
  "group_offload_use_stream": true
}
```
</details>

## Pré-requisitos

Garanta que o Python 3.12 esteja instalado.

```bash
python --version
```

## Instalação

```bash
pip install 'simpletuner[cuda]'
```

Veja [INSTALL.md](../INSTALL.md) para opções avançadas de instalação.

## Configurando o ambiente

### Interface web

```bash
simpletuner server
```
Acesse em http://localhost:8001.

### Configuração manual

Execute o script auxiliar:

```bash
simpletuner configure
```

Ou copie o exemplo e edite manualmente:

```bash
cp config/config.json.example config/config.json
```

#### Parâmetros de configuração

Configurações-chave para Kandinsky 5 Video:

- `model_family`: `kandinsky5-video`
- `model_flavour`:
  - `t2v-lite-sft-5s`: Modelo Lite, saída ~5s. (Padrão)
  - `t2v-lite-sft-10s`: Modelo Lite, saída ~10s.
  - `t2v-pro-sft-5s-hd`: Modelo Pro, ~5s, treino em alta definição.
  - `t2v-pro-sft-10s-hd`: Modelo Pro, ~10s, treino em alta definição.
  - `i2v-lite-5s`: Image-to-video Lite, saídas de 5s (requer imagens de condicionamento).
  - `i2v-pro-sft-5s`: Image-to-video Pro SFT, saídas de 5s (requer imagens de condicionamento).
  - *(Variantes pretrain também estão disponíveis para todas as opções acima)*
- `train_batch_size`: `1`. Não aumente isso a menos que você tenha um A100/H100.
- `validation_resolution`:
  - `512x768` é um padrão seguro para testes.
  - `720x1280` (720p) é possível, mas pesado.
- `validation_num_video_frames`: **Deve ser compatível com a compressão do VAE (4x).**
  - Para 5s (em ~12-24fps): Use `61` ou `49`.
  - Fórmula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: O padrão é 24.

### Opcional: regularizador temporal CREPA

Para reduzir flicker e manter assuntos estáveis entre frames:
- Em **Training → Loss functions**, habilite **CREPA**.
- Valores iniciais recomendados: **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Mantenha o encoder de visão padrão (`dinov2_vitg14`, tamanho `518`) a menos que você precise de um menor (`dinov2_vits14` + `224`).
- Requer rede (ou um torch hub em cache) para buscar os pesos do DINOv2 na primeira vez.
- Só habilite **Drop VAE Encoder** se você estiver treinando inteiramente a partir de latentes em cache; caso contrário, deixe desligado.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Considerações sobre o dataset

Datasets de vídeo exigem configuração cuidadosa. Crie `config/multidatabackend.json`:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 512,
    "video": {
        "num_frames": 61,
        "min_frames": 61,
        "frame_rate": 24,
        "bucket_strategy": "aspect_ratio"
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/kandinsky5",
    "disabled": false
  }
]
```

Na subseção `video`:
- `num_frames`: Contagem de frames alvo para treino.
- `min_frames`: Comprimento mínimo de vídeo (vídeos mais curtos são descartados).
- `max_frames`: Filtro de comprimento máximo de vídeo.
- `bucket_strategy`: Como os vídeos são agrupados em buckets:
  - `aspect_ratio` (padrão): agrupa apenas pela proporção espacial.
  - `resolution_frames`: agrupa pelo formato `WxH@F` (ex.: `1920x1080@61`) para datasets com resolução/duração mistas.
- `frame_interval`: Ao usar `resolution_frames`, arredonde a contagem de frames para este intervalo.

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

#### Configuração de diretórios

```bash
mkdir -p datasets/videos
</details>

# Place .mp4 / .mov files here.
# Place corresponding .txt files with same filename for captions.
```

#### Login

```bash
wandb login
huggingface-cli login
```

### Executando o treinamento

```bash
simpletuner train
```

## Notas e dicas de troubleshooting

### Sem memória (OOM)

Treino de vídeo é extremamente exigente. Se der OOM:

1.  **Reduza a resolução**: Tente 480p (`480x854` ou similar).
2.  **Reduza frames**: Baixe `validation_num_video_frames` e `num_frames` do dataset para `33` ou `49`.
3.  **Cheque o offload**: Garanta que `--enable_group_offload` está ativo.

### Qualidade do vídeo de validação

- **Vídeos pretos/ruído**: Geralmente causados por `validation_guidance` alto demais (> 6.0) ou baixo demais (< 2.0). Fique em `5.0`.
- **Tremor de movimento**: Verifique se o frame rate do seu dataset corresponde ao frame rate em que o modelo foi treinado (geralmente 24fps).
- **Vídeo estático**: O modelo pode estar subtreinado ou o prompt não descreve movimento. Use prompts como "camera pans right", "zoom in", "running", etc.

### Treino TREAD

TREAD funciona para vídeo também e é altamente recomendado para economizar compute.

Adicione ao `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Isso pode acelerar o treino em ~25-40% dependendo da razão.

### Treino I2V (Image-to-Video)

Se usar flavours `i2v`:
- O SimpleTuner extrai automaticamente o primeiro frame dos vídeos de treino para usar como imagem de condicionamento.
- O pipeline mascara automaticamente o primeiro frame durante o treinamento.
- A validação exige fornecer uma imagem de entrada, ou o SimpleTuner usará o primeiro frame da geração de vídeo de validação como condicionador.
