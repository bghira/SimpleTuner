# Guia de Início Rápido do Hunyuan Video 1.5

Este guia mostra como treinar um LoRA no release **Hunyuan Video 1.5** (8.3B) da Tencent (`tencent/HunyuanVideo-1.5`) usando SimpleTuner.

## Requisitos de hardware

Hunyuan Video 1.5 é um modelo grande (8.3B parâmetros).

- **Mínimo**: **24GB-32GB de VRAM** é confortável para um LoRA rank 16 com gradient checkpointing completo em 480p.
- **Recomendado**: A6000 / A100 (48GB-80GB) para treino em 720p ou batch sizes maiores.
- **RAM do sistema**: **64GB+** é recomendado para lidar com o carregamento do modelo.

### Offload de memória (opcional)

Adicione o seguinte ao seu `config.json`:

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

- `--group_offload_use_stream`: Funciona apenas em dispositivos CUDA.
- **Não** combine isso com `--enable_model_cpu_offload`.

## Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.13 python3.13-venv
```

### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para habilitar a compilação de extensões CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

### Etapas adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Checkpoints obrigatórios

O repositório principal `tencent/HunyuanVideo-1.5` contém o transformer/vae/scheduler, mas o **encoder de texto** (`text_encoder/llm`) e o **encoder de visão** (`vision_encoder/siglip`) ficam em downloads separados. Aponte o SimpleTuner para suas cópias locais antes de iniciar:

```bash
export HUNYUANVIDEO_TEXT_ENCODER_PATH=/path/to/text_encoder_root
export HUNYUANVIDEO_VISION_ENCODER_PATH=/path/to/vision_encoder_root
```

Se esses valores não estiverem definidos, o SimpleTuner tenta puxá-los do repositório do modelo; a maioria dos mirrors não os inclui, então defina os caminhos explicitamente para evitar erros de inicialização.

## Configurando o ambiente

### Método da interface web

A WebUI do SimpleTuner torna a configuração bastante direta. Para rodar o servidor:

```bash
simpletuner server
```

Isso vai criar um servidor web na porta 8001 por padrão, que você pode acessar em http://localhost:8001.

### Método manual / linha de comando

Para rodar o SimpleTuner via linha de comando, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo.

**Nota:** Isso não configura seu dataloader. Você ainda precisará fazer isso manualmente depois.

Para executá-lo:

```bash
simpletuner configure
```

Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Sobrescritas de configuração-chave para HunyuanVideo:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "hunyuanvideo",
  "pretrained_model_name_or_path": "tencent/HunyuanVideo-1.5",
  "model_flavour": "t2v-480p",
  "output_dir": "output/hunyuan-video",
  "validation_resolution": "854x480",
  "validation_num_video_frames": 61,
  "validation_guidance": 6.0,
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "learning_rate": 1e-4,
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "lora_rank": 16,
  "enable_group_offload": true,
  "group_offload_type": "block_level",
  "dataset_backend_config": "config/multidatabackend.json"
}
```
</details>

- Opções de `model_flavour`:
  - `t2v-480p` (Padrão)
  - `t2v-720p`
  - `i2v-480p` (Image-to-Video)
  - `i2v-720p` (Image-to-Video)
- `validation_num_video_frames`: Deve ser `(frames - 1) % 4 == 0`. Ex.: 61, 129.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Considerações sobre o dataset

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

```json
[
  {
    "id": "my-video-dataset",
    "type": "local",
    "dataset_type": "video",
    "instance_data_dir": "datasets/videos",
    "caption_strategy": "textfile",
    "resolution": 480,
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
    "cache_dir": "cache/text/hunyuan",
    "disabled": false
  }
]
```

Na subseção `video`:
- `num_frames`: Contagem de frames alvo para treino. Deve satisfazer `(frames - 1) % 4 == 0`.
- `min_frames`: Comprimento mínimo de vídeo (vídeos mais curtos são descartados).
- `max_frames`: Filtro de comprimento máximo de vídeo.
- `bucket_strategy`: Como os vídeos são agrupados em buckets:
  - `aspect_ratio` (padrão): agrupa apenas pela proporção espacial.
  - `resolution_frames`: agrupa pelo formato `WxH@F` (ex.: `854x480@61`) para datasets com resolução/duração mistas.
- `frame_interval`: Ao usar `resolution_frames`, arredonde a contagem de frames para este intervalo.

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

- **Cache de Text Embeds**: Altamente recomendado. Hunyuan usa um grande encoder de texto LLM. O cache economiza bastante VRAM durante o treino.

#### Login no WandB e Huggingface Hub

```bash
wandb login
huggingface-cli login
```

</details>

### Executando o treinamento

A partir do diretório do SimpleTuner:

```bash
simpletuner train
```

## Notas e dicas de troubleshooting

### Otimização de VRAM

- **Group Offload**: Essencial para GPUs de consumidor. Garanta que `enable_group_offload` esteja true.
- **Resolução**: Fique em 480p (`854x480` ou similar) se você tem VRAM limitada. 720p (`1280x720`) aumenta muito o uso de memória.
- **Quantização**: Use `base_model_precision` (padrão `bf16`); `int8-torchao` funciona para economizar mais, ao custo de velocidade.
- **Convolução de patch do VAE**: Para OOMs no VAE do HunyuanVideo, defina `--vae_enable_patch_conv=true` (ou ative na UI). Isso fatia o trabalho de conv/atenção 3D para reduzir o pico de VRAM; espere uma pequena perda de throughput.

### Image-to-Video (I2V)

- Use `model_flavour="i2v-480p"` ou `i2v-720p`.
- O SimpleTuner usa automaticamente o primeiro frame das amostras do seu dataset de vídeo como imagem de condicionamento durante o treinamento.

#### Opções de Validação I2V

Para validação com modelos i2v, você tem duas opções:

1. **Primeiro frame extraído automaticamente**: Por padrão, a validação usa o primeiro frame das amostras de vídeo no seu dataset.

2. **Dataset de imagens separado** (setup mais simples): Use `--validation_using_datasets=true` com `--eval_dataset_id` apontando para um dataset de imagens. Isso permite usar qualquer dataset de imagens como entrada de condicionamento do primeiro frame para vídeos de validação, sem precisar configurar o pareamento complexo de datasets de condicionamento usado durante o treinamento.

Exemplo de config para opção 2:
```json
{
  "validation_using_datasets": true,
  "eval_dataset_id": "my-image-dataset"
}
```

### Encoders de texto

Hunyuan usa um setup com dois encoders de texto (LLM + CLIP). Garanta que sua RAM do sistema consegue lidar com o carregamento deles durante a fase de cache.
