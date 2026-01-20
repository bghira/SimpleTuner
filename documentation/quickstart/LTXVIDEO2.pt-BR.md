# Guia de Início Rápido do LTX Video 2

Neste exemplo, vamos treinar um LoRA do LTX Video 2 usando os VAEs de vídeo/áudio do LTX-2 e um encoder de texto Gemma3.

## Requisitos de hardware

LTX Video 2 é um modelo pesado de **19B**. Ele combina:
1.  **Gemma3**: O encoder de texto.
2.  **LTX-2 Video VAE** (mais o Audio VAE quando há condicionamento por áudio).
3.  **Video Transformer de 19B**: Um backbone DiT grande.

Este setup é intensivo em VRAM, e o pré-cache do VAE pode aumentar o uso de memória.

- **Treino em uma GPU**: Comece com `train_batch_size: 1` e habilite group offload.
  - **Nota**: O **passo inicial de pré-cache do VAE** pode exigir mais VRAM. Talvez seja necessário offload para CPU ou uma GPU maior apenas na fase de cache.
  - **Dica**: Defina `"offload_during_startup": true` no seu `config.json` para garantir que o VAE e o encoder de texto não sejam carregados na GPU ao mesmo tempo, reduzindo bastante a pressão de memória do pré-cache.
- **Treino multi-GPU**: **FSDP2** ou **Group Offload** agressivo é recomendado se você precisa de mais folga.
- **RAM do sistema**: 64GB+ é recomendado para execuções maiores; mais RAM ajuda no cache.

### Desempenho e memória observados (relatos de campo)

- **Configuração base**: 480p, 17 frames, batch size 2 (mínima duração/resolução).
- **RamTorch (incl. encoder de texto)**: ~13 GB de VRAM em AMD 7900XTX.
  - NVIDIA 3090/4090/5090+ deve ter folga similar ou melhor.
- **Sem offload (int8 TorchAO)**: ~29-30 GB de VRAM; hardware de 32 GB recomendado.
  - Pico de RAM do sistema: ~46 GB ao carregar Gemma3 bf16 e depois quantizar para int8 (~32 GB VRAM).
  - Pico de RAM do sistema: ~34 GB ao carregar o transformer LTX-2 bf16 e depois quantizar para int8 (~30 GB VRAM).
- **Sem offload (bf16 completo)**: ~48 GB de VRAM necessários para treinar o modelo sem offload.
- **Throughput**:
  - ~8 s/step em A100-80G SXM4 (sem compilação).
  - ~16 s/step em 7900XTX (execução local).
  - ~30 min para 200 steps em A100-80G SXM4.

### Offload de memória (Crítico)

Para a maioria dos setups de GPU única treinando o LTX Video 2, é recomendável habilitar offload em grupo. É opcional, mas ajuda a manter folga de VRAM em batches/resoluções maiores.

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

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
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

Configurações-chave para LTX Video 2:

- `model_family`: `ltxvideo2`
- `model_flavour`: `dev` (padrão), `dev-fp4` ou `dev-fp8`.
- `pretrained_model_name_or_path`: `Lightricks/LTX-2` (repositório com o checkpoint combinado) ou um arquivo `.safetensors` local.
- `train_batch_size`: `1`. Não aumente isso a menos que você tenha um A100/H100.
- `validation_resolution`:
  - `512x768` é um padrão seguro para testes.
  - `720x1280` (720p) é possível, mas pesado.
- `validation_num_video_frames`: **Deve ser compatível com a compressão do VAE (4x).**
  - Para 5s (em ~12-24fps): Use `61` ou `49`.
  - Fórmula: `(frames - 1) % 4 == 0`.
- `validation_guidance`: `5.0`.
- `frame_rate`: O padrão é 25.

O LTX-2 é distribuído como um único checkpoint `.safetensors` que inclui o transformer, o VAE de vídeo,
o VAE de áudio e o vocoder. O SimpleTuner carrega desse arquivo combinado conforme o `model_flavour` (dev/dev-fp4/dev-fp8).

### Opcional: otimizações de VRAM

Se precisar de mais folga de VRAM:
- **Block swap (Musubi)**: Defina `musubi_blocks_to_swap` (tente `4-8`) e opcionalmente `musubi_block_swap_device` (padrão `cpu`) para streamar os últimos blocos do transformer da CPU. Menor throughput, menor pico de VRAM.
- **Convolução por patches do VAE**: Defina `--vae_enable_patch_conv=true` para habilitar chunking temporal no VAE do LTX-2; espere uma pequena perda de velocidade, mas menor pico de VRAM.
- **Temporal roll do VAE**: Defina `--vae_enable_temporal_roll=true` para um chunking temporal mais agressivo (perda de velocidade maior).
- **VAE tiling**: Defina `--vae_enable_tiling=true` para dividir o encode/decode do VAE em resoluções grandes.

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
        "frame_rate": 25,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1,
        "duration_interval": 3.0
    },
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo2",
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

O auto-split de áudio fica habilitado por padrão em datasets de vídeo. Adicione um bloco `audio` para ajustar
sample rate/canais, defina `audio.auto_split: false` para desativar, ou forneça um dataset de áudio separado e
vincule via `s2v_datasets`. O SimpleTuner vai cachear os latentes de áudio junto com os latentes de vídeo.

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
- **Tremor de movimento**: Verifique se o frame rate do seu dataset corresponde ao frame rate em que o modelo foi treinado (geralmente 25fps).
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

### Configuração de menor uso de VRAM (7900XTX)

Configuração testada em campo que prioriza o menor uso de VRAM no LTX Video 2.

<details>
<summary>Ver configuração 7900XTX (menor uso de VRAM)</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/ltx2/multidatabackend.json",
  "disable_benchmark": true,
  "dynamo_mode": "",
  "evaluation_type": "none",
  "hub_model_id": "simpletuner-ltxvideo2-19b-t2v-lora-test",
  "learning_rate": 0.00006,
  "lr_warmup_steps": 50,
  "lycoris_config": "config/lycoris_config.json",
  "max_grad_norm": 0.1,
  "max_train_steps": 200,
  "minimum_image_size": 0,
  "model_family": "ltxvideo2",
  "model_flavour": "dev",
  "model_type": "lora",
  "num_train_epochs": 0,
  "offload_during_startup": true,
  "optimizer": "adamw_bf16",
  "output_dir": "output/examples/ltxvideo2-19b-t2v.peft-lora",
  "override_dataset_config": true,
  "ramtorch": true,
  "ramtorch_text_encoder": true,
  "report_to": "none",
  "resolution": 480,
  "scheduled_sampling_reflexflow": false,
  "seed": 42,
  "skip_file_discovery": "",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "example-training-run",
  "train_batch_size": 2,
  "vae_batch_size": 1,
  "vae_enable_patch_conv": true,
  "vae_enable_slicing": true,
  "vae_enable_temporal_roll": true,
  "vae_enable_tiling": true,
  "validation_disable": true,
  "validation_disable_unconditional": true,
  "validation_guidance": 5,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "validation_prompt": "🟫 is holding a sign that says hello world from ltxvideo2",
  "validation_resolution": "768x512",
  "validation_seed": 42,
  "validation_using_datasets": false
}
```
</details>

### Fluxos de validação (T2V vs I2V)

- **T2V (texto para vídeo)**: Deixe `validation_using_datasets: false` e use `validation_prompt` ou `validation_prompt_library`.
- **I2V (imagem para vídeo)**: Defina `validation_using_datasets: true` e aponte `eval_dataset_id` para um split de validação que forneça uma imagem de referência. A validação alterna para o pipeline de imagem para vídeo e usa essa imagem como condicionamento.
- **S2V (condicionado por áudio)**: Com `validation_using_datasets: true`, garanta que `eval_dataset_id` aponte para um dataset com `s2v_datasets` (ou o comportamento padrão de `audio.auto_split`). A validação carrega os latentes de áudio em cache automaticamente.

### Adaptadores de validação (LoRAs)

Os LoRAs da Lightricks podem ser aplicados na validação via `validation_adapter_path` (único) ou
`validation_adapter_config` (várias execuções). Esses repos usam filenames de peso não padrão, então use
`repo_id:weight_name`. Veja a coleção LTX-2 para os filenames corretos e assets relacionados:
https://huggingface.co/collections/Lightricks/ltx-2
- `Lightricks/LTX-2-19b-IC-LoRA-Canny-Control:ltx-2-19b-ic-lora-canny-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Depth-Control:ltx-2-19b-ic-lora-depth-control.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Detailer:ltx-2-19b-ic-lora-detailer.safetensors`
- `Lightricks/LTX-2-19b-IC-LoRA-Pose-Control:ltx-2-19b-ic-lora-pose-control.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In:ltx-2-19b-lora-camera-control-dolly-in.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out:ltx-2-19b-lora-camera-control-dolly-out.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left:ltx-2-19b-lora-camera-control-dolly-left.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right:ltx-2-19b-lora-camera-control-dolly-right.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down:ltx-2-19b-lora-camera-control-jib-down.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up:ltx-2-19b-lora-camera-control-jib-up.safetensors`
- `Lightricks/LTX-2-19b-LoRA-Camera-Control-Static:ltx-2-19b-lora-camera-control-static.safetensors`

Para validação mais rápida, use `Lightricks/LTX-2-19b-distilled-lora-384:ltx-2-19b-distilled-lora-384.safetensors`
como adaptador e defina `validation_guidance: 1` junto com `validation_num_inference_steps: 8`.
