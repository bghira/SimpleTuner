# Guia de Início Rápido do Stable Cascade Stage C

Este guia mostra como configurar o SimpleTuner para fazer fine-tuning do **prior Stable Cascade Stage C**. O Stage C aprende o prior de texto-para-imagem que alimenta a pilha de decodificadores Stage B/C, então boas práticas de treinamento aqui melhoram diretamente as saídas do decodificador downstream. Vamos focar no treinamento LoRA, mas os mesmos passos se aplicam a fine-tunes completos se você tiver VRAM suficiente.

> **Aviso:** O Stage C usa o encoder de texto CLIP-G/14 com 1B+ parâmetros e um autoencoder baseado em EfficientNet. Garanta que o torchvision esteja instalado e espere caches grandes de text-embeds (aprox. 5-6x maiores por prompt do que no SDXL).

## Requisitos de hardware

- **Treinamento LoRA:** 20-24 GB de VRAM (RTX 3090/4090, A6000, etc.).
- **Treinamento full-model:** 48 GB+ de VRAM recomendado (A6000, A100, H100). Offload via DeepSpeed/FSDP2 pode reduzir a exigência, mas adiciona complexidade.
- **RAM do sistema:** 32 GB recomendados para que o encoder de texto CLIP-G e as threads de cache não fiquem sem recursos.
- **Disco:** Reserve pelo menos ~50 GB para arquivos de cache de prompts. As embeddings CLIP-G do Stage C têm ~4-6 MB cada.

## Pré-requisitos

1. Python 3.13 (compatível com a `.venv` do projeto).
2. CUDA 12.1+ ou ROCm 5.7+ para aceleração em GPU (ou Apple Metal para Macs M-series, embora o Stage C seja majoritariamente testado em CUDA).
3. `torchvision` (obrigatório para o autoencoder do Stable Cascade) e `accelerate` para iniciar o treinamento.

Verifique sua versão do Python:

```bash
python --version
```

Instale pacotes ausentes (exemplo no Ubuntu):

```bash
sudo apt update && sudo apt install -y python3.13 python3.13-venv
```

## Instalação

Siga a instalação padrão do SimpleTuner (pip ou código-fonte). Para uma workstation CUDA típica:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Para contribuidores ou quem estiver trabalhando diretamente no repositório, instale a partir da fonte e depois rode `pip install -e .[cuda,dev]`.

## Configurando o ambiente

### 1. Copie a configuração base

```bash
cp config/config.json.example config/config.json
```

Defina as seguintes chaves (valores mostrados são um bom ponto de partida para Stage C):

| Chave | Recomendação | Notas |
| --- | -------------- | ----- |
| `model_family` | `"stable_cascade"` | Obrigatório para carregar os componentes do Stage C |
| `model_flavour` | `"stage-c"` (ou `"stage-c-lite"`) | A versão lite reduz parâmetros se você só tiver ~18 GB de VRAM |
| `model_type` | `"lora"` | Fine-tune completo funciona, mas exige bem mais memória |
| `mixed_precision` | `"no"` | O Stage C se recusa a rodar em mixed precision a menos que você defina `i_know_what_i_am_doing=true`; fp32 é a escolha segura |
| `gradient_checkpointing` | `true` | Economiza 3-4 GB de VRAM |
| `vae_batch_size` | `1` | O autoencoder do Stage C é pesado; mantenha baixo |
| `validation_resolution` | `"1024x1024"` | Corresponde às expectativas do decodificador downstream |
| `stable_cascade_use_decoder_for_validation` | `true` | Garante que a validação use o pipeline combinado prior+decodificador |
| `stable_cascade_decoder_model_name_or_path` | `"stabilityai/stable-cascade"` | Mude para um caminho local se você tiver um decodificador Stage B/C customizado |
| `stable_cascade_validation_prior_num_inference_steps` | `20` | Steps de denoising do prior |
| `stable_cascade_validation_prior_guidance_scale` | `3.0–4.0` | CFG no prior |
| `stable_cascade_validation_decoder_guidance_scale` | `0.0–0.5` | CFG do decodificador (0.0 é fotorealista, >0.0 aumenta aderência ao prompt) |

#### Exemplo de `config/config.json`

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "base_model_precision": "int8-torchao",
  "checkpoint_step_interval": 100,
  "data_backend_config": "config/stable_cascade/multidatabackend.json",
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": true,
  "hub_model_id": "stable-cascade-stage-c-lora",
  "learning_rate": 1e-4,
  "lora_alpha": 16,
  "lora_rank": 16,
  "lora_type": "standard",
  "lr_scheduler": "cosine",
  "max_train_steps": 30000,
  "mixed_precision": "no",
  "model_family": "stable_cascade",
  "model_flavour": "stage-c",
  "model_type": "lora",
  "optimizer": "adamw_bf16",
  "output_dir": "output/stable_cascade_stage_c",
  "report_to": "wandb",
  "seed": 42,
  "stable_cascade_decoder_model_name_or_path": "stabilityai/stable-cascade",
  "stable_cascade_decoder_subfolder": "decoder_lite",
  "stable_cascade_use_decoder_for_validation": true,
  "stable_cascade_validation_decoder_guidance_scale": 0.0,
  "stable_cascade_validation_prior_guidance_scale": 3.5,
  "stable_cascade_validation_prior_num_inference_steps": 20,
  "train_batch_size": 4,
  "use_ema": true,
  "vae_batch_size": 1,
  "validation_guidance": 4.0,
  "validation_negative_prompt": "ugly, blurry, low-res",
  "validation_num_inference_steps": 30,
  "validation_prompt": "a cinematic photo of a shiba inu astronaut",
  "validation_resolution": "1024x1024"
}
```
</details>

Principais pontos:

- `model_flavour` aceita `stage-c` e `stage-c-lite`. Use lite se estiver curto de VRAM ou preferir o prior destilado.
- Mantenha `mixed_precision` em `"no"`. Se você sobrescrever, defina `i_know_what_i_am_doing=true` e esteja pronto para NaNs.
- Ativar `stable_cascade_use_decoder_for_validation` conecta a saída do prior ao decodificador Stage B/C para que a galeria de validação mostre imagens reais em vez de latentes do prior.

### 2. Configure o backend de dados

Crie `config/stable_cascade/multidatabackend.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "primary",
    "type": "local",
    "dataset_type": "images",
    "instance_data_dir": "/data/stable-cascade",
    "resolution": "1024x1024",
    "bucket_resolutions": ["1024x1024", "896x1152", "1152x896"],
    "crop": true,
    "crop_style": "random",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "caption_strategy": "filename",
    "prepend_instance_prompt": false,
    "repeats": 1
  },
  {
    "id": "stable-cascade-text-cache",
    "type": "local",
    "dataset_type": "text_embeds",
    "cache_dir": "/data/cache/stable-cascade/text",
    "default": true
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Dicas:

- Os latentes do Stage C são derivados de um autoencoder, então mantenha 1024x1024 (ou um intervalo pequeno de buckets retrato/paisagem). O decodificador espera grades latentes ~24x24 de uma entrada de 1024px.
- Mantenha `target_downsample_size` em 1024 para que recortes estreitos não explodam proporções além de ~2:1.
- Sempre configure um cache dedicado de text-embeds. Sem isso, cada execução vai gastar 30-60 minutos re-embeddando captions com CLIP-G.

### 3. Biblioteca de prompts (opcional)

Crie `config/stable_cascade/prompt_library.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "portrait": "a cinematic portrait photograph lit by studio strobes",
  "landscape": "a sweeping ultra wide landscape with volumetric lighting",
  "product": "a product render on a seamless background, dramatic reflections",
  "stylized": "digital illustration in the style of a retro sci-fi book cover"
}
```
</details>

Habilite no seu config adicionando `"validation_prompt_library": "config/stable_cascade/prompt_library.json"`.

## Treinamento

1. Ative seu ambiente e inicie a configuração do Accelerate caso ainda não tenha feito:

```bash
source .venv/bin/activate
accelerate config
```

2. Inicie o treinamento:

```bash
accelerate launch simpletuner/train.py \
  --config_file config/config.json \
  --data_backend_config config/stable_cascade/multidatabackend.json
```

Durante a primeira época, monitore:

- **Vazão do cache de texto** – O Stage C vai registrar o progresso do cache. Espere ~8-12 prompts/seg em GPUs de ponta.
- **Uso de VRAM** – Busque <95% de utilização para evitar OOM quando a validação rodar.
- **Saídas de validação** – O pipeline combinado deve gerar PNGs em resolução total em `output/<run>/validation/`.

## Notas de validação e inferência

- O prior do Stage C sozinho produz apenas embeddings de imagem. O wrapper de validação do SimpleTuner alimenta automaticamente o decodificador quando `stable_cascade_use_decoder_for_validation=true`.
- Para trocar a versão do decodificador, defina `stable_cascade_decoder_subfolder` como `"decoder"`, `"decoder_lite"` ou uma pasta customizada contendo os pesos do Stage B ou Stage C.
- Para prévias mais rápidas, reduza `stable_cascade_validation_prior_num_inference_steps` para ~12 e `validation_num_inference_steps` para 20. Depois de satisfeito, aumente novamente para maior qualidade.

## Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** permite treinar o Stable Cascade com um objetivo de Flow Matching.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

</details>

## Solução de problemas

| Sintoma | Correção |
| --- | --- |
| "Stable Cascade Stage C requires --mixed_precision=no" | Defina `"mixed_precision": "no"` ou adicione `"i_know_what_i_am_doing": true` (não recomendado) |
| A validação só mostra priors (ruído verde) | Garanta que `stable_cascade_use_decoder_for_validation` esteja `true` e que os pesos do decodificador foram baixados |
| Cache de text-embeds leva horas | Use SSD/NVMe para o diretório de cache e evite mounts de rede. Considere podar prompts ou pré-computar com a CLI `simpletuner-text-cache` |
| Erro de importação do autoencoder | Instale torchvision na sua `.venv` (`pip install torchvision --extra-index-url https://download.pytorch.org/whl/cu124`). O Stage C precisa dos pesos do EfficientNet |

## Próximos passos

- Experimente `lora_rank` (8-32) e `learning_rate` (5e-5 a 2e-4) dependendo da complexidade do assunto.
- Anexe ControlNet/adapters de condicionamento ao Stage B após treinar o prior.
- Se precisar de iteração mais rápida, treine o flavour `stage-c-lite` e mantenha os pesos `decoder_lite` para validação.

Bom treino!
