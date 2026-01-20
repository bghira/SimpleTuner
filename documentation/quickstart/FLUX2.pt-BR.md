# Guia de Início Rápido do FLUX.2

Este guia cobre o treinamento de LoRAs no FLUX.2, a mais recente família de modelos de geração de imagens da Black Forest Labs.

> **Nota**: O flavour padrão é `klein-9b`, mas este guia foca no `dev` (o transformer completo de 12B com encoder de texto Mistral-3 de 24B) por ter os maiores requisitos de recursos. Os modelos Klein são mais fáceis de executar - veja [Variantes do modelo](#variantes-do-modelo) abaixo.

## Variantes do modelo

O FLUX.2 vem em três variantes:

| Variante | Transformer | Encoder de texto | Total de blocos | Padrão |
|---------|-------------|------------------|-----------------|--------|
| `dev` | 12B parâmetros | Mistral-3 (24B) | 56 (8+48) | |
| `klein-9b` | 9B parâmetros | Qwen3 (incluso) | 32 (8+24) | ✓ |
| `klein-4b` | 4B parâmetros | Qwen3 (incluso) | 25 (5+20) | |

**Diferenças principais:**
- **dev**: Usa encoder de texto Mistral-Small-3.1-24B autônomo, possui embeddings de orientação
- **modelos klein**: Usam encoder de texto Qwen3 incluso no repositório do modelo, **sem embeddings de orientação** (opções de treinamento de orientação são ignoradas)

Para selecionar uma variante, defina `model_flavour` na sua configuração:
```json
{
  "model_flavour": "dev"
}
```

## Visão geral do modelo

O FLUX.2-dev introduz mudanças arquiteturais significativas em relação ao FLUX.1:

- **Encoder de texto**: Mistral-Small-3.1-24B (dev) ou Qwen3 (klein)
- **Arquitetura**: 8 DoubleStreamBlocks + 48 SingleStreamBlocks (dev)
- **Canais latentes**: 32 canais no VAE → 128 após pixel shuffle (vs 16 no FLUX.1)
- **VAE**: VAE customizado com batch normalization e pixel shuffling
- **Dimensão de embedding**: 15.360 para dev (3×5.120), 12.288 para klein-9b (3×4.096), 7.680 para klein-4b (3×2.560)

## Requisitos de hardware

Os requisitos de hardware variam significativamente por variante do modelo.

### Modelos Klein (Recomendado para a maioria dos usuários)

Os modelos Klein são muito mais acessíveis:

| Variante | VRAM bf16 | VRAM int8 | RAM do sistema |
|---------|-----------|-----------|----------------|
| `klein-4b` | ~12GB | ~8GB | 32GB+ |
| `klein-9b` | ~22GB | ~14GB | 64GB+ |

**Recomendado para klein-9b**: GPU única de 24GB (RTX 3090/4090, A5000)
**Recomendado para klein-4b**: GPU única de 16GB (RTX 4080, A4000)

### FLUX.2-dev (Avançado)

O FLUX.2-dev tem requisitos de recursos significativos devido ao encoder de texto Mistral-3:

#### Requisitos de VRAM

O encoder de texto Mistral 24B sozinho exige VRAM considerável:

| Componente | bf16 | int8 | int4 |
|-----------|------|------|------|
| Mistral-3 (24B) | ~48GB | ~24GB | ~12GB |
| Transformer FLUX.2 | ~24GB | ~12GB | ~6GB |
| VAE + overhead | ~4GB | ~4GB | ~4GB |

| Configuração | VRAM total aproximada |
|--------------|-----------------------|
| bf16 em tudo | ~76GB+ |
| encoder de texto int8 + transformer bf16 | ~52GB |
| tudo em int8 | ~40GB |
| encoder de texto int4 + transformer int8 | ~22GB |

#### RAM do sistema

- **Mínimo**: 96GB de RAM do sistema (carregar o encoder 24B exige memória substancial)
- **Recomendado**: 128GB+ para operação confortável

#### Hardware recomendado

- **Mínimo**: 2x GPUs 48GB (A6000, L40S) com FSDP2 ou DeepSpeed
- **Recomendado**: 4x H100 80GB com fp8-torchao
- **Com quantização pesada (int4)**: 2x GPUs 24GB podem funcionar, mas é experimental

Treinamento distribuído multi-GPU (FSDP2 ou DeepSpeed) é essencialmente obrigatório para o FLUX.2-dev devido ao tamanho combinado do encoder de texto Mistral-3 e do transformer.

## Pré-requisitos

### Versão do Python

FLUX.2 exige Python 3.10 ou superior com transformers recentes:

```bash
python --version  # Should be 3.10+
pip install transformers>=4.45.0
```

### Acesso ao modelo

Os modelos FLUX.2 exigem aprovação de acesso no Hugging Face:

**Para dev:**
1. Visite [black-forest-labs/FLUX.2-dev](https://huggingface.co/black-forest-labs/FLUX.2-dev)
2. Aceite o acordo de licença

**Para modelos klein:**
1. Visite [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) ou [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)
2. Aceite o acordo de licença

Garanta que você esteja logado no Hugging Face CLI: `huggingface-cli login`

## Instalação

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]'
```

Para setup de desenvolvimento:
```bash
git clone https://github.com/bghira/SimpleTuner
cd SimpleTuner
pip install -e ".[cuda]"
```

## Configuração

### Interface web

```bash
simpletuner server
```

Acesse http://localhost:8001 e selecione FLUX.2 como família de modelo.

### Configuração manual

Crie `config/config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "model_type": "lora",
  "model_family": "flux2",
  "model_flavour": "dev",
  "pretrained_model_name_or_path": "black-forest-labs/FLUX.2-dev",
  "output_dir": "/path/to/output",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "gradient_checkpointing": true,
  "mixed_precision": "bf16",
  "learning_rate": 1e-4,
  "lr_scheduler": "constant",
  "max_train_steps": 10000,
  "validation_resolution": "1024x1024",
  "validation_num_inference_steps": 20,
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0,
  "lora_rank": 16
}
```
</details>

### Opções-chave de configuração

#### Configuração de guidance

> **Nota**: Os modelos Klein (`klein-4b`, `klein-9b`) não possuem embeddings de orientação. As seguintes opções de guidance aplicam-se apenas ao `dev`.

FLUX.2-dev usa embedding de guidance semelhante ao FLUX.1:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "flux_guidance_mode": "constant",
  "flux_guidance_value": 1.0
}
```
</details>

Ou para guidance aleatório durante o treinamento:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "flux_guidance_mode": "random-range",
  "flux_guidance_min": 1.0,
  "flux_guidance_max": 5.0
}
```
</details>

#### Quantização (otimização de memória)

Para reduzir o uso de VRAM:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "base_model_default_dtype": "bf16"
}
```
</details>

#### TREAD (aceleração de treinamento)

FLUX.2 suporta TREAD para treino mais rápido:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
</details>

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

### Configuração de dataset

Crie `config/multidatabackend.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "my-dataset",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux2/my-dataset",
    "instance_data_dir": "datasets/my-dataset",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 10
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux2",
    "write_batch_size": 64
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

### Condicionamento opcional de edição/referência

O FLUX.2 pode treinar tanto **texto-para-imagem** (sem condicionamento) quanto com **pares de imagens referência/edição**. Para adicionar condicionamento, pareie seu dataset principal a um ou mais datasets `conditioning` usando [`conditioning_data`](../DATALOADER.md#conditioning_data) e escolha um [`conditioning_type`](../DATALOADER.md#conditioning_type):

<details>
<summary>Ver exemplo de config</summary>

```jsonc
[
  {
    "id": "flux2-edits",
    "type": "local",
    "instance_data_dir": "/datasets/flux2/edits",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["flux2-references"],
    "cache_dir_vae": "cache/vae/flux2/edits"
  },
  {
    "id": "flux2-references",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/flux2/references",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/flux2/references"
  }
]
```
</details>

- Use `conditioning_type=reference_strict` quando você precisa de recortes alinhados 1:1 com a imagem de edição. `reference_loose` permite proporções diferentes.
- Os nomes de arquivo devem corresponder entre os datasets de edição e referência; cada imagem de edição deve ter um arquivo de referência correspondente.
- Ao fornecer múltiplos datasets de condicionamento, defina `conditioning_multidataset_sampling` (`combined` vs `random`) conforme necessário; veja [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling).
- Sem `conditioning_data`, o FLUX.2 retorna ao treinamento padrão texto-para-imagem.

### Alvos de LoRA

Presets de alvo de LoRA disponíveis:

- `all` (padrão): todas as camadas de atenção e MLP
- `attention`: apenas camadas de atenção (qkv, proj)
- `mlp`: apenas camadas MLP/feed-forward
- `tiny`: treinamento mínimo (somente camadas qkv)

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "--flux_lora_target": "all"
}
```
</details>

## Treinamento

### Login nos serviços

```bash
huggingface-cli login
wandb login  # optional
```

### Iniciar treinamento

```bash
simpletuner train
```

Ou via script:

```bash
./train.sh
```

### Offload de memória

Para setups com memória limitada, o FLUX.2 suporta group offload tanto para o transformer quanto, opcionalmente, para o encoder de texto Mistral-3:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
--group_offload_text_encoder
```

O flag `--group_offload_text_encoder` é recomendado para o FLUX.2, já que o encoder de texto Mistral 24B se beneficia bastante do offload durante o cache de embeddings de texto. Você também pode adicionar `--group_offload_vae` para incluir o VAE no offload durante o cache de latentes.

## Prompts de validação

Crie `config/user_prompt_library.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "portrait_subject": "a professional portrait photograph of <subject>, studio lighting, high detail",
  "artistic_subject": "an artistic interpretation of <subject> in the style of renaissance painting",
  "cinematic_subject": "a cinematic shot of <subject>, dramatic lighting, film grain"
}
```
</details>

## Inferência

### Usando LoRA treinado

LoRAs do FLUX.2 podem ser carregadas com o pipeline de inferência do SimpleTuner ou ferramentas compatíveis, à medida que o suporte da comunidade amadurece.

### Escala de guidance

- Treinar com `flux_guidance_value=1.0` funciona bem para a maioria dos casos
- Na inferência, use valores normais de guidance (3.0-5.0)

## Diferenças em relação ao FLUX.1

| Aspecto | FLUX.1 | FLUX.2-dev | FLUX.2-klein-9b | FLUX.2-klein-4b |
|--------|--------|------------|-----------------|-----------------|
| Encoder de texto | CLIP-L/14 + T5-XXL | Mistral-3 (24B) | Qwen3 (incluso) | Qwen3 (incluso) |
| Dimensão de embedding | CLIP: 768, T5: 4096 | 15.360 | 12.288 | 7.680 |
| Canais latentes | 16 | 32 (→128) | 32 (→128) | 32 (→128) |
| VAE | AutoencoderKL | Custom (BatchNorm) | Custom (BatchNorm) | Custom (BatchNorm) |
| Blocos do transformer | 19 joint + 38 single | 8 double + 48 single | 8 double + 24 single | 5 double + 20 single |
| Guidance embeds | Sim | Sim | Não | Não |

## Solução de problemas

### Sem memória durante a inicialização

- Ative `--offload_during_startup=true`
- Use `--quantize_via=cpu` para quantização do encoder de texto
- Reduza `--vae_batch_size`

### Embedding de texto lento

Mistral-3 é grande; considere:
- Pré-cachear todas as embeddings de texto antes do treinamento
- Usar quantização do encoder de texto
- Processamento em lote com `write_batch_size` maior

### Instabilidade no treinamento

- Reduza a taxa de aprendizado (tente 5e-5)
- Aumente passos de acumulação de gradiente
- Ative gradient checkpointing
- Use `--max_grad_norm=1.0`

### CUDA sem memória

- Ative quantização (`int8-quanto` ou `int4-quanto`)
- Ative gradient checkpointing
- Reduza o tamanho do lote
- Ative group offload
- Use TREAD para eficiência de roteamento de tokens

## Avançado: configuração do TREAD

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion) acelera o treinamento processando tokens de forma seletiva:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": -4
      }
    ]
  }
}
```
</details>

- `selection_ratio`: fração de tokens a manter (0.5 = 50%)
- `start_layer_idx`: primeira camada para aplicar roteamento
- `end_layer_idx`: última camada (negativo = a partir do fim)

Aceleração esperada: 20-40% dependendo da configuração.

## Veja também

- [Guia de Início Rápido do FLUX.1](FLUX.md) - Para treinar FLUX.1
- [Documentação do TREAD](../TREAD.md) - Configuração detalhada de TREAD
- [Guia de treinamento LyCORIS](../LYCORIS.md) - Métodos de treino LoRA e LyCORIS
- [Configuração do dataloader](../DATALOADER.md) - Setup de dataset
