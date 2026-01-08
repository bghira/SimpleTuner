# Guia de Início Rápido do Kandinsky 5.0 Image

Neste exemplo, vamos treinar um LoRA do Kandinsky 5.0 Image.

## Requisitos de hardware

Kandinsky 5.0 usa um **enorme encoder de texto Qwen2.5-VL de 7B parâmetros** além de um encoder CLIP padrão e o VAE do Flux. Isso coloca uma demanda significativa tanto em VRAM quanto em RAM do sistema.

Apenas carregar o encoder Qwen exige cerca de **14GB** de memória por si só. Ao treinar um LoRA rank 16 com gradient checkpointing completo:

- **24GB de VRAM** é o mínimo confortável (RTX 3090/4090).
- **16GB de VRAM** é possível, mas exige offload agressivo e provavelmente quantização `int8` do modelo base.

Você vai precisar:

- **RAM do sistema**: Pelo menos 32GB, idealmente 64GB, para lidar com o carregamento inicial do modelo sem travar.
- **GPU**: NVIDIA RTX 3090 / 4090 ou placas profissionais (A6000, A100, etc.).

### Offload de memória (recomendado)

Dado o tamanho do encoder de texto, você provavelmente deve usar offload em grupo em hardware de consumidor. Isso offloada blocos do transformer para a memória da CPU quando não estão sendo computados.

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

Além disso, defina `"offload_during_startup": true` no seu `config.json` para reduzir uso de VRAM durante a inicialização e cache. Isso garante que o encoder de texto e o VAE não sejam carregados ao mesmo tempo.

## Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.12 python3.12-venv
```

## Instalação

Instale o SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

## Configurando o ambiente

### Método da interface web

A WebUI do SimpleTuner torna a configuração bastante direta. Para rodar o servidor:

```bash
simpletuner server
```

Acesse em http://localhost:8001.

### Método manual / linha de comando

Para rodar o SimpleTuner via linha de comando, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode ajudar você a pular esta seção:

```bash
simpletuner configure
```

Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Você precisará modificar as seguintes variáveis:

- `model_type`: `lora`
- `model_family`: `kandinsky5-image`
- `model_flavour`:
  - `t2i-lite-sft`: (Padrão) Checkpoint SFT padrão. Melhor para fine-tuning de estilos/personagens.
  - `t2i-lite-pretrain`: Checkpoint de pré-treino. Melhor para ensinar conceitos totalmente novos do zero.
  - `i2i-lite-sft` / `i2i-lite-pretrain`: Para treino imagem-para-imagem. Requer imagens de condicionamento no seu dataset.
- `output_dir`: onde salvar seus checkpoints.
- `train_batch_size`: comece com `1`.
- `gradient_accumulation_steps`: use `1` ou maior para simular batches maiores.
- `validation_resolution`: `1024x1024` é o padrão para este modelo.
- `validation_guidance`: `5.0` é o padrão recomendado para Kandinsky 5.
- `flow_schedule_shift`: `1.0` é o padrão. Ajustar isso muda como o modelo prioriza detalhes vs composição (veja abaixo).

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação principal". Você também pode criar uma biblioteca de prompts em `config/user_prompt_library.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "portrait": "A high quality portrait of a woman, cinematic lighting, 8k",
  "landscape": "A beautiful mountain landscape at sunset, oil painting style"
}
```
</details>

Habilite adicionando isto ao `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>

#### Ajuste de schedule de flow

Kandinsky 5 é um modelo de flow-matching. O parâmetro `shift` controla a distribuição de ruído durante treino e inferência.

- **Shift 1.0 (Padrão)**: Treino equilibrado.
- **Shift baixo (< 1.0)**: Foca mais em detalhes de alta frequência (textura, ruído).
- **Shift alto (> 1.0)**: Foca mais em detalhes de baixa frequência (composição, cor, estrutura).

Se o modelo aprende estilos bem, mas falha na composição, tente aumentar o shift. Se aprende composição, mas falta textura, tente diminuir.

#### Treino com modelo quantizado

Você pode reduzir significativamente a VRAM quantizando o transformer para 8 bits.

No `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "base_model_default_dtype": "bf16"
```
</details>

> **Nota**: Não recomendamos quantizar os encoders de texto (`no_change`), pois o Qwen2.5-VL é sensível a quantização e já é a parte mais pesada do pipeline.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Considerações sobre o dataset

Você precisará de um arquivo de configuração de dataset, por exemplo, `config/multidatabackend.json`.

```json
[
  {
    "id": "my-image-dataset",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "datasets/my_images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "crop": true,
    "crop_aspect": "square",
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

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Depois crie o diretório do dataset:

```bash
mkdir -p datasets/my_images
</details>

# Copy your images and .txt caption files here
```

#### Login no WandB e Huggingface Hub

```bash
wandb login
huggingface-cli login
```

### Executando o treinamento

**Opção 1 (Recomendado):**

```bash
simpletuner train
```

**Opção 2 (Legado):**

```bash
./train.sh
```

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

Para rodar em setups com 16GB ou 24GB limitados:

1.  **Habilite Group Offload**: `--enable_group_offload`.
2.  **Quantize o modelo base**: defina `"base_model_precision": "int8-quanto"`.
3.  **Batch size**: mantenha em `1`.

### Artefatos e imagens "queimadas"

Se as imagens de validação parecerem saturadas ou ruidosas ("queimadas"):

- **Cheque guidance**: garanta `validation_guidance` em torno de `5.0`. Valores mais altos (7.0+) costumam fritar a imagem neste modelo.
- **Cheque flow shift**: valores extremos de `flow_schedule_shift` podem causar instabilidade. Comece em `1.0`.
- **Taxa de aprendizado**: 1e-4 é padrão para LoRA, mas se você vir artefatos, tente baixar para 5e-5.

### Treino TREAD

Kandinsky 5 suporta [TREAD](../TREAD.md) para treinar mais rápido ao descartar tokens.

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

Isso descarta 50% dos tokens nas camadas do meio, acelerando o passe do transformer.
