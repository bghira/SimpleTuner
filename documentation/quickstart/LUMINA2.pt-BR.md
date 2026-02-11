## Guia de Início Rápido do Lumina2

Neste exemplo, vamos treinar um LoRA do Lumina2 ou fazer fine-tuning do modelo completo.

### Requisitos de hardware

O Lumina2 é um modelo de 2B parâmetros, tornando-o muito mais acessível do que modelos maiores como Flux ou SD3. O tamanho menor do modelo significa:

Ao treinar um LoRA rank-16, ele usa:
- Aproximadamente 12-14GB de VRAM para treinamento LoRA
- Aproximadamente 16-20GB de VRAM para fine-tuning completo
- Cerca de 20-30GB de RAM do sistema durante a inicialização

Você vai precisar:
- **Mínimo**: Uma única RTX 3060 12GB ou RTX 4060 Ti 16GB
- **Recomendado**: RTX 3090, RTX 4090 ou A100 para treinamento mais rápido
- **RAM do sistema**: Pelo menos 32GB recomendados

### Pré-requisitos

Certifique-se de que você tenha Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8:

```bash
apt -y install nvidia-cuda-toolkit
```

### Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Configurando o ambiente

Para rodar o SimpleTuner, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Lá, você precisará modificar as seguintes variáveis:

- `model_type` - Defina como `lora` para treinamento LoRA ou `full` para fine-tuning completo.
- `model_family` - Defina como `lumina2`.
- `output_dir` - Defina o diretório onde deseja armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - Pode ser 1-4 dependendo da memória da sua GPU e do tamanho do dataset.
- `validation_resolution` - O Lumina2 suporta múltiplas resoluções. Opções comuns: `1024x1024`, `512x512`, `768x768`.
- `validation_guidance` - O Lumina2 usa classifier-free guidance. Valores de 3.5-7.0 funcionam bem.
- `validation_num_inference_steps` - 20-30 steps funcionam bem para o Lumina2.
- `gradient_accumulation_steps` - Pode ser usado para simular lotes maiores. Um valor de 2-4 funciona bem.
- `optimizer` - `adamw_bf16` é recomendado. `lion` e `optimi-stableadamw` também funcionam bem.
- `mixed_precision` - Mantenha em `bf16` para melhores resultados.
- `gradient_checkpointing` - Defina como `true` para economizar VRAM.
- `learning_rate` - Para LoRA: `1e-4` a `5e-5`. Para fine-tuning completo: `1e-5` a `1e-6`.

#### Exemplo de configuração do Lumina2

Isso vai no `config.json`

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
    "base_model_precision": "int8-torchao",
    "checkpoint_step_interval": 50,
    "data_backend_config": "config/lumina2/multidatabackend.json",
    "disable_bucket_pruning": true,
    "eval_steps_interval": 50,
    "evaluation_type": "clip",
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "lumina2-lora",
    "learning_rate": 1e-4,
    "lora_alpha": 16,
    "lora_rank": 16,
    "lora_type": "standard",
    "lr_scheduler": "constant",
    "max_train_steps": 400000,
    "model_family": "lumina2",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/lumina2",
    "push_checkpoints_to_hub": true,
    "push_to_hub": true,
    "quantize_via": "cpu",
    "report_to": "wandb",
    "seed": 42,
    "tracker_project_name": "lumina2-training",
    "tracker_run_name": "lumina2-lora",
    "train_batch_size": 4,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 40,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "1024x1024",
    "validation_seed": 42,
    "validation_step_interval": 50
}
```
</details>

Para treinamento Lycoris, altere `lora_type` para `lycoris`

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação primário". Além disso, crie um arquivo de biblioteca de prompts:

```json
{
  "portrait": "a high-quality portrait photograph with natural lighting",
  "landscape": "a breathtaking landscape photograph with dramatic lighting",
  "artistic": "an artistic rendering with vibrant colors and creative composition",
  "detailed": "a highly detailed image with sharp focus and rich textures",
  "stylized": "a stylized illustration with unique artistic flair"
}
```

Adicione ao seu config:
```json
{
  "--user_prompt_library": "config/user_prompt_library.json"
}
```

#### Considerações sobre o dataset

O Lumina2 se beneficia de dados de treinamento de alta qualidade. Crie um `--data_backend_config` (`config/multidatabackend.json`):

> 💡 **Dica:** Para datasets grandes em que espaço em disco é uma preocupação, você pode usar `--vae_cache_disable` para realizar codificação VAE online sem armazenar os resultados no disco.

```json
[
  {
    "id": "lumina2-training",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 2048,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/lumina2/training",
    "instance_data_dir": "/datasets/training",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/lumina2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Crie o diretório do seu dataset. Não se esqueça de atualizar esse caminho para o local real.

```bash
mkdir -p /datasets/training
</details>

# Coloque suas imagens e arquivos de caption em /datasets/training/
```

Arquivos de caption devem ter o mesmo nome da imagem com a extensão `.txt`.

#### Login no WandB

O SimpleTuner tem suporte **opcional** a trackers, com foco principal no Weights & Biases. Você pode desativar com `report_to=none`.

Para habilitar o wandb, execute os seguintes comandos:

```bash
wandb login
```

#### Login no Huggingface Hub

Para enviar checkpoints ao Huggingface Hub, garanta que:
```bash
huggingface-cli login
```

### Executando o treinamento

A partir do diretório do SimpleTuner, você tem várias opções para iniciar o treinamento:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**Opção 2 (Método Git clone):**
```bash
simpletuner train
```

**Opção 3 (Método legado - ainda funciona):**
```bash
./train.sh
```

Isso vai iniciar o cache de text embeds e saídas VAE em disco.

## Dicas de treinamento para Lumina2

### Taxas de aprendizado

#### Treinamento LoRA
- Comece com `1e-4` e ajuste com base nos resultados
- O Lumina2 treina rápido, então monitore as primeiras iterações de perto
- Ranks 8-32 funcionam bem para a maioria dos casos, 64-128 podem exigir monitoramento mais próximo, e 256-512 podem ser úteis para treinar novas tarefas no modelo

#### Fine-tuning completo
- Use taxas de aprendizado menores: `1e-5` a `5e-6`
- Considere usar EMA (Exponential Moving Average) para estabilidade
- É recomendado clipping de gradiente (`max_grad_norm`) de 1.0

### Considerações de resolução

O Lumina2 suporta resoluções flexíveis:
- Treinar em 1024x1024 oferece a melhor qualidade
- Treinamento em resolução mista (512px, 768px, 1024px) ainda não foi testado para impacto de qualidade
- Bucketing de proporção funciona bem com o Lumina2

### Duração do treinamento

Devido ao tamanho eficiente de 2B parâmetros do Lumina2:
- Treinamento LoRA frequentemente converge em 500-2000 steps
- Fine-tuning completo pode precisar de 2000-5000 steps
- Monitore imagens de validação frequentemente, pois o modelo treina rápido

### Problemas comuns e soluções

1. **Modelo convergindo rápido demais**: Diminua a taxa de aprendizado, troque do otimizador Lion para AdamW
2. **Artefatos nas imagens geradas**: Garanta dados de treinamento de alta qualidade e considere reduzir a taxa de aprendizado
3. **Sem memória**: Habilite gradient checkpointing e reduza o tamanho do batch
4. **Overfitting fácil**: Use datasets de regularização

## Dicas de inferência

### Usando seu modelo treinado

Modelos Lumina2 podem ser usados com:
- Biblioteca Diffusers diretamente
- ComfyUI com os nós apropriados
- Outros frameworks de inferência que suportam modelos baseados em Gemma2

### Configurações ideais de inferência

- Guidance scale: 4.0-6.0
- Steps de inferência: 20-50
- Use a mesma resolução em que você treinou para melhores resultados

## Notas

### Vantagens do Lumina2

- Treinamento rápido devido ao tamanho de 2B parâmetros
- Boa relação qualidade/tamanho
- Suporta vários modos de treinamento (LoRA, LyCORIS, full)
- Uso eficiente de memória

### Limitações atuais

- Sem suporte a ControlNet por enquanto
- Limitado a geração texto-para-imagem
- Exige alta qualidade de captions para melhores resultados

### Otimização de memória

Ao contrário de modelos maiores, o Lumina2 normalmente não requer:
- Quantização de modelo
- Técnicas extremas de otimização de memória
- Estratégias complexas de mixed precision
