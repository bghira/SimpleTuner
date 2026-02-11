## Guia de Início Rápido do Sana Video

Neste exemplo, vamos treinar o modelo Sana Video 2B 480p.

### Requisitos de hardware

Sana Video usa o autoencoder Wan e processa sequências de 81 frames em 480p por padrão. Espere uso de memória semelhante a outros modelos de vídeo; habilite gradient checkpointing cedo e aumente `train_batch_size` só depois de verificar folga de VRAM.

### Offload de memória (opcional)

Se você estiver perto do limite de VRAM, habilite o offload em grupo no seu config:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Usuários CUDA se beneficiam de `--group_offload_use_stream`; outros backends o ignoram automaticamente.
- Evite `--group_offload_to_disk_path` a menos que a RAM do sistema esteja limitada — staging em disco é mais lento, mas mantém a execução estável.
- Desative `--enable_model_cpu_offload` ao usar group offloading.

### Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para habilitar a compilação de extensões CUDA:

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

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não configura seu dataloader. Você ainda precisará fazer isso manualmente depois.

Para executá-lo:

```bash
simpletuner configure
```

> ⚠️ Para usuários localizados em países onde o Hugging Face Hub não é facilmente acessível, você deve adicionar `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc` dependendo de qual `$SHELL` seu sistema usa.

Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Lá, você provavelmente precisará modificar as seguintes variáveis:

- `model_type` - Defina como `full`.
- `model_family` - Defina como `sanavideo`.
- `pretrained_model_name_or_path` - Defina como `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `pretrained_vae_model_name_or_path` - Defina como `Efficient-Large-Model/SANA-Video_2B_480p_diffusers`.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e vídeos de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - comece baixo para treino de vídeo e aumente apenas após confirmar o uso de VRAM.
- `validation_resolution` - Sana Video é um modelo 480p; use `832x480` ou os buckets de aspecto que você pretende validar.
- `validation_num_video_frames` - Defina como `81` para corresponder ao comprimento padrão do sampler.
- `validation_guidance` - Use o que você costuma selecionar na inferência para o Sana Video.
- `validation_num_inference_steps` - Use algo em torno de 50 para qualidade estável.
- `framerate` - Se omitido, Sana Video usa 16 fps por padrão; defina para corresponder ao seu dataset.

- `optimizer` - Você pode usar qualquer otimizador com o qual se sinta confortável e familiarizado, mas usaremos `optimi-adamw` neste exemplo.
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treino mais eficiente, ou `no` (mas vai consumir mais memória e ser mais lento).
- `gradient_checkpointing` - Habilite isso para controlar o uso de VRAM.
- `use_ema` - definir como `true` vai ajudar bastante a obter um resultado mais suavizado junto do seu checkpoint principal treinado.

Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

Ao final, seu config deve se parecer com:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/sanavideo/multidatabackend.json",
  "seed": 42,
  "output_dir": "output/sanavideo",
  "max_train_steps": 400000,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "tracker_project_name": "video-training",
  "tracker_run_name": "sanavideo-2b-480p",
  "report_to": "wandb",
  "model_type": "full",
  "pretrained_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "pretrained_vae_model_name_or_path": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
  "model_family": "sanavideo",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 200,
  "validation_resolution": "832x480",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 6.0,
  "validation_num_inference_steps": 50,
  "validation_num_video_frames": 81,
  "validation_prompt": "A short video of a small, fluffy animal exploring a sunny room with soft window light and gentle camera motion.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "bf16",
  "vae_batch_size": 1,
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "framerate": 16,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### Opcional: regularizador temporal CREPA

Se seus vídeos apresentam flicker ou sujeitos instáveis, habilite CREPA:
- Em **Training → Loss functions**, ative **CREPA**.
- Padrões sugeridos: **Block Index = 10**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Mantenha o encoder padrão (`dinov2_vitg14`, tamanho `518`) a menos que você precise de uma opção menor (`dinov2_vits14` + `224`) para economizar VRAM.
- A primeira execução baixa DINOv2 via torch hub; faça cache ou pré-carregue se estiver offline.
- Só habilite **Drop VAE Encoder** quando treinar puramente a partir de latentes em cache; deixe desativado se ainda estiver codificando pixels.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação principal", que normalmente é o instance_prompt principal que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para rodar durante validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Os nicknames são o nome do arquivo para a validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o trainer para essa biblioteca de prompts, adicione-a ao TRAINER_EXTRA_ARGS adicionando uma nova linha no final de `config.json`:

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversos ajudará a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu sujeito (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style video featuring <token>, capturing her essence with vibrant colors, dynamic motion, and expressive storytelling",
    "chef_<token>": "a high-quality, detailed video of <token> as a sous-chef, immersed in the art of culinary creation with captivating close-ups and engaging sequences",
    "just_<token>": "a lifelike and intimate video portrait of <token>, showcasing her unique personality and charm through nuanced movement and expression",
    "cinematic_<token>": "a cinematic, visually stunning video of <token>, emphasizing her dramatic and captivating presence through fluid camera movements and atmospheric effects",
    "elegant_<token>": "an elegant and timeless video portrait of <token>, exuding grace and sophistication with smooth transitions and refined visuals",
    "adventurous_<token>": "a dynamic and adventurous video featuring <token>, captured in an exciting, action-filled sequence that highlights her energy and spirit",
    "mysterious_<token>": "a mysterious and enigmatic video portrait of <token>, shrouded in shadows and intrigue with a narrative that unfolds in subtle, cinematic layers",
    "vintage_<token>": "a vintage-style video of <token>, evoking the charm and nostalgia of a bygone era through sepia tones and period-inspired visual storytelling",
    "artistic_<token>": "an artistic and abstract video representation of <token>, blending creativity with visual storytelling through experimental techniques and fluid visuals",
    "futuristic_<token>": "a futuristic and cutting-edge video portrayal of <token>, set against a backdrop of advanced technology with sleek, high-tech visuals",
    "woman": "a beautifully crafted video portrait of a woman, highlighting her natural beauty and unique features through elegant motion and storytelling",
    "man": "a powerful and striking video portrait of a man, capturing his strength and character with dynamic sequences and compelling visuals",
    "boy": "a playful and spirited video portrait of a boy, capturing youthful energy and innocence through lively scenes and engaging motion",
    "girl": "a charming and vibrant video portrait of a girl, emphasizing her bright personality and joy with colorful visuals and fluid movement",
    "family": "a heartwarming and cohesive family video, showcasing the bonds and connections between loved ones through intimate moments and shared experiences"
}
```

> ℹ️ Sana Video é um modelo de flow-matching; prompts curtos podem não ter informação suficiente para o modelo fazer um bom trabalho. Use prompts descritivos sempre que possível.

#### Rastreamento de score CLIP

Isso não deve ser habilitado para treino de modelos de vídeo no momento.

</details>

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar a perda de avaliação.

#### Prévias de validação

SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver vídeos de validação sendo gerados passo a passo em tempo real via callbacks de webhook.

Para habilitar:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requisitos:**

- Configuração de webhook
- Validação habilitada

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá frames de prévia nos steps 5, 10, 15 e 20.

#### Schedule de flow-matching

Sana Video usa o schedule canônico de flow-matching do checkpoint. Overrides de shift fornecidos pelo usuário são ignorados; mantenha `flow_schedule_shift` e `flow_schedule_auto_shift` sem definir para este modelo.

#### Treino com modelo quantizado

Opções de precisão (bf16, int8, fp8) estão disponíveis no config; combine-as com seu hardware e volte para maior precisão se encontrar instabilidades.

#### Considerações sobre o dataset

Há poucas limitações no tamanho do dataset além do quanto de compute e tempo serão necessários para processar e treinar.

Você deve garantir que o dataset seja grande o suficiente para treinar seu modelo de forma eficaz, mas não tão grande para o compute disponível.

Note que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps` além de ser maior que `vae_batch_size`. O dataset não será utilizável se for pequeno demais.

> ℹ️ Com poucas amostras, você pode ver a mensagem **no samples detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

Dependendo do dataset que você tem, será necessário configurar seu diretório de dataset e o arquivo de configuração do dataloader de forma diferente.

Neste exemplo, usaremos [video-dataset-disney-organized](https://huggingface.co/datasets/sayakpaul/video-dataset-disney-organized) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sanavideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 81,
        "min_frames": 81,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sanavideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

- Na subseção `video`, temos as seguintes chaves que podemos definir:
  - `num_frames` (opcional, int) é quantos frames de dados vamos treinar.
  - `min_frames` (opcional, int) determina o comprimento mínimo de vídeo considerado para treino.
  - `max_frames` (opcional, int) determina o comprimento máximo de vídeo considerado para treino.
  - `is_i2v` (opcional, bool) determina se o treino i2v será feito em um dataset.
  - `bucket_strategy` (opcional, string) determina como os vídeos são agrupados em buckets:
    - `aspect_ratio` (padrão): agrupa apenas pela proporção espacial (ex.: `1.78`, `0.75`).
    - `resolution_frames`: agrupa por resolução e contagem de frames no formato `WxH@F` (ex.: `832x480@81`). Útil para datasets com resolução/duração mistas.
  - `frame_interval` (opcional, int) ao usar `resolution_frames`, arredonde contagens de frames para este intervalo.

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset sayakpaul/video-dataset-disney-organized --local-dir=disney-black-and-white
popd
```

Isso vai baixar todas as amostras de vídeo da Disney para o diretório `datasets/disney-black-and-white`, que será criado automaticamente.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você pretende enviar itens para um repositório Git LFS manualmente, também deve executar `git config --global credential.helper store`.

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login em ambos os serviços.

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

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

## Notas e dicas de troubleshooting

### Padrões de validação

- Sana Video usa 81 frames e 16 fps por padrão quando configurações de validação não são fornecidas.
- O caminho do autoencoder Wan deve corresponder ao caminho do modelo base; mantenha-os alinhados para evitar erros de carregamento.

### Perda com máscara

Se você está treinando um sujeito ou estilo e gostaria de mascarar um ou outro, veja a seção [treino com loss mascarada](../DREAMBOOTH.md#masked-loss) do guia de Dreambooth.
