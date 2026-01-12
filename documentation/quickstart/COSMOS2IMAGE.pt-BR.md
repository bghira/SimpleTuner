## Guia de Início Rápido do Cosmos2 Predict (Image)

Neste exemplo, vamos treinar um Lycoris LoKr para o Cosmos2 Predict (Image), um modelo de flow matching da NVIDIA.

### Requisitos de hardware

Cosmos2 Predict (Image) é um modelo baseado em vision transformer que usa flow matching.

**Nota**: Devido à arquitetura, ele realmente não deve ser quantizado durante o treino, o que significa que você precisa de VRAM suficiente para acomodar a precisão bf16 completa.

Uma GPU de 24GB é recomendada como mínimo para um treino confortável sem otimizações extensas.

### Offload de memória (opcional)

Para caber o Cosmos2 em GPUs menores, habilite o offload em grupo:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams são respeitados apenas em CUDA; outros dispositivos fazem fallback automaticamente.
- Não combine isso com `--enable_model_cpu_offload`.
- Staging em disco é opcional e ajuda quando a RAM do sistema é o gargalo.

### Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem o Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.12 python3.12-venv
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

- `model_type` - Defina como `lora`.
- `lora_type` - Defina como `lycoris`.
- `model_family` - Defina como `cosmos2image`.
- `base_model_precision` - **Importante**: Defina como `no_change` - Cosmos2 não deve ser quantizado.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - Comece com 1 e aumente se você tiver VRAM suficiente.
- `validation_resolution` - O padrão é `1024x1024` para o Cosmos2.
  - Outras resoluções podem ser especificadas usando vírgulas: `1024x1024,768x768`
- `validation_guidance` - Use um valor em torno de 4.0 para o Cosmos2.
- `validation_num_inference_steps` - Use em torno de 20 steps.
- `use_ema` - Definir como `true` ajuda bastante a obter um resultado mais suavizado junto do seu checkpoint principal treinado.
- `optimizer` - O exemplo usa `adamw_bf16`.
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treino mais eficiente.
- `gradient_checkpointing` - Habilite para reduzir uso de VRAM ao custo de velocidade de treino.

Seu config.json vai parecer com algo assim:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "base_model_precision": "no_change",
    "checkpoint_step_interval": 500,
    "data_backend_config": "config/cosmos2image/multidatabackend.json",
    "disable_bucket_pruning": true,
    "flow_schedule_shift": 0.0,
    "flow_schedule_auto_shift": true,
    "gradient_checkpointing": true,
    "hub_model_id": "cosmos2image-lora",
    "learning_rate": 6e-5,
    "lora_type": "lycoris",
    "lycoris_config": "config/cosmos2image/lycoris_config.json",
    "lr_scheduler": "constant",
    "lr_warmup_steps": 100,
    "max_train_steps": 10000,
    "model_family": "cosmos2image",
    "model_type": "lora",
    "num_train_epochs": 0,
    "optimizer": "adamw_bf16",
    "output_dir": "output/cosmos2image",
    "push_checkpoints_to_hub": false,
    "push_to_hub": false,
    "quantize_via": "cpu",
    "report_to": "tensorboard",
    "seed": 42,
    "tracker_project_name": "cosmos2image-training",
    "tracker_run_name": "cosmos2image-lora",
    "train_batch_size": 1,
    "use_ema": true,
    "vae_batch_size": 1,
    "validation_disable_unconditional": true,
    "validation_guidance": 4.0,
    "validation_guidance_rescale": 0.0,
    "validation_negative_prompt": "ugly, cropped, blurry, low-quality, mediocre average",
    "validation_num_inference_steps": 20,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_prompt_library": false,
    "validation_resolution": "512x512",
    "validation_seed": 42,
    "validation_step_interval": 500
}
```
</details>

> ℹ️ Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

E um arquivo `config/cosmos2image/lycoris_config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "bypass_mode": true,
    "algo": "lokr",
    "multiplier": 1.0,
    "full_matrix": true,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 4,
    "apply_preset": {
        "target_module": [
            "Attention"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 4
            }
        }
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

Para apontar o trainer para essa biblioteca de prompts, adicione-a ao seu config definindo:
```json
"validation_prompt_library": "config/user_prompt_library.json"
```

Um conjunto de prompts diversos ajudará a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu sujeito (instance_prompt).

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

#### Prévias de validação

SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```

**Requisitos:**
- Configuração de webhook
- Validação habilitada

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Ajuste do schedule de flow

Como modelo de flow-matching, o Cosmos2 tem uma propriedade chamada "shift" que permite deslocar a parte treinada do schedule de timesteps usando um valor decimal simples.

A configuração inclui `flow_schedule_auto_shift` habilitado por padrão, que usa shift de timesteps dependente de resolução - valores mais altos para imagens maiores e menores para imagens menores.

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o suficiente para treinar de forma eficaz. Note que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps` além de ser maior que `vae_batch_size`. O dataset não será utilizável se for pequeno demais.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

Dependendo do dataset que você tem, será necessário configurar seu diretório de dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/cosmos2image/multidatabackend.json`) contendo isto:

```json
[
  {
    "id": "pseudo-camera-10k-cosmos2",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/cosmos2/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/cosmos2",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

> ℹ️ Use `caption_strategy=textfile` se você tiver arquivos `.txt` contendo legendas.
> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Isso vai baixar cerca de 10k amostras de fotografias para o diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

Suas imagens de Dreambooth devem ir para o diretório `datasets/dreambooth-subject`.

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

</details>

### Executando o treinamento

A partir do diretório do SimpleTuner, você tem várias opções para iniciar o treinamento:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'
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

### Rodando inferência no LoKr depois

Como o Cosmos2 é um modelo mais novo com documentação limitada, exemplos de inferência podem precisar de ajustes. Uma estrutura básica seria:

<details>
<summary>Mostrar exemplo de inferência em Python</summary>

```py
import torch
from lycoris import create_lycoris_from_weights

# Model and adapter paths
model_id = 'nvidia/Cosmos-1.0-Predict-Image-Text2World-12B'
adapter_repo_id = 'your-username/your-cosmos2-lora'
adapter_filename = 'pytorch_lora_weights.safetensors'

def download_adapter(repo_id: str):
    import os
    from huggingface_hub import hf_hub_download
    adapter_filename = "pytorch_lora_weights.safetensors"
    cache_dir = os.environ.get('HF_PATH', os.path.expanduser('~/.cache/huggingface/hub/models'))
    cleaned_adapter_path = repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")
    path_to_adapter = os.path.join(cache_dir, cleaned_adapter_path)
    path_to_adapter_file = os.path.join(path_to_adapter, adapter_filename)
    os.makedirs(path_to_adapter, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id, filename=adapter_filename, local_dir=path_to_adapter
    )

    return path_to_adapter_file

# Load the model and adapter

import torch
from diffusers import Cosmos2TextToImagePipeline

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Text2Image, nvidia/Cosmos-Predict2-14B-Text2Image
model_id = "nvidia/Cosmos-Predict2-2B-Text2Image"
adapter_repo_id = "youruser/your-repo-name"

adapter_file_path = download_adapter(repo_id=adapter_repo_id)
pipe = Cosmos2TextToImagePipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)

lora_scale = 1.0
wrapper, _ = create_lycoris_from_weights(lora_scale, adapter_file_path, pipeline.transformer)
wrapper.merge_to()

pipe.to("cuda")

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."

output = pipe(
    prompt=prompt, negative_prompt=negative_prompt, generator=torch.Generator().manual_seed(1)
).images[0]
output.save("output.png")

```

</details>

## Notas e dicas de troubleshooting

### Considerações de memória

Como o Cosmos2 não pode ser quantizado durante o treino, o uso de memória será maior do que em modelos quantizados. Configurações-chave para menor uso de VRAM:

- Habilite `gradient_checkpointing`
- Use batch size 1
- Considere usar o otimizador `adamw_8bit` se a memória estiver apertada
- Definir a variável de ambiente `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ajuda a minimizar uso de VRAM ao treinar múltiplas proporções
- Use `--vae_cache_disable` para fazer encoding do VAE online sem cachear em disco, o que economiza espaço mas aumenta tempo/pressão de memória.

### Considerações de treinamento

Como o Cosmos2 é um modelo mais novo, os parâmetros ideais de treino ainda estão sendo explorados:

- O exemplo usa learning rate `6e-5` com AdamW
- Flow schedule auto-shift está habilitado para lidar com diferentes resoluções
- Avaliação CLIP é usada para monitorar o progresso do treino

### Bucketização de aspecto

A configuração tem `disable_bucket_pruning` como true, o que pode ser ajustado conforme seu dataset.

### Treino com múltiplas resoluções

O modelo pode ser treinado em 512px inicialmente, com potencial para treinar em resoluções maiores mais tarde. A configuração `flow_schedule_auto_shift` ajuda com treino multi-resolução.

### Perda com máscara

Se você está treinando um sujeito ou estilo e gostaria de mascarar um ou outro, veja a seção [treino com loss mascarada](../DREAMBOOTH.md#masked-loss) do guia de Dreambooth.

### Limitações conhecidas

- O tratamento de system prompt ainda não está implementado
- As características de treinabilidade do modelo ainda estão sendo exploradas
- Quantização não é suportada e deve ser evitada
