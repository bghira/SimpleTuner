## Guia de Início Rápido do OmniGen

Neste exemplo, vamos treinar um Lycoris LoKr para o OmniGen, focado em melhorias gerais de T2I (não para edit/instruct neste momento).

### Requisitos de hardware

OmniGen é um modelo de tamanho moderado com cerca de 3.8B parâmetros; usa o VAE do SDXL, mas não usa um encoder de texto. Em vez disso, o OmniGen usa IDs de tokens nativos como entradas e se comporta como um modelo multimodal.

O uso de memória durante o treino ainda não é conhecido, mas espera-se que caiba facilmente em uma placa de 24G com batch size 2 ou 3. O modelo pode ser quantizado, economizando mais VRAM.

OmniGen é uma arquitetura estranha em relação a outros modelos treináveis pelo SimpleTuner;

- Atualmente, apenas treino t2i (texto-para-imagem) é suportado, onde a saída do modelo é alinhada com prompts de treino e imagens de entrada.
- Um modo de treino imagem-para-imagem ainda não é suportado, mas pode vir no futuro.
  - Esse modo permite fornecer uma segunda imagem como entrada, e o modelo usa isso como dados de condicionamento/referência para a saída.
- O valor de loss ao treinar OmniGen é muito alto, e não se sabe por quê.


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
- `model_family` - Defina como `omnigen`.
- `model_flavour` - Defina como `v1`
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - para uma placa de 24G com gradient checkpointing completo, isso pode chegar a 6.
- `validation_resolution` - Este checkpoint do OmniGen é um modelo 1024px, você deve definir `1024x1024` ou outra resolução suportada.
  - Outras resoluções podem ser especificadas usando vírgulas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use o que você costuma selecionar na inferência para o OmniGen; um valor mais baixo em torno de 2.5-3.0 gera resultados mais realistas
- `validation_num_inference_steps` - Use algo em torno de 30
- `use_ema` - definir como `true` vai ajudar bastante a obter um resultado mais suavizado junto do seu checkpoint principal treinado.

- `optimizer` - Você pode usar qualquer otimizador com o qual se sinta confortável e familiarizado, mas usaremos `adamw_bf16` neste exemplo.
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treino mais eficiente, ou `no` (mas vai consumir mais memória e ser mais lento).
- `gradient_checkpointing` - Desativar isso será o mais rápido, mas limita seus tamanhos de lote. É obrigatório habilitar para obter o menor uso de VRAM.

Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

Seu config.json vai ficar mais ou menos assim:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "validation_torch_compile": "false",
    "validation_step_interval": 200,
    "validation_seed": 42,
    "validation_resolution": "1024x1024",
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_num_inference_steps": "20",
    "validation_guidance": 2.0,
    "validation_guidance_rescale": "0.0",
    "vae_cache_ondemand": true,
    "vae_batch_size": 1,
    "train_batch_size": 1,
    "tracker_run_name": "eval_loss_test1",
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "output_dir": "output/models-omnigen",
    "optimizer": "adamw_bf16",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "model_type": "lora",
    "model_family": "omnigen",
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 10000,
    "max_grad_norm": 0.01,
    "lycoris_config": "config/lycoris_config.json",
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant",
    "lora_type": "lycoris",
    "learning_rate": "4e-5",
    "gradient_checkpointing": "true",
    "grad_clip_method": "value",
    "eval_steps_interval": 100,
    "disable_benchmark": false,
    "data_backend_config": "config/omnigen/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "base_model_precision": "no_change",
    "aspect_bucket_rounding": 2
}
```
</details>

E um arquivo simples `config/lycoris_config.json` - note que o `FeedForward` pode ser removido para maior estabilidade de treino.

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "algo": "lokr",
    "multiplier": 1.0,
    "linear_dim": 10000,
    "linear_alpha": 1,
    "factor": 16,
    "apply_preset": {
        "target_module": [
            "Attention",
            "FeedForward"
        ],
        "module_algo_map": {
            "Attention": {
                "factor": 16
            },
            "FeedForward": {
                "factor": 8
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

Dentro de `config/config.json` está o "prompt de validação principal" (`--validation_prompt`), que normalmente é o instance_prompt principal que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para rodar durante validações.

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
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```

> ℹ️ OmniGen parece limitar em torno de 122 tokens de entendimento. Não se sabe se ele entende mais do que isso.

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

</details>

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar avaliação de loss.

#### Prévias de validação

SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

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

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Ajuste de schedule de flow

Atualmente, o OmniGen é hard-coded para usar sua própria formulação especial de flow-matching, e o schedule shift não se aplica a ele.

<!--
Flow-matching models such as OmniGen, Sana, Flux, and SD3 have a property called "shift" that allows us to shift the trained portion of the timestep schedule using a simple decimal value.

##### Auto-shift
A commonly-recommended approach is to follow several recent works and enable resolution-dependent timestep shift, `--flow_schedule_auto_shift` which uses higher shift values for larger images, and lower shift values for smaller images. This results in stable but potentially mediocre training results.

##### Manual specification
_Thanks to General Awareness from Discord for the following examples_

When using a `--flow_schedule_shift` value of 0.1 (a very low value), only the finer details of the image are affected:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

When using a `--flow_schedule_shift` value of 4.0 (a very high value), the large compositional features and potentially colour space of the model becomes impacted:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)
-->

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o suficiente para treinar de forma eficaz. Note que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps` além de ser maior que `vae_batch_size`. O dataset não será utilizável se for pequeno demais.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

Dependendo do dataset que você tem, será necessário configurar seu diretório de dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-omnigen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/omnigen/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/omnigen/dreambooth-subject-512",
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
    "cache_dir": "cache/text/omnigen",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ Rodar datasets 512px e 1024px simultaneamente é suportado e pode resultar em melhor convergência.

> ℹ️ Embeddings de encoder de texto não são geradas pelo OmniGen, mas ainda é necessário definir um (por enquanto).

Minha configuração para OmniGen foi bem básica e ficou assim, pois eu usei o conjunto de treino de stable eval loss:

<details>
<summary>Ver exemplo de config</summary>

```json
[
    {
        "id": "something-special-to-remember-by",
        "type": "local",
        "instance_data_dir": "/datasets/pseudo-camera-10k/train",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "filename",
        "cache_dir_vae": "cache/vae/omnigen",
        "vae_cache_clear_each_epoch": false,
        "crop": true,
        "crop_aspect": "square"
    },
    {
        "id": "omnigen-eval",
        "type": "local",
        "dataset_type": "eval",
        "crop": true,
        "crop_aspect": "square",
        "instance_data_dir": "/datasets/test_datasets/squares",
        "resolution": 1024,
        "minimum_image_size": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/omnigen-eval",
        "caption_strategy": "filename"
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/omnigen"
    }
]
```
</details>


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

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

A configuração de menor VRAM para OmniGen ainda não é conhecida, mas espera-se que seja semelhante ao seguinte:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G de memória do sistema
- Precisão do modelo base: `int8-quanto` (ou `fp8-torchao`, `int8-torchao` seguem perfis de uso de memória similares)
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 1024px
- Tamanho de lote: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- PyTorch: 2.7+
- Usar `--quantize_via=cpu` para evitar erro outOfMemory na inicialização em placas <=16G.
- Habilite `--gradient_checkpointing`
- Use uma configuração LoRA ou Lycoris bem pequena (ex.: LoRA rank 1 ou Lokr factor 25)

**NOTA**: O pré-cache de embeddings do VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. Se isso ocorrer, VAE tiling e slicing podem ser habilitados opcionalmente. Encoders de texto podem ser offloadados para CPU durante o cache do VAE com `offload_during_startup=true`. Para evitar uso de disco para cache de VAE em datasets grandes, use `--vae_cache_disable`.

A velocidade foi aproximadamente 3,4 iterações por segundo em uma AMD 7900XTX usando PyTorch 2.7 e ROCm 6.3.

### Perda com máscara

Se você está treinando um sujeito ou estilo e gostaria de mascarar um ou outro, veja a seção [treino com loss mascarada](../DREAMBOOTH.md#masked-loss) do guia de Dreambooth.

### Quantização

Não testado a fundo (ainda).

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

*Não suportado.*

#### LoKr (--lora_type=lycoris)
- Taxas de aprendizado mais suaves são melhores para LoKr (`1e-4` com AdamW, `2e-5` com Lion)
- Outros algoritmos precisam de mais exploração.
- Definir `is_regularisation_data` tem impacto/efeito desconhecido com OmniGen (não testado)

### Artefatos de imagem

OmniGen tem resposta desconhecida a artefatos de imagem, embora use o VAE do SDXL e tenha limitações idênticas em detalhes finos.

Se surgirem problemas de qualidade de imagem, por favor abra uma issue no GitHub.

### Bucketização de aspecto

Este modelo tem resposta desconhecida a dados com bucketização de aspecto. Experimentação será útil.

### Valores altos de loss

OmniGen tem um valor de loss muito alto, e não se sabe por quê. Recomenda-se ignorar a loss e focar na qualidade visual das imagens geradas.
