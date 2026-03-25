## Guia de Início Rápido do Qwen Image

> 🆕 Procurando os checkpoints de edição? Veja o [guia de Início Rápido do Qwen Image Edit](./QWEN_EDIT.md) para instruções de treino com referência pareada.

Neste exemplo, vamos treinar um LoRA para o Qwen Image, um modelo visão-linguagem de 20B parâmetros. Devido ao tamanho, precisaremos de técnicas agressivas de otimização de memória.

Uma GPU de 24GB é o mínimo absoluto e, mesmo assim, você precisará de quantização extensa e configuração cuidadosa. 40GB+ é fortemente recomendado para uma experiência mais tranquila.

Ao treinar em 24G, as validações vão dar OOM a menos que você use resolução menor ou quantização agressiva além de int8.

### Requisitos de hardware

Qwen Image é um modelo de 20B parâmetros com um encoder de texto sofisticado que, sozinho, consome ~16GB de VRAM antes de quantização. O modelo usa um VAE customizado com 16 canais latentes.

**Limitações importantes:**
- **Não suportado em AMD ROCm ou MacOS** devido à falta de flash attention eficiente
- Batch size > 1 não funciona corretamente no momento; use gradient accumulation em vez disso
- TREAD (Text-Representation Enhanced Adversarial Diffusion) ainda não é suportado

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

- `model_type` - Defina como `lora`.
- `lora_type` - Defina como `standard` para PEFT LoRA ou `lycoris` para LoKr.
- `model_family` - Defina como `qwen_image`.
- `model_flavour` - Defina como `v1.0`.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - Ajuste de acordo com a VRAM disponível. Os overrides atuais de Qwen no SimpleTuner suportam batch sizes maiores que 1.
- `gradient_accumulation_steps` - Configure em 2-8 se quiser um batch efetivo maior sem aumentar a VRAM por passo.
- `validation_resolution` - Defina `1024x1024` ou menos por restrições de memória.
  - 24G não aguenta validações 1024x1024 atualmente - você precisará reduzir o tamanho
  - Outras resoluções podem ser especificadas usando vírgulas: `1024x1024,768x768,512x512`
- `validation_guidance` - Use um valor em torno de 3.0-4.0 para bons resultados.
- `validation_num_inference_steps` - Use algo em torno de 30.
- `use_ema` - Definir como `true` ajuda a obter resultados mais suaves, mas usa mais memória.

- `optimizer` - Use `optimi-lion` para bons resultados, ou `adamw-bf16` se tiver memória de sobra.
- `mixed_precision` - Deve ser definido como `bf16` para o Qwen Image.
- `gradient_checkpointing` - **Obrigatório** habilitar (`true`) para uso de memória aceitável.
- `base_model_precision` - **Fortemente recomendado** definir `int8-quanto` ou `nf4-bnb` para placas de 24GB.
- `quantize_via` - Defina como `cpu` para evitar OOM durante a quantização em GPUs menores.
- `quantize_activations` - Mantenha como `false` para preservar qualidade de treino.

Configurações de otimização de memória para GPUs 24GB:
- `lora_rank` - Use 8 ou menos.
- `lora_alpha` - Igual ao valor de lora_rank.
- `flow_schedule_shift` - Defina como 1.73 (ou experimente entre 1.0-3.0).

Seu config.json vai ficar mais ou menos assim para um setup mínimo:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_type": "lora",
    "model_family": "qwen_image",
    "model_flavour": "v1.0",
    "lora_type": "standard",
    "lora_rank": 8,
    "lora_alpha": 8,
    "output_dir": "output/models-qwen_image",
    "train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "validation_resolution": "1024x1024",
    "validation_guidance": 4.0,
    "validation_num_inference_steps": 30,
    "validation_seed": 42,
    "validation_prompt": "A photo-realistic image of a cat",
    "validation_step_interval": 100,
    "vae_batch_size": 1,
    "seed": 42,
    "resume_from_checkpoint": "latest",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "report_to": "tensorboard",
    "optimizer": "optimi-lion",
    "num_train_epochs": 0,
    "num_eval_images": 1,
    "mixed_precision": "bf16",
    "minimum_image_size": 0,
    "max_train_steps": 1000,
    "max_grad_norm": 0.01,
    "lr_warmup_steps": 100,
    "lr_scheduler": "constant_with_warmup",
    "learning_rate": "1e-4",
    "gradient_checkpointing": "true",
    "base_model_precision": "int2-quanto",
    "quantize_via": "cpu",
    "quantize_activations": false,
    "flow_schedule_shift": 1.73,
    "disable_benchmark": false,
    "data_backend_config": "config/qwen_image/multidatabackend.json",
    "checkpoints_total_limit": 5,
    "checkpoint_step_interval": 500,
    "caption_dropout_probability": 0.0,
    "aspect_bucket_rounding": 2
}
```
</details>

> ℹ️ Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

> ⚠️ **Crítico para GPUs 24GB**: O encoder de texto sozinho usa ~16GB de VRAM. Com quantização `int2-quanto` ou `nf4-bnb`, isso pode ser reduzido significativamente.

Para um sanity check rápido com uma configuração conhecida:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train example=qwen_image.peft-lora
```

**Opção 2 (Método Git clone):**
```bash
simpletuner train env=examples/qwen_image.peft-lora
```

**Opção 3 (Método legado - ainda funciona):**
```bash
ENV=examples/qwen_image.peft-lora ./train.sh
```

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

Para apontar o trainer para essa biblioteca de prompts, adicione ao seu config.json:
```json
  "validation_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversos ajudará a determinar se o modelo está aprendendo corretamente:

```json
{
    "anime_style": "a breathtaking anime-style portrait with vibrant colors and expressive features",
    "chef_cooking": "a high-quality, detailed photograph of a sous-chef immersed in culinary creation",
    "portrait": "a lifelike and intimate portrait showcasing unique personality and charm",
    "cinematic": "a cinematic, visually stunning photo with dramatic and captivating presence",
    "elegant": "an elegant and timeless portrait exuding grace and sophistication",
    "adventurous": "a dynamic and adventurous photo captured in an exciting moment",
    "mysterious": "a mysterious and enigmatic portrait shrouded in shadows and intrigue",
    "vintage": "a vintage-style portrait evoking the charm and nostalgia of a bygone era",
    "artistic": "an artistic and abstract representation blending creativity with visual storytelling",
    "futuristic": "a futuristic and cutting-edge portrayal set against advanced technology"
}
```

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

#### Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar avaliação de loss.

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

#### Ajuste de schedule de flow

Qwen Image, como modelo de flow-matching, suporta shift do schedule de timesteps para controlar quais partes do processo de geração são treinadas.

O parâmetro `flow_schedule_shift` controla isso:
- Valores baixos (0.1-1.0): foco em detalhes finos
- Valores médios (1.0-3.0): treino equilibrado (recomendado)
- Valores altos (3.0-6.0): foco em grandes aspectos composicionais

##### Auto-shift
Você pode habilitar o shift dependente de resolução com `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso pode dar resultados estáveis, mas potencialmente medianos.

##### Especificação manual
Um valor de `--flow_schedule_shift` de 1.73 é recomendado como ponto de partida para Qwen Image, embora você possa precisar experimentar baseado no seu dataset e objetivos.

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o suficiente para treinar de forma eficaz.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

> ⚠️ **Importante**: Devido às limitações atuais, mantenha `train_batch_size` em 1 e use `gradient_accumulation_steps` para simular batch sizes maiores.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

```json
[
  {
    "id": "pseudo-camera-10k-qwen",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/pseudo-camera-10k",
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
    "minimum_image_size": 512,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/qwen_image/dreambooth-subject",
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
    "cache_dir": "cache/text/qwen_image",
    "disabled": false,
    "write_batch_size": 16
  }
]
```

> ℹ️ Use `caption_strategy=textfile` se você tiver arquivos `.txt` contendo legendas.
> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).
> ℹ️ Observe o `write_batch_size` reduzido para text embeds para evitar OOM.

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

A partir do diretório do SimpleTuner, basta executar:

```bash
./train.sh
```

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

### Dicas de otimização de memória

#### Configuração de menor VRAM (mínimo 24GB)

A configuração de menor VRAM do Qwen Image exige cerca de 24GB:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (mínimo 24GB)
- Memória do sistema: 64GB+ recomendado
- Precisão do modelo base:
  - Para sistemas NVIDIA: `int2-quanto` ou `nf4-bnb` (obrigatório para placas de 24GB)
  - `int4-quanto` pode funcionar, mas pode ter qualidade menor
- Otimizador: `optimi-lion` ou `bnb-lion8bit-paged` para eficiência de memória
- Resolução: comece com 512px ou 768px, suba para 1024px se a memória permitir
- Batch size: 1 (obrigatório devido às limitações atuais)
- Gradient accumulation steps: 2-8 para simular batches maiores
- Habilite `--gradient_checkpointing` (obrigatório)
- Use `--quantize_via=cpu` para evitar OOM na inicialização
- Use um rank LoRA pequeno (1-8)
- Definir a variável de ambiente `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` ajuda a minimizar uso de VRAM

**NOTA**: O pré-cache de embeddings do VAE e saídas do encoder de texto vai usar memória significativa. Habilite `offload_during_startup=true` se você encontrar OOM.

### Rodando inferência no LoRA depois

Como o Qwen Image é um modelo mais novo, aqui está um exemplo funcional de inferência:

<details>
<summary>Mostrar exemplo de inferência em Python</summary>

```python
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration

model_id = 'Qwen/Qwen-Image'
adapter_id = 'your-username/your-lora-name'

# Load the pipeline
pipeline = QwenImagePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Load LoRA weights
pipeline.load_lora_weights(adapter_id)

# Optional: quantize the model to save VRAM
from optimum.quanto import quantize, freeze, qint8
quantize(pipeline.transformer, weights=qint8)
freeze(pipeline.transformer)

# Move to device
pipeline.to('cuda' if torch.cuda.is_available() else 'cpu')

# Generate an image
prompt = "Your test prompt here"
negative_prompt = 'ugly, cropped, blurry, low-quality, mediocre average'

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=4.0,
    generator=torch.Generator(device='cuda').manual_seed(42),
    width=1024,
    height=1024,
).images[0]

image.save("output.png", format="PNG")
```
</details>

### Notas e dicas de troubleshooting

#### Limitações de batch size

Builds antigos do Qwen no diffusers tinham problemas com batch size > 1 por causa do padding dos embeddings de texto e do mascaramento de atenção. Os overrides atuais de Qwen no SimpleTuner corrigem os dois pontos, então batches maiores funcionam se a sua VRAM permitir.
- Aumente `train_batch_size` somente depois de confirmar a folga de memória.
- Se ainda aparecerem artefatos em uma instalação antiga, atualize e regenere quaisquer text embeds antigos.

#### Quantização

- `int2-quanto` oferece a economia de memória mais agressiva, mas pode impactar a qualidade
- `nf4-bnb` oferece bom equilíbrio entre memória e qualidade
- `int4-quanto` é uma opção intermediária
- Evite `int8` a menos que você tenha 40GB+ de VRAM

#### Taxas de aprendizado

Para treino de LoRA:
- LoRAs pequenas (rank 1-8): use learning rates em torno de 1e-4
- LoRAs maiores (rank 16-32): use learning rates em torno de 5e-5
- Com otimizador Prodigy: comece em 1.0 e deixe adaptar

#### Artefatos de imagem

Se você encontrar artefatos:
- Reduza sua taxa de aprendizado
- Aumente gradient accumulation steps
- Garanta que suas imagens são de alta qualidade e bem pré-processadas
- Considere usar resoluções menores inicialmente

#### Treino com múltiplas resoluções

Comece o treino em resoluções menores (512px ou 768px) para acelerar o aprendizado inicial, depois faça fine-tune em 1024px. Habilite `--flow_schedule_auto_shift` ao treinar em diferentes resoluções.

### Limitações de plataforma

**Não suportado em:**
- AMD ROCm (falta implementação eficiente de flash attention)
- Apple Silicon/MacOS (limitações de memória e atenção)
- GPUs de consumidor com menos de 24GB de VRAM

### Problemas conhecidos atuais

1. Batch size > 1 não funciona corretamente (use gradient accumulation)
2. TREAD ainda não é suportado
3. Alto uso de memória do encoder de texto (~16GB antes da quantização)
4. Problemas de manuseio de comprimento de sequência ([issue upstream](https://github.com/huggingface/diffusers/issues/12075))

Para ajuda adicional e troubleshooting, consulte a [documentação do SimpleTuner](/documentation) ou entre no Discord da comunidade.
