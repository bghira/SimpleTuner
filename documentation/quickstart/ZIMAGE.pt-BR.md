# Guia de Início Rápido do Z-Image [base / turbo]

Neste exemplo, vamos treinar um LoRA de Z-Image Turbo. Z-Image é um transformer de flow-matching de 6B (cerca de metade do Flux) com flavours base e turbo. Turbo espera um adapter assistente; o SimpleTuner pode carregá-lo automaticamente.

## Requisitos de hardware

Z-Image usa menos memória que o Flux, mas ainda se beneficia de GPUs fortes. Quando você treina todos os componentes de um LoRA de rank 16 (MLP, projeções, blocos do transformer), normalmente usa:

- ~32-40G de VRAM quando não quantiza o modelo base
- ~16-24G de VRAM ao quantizar para int8 + pesos base/LoRA em bf16
- ~10–12G de VRAM ao quantizar para NF4 + pesos base/LoRA em bf16

Além disso, Ramtorch e group offload podem ser usados para tentar reduzir ainda mais o uso de VRAM. Para usuários multi-GPU, o FSDP2 permite rodar em várias GPUs menores também.

Você vai precisar:

- **mínimo absoluto**: uma única **3080 10G** (com quantização/offload agressivos)
- **mínimo realista**: uma única 3090/4090 ou V100/A6000
- **idealmente**: várias 4090, A6000, L40S ou melhor

GPUs Apple não são recomendadas para treino.

### Offload de memória (opcional)

Offload agrupado de módulos reduz drasticamente a pressão de VRAM quando o gargalo são os pesos do transformer. Você pode habilitar adicionando as seguintes flags ao `TRAINER_EXTRA_ARGS` (ou na página Hardware da WebUI):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Streams só são efetivos em CUDA; o SimpleTuner desativa automaticamente em ROCm, MPS e CPU.
- **Não** combine isso com outras estratégias de CPU offload.
- Group offload não é compatível com quantização Quanto.
- Prefira um SSD/NVMe local rápido ao fazer offload para disco.

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

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.x para habilitar a compilação de extensões CUDA:

```bash
apt -y install nvidia-cuda-toolkit-12-8
```

## Instalação

Instale o SimpleTuner via pip:

```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Etapas adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Configurando o ambiente

### Método da interface web

A WebUI do SimpleTuner torna a configuração direta. Para rodar o servidor:

```bash
simpletuner server
```

Isso vai criar um servidor web na porta 8001 por padrão, que você pode acessar em http://localhost:8001.

### Método manual / linha de comando

Para rodar o SimpleTuner via linha de comando, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

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
- `model_family` - Defina como `z-image`.
- `model_flavour` - defina como `turbo` (ou `turbo-ostris-v2` para o adapter assistente v2); o flavour base aponta para um checkpoint atualmente indisponível.
- `pretrained_model_name_or_path` - Defina como `TONGYI-MAI/Z-Image-Turbo`.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - mantenha em 1, especialmente se você tem um dataset muito pequeno.
- `validation_resolution` - Z-Image é 1024px; use `1024x1024` ou buckets multi-aspect: `1024x1024,1280x768,2048x2048`.
- `validation_guidance` - Guidance baixo (0–1) é típico para o Z-Image Turbo, mas o flavour base exige um intervalo entre 4-6.
- `validation_num_inference_steps` - Turbo precisa de apenas 8, mas Base pode se virar com algo em torno de 50-100.
- `--lora_rank=4` se você quiser reduzir bastante o tamanho do LoRA treinado. Isso ajuda no uso de VRAM.
- Para turbo, forneça o adapter assistente (veja abaixo) ou desative explicitamente.

- `gradient_accumulation_steps` - aumenta o tempo de execução de forma linear; use se precisar reduzir VRAM.
- `optimizer` - Iniciantes devem ficar com adamw_bf16, embora outras variantes adamw/lion também sejam boas escolhas.
- `mixed_precision` - `bf16` em GPUs modernas; `fp16` caso contrário.
- `gradient_checkpointing` - defina como true em praticamente todas as situações e em todos os dispositivos.
- `gradient_checkpointing_interval` - pode ser definido como 2+ em GPUs maiores para checkpointar a cada _n_ blocos.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

### LoRA assistente (Turbo)

Turbo espera um adapter assistente:

- `assistant_lora_path`: `ostris/zimage_turbo_training_adapter`
- `assistant_lora_weight_name`:
  - `turbo`: `zimage_turbo_training_adapter_v1.safetensors`
  - `turbo-ostris-v2`: `zimage_turbo_training_adapter_v2.safetensors`

O SimpleTuner preenche isso automaticamente para flavours turbo, a menos que você sobrescreva. Desative com `--disable_assistant_lora` se aceitar a perda de qualidade.

### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação principal", que normalmente é o instance_prompt principal que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para rodar durante validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Os nicknames são o nome do arquivo para a validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o trainer para essa biblioteca de prompts, adicione-a ao TRAINER_EXTRA_ARGS adicionando uma nova linha no final de `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

Um conjunto de prompts diversos ajudará a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu sujeito (instance_prompt).

<details>
<summary>Ver exemplo de config</summary>

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
</details>

> ℹ️ Z-Image é um modelo de flow-matching e prompts mais curtos com forte similaridade resultarão praticamente na mesma imagem produzida. Use prompts mais longos e descritivos.

### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

### Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar avaliação de loss.

### Prévias de validação

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

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder.

### Ajuste de schedule de flow (flow matching)

Modelos de flow-matching como Z-Image têm um parâmetro "shift" para mover a parte treinada do schedule de timesteps. Auto-shift baseado em resolução é um padrão seguro. Aumentar o shift manualmente move o aprendizado para features mais grosseiras; reduzir favorece detalhes finos. Para o modelo turbo, é possível que modificar esses valores prejudique o modelo.

### Treino com modelo quantizado

TorchAO ou outra quantização pode reduzir precisão e requisitos de VRAM - Optimum Quanto está em manutenção, mas ainda está disponível.

Para usuários de `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
  "base_model_precision": "int8-torchao",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

### Considerações sobre o dataset

> ⚠️ A qualidade das imagens para treino é crítica; Z-Image absorve artefatos cedo. Um passe final com dados de alta qualidade pode ser necessário.

Mantenha seu dataset grande o suficiente (pelo menos `train_batch_size * gradient_accumulation_steps`, e mais que `vae_batch_size`). Aumente `repeats` se você vir **no images detected in dataset**.

Exemplo de configuração multi-backend (`config/multidatabackend.json`):

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-zimage",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/zimage/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/zimage/dreambooth-subject-512",
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
    "cache_dir": "cache/text/zimage",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Rodar datasets 512px e 1024px simultaneamente é suportado e pode melhorar a convergência.

Crie o diretório de datasets:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

### Login no WandB e Huggingface Hub

Faça login antes do treino, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`:

```bash
wandb login
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

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

## Configuração multi-GPU

O SimpleTuner inclui **detecção automática de GPU** pela WebUI. Durante o onboarding, você vai configurar:

- **Modo Automático**: Usa automaticamente todas as GPUs detectadas com configurações ideais
- **Modo Manual**: Seleciona GPUs específicas ou define contagem de processos personalizada
- **Modo Desativado**: Treino com uma única GPU

A WebUI detecta seu hardware e configura `--num_processes` e `CUDA_VISIBLE_DEVICES` automaticamente.

Para configuração manual ou setups avançados, veja a [seção de treinamento multi-GPU](../INSTALL.md#multiple-gpu-training) no guia de instalação.

## Dicas de inferência

### Configurações de guidance

Z-Image é flow-matching; valores de guidance mais baixos (em torno de 0–1) tendem a preservar qualidade e diversidade. Se você treinar com vetores de guidance mais altos, garanta que seu pipeline de inferência suporte CFG e espere geração mais lenta ou maior uso de VRAM com CFG em batch.

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

- GPU: um único dispositivo NVIDIA CUDA (10–12G) com quantização/offload agressivos
- Memória do sistema: ~32–48G
- Precisão do modelo base: `nf4-bnb` ou `int8`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged` ou variantes adamw
- Resolução: 512px (1024px exige mais VRAM)
- Batch size: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- Use `--quantize_via=cpu` se der OOM na inicialização em placas <=16G
- Habilite `--gradient_checkpointing`
- Habilite Ramtorch ou group offload

O estágio de pré-cache pode ficar sem memória; quantização do encoder de texto e VAE tiling podem ser habilitados via `--text_encoder_precision=int8-torchao` e `--vae_enable_tiling=true`. Mais memória pode ser economizada na inicialização com `--offload_during_startup=true`, o que mantém apenas o encoder de texto ou VAE carregado, e não ambos.

### Quantização

- Quantização mínima de 8 bits costuma ser necessária para uma placa de 16G treinar este modelo.
- Quantizar o modelo para 8 bits geralmente não prejudica o treino e permite batch sizes maiores.
- **int8** se beneficia de aceleração de hardware; **nf4-bnb** reduz ainda mais a VRAM, mas é mais sensível.
- Ao carregar o LoRA depois, você **idealmente** deve usar a mesma precisão do modelo base usada no treino.

### Bucketização de aspecto

- Treinar apenas com recortes quadrados geralmente funciona, mas buckets multi-aspect podem melhorar a generalização.
- Usar buckets naturais pode enviesar formas; recorte aleatório pode ajudar se você precisar de cobertura mais ampla.
- Misturar configurações de dataset definindo seu diretório de imagens múltiplas vezes gerou boa generalização.

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

- Taxas de aprendizado menores costumam se comportar melhor em transformers grandes.
- Comece com ranks modestos (4–16) antes de tentar ranks muito altos.
- Reduza `max_grad_norm` se o modelo ficar instável; aumente se o aprendizado travar.

#### LoKr (--lora_type=lycoris)

- Taxas de aprendizado mais altas (ex.: `1e-3` com AdamW, `2e-4` com Lion) podem funcionar bem; ajuste conforme necessário.
- Marque datasets de regularização com `is_regularisation_data` para ajudar a preservar o modelo base.

### Artefatos de imagem

Z-Image absorve artefatos ruins cedo. Um passe final com dados de alta qualidade pode ser necessário para limpar. Fique atento a artefatos de grade se a taxa de aprendizado for alta, os dados forem de baixa qualidade ou o manejo de aspecto estiver errado.

### Treinando modelos Z-Image fine-tuned customizados

Alguns checkpoints fine-tuned podem não ter a estrutura completa de diretórios. Defina estes campos conforme necessário:

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_family": "z-image",
    "pretrained_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_model_name_or_path": "your-custom-transformer",
    "pretrained_vae_model_name_or_path": "TONGYI-MAI/Z-Image-Base",
    "pretrained_transformer_subfolder": "none"
}
```
</details>

## Troubleshooting

- OOM na inicialização: habilite group offload (não com Quanto), reduza rank do LoRA, ou quantize (`--base_model_precision int8`/`nf4`).
- Saídas borradas: aumente `validation_num_inference_steps` (ex.: 24–28) ou eleve guidance para 1.0.
- Artefatos/overfitting: reduza rank ou taxa de aprendizado, adicione prompts mais diversos, ou encurte o treino.
- Problemas com adapter assistente: turbo espera o caminho/peso do adapter; só desative se aceitar perda de qualidade.
- Validações lentas: reduza resolução ou steps; flow matching converge rápido.
