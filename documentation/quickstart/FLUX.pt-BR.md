# Guia de Início Rápido do Flux[dev] / Flux[schnell]

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

Neste exemplo, vamos treinar um LoRA Flux.1 Krea.

## Requisitos de hardware

Flux requer muita **RAM do sistema** além da memória da GPU. Apenas quantizar o modelo na inicialização exige cerca de 50GB de memória do sistema. Se isso estiver levando tempo demais, talvez seja necessário avaliar as capacidades do seu hardware e se mudanças são necessárias.

Quando você treina todos os componentes de um LoRA de rank 16 (MLP, projeções, blocos multimodais), o uso acaba ficando em:

- pouco mais de 30G de VRAM quando não quantiza o modelo base
- pouco mais de 18G de VRAM quando quantiza para int8 + pesos base/LoRA em bf16
- pouco mais de 13G de VRAM quando quantiza para int4 + pesos base/LoRA em bf16
- pouco mais de 9G de VRAM quando quantiza para NF4 + pesos base/LoRA em bf16
- pouco mais de 9G de VRAM quando quantiza para int2 + pesos base/LoRA em bf16

Você vai precisar:

- **mínimo absoluto**: uma única **3080 10G**
- **mínimo realista**: uma única 3090 ou GPU V100
- **idealmente**: várias 4090, A6000, L40S ou superior

Felizmente, essas GPUs estão disponíveis em provedores como [LambdaLabs](https://lambdalabs.com), que oferece os menores preços, e clusters localizados para treino multi-node.

**Ao contrário de outros modelos, GPUs Apple não funcionam atualmente para treinar Flux.**


## Pré-requisitos

Certifique-se de que você tem Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.12 python3.12-venv
```

### Dependências da imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para habilitar a compilação de extensões CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

## Instalação

Instale o SimpleTuner via pip:

```bash
pip install simpletuner[cuda]
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

A WebUI do SimpleTuner torna a configuração bastante direta. Para rodar o servidor:

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
- `model_family` - Defina como `flux`.
- `model_flavour` - isso é `krea` por padrão, mas pode ser definido como `dev` para treinar o release original FLUX.1-Dev.
  - `krea` - O modelo padrão FLUX.1-Krea [dev], uma variante de pesos abertos do Krea 1, um modelo proprietário em colaboração entre BFL e Krea.ai
  - `dev` - flavour Dev, o padrão anterior
  - `schnell` - flavour Schnell; o quickstart define automaticamente o schedule de ruído rápido e o stack de LoRA assistente
  - `kontext` - treino Kontext (veja [este guia](../quickstart/FLUX_KONTEXT.md) para orientação específica)
  - `fluxbooru` - Um modelo de-destilado (requer CFG) baseado no FLUX.1-Dev chamado [FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3), criado pelo Terminus Research Group
  - `libreflux` - Um modelo de-destilado baseado no FLUX.1-Schnell que requer mascaramento de atenção nas entradas do encoder de texto T5
- `offload_during_startup` - Defina como `true` se faltar memória durante encodes do VAE.
- `pretrained_model_name_or_path` - Defina como `black-forest-labs/FLUX.1-dev`.
- `pretrained_vae_model_name_or_path` - Defina como `black-forest-labs/FLUX.1-dev`.
  - Note que você precisará fazer login no Hugging Face e obter acesso para baixar este modelo. Vamos cobrir o login no Hugging Face mais adiante neste tutorial.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - mantenha em 1, especialmente se você tem um dataset muito pequeno.
- `validation_resolution` - Como o Flux é um modelo 1024px, você pode definir `1024x1024`.
  - Além disso, o Flux foi fine-tuned em buckets multi-aspect, e outras resoluções podem ser especificadas usando vírgulas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use o que você costuma selecionar na inferência para o Flux.
- `validation_guidance_real` - Use >1.0 para usar CFG na inferência do Flux. Isso deixa as validações mais lentas, mas produz melhores resultados. Funciona melhor com `VALIDATION_NEGATIVE_PROMPT` vazio.
- `validation_num_inference_steps` - Use algo em torno de 20 para economizar tempo e ainda ver qualidade razoável. Flux não é muito diverso, e mais steps podem apenas desperdiçar tempo.
- `--lora_rank=4` se você quiser reduzir bastante o tamanho do LoRA treinado. Isso ajuda no uso de VRAM.
- Treinos de LoRA Schnell usam o schedule rápido automaticamente via defaults do quickstart; não são necessários flags extras.

- `gradient_accumulation_steps` - A orientação anterior era evitar isso com treino bf16 porque degradaria o modelo. Testes adicionais mostraram que isso não é necessariamente o caso para o Flux.
  - Essa opção faz com que passos de atualização sejam acumulados por vários steps. Isso aumenta o tempo de treino de forma linear, de modo que um valor 2 torna o treino duas vezes mais lento e dura duas vezes mais.
- `optimizer` - Iniciantes devem ficar com adamw_bf16, embora optimi-lion e optimi-stableadamw também sejam boas escolhas.
- `mixed_precision` - Iniciantes devem manter em `bf16`
- `gradient_checkpointing` - defina como true em praticamente todas as situações e em todos os dispositivos
- `gradient_checkpointing_interval` - pode ser definido como 2 ou mais em GPUs maiores para checkpointar apenas a cada _n_ blocos. Um valor 2 checkpointa metade dos blocos, e 3 checkpointa um terço.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

### Offload de memória (opcional)

Flux suporta offload agrupado de módulos via diffusers v0.33+. Isso reduz dramaticamente a pressão de VRAM quando o gargalo são os pesos do transformer. Você pode habilitar adicionando as seguintes flags ao `TRAINER_EXTRA_ARGS` (ou na página Hardware da WebUI):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` só é efetivo em dispositivos CUDA; o SimpleTuner desativa streams automaticamente em backends ROCm, MPS e CPU.
- **Não** combine isso com `--enable_model_cpu_offload` — as duas estratégias são mutuamente exclusivas.
- Ao usar `--group_offload_to_disk_path`, prefira um SSD/NVMe local rápido.

#### Prompts de validação

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

> ℹ️ Flux é um modelo de flow-matching e prompts mais curtos com forte similaridade resultarão praticamente na mesma imagem produzida pelo modelo. Certifique-se de usar prompts mais longos e descritivos.

#### Rastreamento de score CLIP

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre como configurar e interpretar scores CLIP.

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar a perda de avaliação.

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

#### Ajuste do schedule de tempo do Flux

Modelos de flow-matching como Flux e SD3 têm uma propriedade chamada "shift" que permite deslocar a parte treinada do schedule de timesteps usando um valor decimal simples.

##### Padrões

Por padrão, nenhum shift de schedule é aplicado ao Flux, o que resulta em uma forma de sino sigmoide na distribuição de amostragem de timesteps. Isso provavelmente não é a abordagem ideal para o Flux, mas resulta em maior aprendizado em menos tempo do que o auto-shift.

##### Auto-shift

Uma abordagem comumente recomendada é seguir vários trabalhos recentes e habilitar o shift de timesteps dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em treinamentos estáveis, mas potencialmente medianos.

##### Especificação manual

(_Agradecimentos ao General Awareness no Discord pelos exemplos a seguir_)

Ao usar um valor de `--flow_schedule_shift` de 0.1 (um valor muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (um valor muito alto), os grandes recursos de composição e possivelmente o espaço de cor do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Treinamento com modelo quantizado

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisão e os requisitos de VRAM, treinando o Flux com apenas 16GB.

Para usuários de `config.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

##### Configurações específicas de LoRA (não LyCORIS)

```bash
# When training 'mmdit', we find very stable training that makes the model take longer to learn.
# When training 'all', we can easily shift the model distribution, but it is more prone to forgetting and benefits from high quality data.
# When training 'all+ffs', all attention layers are trained in addition to the feed-forward which can help with adapting the model objective for the LoRA.
# - This mode has been reported to lack portability, and platforms such as ComfyUI might not be able to load the LoRA.
# The option to train only the 'context' blocks is offered as well, but its impact is unknown, and is offered as an experimental choice.
# - An extension to this mode, 'context+ffs' is also available, which is useful for pretraining new tokens into a LoRA before continuing finetuning it via `--init_lora`.
# Other options include 'tiny' and 'nano' which train just 1 or 2 layers.
"--flux_lora_target": "all",

# If you want to use LoftQ initialisation, you can't use Quanto to quantise the base model.
# This possibly offers better/faster convergence, but only works on NVIDIA devices and requires Bits n Bytes and is incompatible with Quanto.
# Other options are 'default', 'gaussian' (difficult), and untested options: 'olora' and 'pissa'.
"--lora_init_type": "loftq",
```

#### Considerações sobre o dataset

> ⚠️ A qualidade das imagens para treinamento é mais importante para o Flux do que para a maioria dos outros modelos, pois ele absorve os artefatos das suas imagens _primeiro_ e depois aprende o conceito/sujeito.

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisa garantir que seu dataset seja grande o suficiente para treinar de forma eficaz. Note que o tamanho mínimo de dataset é `train_batch_size * gradient_accumulation_steps` além de ser maior que `vae_batch_size`. O dataset não será utilizável se for pequeno demais.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` vai superar essa limitação.

Dependendo do dataset que você tem, será necessário configurar seu diretório de dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
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
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ Rodar datasets 512px e 1024px simultaneamente é suportado e pode resultar em melhor convergência para o Flux.

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
pip install simpletuner[cuda]
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

**Nota:** Não está claro se o treinamento em buckets multi-aspect funciona corretamente para o Flux no momento. É recomendado usar `crop_style=random` e `crop_aspect=square`.

## Configuração multi-GPU

O SimpleTuner inclui **detecção automática de GPU** pela WebUI. Durante o onboarding, você vai configurar:

- **Modo Automático**: Usa automaticamente todas as GPUs detectadas com configurações ideais
- **Modo Manual**: Seleciona GPUs específicas ou define uma contagem de processos personalizada
- **Modo Desativado**: Treino com uma única GPU

A WebUI detecta seu hardware e configura `--num_processes` e `CUDA_VISIBLE_DEVICES` automaticamente.

Para configuração manual ou setups avançados, veja a [seção de treinamento multi-GPU](../INSTALL.md#multiple-gpu-training) no guia de instalação.

## Dicas de inferência

### LoRAs treinadas com CFG (flux_guidance_value > 1)

No ComfyUI, você vai precisar passar o Flux por outro nó chamado AdaptiveGuider. Um membro da comunidade forneceu um nó modificado aqui:

(**links externos**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) e o workflow de exemplo [aqui](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### LoRA destilada por CFG (flux_guidance_scale == 1)

Inferir a LoRA destilada por CFG é tão simples quanto usar um guidance_scale mais baixo em torno do valor com que ela foi treinada.

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

Atualmente, a menor utilização de VRAM (9090M) pode ser obtida com:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G de memória do sistema
- Precisão do modelo base: `nf4-bnb`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 512px
  - 1024px requer >= 12G de VRAM
- Tamanho de lote: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- PyTorch: 2.6 Nightly (build de 29 de setembro)
- Usar `--quantize_via=cpu` para evitar erro outOfMemory na inicialização em placas <=16G.
- Com `--attention_mechanism=sageattention` para reduzir ainda mais a VRAM em 0,1GB e melhorar a velocidade de geração de imagens de validação no treino.
- Garanta que `--gradient_checkpointing` esteja habilitado ou nada vai impedir OOM

**NOTA**: O pré-cache de embeddings do VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. Se isso ocorrer, a quantização do encoder de texto e o VAE tiling podem ser habilitados via `--vae_enable_tiling=true`. Mais memória pode ser economizada na inicialização com `--offload_during_startup=true`.

A velocidade foi aproximadamente 1,4 iterações por segundo em uma 4090.

### SageAttention

Ao usar `--attention_mechanism=sageattention`, a inferência pode ficar mais rápida durante a validação.

**Nota**: Isso não é compatível com _todas_ as configurações de modelo, mas vale a tentativa.

### Treino quantizado em NF4

Em termos simples, NF4 é uma representação _quase_ 4 bits do modelo, o que significa que o treino tem sérias preocupações de estabilidade a resolver.

Em testes iniciais, o seguinte se mantém:

- Otimizador Lion causa colapso do modelo, mas usa menos VRAM; variantes AdamW ajudam a manter o treino; bnb-adamw8bit, adamw_bf16 são ótimas escolhas
  - AdEMAMix não se saiu bem, mas as configurações não foram exploradas
- `--max_grad_norm=0.01` ajuda ainda mais a reduzir a quebra do modelo ao evitar mudanças gigantes em pouco tempo
- NF4, AdamW8bit e um batch size maior ajudam a superar os problemas de estabilidade, ao custo de mais tempo de treino ou VRAM
- Aumentar a resolução de 512px para 1024px desacelera o treino de, por exemplo, 1,4 s por step para 3,5 s por step (batch size 1, 4090)
- Tudo que é difícil de treinar em int8 ou bf16 fica mais difícil em NF4
- É menos compatível com opções como SageAttention

NF4 não funciona com torch.compile, então qualquer velocidade que você obtiver é a que terá.

Se VRAM não for uma preocupação (ex.: 48G ou mais), então int8 com torch.compile é sua melhor opção, a mais rápida.

### Perda com máscara

Se você está treinando um sujeito ou estilo e gostaria de mascarar um ou outro, veja a seção [treino com loss mascarada](../DREAMBOOTH.md#masked-loss) do guia de Dreambooth.

### Treino TREAD

> ⚠️ **Experimental**: TREAD é um recurso recém-implementado. Embora funcione, configurações ideais ainda estão sendo exploradas.

[TREAD](../TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. É um método que pode acelerar o treino do Flux ao rotear tokens de forma inteligente pelas camadas do transformer. A aceleração é proporcional a quantos tokens você descarta.

#### Configuração rápida

Adicione isto ao seu `config.json`:

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

Essa configuração vai:

- Manter apenas 50% dos tokens de imagem nas camadas 2 até a penúltima
- Tokens de texto nunca são descartados
- Aceleração de treino de ~25% com impacto mínimo na qualidade

#### Pontos-chave

- **Suporte de arquitetura limitado** - TREAD só está implementado para modelos Flux e Wan
- **Melhor em altas resoluções** - maiores acelerações em 1024x1024+ devido à complexidade O(n²) da atenção
- **Compatível com perda com máscara** - regiões mascaradas são preservadas automaticamente (mas isso reduz a aceleração)
- **Funciona com quantização** - pode ser combinado com treino int8/int4/NF4
- **Espere um pico de loss inicial** - ao iniciar treino LoRA/LoKr, a loss será mais alta no começo, mas corrige rapidamente

#### Dicas de ajuste

- **Conservador (foco em qualidade)**: use `selection_ratio` de 0.3-0.5
- **Agressivo (foco em velocidade)**: use `selection_ratio` de 0.6-0.8
- **Evite camadas iniciais/finais**: não roteie nas camadas 0-1 ou na camada final
- **Para treino LoRA**: pode haver pequenas desacelerações - experimente configs diferentes
- **Maior resolução = melhor aceleração**: mais benefício em 1024px ou mais

#### Comportamento conhecido

- Quanto mais tokens descartados (maior `selection_ratio`), mais rápido o treino, mas maior a loss inicial
- Treino LoRA/LoKr mostra um pico de loss inicial que corrige rapidamente conforme a rede se adapta
- Algumas configurações de LoRA podem treinar um pouco mais devagar - configs ideais ainda estão sendo exploradas
- A implementação de RoPE (rotary position embedding) é funcional, mas talvez não esteja 100% correta

Para opções detalhadas de configuração e troubleshooting, veja a [documentação completa do TREAD](../TREAD.md).

### Guidance sem classificador

#### Problema

O modelo Dev chega destilado por guidance por padrão, o que significa que ele faz uma trajetória muito direta até as saídas do modelo teacher. Isso é feito por meio de um vetor de guidance alimentado no modelo em tempo de treino e inferência - o valor desse vetor impacta muito o tipo de LoRA resultante:

#### Solução

- Um valor de 1.0 (**padrão**) preserva a destilação inicial do modelo Dev
  - Este é o modo mais compatível
  - A inferência é tão rápida quanto no modelo original
  - A destilação por flow-matching reduz a criatividade e a variabilidade das saídas do modelo, como no Flux Dev original (tudo mantém a mesma composição/visual)
- Um valor mais alto (testado em ~3.5-4.5) reintroduz o objetivo de CFG no modelo
  - Isso requer que o pipeline de inferência tenha suporte a CFG
  - A inferência fica 50% mais lenta e 0% de aumento de VRAM **ou** cerca de 20% mais lenta e 20% a mais de VRAM devido à inferência CFG em batch
  - Porém, esse estilo de treino melhora a criatividade e a variabilidade das saídas do modelo, o que pode ser necessário para certas tarefas de treino

Podemos reintroduzir parcialmente a destilação em um modelo de-destilado continuando o ajuste do modelo usando um valor de vetor 1.0. Ele nunca vai se recuperar totalmente, mas pelo menos ficará mais utilizável.

#### Caveats

- Isso tem o impacto final de **ou**:
  - Aumentar a latência de inferência em 2x quando calculamos a saída incondicional de forma sequencial, por exemplo, com duas passagens de forward separadas
  - Aumentar o consumo de VRAM de forma equivalente a usar `num_images_per_prompt=2` e receber duas imagens na inferência, acompanhado do mesmo percentual de desaceleração.
    - Isso costuma ser menos extremo do que computação sequencial, mas o uso de VRAM pode ser alto demais para a maioria do hardware de treino de consumidor.
    - Este método não está _atualmente_ integrado ao SimpleTuner, mas o trabalho continua.
- Fluxos de inferência para ComfyUI ou outros aplicativos (ex.: AUTOMATIC1111) precisarão ser modificados para habilitar o CFG "real", o que talvez não seja possível de imediato.

### Quantização

- Quantização mínima de 8 bits é necessária para uma placa de 16G treinar este modelo
  - Em bfloat16/float16, um LoRA de rank 1 fica com pouco mais de 30GB de uso de memória
- Quantizar o modelo em 8 bits não prejudica o treino
  - Isso permite aumentar batch size e possivelmente obter melhor resultado
  - Comporta-se igual ao treino em precisão total - fp32 não vai fazer seu modelo ficar melhor do que bf16+int8.
- **int8** tem aceleração de hardware e suporte a `torch.compile()` em hardware NVIDIA mais recente (3090 ou melhor)
- **nf4-bnb** reduz a VRAM para 9GB, cabendo em uma placa de 10G (com suporte a bfloat16)
- Ao carregar o LoRA no ComfyUI depois, você **precisa** usar a mesma precisão do modelo base usada no treino do LoRA.
- **int4** depende de kernels bf16 customizados e não funciona se sua placa não suporta bfloat16

### Crashes

- Se você receber SIGKILL após os encoders de texto serem descarregados, isso significa que não há memória de sistema suficiente para quantizar o Flux.
  - Tente carregar com `--base_model_precision=bf16`, mas se isso não funcionar, talvez você só precise de mais memória.
  - Tente `--quantize_via=accelerator` para usar a GPU em vez disso

### Schnell

- Se você treinar um LyCORIS LoKr no Dev, ele **geralmente** funciona muito bem no Schnell com apenas 4 steps depois.
  - Treino direto no Schnell realmente precisa de mais tempo no forno - atualmente, os resultados não parecem bons

> ℹ️ Ao mesclar Schnell com Dev de qualquer forma, a licença do Dev prevalece e ele se torna não comercial. Isso provavelmente não importa para a maioria dos usuários, mas vale a nota.

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

- LoRA tem desempenho geral pior do que LoKr para datasets maiores
- Há relatos de que o LoRA do Flux treina de forma semelhante aos LoRAs do SD 1.5
- Porém, um modelo tão grande quanto 12B se mostrou empiricamente melhor com **taxas de aprendizado menores.**
  - LoRA em 1e-3 pode torrar tudo. LoRA em 1e-5 quase não faz nada.
- Ranks tão grandes quanto 64 a 128 podem ser indesejáveis em um modelo 12B devido a dificuldades gerais que escalam com o tamanho do modelo base.
  - Tente uma rede menor primeiro (rank-1, rank-4) e vá aumentando - elas treinam mais rápido e podem entregar tudo o que você precisa.
  - Se você está achando muito difícil treinar seu conceito no modelo, talvez precise de um rank maior e mais dados de regularização.
- Outros modelos diffusion transformer como PixArt e SD3 se beneficiam bastante de `--max_grad_norm` e o SimpleTuner mantém um valor alto por padrão no Flux.
  - Um valor menor impediria o modelo de desmoronar cedo demais, mas também pode tornar muito difícil aprender novos conceitos que fogem da distribuição de dados do modelo base. O modelo pode travar e nunca melhorar.

#### LoKr (--lora_type=lycoris)

- Taxas de aprendizado mais altas são melhores para LoKr (`1e-3` com AdamW, `2e-4` com Lion)
- Outros algoritmos precisam de mais exploração.
- Definir `is_regularisation_data` nesses datasets pode ajudar a preservar / evitar bleed e melhorar a qualidade do modelo final.
  - Isso se comporta diferente de "prior loss preservation", que é conhecido por dobrar batch sizes e não melhorar muito o resultado
  - A implementação de dados de regularização do SimpleTuner fornece uma maneira eficiente de preservar o modelo base

### Artefatos de imagem

Flux absorve imediatamente artefatos ruins de imagem. É assim mesmo - uma execução final de treino apenas com dados de alta qualidade pode ser necessária para corrigir isso no final.

Quando você faz estas coisas (entre outras), alguns artefatos de grade quadrada **podem** começar a aparecer nas amostras:

- Treinar demais com dados de baixa qualidade
- Usar taxa de aprendizado alta demais
- Overtraining (em geral), uma rede de baixa capacidade com muitas imagens
- Undertraining (também), uma rede de alta capacidade com poucas imagens
- Usar proporções estranhas ou tamanhos de dados de treino

### Bucketização de aspecto

- Treinar por tempo demais com recortes quadrados provavelmente não vai prejudicar esse modelo. Pode exagerar, é ótimo e confiável.
- Por outro lado, usar os buckets de aspecto naturais do seu dataset pode enviesar demais essas formas durante a inferência.
  - Isso pode ser desejável, pois mantém estilos dependentes de aspecto, como coisas cinematográficas, de vazarem para outras resoluções.
  - Porém, se você quer melhorar resultados igualmente em muitos buckets de aspecto, talvez precise experimentar `crop_aspect=random`, o que tem suas próprias desvantagens.
- Misturar configurações de dataset definindo o dataset do diretório de imagens múltiplas vezes produziu resultados muito bons e um modelo bem generalizado.

### Treinando modelos Flux fine-tuned customizados

Alguns modelos Flux fine-tuned no Hugging Face Hub (como Dev2Pro) não têm a estrutura completa de diretórios, exigindo que estas opções específicas sejam definidas.

Certifique-se de definir estas opções `flux_guidance_value`, `validation_guidance_real` e `flux_attention_masked_training` conforme a forma como o criador fez, se essa informação estiver disponível.

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
</details>
