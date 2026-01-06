## Guia de Início Rápido do LTX Video

Neste exemplo, vamos treinar um LoRA de LTX-Video usando o [dataset Disney de domínio público](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) de Sayak Paul.

### Requisitos de hardware

LTX não requer muita memória de sistema **ou** GPU.

Quando você treina todos os componentes de um LoRA de rank 16 (MLP, projeções, blocos multimodais), o uso fica um pouco acima de 12G em um Mac M3 (batch size 4).

Você vai precisar:
- **mínimo realista**: 16GB ou uma única GPU 3090 ou V100
- **idealmente**: várias 4090, A6000, L40S ou melhor

Sistemas Apple Silicon funcionam muito bem com LTX até agora, embora em resoluções menores por conta de limites no backend MPS usado pelo PyTorch.

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
- Evite `--group_offload_to_disk_path` a menos que a RAM do sistema seja <64 GB — staging em disco é mais lento, mas mantém a execução estável.
- Desative `--enable_model_cpu_offload` ao usar group offloading.

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
pip install simpletuner[cuda]
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

#### Etapas adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

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
- `model_family` - Defina como `ltxvideo`.
- `pretrained_model_name_or_path` - Defina como `Lightricks/LTX-Video-0.9.5`.
- `pretrained_vae_model_name_or_path` - Defina como `Lightricks/LTX-Video-0.9.5`.
- `output_dir` - Defina como o diretório onde você quer armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - isso pode ser aumentado para mais estabilidade, mas um valor 4 deve funcionar bem para começar
- `validation_resolution` - Defina como o que você normalmente usa para gerar vídeos com LTX (`768x512`)
  - Múltiplas resoluções podem ser especificadas usando vírgulas: `1280x768,768x512`
- `validation_guidance` - Use o que você costuma selecionar na inferência para LTX.
- `validation_num_inference_steps` - Use algo em torno de 25 para economizar tempo e ainda ver qualidade razoável.
- `--lora_rank=4` se você quiser reduzir bastante o tamanho do LoRA treinado. Isso ajuda no uso de VRAM ao mesmo tempo em que reduz sua capacidade de aprendizado.

- `gradient_accumulation_steps` - Essa opção faz com que passos de atualização sejam acumulados por vários steps.
  - Isso aumenta o tempo de treino de forma linear, de modo que um valor 2 torna o treino duas vezes mais lento e dura duas vezes mais.
- `optimizer` - Iniciantes devem ficar com adamw_bf16, embora optimi-lion e optimi-stableadamw também sejam boas escolhas.
- `mixed_precision` - Iniciantes devem manter em `bf16`
- `gradient_checkpointing` - defina como true em praticamente todas as situações e em todos os dispositivos
- `gradient_checkpointing_interval` - ainda não é suportado no LTX Video e deve ser removido do seu config.

Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

Ao final, seu config deve se parecer com o meu:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "data_backend_config": "config/ltxvideo/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "disable_benchmark": false,
  "offload_during_startup": true,
  "output_dir": "output/ltxvideo",
  "lora_type": "lycoris",
  "lycoris_config": "config/ltxvideo/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "ltxvideo-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "ltxvideo-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Lightricks/LTX-Video-0.9.5",
  "model_family": "ltxvideo",
  "train_batch_size": 8,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 800,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "768x512",
  "validation_negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
  "validation_guidance": 3.0,
  "validation_num_inference_steps": 40,
  "validation_prompt": "The video depicts a long, straight highway stretching into the distance, flanked by metal guardrails. The road is divided into multiple lanes, with a few vehicles visible in the far distance. The surrounding landscape features dry, grassy fields on one side and rolling hills on the other. The sky is mostly clear with a few scattered clouds, suggesting a bright, sunny day. And then the camera switch to a inding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region.",
  "mixed_precision": "bf16",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.0,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "fp8-torchao",
  "vae_batch_size": 1,
  "webhook_config": "config/ltxvideo/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 128,
  "flow_schedule_shift": 1,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

### Opcional: regularizador temporal CREPA

Se suas execuções de LTX apresentarem flicker ou deriva de identidade, tente CREPA (alinhamento entre frames):
- Na WebUI, vá em **Training → Loss functions** e habilite **CREPA**.
- Comece com **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- Mantenha o encoder de visão padrão (`dinov2_vitg14`, tamanho `518`). Mude para `dinov2_vits14` + `224` apenas se precisar de menor VRAM.
- Precisa de internet (ou um torch hub em cache) na primeira vez para buscar pesos do DINOv2.
- Opcional: se treinar puramente a partir de latentes em cache, habilite **Drop VAE Encoder** para economizar memória; mantenha desligado se precisar codificar vídeos novos.

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

> ℹ️ LTX Video é um modelo de flow-matching baseado em T5 XXL; prompts curtos podem não ter informação suficiente para o modelo fazer um bom trabalho. Certifique-se de usar prompts mais longos e descritivos.

#### Rastreamento de score CLIP

Isso não deve ser habilitado para treino de modelos de vídeo no momento.

</details>

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

#### Shift do schedule de flow-matching

Modelos de flow-matching como Flux, Sana, SD3 e LTX Video têm uma propriedade chamada `shift` que permite deslocar a parte treinada do schedule de timesteps usando um valor decimal simples.

##### Padrões
Por padrão, nenhum shift de schedule é aplicado, o que resulta em uma forma de sino sigmoide na distribuição de amostragem de timesteps, também conhecida como `logit_norm`.

##### Auto-shift
Uma abordagem comumente recomendada é seguir vários trabalhos recentes e habilitar o shift de timesteps dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em treinamentos estáveis, mas potencialmente medianos.

##### Especificação manual
_Agradecimentos ao General Awareness no Discord pelos exemplos a seguir_

> ℹ️ Esses exemplos mostram como o valor funciona usando Flux Dev, embora LTX Video deva ser bem semelhante.

Ao usar um valor de `--flow_schedule_shift` de 0.1 (um valor muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (um valor muito alto), os grandes recursos de composição e possivelmente o espaço de cor do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Treino com modelo quantizado

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisão e os requisitos de VRAM, treinando com apenas 16GB.



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
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

- Na subseção `video`, temos as seguintes chaves que podemos definir:
  - `num_frames` (opcional, int) é quantos frames de dados vamos treinar.
    - A 25 fps, 125 frames equivalem a 5 segundos de vídeo, saída padrão. Este deve ser seu alvo.
  - `min_frames` (opcional, int) determina o comprimento mínimo de vídeo considerado para treino.
    - Isso deve ser pelo menos igual a `num_frames`. Não definir garante que será igual.
  - `max_frames` (opcional, int) determina o comprimento máximo de vídeo considerado para treino.
  - `is_i2v` (opcional, bool) determina se o treino i2v será feito em um dataset.
    - Isso é definido como True por padrão para LTX. Você pode desativar.
  - `bucket_strategy` (opcional, string) determina como os vídeos são agrupados em buckets:
    - `aspect_ratio` (padrão): agrupa apenas pela proporção espacial (ex.: `1.78`, `0.75`).
    - `resolution_frames`: agrupa por resolução e contagem de frames no formato `WxH@F` (ex.: `768x512@125`). Útil para datasets com resolução/duração mistas.
  - `frame_interval` (opcional, int) ao usar `resolution_frames`, arredonde contagens de frames para este intervalo. Defina isso para o fator de contagem de frames exigido pelo seu modelo.

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

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

Como outros modelos, é possível que a menor utilização de VRAM seja obtida com:

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 11G de memória do sistema
- Precisão do modelo base: `nf4-bnb`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 480px
- Tamanho de lote: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- PyTorch: 2.6
- Garanta que `--gradient_checkpointing` esteja habilitado ou nada vai impedir OOM

**NOTA**: O pré-cache de embeddings do VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. Se isso ocorrer, a quantização do encoder de texto e o VAE tiling podem ser habilitados. Além dessas opções, `--offload_during_startup=true` ajuda a evitar competição entre uso de memória do VAE e do encoder de texto.

A velocidade foi aproximadamente 0,8 iteração por segundo em um Macbook Pro M3 Max.

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
- Aumentar a resolução desacelera bastante o treino e pode prejudicar o modelo
- Aumentar o comprimento dos vídeos também consome muito mais memória. Reduza `num_frames` para contornar isso.
- Tudo que é difícil de treinar em int8 ou bf16 fica mais difícil em NF4
- É menos compatível com opções como SageAttention

NF4 não funciona com torch.compile, então qualquer velocidade que você obtiver é a que terá.

Se VRAM não for uma preocupação, então int8 com torch.compile é sua melhor opção, a mais rápida.

### Perda com máscara

Não use isso com LTX Video.


### Quantização
- Quantização não é necessária para treinar este modelo

### Artefatos de imagem
Como outros modelos DiT, se você fizer estas coisas (entre outras) alguns artefatos de grade quadrada **podem** começar a aparecer nas amostras:
- Treinar demais com dados de baixa qualidade
- Usar taxa de aprendizado alta demais
- Overtraining (em geral), uma rede de baixa capacidade com muitas imagens
- Undertraining (também), uma rede de alta capacidade com poucas imagens
- Usar proporções estranhas ou tamanhos de dados de treino

### Bucketização de aspecto
- Vídeos são colocados em buckets como imagens.
- Treinar por tempo demais com recortes quadrados provavelmente não vai prejudicar esse modelo. Pode exagerar, é ótimo e confiável.
- Por outro lado, usar os buckets de aspecto naturais do seu dataset pode enviesar demais essas formas durante a inferência.
  - Isso pode ser desejável, pois mantém estilos dependentes de aspecto, como coisas cinematográficas, de vazarem para outras resoluções.
  - Porém, se você quer melhorar resultados igualmente em muitos buckets de aspecto, talvez precise experimentar `crop_aspect=random`, o que tem suas próprias desvantagens.
- Misturar configurações de dataset definindo o dataset do diretório de imagens múltiplas vezes produziu resultados muito bons e um modelo bem generalizado.

### Treinando modelos LTX fine-tuned customizados

Alguns modelos fine-tuned no Hugging Face Hub não têm a estrutura completa de diretórios, exigindo que opções específicas sejam definidas.

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_family": "ltxvideo",
    "pretrained_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Lightricks/LTX-Video",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

## Créditos

O projeto [finetrainers](https://github.com/a-r-r-o-w/finetrainers) e o time do Diffusers.
- Originalmente usou alguns conceitos de design do SimpleTuner
- Agora contribui com insights e código para tornar o treino de vídeo fácil de implementar
