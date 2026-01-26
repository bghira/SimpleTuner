## Guia de Início Rápido do Wan 2.1

Neste exemplo, vamos treinar um LoRA do Wan 2.1 usando o [dataset Disney de domínio público](https://hf.co/datasets/sayakpaul/video-dataset-disney-organized) de Sayak Paul.



https://github.com/user-attachments/assets/51e6cbfd-5c46-407c-9398-5932fa5fa561


### Requisitos de hardware

Wan 2.1 **1.3B** não requer muita memória de sistema **ou** GPU. O modelo **14B**, também suportado, é bem mais exigente.

Atualmente, o treino de imagem-para-vídeo não é suportado para Wan, mas LoRA e Lycoris T2V rodam nos modelos I2V.

#### Texto para vídeo

1.3B - https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Resolução: 832x480
- LoRA rank 16 usa um pouco mais de 12G (batch size 4)

14B - https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
- Resolução: 832x480
- Cabe em 24G, mas você vai precisar ajustar algumas configurações.

<!--
#### Image to Video
14B (720p) - https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers
- Resolution: 1280x720
-->

#### Image to Video (Wan 2.2)

Checkpoints I2V recentes do Wan 2.2 funcionam com o mesmo fluxo de treino:

- Stage high: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/high_noise_model
- Stage low: https://huggingface.co/Wan-AI/Wan2.2-I2V-14B-Diffusers/tree/main/low_noise_model

Você pode escolher o stage desejado com as configurações `model_flavour` e `wan_validation_load_other_stage` descritas mais adiante neste guia.

Você vai precisar:
- **mínimo realista**: 16GB ou uma única GPU 3090 ou V100
- **idealmente**: várias 4090, A6000, L40S ou melhor

Se você encontrar incompatibilidades de shape nas camadas de time embedding ao rodar checkpoints do Wan 2.2, habilite o novo
flag `wan_force_2_1_time_embedding`. Isso força o transformer a usar time embeddings no estilo Wan 2.1 e resolve o problema de compatibilidade.

#### Presets de stage e validação

- `model_flavour=i2v-14b-2.2-high` mira no stage high-noise do Wan 2.2.
- `model_flavour=i2v-14b-2.2-low` mira no stage low-noise (mesmos checkpoints, subpasta diferente).
- Ative `wan_validation_load_other_stage=true` para carregar o stage oposto junto do que você treina para renderizações de validação.
- Deixe o flavour sem definir (ou use `t2v-480p-1.3b-2.1`) para o run padrão de texto-para-vídeo do Wan 2.1.

#### Validação I2V com Datasets de Imagens

Para validação i2v, você pode usar um dataset de imagens simples sem precisar do pareamento completo de datasets de condicionamento:

```json
{
  "validation_using_datasets": true,
  "eval_dataset_id": "my-image-dataset"
}
```

Isso usa imagens do dataset especificado como entradas de condicionamento do primeiro frame para geração de vídeos de validação, sem precisar da configuração complexa de condicionamento usada durante o treinamento.

Sistemas Apple Silicon não funcionam muito bem com o Wan 2.1 até agora; espere algo como 10 minutos por step de treino.

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
pip install 'simpletuner[cuda13]'
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).
#### SageAttention 2

Se você quiser usar SageAttention 2, alguns passos devem ser seguidos.

> Nota: SageAttention oferece ganho mínimo de velocidade, não é muito efetivo; não sei por quê. Testado em 4090.

Execute o seguinte ainda dentro do seu venv do Python:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

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

### Offload de memória (opcional)

Wan é um dos modelos mais pesados que o SimpleTuner suporta. Habilite offload em grupo se estiver perto do teto de VRAM:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Apenas dispositivos CUDA respeitam `--group_offload_use_stream`; ROCm/MPS fazem fallback automaticamente.
- Deixe o staging em disco comentado a menos que a memória da CPU seja o gargalo.
- `--enable_model_cpu_offload` é mutuamente exclusivo com group offload.

### Chunking de feed-forward (opcional)

Se os checkpoints 14B ainda derem OOM durante o gradient checkpointing, faça chunking das camadas feed-forward do Wan:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

Isso corresponde ao novo toggle no assistente de configuração (`Training → Memory Optimisation`). Tamanhos menores economizam mais
memória, mas deixam cada step mais lento. Você também pode definir `WAN_FEED_FORWARD_CHUNK_SIZE=2` no ambiente para testes rápidos.


Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

Seu config no final vai ficar parecido com o meu:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan",
  "lora_type": "standard",
  "lycoris_config": "config/wan/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-disney",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "pretrained_t5_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "model_family": "wan",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "两只拟人化的猫咪身穿舒适的拳击装备，戴着鲜艳的手套，在聚光灯照射的舞台上激烈对战",
  "validation_negative_prompt": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
  "validation_guidance": 5.2,
  "validation_num_inference_steps": 40,
  "validation_num_video_frames": 81,
  "mixed_precision": "bf16",
  "optimizer": "optimi-lion",
  "learning_rate": 0.00005,
  "max_grad_norm": 0.01,
  "grad_clip_method": "value",
  "lr_scheduler": "cosine",
  "lr_warmup_steps": 400000,
  "base_model_precision": "no_change",
  "vae_batch_size": 1,
  "webhook_config": "config/wan/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "validation_guidance_skip_layers": [9],
  "validation_guidance_skip_layers_start": 0.0,
  "validation_guidance_skip_layers_stop": 1.0,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

De particular importância nessa configuração são as configurações de validação. Sem elas, as saídas não ficam muito boas.

### Opcional: regularizador temporal CREPA

Para movimento mais suave e menos deriva de identidade no Wan:
- Em **Training → Loss functions**, habilite **CREPA**.
- Comece com **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- O encoder padrão (`dinov2_vitg14`, tamanho `518`) funciona bem; troque para `dinov2_vits14` + `224` apenas se precisar reduzir VRAM.
- A primeira execução baixa DINOv2 via torch hub; faça cache ou pré-carregue se treinar offline.
- Só habilite **Drop VAE Encoder** quando treinar totalmente a partir de latentes em cache; caso contrário mantenha desativado para ainda codificar pixels.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz viés de exposição e melhora a qualidade ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam o overhead computacional do treinamento.

</details>

### Treino TREAD

> ⚠️ **Experimental**: TREAD é um recurso recém-implementado. Embora funcione, configurações ideais ainda estão sendo exploradas.

[TREAD](../TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. É um método que pode acelerar o treino do Flux ao rotear tokens de forma inteligente pelas camadas do transformer. A aceleração é proporcional a quantos tokens você descarta.

#### Configuração rápida

Adicione isto ao seu `config.json` para uma abordagem simples e conservadora para chegar a cerca de 5 segundos por step com bs=2 e 480p (reduzido de 10 segundos por step na velocidade padrão):

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.1,
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
- Potencialmente melhora a qualidade de treino e a convergência

Para o Wan 1.3B, podemos melhorar essa abordagem usando um setup de rotas progressivas em todas as 29 camadas e chegar a uma velocidade em torno de 7,7 segundos por step com bs=2 e 480p:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "tread_config": {
      "routes": [
          { "selection_ratio": 0.1, "start_layer_idx": 2, "end_layer_idx": 8 },
          { "selection_ratio": 0.25, "start_layer_idx": 9, "end_layer_idx": 11 },
          { "selection_ratio": 0.35, "start_layer_idx": 12, "end_layer_idx": 15 },
          { "selection_ratio": 0.25, "start_layer_idx": 16, "end_layer_idx": 23 },
          { "selection_ratio": 0.1, "start_layer_idx": 24, "end_layer_idx": -2 }
      ]
  }
}
```
</details>

Essa configuração tenta usar dropout de tokens mais agressivo nas camadas internas do modelo onde o conhecimento semântico não é tão importante.

Para alguns datasets, dropout mais agressivo pode ser tolerável, mas um valor de 0,5 é consideravelmente alto para o Wan 2.1.

#### Pontos-chave

- **Suporte de arquitetura limitado** - TREAD só está implementado para modelos Flux e Wan
- **Melhor em altas resoluções** - maiores acelerações em 1024x1024+ devido à complexidade O(n²) da atenção
- **Compatível com perda com máscara** - regiões mascaradas são preservadas automaticamente (mas isso reduz a aceleração)
- **Funciona com quantização** - pode ser combinado com treino int8/int4/NF4
- **Espere um pico de loss inicial** - ao iniciar treino LoRA/LoKr, a loss será mais alta no começo, mas corrige rapidamente

#### Dicas de ajuste

- **Conservador (foco em qualidade)**: use `selection_ratio` de 0.1-0.3
- **Agressivo (foco em velocidade)**: use `selection_ratio` de 0.3-0.5 e aceite o impacto na qualidade
- **Evite camadas iniciais/finais**: não roteie nas camadas 0-1 ou na camada final
- **Para treino LoRA**: pode haver pequenas desacelerações - experimente configs diferentes
- **Maior resolução = melhor aceleração**: mais benefício em 1024px ou mais

#### Comportamento conhecido

- Quanto mais tokens descartados (maior `selection_ratio`), mais rápido o treino, mas maior a loss inicial
- Treino LoRA/LoKr mostra um pico de loss inicial que corrige rapidamente conforme a rede se adapta
  - Usar configuração de treino menos agressiva ou múltiplas rotas com camadas internas mais altas alivia isso
- Algumas configurações de LoRA podem treinar um pouco mais devagar - configs ideais ainda estão sendo exploradas
- A implementação de RoPE (rotary position embedding) é funcional, mas talvez não esteja 100% correta

Para opções detalhadas de configuração e troubleshooting, veja a [documentação completa do TREAD](../TREAD.md).


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
</details>

> ℹ️ Wan 2.1 usa apenas o encoder de texto UMT5, que possui muita informação local nas embeddings, o que significa que prompts mais curtos podem não ter informação suficiente para o modelo fazer um bom trabalho. Certifique-se de usar prompts mais longos e descritivos.

#### Rastreamento de score CLIP

Isso não deve ser habilitado para treino de modelos de vídeo no momento.

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

Modelos de flow-matching como Flux, Sana, SD3, LTX Video e Wan 2.1 têm uma propriedade chamada `shift` que permite deslocar a parte treinada do schedule de timesteps usando um valor decimal simples.

##### Padrões
Por padrão, nenhum shift de schedule é aplicado, o que resulta em uma forma de sino sigmoide na distribuição de amostragem de timesteps, também conhecida como `logit_norm`.

##### Auto-shift
Uma abordagem comumente recomendada é seguir vários trabalhos recentes e habilitar o shift de timesteps dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em treinamentos estáveis, mas potencialmente medianos.

##### Especificação manual
_Agradecimentos ao General Awareness no Discord pelos exemplos a seguir_

> ℹ️ Esses exemplos mostram como o valor funciona usando Flux Dev, embora o Wan 2.1 deva ser bem semelhante.

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

#### Configurações de validação

Durante a exploração inicial para adicionar o Wan 2.1 ao SimpleTuner, saídas de pesadelo horríveis estavam aparecendo, e isso se resume a alguns motivos:

- Passos insuficientes na inferência
  - A menos que você use UniPC, provavelmente precisa de pelo menos 40 steps. UniPC pode reduzir um pouco, mas você terá que experimentar.
- Configuração incorreta do scheduler
  - Estava usando schedule Euler flow matching normal, mas a distribuição Betas parece funcionar melhor
  - Se você não mexeu nessa configuração, deve estar ok agora
- Resolução incorreta
  - Wan 2.1 só funciona corretamente nas resoluções em que foi treinado; às vezes dá sorte, mas é comum ter resultados ruins
- Valor de CFG ruim
  - O Wan 2.1 1.3B parece sensível a valores de CFG, mas algo em torno de 4.0-5.0 parece seguro
- Prompting ruim
  - Claro, modelos de vídeo parecem exigir um time de místicos para passar meses nas montanhas em um retiro zen e aprender a arte sagrada do prompting, porque seus datasets e o estilo de legenda são guardados como o Santo Graal.
  - tl;dr tente prompts diferentes.

Apesar de tudo isso, a menos que seu batch size esteja muito baixo e/ou sua taxa de aprendizado esteja muito alta, o modelo vai rodar corretamente na sua ferramenta de inferência favorita (assumindo que você já tenha uma que dê bons resultados).

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
    "cache_dir_vae": "cache/vae/wan/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

- Execuções I2V do Wan 2.2 criam caches de condicionamento CLIP. Na entrada **video** do dataset, aponte para um backend dedicado e (opcionalmente) sobrescreva o caminho do cache:

<details>
<summary>Ver exemplo de config</summary>

```json
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "conditioning_image_embeds": "disney-conditioning",
    "cache_dir_conditioning_image_embeds": "cache/conditioning_image_embeds/disney-black-and-white"
  }
```
</details>

- Defina o backend de condicionamento uma vez e reutilize em outros datasets se necessário (objeto completo mostrado aqui para clareza):

<details>
<summary>Ver exemplo de config</summary>

```json
  {
    "id": "disney-conditioning",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/disney-conditioning",
    "disabled": false
  }
```
</details>

- Na subseção `video`, temos as seguintes chaves que podemos definir:
  - `num_frames` (opcional, int) é quantos frames de dados vamos treinar.
    - A 15 fps, 75 frames equivalem a 5 segundos de vídeo, saída padrão. Este deve ser seu alvo.
  - `min_frames` (opcional, int) determina o comprimento mínimo de vídeo considerado para treino.
    - Isso deve ser pelo menos igual a `num_frames`. Não definir garante que será igual.
  - `max_frames` (opcional, int) determina o comprimento máximo de vídeo considerado para treino.
  - `bucket_strategy` (opcional, string) determina como os vídeos são agrupados em buckets:
    - `aspect_ratio` (padrão): agrupa apenas pela proporção espacial (ex.: `1.78`, `0.75`).
    - `resolution_frames`: agrupa por resolução e contagem de frames no formato `WxH@F` (ex.: `832x480@75`). Útil para datasets com resolução/duração mistas.
  - `frame_interval` (opcional, int) ao usar `resolution_frames`, arredonde contagens de frames para este intervalo.
<!--  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.
    - This is set to True by default for Wan 2.1. You can disable it, however.
-->

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
pip install 'simpletuner[cuda13]'
simpletuner train
```

**Opção 2 (Método Git clone):**
```bash
simpletuner train
```

> ℹ️ Acrescente `--model_flavour i2v-14b-2.2-high` (ou `low`) e, se desejar, `--wan_validation_load_other_stage` dentro de `TRAINER_EXTRA_ARGS` ou na sua chamada CLI ao treinar Wan 2.2. Adicione `--wan_force_2_1_time_embedding` apenas quando o checkpoint reportar incompatibilidade de shape no time embedding.

**Opção 3 (Método legado - ainda funciona):**
```bash
./train.sh
```

Isso vai iniciar o cache em disco das embeddings de texto e saídas do VAE.

Para mais informações, veja os documentos do [dataloader](../DATALOADER.md) e do [tutorial](../TUTORIAL.md).

## Notas e dicas de troubleshooting

### Configuração de menor VRAM

Wan 2.1 é sensível à quantização e não pode ser usado com NF4 ou INT4 atualmente.

- SO: Ubuntu Linux 24
- GPU: um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 12G de memória do sistema
- Precisão do modelo base: `int8-quanto`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 480px
- Tamanho de lote: 1, zero passos de acumulação de gradiente
- DeepSpeed: desativado / não configurado
- PyTorch: 2.6
- Garanta que `--gradient_checkpointing` esteja habilitado ou nada vai impedir OOM
- Treine apenas em imagens, ou defina `num_frames` como 1 no seu dataset de vídeo

**NOTA**: O pré-cache de embeddings do VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. Como resultado, `--offload_during_startup=true` é basicamente obrigatório. Se necessário, a quantização do encoder de texto e o VAE tiling podem ser habilitados. (Wan não suporta VAE tiling/slicing no momento)

Velocidades:
- 665,8 s/iter em um Macbook Pro M3 Max
- 2 s/iter em uma NVIDIA 4090 com batch size 1
- 11 s/iter em NVIDIA 4090 com batch size 4

### SageAttention

Ao usar `--attention_mechanism=sageattention`, a inferência pode ficar mais rápida durante a validação.

**Nota**: Isso não é compatível com o passo final de decode do VAE e não acelera essa parte.

### Perda com máscara

Não use isso com Wan 2.1.

### Quantização
- Quantização não é necessária para treinar este modelo em 24G

### Artefatos de imagem

Wan exige o schedule de flow-matching Euler Betas ou (por padrão) o solver multistep UniPC, um scheduler de ordem superior que fará previsões mais fortes.

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

### Treinando modelos Wan 2.1 fine-tuned customizados

Alguns modelos fine-tuned no Hugging Face Hub não têm a estrutura completa de diretórios, exigindo que opções específicas sejam definidas.

<details>
<summary>Ver exemplo de config</summary>

```json
{
    "model_family": "wan",
    "pretrained_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_model_name_or_path": "path/to-the-other-model",
    "pretrained_vae_model_name_or_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "pretrained_transformer_subfolder": "none",
}
```
</details>

> Nota: Você pode fornecer um caminho para um único arquivo `.safetensors` para `pretrained_transformer_name_or_path`
