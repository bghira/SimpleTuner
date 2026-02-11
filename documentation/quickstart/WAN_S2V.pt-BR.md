## Guia rápido do Wan 2.2 S2V

Neste exemplo, vamos treinar uma LoRA Wan 2.2 S2V (Speech-to-Video). Modelos S2V geram vídeo condicionado por entrada de áudio, permitindo geração de vídeo guiada por áudio.

### Requisitos de hardware

Wan 2.2 S2V **14B** é um modelo exigente que requer muita memória de GPU.

#### Speech to Video

14B - https://huggingface.co/tolgacangoz/Wan2.2-S2V-14B-Diffusers
- Resolução: 832x480
- Cabe em 24G, mas você vai precisar ajustar um pouco as configurações.

Você vai precisar:
- **um mínimo realista** é 24GB ou uma única GPU 4090 ou A6000
- **idealmente** várias 4090, A6000, L40S ou melhores

Sistemas Apple silicon não funcionam tão bem com Wan 2.2 até agora; algo como 10 minutos para um único passo de treinamento pode ser esperado.

### Pré-requisitos

Certifique-se de ter o Python instalado; o SimpleTuner funciona bem do 3.10 ao 3.12.

Você pode verificar isso executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

```bash
apt -y install python3.13 python3.13-venv
```

#### Dependências de imagem de contêiner

Para Vast, RunPod e TensorDock (entre outros), o seguinte funciona em uma imagem CUDA 12.2-12.8 para permitir a compilação de extensões CUDA:

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

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](/documentation/INSTALL.md).
#### SageAttention 2

Se você quiser usar o SageAttention 2, alguns passos devem ser seguidos.

> Nota: SageAttention oferece ganho de velocidade mínimo, não é muito efetivo; não sei por quê. Testado em 4090.

Execute o seguinte enquanto ainda estiver dentro do seu venv de Python:
```bash
git clone https://github.com/thu-ml/SageAttention
pushd SageAttention
  pip install . --no-build-isolation
popd
```

#### Passos adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

### Configurando o ambiente

Para executar o SimpleTuner, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule totalmente esta seção com uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não configura o seu dataloader. Você ainda terá que fazer isso manualmente mais tarde.

Para executá-lo:

```bash
simpletuner configure
```

> Para usuários em países onde o Hugging Face Hub não é facilmente acessível, você deve adicionar `HF_ENDPOINT=https://hf-mirror.com` ao seu `~/.bashrc` ou `~/.zshrc`, dependendo de qual `$SHELL` seu sistema usa.

### Offloading de memória (opcional)

Wan é um dos modelos mais pesados que o SimpleTuner suporta. Ative o offloading em grupo se você estiver perto do limite de VRAM:

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- Somente dispositivos CUDA respeitam `--group_offload_use_stream`; ROCm/MPS fazem fallback automaticamente.
- Deixe o staging em disco comentado, a menos que a memória da CPU seja o gargalo.
- `--enable_model_cpu_offload` é mutuamente exclusivo com offload em grupo.

### Chunking de feed-forward (opcional)

Se os checkpoints 14B ainda derem OOM durante o gradient checkpointing, quebre as camadas feed-forward do Wan:

```bash
--enable_chunked_feed_forward \
--feed_forward_chunk_size 2 \
```

Isso corresponde ao novo toggle no assistente de configuração (`Training -> Memory Optimisation`). Tamanhos menores de chunk economizam mais memória, mas deixam cada passo mais lento. Você também pode definir `WAN_FEED_FORWARD_CHUNK_SIZE=2` no seu ambiente para experimentos rápidos.


Se você preferir configurar manualmente:

Copie `config/config.json.example` para `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Usuários multi-GPU podem consultar [este documento](/documentation/OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

Sua configuração no final ficará parecida com a minha:

<details>
<summary>View example config</summary>

```json
{
  "resume_from_checkpoint": "latest",
  "quantize_via": "cpu",
  "attention_mechanism": "sageattention",
  "data_backend_config": "config/wan_s2v/multidatabackend.json",
  "aspect_bucket_rounding": 2,
  "seed": 42,
  "minimum_image_size": 0,
  "offload_during_startup": true,
  "disable_benchmark": false,
  "output_dir": "output/wan_s2v",
  "lora_type": "standard",
  "lycoris_config": "config/wan_s2v/lycoris_config.json",
  "max_train_steps": 400000,
  "num_train_epochs": 0,
  "checkpoint_step_interval": 1000,
  "checkpoints_total_limit": 5,
  "hub_model_id": "wan-s2v-lora",
  "push_to_hub": "true",
  "push_checkpoints_to_hub": "true",
  "tracker_project_name": "lora-training",
  "tracker_run_name": "wan-s2v-adamW",
  "report_to": "wandb",
  "model_type": "lora",
  "pretrained_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "pretrained_t5_model_name_or_path": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
  "model_family": "wan_s2v",
  "model_flavour": "s2v-14b-2.2",
  "train_batch_size": 2,
  "gradient_checkpointing": true,
  "gradient_accumulation_steps": 1,
  "caption_dropout_probability": 0.1,
  "resolution_type": "pixel_area",
  "resolution": 480,
  "validation_seed": 42,
  "validation_step_interval": 100,
  "validation_resolution": "832x480",
  "validation_prompt": "A person speaking with natural gestures",
  "validation_negative_prompt": "blurry, low quality, distorted",
  "validation_guidance": 4.5,
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
  "webhook_config": "config/wan_s2v/webhook.json",
  "compress_disk_cache": true,
  "use_ema": true,
  "ema_validation": "ema_only",
  "ema_update_interval": 2,
  "delete_problematic_images": "true",
  "disable_bucket_pruning": true,
  "lora_rank": 16,
  "flow_schedule_shift": 3,
  "validation_prompt_library": false,
  "ignore_final_epochs": true
}
```
</details>

De particular importância nesta configuração são os ajustes de validação. Sem eles, os resultados não ficam muito bons.

### Opcional: regularizador temporal CREPA

Para movimento mais suave e menos desvio de identidade no Wan S2V:
- Em **Training -> Loss functions**, habilite **CREPA**.
- Comece com **Block Index = 8**, **Weight = 0.5**, **Adjacent Distance = 1**, **Temporal Decay = 1.0**.
- O encoder padrão (`dinov2_vitg14`, tamanho `518`) funciona bem; troque para `dinov2_vits14` + `224` apenas se precisar reduzir VRAM.
- A primeira execução baixa o DINOv2 via torch hub; faça cache ou prefetch se treinar offline.
- Só habilite **Drop VAE Encoder** quando treinar inteiramente a partir de latentes em cache; caso contrário, deixe desativado para que a codificação de pixels ainda funcione.

### Recursos experimentais avançados

<details>
<summary>Show advanced experimental details</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](/documentation/experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao permitir que o modelo gere suas próprias entradas durante o treinamento.

> Esses recursos aumentam o overhead computacional do treinamento.

</details>

### Treinamento TREAD

> **Experimental**: TREAD é um recurso implementado recentemente. Embora funcional, as configurações ideais ainda estão sendo exploradas.

[TREAD](/documentation/TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. É um método que pode acelerar o treinamento do Wan S2V ao rotear tokens de forma inteligente pelas camadas do transformer. O ganho de velocidade é proporcional a quantos tokens você descarta.

#### Configuração rápida

Adicione isso ao seu `config.json` para uma abordagem simples e conservadora:

<details>
<summary>View example config</summary>

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

Essa configuração irá:
- Manter apenas 50% dos tokens de imagem durante as camadas 2 até a penúltima
- Tokens de texto nunca são descartados
- Aceleração de treino de ~25% com impacto mínimo na qualidade
- Potencialmente melhora a qualidade do treinamento e a convergência

#### Pontos-chave

- **Suporte de arquitetura limitado** - TREAD só está implementado para modelos Flux e Wan (incluindo S2V)
- **Melhor em altas resoluções** - Maiores ganhos em 1024x1024+ devido à complexidade O(n^2) da atenção
- **Compatível com perda mascarada** - Regiões mascaradas são preservadas automaticamente (mas isso reduz o ganho)
- **Funciona com quantização** - Pode ser combinado com treino int8/int4/NF4
- **Espere um pico inicial de loss** - Ao iniciar treino LoRA/LoKr, a loss será maior no início mas se corrige rapidamente

#### Dicas de ajuste

- **Conservador (foco em qualidade)**: Use `selection_ratio` de 0.1-0.3
- **Agressivo (foco em velocidade)**: Use `selection_ratio` de 0.3-0.5 e aceite o impacto na qualidade
- **Evite camadas iniciais/finais**: Não faça routing nas camadas 0-1 ou na última camada
- **Para treino LoRA**: Pode haver pequenas lentidões - experimente diferentes configs
- **Maior resolução = melhor ganho**: Mais benefício em 1024px e acima

Para opções detalhadas de configuração e troubleshooting, veja a [documentação completa do TREAD](/documentation/TREAD.md).


#### Prompts de validação

Dentro de `config/config.json` está o "primary validation prompt", que normalmente é o instance_prompt principal em que você está treinando para seu único sujeito ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para executar durante as validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

<details>
<summary>View example config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Os nicknames são o nome do arquivo para a validação, então mantenha-os curtos e compatíveis com o seu sistema de arquivos.

Para apontar o trainer para essa biblioteca de prompts, adicione-a ao TRAINER_EXTRA_ARGS incluindo uma nova linha no fim de `config.json`:
<details>
<summary>View example config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

> S2V usa o encoder de texto UMT5, que tem muita informação local em seus embeddings, o que significa que prompts curtos podem não ter informação suficiente para o modelo fazer um bom trabalho. Use prompts mais longos e descritivos.

#### Rastreamento de pontuação CLIP

Isso não deve ser habilitado para treinamento de modelos de vídeo, no momento.

# Loss de avaliação estável

Se você quiser usar loss MSE estável para avaliar o desempenho do modelo, veja [este documento](/documentation/evaluation/EVAL_LOSS.md) para informações sobre como configurar e interpretar a loss de avaliação.

#### Pré-visualizações de validação

O SimpleTuner suporta pré-visualizações intermediárias de validação em streaming durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
<details>
<summary>View example config</summary>

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

Defina `validation_preview_steps` para um valor maior (por exemplo, 3 ou 5) para reduzir o overhead do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de preview nos passos 5, 10, 15 e 20.

#### Deslocamento de cronograma de flow-matching

Modelos de flow-matching como Flux, Sana, SD3, LTX Video e Wan S2V têm uma propriedade chamada `shift` que permite deslocar a parte treinada do cronograma de timesteps usando um valor decimal simples.

##### Padrões
Por padrão, nenhum deslocamento de cronograma é aplicado, o que resulta em um formato de sino sigmoide na distribuição de amostragem de timesteps, também conhecido como `logit_norm`.

##### Auto-shift
Uma abordagem recomendada é seguir vários trabalhos recentes e habilitar o deslocamento de timesteps dependente da resolução, `--flow_schedule_auto_shift`, que usa valores de deslocamento maiores para imagens maiores e valores menores para imagens menores. Isso resulta em resultados estáveis, mas potencialmente medianos.

##### Especificação manual
_Agradecimentos ao General Awareness do Discord pelos exemplos a seguir_

> Esses exemplos mostram como o valor funciona usando Flux Dev, embora Wan S2V deva ser muito similar.

Ao usar um valor de `--flow_schedule_shift` de 0.1 (muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (muito alto), os grandes elementos de composição e possivelmente o espaço de cor do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)


#### Treinamento de modelo quantizado

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisão e os requisitos de VRAM, treinando com apenas 16GB.



Para usuários de `config.json`:
<details>
<summary>View example config</summary>

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

Durante a exploração inicial, a baixa qualidade de saída pode vir do Wan S2V, e isso se resume a alguns motivos:

- Não há passos suficientes para inferência
  - A menos que você esteja usando UniPC, provavelmente precisa de pelo menos 40 passos. O UniPC pode reduzir um pouco esse número, mas você terá que experimentar.
- Configuração incorreta do scheduler
  - Estava usando o cronograma normal de Euler flow matching, mas a distribuição Betas parece funcionar melhor
  - Se você não mexeu nessa configuração, deve estar tudo certo agora
- Resolução incorreta
  - Wan S2V só funciona corretamente nas resoluções em que foi treinado, você dá sorte se funcionar, mas é comum ter resultados ruins
- Valor CFG ruim
  - Um valor em torno de 4.0-5.0 parece seguro
- Prompts ruins
  - Claro, modelos de vídeo parecem exigir uma equipe de místicos que passem meses nas montanhas em um retiro zen para aprender a arte sagrada do prompting, porque seus datasets e estilo de captions são guardados como o Santo Graal.
  - tl;dr experimente prompts diferentes.
- Áudio ausente ou incompatível
  - S2V requer entrada de áudio para validação - garanta que suas amostras de validação tenham arquivos de áudio correspondentes

Apesar de tudo isso, a menos que seu batch size seja muito baixo e/ou sua learning rate seja muito alta, o modelo rodará corretamente na sua ferramenta de inferência favorita (assumindo que você já tenha uma com bons resultados).

#### Considerações sobre o dataset

O treinamento S2V requer dados de vídeo e áudio pareados. Por padrão, o SimpleTuner faz auto-split do áudio a partir
dos datasets de vídeo, então você não precisa de um dataset de áudio separado a menos que queira processamento
personalizado. Use `audio.auto_split: false` para desativar e defina `s2v_datasets` manualmente.

Há poucas limitações no tamanho do dataset além de quanto compute e tempo levará para processar e treinar.

Você deve garantir que o dataset seja grande o suficiente para treinar seu modelo de forma eficaz, mas não tão grande para a quantidade de compute que você tem disponível.

Observe que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps` e também maior que `vae_batch_size`. O dataset não será utilizável se for muito pequeno.

> Com poucas amostras, você pode ver uma mensagem **no samples detected in dataset** - aumentar o valor de `repeats` resolverá essa limitação.

#### Configuração do dataset de áudio

##### Extração automática de áudio de vídeos (Recomendado)

Se seus vídeos já contêm faixas de áudio, o SimpleTuner pode extrair e processar áudio automaticamente sem exigir um dataset de áudio separado. Essa é a abordagem mais simples e a padrão:

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 75,
        "min_frames": 75,
        "bucket_strategy": "aspect_ratio"
    },
    "audio": {
        "auto_split": true,
        "sample_rate": 16000,
        "channels": 1
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

Com o auto-split de áudio habilitado (padrão), o SimpleTuner irá:
1. Gerar automaticamente uma configuração de dataset de áudio (`s2v-videos_audio`)
2. Extrair áudio de cada vídeo durante a descoberta de metadados
3. Fazer cache dos latentes de áudio VAE em um diretório dedicado
4. Vincular automaticamente o dataset de áudio via `s2v_datasets`

**Opções de configuração de áudio:**
- `audio.auto_split` (bool): Habilita a extração automática de áudio de vídeos (padrão: true)
- `audio.sample_rate` (int): Taxa de amostragem alvo em Hz (padrão: 16000 para Wav2Vec2)
- `audio.channels` (int): Número de canais de áudio (padrão: 1 para mono)
- `audio.allow_zero_audio` (bool): Gera áudio preenchido com zeros para vídeos sem streams de áudio (padrão: false)
- `audio.max_duration_seconds` (float): Duração máxima do áudio; arquivos mais longos são ignorados
- `audio.duration_interval` (float): Intervalo de duração para agrupamento em buckets em segundos (padrão: 3.0)
- `audio.truncation_mode` (string): Como truncar áudio longo: "beginning", "end", "random" (padrão: "beginning")

**Nota**: Vídeos sem faixas de áudio são automaticamente ignorados no treinamento S2V, a menos que `audio.allow_zero_audio: true` esteja definido.

##### Dataset de áudio manual (Alternativa)

Se você preferir arquivos de áudio separados, precisar de processamento de áudio personalizado ou desativar o auto-split,
os modelos S2V também podem usar arquivos de áudio pré-extraídos que correspondam aos seus arquivos de vídeo pelo nome do arquivo. Por exemplo:
- `video_001.mp4` deve ter um correspondente `video_001.wav` (ou `.mp3`, `.flac`, `.ogg`, `.m4a`)

Os arquivos de áudio devem estar em um diretório separado que você configurará como um backend `s2v_datasets`.

##### Extração de áudio de vídeos (Manual)

Se seus vídeos já contêm áudio, use o script fornecido para extraí-lo:

```bash
# Extract audio only (keeps original videos unchanged)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio

# Extract audio and remove it from source videos (recommended to avoid redundant data)
python scripts/generate_s2v_audio.py \
    --input-dir datasets/s2v-videos \
    --output-dir datasets/s2v-audio \
    --strip-audio
```

O script:
- Extrai áudio em WAV mono de 16kHz (taxa nativa do Wav2Vec2)
- Faz correspondência automática de nomes de arquivo (ex., `video.mp4` -> `video.wav`)
- Ignora vídeos sem streams de áudio
- Requer `ffmpeg` instalado

##### Configuração do dataset (Manual)

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo isto:

<details>
<summary>View example config</summary>

```json
[
  {
    "id": "s2v-videos",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/wan_s2v/videos",
    "instance_data_dir": "datasets/s2v-videos",
    "s2v_datasets": ["s2v-audio"],
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
    "id": "s2v-audio",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/s2v-audio",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/wan_s2v",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Pontos-chave para a configuração de dataset S2V:
- O campo `s2v_datasets` no seu dataset de vídeo aponta para o(s) backend(s) de áudio
- Arquivos de áudio são correspondidos pelo stem do nome do arquivo (ex., `video_001.mp4` corresponde a `video_001.wav`)
- O áudio é codificado on-the-fly usando Wav2Vec2 (~600MB de VRAM), sem necessidade de cache
- O tipo de dataset de áudio é `audio`

- Na subseção `video`, temos as seguintes chaves que podemos definir:
  - `num_frames` (opcional, int) é quantos frames de dados vamos treinar.
    - A 15 fps, 75 frames são 5 segundos de vídeo, saída padrão. Esse deve ser seu alvo.
  - `min_frames` (opcional, int) determina o comprimento mínimo de um vídeo que será considerado para treinamento.
    - Deve ser pelo menos igual a `num_frames`. Não definir garante que ficará igual.
  - `max_frames` (opcional, int) determina o comprimento máximo de um vídeo que será considerado para treinamento.
  - `bucket_strategy` (opcional, string) determina como os vídeos são agrupados em buckets:
    - `aspect_ratio` (padrão): Agrupa apenas pela proporção espacial (ex., `1.78`, `0.75`).
    - `resolution_frames`: Agrupa por resolução e contagem de frames no formato `WxH@F` (ex., `832x480@75`). Útil para datasets com resolução/duração mistas.
  - `frame_interval` (opcional, int) quando usar `resolution_frames`, arredonda a contagem de frames para esse intervalo.

Depois, crie um diretório `datasets` com seus arquivos de vídeo e áudio:

```bash
mkdir -p datasets/s2v-videos datasets/s2v-audio
# Place your video files in datasets/s2v-videos/
# Place your audio files in datasets/s2v-audio/
```

Garanta que cada vídeo tenha um arquivo de áudio correspondente pelo stem do nome do arquivo.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de começar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você for fazer push manual de itens para um repositório Git LFS, também deve rodar `git config --global credential.helper store`

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login em ambos os serviços.

### Execução do treinamento

A partir do diretório do SimpleTuner, você tem várias opções para iniciar o treinamento:

**Opção 1 (Recomendado - pip install):**
```bash
pip install 'simpletuner[cuda]'

# CUDA 13 / Blackwell users (NVIDIA B-series GPUs)
pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
simpletuner train
```

**Opção 2 (Método git clone):**
```bash
simpletuner train
```

**Opção 3 (Método legado - ainda funciona):**
```bash
./train.sh
```

Isso iniciará o cache em disco dos embeddings de texto e das saídas do VAE.

Para mais informações, veja os documentos [dataloader](/documentation/DATALOADER.md) e [tutorial](/documentation/TUTORIAL.md).

## Notas e dicas de troubleshooting

### Configuração de VRAM mínima

Wan S2V é sensível à quantização e não pode ser usado com NF4 ou INT4 atualmente.

- OS: Ubuntu Linux 24
- GPU: Um único dispositivo NVIDIA CUDA (24G recomendado)
- Memória do sistema: aproximadamente 16G de memória do sistema
- Precisão do modelo base: `int8-quanto`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 480px
- Batch size: 1, zero passos de acumulação de gradiente
- DeepSpeed: desabilitado / não configurado
- PyTorch: 2.6
- Garanta que `--gradient_checkpointing` esteja ativado ou nada evitará o OOM
- Treine apenas em imagens, ou defina `num_frames` como 1 para o seu dataset de vídeo

**NOTA**: O pré-cache de embeddings VAE e saídas do text encoder pode usar mais memória e ainda assim dar OOM. Como resultado, `--offload_during_startup=true` é basicamente obrigatório. Se for o caso, quantização do text encoder e tiling de VAE podem ser habilitados. (Wan não suporta VAE tiling/slicing atualmente)

### SageAttention

Ao usar `--attention_mechanism=sageattention`, a inferência pode ser acelerada durante a validação.

**Nota**: Isso não é compatível com a etapa final de decodificação do VAE e não acelerará essa parte.

### Perda mascarada

Não use isso com Wan S2V.

### Quantização
- A quantização pode ser necessária para treinar este modelo em 24G dependendo do batch size

### Artefatos de imagem
Wan exige o uso do cronograma de flow-matching Euler Betas ou (por padrão) o solver multistep UniPC, um scheduler de ordem superior que fará previsões mais fortes.

Como outros modelos DiT, se você fizer estas coisas (entre outras), alguns artefatos de grade quadrada **podem** começar a aparecer nas amostras:
- Overtrain com dados de baixa qualidade
- Usar uma taxa de aprendizado muito alta
- Overtraining (em geral), uma rede de baixa capacidade com muitas imagens
- Undertraining (também), uma rede de alta capacidade com poucas imagens
- Usar proporções estranhas ou tamanhos de dados de treinamento

### Bucketização de aspecto
- Vídeos são agrupados como imagens.
- Treinar por muito tempo com recortes quadrados provavelmente não prejudica muito este modelo. Vá fundo, ele é ótimo e confiável.
- Por outro lado, usar os buckets de aspecto naturais do seu dataset pode enviesar demais essas formas durante a inferência.
  - Isso pode ser uma qualidade desejável, pois evita que estilos dependentes de aspecto como o cinematográfico vazem para outras resoluções.
  - No entanto, se você busca melhorar resultados igualmente em muitos buckets de aspecto, talvez precise experimentar `crop_aspect=random`, que tem seus próprios contras.
- Misturar configurações de dataset definindo seu diretório de imagens várias vezes produziu resultados muito bons e um modelo bem generalizado.

### Sincronização de áudio

Para melhores resultados com S2V:
- Garanta que a duração do áudio corresponda à duração do vídeo
- O áudio é reamostrado internamente para 16kHz
- O encoder Wav2Vec2 processa áudio on-the-fly (~600MB de overhead de VRAM)
- As features de áudio são interpoladas para corresponder ao número de frames do vídeo
