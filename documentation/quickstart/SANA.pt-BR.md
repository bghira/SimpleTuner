## Guia de Início Rápido do NVLabs Sana

Neste exemplo, vamos treinar o modelo NVLabs Sana em full-rank.

### Requisitos de hardware

O Sana é muito leve e talvez nem precise de gradient checkpointing completo habilitado em uma placa de 24G, o que significa que ele treina muito rápido!

- **o mínimo absoluto** é cerca de 12G de VRAM, embora este guia talvez não ajude você a chegar lá completamente
- **um mínimo realista** é uma única GPU 3090 ou V100
- **idealmente** múltiplas 4090, A6000, L40S ou melhor

O Sana é uma arquitetura estranha em relação a outros modelos que podem ser treinados pelo SimpleTuner;

- Inicialmente, ao contrário de outros modelos, o Sana exigia treinamento em fp16 e travava com bf16
  - Os autores do modelo na NVIDIA foram gentis em fornecer pesos compatíveis com bf16 para fine-tuning
- A quantização pode ser mais sensível nessa família de modelos devido aos problemas com bf16/fp16
- SageAttention não funciona com Sana (ainda) devido ao formato de head_dim atualmente não suportado
- O valor de loss ao treinar Sana é muito alto, e pode precisar de uma taxa de aprendizado bem menor do que outros modelos (ex.: `1e-5` ou algo assim)
- O treinamento pode atingir valores NaN, e não está claro por que isso acontece

O gradient checkpointing pode liberar VRAM, mas desacelera o treinamento. Um gráfico de resultados de teste de uma 4090 com 5800X3D:

![image](https://github.com/user-attachments/assets/310bf099-a077-4378-acf4-f60b4b82fdc4)

O código de modelagem do Sana no SimpleTuner permite especificar `--gradient_checkpointing_interval` para fazer checkpoint a cada _n_ blocos e obter os resultados vistos no gráfico acima.

### Pré-requisitos

Certifique-se de que você tenha Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

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

**Nota:** Isso não configura seu dataloader. Você ainda terá que fazer isso manualmente, mais tarde.

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

Lá, você possivelmente precisará modificar as seguintes variáveis:

- `model_type` - Defina como `full`.
- `model_family` - Defina como `sana`.
- `pretrained_model_name_or_path` - Defina como `terminusresearch/sana-1.6b-1024px`
- `output_dir` - Defina o diretório onde deseja armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `train_batch_size` - Para uma placa 24G com gradient checkpointing completo, pode ser até 6.
- `validation_resolution` - Este checkpoint para Sana é um modelo 1024px, então defina como `1024x1024` ou uma das outras resoluções suportadas pelo Sana.
  - Outras resoluções podem ser especificadas separando por vírgulas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Use qualquer valor que você costuma escolher em inferência para Sana.
- `validation_num_inference_steps` - Use algo em torno de 50 para a melhor qualidade, embora você possa aceitar menos se estiver satisfeito com os resultados.
- `use_ema` - Definir como `true` ajuda muito a obter um resultado mais suavizado junto com o seu checkpoint principal.

- `optimizer` - Você pode usar qualquer otimizador que conheça e com o qual se sinta confortável, mas usaremos `optimi-adamw` neste exemplo.
- `mixed_precision` - É recomendado definir como `bf16` para a configuração de treinamento mais eficiente, ou `no` (mas consumirá mais memória e será mais lento).
  - Um valor de `fp16` não é recomendado aqui, mas pode ser necessário para certos fine-tunes do Sana (e introduz outros problemas para habilitar isso)
- `gradient_checkpointing` - Desativar isso será o mais rápido, mas limita seus tamanhos de batch. É necessário habilitar isso para obter o menor uso de VRAM.
- `gradient_checkpointing_interval` - Se `gradient_checkpointing` parece exagero para sua GPU, você pode definir um valor 2 ou maior para fazer checkpoint apenas a cada _n_ blocos. Um valor 2 faria checkpoint de metade dos blocos, e 3 de um terço.

Usuários multi-GPU podem consultar [este documento](../OPTIONS.md#environment-configuration-variables) para informações sobre como configurar o número de GPUs a usar.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

#### Prompts de validação

Dentro de `config/config.json` está o "prompt de validação primário", que normalmente é o principal instance_prompt no qual você está treinando para seu único assunto ou estilo. Além disso, um arquivo JSON pode ser criado contendo prompts extras para executar durante as validações.

O arquivo de exemplo `config/user_prompt_library.json.example` contém o seguinte formato:

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```

Os nicknames são o nome do arquivo de validação, então mantenha-os curtos e compatíveis com seu sistema de arquivos.

Para apontar o treinador para essa biblioteca de prompts, adicione-a ao TRAINER_EXTRA_ARGS com uma nova linha no final do `config.json`:
```json
  "--user_prompt_library": "config/user_prompt_library.json",
```

Um conjunto de prompts diversificados ajuda a determinar se o modelo está colapsando conforme treina. Neste exemplo, a palavra `<token>` deve ser substituída pelo nome do seu assunto (instance_prompt).

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

> ℹ️ O Sana usa uma configuração estranha de text encoder que significa que prompts curtos possivelmente ficarão muito ruins.

#### Acompanhamento de CLIP score

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre configuração e interpretação de CLIP scores.

</details>

# Perda de avaliação estável

Se você quiser usar perda MSE estável para pontuar o desempenho do modelo, veja [este documento](../evaluation/EVAL_LOSS.md) para informações sobre configuração e interpretação de perda de avaliação.

#### Prévias de validação

O SimpleTuner suporta streaming de prévias intermediárias de validação durante a geração usando modelos Tiny AutoEncoder. Isso permite ver imagens de validação sendo geradas passo a passo em tempo real via callbacks de webhook.

Para habilitar:
<details>
<summary>Ver exemplo de configuração</summary>

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

Defina `validation_preview_steps` para um valor maior (ex.: 3 ou 5) para reduzir a sobrecarga do Tiny AutoEncoder. Com `validation_num_inference_steps=20` e `validation_preview_steps=5`, você receberá imagens de prévia nos steps 5, 10, 15 e 20.

#### Shift do cronograma de tempo do Sana

Modelos flow-matching como Sana, Flux e SD3 têm uma propriedade chamada "shift" que permite deslocar a porção treinada do cronograma de timesteps usando um valor decimal simples.

##### Auto-shift

Uma abordagem comumente recomendada é seguir vários trabalhos recentes e habilitar shift de timestep dependente de resolução, `--flow_schedule_auto_shift`, que usa valores de shift maiores para imagens maiores e menores para imagens menores. Isso resulta em resultados de treinamento estáveis, porém potencialmente medíocres.

##### Especificação manual

_Agradecimentos a General Awareness do Discord pelos exemplos a seguir_

Ao usar um valor de `--flow_schedule_shift` de 0.1 (um valor muito baixo), apenas os detalhes mais finos da imagem são afetados:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Ao usar um valor de `--flow_schedule_shift` de 4.0 (um valor muito alto), as grandes características composicionais e potencialmente o espaço de cores do modelo são impactados:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Considerações sobre o dataset

> ⚠️ A qualidade das imagens para treinamento é mais importante para o Sana do que para a maioria dos outros modelos, pois ele vai absorver os artefatos nas suas imagens *primeiro*, e depois aprender o conceito/assunto.

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisará garantir que seu dataset seja grande o suficiente para treinar seu modelo efetivamente. Observe que o tamanho mínimo do dataset é `train_batch_size * gradient_accumulation_steps`, além de ser maior que `vae_batch_size`. O dataset não será utilizável se for muito pequeno.

> ℹ️ Com poucas imagens, você pode ver a mensagem **no images detected in dataset** - aumentar o valor de `repeats` supera essa limitação.

Dependendo do dataset que você tem, será necessário configurar o diretório do dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crie um documento `--data_backend_config` (`config/multidatabackend.json`) contendo:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sana",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/sana/pseudo-camera-10k",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject",
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
    "cache_dir_vae": "cache/vae/sana/dreambooth-subject-512",
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
    "cache_dir": "cache/text/sana",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ Rodar datasets 512px e 1024px simultaneamente é suportado e pode resultar em melhor convergência para o Sana.

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # coloque suas imagens em dreambooth-subject/ agora
popd
```

Isso vai baixar cerca de 10k amostras de fotos para o seu diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

Suas imagens Dreambooth devem ir para o diretório `datasets/dreambooth-subject`.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `--push_to_hub` e `--report_to=wandb`.

Se você vai enviar itens para um repositório Git LFS manualmente, também deve executar `git config --global credential.helper store`

Execute os seguintes comandos:

```bash
wandb login
```

e

```bash
huggingface-cli login
```

Siga as instruções para fazer login nos dois serviços.

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

Isso vai iniciar o cache de text embeds e saídas VAE em disco.

Para mais informações, veja os documentos [dataloader](../DATALOADER.md) e [tutorial](../TUTORIAL.md).

## Notas e dicas de solução de problemas

### Configuração de VRAM mais baixa

Atualmente, a menor utilização de VRAM pode ser obtida com:

- SO: Ubuntu Linux 24
- GPU: Um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G de memória do sistema
- Precisão do modelo base: `nf4-bnb`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 1024px
- Batch size: 1, zero steps de gradient accumulation
- DeepSpeed: desativado / não configurado
- PyTorch: 2.5.1
- Usando `--quantize_via=cpu` para evitar erro outOfMemory durante a inicialização em placas <=16G.
- Habilitar `--gradient_checkpointing`

**NOTA**: O pré-cache de embeddings VAE e saídas do text encoder pode usar mais memória e ainda dar OOM. Se isso acontecer, a quantização do text encoder pode ser habilitada. VAE tiling pode não funcionar para o Sana no momento. Para datasets grandes em que espaço em disco é uma preocupação, você pode usar `--vae_cache_disable` para fazer codificação online sem cache em disco.

A velocidade foi de aproximadamente 1.4 iterações por segundo em uma 4090.

### Loss mascarada

Se você estiver treinando um assunto ou estilo e quiser mascarar um ou outro, veja a seção de [treinamento com loss mascarada](../DREAMBOOTH.md#masked-loss) no guia de Dreambooth.

### Quantização

Não testado completamente (ainda).

### Taxas de aprendizado

#### LoRA (--lora_type=standard)

*Não suportado.*

#### LoKr (--lora_type=lycoris)
- Taxas de aprendizado moderadas são melhores para LoKr (`1e-4` com AdamW, `2e-5` com Lion)
- Outros algos precisam de mais exploração.
- Definir `is_regularisation_data` tem impacto/efeito desconhecido com Sana (não testado)

### Artefatos de imagem

O Sana tem uma resposta desconhecida a artefatos de imagem.

No momento, não se sabe se algum artefato comum de treinamento será produzido ou qual a causa disso.

Se surgirem problemas de qualidade de imagem, abra uma issue no Github.

### Bucketing de aspecto

Este modelo tem uma resposta desconhecida a dados com buckets de aspecto. Experimentação será útil.
