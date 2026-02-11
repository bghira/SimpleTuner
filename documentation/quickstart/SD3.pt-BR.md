## Stable Diffusion 3

Neste exemplo, vamos treinar um modelo Stable Diffusion 3 usando o toolkit SimpleTuner e o tipo de modelo `lora`.

### Pré-requisitos

Certifique-se de que você tenha Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12.

Você pode verificar executando:

```bash
python --version
```

Se você não tem Python 3.12 instalado no Ubuntu, pode tentar o seguinte:

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

Lá, você precisará modificar as seguintes variáveis:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "model_type": "lora",
  "model_family": "sd3",
  "pretrained_model_name_or_path": "stabilityai/stable-diffusion-3.5-large",
  "output_dir": "/home/user/outputs/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.0,
  "validation_prompt": "your main test prompt here",
  "user_prompt_library": "config/user_prompt_library.json"
}
```
</details>


- `pretrained_model_name_or_path` - Defina como `stabilityai/stable-diffusion-3.5-large`. Note que você precisará fazer login no Huggingface e ter acesso concedido para baixar este modelo. Vamos falar sobre login no Huggingface mais adiante neste tutorial.
  - Se você preferir treinar o SD3.0 Medium (2B) mais antigo, use `stabilityai/stable-diffusion-3-medium-diffusers`.
- `MODEL_TYPE` - Defina como `lora`.
- `MODEL_FAMILY` - Defina como `sd3`.
- `OUTPUT_DIR` - Defina o diretório onde deseja armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `VALIDATION_RESOLUTION` - Como o SD3 é um modelo 1024px, você pode definir como `1024x1024`.
  - Além disso, o SD3 foi fine-tuned em buckets multi-aspect e outras resoluções podem ser especificadas separando por vírgulas: `1024x1024,1280x768`
- `VALIDATION_GUIDANCE` - O SD3 se beneficia de um valor bem baixo. Defina como `3.0`.

Há algumas configurações adicionais se você estiver em um Mac M-series:

- `mixed_precision` deve ser definido como `no`.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

#### Treinamento de modelo quantizado

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisão e os requisitos de VRAM bem abaixo dos requisitos do treinamento base do SDXL.



> ⚠️ Se você estiver usando um arquivo de configuração JSON, use este formato em `config.json` em vez de `config.env`:

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "text_encoder_3_precision": "no_change",
  "optimizer": "adamw_bf16"
}
```

Para usuários de `config.env` (descontinuado):

```bash
</details>

# escolhas: int8-quanto, int4-quanto, int2-quanto, fp8-quanto
# int8-quanto foi testado com um LoRA de dreambooth de assunto único.
# fp8-quanto não funciona em sistemas Apple. você deve usar níveis int.
# int2-quanto é bem extremo e deixa o LoRA rank-1 inteiro em cerca de 13.9GB de VRAM.
# que os deuses tenham misericórdia da sua alma, caso você force demais.
export TRAINER_EXTRA_ARGS="--base_model_precision=int8-quanto"

# Talvez você queira manter os text encoders em precisão total para que seus text embeds fiquem impecáveis.
# Nós descarregamos os text encoders antes do treinamento, então isso não é um problema durante o treino - apenas durante o pré-cache.
# Alternativamente, você pode fazer uma quantização agressiva aqui e rodá-los em int4 ou int8, porque ninguém pode te impedir.
export TRAINER_EXTRA_ARGS="${TRAINER_EXTRA_ARGS} --text_encoder_1_precision=no_change --text_encoder_2_precision=no_change"

# Quando você está quantizando o modelo, --base_model_default_dtype é definido como bf16 por padrão. Essa configuração exige adamw_bf16, mas economiza mais memória.
# adamw_bf16 só suporta treino bf16, mas qualquer outro otimizador suportará precisão de treino bf16 ou fp32.
export OPTIMIZER="adamw_bf16"
```

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisará garantir que seu dataset seja grande o suficiente para treinar seu modelo efetivamente. Observe que o tamanho mínimo do dataset é `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`, além de ser maior que `VAE_BATCH_SIZE`. O dataset não será utilizável se for muito pequeno.

Dependendo do dataset que você tem, será necessário configurar o diretório do dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

No seu diretório `/home/user/simpletuner/config`, crie um multidatabackend.json:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sd3",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 1024,
    "minimum_image_size": 0,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/home/user/simpletuner/output/cache/vae/sd3/pseudo-camera-10k",
    "instance_data_dir": "/home/user/simpletuner/datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery"
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/sd3/pseudo-camera-10k",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

Depois, crie um diretório `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
popd
```

Isso vai baixar cerca de 10k amostras de fotos para o seu diretório `datasets/pseudo-camera-10k`, que será criado automaticamente.

#### Login no WandB e Huggingface Hub

Você vai querer fazer login no WandB e no HF Hub antes de iniciar o treinamento, especialmente se estiver usando `push_to_hub: true` e `--report_to=wandb`.

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

No diretório do SimpleTuner, basta executar:

```bash
bash train.sh
```

Isso vai iniciar o cache de text embeds e saídas VAE em disco.

Para mais informações, veja os documentos [dataloader](../DATALOADER.md) e [tutorial](../TUTORIAL.md).

## Notas e dicas de solução de problemas

### Skip-layer guidance (SD3.5 Medium)

A StabilityAI recomenda habilitar SLG (Skip-layer guidance) na inferência do SD 3.5 Medium. Isso não impacta os resultados de treinamento, apenas a qualidade das amostras de validação.

Os seguintes valores são recomendados para `config.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "--validation_guidance_skip_layers": [7, 8, 9],
  "--validation_guidance_skip_layers_start": 0.01,
  "--validation_guidance_skip_layers_stop": 0.2,
  "--validation_guidance_skip_scale": 2.8,
  "--validation_guidance": 4.0,
  "--flow_use_uniform_schedule": true,
  "--flow_schedule_auto_shift": true
}
```
</details>

- `..skip_scale` determina quanto escalar a previsão do prompt positivo durante skip-layer guidance. O valor padrão de 2.8 é seguro para o valor de skip base do modelo `7, 8, 9`, mas precisará ser aumentado se mais camadas forem puladas, dobrando-o para cada camada adicional.
- `..skip_layers` informa quais camadas pular durante a previsão do prompt negativo.
- `..skip_layers_start` determina a fração do pipeline de inferência durante a qual o skip-layer guidance deve começar a ser aplicado.
- `..skip_layers_stop` define a fração do total de steps de inferência após a qual o SLG não será mais aplicado.

O SLG pode ser aplicado por menos steps para um efeito mais fraco ou menor redução da velocidade de inferência.

Parece que um treinamento extensivo de um modelo LoRA ou LyCORIS exigirá modificação desses valores, embora não esteja claro exatamente como isso muda.

**CFG mais baixo deve ser usado durante a inferência.**

### Instabilidade do modelo

O modelo SD 3.5 Large 8B tem instabilidades potenciais durante o treinamento:

- Valores altos de `--max_grad_norm` permitem que o modelo explore atualizações de peso potencialmente perigosas
- Taxas de aprendizado podem ser extremamente sensíveis; `1e-5` funciona com StableAdamW, mas `4e-5` pode explodir
- Batches maiores ajudam **muito**
- A estabilidade não é impactada por desativar quantização ou treinar em fp32 puro

O código oficial de treinamento não foi lançado junto com o SD3.5, deixando desenvolvedores para adivinhar como implementar o loop de treinamento com base no [conteúdo do repositório SD3.5](https://github.com/stabilityai/sd3.5).

Algumas mudanças foram feitas no suporte do SimpleTuner ao SD3.5:
- Excluir mais camadas da quantização
- Não zerar mais o padding do T5 por padrão (`--t5_padding`)
- Oferecer um switch (`--sd3_clip_uncond_behaviour` e `--sd3_t5_uncond_behaviour`) para usar captions em branco codificadas para previsões incondicionais (`empty_string`, **padrão**) ou zeros (`zero`), configuração não recomendada para ajustar.
- A função de loss do treinamento do SD3.5 foi atualizada para corresponder à encontrada no repositório upstream StabilityAI/SD3.5
- Valor padrão de `--flow_schedule_shift` atualizado para 3 para corresponder ao valor estático de 1024px do SD3
  - A StabilityAI publicou documentação para usar `--flow_schedule_shift=1` com `--flow_use_uniform_schedule`
  - Membros da comunidade relataram que `--flow_schedule_auto_shift` funciona melhor ao usar treinamento multi-aspect ou multi-resolution
- Limite de comprimento da sequência do tokeniser hard-coded atualizado para **154** com opção de revertê-lo para **77** tokens para economizar espaço em disco ou computação ao custo de degradação da qualidade de saída


#### Valores de configuração estáveis

Estas opções são conhecidas por manter o SD3.5 intacto pelo maior tempo possível:
- optimizer=adamw_bf16
- flow_schedule_shift=1
- learning_rate=1e-4
- batch_size=4 * 3 GPUs
- max_grad_norm=0.1
- base_model_precision=int8-quanto
- Sem loss masking ou regularização de dataset, pois sua contribuição para essa instabilidade é desconhecida
- `validation_guidance_skip_layers=[7,8,9]`

### Configuração de VRAM mais baixa

- SO: Ubuntu Linux 24
- GPU: Um único dispositivo NVIDIA CUDA (10G, 12G)
- Memória do sistema: aproximadamente 50G de memória do sistema
- Precisão do modelo base: `nf4-bnb`
- Otimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolução: 512px
- Batch size: 1, zero steps de gradient accumulation
- DeepSpeed: desativado / não configurado
- PyTorch: 2.5

### SageAttention

Ao usar `--attention_mechanism=sageattention`, a inferência pode ser acelerada durante a validação.

**Nota**: Isso não é compatível com _toda_ configuração de modelo, mas vale a pena tentar.

### Loss mascarada

Se você estiver treinando um assunto ou estilo e quiser mascarar um ou outro, veja a seção de [treinamento com loss mascarada](../DREAMBOOTH.md#masked-loss) no guia de Dreambooth.

### Dados de regularização

Para mais informações sobre datasets de regularização, veja [esta seção](../DREAMBOOTH.md#prior-preservation-loss) e [esta seção](../DREAMBOOTH.md#regularisation-dataset-considerations) do guia de Dreambooth.

### Treinamento quantizado

Veja [esta seção](../DREAMBOOTH.md#quantised-model-training-loralycoris-only) do guia de Dreambooth para informações sobre configurar quantização para SD3 e outros modelos.

### Acompanhamento de CLIP score

Se você quiser habilitar avaliações para pontuar o desempenho do modelo, veja [este documento](../evaluation/CLIP_SCORES.md) para informações sobre configuração e interpretação de CLIP scores.

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
