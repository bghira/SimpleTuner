## Guia de Início Rápido do Stable Diffusion XL

Neste exemplo, vamos treinar um modelo Stable Diffusion XL usando o toolkit SimpleTuner e o tipo de modelo `lora`.

Em comparação com modelos modernos maiores, o SDXL é bem modesto em tamanho, então talvez seja possível usar treinamento `full`, mas isso vai exigir VRAM adicional em comparação ao LoRA e alguns ajustes de hiperparâmetros.

### Pré-requisitos

Certifique-se de que você tenha Python instalado; o SimpleTuner funciona bem com 3.10 até 3.12 (máquinas AMD ROCm exigem 3.12).

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
pip install 'simpletuner[cuda13]'
```

Para instalação manual ou setup de desenvolvimento, veja a [documentação de instalação](../INSTALL.md).

### Configurando o ambiente

Para rodar o SimpleTuner, você precisará configurar um arquivo de configuração, os diretórios de dataset e modelo, e um arquivo de configuração do dataloader.

#### Arquivo de configuração

Um script experimental, `configure.py`, pode permitir que você pule esta seção inteiramente por meio de uma configuração interativa passo a passo. Ele contém alguns recursos de segurança que ajudam a evitar armadilhas comuns.

**Nota:** Isso não **configura totalmente** seu dataloader. Você ainda terá que fazer isso manualmente, mais tarde.

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

#### Etapas adicionais para AMD ROCm

O seguinte deve ser executado para um AMD MI300X ser utilizável:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

Lá, você precisará modificar as seguintes variáveis:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "model_type": "lora",
  "model_family": "sdxl",
  "model_flavour": "base-1.0",
  "output_dir": "/home/user/output/models",
  "validation_resolution": "1024x1024,1280x768",
  "validation_guidance": 3.4,
  "use_gradient_checkpointing": true,
  "learning_rate": 1e-4
}
```
</details>

- `model_family` - Defina como `sdxl`.
- `model_flavour` - Defina como `base-1.0`, ou use `pretrained_model_name_or_path` para apontar para um modelo diferente.
- `model_type` - Defina como `lora`.
- `use_dora` - Defina como `true` se você quiser treinar DoRA.
- `output_dir` - Defina o diretório onde deseja armazenar seus checkpoints e imagens de validação. É recomendado usar um caminho completo aqui.
- `validation_resolution` - Defina como `1024x1024` para este exemplo.
  - Além disso, o Stable Diffusion XL foi fine-tuned em buckets multi-aspect e outras resoluções podem ser especificadas separando por vírgulas: `1024x1024,1280x768`
- `validation_guidance` - Use qualquer valor com o qual você se sinta confortável para testes em inferência. Defina entre `4.2` e `6.4`.
- `use_gradient_checkpointing` - Provavelmente deve ser `true`, a menos que você tenha MUITA VRAM e queira sacrificar um pouco para ficar mais rápido.
- `learning_rate` - `1e-4` é bastante comum para redes de baixo rank, embora `1e-5` possa ser uma escolha mais conservadora se você notar algum "burning" ou overtraining precoce.

Há algumas configurações adicionais se você estiver em um Mac M-series:

- `mixed_precision` deve ser definido como `no`.
  - Isso era verdade no pytorch 2.4, mas talvez o bf16 já possa ser usado agora a partir do 2.6+
- `attention_mechanism` pode ser definido como `xformers` para aproveitar isso, mas é meio obsoleto.

#### Treinamento de modelo quantizado

Testado em sistemas Apple e NVIDIA, o Hugging Face Optimum-Quanto pode ser usado para reduzir a precisão e os requisitos de VRAM do Unet, mas não funciona tão bem quanto em modelos Diffusion Transformer como SD3/Flux, então não é recomendado.

Se você estiver com recursos apertados, ainda é possível usar.

Para `config.json`:
<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "optimizer": "optimi-lion"
}
```
</details>

#### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento, especialmente para datasets menores ou arquiteturas mais antigas como o SDXL.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.
*   **[Diff2Flow](../experimental/DIFF2FLOW.md):** permite treinar SDXL com um objetivo de Flow Matching, potencialmente melhorando a consistência e a qualidade da geração.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

</details>

#### Considerações sobre o dataset

É crucial ter um dataset substancial para treinar seu modelo. Existem limitações no tamanho do dataset, e você precisará garantir que seu dataset seja grande o suficiente para treinar seu modelo efetivamente. Observe que o tamanho mínimo do dataset é `TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS`. O dataset não será descoberto pelo treinador se for muito pequeno.

> 💡 **Dica:** Para datasets grandes em que espaço em disco é uma preocupação, você pode usar `--vae_cache_disable` para realizar codificação VAE online sem armazenar os resultados no disco. Isso é implicitamente habilitado se você usar `--vae_cache_ondemand`, mas adicionar `--vae_cache_disable` garante que nada seja gravado em disco.

Dependendo do dataset que você tem, será necessário configurar o diretório do dataset e o arquivo de configuração do dataloader de forma diferente. Neste exemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

No seu diretório `OUTPUT_DIR`, crie um multidatabackend.json:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "pseudo-camera-10k-sdxl",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 1.0,
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "resolution_type": "area",
    "cache_dir_vae": "cache/vae/sdxl/pseudo-camera-10k",
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
    "cache_dir": "cache/text/sdxl/pseudo-camera-10k",
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
huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=datasets/pseudo-camera-10k
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
