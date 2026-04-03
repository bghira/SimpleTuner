# Guia de Início Rápido do ACE-Step

Neste exemplo, vamos treinar o modelo de geração de áudio ACE-Step. O SimpleTuner atualmente suporta o caminho original do ACE-Step v1 3.5B e o treinamento LoRA com compatibilidade futura para o bundle ACE-Step v1.5.

## Visão geral

O ACE-Step é um modelo de áudio baseado em transformer e flow-matching, projetado para síntese de alta qualidade. No SimpleTuner:

- `base` usa o caminho original de treinamento do ACE-Step v1 3.5B.
- `v15-turbo`, `v15-base` e `v15-sft` usam as variantes do bundle ACE-Step v1.5 carregadas de `ACE-Step/Ace-Step1.5`.

## Requisitos de hardware

O ACE-Step é um modelo de 3.5B parâmetros, o que o torna relativamente leve em comparação com modelos grandes de geração de imagem como o Flux.

- **Mínimo:** GPU NVIDIA com 12GB+ de VRAM (ex.: 3060, 4070).
- **Recomendado:** GPU NVIDIA com 24GB+ de VRAM (ex.: 3090, 4090, A10G) para lotes maiores.
- **Mac:** Suportado via MPS no Apple Silicon (requer ~36GB+ de Memória Unificada).

### Requisitos de armazenamento

> ⚠️ **Aviso de uso de disco:** O cache de VAE para modelos de áudio pode ser substancial. Por exemplo, um único clipe de áudio de 60 segundos pode resultar em um arquivo latente em cache de ~89MB. Essa estratégia de cache é usada para reduzir drasticamente a exigência de VRAM durante o treinamento. Garanta que você tenha espaço em disco suficiente para o cache do seu dataset.

> 💡 **Dica:** Para datasets maiores, você pode usar a opção `--vae_cache_disable` para desativar a gravação de embeddings em disco. Isso habilita implicitamente o cache sob demanda, o que economiza espaço em disco, mas aumenta o tempo de treinamento e o uso de memória já que as codificações são executadas durante o loop de treinamento.

> 💡 **Dica:** Usar quantização `int8-quanto` permite treinar em GPUs com menos VRAM (ex.: 12GB-16GB) com perda mínima de qualidade.

## Pré-requisitos

Garanta que você tenha um ambiente funcional com Python 3.10+.

```bash
pip install simpletuner
```

## Configuração

Recomenda-se manter suas configurações organizadas. Vamos criar uma pasta dedicada para esta demonstração.

```bash
mkdir -p config/acestep-training-demo
```

### Configurações críticas

O SimpleTuner atualmente suporta estes flavours do ACE-Step:

- `base`: ACE-Step v1 3.5B original
- `v15-turbo`, `v15-base`, `v15-sft`: variantes do bundle ACE-Step v1.5

Use a configuração correspondente à variante desejada.

Presets de exemplo prontos para uso estão disponíveis em:

- `simpletuner/examples/ace_step-v1-0.peft-lora`
- `simpletuner/examples/ace_step-v1-5.peft-lora`

Você pode iniciá-los diretamente com `simpletuner train example=ace_step-v1-0.peft-lora` ou `simpletuner train example=ace_step-v1-5.peft-lora`.

#### Exemplo ACE-Step v1

Crie `config/acestep-training-demo/config.json` com estes valores:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

#### Exemplo ACE-Step v1.5

Para ACE-Step v1.5, mantenha `model_family: "ace_step"`, selecione um flavour v1.5 e aponte o checkpoint raiz para o bundle compartilhado v1.5:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "v15-base",
  "pretrained_model_name_or_path": "ACE-Step/Ace-Step1.5",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

### Configurações de validação

Adicione estas configurações ao seu `config.json` para monitorar o progresso:

- **`validation_prompt`**: Uma descrição de texto do áudio que você quer gerar (ex.: "Uma música pop cativante com bateria animada").
- **`validation_lyrics`**: (Opcional) Letras para o modelo cantar.
- **`validation_audio_duration`**: Duração em segundos para clipes de validação (padrão: 30.0).
- **`validation_guidance`**: Escala de guidance (padrão: ~3.0 - 5.0).
- **`validation_step_interval`**: Com que frequência gerar amostras (ex.: a cada 100 steps).

> ⚠️ **Limitação do ACE-Step v1.5:** a integração atual do SimpleTuner suporta treinamento v1.5, mas o pipeline embutido de validação/inferência do ACE-Step ainda é somente v1.0. Para execuções v1.5, desative a validação no loop ou valide com tooling upstream/externo.

### Recursos experimentais avançados

<details>
<summary>Mostrar detalhes experimentais avançados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposição e melhora a qualidade de saída ao deixar o modelo gerar suas próprias entradas durante o treinamento.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

</details>

## Configuração do dataset

O ACE-Step requer uma configuração de dataset **específica para áudio**.

### Opção 1: Dataset de demonstração (Hugging Face)

Para um início rápido, você pode usar o preset [ACEStep-Songs](../data_presets/preset_audio_dataset_with_lyrics.md).

Crie `config/acestep-training-demo/multidatabackend.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

> Veja opções e requisitos de caption_strategy em [DATALOADER.md](../DATALOADER.md#caption_strategy).

### Opção 2: Arquivos de áudio locais

Crie `config/acestep-training-demo/multidatabackend.json`:

<details>
<summary>Ver exemplo de configuração</summary>

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

### Estrutura de dados

Coloque seus arquivos de áudio em `datasets/my_audio_files`. O SimpleTuner suporta uma ampla variedade de formatos, incluindo:

- **Sem perda:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Com perda:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Nota:** Para suportar formatos como MP3, AAC e WMA, você deve ter o **FFmpeg** instalado no sistema.

Para captions e letras, coloque os arquivos de texto correspondentes ao lado dos seus arquivos de áudio:

- **Áudio:** `track_01.wav`
- **Caption (Prompt):** `track_01.txt` (Contém a descrição em texto, ex.: "Uma balada de jazz lenta")
- **Letra (Opcional):** `track_01.lyrics` (Contém o texto da letra)

<details>
<summary>Exemplo de layout do dataset</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **Avançado:** Se seu dataset usa uma convenção de nomes diferente (ex.: `_lyrics.txt`), você pode personalizar isso na configuração do dataset.

<details>
<summary>Ver exemplo de nome de arquivo de letras personalizado</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **Nota sobre letras:** Se um arquivo `.lyrics` não for encontrado para uma amostra, os embeddings de letras serão zerados. O ACE-Step espera condicionamento por letras; treinar muito em dados sem letras (instrumentais) pode exigir mais steps de treino para o modelo aprender a gerar áudio instrumental de alta qualidade com entradas de letras zeradas.

## Treinamento

Inicie o treino especificando seu ambiente:

```bash
simpletuner train env=acestep-training-demo
```

Esse comando diz ao SimpleTuner para procurar `config.json` dentro de `config/acestep-training-demo/`.

> 💡 **Dica (Continuar treinamento):** Para continuar o fine-tuning a partir de um LoRA existente (ex.: checkpoints oficiais do ACE-Step ou adaptadores da comunidade), use a opção `--init_lora`:
> ```bash
> simpletuner train env=acestep-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

### Treinando o embedder de letras (estilo upstream)

> ℹ️ **Nota de versão:** `lyrics_embedder_train` atualmente só se aplica ao caminho de treinamento ACE-Step v1. O caminho LoRA forward-compatible de v1.5 no SimpleTuner é somente do decoder.

O treinador upstream do ACE-Step faz fine-tuning do embedder de letras junto com o denoiser. Para espelhar esse comportamento no SimpleTuner (apenas LoRA completo ou standard):

- Habilite: `lyrics_embedder_train: true`
- Substituições opcionais (caso contrário, o otimizador/scheduler principal é reutilizado):
  - `lyrics_embedder_lr`
  - `lyrics_embedder_optimizer`
  - `lyrics_embedder_lr_scheduler`

Exemplo:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
  "lyrics_embedder_train": true,
  "lyrics_embedder_lr": 5e-5,
  "lyrics_embedder_optimizer": "torch-adamw",
  "lyrics_embedder_lr_scheduler": "cosine_with_restarts"
}
```
</details>
Os pesos do embedder são salvos junto com os checkpoints do LoRA e restaurados ao retomar.

## Solução de problemas

- **Erros de validação:** Garanta que você não esteja tentando usar recursos de validação centrados em imagem, como `num_validation_images` > 1 (mapeado conceitualmente para tamanho do lote no áudio) ou métricas baseadas em imagem (CLIP score).
- **Problemas de memória:** Se ocorrer OOM, tente reduzir `train_batch_size` ou habilitar `gradient_checkpointing`.

## Migrando do treinador upstream

Se você vem dos scripts originais de treinamento do ACE-Step, veja como os parâmetros mapeiam para o `config.json` do SimpleTuner:

| Parâmetro upstream | `config.json` do SimpleTuner | Padrão / Notas |
| :--- | :--- | :--- |
| `--learning_rate` | `learning_rate` | `1e-4` |
| `--num_workers` | `dataloader_num_workers` | `8` |
| `--max_steps` | `max_train_steps` | `2000000` |
| `--every_n_train_steps` | `checkpointing_steps` | `2000` |
| `--precision` | `mixed_precision` | `"fp16"` ou `"bf16"` (use `"no"` para fp32) |
| `--accumulate_grad_batches` | `gradient_accumulation_steps` | `1` |
| `--gradient_clip_val` | `max_grad_norm` | `0.5` |
| `--shift` | `flow_schedule_shift` | `3.0` (Específico do ACE-Step) |

### Convertendo dados brutos

Se você tem arquivos brutos de áudio/texto/letras e quer usar o formato de dataset do Hugging Face (como usado pela ferramenta upstream `convert2hf_dataset.py`), você pode usar o dataset resultante diretamente no SimpleTuner.

O conversor upstream produz um dataset com colunas `tags` e `norm_lyrics`. Para usar isso, configure seu backend assim:

<details>
<summary>Ver exemplo de configuração</summary>

```json
{
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "path/to/converted/dataset",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "norm_lyrics"
    }
}
```
</details>
