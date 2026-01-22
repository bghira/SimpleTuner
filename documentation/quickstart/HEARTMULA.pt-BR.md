# Inicio rapido do HeartMuLa

Neste exemplo, vamos treinar o modelo HeartMuLa oss 3B de geracao de audio.

## Visao geral

HeartMuLa e um transformer autoregressivo de 3B parametros que prediz tokens de audio discretos a partir de tags e letras. Os tokens sao decodificados com o HeartCodec para produzir formas de onda.

## Requisitos de hardware

HeartMuLa e um modelo de 3B parametros, tornando-o relativamente leve comparado a modelos grandes de geracao de imagem como o Flux.

- **Minimo:** GPU NVIDIA com 12GB+ de VRAM (ex.: 3060, 4070).
- **Recomendado:** GPU NVIDIA com 24GB+ de VRAM (ex.: 3090, 4090, A10G) para batches maiores.
- **Mac:** Suportado via MPS no Apple Silicon (requer ~36GB+ de memoria unificada).

### Requisitos de armazenamento

> ⚠️ **Aviso sobre dataset de tokens:** HeartMuLa treina com tokens de audio precomputados. O SimpleTuner nao gera tokens durante o treino, entao seu dataset deve fornecer metadados `audio_tokens` ou `audio_tokens_path`. Arquivos de tokens podem ser grandes, entao planeje o espaco em disco.

> 💡 **Dica:** Usar quantizacao `int8-quanto` permite treinar em GPUs com menos VRAM (ex.: 12GB-16GB) com perda minima de qualidade.

## Pre-requisitos

Garanta um ambiente Python 3.10+ funcional.

```bash
pip install simpletuner
```

## Configuracao

Recomendamos manter suas configuracoes organizadas. Vamos criar uma pasta dedicada para esta demo.

```bash
mkdir -p config/heartmula-training-demo
```

### Ajustes criticos

Crie `config/heartmula-training-demo/config.json` com estes valores:

<details>
<summary>Ver exemplo de config</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### Ajustes de validacao

Adicione estes valores ao seu `config.json` para monitorar o progresso:

- **`validation_prompt`**: Tags ou uma descricao do audio (ex.: "Pop animado com sintetizadores brilhantes").
- **`validation_lyrics`**: (Opcional) Letras para o modelo cantar. Use string vazia para instrumentais.
- **`validation_audio_duration`**: Duracao em segundos para clipes de validacao (padrao: 30.0).
- **`validation_guidance`**: Escala de guidance (comece em torno de 1.5 - 3.0).
- **`validation_step_interval`**: Com que frequencia gerar amostras (ex.: a cada 100 passos).

### Recursos experimentais avancados

<details>
<summary>Mostrar detalhes experimentais avancados</summary>


O SimpleTuner inclui recursos experimentais que podem melhorar significativamente a estabilidade e o desempenho do treinamento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduz o viés de exposicao e melhora a qualidade da saida deixando o modelo gerar suas proprias entradas durante o treino.

> ⚠️ Esses recursos aumentam a sobrecarga computacional do treinamento.

</details>

## Configuracao do dataset

HeartMuLa requer um dataset **especifico para audio** com tokens precomputados.

Cada amostra deve fornecer:

- `tags` (string)
- `lyrics` (string; pode estar vazia)
- `audio_tokens` ou `audio_tokens_path`

Os arrays de tokens devem ser 2D com formato `[frames, num_codebooks]` ou `[num_codebooks, frames]`.

> 💡 **Nota:** HeartMuLa nao usa um codificador de texto separado, entao nao e necessario um backend de text-embeds.

### Opcao 1: Dataset do Hugging Face (tokens em colunas)

Crie `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>Ver exemplo de config</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

Garanta que seu dataset inclua colunas `audio_tokens` ou `audio_tokens_path` junto com os campos de texto.

### Opcao 2: Arquivos de audio locais + metadados de tokens

Crie `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>Ver exemplo de config</summary>

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
  }
]
```
</details>

Garanta que seu backend de metadados forneca `audio_tokens` ou `audio_tokens_path` para cada amostra.

### Estrutura de dados

Coloque seus arquivos de audio em `datasets/my_audio_files`. O SimpleTuner suporta uma ampla gama de formatos incluindo:

- **Sem perda:** `.wav`, `.flac`, `.aiff`, `.alac`
- **Com perda:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **Nota:** Para suportar formatos como MP3, AAC e WMA, voce deve ter o **FFmpeg** instalado no seu sistema.

Para tags e letras, coloque arquivos de texto correspondentes ao lado dos seus arquivos de audio se voce usa `caption_strategy: textfile`:

- **Audio:** `track_01.wav`
- **Tags (Prompt):** `track_01.txt` (Contem a descricao de texto, ex.: "Uma balada de jazz lenta")
- **Letras (Opcional):** `track_01.lyrics` (Contem o texto das letras)

Forneca os arrays de tokens via metadados (por exemplo, entradas `audio_tokens_path` apontando para arquivos `.npy` ou `.npz`).

<details>
<summary>Exemplo de layout do dataset</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **Nota sobre letras:** HeartMuLa espera uma string de letras para cada amostra. Para dados instrumentais, forneca uma string vazia em vez de omitir o campo.

## Treinamento

Inicie o treinamento especificando seu ambiente:

```bash
simpletuner train env=heartmula-training-demo
```

Este comando diz ao SimpleTuner para procurar `config.json` dentro de `config/heartmula-training-demo/`.

> 💡 **Dica (Continuar treinamento):** Para continuar o fine-tuning a partir de uma LoRA existente, use a opcao `--init_lora`:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## Solucao de problemas

- **Erros de validacao:** Garanta que voce nao esta tentando usar recursos de validacao focados em imagem como `num_validation_images` > 1 (mapeado conceitualmente para batch size em audio) ou metricas baseadas em imagem (pontuacao CLIP).
- **Problemas de memoria:** Se ocorrer OOM, tente reduzir `train_batch_size` ou habilitar `gradient_checkpointing`.
