# Integracao com Hugging Face Datasets

O SimpleTuner suporta carregar datasets diretamente do Hugging Face Hub, permitindo treinamento eficiente em datasets de grande escala sem baixar todas as imagens localmente.

## Visao geral

O backend de datasets do Hugging Face permite que voce:
- Carregue datasets diretamente do Hugging Face Hub
- Aplique filtros com base em metadados ou metricas de qualidade
- Extraia captions de colunas do dataset
- Lide com imagens compostas/em grade
- Armazene em cache apenas os embeddings processados localmente

**Importante**: O SimpleTuner requer acesso ao dataset completo para construir buckets de proporcao e calcular batch sizes. Embora o Hugging Face suporte datasets em streaming, esse recurso nao e compativel com a arquitetura do SimpleTuner. Use filtros para reduzir datasets muito grandes a tamanhos gerenciaveis.

## Configuracao basica

Para usar um dataset do Hugging Face, configure seu dataloader com `"type": "huggingface"`:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "text",
  "image_column": "image",
  "cache_dir": "cache/my-hf-dataset"
}
```

### Campos obrigatorios

- `type`: Deve ser `"huggingface"`
- `dataset_name`: O identificador do dataset no Hugging Face (ex.: "laion/laion-aesthetic")
- `caption_strategy`: Deve ser `"huggingface"`
- `metadata_backend`: Deve ser `"huggingface"`

### Campos opcionais

- `split`: Split do dataset a usar (padrao: "train")
- `revision`: Revisao especifica do dataset
- `image_column`: Coluna que contem imagens (padrao: "image")
- `caption_column`: Coluna(s) que contem captions
- `cache_dir`: Diretorio de cache local para arquivos do dataset
- `streaming`: ⚠️ **Atualmente nao funcional** - o SimpleTuner tenta escanear o dataset de forma eficiente para construir metadados e caches do encoder.
- `num_proc`: Numero de processos para filtragem (padrao: 16)

## Configuracao de captions

O backend do Hugging Face suporta extracao flexivel de captions:

### Coluna unica de caption
```json
{
  "caption_column": "caption"
}
```

### Multiplas colunas de caption
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### Acesso a coluna aninhada
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### Configuracao avancada de captions
```json
{
  "huggingface": {
    "caption_column": "caption",
    "fallback_caption_column": "description",
    "description_column": "detailed_description",
    "width_column": "width",
    "height_column": "height"
  }
}
```

## Filtrando datasets

Aplique filtros para selecionar apenas amostras de alta qualidade:

### Filtragem por qualidade
```json
{
  "huggingface": {
    "filter_func": {
      "quality_thresholds": {
        "clip_score": 0.3,
        "aesthetic_score": 5.0,
        "resolution": 0.8
      },
      "quality_column": "quality_assessment"
    }
  }
}
```

### Filtragem por colecao/subconjunto
```json
{
  "huggingface": {
    "filter_func": {
      "collection": ["photo", "artwork"],
      "min_width": 512,
      "min_height": 512
    }
  }
}
```

## Suporte a imagens compostas

Lide com datasets com multiplas imagens em uma grade:

```json
{
  "huggingface": {
    "composite_image_config": {
      "enabled": true,
      "image_count": 4,
      "select_index": 0
    }
  }
}
```

Essa configuracao vai:
- Detectar grades de 4 imagens
- Extrair apenas a primeira imagem (indice 0)
- Ajustar as dimensoes adequadamente

## Exemplos completos de configuracao

### Dataset basico de fotos
```json
{
  "id": "aesthetic-photos",
  "type": "huggingface",
  "dataset_name": "aesthetic-foundation/aesthetic-photos",
  "split": "train",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image",
  "resolution": 1024,
  "resolution_type": "pixel",
  "minimum_image_size": 512,
  "cache_dir": "cache/aesthetic-photos"
}
```

### Dataset filtrado de alta qualidade
```json
{
  "id": "high-quality-art",
  "type": "huggingface",
  "dataset_name": "example/art-dataset",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": ["title", "description", "tags"],
    "fallback_caption_column": "filename",
    "width_column": "original_width",
    "height_column": "original_height",
    "filter_func": {
      "quality_thresholds": {
        "aesthetic_score": 6.0,
        "technical_quality": 0.8
      },
      "min_width": 768,
      "min_height": 768
    }
  },
  "resolution": 1024,
  "resolution_type": "pixel_area",
  "crop": true,
  "crop_aspect": "square"
}
```

### Dataset de video
```json
{
  "id": "video-dataset",
  "type": "huggingface",
  "dataset_type": "video",
  "dataset_name": "example/video-clips",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "huggingface": {
    "caption_column": "description",
    "num_frames_column": "frame_count",
    "fps_column": "fps"
  },
  "video": {
    "num_frames": 125,
    "min_frames": 100
  },
  "resolution": 480,
  "resolution_type": "pixel"
}
```

## Sistema de arquivos virtual

O backend do Hugging Face usa um sistema de arquivos virtual onde as imagens sao referenciadas pelo indice do dataset:
- `0.jpg` → Primeiro item no dataset
- `1.jpg` → Segundo item no dataset
- etc.

Isso permite que o pipeline padrao do SimpleTuner funcione sem modificacoes.

## Comportamento de cache

- **Arquivos do dataset**: Armazenados em cache de acordo com os defaults da biblioteca Hugging Face datasets
- **Embeddings do VAE**: Armazenados em `cache_dir/vae/{backend_id}/`
- **Embeddings de texto**: Usam a configuracao padrao de cache de text embeds
- **Metadados**: Armazenados em `cache_dir/huggingface_metadata/{backend_id}/`

## Consideracoes de desempenho

1. **Varredura inicial**: A primeira execucao baixara metadados do dataset e construira buckets de proporcao
2. **Tamanho do dataset**: Os metadados do dataset inteiro precisam ser carregados para montar listas de arquivos e calcular tamanhos
3. **Filtragem**: Aplicada durante o carregamento inicial - itens filtrados nao serao baixados
4. **Reuso de cache**: Execucoes posteriores reutilizam metadados e embeddings em cache

**Nota**: Embora o Hugging Face datasets suporte streaming, o SimpleTuner requer acesso completo ao dataset para construir buckets e calcular batch sizes. Datasets muito grandes devem ser filtrados para um tamanho gerenciavel.

## Limitacoes

- Acesso somente leitura (nao pode modificar o dataset fonte)
- Requer conexao com a internet para acesso inicial ao dataset
- Alguns formatos de dataset podem nao ser suportados
- Modo streaming nao e suportado - o SimpleTuner requer acesso completo ao dataset
- Datasets muito grandes devem ser filtrados para tamanhos gerenciaveis
- Carregamento inicial de metadados pode ser intensivo em memoria para datasets enormes

## Solucao de problemas

### Dataset nao encontrado
```
Error: Dataset 'username/dataset' not found
```
- Verifique se o dataset existe no Hugging Face Hub
- Verifique se o dataset e privado (requer autenticacao)
- Garanta a grafia correta do nome do dataset

### Carregamento inicial lento
- Datasets grandes levam tempo para carregar metadados e construir buckets
- Use filtragem agressiva para reduzir o tamanho do dataset
- Considere usar um subconjunto ou uma versao filtrada do dataset
- Arquivos de cache aceleram execucoes posteriores

### Problemas de memoria
- Use filtros para reduzir o tamanho do dataset antes de carregar
- Reduza `num_proc` para operacoes de filtragem
- Considere dividir datasets muito grandes em partes menores
- Use thresholds de qualidade para limitar o dataset a amostras de alta qualidade

### Problemas de extracao de captions
- Verifique se os nomes das colunas correspondem ao schema do dataset
- Verifique estruturas de coluna aninhada
- Use `fallback_caption_column` para captions ausentes

## Uso avancado

### Funcoes de filtro personalizadas

Embora a configuracao suporte filtragem basica, voce pode implementar filtros mais complexos modificando o codigo. A funcao de filtro recebe cada item do dataset e retorna True/False.

### Treinamento com multiplos datasets

Combine datasets do Hugging Face com dados locais:

```json
[
  {
    "id": "hf-dataset",
    "type": "huggingface",
    "dataset_name": "laion/laion-art",
    "probability": 0.7
  },
  {
    "id": "local-dataset",
    "type": "local",
    "instance_data_dir": "/path/to/local/data",
    "probability": 0.3
  }
]
```

Essa configuracao vai amostrar 70% do dataset do Hugging Face e 30% dos dados locais.

## Datasets de audio

Para modelos de audio (como ACE-Step), voce pode definir `dataset_type: "audio"`.

```json
{
    "id": "audio-dataset",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "my-audio-data",
    "audio_column": "audio",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "lyrics"
    }
}
```

*   **`audio_column`**: A coluna que contem dados de audio (decodificados ou bytes). Padrao: `"audio"`.
*   **`audio_caption_fields`**: Uma lista de nomes de colunas para combinar e formar o **prompt** (condicionamento de texto). Padrao: `["prompt", "tags"]`.
*   **`lyrics_column`**: A coluna que contem as letras da musica. Padrao: `"lyrics"`. Se essa coluna estiver ausente, o SimpleTuner verificara `"norm_lyrics"` como fallback.

### Colunas esperadas
*   **`audio`**: Os dados de audio.
*   **`prompt`** / **`tags`**: Tags descritivas ou prompts usados para o codificador de texto.
*   **`lyrics`**: Letras da musica usadas para o codificador de letras.
