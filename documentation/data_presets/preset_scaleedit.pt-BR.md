# ScaleEdit-12M: imagens Parquet pareadas com metadados Webshart

## Detalhes

- **Dataset de origem**: [InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M).
- **Metadados complementares**: [webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata).
- **Descrição**: 11.845.023 linhas de edição em 251 shards Parquet e 23 categorias. Os metadados ocupam cerca de 423 MB; as imagens permanecem na origem.
- **Formato das legendas**: strings `edit_instruction`. São instruções de edição, não descrições independentes de cenas.
- **Licença**: CC-BY-NC-SA-4.0, herdada da origem. Consulte o card original para os direitos das imagens e restrições de uso.

## Pré-processamento necessário

Não é necessário extrair tar, criar legendas por imagem ou compilar metadados. Use SimpleTuner com o [backend Parquet do Webshart](../DATALOADER.md#webshart-datasets), incluindo o [PR #3325](https://github.com/bghira/SimpleTuner/pull/3325), e instale a integração publicada com o Hub:

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

O acesso público ao Hub não exige token. Defina `HF_TOKEN` para acesso autenticado. O manifesto fixa a revisão original em `05d2eea7b80ba72c077d39be43b7aca1a7857e6a` e verifica as somas dos metadados.

## Estrutura do dataset

Cada linha original contém `source_image`, `edited_image`, `edit_instruction`, dimensões das duas imagens e notas de qualidade. `webshart-parquet.json` relaciona os shards a `metadata/<task>/<part>.parquet`. Os metadados não contêm imagens. Como `id` se repete entre tarefas, as junções usam o caminho do shard e a posição física da linha.

## Exemplo de configuração do dataloader

Coloque este array em `multidatabackend.json`. Ele começa pelos 458 pares de `3.4_count_change`, um shard de cerca de 54 MB. Todas as imagens originais e editadas estão embutidas nesse subconjunto fixado. Use um modelo de edição com suporte a `reference_strict`, como [Flux Kontext](../quickstart/FLUX_KONTEXT.md); a configuração de treinamento do modelo é separada.

```json
[
  {
    "id": "scaleedit-edits",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "image",
    "conditioning_data": [
      "scaleedit-sources"
    ],
    "cache_dir_vae": "cache/vae/scaleedit/edited_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/edited_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "edited_image",
        "captions": "edit_instruction",
        "width": "edited_image_width",
        "height": "edited_image_height"
      }
    }
  },
  {
    "id": "scaleedit-sources",
    "type": "webshart",
    "source": "InternVL-U/ScaleEdit-12M",
    "metadata": "webshart/ScaleEdit-12M-metadata",
    "subfolder": "3.4_count_change",
    "caption_strategy": "webshart",
    "metadata_backend": "webshart",
    "resolution": 512,
    "resolution_type": "pixel",
    "crop": false,
    "dataset_type": "conditioning",
    "conditioning_type": "reference_strict",
    "source_dataset_id": "scaleedit-edits",
    "cache_dir_vae": "cache/vae/scaleedit/source_image",
    "webshart": {
      "format": "parquet",
      "cache_dir": "cache/webshart/scaleedit/source_image",
      "shard_cache_gb": 0,
      "parallel_downloads": 4,
      "columns": {
        "image": "source_image",
        "captions": "edit_instruction",
        "width": "source_image_width",
        "height": "source_image_height"
      }
    }
  },
  {
    "id": "scaleedit-text",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/scaleedit"
  }
]
```

## Explicação da configuração

- `scaleedit-edits` usa `edited_image` como alvo e `edit_instruction` como legenda. Suas colunas de dimensões correspondem à imagem editada.
- `scaleedit-sources` usa `source_image` como condicionamento. `conditioning_data` liga o alvo às referências e `source_dataset_id` liga as referências de volta ao alvo. As dimensões originais são escolhidas explicitamente.
- As duas entradas usam a mesma origem, os mesmos metadados e `subfolder`, preservando as identidades shard/linha. Mantenha separados os diretórios de cache VAE e Webshart.
- `webshart.shard_cache_gb: 0` mantém leituras por intervalo. A maioria dos shards completos tem cerca de 31 GB; o cache completo pode baixar um shard inteiro na primeira solicitação de imagem.
- `crop: false` evita recortes independentes nesta receita inicial. `reference_strict` usa a geometria de preparação do alvo na referência. Mantenha o pré-processamento compatível com o modelo de edição.

## Expandir além do exemplo

Antes de alterar ou remover `subfolder`, trate as 702.078 imagens originais disponíveis apenas por URL no dataset completo. Os metadados preservam URL, SHA-256 e data de coleta, mas não fazem o download: um `source_image` ausente continua `None`. Baixe e verifique explicitamente essas origens ou crie subconjuntos alinhados por caminho e linha.

Não limite nem filtre alvos e referências de forma independente: eles precisam selecionar as mesmas amostras. Um `max_num_samples` igual não garante os mesmos pares se os buckets forem diferentes.

Grupos de linhas e dicionários grandes ainda podem usar centenas de MiB ao ler imagens, mesmo sem cache completo. O bucketing usa os metadados complementares; o acesso às imagens tem outro custo de memória e rede. O [benchmark do loader Webshart](../DATALOADER.md#parquet-image-bytes-on-s3) permite medir uma amostra limitada antes de ampliar.
