# Subjects200K

## Detalhes

- **Link do Hub**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **Descricao**: 200k+ imagens compostas de alta qualidade com descricoes pareadas e avaliacoes de qualidade. Cada amostra contem duas imagens lado a lado do mesmo sujeito em contextos diferentes.
- **Formato(s) de caption**: JSON estruturado com descricoes separadas para cada metade da imagem
- **Recursos especiais**: Scores de avaliacao de qualidade, tags de colecao, imagens compostas

## Estrutura do dataset

O dataset Subjects200K e unico porque:
- Cada campo `image` contem **duas imagens combinadas lado a lado** em uma unica imagem larga
- Cada amostra tem **duas captions separadas** - uma para a imagem da esquerda (`description.description_0`) e outra para a direita (`description.description_1`)
- Metadados de avaliacao de qualidade permitem filtrar por metricas de qualidade da imagem
- Imagens sao pre-organizadas em colecoes

Exemplo de estrutura de dados:
```python
{
    'image': PIL.Image,  # Combined image (e.g., 1056x528 for two 528x528 images)
    'collection': 'collection_1',
    'quality_assessment': {
        'compositeStructure': 5,
        'objectConsistency': 5,
        'imageQuality': 5
    },
    'description': {
        'item': 'Eames Lounge Chair',
        'description_0': 'The Eames Lounge Chair is placed in a modern city living room...',
        'description_1': 'Nestled in a cozy nook of a rustic cabin...',
        'category': 'Furniture',
        'description_valid': True
    }
}
```

## Uso direto (sem pre-processamento)

Diferente de datasets que exigem extracao e pre-processamento, o Subjects200K pode ser usado diretamente do HuggingFace! O dataset ja esta formatado e hospedado.

Basta garantir que voce tenha a biblioteca `datasets` instalada:
```bash
pip install datasets
```

## Configuracao do dataloader

Como cada amostra contem duas imagens, precisamos configurar **duas entradas de dataset separadas** - uma para cada metade da imagem composta:

```json
[
    {
        "id": "subjects200k-left",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-left",
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "subjects200k-right",
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "cache_dir_vae": "cache/vae/subjects-right",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            },
            "filter_func": {
                "collection": "collection_1",
                "quality_thresholds": {
                    "compositeStructure": 4.5,
                    "objectConsistency": 4.5,
                    "imageQuality": 4.5
                }
            }
        }
    },
    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```

### Explicando a configuracao

#### Configuracao de imagem composta
- `composite_image_config.enabled`: Ativa o manuseio de imagem composta
- `composite_image_config.image_count`: Numero de imagens na composicao (2 para lado a lado)
- `composite_image_config.select_index`: Qual imagem extrair (0 = esquerda, 1 = direita)

#### Filtragem de qualidade
O `filter_func` permite filtrar amostras com base em:
- `collection`: Usar apenas imagens de colecoes especificas
- `quality_thresholds`: Scores minimos para metricas de qualidade:
  - `compositeStructure`: Quao bem as duas imagens funcionam juntas
  - `objectConsistency`: Consistencia do sujeito entre as duas imagens
  - `imageQuality`: Qualidade geral da imagem

#### Selecao de caption
- Imagem da esquerda usa: `"caption_column": "description.description_0"`
- Imagem da direita usa: `"caption_column": "description.description_1"`

### Opcoes de personalizacao

1. **Ajustar thresholds de qualidade**: Valores menores (ex.: 4.0) incluem mais imagens, valores maiores (ex.: 4.8) sao mais seletivos

2. **Usar colecoes diferentes**: Mude `"collection": "collection_1"` para outras colecoes disponiveis no dataset

3. **Alterar resolucao**: Ajuste o valor de `resolution` de acordo com suas necessidades de treino

4. **Desativar filtragem**: Remova a secao `filter_func` para usar todas as imagens

5. **Usar nomes de itens como captions**: Mude a coluna de caption para `"description.item"` para usar apenas o nome do sujeito

### Dicas

- O dataset sera baixado e colocado em cache automaticamente no primeiro uso
- Cada "metade" e tratada como um dataset independente, efetivamente dobrando suas amostras de treino
- Considere usar thresholds de qualidade diferentes para cada metade se quiser variedade
- Os diretorios de cache VAE devem ser diferentes para cada metade para evitar conflitos
