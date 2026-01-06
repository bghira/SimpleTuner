# Niji v6 520k

## Detalhes

- **Link do Hub**: [terminusresearch/nijijourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/nijijourney-v6-520k-raw)
- **Descricao**: ~520.000 saidas de alta qualidade onde quaisquer prompts em japones foram relegendados com GPT-3.5-Turbo.
- **Formato(s) de caption**: Parquet

## Armazenamento necessario

Este dataset contem todos os dados de imagem e, por isso, sera dificil extrair sem espaco em disco adequado. **Garanta pelo menos 1,5TB de espaco em disco para extrair.**

Text embeds T5-XXL para este modelo consumirao ~520GB mesmo com `--compress_disk_cache` habilitado.
Os VAE embeds consumirao pouco menos de 80 a 100GB de espaco, dependendo do modelo treinado e da resolucao dos embeds.

## Download

```bash
huggingface-cli download --repo-type=dataset terminusresearch/nijijourney-v6-520k-raw --local-dir=nijijourney-v6-520k-raw
```

Isso baixara simultaneamente os segmentos tar em partes do Hugging Face Hub.

## Extrair

```bash
cd nijijourney-v6-520k-raw
cat *.tar | tar x
```

Isso criara uma pasta contendo todas as amostras dentro do diretorio atual.

## Exemplo de configuracao do dataloader

```json
{
    "id": "nijijourney-v6-520k-raw",
    "type": "local",
    "cache_dir_vae": "cache/vae-nj-520k/",
    "crop": true,
    "crop_aspect": "square",
    "resolution": 1.0,
    "maximum_image_size": 1.0,
    "minimum_image_size": 0.75,
    "target_downsample_size": 1.00,
    "resolution_type": "area",
    "caption_strategy": "parquet",
    "metadata_backend": "parquet",
    "parquet": {
        "path": "/path/to/nijijourney-v6-520k-raw/train.parquet",
        "caption_column": "gpt_caption",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height",
        "identifier_includes_extension": false
    }
}
```
