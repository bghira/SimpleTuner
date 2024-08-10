# Midjourney v6 520k

## Details

- **Hub link**: [terminusresearch/midjourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/midjourney-v6-520k-raw)
- **Description**: ~520,000 high quality outputs where any Japanese user prompts have been re-captioned with GPT-3.5-Turbo.
- **Caption format(s)**: Parquet

## Required storage

This dataset contains all image data, and as such, it will be difficult to extract without adequate disk space. **Ensure you have at least 1.5TB of disk space available to extract it.**

T5-XXL text embeds for this model will consume ~520GB even with `--compress_disk_cache` enabled.
The VAE embeds will consume just under 80 to 100GB of space, depending on the model being trained and the resolution of the embeds.


## Download

```bash
huggingface-cli download --repo-type=dataset terminusresearch/midjourney-v6-520k-raw --local-dir=midjourney-v6-520k-raw
```

This will simultaneously download the chunked tar segments from Hugging Face Hub.

## Extract

```bash
cd midjourney-v6-520k-raw
cat *.tar | tar x
```

This will create a folder containing all of the samples inside the current directory.

## Dataloader configuration example

```json
{
    "id": "midjourney-v6-520k-raw",
    "type": "local",
    "cache_dir_vae": "cache/vae-mj-520k/",
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
        "path": "/path/to/midjourney-v6-520k-raw/train.parquet",
        "caption_column": "gpt_caption",
        "filename_column": "id",
        "width_column": "width",
        "height_column": "height",
        "identifier_includes_extension": false
    }
}
```
