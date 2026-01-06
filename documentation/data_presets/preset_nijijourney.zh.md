# Niji v6 520k

## 详情

- **Hub 链接**: [terminusresearch/nijijourney-v6-520k-raw](https://huggingface.co/datasets/terminusresearch/nijijourney-v6-520k-raw)
- **描述**: 约 520,000 张高质量输出，日本语用户提示词已由 GPT-3.5-Turbo 重新生成字幕。
- **字幕格式**: Parquet

## 所需存储

该数据集包含全部图像数据，没有足够磁盘空间将难以解压。**请确保至少有 1.5TB 空间用于解压。**

即使启用 `--compress_disk_cache`，该模型的 T5-XXL 文本嵌入仍会占用约 520GB。
VAE 嵌入将占用约 80–100GB，具体取决于训练模型与嵌入分辨率。

## 下载

```bash
huggingface-cli download --repo-type=dataset terminusresearch/nijijourney-v6-520k-raw --local-dir=nijijourney-v6-520k-raw
```

这会从 Hugging Face Hub 同步下载分片 tar 包。

## 解压

```bash
cd nijijourney-v6-520k-raw
cat *.tar | tar x
```

这会在当前目录创建一个包含全部样本的文件夹。

## 数据加载器配置示例

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
