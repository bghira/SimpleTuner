# Hugging Face Datasets Integration

SimpleTuner supports loading datasets directly from the Hugging Face Hub, enabling efficient training on large-scale datasets without downloading all images locally.

## Overview

The Hugging Face datasets backend allows you to:
- Load datasets directly from Hugging Face Hub
- Apply filters based on metadata or quality metrics
- Extract captions from dataset columns
- Handle composite/grid images
- Cache only the processed embeddings locally

**Important**: SimpleTuner requires full dataset access to build aspect ratio buckets and calculate batch sizes. While Hugging Face supports streaming datasets, this feature is not compatible with SimpleTuner's architecture. Use filtering to reduce very large datasets to manageable sizes.

## Basic Configuration

To use a Hugging Face dataset, configure your dataloader with `"type": "huggingface"`:

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

### Required Fields

- `type`: Must be `"huggingface"`
- `dataset_name`: The Hugging Face dataset identifier (e.g., "laion/laion-aesthetic")
- `caption_strategy`: Must be `"huggingface"`
- `metadata_backend`: Must be `"huggingface"`

### Optional Fields

- `split`: Dataset split to use (default: "train")
- `revision`: Specific dataset revision
- `image_column`: Column containing images (default: "image")
- `caption_column`: Column(s) containing captions
- `cache_dir`: Local cache directory for dataset files
- `streaming`: ⚠️ **Currently not functional** - SimpleTuner tries to efficiently scan the dataset to build metadata and encoder caches.
- `num_proc`: Number of processes for filtering (default: 16)

## Caption Configuration

The Hugging Face backend supports flexible caption extraction:

### Single Caption Column
```json
{
  "caption_column": "caption"
}
```

### Multiple Caption Columns
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### Nested Column Access
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### Advanced Caption Configuration
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

## Filtering Datasets

Apply filters to select only high-quality samples:

### Quality-Based Filtering
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

### Collection/Subset Filtering
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

## Composite Image Support

Handle datasets with multiple images in a grid:

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

This configuration will:
- Detect 4-image grids
- Extract only the first image (index 0)
- Adjust dimensions accordingly

## Complete Example Configurations

### Basic Photo Dataset
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

### Filtered High-Quality Dataset
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

### Video Dataset
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

## Virtual File System

The Hugging Face backend uses a virtual file system where images are referenced by their dataset index:
- `0.jpg` → First item in dataset
- `1.jpg` → Second item in dataset
- etc.

This allows the standard SimpleTuner pipeline to work without modification.

## Caching Behavior

- **Dataset files**: Cached according to Hugging Face datasets library defaults
- **VAE embeddings**: Stored in `cache_dir/vae/{backend_id}/`
- **Text embeddings**: Use standard text embed cache configuration
- **Metadata**: Stored in `cache_dir/huggingface_metadata/{backend_id}/`

## Performance Considerations

1. **Initial scan**: The first run will download dataset metadata and build aspect ratio buckets
2. **Dataset size**: The entire dataset metadata must be loaded to build file lists and calculate lengths
3. **Filtering**: Applied during initial load - filtered items won't be downloaded
4. **Cache reuse**: Subsequent runs reuse cached metadata and embeddings

**Note**: While Hugging Face datasets support streaming, SimpleTuner requires full dataset access to build aspect buckets and calculate batch sizes. Very large datasets should be filtered to a manageable size.

## Limitations

- Read-only access (cannot modify source dataset)
- Requires internet connection for initial dataset access
- Some dataset formats may not be supported
- Streaming mode is not supported - SimpleTuner requires full dataset access
- Very large datasets must be filtered to manageable sizes
- Initial metadata loading can be memory-intensive for huge datasets

## Troubleshooting

### Dataset Not Found
```
Error: Dataset 'username/dataset' not found
```
- Verify the dataset exists on Hugging Face Hub
- Check if the dataset is private (requires authentication)
- Ensure correct spelling of dataset name

### Slow Initial Loading
- Large datasets take time to load metadata and build buckets
- Use aggressive filtering to reduce dataset size
- Consider using a subset or filtered version of the dataset
- Cache files will speed up subsequent runs

### Memory Issues
- Use filters to reduce dataset size before loading
- Reduce `num_proc` for filtering operations
- Consider splitting very large datasets into smaller chunks
- Use quality thresholds to limit the dataset to high-quality samples

### Caption Extraction Issues
- Verify column names match dataset schema
- Check for nested column structures
- Use `fallback_caption_column` for missing captions

## Advanced Usage

### Custom Filter Functions

While the configuration supports basic filtering, you can implement more complex filters by modifying the code. The filter function receives each dataset item and returns True/False.

### Multi-Dataset Training

Combine Hugging Face datasets with local data:

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

This configuration will sample 70% from the Hugging Face dataset and 30% from local data.
