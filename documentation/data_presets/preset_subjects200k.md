# Subjects200K

## Details

- **Hub link**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **Description**: 200K+ high-quality composite images with paired descriptions and quality assessments. Each sample contains two side-by-side images of the same subject in different contexts.
- **Caption format(s)**: Structured JSON with separate descriptions for each image half
- **Special features**: Quality assessment scores, collection tags, composite images

## Dataset Structure

The Subjects200K dataset is unique because:
- Each `image` field contains **two images combined side-by-side** into a single wide image
- Each sample has **two separate captions** - one for the left image (`description.description_0`) and one for the right (`description.description_1`)
- Quality assessment metadata allows filtering by image quality metrics
- Images are pre-organized into collections

Example data structure:
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

## Direct Usage (No Preprocessing Required)

Unlike datasets that require extraction and preprocessing, Subjects200K can be used directly from HuggingFace! The dataset is already properly formatted and hosted.

Simply ensure you have the `datasets` library installed:
```bash
pip install datasets
```

## Dataloader Configuration

Since each sample contains two images, we need to configure **two separate dataset entries** - one for each half of the composite image:

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

### Configuration Explained

#### Composite Image Configuration
- `composite_image_config.enabled`: Activates composite image handling
- `composite_image_config.image_count`: Number of images in the composite (2 for side-by-side)
- `composite_image_config.select_index`: Which image to extract (0 = left, 1 = right)

#### Quality Filtering
The `filter_func` allows you to filter samples based on:
- `collection`: Only use images from specific collections
- `quality_thresholds`: Minimum scores for quality metrics:
  - `compositeStructure`: How well the two images work together
  - `objectConsistency`: Consistency of the subject across both images
  - `imageQuality`: Overall image quality

#### Caption Selection
- Left image uses: `"caption_column": "description.description_0"`
- Right image uses: `"caption_column": "description.description_1"`

### Customization Options

1. **Adjust quality thresholds**: Lower values (e.g., 4.0) include more images, higher values (e.g., 4.8) are more selective

2. **Use different collections**: Change `"collection": "collection_1"` to other available collections in the dataset

3. **Change resolution**: Adjust the `resolution` value based on your training needs

4. **Disable filtering**: Remove the `filter_func` section to use all images

5. **Use item names as captions**: Change caption column to `"description.item"` to use just the subject name

### Tips

- The dataset will be automatically downloaded and cached on first use
- Each "half" is treated as an independent dataset, effectively doubling your training samples
- Consider using different quality thresholds for each half if you want variety
- The VAE cache directories should be different for each half to avoid conflicts