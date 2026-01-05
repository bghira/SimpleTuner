# Subjects200K

## 详情

- **Hub 链接**: [Yuanshi/Subjects200K](https://huggingface.co/datasets/Yuanshi/Subjects200K)
- **描述**: 超过 20 万张高质量合成图像，带有成对描述与质量评估。每个样本包含同一主体在不同语境下的左右并排图像。
- **字幕格式**: 结构化 JSON，分别提供左右图像的描述
- **特殊特性**: 质量评估分数、合集标签、合成图像

## 数据集结构

Subjects200K 的特殊之处在于：
- 每个 `image` 字段包含 **两张并排合成的图像**（一张宽图）
- 每个样本有 **两个独立字幕**：左图为 `description.description_0`，右图为 `description.description_1`
- 质量评估元数据可用于按图像质量指标过滤
- 图像已按合集整理

示例数据结构：
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

## 直接使用（无需预处理）

不同于需要解压与预处理的数据集，Subjects200K 可以直接从 HuggingFace 使用！数据已按正确格式托管。

只需确保安装了 `datasets` 库：
```bash
pip install datasets
```

## 数据加载器配置

由于每个样本包含两张图像，需要配置 **两个独立的数据集条目**，分别对应合成图的左右两半：

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

### 配置说明

#### 合成图像配置
- `composite_image_config.enabled`: 启用合成图像处理
- `composite_image_config.image_count`: 合成图中的图像数量（左右并排为 2）
- `composite_image_config.select_index`: 选择要提取的图像（0 = 左，1 = 右）

#### 质量过滤
`filter_func` 可用于按以下条件筛选样本：
- `collection`: 仅使用特定合集的图像
- `quality_thresholds`: 质量指标最低分：
  - `compositeStructure`: 两张图像的整体一致性
  - `objectConsistency`: 物体在两张图中的一致性
  - `imageQuality`: 整体图像质量

#### 字幕选择
- 左图使用：`"caption_column": "description.description_0"`
- 右图使用：`"caption_column": "description.description_1"`

### 可定制选项

1. **调整质量阈值**：降低数值（如 4.0）可包含更多图像，提高数值（如 4.8）更为严格

2. **使用不同合集**：将 `"collection": "collection_1"` 改为数据集中可用的其他合集

3. **改变分辨率**：根据训练需求调整 `resolution` 值

4. **禁用过滤**：移除 `filter_func` 部分以使用全部图像

5. **用条目名作字幕**：将字幕列改为 `"description.item"` 以使用主体名称

### 提示

- 数据集首次使用会自动下载并缓存
- 每一“半边”都作为独立数据集处理，等效于样本数量翻倍
- 若需要更多多样性，可为左右两边设置不同质量阈值
- 左右两边的 VAE 缓存目录应不同，以避免冲突
