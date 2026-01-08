# Hugging Face 数据集集成

SimpleTuner 支持直接从 Hugging Face Hub 加载数据集，在不下载全部图像到本地的情况下高效训练大规模数据集。

## 概览

Hugging Face 数据集后端允许你：
- 直接从 Hugging Face Hub 加载数据集
- 基于元数据或质量指标进行过滤
- 从数据集列中提取字幕
- 处理合成/网格图像
- 仅在本地缓存处理后的嵌入

**重要**：SimpleTuner 需要完整数据集访问来构建纵横比桶并计算批大小。虽然 Hugging Face 支持流式数据集，但该功能与 SimpleTuner 架构不兼容。请使用过滤将超大数据集缩减到可管理规模。

## 基础配置

要使用 Hugging Face 数据集，在 dataloader 中配置 `"type": "huggingface"`：

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

### 必填字段

- `type`: 必须是 `"huggingface"`
- `dataset_name`: Hugging Face 数据集标识（例如 "laion/laion-aesthetic"）
- `caption_strategy`: 必须是 `"huggingface"`
- `metadata_backend`: 必须是 `"huggingface"`

### 可选字段

- `split`: 使用的数据集分割（默认: "train"）
- `revision`: 特定数据集修订版
- `image_column`: 图像列（默认: "image"）
- `caption_column`: 字幕列
- `cache_dir`: 数据集文件的本地缓存目录
- `streaming`: ⚠️ **当前不可用** - SimpleTuner 需要高效扫描数据集以构建元数据和编码器缓存
- `num_proc`: 过滤处理的进程数（默认: 16）

## 字幕配置

Hugging Face 后端支持灵活的字幕提取：

### 单一字幕列
```json
{
  "caption_column": "caption"
}
```

### 多字幕列
```json
{
  "caption_column": ["short_caption", "detailed_caption", "tags"]
}
```

### 嵌套列访问
```json
{
  "caption_column": "metadata.caption",
  "fallback_caption_column": "basic_caption"
}
```

### 高级字幕配置
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

## 数据集过滤

应用过滤以仅选择高质量样本：

### 基于质量的过滤
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

### 集合/子集过滤
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

## 合成图像支持

处理包含多张图像网格的数据集：

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

该配置将：
- 检测 4 图网格
- 仅提取第一张图像（index 0）
- 相应调整尺寸

## 完整配置示例

### 基础照片数据集
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

### 过滤后的高质量数据集
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

### 视频数据集
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

## 虚拟文件系统

Hugging Face 后端使用虚拟文件系统，图像通过数据集索引引用：
- `0.jpg` → 数据集第 1 项
- `1.jpg` → 数据集第 2 项
- 等等

这使得标准 SimpleTuner 流水线无需修改即可工作。

## 缓存行为

- **数据集文件**：按 Hugging Face datasets 库默认规则缓存
- **VAE 嵌入**：存储在 `cache_dir/vae/{backend_id}/`
- **文本嵌入**：使用标准文本嵌入缓存配置
- **元数据**：存储在 `cache_dir/huggingface_metadata/{backend_id}/`

## 性能注意事项

1. **初始扫描**：首次运行会下载数据集元数据并构建纵横比桶
2. **数据集规模**：必须加载完整元数据以构建文件列表并计算长度
3. **过滤**：在初始加载时应用，过滤后的项不会下载
4. **缓存复用**：后续运行将复用缓存的元数据和嵌入

**注记**：尽管 Hugging Face datasets 支持流式处理，但 SimpleTuner 需要完整数据集访问以构建纵横比桶并计算批大小。超大数据集应过滤到可管理规模。

## 限制

- 只读访问（无法修改源数据集）
- 初次访问需要互联网连接
- 部分数据集格式可能不支持
- 不支持 streaming 模式——SimpleTuner 需要完整数据集访问
- 超大数据集必须过滤到可管理规模
- 初次元数据加载在超大数据集上可能消耗大量内存

## 故障排查

### 数据集未找到
```
Error: Dataset 'username/dataset' not found
```
- 确认数据集存在于 Hugging Face Hub
- 检查数据集是否为私有（需要认证）
- 确保数据集名称拼写正确

### 初始加载缓慢
- 大型数据集加载元数据和构建桶需要时间
- 使用更严格的过滤减少数据集规模
- 考虑使用子集或过滤后的版本
- 缓存文件可加快后续运行

### 内存问题
- 在加载前使用过滤减少数据集规模
- 降低过滤操作的 `num_proc`
- 将超大数据集拆分为更小的块
- 使用质量阈值限制为高质量样本

### 字幕提取问题
- 确认列名与数据集 schema 匹配
- 检查是否存在嵌套列结构
- 使用 `fallback_caption_column` 处理缺失字幕

## 高级用法

### 自定义过滤函数

虽然配置支持基础过滤，但更复杂的过滤可通过修改代码实现。过滤函数会接收每条数据项并返回 True/False。

### 多数据集训练

将 Hugging Face 数据集与本地数据结合：

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

该配置会以 70% 概率采样 Hugging Face 数据集，30% 采样本地数据。

## 音频数据集

对于 ACE-Step 等音频模型，可指定 `dataset_type: "audio"`。

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

*   **`audio_column`**：包含音频数据的列（解码或字节）。默认 `"audio"`。
*   **`audio_caption_fields`**：用于组合成**提示词**（文本条件）的列名列表。默认 `["prompt", "tags"]`。
*   **`lyrics_column`**：包含歌词的列。默认 `"lyrics"`。若缺失，SimpleTuner 会检查 `"norm_lyrics"` 作为回退。

### 预期列
*   **`audio`**：音频数据。
*   **`prompt`** / **`tags`**：用于文本编码器的描述标签或提示词。
*   **`lyrics`**：用于歌词编码器的歌词。
