# Dataloader 配置文件

下面是最基础的 dataloader 配置文件示例，文件名为 `multidatabackend.example.json`。

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": true,
    "crop_style": "center",
    "crop_aspect": "square",
    "resolution": 1024,
    "minimum_image_size": 768,
    "maximum_image_size": 2048,
    "minimum_aspect_ratio": 0.50,
    "maximum_aspect_ratio": 3.00,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "repeats": 0
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  }
]
```

## 配置选项

### `id`

- **说明:** 数据集的唯一标识符。设置后应保持不变，用于与状态跟踪条目关联。

### `disabled`

- **取值:** `true` | `false`
- **说明:** 设为 `true` 时，该数据集在训练期间将被完全跳过。适合在不删除配置的情况下临时排除数据集。
- **注记:** 也接受拼写 `disable`。

### `dataset_type`

- **取值:** `image` | `video` | `audio` | `text_embeds` | `image_embeds` | `conditioning_image_embeds` | `conditioning`
- **说明:** `image`、`video`、`audio` 数据集包含主要训练样本。`text_embeds` 存放文本编码器缓存输出，`image_embeds` 存放 VAE 潜变量（当模型使用时），`conditioning_image_embeds` 存放条件图像嵌入缓存（例如 Wan 2.2 I2V 的 CLIP 视觉特征）。当数据集标记为 `conditioning` 时，可通过 [conditioning_data 选项](#conditioning_data) 与 `image` 数据集配对。
- **注记:** 文本/图像嵌入数据集的定义不同于图像数据集。文本嵌入数据集只存储文本嵌入对象；图像数据集存储训练数据。
- **注记:** 不要在**同一个**数据集中混合图片和视频。请分开配置。

### `default`

- **仅适用于 `dataset_type=text_embeds`**
- 设为 `true` 时，该文本嵌入数据集会作为 SimpleTuner 的文本嵌入缓存位置（例如验证提示词嵌入）。由于它们不与图像数据配对，需要明确的存储位置。

### `cache_dir`

- **仅适用于 `dataset_type=text_embeds` 与 `dataset_type=image_embeds`**
- **说明:** 指定该数据集嵌入缓存文件的存储位置。`text_embeds` 写入文本编码器输出，`image_embeds` 存放 VAE 潜变量。
- **注记:** 这与主图像/视频数据集上的 `cache_dir_vae` 不同，后者用于指定其 VAE 缓存位置。

### `write_batch_size`

- **仅适用于 `dataset_type=text_embeds`**
- **说明:** 单次批量写入的文本嵌入数量。数值越大写入吞吐越高，但内存占用也更高。
- **默认值:** 回退到训练器的 `--write_batch_size` 参数（通常为 128）。

### `text_embeds`

- **仅适用于 `dataset_type=image`**
- 未设置时将使用 `default` 的 text_embeds 数据集。若设置为某个现有 `text_embeds` 数据集的 `id`，则会使用该数据集。用于将特定文本嵌入数据集与图像数据集关联。

### `image_embeds`

- **仅适用于 `dataset_type=image`**
- 未设置时，VAE 输出将存放在图像后端。也可设置为某个 `image_embeds` 数据集的 `id`，将 VAE 输出存放到该位置。用于将 image_embeds 数据集与图像数据关联。

### `conditioning_image_embeds`

- **适用于 `dataset_type=image` 与 `dataset_type=video`**
- 当模型报告 `requires_conditioning_image_embeds` 时，将其设置为某个 `conditioning_image_embeds` 数据集的 `id`，用于存储条件图像嵌入缓存（例如 Wan 2.2 I2V 的 CLIP 视觉特征）。若未设置，SimpleTuner 默认写入 `cache/conditioning_image_embeds/<dataset_id>`，确保不与 VAE 缓存冲突。
- 需要这些嵌入的模型必须通过主流程暴露图像编码器。如果模型无法提供编码器，预处理将会提前失败，而不会静默生成空文件。

#### `cache_dir_conditioning_image_embeds`

- **可选：覆盖条件图像嵌入缓存的保存位置。**
- 当你希望将缓存固定到特定文件系统，或使用专用远程后端（`dataset_type=conditioning_image_embeds`）时设置。省略时将自动使用上述默认路径。

#### `conditioning_image_embed_batch_size`

- **可选：覆盖生成条件图像嵌入时的批大小。**
- 未显式设置时，默认使用 `conditioning_image_embed_batch_size` 训练器参数或 VAE 批大小。

### 音频数据集配置（`dataset_type=audio`）

音频后端支持专用 `audio` 块，使元数据与分桶计算基于时长进行处理。示例：

```json
"audio": {
  "max_duration_seconds": 90,
  "channels": 2,
  "bucket_strategy": "duration",
  "duration_interval": 15,
  "truncation_mode": "beginning"
}
```

- **`bucket_strategy`** – 当前默认是 `duration`，会将片段截断到等间隔桶中，使每 GPU 的采样符合批次计算。
- **`duration_interval`** – 以秒为单位的分桶舍入（未设置时默认 **3**）。例如设为 `15` 时，77 秒的片段会归入 75 秒桶。这样可防止单个超长片段占用过多桶，并强制截断到统一间隔。
- **`max_duration_seconds`** – 超过此时长的片段在元数据发现时会被完全跳过，避免异常长的音轨占用桶。
- **`truncation_mode`** – 规定对齐到桶间隔时保留片段的哪一部分。可选：`beginning`、`end`、`random`（默认：`beginning`）。
- 标准音频设置（声道数、缓存目录）会直接映射到 `simpletuner.helpers.data_backend.factory` 创建的运行时音频后端。刻意避免 padding——片段被截断而不是延长，以保持与 ACE-Step 等扩散训练器的行为一致。

### 音频字幕（Hugging Face）
对 Hugging Face 音频数据集，可以指定哪些列用于组成字幕（提示词），以及哪一列包含歌词：
```json
"config": {
    "audio_caption_fields": ["prompt", "tags"],
    "lyrics_column": "lyrics"
}
```
*   `audio_caption_fields`: 合并多个列形成文本提示词（供文本编码器使用）。
*   `lyrics_column`: 指定歌词列（供歌词编码器使用）。

在元数据发现阶段，加载器会记录每个文件的 `sample_rate`、`num_samples`、`num_channels` 和 `duration_seconds`。CLI 的分桶报告现在以 **samples** 而非 **images** 表示，空数据集诊断将列出当前 `bucket_strategy`/`duration_interval`（以及任何 `max_duration_seconds` 限制），方便你无需深入日志即可调整。

### `type`

- **取值:** `aws` | `local` | `csv` | `huggingface`
- **说明:** 决定此数据集所用的存储后端（本地、csv 或云）。

### `conditioning_type`

- **取值:** `controlnet` | `mask` | `reference_strict` | `reference_loose`
- **说明:** 指定 `conditioning` 数据集的条件类型。
  - **controlnet**: 控制信号训练的 ControlNet 条件输入。
  - **mask**: 修补训练用的二值掩码。
  - **reference_strict**: 严格像素对齐的参考图像（如 Qwen Edit 等编辑模型）。
  - **reference_loose**: 宽松对齐的参考图像。

### `source_dataset_id`

- **仅适用于 `dataset_type=conditioning`** 且 `conditioning_type` 为 `reference_strict`、`reference_loose` 或 `mask`
- **说明:** 将条件数据集与其源图像/视频数据集关联以保证像素对齐。设置后，SimpleTuner 会复制源数据集的元数据以确保条件图像与目标对齐。
- **注记:** 严格对齐模式必需；宽松对齐为可选。

### `conditioning_data`

- **取值:** 条件数据集的 `id` 或 `id` 数组
- **说明:** 如 [ControlNet 指南](CONTROLNET.md) 所述，可通过该选项将 `image` 数据集与 ControlNet 或图像掩码数据配对。
- **注记:** 如果有多个条件数据集，可指定 `id` 数组。训练 Flux Kontext 时，可在条件之间随机切换或拼接输入，用于更高级的多图像合成任务。

### `instance_data_dir` / `aws_data_prefix`

- **Local:** 文件系统中的数据路径。
- **AWS:** 存储桶中的 S3 前缀。

### `caption_strategy`

- **textfile** 要求 image.png 旁边有 image.txt，包含一个或多个用换行分隔的字幕。这些图像+文本对**必须位于同一目录**。
- **instanceprompt** 需要提供 `instance_prompt`，并仅使用该值作为集合中每张图片的字幕。
- **filename** 使用转换和清理后的文件名作为字幕，例如将下划线替换为空格。
- **parquet** 从包含其他图像元数据的 parquet 表中提取字幕。通过 `parquet` 字段配置。参见 [Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets)。

`textfile` 与 `parquet` 都支持多字幕：
- textfile 以换行拆分，每行作为一个字幕。
- parquet 表的字段可以是可迭代类型。

### `metadata_backend`

- **取值:** `discovery` | `parquet` | `huggingface`
- **说明:** 控制 SimpleTuner 在数据集准备阶段如何获取图像尺寸和其他元数据。
  - **discovery**（默认）: 扫描实际图像文件读取尺寸。适用于任何存储后端，但大数据集可能较慢。
  - **parquet**: 从 parquet/JSONL 文件的 `width_column` 与 `height_column` 读取尺寸，跳过文件访问。参见 [Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets)。
  - **huggingface**: 使用 Hugging Face 数据集提供的元数据。参见 [Hugging Face Datasets Support](#hugging-face-datasets-support)。
- **注记:** 使用 `parquet` 时必须配置包含 `width_column` 与 `height_column` 的 `parquet` 块。这能显著提升大型数据集的启动速度。

### `metadata_update_interval`

- **取值:** 整数（秒）
- **说明:** 训练期间刷新数据集元数据的频率（秒）。适用于长时间训练中可能变化的数据集。
- **默认值:** 回退到训练器的 `--metadata_update_interval` 参数。

### 裁剪选项

- `crop`: 启用或禁用图像裁剪。
- `crop_style`: 选择裁剪方式（`random`、`center`、`corner`、`face`）。
- `crop_aspect`: 选择裁剪纵横比（`closest`、`random`、`square` 或 `preserve`）。
- `crop_aspect_buckets`: 当 `crop_aspect` 为 `closest` 或 `random` 时，从该列表中选择桶。默认可用全部桶（允许无限放大）。如需限制放大请使用 `max_upscale_threshold`。

### `resolution`

- **resolution_type=area:** 最终图像尺寸按百万像素数确定，这里 1.05 对应 1024^2（1024x1024）左右的总像素面积，约 1_050_000 像素。
- **resolution_type=pixel_area:** 与 `area` 类似，但以像素而非百万像素计。这里设为 1024 时，对应 1024^2（1024x1024）左右的总像素面积，约 1_050_000 像素。
- **resolution_type=pixel:** 最终图像尺寸由较短边为该值来决定。

> **注记**: 图像是否被放大、缩小或裁剪，取决于 `minimum_image_size`、`maximum_target_size`、`target_downsample_size`、`crop` 和 `crop_aspect` 的设置。

### `minimum_image_size`

- 尺寸低于该值的图像将被**排除**在训练之外。
- 当 `resolution` 以百万像素计量（`resolution_type=area`）时，这里也应使用百万像素单位（例如 `1.05` 以排除 1024x1024 **面积**以下的图像）。
- 当 `resolution` 以像素计量时，请使用相同单位（例如 `1024` 以排除短边小于 1024px 的图像）。
- **建议:** 除非想承担低质量放大图像的风险，否则应将 `minimum_image_size` 设为 `resolution`。

### `minimum_aspect_ratio`

- **说明:** 图像的最小纵横比。低于该值的图像将被排除。
- **注记**: 若排除数量过多，训练器在启动时可能会尝试扫描并分桶，从而浪费时间。

> **注记**: 一旦数据集的纵横比与元数据列表建立完成，使用 `skip_file_discovery="vae aspect metadata"` 可避免启动时扫描，节省大量时间。

### `maximum_aspect_ratio`

- **说明:** 图像的最大纵横比。高于该值的图像将被排除。
- **注记**: 若排除数量过多，训练器在启动时可能会尝试扫描并分桶，从而浪费时间。

> **注记**: 一旦数据集的纵横比与元数据列表建立完成，使用 `skip_file_discovery="vae aspect metadata"` 可避免启动时扫描，节省大量时间。

### `conditioning`

- **取值:** 条件配置对象数组
- **说明:** 从源图像自动生成条件数据集。每种条件类型都会创建一个独立数据集，可用于 ControlNet 等条件训练任务。
- **注记:** 指定后，SimpleTuner 会自动创建 ID 类似 `{source_id}_conditioning_{type}` 的条件数据集。

每个条件对象可包含:
- `type`: 要生成的条件类型（必需）
- `params`: 类型特定参数（可选）
- `captions`: 生成数据集的字幕策略（可选）
  - 可为 `false`（无字幕）
  - 单个字符串（作为所有图像的 instance prompt）
  - 字符串数组（每张图随机选取）
  - 省略时使用源数据集字幕

#### 可用的条件类型

##### `superresolution`
生成用于超分辨率训练的低质量图像版本：
```json
{
  "type": "superresolution",
  "blur_radius": 2.5,
  "blur_type": "gaussian",
  "add_noise": true,
  "noise_level": 0.03,
  "jpeg_quality": 85,
  "downscale_factor": 2
}
```

##### `jpeg_artifacts`
为伪影去除训练生成 JPEG 压缩伪影：
```json
{
  "type": "jpeg_artifacts",
  "quality_mode": "range",
  "quality_range": [10, 30],
  "compression_rounds": 1,
  "enhance_blocks": false
}
```

##### `depth` / `depth_midas`
使用 DPT 模型生成深度图：
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**注记:** 深度生成需要 GPU，并在主进程中运行，可能比基于 CPU 的生成器更慢。

##### `random_masks` / `inpainting`
为修补训练生成随机掩码：
```json
{
  "type": "random_masks",
  "mask_types": ["rectangle", "circle", "brush", "irregular"],
  "min_coverage": 0.1,
  "max_coverage": 0.5,
  "output_mode": "mask"
}
```

##### `canny` / `edges`
生成 Canny 边缘检测图：
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

关于如何使用这些条件数据集的更多信息，请参见 [ControlNet 指南](CONTROLNET.md)。

#### 示例

##### 视频数据集

视频数据集应为包含（例如 mp4）视频文件的文件夹，并使用常规字幕存储方式。

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- 在 `video` 子部分中可设置以下键：
  - `num_frames`（可选，int）是用于训练的帧数。
    - 在 25 fps 下，125 帧约等于 5 秒视频，是标准输出，应作为目标。
  - `min_frames`（可选，int）指定用于训练的最短视频长度。
    - 该值应至少等于 `num_frames`，未设置时会与其相同。
  - `max_frames`（可选，int）指定用于训练的最长视频长度。
  - `is_i2v`（可选，bool）指定数据集中是否进行 i2v 训练。
    - LTX 默认设置为 True，但你可以禁用。
  - `bucket_strategy`（可选，string）指定视频分桶方式：
    - `aspect_ratio`（默认）：仅按空间纵横比分桶（如 `1.78`、`0.75`）。行为与图像数据集相同。
    - `resolution_frames`：按分辨率与帧数以 `WxH@F` 格式分桶（如 `1920x1080@125`）。适用于分辨率与时长变化较大的数据集。
  - `frame_interval`（可选，int）：当使用 `bucket_strategy: "resolution_frames"` 时，帧数会向下舍入到该值的最近倍数。应设置为模型要求的帧数因子（某些模型要求 `num_frames - 1` 可被某值整除）。

**自动帧数调整:** SimpleTuner 自动调整视频帧数以满足特定的模型约束。例如，LTX-2 要求帧数满足 `frames % 8 == 1`（如 49、57、65、73、81 等）。如果你的视频帧数不同（如 119 帧），会自动调整至最近的有效帧数（如 113 帧）。调整后短于 `min_frames` 的视频会被跳过并显示警告信息。此自动调整可防止训练错误，无需任何配置。

**注记:** 当使用 `bucket_strategy: "resolution_frames"` 且设置 `num_frames` 时，会得到单一帧桶，且短于 `num_frames` 的视频会被丢弃。若希望更多帧桶且减少丢弃，请取消设置 `num_frames`。

混合分辨率视频数据集使用 `resolution_frames` 分桶的示例：

```json
{
  "id": "mixed-resolution-videos",
  "type": "local",
  "dataset_type": "video",
  "resolution": 720,
  "resolution_type": "pixel_area",
  "instance_data_dir": "datasets/videos",
  "video": {
      "bucket_strategy": "resolution_frames",
      "frame_interval": 25,
      "min_frames": 25,
      "max_frames": 250
  }
}
```

此配置将创建诸如 `1280x720@100`、`1920x1080@125`、`640x480@75` 等桶。视频按训练分辨率与帧数（向最近的 25 帧舍入）分组。


##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Outcome
- 短边小于 **1024px** 的图像会被完全排除出训练。
- `768x1024` 或 `1280x768` 会被排除，但 `1760x1024` 和 `1024x1024` 不会。
- 因为 `minimum_image_size` 等于 `resolution`，不会进行上采样。

##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # different from the above configuration, which is 'pixel'
```
##### Outcome
- 图像的总面积（宽 * 高）小于最小面积（1024 * 1024）时会被排除。
- 例如 `1280x960` 的面积 `(1280 * 960)` 大于 `(1024 * 1024)`，因此**不会**被排除。
- 因为 `minimum_image_size` 等于 `resolution`，不会进行上采样。

##### Configuration
```json
    "minimum_image_size": 0, # or completely unset, not present in the config
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Outcome
- 图像将保持纵横比并缩放到短边 1024px。
- 不会因尺寸而排除图像。
- 小图会通过简单的 `PIL.resize` 方法上采样，效果不佳。
  - 建议在开始训练前使用你选定的超分工具手动放大，避免自动上采样。

### `maximum_image_size` 与 `target_downsample_size`

除非同时设置 `maximum_image_size` 与 `target_downsample_size`，否则图像不会在裁剪前调整大小。也就是说，`4096x4096` 图像会直接裁剪成 `1024x1024`，这可能不理想。

- `maximum_image_size` 指定开始缩放的阈值。超过该值的图像会在裁剪前下采样。
- `target_downsample_size` 指定重采样后、裁剪前的图像大小。

#### 示例

##### Configuration
```json
    "resolution_type": "pixel_area",
    "resolution": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1280,
    "crop": true,
    "crop_aspect": "square"
```

##### Outcome
- 像素面积大于 `(1536 * 1536)` 的图像会在保持原始纵横比的情况下缩放到约 `(1280 * 1280)` 的面积
- 最终图像会被随机裁剪到 `(1024 * 1024)` 的像素面积
- 适合对 2000 万像素等高分辨率数据集训练，在裁剪前先显著缩小，避免裁剪时丢失过多场景上下文（比如只裁到人物背后的瓷砖或模糊背景）

### `max_upscale_threshold`

默认情况下，SimpleTuner 会将小图放大到目标分辨率，可能造成质量下降。`max_upscale_threshold` 可用于限制这种放大行为。

- **默认值**: `null`（允许无限放大）
- **设置后**: 过滤掉需要超过阈值放大的纵横比桶
- **取值范围**: 0 到 1（例如 `0.2` 表示最多允许 20% 放大）
- **适用范围**: 当 `crop_aspect` 为 `closest` 或 `random` 时的纵横比桶选择

#### 示例

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": null
```

##### Outcome
- 所有纵横比桶都可选
- 256x256 图像可以放大到 1024x1024（4 倍）
- 对非常小的图像可能导致质量下降

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": 0.2
```

##### Outcome
- 仅允许需要 ≤20% 放大的纵横比桶
- 256x256 图像尝试放大到 1024x1024（4 倍 = 300% 放大）将没有可用桶
- 850x850 图像可使用所有桶，因为 1024/850 ≈ 1.2（20% 放大）
- 通过排除劣质放大的图像来帮助保持训练质量

---

### `prepend_instance_prompt`

- 启用后，所有字幕都会在开头加入 `instance_prompt` 的值。

### `only_instance_prompt`

- 除 `prepend_instance_prompt` 外，还会将数据集中所有字幕替换为单一短语或触发词。

### `repeats`

- 指定一个 epoch 内数据集中所有样本被看到的次数。用于加强小数据集影响力或最大化 VAE 缓存对象利用率。
- 若有 1000 张图像的数据集与 100 张图像的数据集，通常会给后者设置 **9 或更高** 的 repeats，使其总采样量达到 1000。

> ℹ️ 该值与 Kohya 脚本中的同名选项行为不同，在 Kohya 中 1 表示不重复，**而在 SimpleTuner 中 0 表示不重复**。从 Kohya 配置值减去 1 即可得到 SimpleTuner 的等效值，因此计算 `(dataset_length + repeats * dataset_length)` 会得到 **9**。

#### 多 GPU 训练与数据集规模

使用多 GPU 训练时，数据集必须足以满足**有效批大小**，计算如下：

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

例如，4 张 GPU、`train_batch_size=4`、`gradient_accumulation_steps=1` 时，每个纵横比桶至少需要 **16 个样本**（应用 repeats 后）。

**重要:** 若数据集配置导致可用批次数为 0，SimpleTuner 会报错。错误信息将显示：
- 当前配置值（批大小、GPU 数、repeats）
- 样本不足的纵横比桶
- 每个桶所需的最小 repeats
- 建议的解决方案
- 建议的解决方案

##### 自动数据集超订

当数据集小于有效批大小时，可使用 `--allow_dataset_oversubscription` 标志自动调整 `repeats`（见 [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)）。

启用后，SimpleTuner 将：
- 计算训练所需的最小 repeats
- 自动增加 `repeats` 以满足要求
- 记录一条警告日志说明调整
- **尊重手动设置的 repeats** —— 若在数据集配置中显式设置了 `repeats`，则跳过自动调整

在以下场景尤其有用：
- 训练小数据集（< 100 张）
- 使用较多 GPU 但数据集较小
- 试验不同 batch 大小时不想反复修改数据集配置

**示例场景：**
- 数据集：25 张
- 配置：8 GPU，`train_batch_size=4`，`gradient_accumulation_steps=1`
- 有效批大小：需要 32 个样本
- 不启用超订：报错
- 启用 `--allow_dataset_oversubscription`：自动将 repeats 设为 1（25 × 2 = 50 样本）

### `start_epoch` / `start_step`

- 规划数据集开始采样的时间点。
- `start_epoch`（默认：`1`）以 epoch 为门槛；`start_step`（默认：`0`）以优化器步数（梯度累积后）为门槛。两者条件必须同时满足才会抽取样本。
- 至少有一个数据集需要满足 `start_epoch<=1` **且** `start_step<=1`；否则启动时无数据可用，训练将报错。
- 永远达不到开始条件的数据集（例如 `start_epoch` 超过 `--num_train_epochs`）会被跳过，并在模型卡中注明。
- 当计划的数据集在训练中途激活时，epoch 长度可能增加，因此进度条步数估计为近似值。

### `is_regularisation_data`

- 也可写作 `is_regularization_data`
- 为 LyCORIS 适配器启用父教师训练，使该数据集的预测目标更倾向于基础模型的结果。
  - 目前不支持标准 LoRA。

### `delete_unwanted_images`

- **取值:** `true` | `false`
- **说明:** 启用后，无法通过尺寸或纵横比过滤（例如低于 `minimum_image_size` 或超出 `minimum_aspect_ratio`/`maximum_aspect_ratio`）的图像会从数据集目录中永久删除。
- **警告:** 此操作不可恢复，请谨慎使用。
- **默认值:** 回退到训练器的 `--delete_unwanted_images` 参数（默认：`false`）。

### `delete_problematic_images`

- **取值:** `true` | `false`
- **说明:** 启用后，VAE 编码失败的图像（损坏文件、不支持格式等）会从数据集目录中永久删除。
- **警告:** 此操作不可恢复，请谨慎使用。
- **默认值:** 回退到训练器的 `--delete_problematic_images` 参数（默认：`false`）。

### `slider_strength`

- **取值:** 任意浮点值（正、负或零）
- **说明:** 将数据集标记为 slider LoRA 训练，用于学习对比“相反概念”，从而创建可控概念适配器。
  - **正值**（例如 `0.5`）：“更强的概念”——更亮的眼睛、更明显的微笑等。
  - **负值**（例如 `-0.5`）：“更弱的概念”——更暗的眼睛、更中性的表情等。
  - **零或省略**：中性样本，不在任何方向上推动概念。
- **注记:** 当数据集包含 `slider_strength` 值时，SimpleTuner 会按固定循环旋转批次：正 → 负 → 中性。在每个组内仍应用标准后端概率。
- **另见:** [SLIDER_LORA.md](SLIDER_LORA.md) 提供 slider LoRA 训练的完整指南。

### `vae_cache_clear_each_epoch`

- 启用后，在每个数据集重复周期结束时，所有 VAE 缓存对象都会从文件系统中删除。对大型数据集而言这可能开销较高，但配合 `crop_style=random` 和/或 `crop_aspect=random` 使用时建议启用，以确保从每张图像中采样到完整的裁剪范围。
- 实际上，在使用随机分桶或随机裁剪时该选项**默认启用**。

### `vae_cache_disable`

- **取值:** `true` | `false`
- **说明:** 启用后（通过命令行参数 `--vae_cache_disable`），会隐式启用按需 VAE 缓存，同时禁止将生成的嵌入写入磁盘。这适用于磁盘空间紧张或写入不便的大型数据集。
- **注记:** 这是训练器级别的参数，而非逐数据集配置，但会影响 dataloader 与 VAE 缓存的交互方式。

### `skip_file_discovery`

- 通常不建议设置，仅在超大数据集场景有用。
- 该参数接受逗号或空格分隔的值列表，例如 `vae metadata aspect text`，用于跳过加载配置的一个或多个阶段的文件发现。
- 等同于命令行选项 `--skip_file_discovery`。
- 若你的数据集不需要训练器在每次启动时扫描（例如其潜变量/嵌入已完全缓存），该选项可加快启动与恢复。

### `preserve_data_backend_cache`

- 通常不建议设置，仅在超大 AWS 数据集场景有用。
- 与 `skip_file_discovery` 类似，该选项可避免启动时不必要、耗时且成本高的文件系统扫描。
- 这是一个布尔值，设为 `true` 时，生成的文件系统列表缓存不会在启动时删除。
- 对于 S3 或本地 SMR 机械硬盘等响应极慢的存储系统尤其有用。
- 另外在 S3 上，列举后端对象本身就会产生费用，应尽量避免。

> ⚠️ **如果数据正在持续变动，则不能设置此项。** 训练器将无法看到新增数据，必须再次进行完整扫描。

### `hash_filenames`

- VAE 缓存条目的文件名始终会进行哈希处理。这不可由用户配置，可避免超长文件名导致的路径长度问题。配置中的 `hash_filenames` 会被忽略。

## 过滤字幕

### `caption_filter_list`

- **仅适用于文本嵌入数据集。** 可以是 JSON 列表、txt 文件路径或 JSON 文档路径。过滤字符串可以是要从所有字幕中移除的普通词，也可以是正则表达式。另外还支持 sed 风格的 `s/search/replace/` 语法，用于在字幕中**替换**字符串而非删除。

#### 过滤列表示例

完整示例见 [这里](/config/caption_filter_list.txt.example)。其中包含 BLIP（通用版）、LLaVA、CogVLM 常见的重复或负面字符串。

以下是一个简化示例，后续将解释：

```
arafed
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

各行行为如下：

- `arafed `（末尾带空格）会从任意字幕中移除。保留末尾空格能避免双空格残留，效果更好。虽非必要，但更美观。
- `this .* has a` 是正则表达式，会移除包含 “this ... has a” 的任意字符串；`.*` 表示“匹配所有内容”，直到遇到 “has a” 才停止。
- `^this is the beginning of the string` 会移除短语 “this is the beginning of the string”，但仅在其位于字幕开头时生效。
- `s/this/will be found and replaced/` 会将字幕中第一个 “this” 替换为 “will be found and replaced”。

> ❗调试与测试正则表达式可使用 [regex 101](https://regex101.com)。

# 高级技巧

## 高级配置示例

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": false,
    "crop_style": "random|center|corner|face",
    "crop_aspect": "square|preserve|closest|random",
    "crop_aspect_buckets": [0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "resolution": 1.0,
    "resolution_type": "area|pixel",
    "minimum_image_size": 1.0,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "start_epoch": 1,
    "start_step": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example",
    "conditioning_image_embeds": "conditioning-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir_vae": "s3prefix/for/vaecache",
    "vae_cache_clear_each_epoch": true,
    "repeats": 0
  },
  {
      "id": "vae-embeds-example",
      "type": "local",
      "dataset_type": "image_embeds",
      "disabled": false,
  },
  {
      "id": "conditioning-embeds-example",
      "type": "local",
      "dataset_type": "conditioning_image_embeds",
      "disabled": false
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": false,
    "type": "local",
    "cache_dir": "/path/to/textembed_cache"
  }
]
```

## 直接从 CSV URL 列表训练

**注记：注意 CSV 必须包含图像字幕。**

> ⚠️ 这是一个高级且**实验性**功能，可能会遇到问题。如遇问题，请提交 [issue](https://github.com/bghira/simpletuner/issues)。

你可以不手动下载 URL 列表中的数据，而是直接交给训练器处理。

**注记:** 通常仍建议手动下载图像数据。若想节省本地磁盘空间，也可以尝试 [云端数据集 + 本地编码器缓存](#local-cache-with-cloud-dataset)。

### 优点

- 无需直接下载数据
- 可使用 SimpleTuner 的字幕工具直接为 URL 列表生成字幕
- 节省磁盘空间，因为只存储图像嵌入（如适用）与文本嵌入

### 缺点

- 需要耗时且可能昂贵的纵横比分桶扫描，会下载每张图像并收集其元数据
- 下载的图像会被缓存到磁盘，可能变得很大。目前缓存管理仍较基础，为只写不删
- 若数据集中有大量无效 URL，当前坏样本**不会**从 URL 列表移除，恢复训练时可能浪费时间
  - **建议:** 预先运行 URL 校验并删除无效样本。

### 配置

必需键：

- `type: "csv"`
- `csv_caption_column`
- `csv_cache_dir`
- `caption_strategy: "csv"`

```json
[
    {
        "id": "csvtest",
        "type": "csv",
        "csv_caption_column": "caption",
        "csv_file": "/Volumes/ml/dataset/test_list.csv",
        "csv_cache_dir": "/Volumes/ml/cache/csv/test",
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl",
        "caption_strategy": "csv",
        "image_embeds": "image-embeds",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel",
        "minimum_image_size": 0,
        "disabled": false,
        "skip_file_discovery": "",
        "preserve_data_backend_cache": false
    },
    {
      "id": "image-embeds",
      "type": "local"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/Volumes/ml/cache/text/sdxl",
        "disabled": false,
        "preserve_data_backend_cache": false,
        "skip_file_discovery": "",
        "write_batch_size": 128
    }
]
```

## Parquet 字幕策略 / JSON Lines 数据集

> ⚠️ 这是高级功能，大多数用户不需要。

当训练拥有数十万或数百万图像的超大数据集时，将元数据存入 parquet 数据库比 txt 文件更快——尤其当训练数据在 S3 存储桶中时。

使用 parquet 字幕策略可将文件命名为其 `id` 值，并通过配置值切换字幕列，而无需更新大量文本文件或重命名文件。

以下是使用 [photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket) 数据集字幕与数据的 dataloader 配置示例：

```json
{
  "id": "photo-concept-bucket",
  "type": "local",
  "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
  "caption_strategy": "parquet",
  "metadata_backend": "parquet",
  "parquet": {
    "path": "photo-concept-bucket.parquet",
    "filename_column": "id",
    "caption_column": "cogvlm_caption",
    "fallback_caption_column": "tags",
    "width_column": "width",
    "height_column": "height",
    "identifier_includes_extension": false
  },
  "resolution": 1.0,
  "minimum_image_size": 0.75,
  "maximum_image_size": 2.0,
  "target_downsample_size": 1.5,
  "prepend_instance_prompt": false,
  "instance_prompt": null,
  "only_instance_prompt": false,
  "disable": false,
  "cache_dir_vae": "/models/training/vae_cache/photo-concept-bucket",
  "probability": 1.0,
  "skip_file_discovery": "",
  "preserve_data_backend_cache": false,
  "vae_cache_clear_each_epoch": true,
  "repeats": 1,
  "crop": true,
  "crop_aspect": "closest",
  "crop_style": "random",
  "crop_aspect_buckets": [1.0, 0.75, 1.23],
  "resolution_type": "area"
}
```

在该配置中：

- `caption_strategy` 设为 `parquet`。
- `metadata_backend` 设为 `parquet`。
- 必须定义新的 `parquet` 区块：
  - `path` 是 parquet 或 JSONL 文件路径。
  - `filename_column` 是表中包含文件名的列名。此处使用数值 `id` 列（推荐）。
  - `caption_column` 是表中包含字幕的列名。此处使用 `cogvlm_caption` 列。对 LAION 数据集则为 TEXT 字段。
  - `width_column` 与 `height_column` 可为包含字符串、整数或单条 Series 数据类型的列，用于表示实际图像尺寸。这可以显著提升数据集准备速度，因为无需访问真实图像即可获知尺寸。
  - `fallback_caption_column` 是可选的备用字幕列名，当主字幕为空时使用。本例使用 `tags` 列。
  - `identifier_includes_extension` 当文件名列包含扩展名时应设为 `true`，否则默认扩展名为 `.png`。建议在文件名列中包含扩展名。

> ⚠️ Parquet 支持仅限于读取字幕。你需要用 `{id}.png` 的文件名单独填充图像样本数据源。可参考 [scripts/toolkit/datasets](scripts/toolkit/datasets) 目录中的脚本。

与其他 dataloader 配置类似：

- `prepend_instance_prompt` 与 `instance_prompt` 正常工作。
- 在两次训练之间更新样本字幕会缓存新的嵌入，但不会删除旧的（孤立）单元。
- 当数据集中不存在图像时，其文件名会被用作字幕，并记录错误。

## 云端数据集与本地缓存

为了最大化利用昂贵的本地 NVMe 存储，你可以将图像文件（png、jpg）存放在 S3 存储桶中，并使用本地存储来缓存文本编码器和 VAE（如适用）的特征图。

该配置示例中：

- 图像数据存放在 S3 兼容存储桶
- VAE 数据存放在 /local/path/to/cache/vae
- 文本嵌入存放在 /local/path/to/cache/textencoder

> ⚠️ 记得配置其他数据集选项，例如 `resolution` 与 `crop`。

```json
[
    {
        "id": "data",
        "type": "aws",
        "aws_bucket_name": "text-vae-embeds",
        "aws_endpoint_url": "https://storage.provider.example",
        "aws_access_key_id": "exampleAccessKey",
        "aws_secret_access_key": "exampleSecretKey",
        "aws_region_name": null,
        "cache_dir_vae": "/local/path/to/cache/vae/",
        "caption_strategy": "parquet",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "train.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        },
        "preserve_data_backend_cache": false,
        "image_embeds": "vae-embed-storage"
    },
    {
        "id": "vae-embed-storage",
        "type": "local",
        "dataset_type": "image_embeds"
    },
    {
        "id": "text-embed-storage",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/local/path/to/cache/textencoder/",
        "write_batch_size": 128
    }
]
```

**注记:** `image_embeds` 数据集没有用于设置数据路径的选项。它们通过图像后端的 `cache_dir_vae` 配置。

### Hugging Face 数据集支持

SimpleTuner 现在支持直接从 Hugging Face Hub 加载数据集，无需完整下载到本地。该实验性功能适用于：

- 托管在 Hugging Face 上的大规模数据集
- 内置元数据与质量评估的数据集
- 无需本地存储即可快速试验

有关此功能的完整文档请参见 [此文档](HUGGINGFACE_DATASETS.md)。

使用 Hugging Face 数据集的基本示例：在 dataloader 配置中设置 `"type": "huggingface"`：

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image"
}
```

## 自定义纵横比到分辨率映射

SimpleTuner 首次启动时，会生成分辨率专用的纵横比映射列表，将十进制纵横比值映射到目标像素尺寸。

你可以创建自定义映射，强制训练器按你选择的目标分辨率进行调整，而不是使用自身计算结果。此功能需自行承担风险，若配置错误可能造成严重后果。

创建自定义映射：

- 按照下面的示例创建文件
- 使用 `aspect_ratio_map-{resolution}.json` 格式命名
  - 若配置值为 `resolution=1.0` / `resolution_type=area`，映射文件名为 `aspect_resolution_map-1.0.json`
- 将文件放置在 `--output_dir` 指定的位置
  - 该位置与检查点与验证图像存放位置相同
- 不需要额外配置标志或选项。只要名称与位置正确，即会自动发现并使用。

### 映射配置示例

这是 SimpleTuner 生成的纵横比映射示例。你无需手动配置，但若想完全控制输出分辨率，可将其作为修改起点。

- 数据集包含超过 100 万张图像
- dataloader `resolution` 设为 `1.0`
- dataloader `resolution_type` 设为 `area`

这是最常见的配置，也是 1 百万像素模型可训练的纵横比桶列表。

```json
{
    "0.07": [320, 4544],    "0.38": [640, 1664],    "0.88": [960, 1088],    "1.92": [1472, 768],    "3.11": [1792, 576],    "5.71": [2560, 448],
    "0.08": [320, 3968],    "0.4": [640, 1600],     "0.89": [1024, 1152],   "2.09": [1472, 704],    "3.22": [1856, 576],    "6.83": [2624, 384],
    "0.1": [320, 3328],     "0.41": [704, 1728],    "0.94": [1024, 1088],   "2.18": [1536, 704],    "3.33": [1920, 576],    "7.0": [2688, 384],
    "0.11": [384, 3520],    "0.42": [704, 1664],    "1.06": [1088, 1024],   "2.27": [1600, 704],    "3.44": [1984, 576],    "8.0": [3072, 384],
    "0.12": [384, 3200],    "0.44": [704, 1600],    "1.12": [1152, 1024],   "2.5": [1600, 640],     "3.88": [1984, 512],
    "0.14": [384, 2688],    "0.46": [704, 1536],    "1.13": [1088, 960],    "2.6": [1664, 640],     "4.0": [2048, 512],
    "0.15": [448, 3008],    "0.48": [704, 1472],    "1.2": [1152, 960],     "2.7": [1728, 640],     "4.12": [2112, 512],
    "0.16": [448, 2816],    "0.5": [768, 1536],     "1.36": [1216, 896],    "2.8": [1792, 640],     "4.25": [2176, 512],
    "0.19": [448, 2304],    "0.52": [768, 1472],    "1.46": [1216, 832],    "3.11": [1792, 576],    "4.38": [2240, 512],
    "0.24": [512, 2112],    "0.55": [768, 1408],    "1.54": [1280, 832],    "3.22": [1856, 576],    "5.0": [2240, 448],
    "0.26": [512, 1984],    "0.59": [832, 1408],    "1.83": [1408, 768],    "3.33": [1920, 576],    "5.14": [2304, 448],
    "0.29": [576, 1984],    "0.62": [832, 1344],    "1.92": [1472, 768],    "3.44": [1984, 576],    "5.71": [2560, 448],
    "0.31": [576, 1856],    "0.65": [832, 1280],    "2.09": [1472, 704],    "3.88": [1984, 512],    "6.83": [2624, 384],
    "0.34": [640, 1856],    "0.68": [832, 1216],    "2.18": [1536, 704],    "4.0": [2048, 512],     "7.0": [2688, 384],
    "0.38": [640, 1664],    "0.74": [896, 1216],    "2.27": [1600, 704],    "4.12": [2112, 512],    "8.0": [3072, 384],
    "0.4": [640, 1600],     "0.83": [960, 1152],    "2.5": [1600, 640],     "4.25": [2176, 512],
    "0.41": [704, 1728],    "0.88": [960, 1088],    "2.6": [1664, 640],     "4.38": [2240, 512],
    "0.42": [704, 1664],    "0.89": [1024, 1152],   "2.7": [1728, 640],     "5.0": [2240, 448],
    "0.44": [704, 1600],    "0.94": [1024, 1088],   "2.8": [1792, 640],     "5.14": [2304, 448]
}
```

针对 Stable Diffusion 1.5 / 2.0-base（512px）模型，可使用以下映射：

```json
{
    "1.3": [832, 640], "1.0": [768, 768], "2.0": [1024, 512],
    "0.64": [576, 896], "0.77": [640, 832], "0.79": [704, 896],
    "0.53": [576, 1088], "1.18": [832, 704], "0.85": [704, 832],
    "0.56": [576, 1024], "0.92": [704, 768], "1.78": [1024, 576],
    "1.56": [896, 576], "0.67": [640, 960], "1.67": [960, 576],
    "0.5": [512, 1024], "1.09": [768, 704], "1.08": [832, 768],
    "0.44": [512, 1152], "0.71": [640, 896], "1.4": [896, 640],
    "0.39": [448, 1152], "2.25": [1152, 512], "2.57": [1152, 448],
    "0.4": [512, 1280], "3.5": [1344, 384], "2.12": [1088, 512],
    "0.3": [448, 1472], "2.71": [1216, 448], "8.25": [2112, 256],
    "0.29": [384, 1344], "2.86": [1280, 448], "6.2": [1984, 320],
    "0.6": [576, 960]
}
```
