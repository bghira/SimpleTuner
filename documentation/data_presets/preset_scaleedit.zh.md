# ScaleEdit-12M：使用 Webshart 元数据的配对 Parquet 图像

## 详情

- **源数据集**：[InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M)。
- **元数据配套仓库**：[webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata)。
- **说明**：251 个 Parquet 分片、23 类任务，共 11,845,023 行图像编辑数据。配套元数据约 423 MB；图像字节仍在上游仓库中。
- **描述格式**：`edit_instruction` 字符串。这是编辑指令，不是独立的场景描述。
- **许可证**：继承源数据的 CC-BY-NC-SA-4.0。图像权利和使用限制请参阅上游数据集说明。

## 必要的预处理步骤

无需解压 tar、生成逐图描述文件或编译元数据。请使用包含 [PR #3325](https://github.com/bghira/SimpleTuner/pull/3325) 的 SimpleTuner [Webshart Parquet 后端](../DATALOADER.md#webshart-datasets)，并安装已发布的 Hub 集成：

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

公开 Hub 访问不需要令牌；如需身份验证，可设置 `HF_TOKEN`。配套清单将上游版本固定为 `05d2eea7b80ba72c077d39be43b7aca1a7857e6a`，并校验元数据校验和。

## 数据集结构

每个源 Parquet 行包含 `source_image`、`edited_image`、`edit_instruction`、两张图像的尺寸及质量评分。`webshart-parquet.json` 将源分片映射到 `metadata/<task>/<part>.parquet`。元数据文件不包含图像字节；由于 `id` 在不同任务中重复，连接必须使用源分片路径和物理行号。

## 数据加载器配置示例

将此数组放入 `multidatabackend.json`。示例从包含 458 对图像的 `3.4_count_change` 类别开始，其单个源分片约 54 MB。在固定版本的该子集中，所有源图和编辑图均为内嵌字节。请使用支持 `reference_strict` 的图像编辑模型，例如 [Flux Kontext](../quickstart/FLUX_KONTEXT.md)；模型训练配置需另外设置。

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

## 配置说明

- `scaleedit-edits` 将 `edited_image` 用作训练目标，将 `edit_instruction` 用作描述；宽高列对应编辑后的图像。
- `scaleedit-sources` 将 `source_image` 用作条件。`conditioning_data` 将目标连接到条件集，`source_dataset_id` 将参考图连接回目标集；源图宽高列已显式选择。
- 两个条目使用相同的数据源、配套元数据和 `subfolder`，从而保留一致的分片/行身份。VAE 和 Webshart 缓存目录应分别设置。
- `webshart.shard_cache_gb: 0` 保持范围读取。完整数据集的大部分分片约 31 GB；开启整片缓存可能在首次读取图像时下载整个分片。
- 初始配置使用 `crop: false`，避免独立裁剪。`reference_strict` 使用目标图像的准备几何参数处理参考图；预处理应符合所选编辑模型的要求。

## 扩展到示例之外

更改或移除 `subfolder` 前，必须处理完整数据集中 702,078 个仅提供 URL 的源图像。配套仓库保留 URL、SHA-256 和获取日期，但不会下载它们；缺少内嵌 `source_image` 时仍返回 `None`。请显式下载并验证，或根据分片路径与行号构建匹配的源图/目标图子集。

不要分别限制或筛选目标集与参考集；两者必须选择相同的样本。分桶不同时，设置相同的 `max_num_samples` 并不能保证配对一致。

即使关闭整片缓存，读取图像时，大行组和字典仍可能占用数百 MiB。元数据分桶使用配套仓库，而图像读取有单独的内存和网络开销。较大训练前，可用 [Webshart 加载器基准](../DATALOADER.md#parquet-image-bytes-on-s3) 测量有限样本。
