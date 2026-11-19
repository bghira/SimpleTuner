# ScaleEdit-12M: paired Parquet images with Webshart metadata

## Details

- **Source dataset**: [InternVL-U/ScaleEdit-12M](https://huggingface.co/datasets/InternVL-U/ScaleEdit-12M).
- **Metadata companion**: [webshart/ScaleEdit-12M-metadata](https://huggingface.co/datasets/webshart/ScaleEdit-12M-metadata).
- **Description**: 11,845,023 image-editing rows across 251 Parquet shards and 23 task categories. The companion contains about 423 MB of metadata; image bytes remain in the upstream dataset.
- **Caption format**: the `edit_instruction` string. These are editing instructions, not standalone scene descriptions.
- **License**: CC-BY-NC-SA-4.0, inherited from the source. Consult the upstream card for image rights and usage restrictions.

## Required preprocessing steps

No tar extraction, per-image caption files, or metadata compilation is required. Use a SimpleTuner revision with the [Webshart Parquet backend](../DATALOADER.md#webshart-datasets), including [PR #3325](https://github.com/bghira/SimpleTuner/pull/3325), and install the released Hub integration:

```bash
python -m pip install --upgrade "webshart[hub]>=0.8.0"
```

Public Hub access works without a token. Set `HF_TOKEN` if you use authenticated access. The companion manifest pins the upstream revision to `05d2eea7b80ba72c077d39be43b7aca1a7857e6a` and verifies metadata checksums.

## Dataset structure

Each source Parquet row contains `source_image`, `edited_image`, `edit_instruction`, both images’ dimensions, and quality scores. `webshart-parquet.json` maps source parts to `metadata/<task>/<part>.parquet`. The metadata files contain no image payloads; joins use the source part path and physical row position because `id` values repeat between task categories.

## Dataloader configuration example

Put this array in `multidatabackend.json`. It starts with the 458-pair `3.4_count_change` category, whose single source shard is about 54 MB. All source and edited images are embedded for this pinned subset. Use an image-editing model supporting `reference_strict`, such as [Flux Kontext](../quickstart/FLUX_KONTEXT.md); the model’s training configuration is separate.

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

## Configuration explained

- `scaleedit-edits` reads `edited_image` as the training target and `edit_instruction` as its caption. Its width and height columns describe the edited image.
- `scaleedit-sources` reads `source_image` as conditioning. `conditioning_data` links the target to it, while `source_dataset_id` links references back to the target. The source width and height columns are selected explicitly.
- Both entries use the same source, companion, and `subfolder`, preserving identical part/row sample identities. Keep their VAE and Webshart cache directories separate.
- `webshart.shard_cache_gb: 0` keeps image access on range reads. Most full-dataset parts are around 31 GB; enabling whole-shard caching can download an entire part for the first payload request.
- `crop: false` avoids independent crops in this starting recipe. `reference_strict` uses the target’s preparation geometry for its reference. Keep preprocessing compatible with the chosen editing model.

## Scaling beyond the example

Before changing or removing `subfolder`, handle the full dataset’s 702,078 URL-only source images. The companion preserves their URLs, SHA-256 values, and fetch dates. It does not fetch them; a missing embedded `source_image` remains `None`. Retrieve and verify those sources explicitly, or build matching source/target subsets using part paths and row positions.

Do not independently limit or filter the target and reference entries: their selected sample sets must stay aligned. In particular, an equal `max_num_samples` value does not by itself guarantee the same pairs when bucketing differs.

Large source row groups and dictionaries can still require hundreds of MiB when reading images, even with the whole-shard cache disabled. Metadata-only bucketing uses the companion, but image access has a separate memory and network cost. The [Webshart loader benchmark](../DATALOADER.md#parquet-image-bytes-on-s3) can measure a bounded sample before a larger run.
