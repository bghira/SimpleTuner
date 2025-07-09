# Dataloader configuration file

Here is the most basic example of a dataloader configuration file, as `multidatabackend.example.json`.

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

## Configuration Options

### `id`

- **Description:** Unique identifier for the dataset. It should remain constant once set, as it links the dataset to its state tracking entries.

### `dataset_type`

- **Values:** `image` | `video` | `text_embeds` | `image_embeds` | `conditioning`
- **Description:** `image` and `video` datasets contain your training data. `text_embeds` contain the outputs of the text encoder cache, and `image_embeds` contain the VAE outputs, if the model uses one. When a dataset is marked as `conditioning`, it is possible to pair it to your `image` dataset via [the conditioning_data option](#conditioning_data)
- **Note:** Text and image embed datasets are defined differently than image datasets are. A text embed dataset stores ONLY the text embed objects. An image dataset stores the training data.
- **Note:** Don't combine images and video in a **single** dataset. Split them out.

### `default`

- **Only applies to `dataset_type=text_embeds`**
- If set `true`, this text embed dataset will be where SimpleTuner stores the text embed cache for eg. validation prompt embeds. As they do not pair to image data, there needs to be a specific location for them to end up.

### `text_embeds`

- **Only applies to `dataset_type=image`**
- If unset, the `default` text_embeds dataset will be used. If set to an existing `id` of a `text_embeds` dataset, it will use that instead. Allows specific text embed datasets to be associated with a given image dataset.

### `image_embeds`

- **Only applies to `dataset_type=image`**
- If unset, the VAE outputs will be stored on the image backend. Otherwise, you may set this to the `id` of an `image_embeds` dataset, and the VAE outputs will be stored there instead. Allows associating the image_embed dataset to the image data.

### `type`

- **Values:** `aws` | `local` | `csv`
- **Description:** Determines the storage backend (local, csv or cloud) used for this dataset.

### `conditioning_type`

- **Values:** `controlnet` | `mask`
- **Description:** A dataset may contain ControlNet conditioning inputs or masks to use during loss calculations. Only one or the other may be used.

### `conditioning_data`

- **Values:** `id` value of conditioning dataset
- **Description:** As described in [the ControlNet guide](/documentation/CONTROLNET.md), an `image` dataset can be paired to its ControlNet or image mask data via this option.

### `instance_data_dir` / `aws_data_prefix`

- **Local:** Path to the data on the filesystem.
- **AWS:** S3 prefix for the data in the bucket.

### `caption_strategy`

- **textfile** requires your image.png be next to an image.txt that contains one or more captions, separated by newlines. These image+text pairs **must be in the same directory**.
- **instanceprompt** requires a value for `instance_prompt` also be provided, and will use **only** this value for the caption of every image in the set.
- **filename** will use a converted and cleaned-up version of the filename as its caption, eg. after swapping underscores for spaces.
- **parquet** will pull captions from the parquet table that contains the rest of the image metadata. use the `parquet` field to configure this. See [Parquet caption strategy](#parquet-caption-strategy--json-lines-datasets).

Both `textfile` and `parquet` support multi-captions:
- textfiles are split by newlines. Each new line will be its own separate caption.
- parquet tables can have an iterable type in the field.

### Cropping Options

- `crop`: Enables or disables image cropping.
- `crop_style`: Selects the cropping style (`random`, `center`, `corner`, `face`).
- `crop_aspect`: Chooses the cropping aspect (`closest`, `random`, `square` or `preserve`).
- `crop_aspect_buckets`: When `crop_aspect` is set to `closest` or `random`, a bucket from this list will be selected, so long as the resulting image size would not result more than 20% upscaling.

### `resolution`

- **resolution_type=area:** The final image size is determined by megapixel count - a value of 1.05 here will correspond to aspect buckets around 1024^2 (1024x1024) total pixel area, ~1_050_000 pixels.
- **resolution_type=pixel_area:** Like `area`, the final image size is by its area, but measures in pixels rather than megapixels. A value of 1024 here will generate aspect buckets around 1024^2 (1024x1024) total pixel area, ~1_050_000 pixels.
- **resolution_type=pixel:** The final image size will be determined by the smaller edge being this value.

> **NOTE**: Whether images are upscaled, downscaled, or cropped, rely on the values of `minimum_image_size`, `maximum_target_size`, `target_downsample_size`, `crop`, and `crop_aspect`.

### `minimum_image_size`

- Any images whose size ends up falling underneath this value will be **excluded** from training.
- When `resolution` is measured in megapixels (`resolution_type=area`), this should be in megapixels too (eg. `1.05` megapixels to exclude images under 1024x1024 **area**)
- When `resolution` is measured in pixels, you should use the same unit here (eg. `1024` to exclude images under 1024px **shorter edge length**)
- **Recommendation**: Keep `minimum_image_size` equal to `resolution` unless you want to risk training on poorly-upsized images.

### `minimum_aspect_ratio`

- **Description:** The minimum aspect ratio of the image. If the image's aspect ratio is less than this value, it will be excluded from training.
- **Note**: If the number of images qualifying for exclusion is excessive, this might waste time at startup as the trainer will try to scan them and bucket if they are missing from the bucket lists.

> **Note**: Once the aspect and metadata lists are built for your dataset, using `skip_file_discovery="vae aspect metadata"` will prevent the trainer from scanning the dataset on startup, saving a lot of time.

### `maximum_aspect_ratio`

- **Description:** The maximum aspect ratio of the image. If the image's aspect ratio is greater than this value, it will be excluded from training.
- **Note**: If the number of images qualifying for exclusion is excessive, this might waste time at startup as the trainer will try to scan them and bucket if they are missing from the bucket lists.

> **Note**: Once the aspect and metadata lists are built for your dataset, using `skip_file_discovery="vae aspect metadata"` will prevent the trainer from scanning the dataset on startup, saving a lot of time.

### `conditioning`

- **Values:** Array of conditioning configuration objects
- **Description:** Automatically generates conditioning datasets from your source images. Each conditioning type creates a separate dataset that can be used for ControlNet training or other conditioning tasks.
- **Note:** When specified, SimpleTuner will automatically create conditioning datasets with IDs like `{source_id}_conditioning_{type}`

Each conditioning object can contain:
- `type`: The type of conditioning to generate (required)
- `params`: Type-specific parameters (optional)
- `captions`: Caption strategy for the generated dataset (optional)
  - Can be `false` (no captions)
  - A single string (used as instance prompt for all images)
  - An array of strings (randomly selected for each image)
  - If omitted, captions from the source dataset are used

#### Available Conditioning Types

##### `superresolution`
Generates low-quality versions of images for super-resolution training:
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
Creates JPEG compression artifacts for artifact removal training:
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
Generates depth maps using DPT models:
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**Note:** Depth generation requires GPU and runs in the main process, which may be slower than CPU-based generators.

##### `random_masks` / `inpainting`
Creates random masks for inpainting training:
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
Generates Canny edge detection maps:
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

See [the ControlNet guide](/documentation/CONTROLNET.md) for more details on how to use these conditioning datasets.

#### Examples

##### Video dataset

A video dataset should be a folder of (eg. mp4) video files and the usual methods of storing captions.

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

- In the `video` subsection, we have the following keys we can set:
  - `num_frames` (optional, int) is how many seconds of data we'll train on.
    - At 25 fps, 125 frames is 5 seconds of video, standard output. This should be your target.
  - `min_frames` (optional, int) determines the minimum length of a video that will be considered for training.
    - This should be at least equal to `num_frames`. Not setting it ensures it'll be equal.
  - `max_frames` (optional, int) determines the maximum length of a video that will be considered for training.
  - `is_i2v` (optional, bool) determines whether i2v training will be done on a dataset.
    - This is set to True by default for LTX. You can disable it, however.


##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Outcome
- Any images with a shorter edge less than **1024px** will be completely excluded from training.
- Images like `768x1024` or `1280x768` would be excluded, but `1760x1024` and `1024x1024` would not.
- No image will be upsampled, because `minimum_image_size` is equal to `resolution`

##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # different from the above configuration, which is 'pixel'
```
##### Outcome
- The image's total area (width * height) being less than the minimum area (1024 * 1024) will result in it being excluded from training.
- Images like `1280x960` would **not** be excluded because `(1280 * 960)` is greater than `(1024 * 1024)`
- No image will be upsampled, because `minimum_image_size` is equal to `resolution`

##### Configuration
```json
    "minimum_image_size": 0, # or completely unset, not present in the config
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Outcome
- Images will be resized so their shorter edge is 1024px while maintaining their aspect ratio
- No images will be excluded based on size
- Small images will be upscaled using naive `PIL.resize` methods that do not look good
  - Upscaling is recommended to avoid unless done by hand using an upscaler of your choice before beginning training

### `maximum_image_size` and `target_downsample_size`

Images are not resized before cropping **unless** `maximum_image_size` and `target_downsample_size` are both set. In other words, a `4096x4096` image will be directly cropped to a `1024x1024` target, which may be undesirable.

- `maximum_image_size` specifies the threshold at which the resizing will begin. It will downsample images before cropping if they are larger than this.
- `target_downsample_size` specifies how large the image will be after resample and before it is cropped.

#### Examples

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
- Any images with a pixel area greater than `(1536 * 1536)` will be resized so that its pixel area is roughly `(1280 * 1280)` while maintaining its original aspect ratio
- Final image size will be random-cropped to a pixel area of `(1024 * 1024)`
- Useful for training on eg. 20 megapixel datasets that need to be resized substantially before cropping to avoid massive loss of scene context in the image (like cropping a picture of a person to just a tile wall or a blurry section of the background)


---

### `prepend_instance_prompt`

- When enabled, all captions will include the `instance_prompt` value at the beginning.

### `only_instance_prompt`

- In addition to `prepend_instance_prompt`, replaces all captions in the dataset with a single phrase or trigger word.

### `repeats`

- Specifies the number of times all samples in the dataset are seen during an epoch. Useful for giving more impact to smaller datasets or maximizing the usage of VAE cache objects.
- If you have a dataset of 1000 images vs one with 100 images, you would likely want to give the lesser dataset a repeats of `9` **or greater** to bring it to 1000 total images sampled.

> ℹ️ This value behaves differently to the same option in Kohya's scripts, where a value of 1 means no repeats. **For SimpleTuner, a value of 0 means no repeats**. Subtract one from your Kohya config value to obtain the equivalent for SimpleTuner, hence a value of **9** resulting from the calculation `(dataset_length + repeats * dataset_length)` .

### `is_regularisation_data`

- Also may be spelt `is_regularization_data`
- Enables parent-teacher training for LyCORIS adapters so that the prediction target prefers the base model's result for a given dataset.
  - Standard LoRA are not currently supported.

### `vae_cache_clear_each_epoch`

- When enabled, all VAE cache objects are deleted from the filesystem at the end of each dataset repeat cycle. This can be resource-intensive for large datasets, but combined with `crop_style=random` and/or `crop_aspect=random` you'll want this enabled to ensure you sample a full range of crops from each image.
- In fact, this option is **enabled by default** when using random bucketing or crops.

### `skip_file_discovery`

- You probably don't want to ever set this - it is useful only for very large datasets.
- This parameter accepts a comma or space separated list of values, eg. `vae metadata aspect text` to skip file discovery for one or more stages of the loader configuration.
- This is equivalent to the commandline option `--skip_file_discovery`
- This is helpful if you have datasets you don't need the trainer to scan on every startup, eg. their latents/embeds are already cached fully. This allows quicker startup and resumption of training.

### `preserve_data_backend_cache`

- You probably don't want to ever set this - it is useful only for very large AWS datasets.
- Like `skip_file_discovery`, this option can be set to prevent unnecessary, lengthy and costly filesystem scans at startup.
- It takes a boolean value, and if set to be `true`, the generated filesystem list cache file will not be removed at launch.
- This is helpful for very large and slow storage systems such as S3 or local SMR spinning hard drives that have extremely slow response times.
- Additionally, on S3, backend listing can add up in cost and should be avoided.

> ⚠️ **Unfortunately, this cannot be set if the data is actively being changed.** The trainer will not see any new data that is added to the pool, it will have to do another full scan.

### `hash_filenames`

- When set, the VAE cache entries' filenames will be hashed. This is not set by default for backwards compatibility, but it allows for datasets with very long filenames to be easily used.

## Filtering captions

### `caption_filter_list`

- **For text embed datasets only.** This may be a JSON list, a path to a txt file, or a path to a JSON document. Filter strings can be simple terms to remove from all captions, or they can be regular expressions. Additionally, sed-style `s/search/replace/` entries may be used to _replace_ strings in the caption rather than simply remove it.

#### Example filter list

A complete example list can be found [here](/config/caption_filter_list.txt.example). It contains common repetitive and negative strings that would be returned by BLIP (all common variety), LLaVA, and CogVLM.

This is a shortened example, which will be explained below:

```
arafed 
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

In order, the lines behave as follows:

- `arafed ` (with a space at the end) will be removed from any caption it is found in. Including a space at the end means the caption will look nicer, as double-spaces won't remain. This is unnecessary, but it looks nice.
- `this .* has a` is a regular expression that will remove anything that contains "this ... has a", including any random text in between those two strings; `.*` is a regular expression that means "everything we find" until it finds the "has a" string, when it stops matching.
- `^this is the beginning of the string` will remove the phrase "this is the beginning of the string" from any caption, but only when it appears at the start of the caption.
- `s/this/will be found and replaced/` will result in the first instance of the term "this" in any caption being replaced with "will be found and replaced".

> ❗Use [regex 101](https://regex101.com) for help debugging and testing regular expressions.

# Advanced techniques

## Advanced Example Configuration

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
    "hash_filenames": true,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
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

## Train directly from CSV URL list

**Note: Your CSV must contain the captions for your images.**

> ⚠️ This is an advanced **and** experimental feature, and you may run into problems. If you do, please open an [issue](https://github.com/bghira/simpletuner/issues)!

Instead of manually downloading your data from a URL list, you might wish to plug them in directly to the trainer.

**Note:** It's always better to manually download the image data. Another strategy to save local disk space might be to try [using cloud storage with local encoder caches](#local-cache-with-cloud-dataset) instead.

### Advantages

- No need to directly download the data
- Can make use of SimpleTuner's caption toolkit to directly caption the URL list
- Saves on disk space, since only the image embeds (if applicable) and text embeds are stored

### Disadvantages

- Requires a costly and potentially slow aspect bucket scan where each image is downloaded and its metadata collected
- The downloaded images are cached on-disk, which can grow to be very large. This is an area of improvement to work on, as the cache management in this version is very basic, write-only/delete-never
- If your dataset has a large number of invalid URLs, these might waste time on resumption as, currently, bad samples are **never** removed from the URL list
  - **Suggestion:** Run a URL validation task beforehand and remove any bad samples.

### Configuration

Required keys:

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
        "preserve_data_backend_cache": false,
        "hash_filenames": true
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

## Parquet caption strategy / JSON Lines datasets

> ⚠️ This is an advanced feature, and will not be necessary for most users.

When training a model with a very-large dataset numbering in the hundreds of thousands or millions of images, it's fastest to store your metadata inside a parquet database instead of txt files - especially when your training data is stored on an S3 bucket.

Using the parquet caption strategy allows you to name all of your files by their `id` value, and change their caption column via a config value rather than updating many text files, or having to rename the files to update their captions.

Here is an example dataloader configuration that makes use of the captions and data in the [photo-concept-bucket](https://huggingface.co/datasets/ptx0/photo-concept-bucket) dataset:

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

In this configuration:

- `caption_strategy` is set to `parquet`.
- `metadata_backend` is set to `parquet`.
- A new section, `parquet` must be defined:
  - `path` is the path to the parquet or JSONL file.
  - `filename_column` is the name of the column in the table that contains the filenames. For this case, we are using the numeric `id` column (recommended).
  - `caption_column` is the name of the column in the table that contains the captions. For this case, we are using the `cogvlm_caption` column. For LAION datasets, this would be the TEXT field.
  - `width_column` and `height_column` can be a column containing strings, int, or even a single-entry Series data type, measuring the actual image's dimensions. This notably improves the dataset preparation time, as we don't need to access the real images to discover this information.
  - `fallback_caption_column` is an optional name of a column in the table that contains fallback captions. These are used if the primary caption field is empty. For this case, we are using the `tags` column.
  - `identifier_includes_extension` should be set to `true` when your filename column contains the image extension. Otherwise, the extension will be assumed as `.png`. It is recommended to include filename extensions in your table filename column.

> ⚠️ Parquet support capability is limited to reading captions. You must separately populate a data source with your image samples using "{id}.png" as their filename. See scripts in the [toolkit/datasets](toolkit/datasets) directory for ideas.

As with other dataloader configurations:

- `prepend_instance_prompt` and `instance_prompt` behave as normal.
- Updating a sample's caption in between training runs will cache the new embed, but not remove the old (orphaned) unit.
- When an image doesn't exist in a dataset, its filename will be used as its caption and an error will be emitted.

## Local cache with cloud dataset

In order to maximise the use of costly local NVMe storage, you may wish to store just the image files (png, jpg) on an S3 bucket, and use the local storage to cache your extracted feature maps from the text encoder(s) and VAE (if applicable).

In this example configuration:

- Image data is stored on an S3-compatible bucket
- VAE data is stored in /local/path/to/cache/vae
- Text embeds are stored in /local/path/to/cache/textencoder

> ⚠️ Remember to configure the other dataset options, such as `resolution` and `crop`

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

**Note:** The `image_embeds` dataset does not have any options to set for data paths. Those are configured via `cache_dir_vae` on the image backend.

### Hugging Face Datasets Support

SimpleTuner now supports loading datasets directly from Hugging Face Hub without downloading the entire dataset locally. This experimental feature is ideal for:

- Large-scale datasets hosted on Hugging Face
- Datasets with built-in metadata and quality assessments
- Quick experimentation without local storage requirements

For thorough documentation on this feature, refer to [this document](/documentation/HUGGINGFACE_DATASETS.md).

For a basic example of how to use a Hugging Face dataset, set `"type": "huggingface"` in your dataloader configuration:

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

## Custom aspect ratio-to-resolution mapping

When SimpleTuner first launches, it generates resolution-specific aspect mapping lists that link a decimal aspect-ratio value to its target pixel size.

It's possible to create a custom mapping that forces the trainer to adjust to your chosen target resolution instead of its own calculations. This functionality is provided at your own risk, as it can obviously cause great harm if configured incorrectly.

To create the custom mapping:

- Create a file that follows the example (below)
- Name the file using the format `aspect_ratio_map-{resolution}.json`
  - For a configuration value of `resolution=1.0` / `resolution_type=area`, the mapping filename will be `aspect_resolution_map-1.0.json`
- Place this file in the location specified as `--output_dir`
  - This is the same location where your checkpoints and validation images will be found.
- No additional configuration flags or options are required. It will be automatically discovered and used, as long as the name and location are correct.

### Example mapping configuration

This is an example aspect ratio mapping generated by SimpleTuner. You don't need to manually configure this, as the trainer will automatically create one. However, for full control over the resulting resolutions, these mappings are supplied as a starting point for modification.

- The dataset had more than 1 million images
- The dataloader `resolution` was set to `1.0`
- The dataloader `resolution_type` was set to `area`

This is the most common configuration, and list of aspect buckets trainable for a 1 megapixel model.

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

For Stable Diffusion 1.5 / 2.0-base (512px) models, the following mapping will work:

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
