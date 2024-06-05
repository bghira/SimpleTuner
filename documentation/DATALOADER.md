# Dataloader configuration file

Here is an example dataloader configuration file, as `multidatabackend.example.json`:

```json
[
    {
        "id": "something-special-to-remember-by",
        "type": "local",
        "instance_data_dir": "/path/to/data/tree",
        "crop": false,
        "crop_style": "random|center|corner|face",
        "crop_aspect": "square|preserve|random",
        "crop_aspect_buckets": [
            0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ],
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
        "repeats": 5,
        "text_embeds": "alt-embed-cache"
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
        "repeats": 2
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

## Configuration Options

### `id`
- **Description:** Unique identifier for the dataset. It should remain constant once set, as it links the dataset to its state tracking entries.

### `dataset_type`
- **Values:** `image` | `text_embeds`
- **Description:** Text embed datasets are defined differently than image datasets are. A text embed dataset stores ONLY the text embed objects. An image dataset stores the training data.

### `default`
- **Only applies to `dataset_type=text_embeds`**
- If set `true`, this text embed dataset will be where SimpleTuner stores the text embed cache for eg. validation prompt embeds. As they do not pair to image data, there needs to be a specific location for them to end up.

### `text_embeds`
- **Only applies to `dataset_type=image`**
- If unset, the `default` text_embeds dataset will be used. If set to an existing `id` of a `text_embeds` dataset, it will use that instead. Allows specific text embed datasets to be associated with a given image dataset.

### `type`
- **Values:** `aws` | `local`
- **Description:** Determines the storage backend (local or cloud) used for this dataset.

### `instance_data_dir` / `aws_data_prefix`
- **Local:** Path to the data on the filesystem.
- **AWS:** S3 prefix for the data in the bucket.

### Cropping Options
- `crop`: Enables or disables image cropping.
- `crop_style`: Selects the cropping style (`random`, `center`, `corner`, `face`).
- `crop_aspect`: Chooses the cropping aspect (`random`, `square` or `preserve`).
- `crop_aspect_buckets`: When `crop_aspect` is set to `random`, a bucket from this list will be selected, so long as the resulting image size would not result more than 20% upscaling.

### `resolution`
- **Area-Based:** Cropping/sizing is done by megapixel count.
- **Pixel-Based:** Resizing or cropping uses the smaller edge as the basis for calculation.

### `minimum_image_size`
- **Area Comparison:** Specified in megapixels. Considers the entire pixel area.
- **Pixel Comparison:** Both image edges must exceed this value, specified in pixels.

### `maximum_image_size` and `target_downsample_size`
- `maximum_image_size` specifies the maximum image size that will be considered croppable. It will downsample images before cropping if they are larger than this.
- `target_downsample_size` specifies how large the image will be after resample and before it is cropped.
- **Example**: A 20 megapixel image is too large to crop to 1 megapixel without losing context. Set `maximum_size_image=5.0` and `target_downsample_size=2.0` to resize any images larger than 5 megapixels down to 2 megapixels before cropping to 1 megapixel.

### `prepend_instance_prompt`
- When enabled, all captions will include the `instance_prompt` value at the beginning.

### `only_instance_prompt`
- In addition to `prepend_instance_prompt`, replaces all captions in the dataset with a single phrase or trigger word.

### `repeats`
- Specifies the number of times all samples in the dataset are seen during an epoch. Useful for giving more impact to smaller datasets or maximizing the usage of VAE cache objects.

### `vae_cache_clear_each_epoch`
- When enabled, all VAE cache objects are deleted from the filesystem at the end of each dataset repeat cycle. This can be resource-intensive for large datasets, but combined with `crop_style=random` and/or `crop_aspect=random` you'll want this enabled to ensure you sample a full range of crops from each image.

### `ignore_epochs`
- When enabled, this dataset will not hold up the rest of the datasets from completing an epoch. This will inherently make the value for the current epoch inaccurate, as it reflects only the number of times any datasets *without* this flag have completed all of their repeats. The state of the ignored dataset isn't reset upon the next epoch, it is simply ignored. It will eventually run out of samples as a dataset typically does. At that time it will be removed from consideration until the next natural epoch completes.

### `skip_file_discovery`
- This allows specifying the commandline option `--skip_file_discovery` just for a particular dataset at a time. This is helpful if you have datasets you don't need the trainer to scan on every startup, eg. their latents/embeds are already cached fully. This allows quicker startup and resumption of training. This parameter accepts a comma or space separated list of values, eg. `vae metadata aspect text` to skip file discovery for one or more stages of the loader configuration.

### `preserve_data_cache_backend`
- Like `skip_file_discovery`, this option can be set to prevent repeated lookups of file lists during startup. It takes a boolean value, and if set to be `true`, the generated cache file will not be removed at launch. This is helpful for very large and slow storage systems such as S3 or local SMR spinning hard drives that have extremely slow response times. Additionally, on S3, backend listing can add up in cost and should be avoided. **Unfortunately, this cannot be set if the data is actively being changed.** The trainer will not see any new data that is added to the pool, it will have to do another full scan.

## Filtering captions

### `caption_filter_list`
- **For text embed datasets only.** This may be a JSON list, a path to a txt file, or a path to a JSON document. Filter strings can be simple terms to remove from all captions, or they can be regular expressions. Additionally, sed-style `s/search/replace/` entries may be used to *replace* strings in the caption rather than simply remove it.

#### Example filter list

A complete example list can be found [here](/caption_filter_list.example.txt). It contains common repetitive and negative strings that would be returned by BLIP (all common variety), LLaVA, and CogVLM.

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
    "crop_aspect": "random",
    "crop_style": "random",
    "crop_aspect_buckets": [
        1.0,
        0.75,
        1.23
    ],
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