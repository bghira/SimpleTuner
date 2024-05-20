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
        "instance_prompt": "cat girls",
        "only_instance_prompt": false,
        "caption_strategy": "filename",
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

## Parquet caption strategy

> ⚠️ This is an advanced feature, and will not be necessary for most users.

When training a model with a very-large dataset numbering in the hundreds of thousands or millions of images, it is expedient to store your metadata inside a parquet database instead of txt files - especially when your training data is stored on an S3 bucket.

Using the parquet caption strategy allows you to name all of your files by their `id` value, and change their caption column via a config value rather than updating many text files, or having to rename the files to update their captions.

Here is an example dataloader configuration that makes use of the captions and data in the [photo-concept-bucket](https://huggingface.co/datasets/ptx0/photo-concept-bucket) dataset:

```json
{
    "id": "photo-concept-bucket",
    "type": "local",
    "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
    "caption_strategy": "parquet",
    "parquet_path": "/models/training/datasets/photo-concept-bucket/photo-concept-bucket.parquet",
    "parquet_filename_column": "id",
    "parquet_caption_column": "cogvlm_caption",
    "parquet_fallback_caption_column": "tags",
    "minimum_image_size": 0.25,
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
    "crop_aspect": "square",
    "crop_style": "random",
    "resolution": 0.5,
    "resolution_type": "area"
}
```

In this configuration:

- `caption_strategy` is set to `parquet`.
- `parquet_path` is the path to the parquet file.
- `parquet_filename_column` is the name of the column in the parquet file that contains the filenames. For this case, we are using the numeric `id` column (recommended).
- `parquet_caption_column` is the name of the column in the parquet file that contains the captions. For this case, we are using the `cogvlm_caption` column. For LAION datasets, this would be the TEXT field.
- `parquet_fallback_caption_column` is an optional name of a column in the parquet file that contains fallback captions. These are used if the primary caption field is empty. For this case, we are using the `tags` column.

> ⚠️ Parquet support capability is limited to reading captions. You must separately populate a data source with your image samples using "{id}.png" as their filename. See scripts in the [toolkit/datasets](toolkit/datasets) directory for ideas.

As with other dataloader configurations:

- `prepend_instance_prompt` and `instance_prompt` behave as normal.
- Updating a sample's caption in between training runs will cache the new embed, but not remove the old (orphaned) unit.