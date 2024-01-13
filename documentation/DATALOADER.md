# Dataloader configuration file

Here is an example dataloader configuration file, as `multidatabackend.example.json`:

```json
[
    {
        "id": "something-special-to-remember-by",
        "type": "local",
        "instance_data_dir": "/path/to/data/tree",
        "crop": false,
        "crop_style": "random|center|corner",
        "crop_aspect": "square|preserve",
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
        "repeats": 5
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
        "cache_dir_vae": "/path/to/cache/dir",
        "vae_cache_clear_each_epoch": true,
        "repeats": 2
    }
]
```

## Configuration Options

### `id`
- **Description:** Unique identifier for the dataset. It should remain constant once set, as it links the dataset to its state tracking entries.

### `type`
- **Values:** `aws` | `local`
- **Description:** Determines the storage backend (local or cloud) used for this dataset.

### `instance_data_dir` / `aws_data_prefix`
- **Local:** Path to the data on the filesystem.
- **AWS:** S3 prefix for the data in the bucket.

### Cropping Options
- `crop`: Enables or disables image cropping.
- `crop_style`: Selects the cropping style (`random`, `center`, `corner`).
- `crop_aspect`: Chooses the cropping aspect (`square` or `preserve`).

### `resolution`
- **Area-Based:** Cropping/sizing is done by megapixel count.
- **Pixel-Based:** Resizing or cropping uses the smaller edge as the basis for calculation.

### `minimum_image_size`
- **Area Comparison:** Specified in megapixels. Considers the entire pixel area.
- **Pixel Comparison:** Both image edges must exceed this value, specified in pixels.

### `prepend_instance_prompt`
- When enabled, all captions will include the `instance_prompt` value at the beginning.

### `only_instance_prompt`
- In addition to `prepend_instance_prompt`, replaces all captions in the dataset with a single phrase or trigger word.

### `repeats`
- Specifies the number of times all samples in the dataset are seen during an epoch. Useful for giving more impact to smaller datasets or maximizing the usage of VAE cache objects.

### `vae_cache_clear_each_epoch`
- When enabled, all VAE cache objects are deleted from the filesystem at the end of each dataset repeat cycle. This can be resource-intensive for large datasets.