SimpleTuner contains some ad-hoc tools for generating and managing the training data and checkpoints.

#### Captioning

* `interrogate.py` - This is useful for labelling datasets using BLIP. Not very accurate, but good enough for a LARGE dataset that's being used for fine-tuning.

#### Datasets

* `csv_to_s3.py` - given a folder of CSV webdataset as inputs, download/caption/transform images before stuffing them into an S3 bucket.
* `dataset_from_csv.py` - Download a chunk of data to local storage from a single csv dataset document.
* `dataset_from_laion.py` - A variant of the above script.
* `analyze_laion_data.py` - After downloading a lot of LAION's data, you can use this to throw a lot of it away.
* `analyze_aspect_ratios_json.py` - Use the output from `analyze_laion_data.py` to nuke images that do not fit our aspect goals.
* `helpers/broken_images.py` - Scan and remove any images that will not load properly.

#### Inference

* `inference.py` - Generate validation results from the prompts catalogue (`prompts.py`) using DDIMScheduler.
* `inference_ddpm.py` - Use DDPMScheduler to assemble a checkpoint from a base model configuration and run through validation prompts.
* `inference_karras.py` - Use the Karras sigmas with DPM 2M Karras. Useful for testing what might happen in Automatic1111.
* `tile_shortnames.py` - Tile the outputs from the above scripts into strips.

* `inference_snr_test.py` - Generate a large number of CFG range images, and catalogue the results for tiling.
* `tile_images.py` - Generate large image tiles to compare CFG results for zero SNR training / inference tuning.