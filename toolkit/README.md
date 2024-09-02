SimpleTuner contains some ad-hoc tools for generating and managing the training data and checkpoints.

#### Captioning

When captioning a dataset, relying on a single caption model is a bad practice as it pins the model to whatever the chosen caption model knows.

A variety of caption options are provided:

* `caption_with_blip.py` - This is the original BLIP / BLIP2 captioning script which leverages the `interrogate` python library to run the (by default Flan-T5) BLIP model.
* `caption_with_blip3.py` - Built on top of the Phi LLM, BLIP3 aka XGEN-MM is an excellent option for captioning, relatively lightweight and yet very powerful.
* `caption_with_cogvlm_remote.py` - A script used by the volunteer cluster run via the Terminus Research Group
* `caption_with_cogvlm.py` - If you want CogVLM captioning, use this - though there's some potentially erratic results from Cog where it might repeat words.
* `caption_with_gemini.py` - Set `GEMINI_API_KEY` in your environment from one obtained via [Google AI](https://ai.google.dev) and you can caption images for free using Gemini Pro Vision.
* `caption_with_llava.py` - Use Llava 1.5 or 1.6 and run pretty much the same way the CogVLM script does, albeit in a different style.
* `caption_with_internvl.py` - Uses InternVL2 by default to caption images direclty into parquet tables for use by SimpleTuner.


#### Datasets

* `csv_to_s3.py` - given a folder of CSV webdataset as inputs, download/caption/transform images before stuffing them into an S3 bucket.
* `clear_s3_bucket.py` - Just a convenient way to clear an S3 bucket that's been used with this tool.
* `dataset_from_kellyc.py` - If you use the KellyC browser extension for image scraping, this will build a dataset from the URL list it saves.
* `dataset_from_csv.py` - Download a chunk of data to local storage from a single csv dataset document.
* `dataset_from_laion.py` - A variant of the above script.
* `analyze_laion_data.py` - After downloading a lot of LAION's data, you can use this to throw a lot of it away.
* `analyze_aspect_ratios_json.py` - Use the output from `analyze_laion_data.py` to nuke images that do not fit our aspect goals.
* `check_latent_corruption.py` - Scan and remove any images that will not load properly.
* `update_parquet.py` - A scaffold for updating the contents of a parquet file.
* `folder_to_parquet.py` - Import a folder of images into a parquet file.
* `discord_scrape.py` - Scrape the Midjourney server into a local folder and/or parquet files.
* `enhance_with_controlnet.py` - An incomplete script which aims to demonstrate improving a dataset using ControlNet Tile before training.

#### Inference

* `inference.py` - Generate validation results from the prompts catalogue (`prompts.py`) using DDIMScheduler.
* `inference_ddpm.py` - Use DDPMScheduler to assemble a checkpoint from a base model configuration and run through validation prompts.
* `inference_karras.py` - Use the Karras sigmas with DPM 2M Karras. Useful for testing what might happen in Automatic1111.
* `tile_shortnames.py` - Tile the outputs from the above scripts into strips.

* `inference_snr_test.py` - Generate a large number of CFG range images, and catalogue the results for tiling.
* `tile_images.py` - Generate large image tiles to compare CFG results for zero SNR training / inference tuning.