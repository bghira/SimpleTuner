# SimpleTuner WebUI Tutorial

## Introduction

This tutorial will help you get started with the SimpleTuner Web interface.

## Installing requirements

For Ubuntu systems, start by installing the required packages:

```bash
apt -y install python3.12-venv python3.12-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # if you're using DeepSpeed
```

## Creating a workspace directory

A workspace contains your configurations, output models, validation images, and potentially your datasets.

On Vast or similar providers, you can use the `/workspace/simpletuner` directory:

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

If you'd like to make it in your home directory instead:
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## Installing SimpleTuner into your workspace

Create a virtual environment to install dependencies to:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
```

### CUDA-specific dependencies

NVIDIA users will have to use the CUDA extras to pull in all the right dependencies:

```bash
pip install -e 'simpletuner[cuda]'
# or, if you've cloned via git:
# pip install -e '.[cuda]'
```

There are other extras for users on apple and rocm hardware, see the [installation instructions](../INSTALL.md).

## Starting the server

To start the server with SSL on port 8080:

```bash
# for DeepSpeed, we'll need CUDA_HOME pointing to the correct location
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

Now, visit https://localhost:8080 in your web browser. You may need to forward the port over SSH, for example:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

## Using the WebUI

### Onboarding steps

Once you have the page loaded, you'll be asked onboarding questions to set up your environment.

#### Configuration directory

The special configuration value `configs_dir` is introduced to point to a folder that contains all of your SimpleTuner configurations, which are recommended to be sorted into subdirectories - **the Web UI will do this for you**:

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="788" height="465" alt="image" src="https://github.com/user-attachments/assets/656aa287-3b59-476d-ac45-6ede325fe858" />

##### Migrating from command-line usage

If you've been using SimpleTuner before without a WebUI, you can point to your existing config/ folder and all of your environments will be auto-discovered.

For new users, the default location of your configs and datasets will `~/.simpletuner/` and it's recommended to move your datasets somewhere with more space:

<img width="775" height="454" alt="image" src="https://github.com/user-attachments/assets/39238810-da26-4bde-8fc9-1002251f778a" />


#### (Multi-)GPU selection and configuration

After configuring the default paths, you'll reach a step where multi-GPU can be configured (pictured on a Macbook)

<img width="755" height="646" alt="image" src="https://github.com/user-attachments/assets/de43a09d-06a7-45c0-8111-7a0b014499c8" />

If you've got multiple GPUs and would like to just use the second one, this is where you can do that.

> **Note for multi-GPU users:** When training with multiple GPUs, your dataset size requirements increase proportionally. The effective batch size is calculated as `train_batch_size × num_gpus × gradient_accumulation_steps`. If your dataset is smaller than this value, you'll need to either increase the `repeats` setting in your dataset configuration or enable the `--allow_dataset_oversubscription` option in the Advanced settings. See the [batch size section](#multi-gpu-batch-size-considerations) below for more details.

#### Creating your first training environment

If you did not have any pre-existing configurations found in your `configs_dir`, you'll be asked to create **your first training environment**:

<img width="750" height="1381" alt="image" src="https://github.com/user-attachments/assets/4a3ee88f-c70f-416c-ae5d-6593deb9ca35" />

Use **Bootstrap From Example** to select an example config to start from, or simply enter a descriptive name and create a random environment if you prefer to use a setup wizard instead.

### Switching between training environments

If you had any pre-existing configuration environments, they will show up in this drop-down menu.

Otherwise, the option we just created while onboarding will be selected and active already.

<img width="965" height="449" alt="image" src="https://github.com/user-attachments/assets/d8c73cef-ecbb-4229-ad54-9ccd55f8175a" />

Use **Manage Configs** to get to the `Environment` tab where a list of your environments, dataloader and other configurations can be found.

### Configuration wizard

I've worked hard to provide a comprehensive setup wizard that will help you configure some of the most important settings in a no-nonsense bootstrap to get started.

<img width="394" height="286" alt="image" src="https://github.com/user-attachments/assets/21e99854-1d75-4ba9-8be6-15e715d77f4e" />

In the upper left navigation menu, the Wizard button will bring you to a selection dialogue:

<img width="1186" height="1756" alt="image" src="https://github.com/user-attachments/assets/f6d4ac57-e3f6-4060-a4d3-b7f0829d7350" />

And then all built-in model variants are offered. Each variant will pre-enable required settings like Attention Masking or extended token limits.

#### LoRA model options

If you wish to train a LoRA, you'll be able to set the model quantisation options here.

In general, unless you're training a Stable Diffusion type model, int8-quanto is recommended as it won't harm quality, and allows higher batch sizes.

Some small models like Cosmos2, Sana, and PixArt, really do not like being quantised.

<img width="1106" height="1464" alt="image" src="https://github.com/user-attachments/assets/0284d987-6060-4692-934a-0905ef2d5ca1" />

#### Full-rank training

Full-rank training is discouraged, as it generally takes a lot longer and costs more in resources than a LoRA/LyCORIS, for the same dataset.

However, if you do wish to train a full checkpoint, you're able to configure DeepSpeed ZeRO stages here which will be required for larger models like Auraflow, Flux, and larger.

FSDP2 is supported, but not configurable in this wizard. Simply leave DeepSpeed disabled and manually configure FSDP2 later if you wish to use it

<img width="1097" height="1278" alt="image" src="https://github.com/user-attachments/assets/60475318-facd-4da1-a2a1-67cecff18e04" />


#### How long do you want to train for?

You'll have to decide whether you wish to measure training time in epochs or steps. It all is pretty much equal in the end, though some people develop a preference one way or the other.

<img width="1136" height="1091" alt="image" src="https://github.com/user-attachments/assets/9146cdcd-f277-45e5-92cb-f74f23039d51" />

#### Sharing your model via Hugging Face Hub

Optionally, you can publish your final *and* intermediate checkpoints to [Hugging Face Hub](https://hf.co), but you'll require an account - you can login to the hub via the wizard, or the Publishing tab. Either way, you can always change your mind and enable or disable it.

If you do select to publish your model, be mindful to select `Private repo` if you don't want your model to be accessible to the broader public.

<img width="1090" height="859" alt="image" src="https://github.com/user-attachments/assets/d1f86b6b-b0d5-4caa-b3ff-6bd106928094" />

#### Model validations

If you want the trainer to generate images periodically, you can configure a single validation prompt at this point of the wizard. Multiple prompt library can be configured inside the `Validations & Output` tab after the wizard is complete.

Want to outsource validation to your own script or service? Switch the **Validation Method** to `external-script` in the validation tab after the wizard and provide `--validation_external_script`. You can pass training context into the script with placeholders like `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, and any `validation_*` config value (e.g., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Enable `--validation_external_background` to fire-and-forget without blocking training.

If you want to keep SimpleTuner's built-in publishing providers (or Hugging Face Hub uploads) but still trigger your own automation with the remote URL, use `--post_upload_script` instead. It runs once per upload with placeholders `{remote_checkpoint_path}`, `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`. SimpleTuner doesn't capture the script's output—emit any tracker updates directly from your script.

Example hook:

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

Where `notify.sh` posts the URL to your tracker web API. Feel free to adapt to Slack, custom dashboards, or any other integration.

Working sample: `simpletuner/examples/external-validation/replicate_post_upload.py` demonstrates using `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}`, and `{huggingface_path}` to trigger a Replicate inference after uploads.

Another sample: `simpletuner/examples/external-validation/wavespeed_post_upload.py` calls the WaveSpeed API and polls for the result, using the same placeholders.

Flux-focused sample: `simpletuner/examples/external-validation/fal_post_upload.py` calls the fal.ai Flux LoRA endpoint; it requires `FAL_KEY` and only runs when `model_family` includes `flux`.

Local GPU sample: `simpletuner/examples/external-validation/use_second_gpu.py` runs Flux LoRA inference on another GPU (defaults to `cuda:1`) and can be used even when no uploads occur.

<img width="1101" height="1357" alt="image" src="https://github.com/user-attachments/assets/97bdd3f1-b54c-4087-b4d5-05da8b271751" />

#### Logging training statistics

SimpleTuner has support for multiple target APIs if you wish to send your training statistics to one.

Note: None of your personal data, training logs, captions, or data are **ever** sent to SimpleTuner project developers. Control of your data is in **your** hands.

<img width="1099" height="1067" alt="image" src="https://github.com/user-attachments/assets/c9be9a20-12ad-402a-9605-66ba5771e630" />

#### Dataset Configuration

At this point, you can decide whether to keep any existing dataset, or create a new configuration (leaving any others untouched) through the Dataset Creation Wizard, which will appear upon clicking.

<img width="1103" height="877" alt="image" src="https://github.com/user-attachments/assets/3d3cc391-52ed-422e-a4a1-676ca342df10" />

##### Dataset Wizard

If you elected to create a new dataset, you'll see the following wizard, which will walk you through the adding of a local or cloud dataset.

<img width="1110" height="857" alt="image" src="https://github.com/user-attachments/assets/3719e0f5-774e-461d-be02-902e08a679f6" />

<img width="1082" height="1255" alt="image" src="https://github.com/user-attachments/assets/ac38a3de-364a-447f-a734-cab2bdd5338d" />

For a local dataset, you'll be able to use the **Browse directories** button to access a dataset browser modal.

<img width="1201" height="1160" alt="image" src="https://github.com/user-attachments/assets/66a333d0-30fa-45d1-a5b2-1e859d789677" />

If you've pointed the datasets directory correctly during onboarding, you'll see your stuff here.

Click the directory you wish to add, and then **Select Directory**.

<img width="907" height="709" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

After this, you'll be guided through configuring resolution values and cropping.

**NOTE**: SimpleTuner doesn't *upscale* images, so ensure they are at least as large as your configured resolution.

When you reach the step to configure your captions, **carefully consider** which option is correct.

If you're just wanting to use a single trigger word, that'd be the **Instance Prompt** option.

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

#### Learning rate, batch size & optimiser

Once you complete the dataset wizard (or if you elected to keep your existing datasets), you'll be offered presets for optimiser/learning rate and batch size.

These are just starting points that help newcomers make somewhat better choices for their first few training runs - for experienced users, use **Manual configuration** for complete control.

**NOTE**: If you plan on using DeepSpeed later, the optimiser choice doesn't matter much here.

##### Multi-GPU Batch Size Considerations

When training with multiple GPUs, be aware that your dataset must accommodate the **effective batch size**:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

If your dataset is smaller than this value, SimpleTuner will raise an error with specific guidance. You can:
- Reduce the batch size
- Increase the `repeats` value in your dataset configuration
- Enable **Allow Dataset Oversubscription** in the Advanced settings to automatically adjust repeats

See [DATALOADER.md](/documentation/DATALOADER.md#multi-gpu-training-and-dataset-sizing) for more details on dataset sizing.

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### Review & save

If you're happy with all of your selected values, go ahead and **Finish** the wizard.

You'll then see your new environment actively selected and ready for training!

In most cases, these settings will be all you'll have needed to configure. You may want to add extra datasets or fiddle with other settings.

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

On the **Environment** page, you'll see the newly-configured training job, and buttons to download or duplicate the configuration, if you wished to use it like a template.

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**NOTE**: The **Default** environment is special, and not recommended for use as a general training environment; its settings can be automatically merged into any environment that enables the option to do so, **Use environment defaults**:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
