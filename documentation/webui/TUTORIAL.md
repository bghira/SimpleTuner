# SimpleTuner WebUI Tutorial

## Introduction

This tutorial will help you get started with the SimpleTuner Web interface.

## Installing requirements

For Ubuntu systems, start by installing the required packages:

```bash
apt -y install python3.13-venv python3.13-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # if you're using DeepSpeed
apt -y install ffmpeg # if training video models
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
python3.13 -m venv .venv
. .venv/bin/activate
```

### CUDA-specific dependencies

NVIDIA users will have to use the CUDA extras to pull in all the right dependencies:

```bash
pip install 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

Now, visit https://localhost:8080 in your web browser.

You may need to forward the port over SSH, for example:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **Tip:** If you have an existing configuration environment (e.g., from previous CLI usage), you can start the server with `--env` to automatically begin training once the server is ready:
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> This is equivalent to starting the server and then manually clicking "Start Training" in the WebUI, but allows for unattended startup.

## First-time setup: Creating an admin account

On first launch, SimpleTuner requires you to create an administrator account. When you visit the WebUI for the first time, you'll see a setup screen prompting you to create the first admin user.

Enter your email, username, and a secure password. This account will have full administrative privileges.

### Managing users

After setup, you can manage users from the **Manage Users** page (accessible from the sidebar when logged in as an admin):

- **Users tab**: Create, edit, and delete user accounts. Assign permission levels (viewer, researcher, lead, admin).
- **Levels tab**: Define custom permission levels with fine-grained access control.
- **Auth Providers tab**: Configure external authentication (OIDC, LDAP) for single sign-on.
- **Registration tab**: Control whether new users can self-register (disabled by default).

### API keys for automation

Users can generate API keys for scripted access from their profile or the admin panel. API keys use the `st_` prefix and can be used with the `X-API-Key` header:

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **Note:** For private/internal deployments, keep public registration disabled and create user accounts manually through the admin panel.

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

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/4be22081-f13d-4aed-a87c-2313ddefc8a4" />

##### Migrating from command-line usage

If you've been using SimpleTuner before without a WebUI, you can point to your existing config/ folder and all of your environments will be auto-discovered.

For new users, the default location of your configs and datasets will `~/.simpletuner/` and it's recommended to move your datasets somewhere with more space:

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/c5b3ab53-654e-4a9b-8e2d-7951f11619ef" />


#### (Multi-)GPU selection and configuration

After configuring the default paths, you'll reach a step where multi-GPU can be configured (pictured on a NVIDIA system)

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/61d5a7bc-0a02-4a0a-8df0-207cce4b7bc1" />

If you've got multiple GPUs and would like to just use the second one, this is where you can do that.

> **Note for multi-GPU users:** When training with multiple GPUs, your dataset size requirements increase proportionally. The effective batch size is calculated as `train_batch_size × num_gpus × gradient_accumulation_steps`. If your dataset is smaller than this value, you'll need to either increase the `repeats` setting in your dataset configuration or enable the `--allow_dataset_oversubscription` option in the Advanced settings. See the [batch size section](#multi-gpu-batch-size-considerations) below for more details.

#### Creating your first training environment

If you did not have any pre-existing configurations found in your `configs_dir`, you'll be asked to create **your first training environment**:

<img width="500" height="640" alt="image" src="https://github.com/user-attachments/assets/2110287a-16fd-4f87-893b-86d2a555a10f" />

Use **Bootstrap From Example** to select an example config to start from, or simply enter a descriptive name and create a random environment if you prefer to use a setup wizard instead.

### Switching between training environments

If you had any pre-existing configuration environments, they will show up in this drop-down menu.

Otherwise, the option we just created while onboarding will be selected and active already.

<img width="448" height="225" alt="image" src="https://github.com/user-attachments/assets/66fef6a9-2040-47fd-b22d-918470677992" />

Use **Manage Configs** to get to the `Environment` tab where a list of your environments, dataloader and other configurations can be found.

### Configuration wizard

I've worked hard to provide a comprehensive setup wizard that will help you configure some of the most important settings in a no-nonsense bootstrap to get started.

<img width="470" height="358" alt="image" src="https://github.com/user-attachments/assets/e4bf1a4e-716c-4101-b753-e9e24bb42d8a" />

In the upper left navigation menu, the Wizard button will bring you to a selection dialogue:

<img width="448" height="440" alt="image" src="https://github.com/user-attachments/assets/68324fa8-3ca9-45b1-b947-1e7738fd1d8c" />

And then all built-in model variants are offered. Each variant will pre-enable required settings like Attention Masking or extended token limits.

#### LoRA model options

If you wish to train a LoRA, you'll be able to set the model quantisation options here.

In general, unless you're training a Stable Diffusion type model, int8-quanto is recommended as it won't harm quality, and allows higher batch sizes.

Some small models like Cosmos2, Sana, and PixArt, really do not like being quantised.

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/c2e721f2-b4da-4cd0-84fd-7ac81993e87c" />

#### Full-rank training

Full-rank training is discouraged, as it generally takes a lot longer and costs more in resources than a LoRA/LyCORIS, for the same dataset.

However, if you do wish to train a full checkpoint, you're able to configure DeepSpeed ZeRO stages here which will be required for larger models like Auraflow, Flux, and larger.

FSDP2 is supported, but not configurable in this wizard. Simply leave DeepSpeed disabled and manually configure FSDP2 later if you wish to use it

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/88438f1c-b0a2-4249-afd0-7878aa1abada" />

#### How long do you want to train for?

You'll have to decide whether you wish to measure training time in epochs or steps. It all is pretty much equal in the end, though some people develop a preference one way or the other.

<img width="508" height="475" alt="image" src="https://github.com/user-attachments/assets/dcb54279-0ce7-4c66-a9ab-4dd26f87278c" />

#### Sharing your model via Hugging Face Hub

Optionally, you can publish your final *and* intermediate checkpoints to [Hugging Face Hub](https://hf.co), but you'll require an account - you can login to the hub via the wizard, or the Publishing tab. Either way, you can always change your mind and enable or disable it.

If you do select to publish your model, be mindful to select `Private repo` if you don't want your model to be accessible to the broader public.

<img width="508" height="370" alt="image" src="https://github.com/user-attachments/assets/8d2d282b-e66f-48a8-a40e-4e4ecc2d280b" />

#### Checkpoint frequency

When training, your model will be saved periodically to disk. Keeping more checkpoints requires greater disk space.

Checkpoints allow resuming training later without having to repeat all of the steps. Keeping a handful of checkpoints allows testing multiple versions of your model to keep the one that works the best for you.

It's recommended to keep a checkpoint every 10% though it depends on how much data you're training on. With a small dataset, you'll want to checkpoint often to ensure you're not overfitting.

Extremely large datasets will benefit from longer checkpoint intervals to avoid wasting time writing them to disk.

<img width="508" height="485" alt="image" src="https://github.com/user-attachments/assets/c7b1cd0b-a1b9-47ec-87f9-1ecac2e0841a" />


#### Model validations

If you want the trainer to generate images periodically, you can configure a single validation prompt at this point of the wizard. Multiple prompt library can be configured inside the `Validations & Output` tab after the wizard is complete.

Want to outsource validation to your own script or service? Switch the **Validation Method** to `external-script` in the validation tab after the wizard and provide `--validation_external_script`. You can pass training context into the script with placeholders like `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`, and any `validation_*` config value (e.g., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Enable `--validation_external_background` to fire-and-forget without blocking training.

Need a hook the moment a checkpoint hits disk? Use `--post_checkpoint_script` to fire a script right after each save (before uploads begin). It accepts the same placeholders, with `{remote_checkpoint_path}` left empty.

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


<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/e699ba57-526b-4f60-9e8c-0ba410761c9f" />

#### Logging training statistics

SimpleTuner has support for multiple target APIs if you wish to send your training statistics to one.

Note: None of your personal data, training logs, captions, or data are **ever** sent to SimpleTuner project developers. Control of your data is in **your** hands.

<img width="508" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8d15c5-456f-4637-af7e-c2f5f31cb968" />

#### Dataset Configuration

At this point, you can decide whether to keep any existing dataset, or create a new configuration (leaving any others untouched) through the Dataset Creation Wizard, which will appear upon clicking.

<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/b5a7f883-e180-4662-b84c-fff609c6b1df" />

##### Dataset Wizard

If you elected to create a new dataset, you'll see the following wizard, which will walk you through the adding of a local or cloud dataset.

<img width="508" height="332" alt="image" src="https://github.com/user-attachments/assets/c523930b-563e-4b5d-b104-8e7ce4658b2c" />

<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/c263f58e-fd85-437e-811a-967b94e309fd" />

For a local dataset, you'll be able to use the **Browse directories** button to access a dataset browser modal.

<img width="396" height="576" alt="image" src="https://github.com/user-attachments/assets/14c51685-3559-4d16-be59-ed4b0959ca32" />

If you've pointed the datasets directory correctly during onboarding, you'll see your stuff here.

Click the directory you wish to add, and then **Select Directory**.

<img width="454" height="356" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

After this, you'll be guided through configuring resolution values and cropping.

When you reach the step to configure your captions, **carefully consider** which option is correct.

If you're just wanting to use a single trigger word, that'd be the **Instance Prompt** option.

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### Optional: Upload a dataset from your browser

If your images and captions aren't on the box yet, the dataset wizard now includes an **Upload** button next to **Browse directories**. You can:

- Create a new subfolder under your configured datasets directory, then upload individual files or a ZIP (images plus .txt/.jsonl/.csv metadata are accepted).
- Let SimpleTuner extract the ZIP into that folder (sized for local backends; very large archives are rejected).
- Immediately pick the freshly uploaded folder in the browser and continue the wizard without leaving the UI.

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

See [DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing) for more details on dataset sizing.

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### Memory optimisation presets

For easier setup on consumer hardware, each model has custom presets included that allow selecting for light, balanced, or aggressive memory savings.

On the **Training** tab's **Memory Optimisation** section, you'll find the **Load Presets** button:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

Which brings up this interface:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### Review & save

If you're happy with all of your selected values, go ahead and **Finish** the wizard.

You'll then see your new environment actively selected and ready for training!

In most cases, these settings will be all you'll have needed to configure. You may want to add extra datasets or fiddle with other settings.

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

On the **Environment** page, you'll see the newly-configured training job, and buttons to download or duplicate the configuration, if you wished to use it like a template.

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**NOTE**: The **Default** environment is special, and not recommended for use as a general training environment; its settings can be automatically merged into any environment that enables the option to do so, **Use environment defaults**:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
