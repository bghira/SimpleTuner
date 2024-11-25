# DeepSpeed offload / multi-GPU training

SimpleTuner v0.7 includes preliminary support for training SDXL using DeepSpeed ZeRO stages 1 through 3.

> ⚠️ Stable Diffusion 3 support for DeepSpeed is not tested, and will be unlikely to work without modification.

**Training SDXL 1.0 on 9237MiB of VRAM**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.11     9232MiB |
+-----------------------------------------------------------------------------+
```

These memory savings have been achieved through the use of DeepSpeed ZeRO Stage 2 offload. Without that, the SDXL U-net will consume more than 24G of VRAM, causing the dreaded CUDA Out of Memory exception.

## What is DeepSpeed?

ZeRO stands for **Zero Redundancy Optimizer**. This technique reduces the memory consumption of each GPU by partitioning the various model training states (weights, gradients, and optimizer states) across the available devices (GPUs and CPUs).

ZeRO is being implemented as incremental stages of optimizations, where optimizations in earlier stages are available in the later stages. To deep dive into ZeRO, please see the original [paper](https://arxiv.org/abs/1910.02054v3) (1910.02054v3).

## Known issues

### LoRA support

Due to how DeepSpeed changes the model saving routines, it's not currently supported to train LoRA with DeepSpeed.

This may change in a future release.

### Enabling / disabling DeepSpeed on existing checkpoints

Currently in SimpleTuner, DeepSpeed cannot be **enabled** when resuming from a checkpoint that did **not** previously use DeepSpeed.

Conversely, DeepSpeed cannot be **disabled** when attempting to resume training from a checkpoint that was trained using DeepSpeed.

To workaround this issue, export the training pipeline to a complete set of model weights before attempting to enable/disable DeepSpeed on an in-progress training session.

It's unlikely this support will ever come to fruition, as DeepSpeed's optimiser is very different from any of the usual choices.

## DeepSpeed Stages

DeepSpeed offers three levels of optimisation for training a model, with each increase having more and more overhead.

Especially for multi-GPU training, the CPU transfers are currently not highly optimised within DeepSpeed.

Because of this overhead, it is recommended that the **lowest** level of DeepSpeed that works, be the one you select.

### Stage 1

The optimizer states (e.g., for Adam optimizer, 32-bit weights, and the first, and second moment estimates) are partitioned across the processes, so that each process updates only its partition.

### Stage 2

The reduced 32-bit gradients for updating the model weights are also partitioned such that each process retains only the gradients corresponding to its portion of the optimizer states.

### Stage 3

The 16-bit model parameters are partitioned across the processes. ZeRO-3 will automatically collect and partition them during the forward and backward passes.

## Enabling DeepSpeed

The [official tutorial](https://www.deepspeed.ai/tutorials/zero/) is very well-structured and includes many scenarios not outlined here.

DeepSpeed is supported by 🤗Accelerate, and can be easily enabled through `accelerate config`:

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO  
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4                                                  
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16                                                                                                                        
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml                              
```

This results in the following yaml file:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## Configuring SimpleTuner

SimpleTuner requires no special configuration for its use of DeepSpeed.

If using ZeRO stage 2 or 3 with NVMe offload, `--offload_param_path=/path/to/offload` can be supplied, to store the parameter/optimiser offload files on a dedicated partition. This storage should ideally be an NVMe device, but any storage will do.

### DeepSpeed Optimizer / Learning rate scheduler

DeepSpeed uses its own learning rate scheduler and, by default, a heavily-optimised version of AdamW - though, not 8bit. That seems less important for DeepSpeed, as things will tend to stay closer to the CPU.

If a `scheduler` or `optimizer` are configured in your `default_config.yaml`, those will be used. If no `scheduler` or `optimizer` are defined, the default `AdamW` and `WarmUp` options will be used as optimiser and scheduler, respectively.

## Some quick test results

Using a 4090 24G GPU:

* We can now train the full U-net at 1 megapixel (1024^2 pixel area) using just **13102MiB of VRAM for batch size 8**
  * This operated at 8 seconds per iteration. This means 1000 steps of training can be done in a little under 2 and 1/2 hours.
  * As indicated in the DeepSpeed tutorial, it may be advantageous to attempt to tune the batch size to a lower value, so that the available VRAM is used for parameters and optimiser states.
    * However, SDXL is a relatively small model, and we can potentially avoid some of the recommendations if the performance impact is acceptable.
* At **128x128** image size on a batch size of 8, training consumes as little as **9237MiB of VRAM**. This is a potentially niche use case for pixel art training, which requires a 1:1 mapping with the latent space.

Within these parameters, you will find varying degrees of success and can quite possibly even fit the full u-net training into as little as 8GiB of VRAM at 1024x1024 for a batch size of 1 (untested).

Since SDXL was trained for many steps on a large distribution of image resolutions and aspect ratios, you can even reduce the pixel area down to .75 megapixels, roughly 768x768 and further optimise memory use.

# AMD device support

I do not have any consumer or workstation-grade AMD GPUs, however, there are some reports that the MI50 (now going out of support) and other higher grade Instinct cards **do** work with DeepSpeed. AMD maintains a repository for their implementation.

# EMA training (Exponential moving average)

While EMA is a great way to smooth out gradients and improve generalisation abilities of the resulting weights, it is a very memory heavy affair.

EMA holds a shadow copy of the model parameters in memory, essentially doubling the footprint of the model. For SimpleTuner, EMA is not passed through the Accelerator module, which means it is not impacted by DeepSpeed. This means the memory savings that we saw with the base U-net, are not realised with the EMA model.

However, by default, the EMA model is kept on CPU.