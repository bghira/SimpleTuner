# Distributed Training (Multi-node)

This document contains notes_*_ on configuring a 4-way 8xH100 cluster for use with SimpleTuner.

> _*_This guide does not contain full end-to-end installation instructions. Instead, these serve as considerations to take when following the [INSTALL](/INSTALL.md) document or one of the [quickstart guides](/documentation/QUICKSTART.md).

## Dataloader configuration

Very-large datasets can be a challenge to efficiently manage. SimpleTuner will automatically shard datasets over each node and distribute pre-processing across every available GPU in the cluster, while using asynchronous queues and threads to maintain throughput.

### Slow image scan / discovery

The **discovery** backend currently restricts aspect bucket data collection to a single node. This can take an **extremely** long time with very-large datasets as each image has to be read from storage to retrieve its geometry.

To work-around this problem, the [parquet metadata_backend](/documentation/DATALOADER.md#parquet-caption-strategy--json-lines-datasets) should be used, allowing you to preprocess your data in any manner accessible to you. As outlined in the linked document section, the parquet table contains the `filename`, `width`, `height`, and `caption` columns to help quickly and efficiently sort the data into its respective buckets.


### Storage space

Huge datasets, especially when using the T5-XXL text encoder, will consume enormous quantities of space for the original data, the image embeds, and the text embeds.

#### Cloud storage

Using a provider such as Cloudflare R2, one can generate extremely large datasets with very little storage fees.

See the [dataloader configuration guide](/documentation/DATALOADER.md#local-cache-with-cloud-dataset) for an example of how to configure the `aws` type in `multidatabackend.json`

- Image data can be stored locally or via S3
  - If images are in S3, the preprocessing speed reduces according to network bandwidth
  - If images are stored locally, this does not take advantage of NVMe throughput during **training**
- Image embeds and text embeds can be separately stored on local or cloud storage
  - Placing embeds on cloud storage reduce the training rate very little, as they are fetched in parallel

Ideally, all images and all embeds are simply maintained in a cloud storage bucket. This greatly simplifies the risk of issues during pre-processing and resuming training.

#### Preserving filesystem scan caches

If your datasets so large that scanning for new images becomes a bottleneck, adding `preserve_data_backend_cache=true` to each dataloader config entry will prevent the backend from being scanned for new images.

**Note** that you should then use the `image_embeds` data backend type ([more information here](/documentation/DATALOADER.md#local-cache-with-cloud-dataset)) to allow these cache lists to live separately in case your pre-processing job is interrupted. This will prevent the **image list** from being re-scanned at startup.

#### Data compression

Data compression should be enabled by adding the following to `config.json`:

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

This will use inline gzip to reduce the amount of redundant disk space consumed by the rather-large text and image embeds.

## Configuring via 🤗 Accelerate

When using `accelerate config` (`/home/user/.cache/huggingface/accelerate/default_config.yaml`) to deploy SimpleTuner, these options will take priority over the contents of `config/config.env`

An example default_config for Accelerate that does not include DeepSpeed:

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

This document doesn't go into as much detail as the [dedicated page](/documentation/DEEPSPEED.md).

When optimising training on DeepSpeed for multi-node, using the lowest-possible ZeRO level is **essential**.

For example, an 80G NVIDIA GPU can successfully train Flux with ZeRO level 1 offload, minimising overhead substantially.

Adding the following lines 

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### torch compile optimisation

For extra performance (with a drawback of compatibility issues) you can enable torch compile by adding the following lines into each node's yaml config:

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## Expected performance

- 4x H100 SXM5 nodes connected via local network
- 1TB of memory per node
- Training cache streaming from shared S3-compatible data backend (Cloudflare R2) in same region
- Batch size of **8** per accelerator, and **no** gradient accumulation steps
  - Total effective batch size is **256**
- Resolution is at 1024px with data bucketing enabled
- **Speed**: 15 seconds per step with 1024x1024 data when full-rank training Flux.1-dev (12B)

Lower batch sizes, lower resolution, and enabling torch compile can bring the speed into **iterations per second**:

- Reduce resolution to 512px and disable data bucketing (square crops only)
- Swap DeepSpeed from AdamW to Lion fused optimiser
- Enable torch compile with max-autotune
- **Speed**: 2 iterations per second

## Distributed training caveats

- Every node must have the same number of accelerators available
- Only LoRA/LyCORIS can be quantised, so full distributed model training requires DeepSpeed instead
- This is a very high-cost operation, and high batch sizes might slow you down more than you want, requiring scaling up the count of GPUs in the cluster. A careful balance of budgeting should be considered.
- (DeepSpeed) Validations might need to be disabled when training with DeepSpeed ZeRO 3
- (DeepSpeed) Model saving ends up creating weird sharded copies when saving with ZeRO level 3, but levels 1 and 2 function as expected
- (DeepSpeed) The use of DeepSpeed's CPU-based optimisers becomes required as it handles sharding and offload of the optim states.
