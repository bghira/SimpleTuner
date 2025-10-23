# FSDP2 sharded / multi-GPU training

SimpleTuner now ships with first-class support for PyTorch Fully Sharded Data Parallel v2 (DTensor-backed FSDP). The WebUI defaults to the v2 implementation for full-model runs and exposes the most important accelerate flags so you can scale to multi-GPU hardware without writing custom launch scripts.

> ⚠️ FSDP2 targets recent PyTorch 2.x releases with the distributed DTensor stack enabled on CUDA builds. The WebUI only surfaces context-parallel controls on CUDA hosts; other backends are considered experimental.

## What is FSDP2?

FSDP2 is the next iteration of PyTorch’s sharded data-parallel engine. Instead of the legacy flat-parameter logic from FSDP v1, the v2 plugin sits on top of DTensor. It shards model parameters, gradients, and optimizers across ranks while keeping a small per-rank working set. Compared to the classic ZeRO-style approaches it keeps the Hugging Face accelerate launch flow, so checkpoints, optimizers, and inference paths stay compatible with the rest of SimpleTuner.

## Feature overview

- WebUI toggle (Hardware → Accelerate) that generates a FullyShardedDataParallelPlugin with sane defaults
- Automatic CLI normalisation (`--fsdp_version`, `--fsdp_state_dict_type`, `--fsdp_auto_wrap_policy`) so manual flag spelling is forgiving
- Optional context-parallel sharding (`--context_parallel_size`, `--context_parallel_comm_strategy`) layered on top of FSDP2 for long-sequence models
- Built-in transformer block discovery modal that inspects the base model and suggests class names for auto-wrapping
- Cached detection metadata under `~/.simpletuner/fsdp_block_cache.json`, with one-click maintenance actions in the WebUI settings
- Checkpoint format switcher (sharded vs full) plus a CPU-RAM-efficient resume mode for tight host memory ceilings

## Known limitations

- FSDP2 can only be enabled when `model_type` is `full`. PEFT/LoRA style runs continue to use standard single-device paths.
- DeepSpeed and FSDP are mutually exclusive. Supplying both `--fsdp_enable` and a DeepSpeed config raises an explicit error in CLI and WebUI flows.
- Context parallelism is limited to CUDA systems and requires `--context_parallel_size > 1` with `--fsdp_version=2`.
- Validation passes are disabled automatically when `--fsdp_reshard_after_forward` remains true; this mirrors the guard in `trainer.py`.
- Block detection instantiates the base model locally. Expect a short pause and elevated host memory usage when scanning large checkpoints.
- FSDP v1 remains for backwards compatibility but is marked deprecated throughout the UI and CLI logs.

## Enabling FSDP2

### Method 1: WebUI (recommended)

1. Open the SimpleTuner WebUI and load the training configuration you plan to run.
2. Switch to **Hardware → Accelerate**.
3. Toggle **Enable FSDP v2**. The version selector will default to `2`; leave it unless you intentionally need v1.
4. (Optional) Adjust:
   - **Reshard After Forward** to trade VRAM for communication
   - **Checkpoint Format** between `Sharded` and `Full`
   - **CPU RAM Efficient Loading** if resuming with tight host memory limits
   - **Auto Wrap Policy** and **Transformer Classes to Wrap** (see detection workflow below)
   - **Context Parallel Size / Rotation** when you need sequence sharding
5. Save the configuration. The trainer launch surface will now pass the correct accelerate plugin.

### Method 2: CLI

Use `simpletuner-train` with the same flags surfaced in the WebUI. Example for an SDXL full-model run across two GPUs:

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

If you already maintain an accelerate config file you can keep using it; SimpleTuner merges the FSDP plugin into the launch parameters instead of overriding your entire configuration.

## Context parallelism

Context parallelism is available as an optional layer on top of FSDP2 for CUDA hosts. Set `--context_parallel_size` (or the matching WebUI field) to the number of GPUs that should split the sequence dimension. Communication happens via:

- `allgather` (default) – prioritises overlap and is the best starting point
- `alltoall` – niche workloads with very large attention windows may benefit, at the cost of extra orchestration

The trainer enforces `fsdp_enable` and `fsdp_version=2` when context parallelism is requested. Setting the size back to `1` cleanly disables the feature and normalises the rotation string so saved configs stay consistent.

## FSDP block detection workflow

SimpleTuner bundles a detector that inspects the selected base model and surfaces the module classes most suitable for FSDP auto wrapping:

1. Select a **Model Family** (and optionally a **Model Flavour**) in the trainer form.
2. Enter the checkpoint path if you are training from a custom weight directory.
3. Click **Detect Blocks** next to **Transformer Classes to Wrap**. SimpleTuner will instantiate the model, walk its modules, and record parameter totals per class.
4. Review the modal analysis:
   - **Select** the classes that should be wrapped (check boxes in the first column)
   - **Total Params** highlights which modules dominate your parameter budget
   - `_no_split_modules` (if present) are displayed as badges and should be added to your exclusion lists
5. Press **Apply Selection** to populate `--fsdp_transformer_layer_cls_to_wrap`.
6. Subsequent openings reuse the cached result unless you hit **Refresh Detection**.

Detection results live in `~/.simpletuner/fsdp_block_cache.json` keyed by model family, checkpoint path, and flavour. Use **Settings → WebUI Preferences → Cache Maintenance → Clear FSDP Detection Cache** when switching between divergent checkpoints or after updating model weights.

## Checkpoint handling

- **Sharded state dict** (`SHARDED_STATE_DICT`) saves rank-local shards and scales gracefully to large models.
- **Full state dict** (`FULL_STATE_DICT`) gathers parameters to rank 0 for compatibility with external tooling; expect higher memory pressure.
- **CPU RAM Efficient Loading** delays all-rank materialisation during resume to flatten host memory spikes.
- **Reshard After Forward** keeps parameter shards lean between forward passes but disables validation (matching Accelerate’s limitation).

Pick the combination that aligns with your resume cadence and downstream tooling. Sharded checkpoints plus RAM-efficient loading is the safest pairing for very large models.

## Maintenance tooling

The WebUI exposes maintenance helpers under **WebUI Preferences → Cache Maintenance**:

- **Clear FSDP Detection Cache** removes all cached block scans (wrapper over `FSDP_SERVICE.clear_cache()`).
- **Clear DeepSpeed Offload Cache** remains available for ZeRO users; it operates independently of FSDP.

Both actions show toast notifications and update the maintenance status area so you can confirm the result without digging through log files.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | Both plugins specified (e.g., DeepSpeed JSON plus `--fsdp_enable`). | Remove the DeepSpeed config or disable FSDP. |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1` while FSDP is off or still on v1. | Enable FSDP, keep `--fsdp_version=2`, or drop the size back to `1`. |
| Validation silently disabled | `fsdp_enable` true with `fsdp_reshard_after_forward` left on. | Either accept disabled evals or toggle resharding off for runs that include validation passes. |
| Block detection fails with `Unknown model_family` | The form lacks a supported family or flavour. | Pick a model from the dropdown; custom families must register in `model_families`. |
| Detection shows stale classes | Cached result reused. | Click **Refresh Detection** or clear the cache from WebUI Preferences. |
| Resume exhausts host RAM | Full state dict gathering during load. | Switch to `SHARDED_STATE_DICT` and/or enable CPU RAM efficient loading. |

## CLI flag reference

- `--fsdp_enable` – turn on FullyShardedDataParallelPlugin
- `--fsdp_version` – choose between `1` and `2` (default `2`, v1 is deprecated)
- `--fsdp_reshard_after_forward` – release parameter shards post-forward (default `true`)
- `--fsdp_state_dict_type` – `SHARDED_STATE_DICT` (default) or `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` – reduce host memory spikes on resume
- `--fsdp_auto_wrap_policy` – `TRANSFORMER_BASED_WRAP`, `SIZE_BASED_WRAP`, `NO_WRAP`, or a dotted callable path
- `--fsdp_transformer_layer_cls_to_wrap` – comma-separated class list populated by the detector
- `--context_parallel_size` – shard attention across this many ranks (CUDA + FSDP2 only)
- `--context_parallel_comm_strategy` – `allgather` (default) or `alltoall` rotation strategy
- `--num_processes` – total ranks passed to accelerate when no config file is provided

These map 1:1 with the WebUI controls under Hardware → Accelerate, so a configuration exported from the interface can be replayed on the CLI without further tweaks.
