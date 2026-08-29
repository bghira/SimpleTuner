# Voice Cloning Data Transforms

Voice cloning data transforms are a planned experimental audio-dataset feature for expanding a training set by transferring one vocal identity onto additional songs, stems, or performances before the main model training begins.

The goal is not to make SimpleTuner a separate voice-conversion workstation. The goal is to make audio fine-tuning datasets less entangled. If a singer only appears in one narrow style, a LoRA can learn "this singer inside this arrangement" instead of the singer identity itself. A voice-cloned expansion split can place the same vocal identity across more varied arrangements, captions, lyrics, and song structures.

This feature is intended for audio datasets only.

!!! warning "Consent and rights"
    Use this workflow only with voices and recordings you are allowed to use. A voice identity is sensitive biometric and creative data. The transform can make derivative audio that sounds like a real person, so permission, licensing, and disclosure matter.

## ELI5

Imagine you have six recordings of one singer, but every recording is in the same band and genre. If you train on only those songs, the model may learn that the singer, guitar tone, drum feel, tempo range, and song structure are one inseparable thing.

Voice cloning data transforms try to separate those ideas:

1. Learn a small voice-conversion model from the singer examples.
2. Take a broader set of songs or vocal stems.
3. Replace the source vocal timbre with the target singer timbre.
4. Keep the new captions and lyrics aligned to the new generated audio.
5. Add the generated audio as another normal training split.

The main model then sees the target voice in more contexts instead of only memorizing the original narrow dataset.

## What This Is For

Use voice cloning transforms when:

- you have permissioned recordings for a target vocalist
- the target identity is too entangled with one genre, band, production style, or song structure
- prompts work only in-domain and fail when the genre changes
- two or more singers in one dataset blend into an averaged voice
- you want separate LoRAs for separate vocal identities
- you want SimpleTuner to prepare the expanded split as part of the same training setup

Do not use it when:

- you already have a large, varied, clean dataset for the same voice
- the source expansion audio is low quality or badly aligned with captions
- you need legally clean public release material and do not have explicit rights
- the base generative model cannot learn the target identity even from clean direct examples

## How It Fits Into Training

Voice cloning is a data preparation transform, not a conditioning dataset.

`conditioning_data` is for paired auxiliary inputs that stay attached to a primary sample during training, such as reference images or generated conditioning maps.

Voice cloning should instead live under a dataset-level `data_transforms` list. The transform materializes new audio files, captions, and optional lyrics, then registers the result as another primary `audio` dataset. After that point, the normal dataloader sees it like any other training split.

Pseudo config shape:

```text
audio dataset:
    id: target-voice
    dataset_type: audio
    data_transforms:
        - task: identity_transfer
          source: expansion-audio-backend
          target: generated-audio-backend
          method: rvc
          audio_mode: separate_convert_remix
```

Pseudo startup behavior:

```text
for each audio dataset:
    for each data transform:
        if task is identity_transfer:
            prepare or reuse the target voice-conversion model
            prepare or reuse generated audio
            append generated audio as a normal train split

continue with normal metadata discovery, bucketing, caching, and training
```

## RVC-Style Identity Transfer

The first implementation is RVC-style voice conversion using the same core ingredients that make community RVC trainers work: HuBERT/ContentVec-style content features, RMVPE pitch extraction, an NSF/VITS generator, a multi-period discriminator, adversarial and mel reconstruction losses, and an optional nearest-neighbor retrieval index.

In this context, the "RVC model" is voice-specific. It is trained from the target identity dataset. The retrieval index is also voice-specific and is built from features from the same target voice. Broad pretrained components, such as content features, pitch extraction, or separation models, are reusable infrastructure; the conversion model and index are the artist- or speaker-specific artifacts.

SimpleTuner should be able to:

1. Reuse a provided voice-conversion model and index.
2. Train the voice-conversion model if no model is provided.
3. Build the retrieval index from the target voice data.
4. Cache the model, index, and generated audio under the training output directory.
5. Reuse cached artifacts at startup when the source data and transform settings have not changed.
6. Optionally reuse or publish the voice-conversion model through a Hub model repository.

## Default Behavior

The defaults are conservative. In this workflow, the audio backend is the expansion music that should be converted, `model.identity_data_dir` is the target voice dataset, and `target.instance_data_dir` is the generated output split.

| Setting | Default | Why |
| --- | --- | --- |
| `task` | `identity_transfer` | Explicitly identifies the transform. |
| `method` | `rvc` | The first supported voice-transfer backend. |
| `train_if_missing` | `true` | SimpleTuner should be able to bootstrap the voice model from the target dataset. |
| `force_retrain` | `false` | Reuse a valid cached voice model when possible. |
| `build_index` | `true` | Retrieval usually improves identity stability and reduces leakage. |
| `identity_data_dir` | required when training on demand | Points to clean vocal examples of the voice to transfer into the expansion songs. |
| `identity_audio_mode` | `separate` | Runs Demucs on identity clips before training. Use `vocal_only` when the identity dataset already contains vocal stems. |
| `identity_stem_debug_dir` | unset | Optional directory for saved identity `vocals.wav` and `no_vocals.wav` previews. Use it to verify that RVC is training from isolated vocals rather than instrument bleed. |
| `asset_hub_model_id` | `lj1995/VoiceConversionWebUI` | Provides the default MIT-licensed RVC assets: HuBERT, RMVPE, and v2 48k pretrained generator/discriminator checkpoints. |
| `model_name` | transform or Hub repo name | Human-readable name saved into the RVC artifact so downloaded caches are identifiable outside their folder name. |
| `sample_rate` | `48000` | The current implementation targets RVC v2 48k assets. Other rates need matching pretrained assets and configs. |
| `training_steps` | `1000` | Runs the RVC generator/discriminator fine-tuning stage during startup. Increase for larger or more varied identity datasets. |
| `batch_size` | `4` | RVC training batch size before distributed sharding. Lower it for memory pressure. |
| `learning_rate` | `1e-4` | Matches the standard RVC AdamW default. |
| `hub_model_id` | unset | No remote voice-model cache is used unless the user opts in. |
| `reuse_from_hub` | `true` when `hub_model_id` is set | Check the Hub before spending time training an on-demand model. |
| `push_to_hub` | `false` | Uploading a voice model should be explicit because the artifact represents a voice identity. |
| `public` | `false` | Hub uploads are private by default. Set this to `true` only when the voice artifact can be published publicly. |
| `audio_mode` | `separate_convert_remix` for full songs, `vocal_only` for vocal stems | Full mixes need separation; stems do not. |
| `separation_method` | `demucs` when separation is needed | Demucs is the expected default stem separator. |
| `timbre_strength` | `1.0` | Controls how strongly the synthesized target-voice vocal replaces the source vocal. Lower values blend source and converted vocals. |
| `retrieval_strength` | `0.75` | Blends nearest target-voice content frames from the saved retrieval index into the generator input. |
| generated split type | primary `audio` dataset | The generated data is trained like normal audio, not used as conditioning. |
| cache location | inside `output_dir` | Keeps generated artifacts tied to the training run and reusable on restart. |
| captions | copy source captions unless configured otherwise | The new split should preserve lyrics and arrangement context. |

If an existing voice-conversion model is supplied, SimpleTuner should use it and only train a new model when explicitly requested or when required artifacts are missing.

## Hub Cache

A voice-conversion model can be expensive enough that repeated on-demand training becomes a footgun. The transform should therefore support an optional Hub-backed cache for the voice model and retrieval index.

The safe lookup order is:

```text
if local voice-conversion cache matches:
    reuse local model and index
else if hub_model_id is configured and reuse_from_hub is enabled:
    check the Hub repository
    download only if it has a SimpleTuner voice-transform manifest
    reuse only if the manifest matches this transform
else if train_if_missing is enabled:
    train the voice-conversion model
    build the retrieval index
    cache locally
    push to hub only when push_to_hub is true
else:
    stop and ask for a model path or a reusable cache
```

The Hub repository uses the `huggingface-hub-rvc` layout rather than a loose collection of files:

```text
config.json
voice_transform/
    manifest.json
    model.safetensors
    features.safetensors
    index.index
```

`config.json` records the package-level RVC metadata, including `model_name`. The voice manifest is the SimpleTuner contract. It should record the target identity dataset fingerprint, RVC training settings, index settings, expected sample rate, tool versions, and the SimpleTuner voice-transform format version. SimpleTuner should not reuse a Hub artifact that lacks this manifest or whose manifest does not match the current transform. That avoids silently applying the wrong voice model to a new dataset.

Publishing should be opt-in. A reasonable pseudo config is:

```text
identity_transfer:
    method: rvc
    model:
        train_if_missing: true
        model_name: Target voice RVC
        hub_model_id: org/target-voice-rvc
        reuse_from_hub: true
        push_to_hub: true
        public: false
```

For private identities, keep the Hub repository private unless you have explicit permission to publish the voice model. Generated audio and model artifacts may have different sharing rights, so treat their upload settings separately.

## WebUI Configuration

RVC model training should be configurable from the WebUI, not only through raw dataloader JSON.

The expected WebUI shape is a dataset transform editor for audio datasets:

```text
Audio dataset
    Data transforms
        Add transform: Identity transfer
            Method: RVC
            Audio mode: vocal_only / separate_convert_remix / full_mix_convert
            Train RVC model if missing: on
            Force retrain: off
            Build retrieval index: on
            Hub model id: optional
            RVC asset repo: lj1995/VoiceConversionWebUI
            Training steps: 1000
            Batch size: 4
            Learning rate: 1e-4
            Reuse from Hub: on when Hub model id is set
            Push RVC model to Hub: off by default
            Hub repo privacy: private by default
            Caption and lyrics sidecars: copied from source audio
```

The WebUI should make the two common setups obvious:

- **Already have vocal stems:** choose `vocal_only`, leave Demucs disabled, and write generated vocal stems.
- **Have full songs:** choose `separate_convert_remix`, use Demucs separation, convert only the vocal stem, and remix with the original instrumental stems.

The interface should show that generated audio becomes another primary audio training split. It should not present identity transfer as `conditioning_data`, because that would imply paired conditioning behavior during training.

## Distributed Startup Behavior

When SimpleTuner starts with multiple data-parallel ranks, voice cloning startup should use the available GPUs instead of making rank 0 do all of the work.

There are two separate distributed phases:

1. **RVC model training:** if `train_if_missing=true`, no matching local cache exists, and no matching Hub artifact is available, the RVC training loop should run under DDP when `world_size > 1`. Each rank should receive different target-voice batches through the normal distributed sampler pattern.
2. **Generated audio preparation:** source expansion inputs should be split by rank, similar to TextEmbedCache and VAECache. Each rank separates, converts, and writes only its assigned shard, then all ranks synchronize before metadata discovery continues.

Pseudo behavior:

```text
if world_size > 1:
    if RVC model must be trained:
        train RVC with DDP across all ranks
        save final model and index once

    split expansion inputs by global rank
    each rank generates its own audio shard
    barrier
    rank 0 writes or verifies the combined manifest
    barrier
else:
    train and generate serially
```

Only one process should publish the final voice model to the Hub. The same applies to final manifest updates. Per-rank generated outputs can be written independently as long as filenames are deterministic and non-overlapping.

This avoids wasting GPU time on multi-GPU systems and keeps startup behavior aligned with SimpleTuner's existing cache preparation model.

## RVC Training Logs

Startup RVC training should not create TensorBoard or WandB runs yet. Those loggers are configured for the main SimpleTuner training job, and reusing them for a nested voice-conversion job would require extra run names, paths, resume rules, and artifact policies.

The RVC stage can still report useful stats through SimpleTuner's native training logger:

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

Useful local stats include generator loss, discriminator loss, mel loss, KL loss, samples processed, elapsed time, DDP world size, cache hit or miss reason, and whether the final model came from local cache, Hub cache, or on-demand training.

These stats are local-only unless a future implementation explicitly adds external logger integration for RVC transforms.

## Choosing `audio_mode`

### `vocal_only`

Use this when your expansion dataset is already preprocessed into clean vocal stems.

This is the simplest and least destructive mode:

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

Gotchas:

- Do not run Demucs again on clean stems unless you have a reason.
- Captions should describe vocals and lyrics, not a full band arrangement, unless you will later remix the stem.
- If the main training model expects full songs, vocal-only generated data may teach a different distribution than your full-song dataset.

### `separate_convert_remix`

Use this when your expansion dataset contains full mixed songs.

The expected flow is:

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

This is the preferred mode for full-song expansion because it avoids converting drums, bass, guitars, room sound, and mastering artifacts as if they were part of the voice.

Gotchas:

- Stem separation can leave bleed, artifacts, or phase issues.
- If the vocal stem is weak, reverberant, or buried, the converted voice can sound unstable.
- Remix loudness matters. A generated split that is consistently louder or quieter than the original data can bias training.
- Captions should describe the final remixed result, not only the source song.

### `full_mix_convert`

Use this only for quick tests.

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

This is fast but usually lower quality. It can drag instruments through the voice converter and teach unwanted artifacts to the final LoRA.

## Caption and Lyrics Policy

The generated split should have captions that match the generated audio.

A good default is:

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

For lyrics, copying is usually correct when the source vocal performance and converted performance use the same words. It is not correct when the transform changes the song, edits sections, drops vocals, or uses a non-lyrical source.

For captions, copying blindly can be wrong. If the source caption says "female pop vocal" and the converted output is a male rock vocal timbre, the caption should be adjusted. The transform should support simple append/remove rules, and more advanced caption rewriting can be layered later.

## Cache and Reuse

The transform should write two kinds of cache:

```text
voice-conversion cache:
    model checkpoint
    retrieval index
    manifest

generated audio cache:
    generated audio files
    captions
    lyrics, when available
    manifest
```

The manifest should record the identity dataset fingerprint, transform settings, source expansion data fingerprint, and tool versions. If those values match, startup can reuse the existing artifacts. If they change, SimpleTuner should regenerate the affected stage.

This is important because voice-conversion training and full-song separation can be expensive. Restarting a main LoRA training run should not redo every generated audio file when nothing relevant changed.

## Practical Dataset Advice

For the target voice in `model.identity_data_dir`, duration matters less than clean voiced coverage.

- **Smoke test:** 30-60 seconds of clean vocal audio can prove the pipeline runs, but the converted voice will usually be rough.
- **Usable starter:** 5-10 minutes of clean isolated voice is a reasonable first target for a personal voice dataset.
- **Singing identity:** 10-30 minutes is better when you need pitch range, vowels, dynamics, articulation, and expressive phrasing.

Use many short clips rather than one long file. Clips around 5-20 seconds are easier to inspect, separate, and reuse. The current RVC trainer resamples identity audio to 48 kHz and truncates each identity file to `max_seconds_per_file`, which defaults to `180`. If a user provides one 30-minute file, only the first three minutes are used by default. Splitting the dataset avoids accidentally throwing away useful vocal coverage.

The standalone [`huggingface-hub-rvc`](https://github.com/SimpleTuner-io/huggingface-hub-rvc) project can train, save, load, and publish the RVC artifact without running a full SimpleTuner training job. Inside SimpleTuner, `scripts/run_rvc_model.py` provides a direct entrypoint for experimenting with the RVC training and conversion portion of the pipeline. Use it when you want to tune the identity dataset, Demucs mode, retrieval strength, transfer strength, or Hub artifact reuse before spending time on the main LoRA training run.

- Keep one target vocalist per generated LoRA when identity control matters.
- Prefer clean, dry vocal examples for training the voice-conversion model.
- Avoid duets unless the goal is specifically to learn the duet blend.
- Use expansion songs with varied tempo, key, genre, dynamics, and lyrical phrasing.
- Keep captions varied enough that identity tokens and style descriptors are not glued to one arrangement.
- Spot-check generated audio before long training runs.
- Compare direct training data, generated data, and mixed training runs separately before combining everything.

## Common Failure Modes

| Symptom | Likely cause |
| --- | --- |
| The LoRA only works in one genre | Voice identity is still entangled with arrangement captions or source data. |
| The generated split sounds phasey or hollow | Separation/remix artifacts from full-song processing. |
| Instruments sound like they were voice-converted | `full_mix_convert` was used where separation was needed. |
| The voice model seems to learn instruments | Identity separation produced vocal stems with too much accompaniment bleed. Set `model.identity_stem_debug_dir`, inspect the saved stems, or preprocess cleaner vocal stems and use `identity_audio_mode=vocal_only`. |
| Vocal identity is weak | Voice-conversion model needs cleaner target data, more target data, or a stronger retrieval index. |
| Captions do not control the voice | Captions still mention source-vocal identity or omit the target identity. |
| The main model learns artifacts | Generated audio quality is too low or too dominant in the train mix. |
| Converted vocals are monotonic or robotic | The RVC path is missing proper F0 extraction, pretrained generator/discriminator initialization, adversarial training, or enough clean target vocal data. |

## Relationship To Regularisation Data

Generated identity-transfer data is not regularisation data by default.

Regularisation data usually teaches the LoRA to preserve the base model's behavior. Identity-transfer data teaches the LoRA a target voice in more contexts. Mixing too much regularisation with too little direct identity data can weaken the identity tokens. Mixing too much generated data can teach conversion artifacts.

Treat these as separate levers:

- direct target dataset: strongest identity signal
- generated identity-transfer dataset: broader context and style coverage
- regularisation dataset: base-model preservation

## Status

This page describes an experimental `data_transforms` workflow. The current implementation trains or reuses a SimpleTuner RVC v2 F0 artifact, extracts HuBERT content features and RMVPE pitch from identity clips, fine-tunes the pretrained RVC generator/discriminator, builds a retrieval index, generates the expanded split, caches the results, and then continues into normal training without requiring a separate manual preprocessing stage.
