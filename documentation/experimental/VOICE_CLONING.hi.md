# Voice Cloning Data Transforms

Voice cloning data transforms audio datasets के लिए planned experimental feature हैं। ये main model training शुरू होने से पहले किसी target vocal identity को extra songs, stems, या performances पर transfer करके training set expand करते हैं।

मकसद SimpleTuner को अलग voice-conversion workstation बनाना नहीं है। मकसद audio fine-tuning dataset में identity और arrangement के entanglement को कम करना है। अगर कोई singer सिर्फ एक narrow style में आता है, तो LoRA "इस arrangement में यह singer" सीख सकती है, singer identity खुद नहीं। Voice-cloned expansion split उसी vocal identity को ज्यादा varied arrangements, captions, lyrics, और song structures में दिखा सकता है।

यह feature सिर्फ audio datasets के लिए है।

!!! warning "Consent और rights"
    इस workflow का उपयोग केवल उन voices और recordings पर करें जिन्हें इस्तेमाल करने की अनुमति आपके पास है। Voice identity sensitive biometric और creative data है। Transform ऐसा derivative audio बना सकता है जो किसी real person जैसा सुनाई दे, इसलिए permission, licensing, और disclosure जरूरी हैं।

## ELI5

मान लीजिए आपके पास एक singer की छह recordings हैं, लेकिन सभी एक ही band और genre में हैं। अगर आप सिर्फ उन्हीं songs पर train करते हैं, model singer, guitar tone, drums, tempo range, और song structure को एक ही चीज मानकर सीख सकता है।

Voice cloning data transforms इन ideas को अलग करने की कोशिश करते हैं:

1. Singer examples से छोटा voice-conversion model सीखना।
2. ज्यादा broad songs या vocal stems लेना।
3. Source vocal timbre को target singer timbre से बदलना।
4. नए captions और lyrics को generated audio से aligned रखना।
5. Generated audio को एक और normal training split की तरह add करना।

फिर main model target voice को ज्यादा contexts में देखता है, सिर्फ original narrow dataset याद नहीं करता।

## कब उपयोग करें

Use करें जब:

- आपके पास target vocalist की permissioned recordings हों
- target identity एक genre, band, production style, या song structure से बहुत entangled हो
- trigger words सिर्फ original domain में काम करें
- एक dataset में कई singers हों और model averaged voice बना रहा हो
- आप अलग vocal identities के लिए अलग LoRAs चाहते हों
- आप चाहते हों कि SimpleTuner उसी training setup में expanded split तैयार करे

Avoid करें जब:

- उसी voice का large, varied, clean dataset पहले से हो
- source expansion audio low quality हो या captions से aligned न हो
- public release के लिए clear rights न हों
- base generative model clean direct examples से भी target identity न सीख पाए

## Training में कैसे फिट होता है

Voice cloning data preparation transform है, conditioning dataset नहीं।

`conditioning_data` paired auxiliary inputs के लिए है जो training के दौरान primary sample से जुड़े रहते हैं, जैसे reference images या generated conditioning maps।

Voice cloning dataset-level `data_transforms` list में रहना चाहिए। Transform नए audio files, captions, और optional lyrics materialize करता है, फिर result को दूसरे primary `audio` dataset की तरह register करता है। उसके बाद normal dataloader इसे किसी भी training split जैसा देखता है।

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

पहली implementation RVC-style voice conversion है: HuBERT content features, RMVPE pitch extraction, NSF/VITS generator, multi-period discriminator, mel/adversarial losses, और optional retrieval index।

इस context में "RVC model" voice-specific है। यह target identity dataset से train होता है। Retrieval index भी voice-specific है और उसी target voice की features से बनता है। Content features, pitch extraction, या separation models जैसे broad pretrained components reusable infrastructure हैं; conversion model और index singer या speaker-specific artifacts हैं।

SimpleTuner को ये कर पाना चाहिए:

1. Provided voice-conversion model और index reuse करना।
2. Model न मिलने पर voice-conversion model train करना।
3. Target voice data से retrieval index build करना।
4. Training output directory में model, index, और generated audio cache करना।
5. Source data और transform settings न बदलें तो startup पर cached artifacts reuse करना।
6. Optional रूप से Hub model repository से voice-conversion model reuse या publish करना।

## Default Behavior

Defaults conservative हैं। इस workflow में audio backend वह expansion music है जिसे convert करना है, `model.identity_data_dir` target voice dataset है, और `target.instance_data_dir` सिर्फ generated output split का path है।

| Setting | Default | Why |
| --- | --- | --- |
| `task` | `identity_transfer` | Transform को clearly identify करता है। |
| `method` | `rvc` | पहला supported voice-transfer backend। |
| `train_if_missing` | `true` | SimpleTuner target dataset से voice model bootstrap कर सके। |
| `force_retrain` | `false` | Valid cached voice model reuse हो सके। |
| `build_index` | `true` | Retrieval identity stability improve कर सकता है और leakage घटा सकता है। |
| `identity_data_dir` | on-demand training में required | उस clean target voice के vocal examples की ओर point करता है जिसे expansion songs में transfer करना है। |
| `identity_audio_mode` | `separate` | Training से पहले identity clips पर Demucs चलाता है। अगर identity dataset में पहले से vocal stems हैं तो `vocal_only` use करें। |
| `identity_stem_debug_dir` | unset | Identity `vocals.wav` और `no_vocals.wav` previews save करने की optional directory। इससे verify करें कि RVC isolated vocals से train कर रहा है, instrument bleed से नहीं। |
| `asset_hub_model_id` | `lj1995/VoiceConversionWebUI` | Default RVC asset repository for HuBERT, RMVPE, and v2 48k pretrained generator/discriminator checkpoints. |
| `model_name` | transform or Hub repo name | Human-readable name saved into the RVC artifact so downloaded caches are identifiable outside their folder name. |
| `sample_rate` | `48000` | Current implementation targets RVC v2 48k assets. Other rates need matching pretrained assets and configs. |
| `training_steps` | `1000` | Runs RVC generator/discriminator fine-tuning during startup. Increase for larger or more varied identity datasets. |
| `batch_size` | `4` | RVC training batch size before distributed sharding. Lower it for memory pressure. |
| `learning_rate` | `1e-4` | Standard RVC AdamW default. |
| `hub_model_id` | unset | User opt-in न करे तो remote voice-model cache use नहीं होती। |
| `reuse_from_hub` | `hub_model_id` set हो तो `true` | On-demand model training से पहले Hub check करता है। |
| `push_to_hub` | `false` | Voice model upload explicit होना चाहिए क्योंकि artifact voice identity represent करता है। |
| `public` | `false` | Hub uploads are private by default. Set this to `true` only when the voice artifact can be published publicly. |
| `audio_mode` | full songs के लिए `separate_convert_remix`, vocal stems के लिए `vocal_only` | Full mixes को separation चाहिए; stems को नहीं। |
| `separation_method` | separation चाहिए तो `demucs` | Demucs expected default stem separator है। |
| `timbre_strength` | `1.0` | Controls how strongly the synthesized target voice replaces the source vocal. Lower values blend source and converted vocals. |
| `retrieval_strength` | `0.75` | Blends nearest target-voice content frames from the retrieval index into the generator input. |
| generated split type | primary `audio` dataset | Generated data normal audio की तरह train होता है, conditioning नहीं। |
| cache location | `output_dir` के अंदर | Artifacts training run से जुड़े रहते हैं और restart पर reuse हो सकते हैं। |
| captions | configured न हो तो source captions copy | नया split lyrics और arrangement context preserve करे। |

अगर existing voice-conversion model दिया गया है, SimpleTuner को उसे use करना चाहिए और नया model सिर्फ explicit request या missing required artifacts पर train करना चाहिए।

## Hub Cache

Voice-conversion model इतना expensive हो सकता है कि repeated on-demand training user के लिए footgun बन जाए। इसलिए transform को voice model और retrieval index के लिए optional Hub-backed cache support करना चाहिए।

Safe lookup order:

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

Hub repository को loose files की जगह SimpleTuner-specific layout use करना चाहिए:

```text
config.json
voice_transform/
    manifest.json
    model.safetensors
    features.safetensors
    index.index
```

Manifest contract है। इसमें target identity dataset fingerprint, RVC training settings, index settings, expected sample rate, tool versions, और SimpleTuner voice-transform format version record होने चाहिए। SimpleTuner को ऐसा Hub artifact reuse नहीं करना चाहिए जिसमें manifest न हो या manifest current transform से match न करे। इससे गलत voice model को नए dataset पर silently apply करने से बचते हैं।

Publishing opt-in होना चाहिए। Reasonable pseudo config:

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

Private identities के लिए Hub repository private रखें जब तक voice model publish करने की explicit permission न हो। Generated audio और model artifacts के sharing rights अलग हो सकते हैं, इसलिए upload settings अलग-अलग treat करें।

## WebUI Configuration

RVC model training WebUI से configurable होनी चाहिए, सिर्फ raw dataloader JSON से नहीं।

Expected WebUI shape audio datasets के लिए dataset transform editor है:

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
            Reuse from Hub: on when Hub model id is set
            Push RVC model to Hub: off by default
            Hub repo privacy: private by default
            Caption और lyrics sidecars: source audio से copy होते हैं
```

WebUI को दो common setups साफ दिखाने चाहिए:

- **Vocal stems पहले से हैं:** `vocal_only` चुनें, Demucs disabled रखें, और generated vocal stems लिखें।
- **Full songs हैं:** `separate_convert_remix` चुनें, Demucs separation use करें, सिर्फ vocal stem convert करें, और original instrumental stems के साथ remix करें।

Interface को दिखाना चाहिए कि generated audio दूसरा primary audio training split बनता है। Identity transfer को `conditioning_data` की तरह present नहीं करना चाहिए, क्योंकि इससे training के दौरान paired conditioning behavior का गलत संकेत मिलेगा।

## Distributed Startup Behavior

जब SimpleTuner multiple data-parallel ranks के साथ start होता है, voice cloning startup को available GPUs use करनी चाहिए, rank 0 से सारा काम नहीं करवाना चाहिए।

दो अलग distributed phases हैं:

1. **RVC model training:** अगर `train_if_missing=true` है, matching local cache नहीं है, और matching Hub artifact भी नहीं है, तो `world_size > 1` होने पर RVC training loop DDP में चलना चाहिए। हर rank normal distributed sampler pattern से अलग target-voice batches ले।
2. **Generated audio preparation:** source expansion inputs rank के हिसाब से split होने चाहिए, TextEmbedCache और VAECache जैसे। हर rank सिर्फ अपना shard separate, convert, और write करे, फिर सभी ranks synchronize हों और metadata discovery continue हो।

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

Final voice model को Hub पर सिर्फ एक process publish करे। Final manifest updates के लिए भी यही rule है। Per-rank generated outputs independently लिखे जा सकते हैं, जब filenames deterministic और non-overlapping हों।

इससे multi-GPU systems पर GPU time waste नहीं होता और startup behavior SimpleTuner के existing cache preparation model से aligned रहता है।

## RVC Training Logs

Startup RVC training अभी TensorBoard या WandB runs create नहीं करनी चाहिए। ये loggers main SimpleTuner training job के लिए configured होते हैं, और nested voice-conversion job के लिए reuse करने पर extra run names, paths, resume rules, और artifact policies चाहिए होंगी।

RVC stage फिर भी SimpleTuner native training logger से useful stats report कर सकता है:

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

Useful local stats में generator loss, discriminator loss, mel loss, KL loss, samples processed, elapsed time, DDP world size, cache hit या miss reason, और final model local cache, Hub cache, या on-demand training से आया या नहीं शामिल हैं।

जब तक future implementation RVC transforms के लिए external logger integration explicitly add न करे, ये stats local-only हैं।

## `audio_mode` चुनना

### `vocal_only`

जब expansion dataset पहले से clean vocal stems में preprocessed हो, इसे use करें।

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

Gotchas:

- Clean stems पर Demucs दोबारा न चलाएं जब तक कोई साफ कारण न हो।
- Captions vocals और lyrics describe करें; full band arrangement नहीं, जब तक आप बाद में remix नहीं करेंगे।
- अगर main training model full songs expect करता है, vocal-only generated data अलग distribution सिखा सकता है।

### `separate_convert_remix`

जब expansion dataset full mixed songs हो, इसे use करें।

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

Full-song expansion के लिए यह preferred mode है क्योंकि drums, bass, guitars, room sound, और mastering artifacts को voice का हिस्सा मानकर convert करने से बचता है।

Gotchas:

- Stem separation bleed, artifacts, या phase issues छोड़ सकता है।
- Vocal stem weak, reverberant, या buried हो तो converted voice unstable हो सकती है।
- Remix loudness मायने रखता है। Generated split हमेशा ज्यादा loud या quiet हो तो training bias हो सकता है।
- Captions final remixed result describe करें, सिर्फ source song नहीं।

### `full_mix_convert`

इसे सिर्फ quick tests के लिए use करें।

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

यह fast है, लेकिन usually lower quality है। Instruments voice converter से गुजर सकते हैं और final LoRA unwanted artifacts सीख सकती है।

## Captions और Lyrics Policy

Generated split की captions generated audio से match करनी चाहिए।

अच्छा default:

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

Lyrics copy करना usually सही है जब source vocal performance और converted performance वही words गाते हैं। अगर transform song बदलता है, sections edit करता है, vocals drop करता है, या non-lyrical source use करता है, तो copy सही नहीं है।

Captions blind copy नहीं होनी चाहिए। अगर source caption कहती है "female pop vocal" और converted output male rock vocal timbre है, caption adjust होनी चाहिए। Transform simple append/remove rules support करे; advanced caption rewriting बाद में add हो सकती है।

## Cache और Reuse

Transform को दो तरह की cache लिखनी चाहिए:

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

Manifest में identity dataset fingerprint, transform settings, source expansion data fingerprint, और tool versions record होने चाहिए। Values match करें तो startup existing artifacts reuse कर सकता है। Values change हों तो SimpleTuner affected stage regenerate करे।

## Practical Dataset Advice

`model.identity_data_dir` में target voice के लिए duration से ज्यादा clean voiced coverage मायने रखती है।

- **Smoke test:** 30-60 seconds का clean vocal audio pipeline चलने की पुष्टि कर सकता है, लेकिन converted voice आम तौर पर rough होगी।
- **Usable starter:** Personal voice dataset के लिए 5-10 minutes की clean isolated voice एक reasonable first target है।
- **Singing identity:** Pitch range, vowels, dynamics, articulation, और expressive phrasing चाहिए तो 10-30 minutes बेहतर है।

एक लंबे file की जगह कई short clips use करें। 5-20 seconds के clips inspect, separate, और reuse करना आसान बनाते हैं। Current RVC trainer identity audio को 48 kHz पर resample करता है और हर identity file को `max_seconds_per_file` तक truncate करता है, जिसका default `180` है। अगर user एक 30-minute file देता है, default में सिर्फ पहले तीन minutes use होंगे। Dataset split करने से useful vocal coverage गलती से discard नहीं होती।

Standalone [`huggingface-hub-rvc`](https://github.com/SimpleTuner-io/huggingface-hub-rvc) project full SimpleTuner training job चलाए बिना RVC artifact train, save, load, और publish कर सकता है। SimpleTuner के अंदर `scripts/run_rvc_model.py` pipeline के RVC training और conversion हिस्से के साथ सीधे experiment करने का entrypoint देता है। Main LoRA training पर समय खर्च करने से पहले identity dataset, Demucs mode, retrieval strength, transfer strength, या Hub artifact reuse tune करने के लिए इसका उपयोग करें।

- Identity control important हो तो एक LoRA में एक target vocalist रखें।
- Voice-conversion model के लिए clean, dry vocal examples prefer करें।
- Duets avoid करें, जब तक goal duet blend सीखना न हो।
- Expansion songs में tempo, key, genre, dynamics, और lyrical phrasing varied रखें।
- Captions varied रखें ताकि identity tokens एक arrangement से चिपक न जाएं।
- Long training runs से पहले generated audio spot-check करें।
- सब कुछ combine करने से पहले direct training data, generated data, और mixed training runs को अलग-अलग compare करें।

## Common Failure Modes

| Symptom | Likely cause |
| --- | --- |
| LoRA सिर्फ एक genre में काम करती है | Voice identity अभी भी arrangement captions या source data से entangled है। |
| Generated split hollow या phasey सुनाई देता है | Full-song processing में separation/remix artifacts। |
| Instruments voice-converted जैसे सुनते हैं | Separation चाहिए थी लेकिन `full_mix_convert` use हुआ। |
| Voice model instruments सीखता हुआ लगता है | Identity separation के vocal stems में accompaniment bleed बहुत ज्यादा है। `model.identity_stem_debug_dir` set करके saved stems inspect करें, या cleaner vocal stems preprocess करके `identity_audio_mode=vocal_only` use करें। |
| Vocal identity weak है | Voice-conversion model को cleaner target data, more data, या stronger retrieval index चाहिए। |
| Captions voice control नहीं करतीं | Captions में source-vocal identity बची है या target identity missing है। |
| Main model artifacts सीखता है | Generated audio low quality है या train mix में बहुत dominant है। |
| Converted vocals monotonic या robotic सुनाई देते हैं | RVC path में proper F0 extraction, pretrained generator/discriminator initialization, adversarial training, या enough clean target vocal data missing है। |

## Regularisation Data से संबंध

Generated identity-transfer data default रूप से regularisation data नहीं है।

Regularisation data usually LoRA को base model behavior preserve करना सिखाता है। Identity-transfer data LoRA को target voice ज्यादा contexts में सिखाता है। Too much regularisation और too little direct identity data identity tokens कमजोर कर सकते हैं। Too much generated data conversion artifacts सिखा सकता है।

इन्हें अलग controls मानें:

- direct target dataset: सबसे strong identity signal
- generated identity-transfer dataset: broader context और style coverage
- regularisation dataset: base-model preservation

## Status

यह page experimental `data_transforms` workflow describe करता है। Current implementation SimpleTuner RVC v2 F0 artifact train या reuse करती है, identity clips से HuBERT content features और RMVPE pitch extract करती है, pretrained RVC generator/discriminator fine-tune करती है, retrieval index build करती है, expanded split generate करती है, results cache करती है, और separate manual preprocessing stage के बिना normal training में continue करती है।
