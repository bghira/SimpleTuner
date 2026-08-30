# 音声クローニング Data Transform

音声クローニング data transform は、メインのモデル学習が始まる前に、あるボーカル identity を追加の曲、stem、歌唱へ転写して学習セットを拡張するための、計画中の実験的な音声データセット機能です。

目的は SimpleTuner を別の voice conversion 作業環境にすることではありません。音声 fine-tuning データの絡み合いを減らすことです。ある歌手が狭いスタイルにしか出てこない場合、LoRA は歌手本人ではなく「この編曲の中のこの歌手」を学んでしまうことがあります。voice-cloned expansion split により、同じ声をより多様な編曲、caption、歌詞、曲構造で見せられます。

この機能は音声データセット専用です。

!!! warning "同意と権利"
    この workflow は、使用する権利がある声と録音だけに使ってください。声の identity はセンシティブな生体情報であり創作データです。この transform は実在人物のように聞こえる派生音声を作れるため、許可、ライセンス、開示が重要です。

## ELI5

同じ歌手の録音が 6 曲あるとして、その全てが同じバンド、同じジャンルだとします。それだけで学習すると、モデルは歌手、ギター音色、ドラムの感触、テンポ範囲、曲構造を一体のものとして覚えるかもしれません。

音声クローニング data transform は、それらを分けるためのものです。

1. 歌手サンプルから小さな voice-conversion model を学習します。
2. より広い曲や vocal stem のセットを読み込みます。
3. 元のボーカル timbre を対象歌手の timbre に置き換えます。
4. 新しい caption と歌詞を生成音声に合わせます。
5. 生成音声を通常の training split として追加します。

これにより、メインモデルは元の狭いデータセットを暗記するだけでなく、より多くの文脈で対象の声を見ることができます。

## 使う場面

使うべき場面:

- 対象ボーカリストの許可された録音がある
- 対象 identity が単一ジャンル、バンド、制作スタイル、曲構造と絡みすぎている
- trigger word が元のドメイン内でしか効かない
- 1 つのデータセットに複数の歌手がいて平均化された声になる
- 声ごとに別々の LoRA を作りたい
- SimpleTuner に同じ training setup の中で expansion split を準備してほしい

避けるべき場面:

- 同じ声の大規模で多様かつクリーンなデータが既にある
- expansion source の音質が悪い、または caption と合っていない
- 公開リリース向けの権利が明確でない
- クリーンな直接サンプルでもベース生成モデルが対象 identity を学べない

## Training への入り方

音声クローニングはデータ準備 transform であり、conditioning dataset ではありません。

`conditioning_data` は、reference image や generated conditioning map のように、training 中に主サンプルへ付いたままになる補助入力のためのものです。

音声クローニングは、dataset レベルの `data_transforms` リストに入るべきです。transform は新しい音声ファイル、caption、必要なら歌詞を materialize し、その結果を別の primary `audio` dataset として登録します。その後、通常の dataloader は他の training split と同じように扱います。

疑似 config:

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

起動時の疑似処理:

```text
for each audio dataset:
    for each data transform:
        if task is identity_transfer:
            prepare or reuse the target voice-conversion model
            prepare or reuse generated audio
            append generated audio as a normal train split

continue with normal metadata discovery, bucketing, caching, and training
```

## RVC 形式の Identity Transfer

最初の実装は RVC 形式の voice conversion で、HuBERT content features、RMVPE pitch extraction、NSF/VITS generator、multi-period discriminator、mel/adversarial losses、optional retrieval index を使います。

ここでいう「RVC model」は voice-specific です。対象 identity dataset から学習されます。retrieval index も voice-specific で、同じ対象声の特徴から構築されます。content feature、pitch 抽出、separation model などの広い pretrained component は再利用される基盤です。一方、conversion model と index は歌手または話者固有の artifact です。

SimpleTuner は以下を行えるべきです。

1. 提供済みの voice-conversion model と index を再利用する。
2. model がなければ voice-conversion model を学習する。
3. 対象声データから retrieval index を構築する。
4. model、index、生成音声を training output directory 以下に cache する。
5. source data と transform 設定が変わっていなければ起動時に cache を再利用する。
6. 必要に応じて Hub model repository から voice-conversion model を再利用、またはそこへ publish する。

## Default Behavior

default は保守的です。この workflow では、audio backend は変換したい拡張用の楽曲、`model.identity_data_dir` は対象 voice dataset、`target.instance_data_dir` は生成 output split の path です。

| Setting | Default | 理由 |
| --- | --- | --- |
| `task` | `identity_transfer` | transform を明示します。 |
| `method` | `rvc` | 最初の voice-transfer backend です。 |
| `train_if_missing` | `true` | SimpleTuner が対象 dataset から voice model を bootstrap できるようにします。 |
| `force_retrain` | `false` | 有効な cache model をできるだけ再利用します。 |
| `build_index` | `true` | retrieval は identity の安定性を上げ、漏れを減らしやすいです。 |
| `identity_data_dir` | on-demand training では必須 | 拡張楽曲へ移したい対象 voice の clean vocal examples を指します。 |
| `identity_audio_mode` | `separate` | training 前に identity clips へ Demucs を実行します。identity dataset が既に vocal stems の場合は `vocal_only` を使います。 |
| `identity_stem_debug_dir` | unset | identity の `vocals.wav` と `no_vocals.wav` preview を保存する任意の directory です。RVC が楽器 bleed ではなく分離済み vocal から training しているか確認できます。 |
| `asset_hub_model_id` | `lj1995/VoiceConversionWebUI` | Default RVC asset repository for HuBERT, RMVPE, and v2 48k pretrained generator/discriminator checkpoints. |
| `model_name` | transform or Hub repo name | Human-readable name saved into the RVC artifact so downloaded caches are identifiable outside their folder name. |
| `sample_rate` | `48000` | Current implementation targets RVC v2 48k assets. Other rates need matching pretrained assets and configs. |
| `training_steps` | `1000` | Runs RVC generator/discriminator fine-tuning during startup. Increase for larger or more varied identity datasets. |
| `batch_size` | `4` | RVC training batch size before distributed sharding. Lower it for memory pressure. |
| `learning_rate` | `1e-4` | Standard RVC AdamW default. |
| `hub_model_id` | unset | user が opt-in しない限り remote voice-model cache は使いません。 |
| `reuse_from_hub` | `hub_model_id` が設定されている場合は `true` | on-demand model を学習する前に Hub を確認します。 |
| `push_to_hub` | `false` | voice model は声の identity を表すため、upload は明示的であるべきです。 |
| `public` | `false` | Hub uploads are private by default. Set this to `true` only when the voice artifact can be published publicly. |
| `audio_mode` | full song は `separate_convert_remix`、vocal stem は `vocal_only` | full mix には分離が必要で、stem には不要です。 |
| `separation_method` | 分離が必要なら `demucs` | Demucs が想定 default の stem separator です。 |
| `timbre_strength` | `1.0` | Controls how strongly the synthesized target voice replaces the source vocal. Lower values blend source and converted vocals. |
| `retrieval_strength` | `0.75` | Blends nearest target-voice content frames from the retrieval index into the generator input. |
| generated split type | primary `audio` dataset | 生成データは conditioning ではなく通常音声として学習します。 |
| cache location | `output_dir` 内 | artifact を training run に結び付け、restart で再利用しやすくします。 |
| captions | 設定がなければ source captions を copy | 新しい split は歌詞と編曲文脈を保持すべきです。 |

既存の voice-conversion model が指定されている場合、SimpleTuner はそれを使い、明示的に要求された場合や必要 artifact が欠けている場合だけ新規学習すべきです。

## Hub Cache

voice-conversion model は再学習コストが高くなることがあり、毎回 on-demand training すると無駄が大きくなります。そのため transform は、voice model と retrieval index の optional Hub-backed cache をサポートすべきです。

安全な lookup order:

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

Hub repository は、単なる file collection ではなく SimpleTuner-specific layout を使うべきです。

```text
config.json
voice_transform/
    manifest.json
    model.safetensors
    features.safetensors
    index.index
```

manifest が contract です。target identity dataset fingerprint、RVC training settings、index settings、expected sample rate、tool versions、SimpleTuner voice-transform format version を記録します。manifest がない、または current transform と一致しない Hub artifact を SimpleTuner は再利用すべきではありません。これにより、間違った voice model を新しい dataset に黙って適用する事故を避けられます。

publish は opt-in にします。妥当な pseudo config:

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

private identity の場合、明示的な許可がない限り Hub repository は private のままにしてください。generated audio と model artifact は共有権利が異なる場合があるため、upload settings は別々に扱います。

## WebUI Configuration

RVC model training は raw dataloader JSON だけでなく WebUI から設定できるべきです。

想定される WebUI 形状は、audio dataset 用の dataset transform editor です。

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
            Caption と lyrics の sidecar: source audio からコピー
```

WebUI はよくある 2 つの setup を分かりやすくするべきです。

- **既に vocal stems がある場合:** `vocal_only` を選び、Demucs は無効のまま、generated vocal stems を書き出します。
- **full songs がある場合:** `separate_convert_remix` を選び、Demucs separation を使い、vocal stem だけを変換して original instrumental stems と remix します。

interface は、generated audio が別の primary audio training split になることを示すべきです。identity transfer を `conditioning_data` として見せてはいけません。training 中の paired conditioning behavior だと誤解されるためです。

## Distributed Startup Behavior

SimpleTuner が複数の data-parallel rank で起動する場合、voice cloning startup は rank 0 だけに作業させるのではなく、利用可能な GPU を使うべきです。

分散処理には 2 つの段階があります。

1. **RVC model training:** `train_if_missing=true` で、matching local cache がなく、matching Hub artifact もない場合、`world_size > 1` なら RVC training loop は DDP で動くべきです。各 rank は通常の distributed sampler pattern で異なる target-voice batch を受け取ります。
2. **Generated audio preparation:** expansion source inputs は TextEmbedCache や VAECache と同じように rank ごとに分割します。各 rank は自分の shard だけを separate、convert、write し、全 rank が同期してから metadata discovery を続けます。

疑似処理:

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

final voice model を Hub に publish する process は 1 つだけにします。final manifest update も同じです。per-rank generated outputs は、filenames が deterministic で重複しない限り、各 rank が独立して書けます。

これにより multi-GPU system で GPU 時間を無駄にせず、startup behavior を SimpleTuner の既存 cache preparation model と揃えられます。

## RVC Training Logs

startup RVC training は、現時点では TensorBoard や WandB run を作成すべきではありません。これらの logger は main SimpleTuner training job 用に設定されており、nested voice-conversion job に再利用すると、追加の run names、paths、resume rules、artifact policies が必要になります。

RVC stage は SimpleTuner native training logger を通して有用な stats を記録できます。

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

有用な local stats には、generator loss、discriminator loss、mel loss、KL loss、processed samples、elapsed time、DDP world size、cache hit/miss reason、final model が local cache、Hub cache、on-demand training のどれから来たかが含まれます。

これらの stats は、将来 RVC transforms 向けの external logger integration が明示的に追加されるまでは local-only です。

## `audio_mode` の選び方

### `vocal_only`

expansion dataset が既にクリーンな vocal stem に preprocessing 済みの場合に使います。

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

注意点:

- 理由がない限り、クリーンな stem に Demucs を再実行しないでください。
- caption は vocal と lyrics を説明します。後で伴奏へ remix しないなら、full band arrangement として説明しないでください。
- メイン training model が full song を期待する場合、vocal-only 生成データは別の分布を教える可能性があります。

### `separate_convert_remix`

expansion dataset が full mixed song の場合に使います。

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

これは full-song expansion の推奨 mode です。drums、bass、guitars、room sound、mastering artifact を声の一部として変換するのを避けられます。

注意点:

- stem separation は bleed、artifact、phase 問題を残すことがあります。
- vocal stem が弱い、reverb が多い、埋もれている場合、変換声が不安定になります。
- remix loudness は重要です。生成 split が常に大きすぎたり小さすぎたりすると training に偏りが出ます。
- caption は source song だけでなく最終 remix 結果を説明するべきです。

### `full_mix_convert`

簡単なテストだけに使います。

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

高速ですが、通常は品質が低くなります。楽器まで voice converter に通してしまい、不要な artifact を最終 LoRA に教える可能性があります。

## Caption と Lyrics

生成 split の caption は、生成された音声と一致している必要があります。

よい default:

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

source vocal performance と converted performance が同じ歌詞を歌う場合、lyrics の copy は普通は正しいです。曲を変える、section を落とす、vocal を削除する、非 lyrics source を使う場合は正しくありません。

caption は盲目的に copy できません。source caption が "female pop vocal" で、変換後が male rock vocal timbre なら調整が必要です。transform は単純な append/remove rules を持つべきで、高度な caption rewrite は後から重ねられます。

## Cache と Reuse

transform は 2 種類の cache を書くべきです。

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

manifest には identity dataset fingerprint、transform settings、source expansion data fingerprint、tool versions を記録します。一致すれば起動時に既存 artifact を再利用できます。変わった場合は、影響を受ける stage だけ再生成します。

## Dataset Advice

`model.identity_data_dir` の target voice では、長さそのものより clean voiced coverage が重要です。

- **Smoke test:** 30-60 秒の clean vocal audio で pipeline が動くことは確認できますが、converted voice は通常かなり粗くなります。
- **Usable starter:** 個人 voice dataset の最初の目標としては、5-10 分の clean isolated voice が現実的です。
- **Singing identity:** pitch range、vowels、dynamics、articulation、expressive phrasing が必要な場合は 10-30 分の方が良いです。

1 つの長い file ではなく、多数の短い clip を使ってください。5-20 秒程度の clip は確認、分離、再利用がしやすくなります。現在の RVC trainer は identity audio を 48 kHz に resample し、各 identity file を `max_seconds_per_file` で truncate します。default は `180` です。ユーザーが 30 分の file を 1 つ渡した場合、default では最初の 3 分だけが使われます。dataset を分割すると、有用な vocal coverage を誤って捨てることを避けられます。

standalone の [`huggingface-hub-rvc`](https://github.com/SimpleTuner-io/huggingface-hub-rvc) project は、SimpleTuner の full training job を実行せずに RVC artifact を train、save、load、publish できます。SimpleTuner 内では `scripts/run_rvc_model.py` が、pipeline の RVC training/conversion 部分を直接試す entrypoint です。main LoRA training に時間を使う前に、identity dataset、Demucs mode、retrieval strength、transfer strength、Hub artifact reuse を調整したい場合に使ってください。

- identity control が重要なら、1 つの LoRA に target vocalist は 1 人だけにします。
- voice-conversion model にはクリーンで dry な vocal examples を優先します。
- duet blend を学びたい場合以外は duet を避けます。
- expansion songs は tempo、key、genre、dynamics、lyrical phrasing を広くします。
- caption を十分に多様化し、identity token が 1 つの編曲に貼り付かないようにします。
- 長い training run の前に生成音声を spot-check します。
- すべてを組み合わせる前に、direct training data、generated data、mixed training run を別々に比較します。

## Common Failure Modes

| 症状 | ありがちな原因 |
| --- | --- |
| LoRA が 1 ジャンルでしか効かない | 声 identity が arrangement caption や source data とまだ絡んでいます。 |
| generated split が phasey または hollow に聞こえる | full-song 処理の separation/remix artifact。 |
| 楽器まで変声されたように聞こえる | 分離が必要なのに `full_mix_convert` を使っています。 |
| voice model が楽器まで学習したように聞こえる | identity separation の vocal stem に accompaniment bleed が多すぎます。`model.identity_stem_debug_dir` で保存された stems を確認するか、より clean な vocal stems を前処理して `identity_audio_mode=vocal_only` を使ってください。 |
| vocal identity が弱い | target data の品質、量、または retrieval index が不足しています。 |
| caption が声を制御しない | source vocal identity が caption に残っている、または target identity がありません。 |
| main model が artifact を学ぶ | generated audio の品質が低い、または train mix 内で強すぎます。 |
| converted vocal が monotonic または robotic に聞こえる | RVC path に適切な F0 extraction、pretrained generator/discriminator initialization、adversarial training、または十分に clean な target vocal data が不足しています。 |

## Regularisation Data との関係

生成された identity-transfer data は、default では regularisation data ではありません。

regularisation data は通常、LoRA に base model の挙動を保たせるためのものです。identity-transfer data は、より多くの文脈で target voice を教えるためのものです。regularisation が多すぎて direct identity data が少なすぎると identity token が弱くなります。generated data が多すぎると conversion artifact を教えることがあります。

別々のレバーとして扱ってください。

- direct target dataset: 最も強い identity signal
- generated identity-transfer dataset: より広い context と style coverage
- regularisation dataset: base-model preservation

## Status

このページは、実験的な `data_transforms` workflow を説明します。現在の実装は SimpleTuner RVC v2 F0 artifact を学習または再利用し、identity clips から HuBERT content features と RMVPE pitch を抽出し、pretrained RVC generator/discriminator を fine-tune し、expanded split を生成して cache し、別の手動 preprocessing stage なしで通常 training へ進みます。
