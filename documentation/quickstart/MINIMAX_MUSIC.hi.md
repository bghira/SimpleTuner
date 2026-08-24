# MiniMax Music 3 Quickstart

यह गाइड SimpleTuner में MiniMax Music 3 LoRA training कॉन्फिगर करता है।

## Overview

MiniMax Music 3 caption और lyrics conditioned music generator है। Diffusers layout में Qwen3 autoregressive language model text/audio conditioning बनाता है, flow-matching transformer 128-channel DAV latents पर train होता है, और decoder/vocoder validation audio बनाता है।

SimpleTuner support करता है:

- transformer के लिए LoRA, LyCORIS और full-rank training
- original `dav.pth` autoencoder से raw audio VAECache encoding
- audio dataset metadata से caption, lyrics और duration
- `validation_prompt`, `validation_lyrics`, `validation_audio_duration` और prompt libraries के साथ validation audio
- `lora_format: "comfyui"` के साथ ComfyUI MiniMax Music LoRA import/export
- AnyFlow, TwinFlow, CREPA self-flow और LayerSync

## Hardware Requirements

MiniMax Music 3 में 2.4B flow transformer और 8B Qwen3 AR conditioning model है।

- **Minimum:** conservative LoRA training के लिए 24GB+ VRAM वाली NVIDIA GPU।
- **Recommended:** बड़े rank, लंबे clips और frequent validation के लिए 48GB+ VRAM या CPU/RAM offload।
- **Mac:** कुछ हिस्सों में MPS काम कर सकता है, लेकिन training और validation के लिए practical target CUDA है।

`base_model_precision: "int8-quanto"`, `text_encoder_1_precision: "int8-quanto"` और `gradient_checkpointing: true` से शुरू करें। अगर text encoder bottleneck रहे, तो LoRA rank बढ़ाने से पहले text encoder offload इस्तेमाल करें।

## Prerequisites

SimpleTuner और audio loading के लिए FFmpeg install करें:

```bash
pip install simpletuner
```

Manual install या development setup के लिए [installation documentation](../INSTALL.md) देखें।

## Configuration

Dedicated config folder बनाएँ:

```bash
mkdir -p config/minimaxmusic-training-demo
```

`config/minimaxmusic-training-demo/config.json` बनाएँ:

<details>
<summary>Example config देखें</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

Ready-made templates:

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

Example run करें:

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

MiniMax Music 3 raw audio caching DAV audio autoencoder इस्तेमाल करता है। Recommended SimpleTuner VAE repo `SimpleTuner/MiniMax-Music-3-Encoder` है, जहाँ converted component `audio_vae/` में Diffusers-style loading के लिए रखा गया है।

Upstream `MiniMaxAI/MiniMax-Music3` repo में original `dav.pth` भी है, और SimpleTuner उसे सीधे load कर सकता है। अगर local converted Diffusers directory इस्तेमाल कर रहे हैं, तो checkpoint root में `dav.pth` रखें या `pretrained_vae_model_name_or_path` को ऐसे path या Hub repo पर point करें जिसमें `dav.pth` या `audio_vae/` हो। केवल `vocoder/` subfolder validation decode के लिए काफी है, लेकिन raw audio VAE caching के लिए नहीं।

## Dataset

MiniMax Music 3 को एक **audio** dataset और एक **text embeds** cache backend चाहिए।

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```

Local audio files के लिए `.txt` में description और `.lyrics` में lyrics रखें:

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## Validation

- **`validation_prompt`**: music description या tags।
- **`validation_lyrics`**: sung lyrics। instrumental validation के लिए empty string इस्तेमाल करें।
- **`validation_audio_duration`**: generated clip duration in seconds।
- **`validation_guidance`**: CFG scale। `1.5` से `2.0` के आसपास शुरू करें।
- **`validation_num_inference_steps`**: sampling steps। `30` के आसपास शुरू करें।
- **`validation_steps`**: कितनी training steps के बाद validation render हो।
- **`validation_prompt_library`**: built-in music caption + lyrics library के लिए `"audio"` इस्तेमाल करें।
- **`user_prompt_library`**: JSON library path। Entries `prompt` या `caption` और optional multiline `lyrics` इस्तेमाल कर सकती हैं।

## Training

```bash
simpletuner train env=minimaxmusic-training-demo
```

Existing MiniMax Music 3 LoRA से शुरू करने के लिए:

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

अगर adapter native ComfyUI format में है, तो config में `lora_format: "comfyui"` रखें। SimpleTuner training के समय convert करेगा और उसी format में export करेगा।

## Advanced Features

MiniMax Music 3 SimpleTuner के flow-matching training path का उपयोग करता है, इसलिए AnyFlow, TwinFlow, CREPA self-flow और LayerSync उपलब्ध हैं। पहले standard LoRA से शुरू करें और advanced features एक-एक करके enable करें।

## भाषा मॉडल (AR चरण) प्रशिक्षण

MiniMax Music 3 के सिमेंटिक कोड की योजना बनाने वाले Qwen3 भाषा मॉडल को संगीत DiT के बजाय प्रशिक्षित किया जा सकता है — dreambooth-शैली के ट्रिगर शब्दों के लिए उपयोगी, जो किसी संगीत शैली को एक कीवर्ड से बांधते हैं।

इस मोड से बनाए गए पूर्ण LM LoRA प्रशिक्षण उदाहरण के लिए [fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple) देखें; इसमें सेटिंग्स, चेकपॉइंट और ऑडियो तुलनाएँ शामिल हैं।

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

आवश्यकताएँ और DiT प्रशिक्षण से अंतर:

- प्रत्येक डेटासेट नमूने में `prompt` (या `tags`), `lyrics`, और `audio_tokens_path` मेटाडेटा होना चाहिए जो `[frames, codebooks]` आकार के कच्चे प्रति-कोडबुक RVQ कोड की `.pt` फ़ाइल की ओर इशारा करे (सिमेंटिक कोड `< 16384`, अवशिष्ट कोड `< audio_vocab_size`, कोई शब्दावली ऑफ़सेट नहीं)। इन्हें समर्पित `minimax-music3-latent-replanner` रिपॉजिटरी के `precompute_rvq_codes.py --raw-codes` से निर्यात करें।
- हानि सिमेंटिक कोडबुक पर नेक्स्ट-टोकन क्रॉस-एंट्रॉपी है, जो ऑडियो स्थितियों तक सीमित है; RVQ depth decoder स्थिर रहता है और अवशिष्ट-कोड इनपुट एम्बेडिंग प्रदान करता है।
- केवल मानक PEFT LoRA समर्थित है और `lora_format: "comfyui"` अस्वीकार किया जाता है। चेकपॉइंट `language_model.` उपसर्ग वाली कुंजियों के साथ `pytorch_lora_weights.safetensors` सहेजते हैं।
- इस मोड में ट्रेनर के भीतर सत्यापन ऑडियो अक्षम है; सहेजे गए चेकपॉइंट से मानक जनरेशन स्टैक के साथ रेंडर करें।
- इस मोड में कोई VAE या टेक्स्ट-एम्बेड कैशिंग नहीं होती — प्रशिक्षण सीधे टोकन पढ़ता है, इसलिए `cache_dir_vae` और टेक्स्ट एम्बेड बैकएंड उपयोग नहीं होते।
- अपना ट्रिगर कीवर्ड (जैसे `"fiona crapple"`) हर नमूने के caption/`prompt` फ़ील्ड में रखें; गीत ज्यों के त्यों रखें।
- Short capped runs के लिए `minimax_music_lm_window_mode: "random"` सेट करें ताकि हमेशा intros पर प्रशिक्षण न होकर positioned RVQ windows sample हों। Random windows prompt में start/end/duration जोड़ती हैं और full-track lyrics हटाती हैं, जब तक sample `lyrics_window` न दे।
- Cropped-window training को हर crop को finished clip की तरह सिखाने न दें। अगर outputs बार-बार crop boundaries पर fade out या resolve करते हैं, तो crop labels और targets जाँचें: interior windows को interior windows की तरह supervise किया जाना चाहिए, और end-of-audio behavior केवल वास्तविक song endings पर सिखाया जाना चाहिए।
- Song structure training के लिए `minimax_music_lm_window_mode: "continuation"` इस्तेमाल करें। यह target window sample करता है, track की शुरुआत से उस window तक सभी audio tokens को causal context रखता है, और पिछले context का loss mask करता है। यह isolated random crop से अधिक memory लेता है, लेकिन हर excerpt को song opening की तरह सिखाने से बचाता है।
- छोटे LM audio datasets पर aggressive optimizers सावधानी से इस्तेमाल करें। High learning rates पर Prodigy बहुत आगे निकल सकता है, और Lion पहले हजार steps में over-adapt कर सकता है; तेज optimizers आज़माने से पहले AdamW को baseline रखें।
- **प्रायर संरक्षण**: `is_regularisation_data: true` के साथ instrumentals या असंबंधित गीतों वाला दूसरा ऑडियो बैकएंड जोड़ें (खाली गीत मान्य हैं)। उन बैचों पर हानि वास्तविक कोड के बजाय स्थिर आधार मॉडल के अपने नेक्स्ट-टोकन वितरण को लक्षित करती है, जिससे LoRA सटीक रहता है: regularisation captions ठीक वैसे ही भविष्यवाणी करते रहते हैं जैसे आधार मॉडल करता, और शैली का रिसाव बहुत कम हो जाता है।

### Style और singer datasets कैसे configure करें

Music style adaptation और singer-identity adaptation को अलग dataset design चाहिए। Singer name को detailed music caption का shortcut न मानें।

#### Music styles

Music styles अपेक्षाकृत forgiving होते हैं। यदि लक्ष्य genre, arrangement या production style है, न कि कोई खास vocal timbre, तो 24 या अधिक varied tracks वाला set उपयोगी adapter के लिए पर्याप्त हो सकता है।

- Target style से बाहर गए बिना diversity बढ़ाएँ। ऐसे tempos, instruments, production choices, moods और neighboring subgenres शामिल करें जिन्हें user inference में मांग सकता है।
- हर audio sample को कई complete style captions दें। केवल trigger word dataset को averaged association में compress करता है और उसके range को reproduce करने के controls नहीं सिखाता।
- Vocal timbre को incidental मानें। Multiple vocalists या instrumental material इस्तेमाल करें ताकि कोई एक voice गलती से learned style का हिस्सा न बन जाए।
- Fixed validation prompts और कई checkpoint strengths से collapse देखें। Style adapters अक्सर बहुत अधिक optimizer steps से पहले ही useful हो जाते हैं।

`caption_strategy: "textfile"` और default `disable_multiline_split: false` के साथ, `.txt` sidecar की हर non-empty line अलग caption candidate होती है। SimpleTuner हर बार उस audio item को sample करते समय एक candidate चुनता है; वह सभी lines को एक grouped caption में combine नहीं करता। DiT workflows हर distinct caption को अलग cache करते हैं, जबकि LM training selected caption को online tokenize करती है और text-embedding cache का उपयोग नहीं करती। उदाहरण:

```text
syncopated art rock, dry drums, angular guitar, abrupt dynamic changes
melodic alternative metal, layered harmonies, restless bass, theatrical pacing
tense progressive rock, odd-meter accents, sparse verse, explosive refrain
```

यह caption augmentation है, multiline prompt नहीं: किसी training example के लिए model इन lines में से एक देखता है।

#### Singer identity

Singer identity बहुत कम forgiving है। हर singer के लिए अलग adapter बनाएँ और ऐसी हर track या section हटाएँ जिसमें दूसरा vocalist हो, जैसे duet, alternating verses, backing leads और guest appearances। `[Verse: ...]` या `[Chorus: ...]` जैसे lyric tags mixed voices को reliably disentangle नहीं करते।

- हर caption candidate में वही unique singer trigger रखें, उसके बाद complete और varied style description दें। Trigger को एक line पर और descriptions को अलग lines पर रखना गलत है, क्योंकि एक समय में केवल एक line चुनी जाती है।
- Narrow single-genre singer dataset आम तौर पर उस arrangement के भीतर singer सीखता है, portable singer identity नहीं। Identity delta उस genre, instrumentation, mix और song structure से entangled हो जाता है जिसके साथ वह हमेशा co-occur करता है, इसलिए trigger केवल in-domain काम कर सकता है। Cross-genre vocal control के लिए singer dataset में meaningful genre और arrangement variety चाहिए।
- Lyrics faithful रखें, लेकिन identity सिखाने के लिए lyric section labels पर निर्भर न रहें। उपयोगी signal audio और caption association से आता है।
- बहुत छोटे corpus के लिए instrumental counterparts prior preservation दे सकते हैं। Six carefully isolated vocal tracks workable हो सकते हैं जब उन्हें उन्हीं tracks से बनी regularisation के साथ pair किया जाए।

```text
vocalist_xyz, sparse alternative rock, dry drums, tense verse, explosive refrain
vocalist_xyz, melodic art metal, layered guitar, mid-tempo groove, close vocal
vocalist_xyz, acoustic chamber rock, hand percussion, soft opening, dramatic lift
```

एक practical regularisation workflow vocals हटाने के लिए Demucs इस्तेमाल करता है:

```bash
python -m demucs --two-stems=vocals path/to/track.wav
```

हर resulting `no_vocals.wav` को अलग audio backend में रखें, style-only `.txt` caption के साथ, singer trigger के बिना, और `.lyrics` sidecar में `[Instrumental]` लिखें। उस backend पर `is_regularisation_data: true` set करें। Regularisation batches frozen base planner को target करते हैं, जिससे adapter "यह music" और "यह singer" को अलग करना सीखता है, छोटे vocal corpus के आसपास पूरे style को rewrite करने के बजाय।

बड़े और diverse single-singer corpus के लिए इस regularisation branch के बिना शुरू करें और इसे केवल तब जोड़ें जब validation style bleed या base-model damage दिखाए। जब vocal dataset पहले से पर्याप्त coverage देता है, regularisation identity acquisition को धीमा कर सकता है। एक plausible explanation यह है कि extra preservation signal पहले से diverse identity gradient को और dilute करता है, लेकिन इसे tuning hypothesis मानें, general rule नहीं।

## Troubleshooting

- **`VAE caching requires the original dav.pth checkpoint`**: `SimpleTuner/MiniMax-Music-3-Encoder` या `MiniMaxAI/MiniMax-Music3` इस्तेमाल करें, local checkpoint root में `dav.pth` रखें, या `pretrained_vae_model_name_or_path` को ऐसे location पर point करें जहाँ यह मौजूद हो।
- **Lyrics missing हैं**: backend metadata में `lyrics` confirm करें, या `caption_strategy: "textfile"` के साथ audio files के पास `.lyrics` sidecar रखें।
- **Text embedding या validation OOM**: `validation_audio_duration` घटाएँ, int8 text encoder precision इस्तेमाल करें, या text encoder offload enable करें।

## संबंधित MiniMax Music 3 प्रयोग

- [ओपन RVQ encoder](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [RVQ reference-audio integration](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [Fiona Crapple LM LoRA](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Latent refiner](https://github.com/bghira/minimax-music3-latent-refiner) और [v0.10 weights](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Latent replanner](https://github.com/bghira/minimax-music3-latent-replanner) और [experiment log](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
