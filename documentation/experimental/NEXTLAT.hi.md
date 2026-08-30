# NextLat

NextLat एक auxiliary training objective है जो transformer को अपने hidden states से next hidden state predict करना सिखाता है।

Original Next-Latent Prediction paper language-style transformers पर केंद्रित है और बताता है कि standard next-token prediction model को history को compact और stable internal states में compress करने के लिए मजबूर नहीं करता। NextLat latent space में self-supervised transition objective जोड़ता है: current hidden state से next hidden state predict करना। SimpleTuner में यह supported transformer model families के लिए experimental regularizer है।

Inference नहीं बदलता। NextLat training loss और एक छोटा predictor add करता है; नया sampler नहीं।

## ELI5

Standard training कहता है: "जो देखा है उससे next output predict करो।"

NextLat जोड़ता है: "अपनी internal notes को भी इतनी अच्छी बनाओ कि वे अगली internal notes predict कर सकें।"

Image, video, और audio models में ये internal notes transformer के hidden tokens हैं। अगर model smooth hidden-state transitions सीखता है, तो वह tokens, frames, patches, या RVQ code positions के बीच ज्यादा coherent plan बना सकता है।

## NextLat क्या बदलता है

Training के दौरान:

1. SimpleTuner एक transformer block से hidden states capture करता है।
2. Predictor last hidden token को छोड़कर बाकी hidden tokens लेता है।
3. वह following hidden token predict करता है।
4. Real following hidden token detached target के रूप में उपयोग होता है।
5. Auxiliary loss normal training loss में add होती है।

Base model अपना normal objective सीखता रहता है। NextLat side objective है जो internal states को predictive dynamics देता है।

## Pseudocode

```text
for each batch:
    prediction = model(batch)
    main_loss = normal_training_loss(prediction, target)

    hidden = captured_hidden_states
    current_hidden = hidden tokens 0..N-2
    next_hidden = hidden tokens 1..N-1

    predicted_next_hidden = nextlat_predictor(current_hidden)
    nextlat_loss = distance(predicted_next_hidden, stop_gradient(next_hidden))

    total_loss = main_loss + nextlat_weight * nextlat_loss
    train_on(total_loss)
```

यदि model family compatible logits head देती है, optional KL भी जोड़ा जा सकता है:

```text
predicted_logits = logits_head(predicted_next_hidden)
target_logits = logits_head(stop_gradient(next_hidden))
total_loss += nextlat_kl_weight * agreement_loss(predicted_logits, target_logits)
```

अधिकतर users को `nextlat_kl_weight=0` रखना चाहिए।

## SimpleTuner में व्यवहार

- यह hidden states expose करने वाली transformer families पर काम करता है।
- `nextlat_block_index` से एक block capture होता है।
- `-1` final supported block का मतलब है।
- Image, video, audio, या token hidden states को sequence में flatten किया जाता है।
- Hidden-token order में one step ahead predict किया जाता है।
- Target hidden state detached होता है।
- Training mode support करे तो predictor extra trainable module की तरह save होता है।

Model guide अलग adapter mode न बताए तो standard PEFT LoRA या full-model training उपयोग करें।

## Quick Setup

### WebUI

1. **Training → Loss functions** खोलें।
2. **NextLat** enable करें।
3. First run के लिए **NextLat Block Index** को `-1` रखें।
4. **NextLat Weight** को छोटा positive value दें।
5. **NextLat State Loss** को `smooth_l1` रखें।
6. Model guide recommendation न हो तो **NextLat KL Weight** को `0` रखें।

### Config JSON / CLI

```json
{
  "nextlat_enabled": true,
  "nextlat_block_index": -1,
  "nextlat_weight": 0.05,
  "nextlat_state_loss": "smooth_l1",
  "nextlat_kl_weight": 0.0
}
```

## Settings

- `nextlat_enabled`: NextLat enable करता है।
- `nextlat_block_index`: zero-based transformer block; `-1` final supported block उपयोग करता है।
- `nextlat_weight`: auxiliary hidden-state prediction loss का multiplier; enabled हो तो zero से बड़ा चाहिए।
- `nextlat_state_loss`: default `smooth_l1`; `mse` भी उपलब्ध।
- `nextlat_kl_weight`: compatible logits head होने पर optional KL agreement weight।

## Values कैसे चुनें

| स्थिति | Suggested start |
| --- | --- |
| First transformer LoRA run | `nextlat_block_index=-1`, `nextlat_weight=0.02` से `0.05` |
| AR/RVQ planner | late block, `smooth_l1`, छोटा weight |
| Video transformer | final block बहुत restrictive लगे तो middle-to-late block |
| Auxiliary loss unstable | block बदलने से पहले `nextlat_weight` घटाएँ |
| Model guide KL recommend करे | सिर्फ documented value उपयोग करें |

## Logs

- `nextlat_loss`: training objective में added weighted auxiliary loss।
- `nextlat_state_loss`: raw hidden-state prediction loss।
- `nextlat_kl_loss`: optional KL term।

Raw state loss trend देखने के लिए है; इसका scale main diffusion, flow, या token loss जैसा होना जरूरी नहीं।

## Compatibility

Current support के लिए [Quick Start](../QUICKSTART.hi.md) की feature table देखें।

Requirements:

- Model transformer hidden states expose करे।
- Selected block मौजूद और capturable हो।
- Captured sequence में कम से कम दो hidden tokens हों।
- Training mode NextLat predictor save कर सके।

NextLat LayerSync, Internal Guidance, और CREPA जैसी hidden-state capture features के साथ naturally pair कर सकता है, लेकिन memory pressure बढ़ता है क्योंकि hidden states auxiliary loss तक available रहने चाहिए।

## क्या Expect करें

NextLat तब ज्यादा useful हो सकता है जब coherent internal transitions important हों:

- autoregressive audio-code या RVQ planners
- temporal structure वाले video transformers
- spatial token order वाले image transformers
- stable internal plan चाहने वाले multimodal models

बहुत छोटे experiments, बहुत बड़े auxiliary weight, या useful hidden states expose न करने वाली family पर असर कमजोर हो सकता है।

## Practical Advice

- पहले short ablation run करें।
- `nextlat_weight` छोटा रखें; validation improve हो तभी बढ़ाएँ।
- Reason न हो तो `smooth_l1` उपयोग करें।
- पहले `-1`, फिर जरूरत हो तो middle-to-late block try करें।
- Model guide न कहे तो KL disabled रखें।
- VRAM ज्यादा बढ़े तो batch size घटाएँ या other hidden-state regularizers disable करें।

## References

- [Next-Latent Prediction paper](https://arxiv.org/abs/2511.05963)
- [NextLat reference code](https://github.com/JaydenTeoh/NextLat)
