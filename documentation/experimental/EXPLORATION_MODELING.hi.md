# Explorative Modeling (XM)

Explorative Modeling, जिसे SimpleTuner में XM कहा जाता है, एक training-time technique है जिसमें model एक ही supervised example के लिए कई hidden choices try करता है और फिर सिर्फ उस choice से सीखता है जो target को सबसे अच्छी तरह explain करती है।

Original Explorative Modeling work exploration को generative training का तीसरा scaling axis मानता है: data और parameters के अलावा, training compute को extra candidates explore करने पर खर्च किया जा सकता है। SimpleTuner में XM supported image, video, audio, और autoregressive model families के लिए experimental training objective है।

Inference नहीं बदलता। XM सिर्फ training batch बनाने, score करने, और loss reduce करने का तरीका बदलता है।

## ELI5

सोचिए किसी student को target image बनानी है, लेकिन final grade से पहले उसे चार rough attempts करने की अनुमति है। चारों attempts का average लेने के बजाय, आप target के सबसे पास वाले attempt को grade करते हैं और उसी से सिखाते हैं।

XM का core idea:

1. एक ही sample के लिए कई candidates बनाओ।
2. सभी candidates पर model चलाओ।
3. हर candidate को real target से compare करो।
4. Sample या token block के लिए best candidate चुनो।
5. सिर्फ selected loss पर backprop करो।

जब target कई valid तरीकों से explain हो सकता है, single forced path model को possibilities average करना सिखा सकता है। Multiple explored paths model को एक plausible mode चुनने देते हैं।

## XM क्या बदलता है

XM नया inference sampler, नया checkpoint format, या दूसरा teacher model add नहीं करता। यह training-time selection बदलता है:

- Standard training एक candidate sample करता है और उसी से सीखता है।
- XM `K` candidates sample करता है और lowest-loss candidate से सीखता है।
- बड़ा `K` ज्यादा exploration देता है, लेकिन training compute भी बढ़ाता है।

Diffusion और flow models में candidate आम तौर पर वह noise होता है जिससे selected timestep पर noised latent बनता है।

Autoregressive token models, जैसे RVQ/audio planners, में candidate एक learned route embedding होता है जो model को same supervised token sequence के लिए कई internal paths देता है।

## SimpleTuner में व्यवहार

### Diffusion और Flow Models

Supported diffusion या flow matching families के लिए `xm_training_target=noise` उपयोग करें।

SimpleTuner:

1. Normal training timestep या sigma sample करता है।
2. Batch को `xm_candidate_count` बार repeat करता है।
3. हर candidate के लिए अलग noise tensor बनाता है।
4. हर candidate noise से noised latents बनाता है।
5. Expanded candidate batch पर model चलाता है।
6. हर candidate की normal training loss compute करता है।
7. Original sample के लिए lowest-loss candidate चुनता है।
8. Selected loss पर backprop करता है।

Model वही normal prediction type सीखता है: family के अनुसार flow velocity, epsilon, v-prediction, या sample prediction।

### Autoregressive और RVQ Models

Supported autoregressive planners के लिए `xm_training_target=route` उपयोग करें।

SimpleTuner:

1. Candidate routes के लिए छोटी learned route embedding table जोड़ता है।
2. हर supervised token sequence को route candidates पर repeat करता है।
3. Route signal को model input में insert करता है।
4. हर route की token losses compute करता है।
5. Whole sample या configured token blocks के लिए best route चुनता है।
6. सिर्फ selected route loss पर backprop करता है।

यह global LM style planners के लिए उपयोगी है जो RVQ audio codes या दूसरे discrete token streams predict करते हैं। Route embedding inference-time decoding बदले बिना same target sequence के लिए कई internal explanations देता है।

## Pseudocode

```text
for each batch:
    candidates = []

    for candidate_id in 1..K:
        candidate_input = make_candidate(batch, candidate_id)
        prediction = model(candidate_input)
        loss = compare(prediction, target)
        candidates.append(loss)

    selected_loss = minimum_loss_per_sample_or_block(candidates)
    train_on(selected_loss)
```

Diffusion के लिए:

```text
candidate_input = add_noise(clean_latent, random_noise_candidate, timestep)
loss = diffusion_or_flow_loss(model(candidate_input), training_target)
```

Autoregressive route selection के लिए:

```text
candidate_input = add_route_embedding(token_sequence, route_candidate)
loss = token_loss(model(candidate_input), target_tokens)
```

## Quick Setup

### WebUI

1. **Training → Loss functions** खोलें।
2. **XM** enable करें।
3. **XM Candidates** को `2` या `4` रखें।
4. **XM Training Target** चुनें:
   - diffusion या flow models के लिए `noise`।
   - autoregressive/RVQ planners के लिए `route`।
5. Model guide recommendation न हो तो **XM Selection Scope** को `sample` रखें।
6. Route block selection उपयोग न कर रहे हों तो **XM Block Size** को `0` रखें।

### Config JSON / CLI

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "noise",
  "xm_selection_scope": "sample",
  "xm_block_size": 0
}
```

AR/RVQ route training:

```json
{
  "xm_enabled": true,
  "xm_candidate_count": 4,
  "xm_training_target": "route",
  "xm_selection_scope": "block",
  "xm_block_size": 16
}
```

## Settings

- `xm_enabled`: XM enable करता है।
- `xm_candidate_count`: हर sample के candidates; XM enabled हो तो कम से कम `2`।
- `xm_training_target`: candidate type। Diffusion/flow के लिए `noise`, token planners के लिए `route`।
- `xm_selection_scope`: winner selection granularity। `sample` whole sample के लिए winner चुनता है; `block` supported families में token/frame blocks के लिए।
- `xm_block_size`: block-level selection का token या frame span। `0` का मतलब full supervised sequence।

## Values कैसे चुनें

| स्थिति | Suggested start |
| --- | --- |
| Image या video diffusion LoRA | `xm_candidate_count=2`, `xm_training_target=noise`, `xm_selection_scope=sample` |
| Ambiguous dataset या बड़ा batch | `xm_candidate_count=4` try करें |
| RVQ/audio planner | `xm_training_target=route`, `xm_selection_scope=block`, model guide का block size |
| New family पर first run | block size `0` रखें और non-XM baseline से validation compare करें |

Candidate count बढ़ने पर cost आम तौर पर लगभग linearly बढ़ता है।

## Logs

XM active हो तो logs में ये आ सकते हैं:

- `xm_loss`: selected loss।
- `xm_candidate_loss_mean`: selection से पहले candidate losses का average।
- `xm_candidate_0_wins`, `xm_candidate_1_wins`: हर candidate कितनी बार जीता।
- `xm_route_usage`: AR/RVQ route usage।

अच्छे संकेत: कई candidates कभी-कभी जीतते हैं, validation improve होता है, और route usage लंबे समय तक collapse नहीं होता।

चिंता के संकेत: शुरुआत से एक ही candidate हमेशा जीतता है, training loss गिरता है लेकिन validation खराब होता है, या memory/step time cost बहुत बढ़ जाता है।

## Compatibility

Current family-level support के लिए [Quick Start](../QUICKSTART.hi.md) की feature table देखें।

General rules:

- Diffusion/flow XM noise candidates और sample-level selection उपयोग करता है।
- AR/RVQ XM route candidates उपयोग करता है और block-level selection support कर सकता है।
- Unsupported families option को silently ignore नहीं करतीं; explicit error देती हैं।

Diffusion noise-candidate XM के लिए, जब तक model family अलग से support न बताए, SimpleTuner TwinFlow, Scheduled Sampling, `input_perturbation`, CREPA self-flow, और stochastic segmentation masked loss को incompatible मानता है।

## Other Features से संबंध

- **MixFlow** flow models की training trajectory बदलता है; XM candidate selection बदलता है।
- **Diff2Flow** legacy diffusion models का target बदलता है।
- **NextLat** hidden-state dynamics regularize करता है; XM routes या noises चुनता है।
- **LayerSync और CREPA** representations align करते हैं; XM सबसे explanatory candidate चुनता है।

## Practical Advice

- Baseline से compare करते समय fixed validation seeds रखें।
- `xm_candidate_count` से VRAM pressure हो तो batch size घटाएँ।
- XM को सिर्फ training loss से judge न करें; validation और sample diversity देखें।
- AR/RVQ में guide recommendation न हो तो block size `1` avoid करें।
- पहले short ablation run करें: same model, dataset, seed, सिर्फ XM on/off।

## References

- [Explorative Modeling project page](https://explorative-modeling.github.io/)
- [Explorative Modeling paper](https://arxiv.org/abs/2607.27372)
