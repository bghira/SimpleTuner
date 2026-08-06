# MiniMax H3 Drift Distillation

MiniMax H3 एक distilled flow-matching video/audio model है। सामान्य LoRA या LyCORIS training में adapter dataset target सीखता है, लेकिन base checkpoint का distilled behavior drift कर सकता है: guidance behavior, modality balance, और packed video/audio sequence layout बदल सकते हैं।

`h3_drift` adapter prediction को उसी model की frozen-base prediction से compare करता है जब adapter disabled होता है। यह अलग teacher checkpoint load नहीं करता और distillation cache इस्तेमाल नहीं करता। हर batch में SimpleTuner:

1. adapter enabled path से normal MiniMax H3 SFT loss निकालता है;
2. adapter अस्थायी रूप से disable करता है;
3. उसी prepared batch को `torch.no_grad()` में frozen base से चलाता है;
4. video/audio predictions के बीच MSE निकालता है;
5. adapter फिर enable करके combined loss backpropagate करता है।

```text
total = sft_loss_weight * normal_h3_loss + loss_weight * frozen_base_prediction_mse
```

## कब इस्तेमाल करें

MiniMax H3 LoRA या LyCORIS training में इसे default रखें, जब तक आपका लक्ष्य original distillation को हटाना न हो। यह style/concept LoRAs, FL2VA/Ref2VA, joint audio/video training, और `convrot-int8` / `convrot-int4` जैसे quantized flavours के लिए उपयोगी है।

Full-rank H3 drift supported नहीं है। जब पूरा transformer train हो रहा हो, frozen base comparison path भरोसेमंद नहीं रहता।

## Quick Config

```json
{
  "model_family": "minimaxh3",
  "model_flavour": "convrot-int8",
  "model_type": "lora",
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "balance": "token",
      "video_weight": 1.0,
      "audio_weight": 1.0
    }
  }
}
```

Checked-in H3 examples इसे default रूप से enable करते हैं। `loss_weight: 0.5` normal dataset target को primary रखता है, लेकिन base-reference drift को भी meaningful बनाता है।

## Config Keys

- `loss_weight`: frozen-base prediction loss का multiplier। Narrow LoRA के लिए `0.25` से `0.5` शुरू करें; base behavior टूटे तो `1.0` करें।
- `sft_loss_weight`: normal MiniMax H3 training loss का multiplier। सामान्य fine-tuning में `1.0` रखें।
- `balance`: `token` valid elements से average करता है; `modality` video और audio means को modality level पर balance करता है।
- `video_weight`: video drift term का multiplier।
- `audio_weight`: audio drift term का multiplier।
- `inner_distillation_method`: optional distiller जो `h3_drift` के अंदर चलेगा, जैसे `anyflow`, `dmd`, `perflow`, `flow_dpo`, या `self_forcing`।
- `inner_distillation_config`: inner distiller को दी जाने वाली config।

## दूसरा Distiller Compose करना

`h3_drift` किसी और distiller को wrap कर सकता है, ताकि step distillation या preference objective के साथ MiniMax H3 का frozen-base behavior भी preserve रहे।

```json
{
  "distillation_method": "h3_drift",
  "distillation_config": {
    "h3_drift": {
      "loss_weight": 0.5,
      "sft_loss_weight": 1.0,
      "inner_distillation_method": "anyflow",
      "inner_distillation_config": {
        "target_mode": "linear",
        "r_timestep_sampler": "zero",
        "loss_weight": 1.0
      }
    }
  }
}
```

Wrapper batch preparation, validation scheduler, distillation cache, caption batches, और generator/discriminator lifecycle hooks inner distiller को delegate करता है। Inner distiller की compatibility checks फिर भी लागू रहती हैं।

## Video और Audio Modes

`minimax_h3_target_mode: "auto"` video-only बनता है। `"video"` audio target rows बंद रखता है। `"av"` joint audio/video rows train करता है। इसे global config में या data backend में `h3_target_mode` / `minimax_h3_target_mode` से set कर सकते हैं।

Distiller prepared batch follow करता है: video-only batch में सिर्फ `model_prediction`, `av` batch में video और `audio_prediction`, `audio_latent_mask`, `sample_weight`, और visual masks सभी लागू होते हैं।

## CFG Distillation बनाए रखना

MiniMax H3 CFG-distilled है। Base checkpoint आम तौर पर `validation_guidance: 1.0`, `validation_guidance_real: 1.0`, और `validation_disable_unconditional: true` के साथ validate होता है। Negative prompting base contract का हिस्सा नहीं है।

SimpleTuner real CFG और negative prompt encode कर सकता है क्योंकि community H3 को de-distill कर सकती है। `h3_drift` उल्टा pressure देता है: adapter को base conditional prediction के पास रखता है। अगर आपका लक्ष्य negative prompt behavior सिखाना या de-distillation है, तो `loss_weight` घटाएं या distiller disable करें।

## Logs और Cost

मुख्य logs: `h3_drift_loss`, `h3_drift_video_loss`, `h3_drift_audio_loss`, element counts, `h3_drift_weighted_loss`, `h3_drift_sft_loss`, inner distiller enabled होने पर `h3_drift_inner_total`, और `total`।

हर step में एक extra forward pass लगता है, लेकिन दूसरा transformer memory में नहीं रखा जाता। ConvRot, RamTorch, musubi block swap, gradient checkpointing, और attention offload के साथ यह compatible है; फिर भी presets benchmark करें क्योंकि extra forward fastest backend बदल सकता है।

## Troubleshooting

- Low-rank error: `model_type: "lora"` इस्तेमाल करें।
- Audio loss zero: batch video-only है, target mode `av` नहीं है, या `audio_latent_mask` सब exclude कर रहा है।
- Adapter concept कम सीख रहा है: `loss_weight` घटाएं, rank बढ़ाएं, या training लंबी करें।
- Audio drift कर रहा है: `balance: "modality"` या बड़ा `audio_weight` आजमाएं।
