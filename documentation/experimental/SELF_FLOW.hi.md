# Self-Flow (internal alignment)

Self-Flow, CREPA का एक mode है जो external vision encoder की जगह उसी model का cleaner EMA teacher view उपयोग करता है। यह Black Forest Labs paper की core idea के काफी करीब है: student को mixed tokenwise noise schedule पर train करना, EMA teacher को cleaner view पर चलाना, और normal generative loss को बनाए रखते हुए internal hidden states align करना।

> **अगर external encoder alignment चाहिए**: REPA / U-REPA के लिए [IMAGE_REPA.hi.md](IMAGE_REPA.hi.md) और temporal CREPA के लिए [VIDEO_CREPA.hi.md](VIDEO_CREPA.hi.md) देखें।

## कब उपयोग करें

- आप external encoder की जगह BFL-style self-supervised regularizer चाहते हैं।
- आप ऐसी transformer family train कर रहे हैं जिसमें SimpleTuner में Self-Flow hooks मौजूद हैं।
- आप चाहते हैं कि वही regularizer normal generation, editing, और multimodal training में मदद करे।
- आप पहले से EMA उपयोग करते हैं या इसे enable कर सकते हैं। Self-Flow के लिए EMA teacher जरूरी है।

अभी supported families:

- Image / edit: `flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- Video / multimodal: `wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## Quick setup (WebUI)

1. **Training → Loss functions** खोलें।
2. **CREPA** enable करें।
3. **CREPA Feature Source** को `self_flow` पर सेट करें।
4. **CREPA Block Index** को earlier student block रखें। 24-layer DiT पर `8` से शुरू करें, deeper stacks पर `10` से।
5. **CREPA Teacher Block Index** को deeper teacher block रखें। `16` या `20` अच्छे starting points हैं।
6. **Weight** को `0.5` पर रखें।
7. **Self-Flow Mask Ratio** के लिए:
   - image: `0.25`
   - video: `0.10`
   - audio-heavy models जैसे `ace_step`: `0.50`
8. **EMA** enable रखें।
9. TwinFlow के साथ इसे combine न करें।

## Quick setup (config JSON / CLI)

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

Legacy alias `crepa_self_flow=true` अभी भी काम करता है, लेकिन नए configs के लिए `crepa_feature_source=self_flow` बेहतर है।

## महत्वपूर्ण knobs

- `crepa_block_index`: student block
- `crepa_teacher_block_index`: EMA teacher block। जरूरी
- `crepa_lambda`: alignment strength। `0.5` से शुरू करें
- `crepa_self_flow_mask_ratio`: alternate timestep पाने वाले tokens का fraction। यह `[0.0, 0.5]` में होना चाहिए
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`: CREPA जैसे ही scheduling controls
- `crepa_use_backbone_features`: यह अलग mode है। इसे Self-Flow के साथ mix न करें

## Sampling / validation

Self-Flow training को बदलता है, basic inference algorithm को नहीं।

- Training में student mixed tokenwise noise देखता है और teacher cleaner EMA view देखता है।
- Validation loss अब भी requested homogeneous timestep schedule को evaluate करती है।
- Normal sampling unchanged रहता है। Inference में dual-timestep masking नहीं चलता।

<details>
<summary>कैसे काम करता है (practitioner)</summary>

- दो timesteps sample किए जाते हैं और random mask के जरिए tokens पर assign किए जाते हैं।
- Student के लिए mixed corruption वाला view और teacher के लिए cleaner timestep वाला view बनाया जाता है।
- Student normal forward करता है, EMA teacher `no_grad` में चलता है।
- Earlier student layer को deeper teacher layer से cosine similarity द्वारा align किया जाता है, जबकि normal generative loss भी train होती रहती है।

</details>

<details>
<summary>Technical (SimpleTuner internals)</summary>

- Mode selection `simpletuner/helpers/training/crepa.py` में `CrepaFeatureSource.SELF_FLOW` के रूप में है
- Shared batch builders `_prepare_image_crepa_self_flow_batch` और `_prepare_video_crepa_self_flow_batch` में हैं
- EMA teacher forward `auxiliary_loss` से `_run_crepa_teacher_forward` के जरिए चलता है
- Validation, `custom_timesteps` मांगने पर homogeneous eval batch rebuild करती है ताकि mixed Self-Flow training batch eval loss को प्रभावित न करे

</details>

## Common pitfalls

- **EMA disabled**: Self-Flow के लिए `use_ema=true` जरूरी है
- **Teacher block unset**: `crepa_teacher_block_index` सेट करें
- **TwinFlow enabled**: दोनों साथ supported नहीं हैं
- **Unsupported family**: केवल वही families काम करेंगी जो `supports_crepa_self_flow()` implement करती हैं
- **Mask ratio बहुत ऊंचा**: `0.5` या उससे कम रखें
- **Special sampler की उम्मीद**: inference normal ही रहता है

## References

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)
