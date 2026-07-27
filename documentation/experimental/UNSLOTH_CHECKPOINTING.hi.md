# Unsloth-Style Checkpointing

छोटा जवाब: इसे तब इस्तेमाल करें जब job बस थोड़ा सा fit नहीं हो रहा, और model support करे तो FFN-only पहले try करें।

`unsloth` backend saved activation tensors को CPU पर offload करता है। `torch` backend उन्हें drop करके backward में recompute करता है। Unsloth आखिरी कुछ GiB बचा सकता है ताकि batch, resolution, या frames बढ़ सकें। यह free speed नहीं है। अगर run `torch` backend से fit हो रहा है, तो आम तौर पर `torch` बेहतर default है।

## Controls

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` के चार useful values हैं:

| Value | Scope | Path | कब use करें |
| --- | --- | --- | --- |
| `torch` | whole block | recompute | CPU offload से पहले सबसे बड़ा built-in memory cut चाहिए। |
| `torch-ffn` | feed-forward | recompute | Flash Attention attention memory संभाल चुका है और cheap win चाहिए। |
| `unsloth` | whole block | CPU offload | torch layer checkpointing अभी भी fit नहीं हो रहा। |
| `unsloth-ffn` | feed-forward | CPU offload | torch FFN-only almost fit है और CPU offload आखिरी gap भर सकता है। |

Supported model families में आप कम blocks भी checkpoint कर सकते हैं:

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` हर दूसरे supported block को checkpoint करता है। Higher values कम checkpointing करती हैं और VRAM में ज्यादा activations रखती हैं।

`torch-ffn` और `unsloth-ffn` अभी Flux.1-style blocks और MageFlow support करते हैं। बाकी model families साफ error देंगी जब तक उनके blocks वही safe boundary expose नहीं करते।

## Tradeoff

- `torch`: intermediate activations drop करता है और backward में recompute करता है।
- `unsloth`: कुछ tensors CPU पर रखता है और backward के लिए वापस GPU पर copy करता है।
- `*-ffn`: साफ FFN boundary वाले models पर केवल feed-forward side checkpoint करता है।
- Flash Attention पहले से बड़ी attention matrix materialize नहीं करता। "free checkpointing" वाली बात mostly attention के लिए है, पूरे transformer block के लिए नहीं।
- CPU offload तब सबसे अच्छा है जब activations बड़े हों और peak memory parameters या optimizer से dominate न हो।

CUDA और पर्याप्त CPU RAM चाहिए। PCIe bandwidth मायने रखती है। अगर CPU-GPU copies hide नहीं होतीं, step slow हो जाता है।

## हमारा Sweep

Synthetic transformer block, bf16, flash SDPA, frozen base weights, batch 1। ये model guarantees नहीं हैं; ये tradeoff दिखाते हैं।

### Packed Image Latents

2x2 packing में `64x64`, `128x128`, और `256x256` latents `1024`, `4096`, और `16384` transformer tokens बनते हैं।

| GPU | Tokens | No checkpoint | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

`1024` tokens पर extra offload लगभग noise है, जब तक आप पहले से VRAM wall पर न हों। `16384` tokens पर `torch-ffn` cheap step है और whole-layer checkpointing बड़ा fit lever है। `unsloth` torch layer checkpointing से करीब `1.3 GiB` और बचाता है।

### बड़ा Transformer Shape

Frozen `32` layers, width `4096`, `3072` tokens:

| GPU | No checkpoint | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

Full trainable weights ने picture बदल दी: gradients और optimizer state peak dominate कर रहे थे, इसलिए उस synthetic run में `unsloth` ने `torch` से ज्यादा memory नहीं बचाई। PEFT frozen-weight case के ज्यादा करीब है।

## Practical Rule

1. अगर job checkpointing के बिना fit होता है, इसे off रखें।
2. अगर fit नहीं होता, `gradient_checkpointing_backend: torch-ffn` try करें।
3. अगर अभी भी tight है, `torch` try करें।
4. अगर torch layer checkpointing भी fit नहीं होता, `unsloth-ffn`, फिर `unsloth` try करें।
5. अगर model `gradient_checkpointing_interval` support करता है, run fit होने के बाद speed वापस पाने के लिए `2` या higher try करें।

यह तभी worth it है जब इससे आपका desired batch, resolution, frames, या rank fit हो। छोटे token counts या ऐसे jobs में फायदा कम है जहाँ peak trainable weights, gradients, optimizer, VAE cache, या validation से आता है।

## Notes

- FSDP activation checkpointing enabled होने पर SimpleTuner model-level gradient checkpointing disable करता है ताकि systems conflict न करें।
- `torch-ffn` और `unsloth-ffn` के लिए model support चाहिए। SimpleTuner silently दूसरा scope नहीं चलाता; यह साफ error देता है।
- `gradient_checkpointing_interval: 1` normal every-block checkpointing जैसा है।
- कुछ model families interval checkpointing expose नहीं करतीं। SimpleTuner warning देकर interval ignore करता है।
- हमारे sweep में `torch.compile` offload path को rescue नहीं कर पाया।
