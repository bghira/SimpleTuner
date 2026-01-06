# SimpleTuner

**SimpleTuner** एक मल्टी‑मोडल diffusion मॉडल fine‑tuning टूलकिट है जो सरलता और समझने में आसानी पर केंद्रित है।

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __शुरू करें__

    ---

    SimpleTuner इंस्टॉल करें और कुछ मिनटों में अपना पहला मॉडल ट्रेन करें

    [:octicons-arrow-right-24: इंस्टॉलेशन](INSTALL.md)

-   :material-cog:{ .lg .middle } __वेब UI__

    ---

    एक स्लीक वेब इंटरफ़ेस के जरिए प्रशिक्षण कॉन्फ़िगर करें और चलाएँ

    [:octicons-arrow-right-24: वेब UI ट्यूटोरियल](webui/TUTORIAL.md)

-   :material-api:{ .lg .middle } __REST API__

    ---

    HTTP API के साथ प्रशिक्षण वर्कफ़्लो स्वचालित करें

    [:octicons-arrow-right-24: API ट्यूटोरियल](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __क्लाउड प्रशिक्षण__

    ---

    Replicate या वितरित वर्कर्स पर प्रशिक्षण चलाएँ

    [:octicons-arrow-right-24: क्लाउड प्रशिक्षण](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __मल्टी‑यूज़र__

    ---

    एंटरप्राइज फीचर्स: SSO, quotas, RBAC, worker orchestration

    [:octicons-arrow-right-24: एंटरप्राइज गाइड](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __मॉडल गाइड__

    ---

    Flux, SD3, SDXL, वीडियो मॉडल्स, और अधिक के लिए स्टेप‑बाय‑स्टेप गाइड

    [:octicons-arrow-right-24: मॉडल गाइड](quickstart/index.md)

</div>

## विशेषताएँ

- **मल्टी‑मोडल प्रशिक्षण** - छवि, वीडियो, और ऑडियो जनरेशन मॉडल्स
- **वेब UI और API** - ब्राउज़र से ट्रेन करें या REST से ऑटोमेट करें
- **वर्कर ऑर्केस्ट्रेशन** - GPU मशीनों पर जॉब्स वितरित करें
- **एंटरप्राइज‑रेडी** - LDAP/OIDC SSO, RBAC, quotas, audit logging
- **क्लाउड इंटीग्रेशन** - Replicate, self‑hosted workers
- **मेमोरी ऑप्टिमाइज़ेशन** - DeepSpeed, FSDP2, quantization

## समर्थित मॉडल

| प्रकार | मॉडल |
|------|--------|
| **छवि** | Flux.1/2, SD3, SDXL, Chroma, Auraflow, PixArt, Sana, Lumina2, HiDream, और अधिक |
| **वीडियो** | Wan, LTX Video, Hunyuan Video, Kandinsky 5, LongCat |
| **ऑडियो** | ACE-Step |

पूरी डॉक्यूमेंटेशन के लिए [मॉडल गाइड](quickstart/index.md) देखें।

## समुदाय

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - बग रिपोर्ट्स और फीचर अनुरोध

## लाइसेंस

SimpleTuner ओपन सोर्स सॉफ़्टवेयर है।
