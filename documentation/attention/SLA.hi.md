# SimpleTuner में Sparse–Linear Attention (SLA)

Sparse–Linear Attention (SLA) एक ही CUDA kernel के अंदर sparse FlashAttention और एक linear attention compensator को फ्यूज़ करता है। महत्वपूर्ण query/key blocks महँगे sparse path पर जाते हैं, जबकि marginal blocks lightweight linear attention और learnable projection का उपयोग करते हैं। इससे गुणवत्ता लगभग full attention जैसी रहती है और FLOPs काफी कम हो जाते हैं।

SimpleTuner SLA को नियमित `--attention_mechanism` फ्लैग के जरिए एक्सपोज़ करता है, इसलिए आप SLA के साथ मॉडल fine-tune कर सकते हैं और बाद में inference भी उसी kernel से चला सकते हैं।

## आवश्यकताएँ

1. रेफरेंस इम्प्लीमेंटेशन इंस्टॉल करें:

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. PyTorch का CUDA बिल्ड उपयोग करें (SLA kernels फिलहाल केवल CUDA पर हैं)।

## SLA सक्षम करना

- `--attention_mechanism=sla` पास करें (या configs में `attention_mechanism: "sla"` सेट करें)।
- कोई अतिरिक्त फ्लैग की आवश्यकता नहीं है; SimpleTuner PyTorch के SDPA entrypoint को wrap करके SLA इंजेक्ट करता है।
- SLA सेटिंग्स (top-k ratio, block sizes, feature map type, query/key feature maps को tie करना) `--sla_config` / `sla_config` के जरिए JSON/Python dict फॉर्म में बदलें। उदाहरण: `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`। डिफ़ॉल्ट्स: top 20 %, block size 64, tied feature maps।

## ट्रेनिंग व्यवहार

- SLA ट्रेन करने योग्य है। controller linear projection head (`proj_l`) को `float32` में रखता है, भले ही SLA का बाकी हिस्सा BF16/FP16 में चले, ताकि AMP/GradScaler स्थिर रहें।
- क्योंकि backbone को SLA के mixed sparse/linear व्यवहार की उम्मीद के अनुसार fine-tune किया जाता है, आपको inference के दौरान भी SLA का उपयोग जारी रखना चाहिए। ट्रेनिंग के बाद Diffusers SDPA/XFormers पर लौटने से गुणवत्ता घटेगी।
- चेकपॉइंट सेव के दौरान, SimpleTuner सामान्य accelerator state के साथ `sla_attention.pt` लिखता है। इस फ़ाइल में हर unique head dimension/dtype pair के लिए SLA projection वेट्स और संबंधित buffers होते हैं। इसे अपने चेकपॉइंट के साथ रखें; हटाने पर अगला resume/inference रन SLA के projection लेयर को फिर से initialize करेगा।

## इंफरेंस

- जब भी आप ट्रेनिंग फिर से शुरू करें या वैलिडेशन स्टेप्स चलाएँ, `--attention_mechanism=sla` सक्षम रखें ताकि चेकपॉइंट उसी SLA kernel का उपयोग करे जिस पर उसे fine-tune किया गया था।
- loader अपने आप `sla_attention.pt` को replay करता है यदि वह चेकपॉइंट डायरेक्टरी में मौजूद हो, इसलिए अतिरिक्त फ्लैग की जरूरत नहीं है।
- यदि आप जानबूझकर SLA-trained वेट्स को standard SDPA से तुलना करना चाहते हैं, तो गुणवत्ता में गिरावट की उम्मीद करें। SLA पेपर बताता है कि backbone को अनुकूल करने के लिए कुछ हजार tuning steps की जरूरत होती है, इसलिए SLA के बिना inference को unsupported समझें।

## ट्रबलशूटिंग और नोट्स

- **`sla_attention.pt` गायब:** इसका मतलब है कि चेकपॉइंट SLA state saving से पहले बनाया गया था या फ़ाइल हटा दी गई है। SLA सक्षम करके एक छोटा ट्रेनिंग सत्र (एक स्टेप भी) चलाएँ ताकि फ़ाइल फिर से बने।
- **AMP/GradScaler त्रुटियाँ:** सुनिश्चित करें कि आप SLA मॉड्यूल्स को मैन्युअली BF16/FP16 में कास्ट नहीं कर रहे हैं। SimpleTuner projection head को स्वतः FP32 में रखता है; अतिरिक्त कास्टिंग से ट्रेनिंग अस्थिर हो सकती है।
- **Hub अपलोड्स:** Hugging Face Hub (या किसी artifact store) पर चेकपॉइंट पुश करते समय `sla_attention.pt` शामिल करें। इससे डाउनलोड करने वाले उपयोगकर्ताओं को trained SLA weights बिना अतिरिक्त चरणों के मिलेंगे।

SLA के डिज़ाइन और पूरे एल्गोरिथ्म की पृष्ठभूमि के लिए देखें: [SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://www.arxiv.org/abs/2509.24006)।
