# SimpleTuner दस्तावेज़ों का अनुवाद

SimpleTuner दस्तावेज़ों के अनुवाद में मदद करने के लिए धन्यवाद! यह गाइड बताता है कि अनुवाद में कैसे योगदान दें।

## समर्थित भाषाएँ

| भाषा | लोकेल | स्थिति |
|----------|--------|--------|
| अंग्रेज़ी | `en` | पूर्ण (स्रोत) |
| चीनी (सरलीकृत) | `zh` | योगदान स्वीकार किए जा रहे हैं |
| जापानी | `ja` | योगदान स्वीकार किए जा रहे हैं |
| कोरियाई | `ko` | योगदान स्वीकार किए जा रहे हैं |

## अनुवाद कैसे काम करता है

SimpleTuner [mkdocs-static-i18n](https://github.com/ultrabug/mkdocs-static-i18n) के साथ **suffix-based i18n** का उपयोग करता है। अनुवाद मूल फ़ाइलों के साथ भाषा suffix के रूप में संग्रहीत होते हैं:

```
documentation/
├── index.md          # English (default)
├── index.zh.md       # Chinese translation
├── index.ja.md       # Japanese translation
├── INSTALL.md        # English
├── INSTALL.zh.md     # Chinese translation
└── ...
```

## अनुवाद में योगदान

### 1. अनुवाद के लिए एक पेज चुनें

उच्च‑प्रभाव वाले पेजों से शुरू करें:

- `index.md` - होम पेज
- `INSTALL.md` - इंस्टॉलेशन गाइड
- `QUICKSTART.md` - क्विक स्टार्ट गाइड
- `quickstart/FLUX.md` - लोकप्रिय मॉडल गाइड

### 2. अनुवाद फ़ाइल बनाएँ

English फ़ाइल कॉपी करें और भाषा suffix जोड़ें:

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. सामग्री का अनुवाद करें

- सभी Markdown formatting को बनाए रखें
- code blocks को English में रखें (commands, config examples)
- prose, headings, और descriptions का अनुवाद करें
- तकनीकी शब्दों को एक‑सा रखें (नीचे glossary देखें)

### 4. Pull request सबमिट करें

- एक फ़ाइल प्रति PR ठीक है
- Title: `docs(i18n): Add Chinese translation for INSTALL.md`
- PR विवरण में भाषा शामिल करें

## अनुवाद दिशानिर्देश

### क्या अनुवाद करना है

- headings और prose
- UI labels और descriptions
- images के लिए alt text
- admonition titles (Note, Warning, आदि)

### क्या English में रखें

- code blocks और commands
- फ़ाइल पाथ और फ़ाइल नाम
- configuration keys और values
- API endpoints और parameters
- तकनीकी identifiers (model names, आदि)

### शब्दावली

इन शब्दों को अनुवादों में एक‑सा रखें:

| अंग्रेज़ी | 中文 | 日本語 | 한국어 |
|---------|------|--------|--------|
| Training | 训练 | トレーニング | 훈련 |
| Fine-tuning | 微调 | ファインチューニング | 파인튜닝 |
| Model | 模型 | モデル | 모델 |
| Dataset | 数据集 | データセット | 데이터셋 |
| Checkpoint | 检查点 | チェックポイント | 체크포인트 |
| LoRA | LoRA | LoRA | LoRA |
| Batch size | 批次大小 | バッチサイズ | 배치 크기 |
| Learning rate | 学习率 | 学習率 | 학습률 |
| Epoch | 轮次 | エポック | 에포크 |
| Validation | 验证 | 検証 | 검증 |
| Inference | 推理 | 推論 | 추론 |

## लोकल में बिल्ड करना

अपने अनुवादों का लोकल टेस्ट करें:

```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocs-static-i18n pymdown-extensions

# Serve with hot reload
mkdocs serve

# Open http://localhost:8000 and switch languages
```

## प्रश्न?

- `documentation` और `i18n` लेबल्स के साथ एक issue खोलें
- चर्चा के लिए हमारे [Discord](https://discord.gg/JGkSwEbjRb) से जुड़ें
