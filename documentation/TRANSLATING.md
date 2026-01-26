# Translating SimpleTuner Documentation

Thank you for helping translate SimpleTuner documentation! This guide explains how to contribute translations.

## Supported Languages

| Language | Locale | Status |
|----------|--------|--------|
| English | `en` | Complete (source) |
| Chinese (Simplified) | `zh` | Accepting contributions |
| Japanese | `ja` | Accepting contributions |
| Korean | `ko` | Accepting contributions |

## How Translation Works

SimpleTuner uses **suffix-based i18n** with Zensical. Translations are stored alongside the original files with a language suffix:

```
documentation/
├── index.md          # English (default)
├── index.zh.md       # Chinese translation
├── index.ja.md       # Japanese translation
├── INSTALL.md        # English
├── INSTALL.zh.md     # Chinese translation
└── ...
```

## Contributing a Translation

### 1. Find a page to translate

Start with high-impact pages:

- `index.md` - Home page
- `INSTALL.md` - Installation guide
- `QUICKSTART.md` - Quick start guide
- `quickstart/FLUX.md` - Popular model guide

### 2. Create the translation file

Copy the English file and add the language suffix:

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. Translate the content

- Keep all Markdown formatting intact
- Keep code blocks in English (commands, config examples)
- Translate prose, headings, and descriptions
- Keep technical terms consistent (see glossary below)

### 4. Submit a pull request

- One file per PR is fine
- Title: `docs(i18n): Add Chinese translation for INSTALL.md`
- Include the language in the PR description

## Translation Guidelines

### What to translate

- Headings and prose
- UI labels and descriptions
- Alt text for images
- Admonition titles (Note, Warning, etc.)

### What to keep in English

- Code blocks and commands
- File paths and filenames
- Configuration keys and values
- API endpoints and parameters
- Technical identifiers (model names, etc.)

### Glossary

Keep these terms consistent across translations:

| English | 中文 | 日本語 | 한국어 |
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

## Building Locally

Test your translations locally:

```bash
# Install dependencies
pip install zensical

# Serve with hot reload
zensical serve

# Open http://localhost:8000 and switch languages
```

## Questions?

- Open an issue with the `documentation` and `i18n` labels
- Join our [Discord](https://discord.gg/JGkSwEbjRb) for discussion
