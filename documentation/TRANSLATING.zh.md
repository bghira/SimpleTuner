# 翻译 SimpleTuner 文档

感谢你帮助翻译 SimpleTuner 文档！本指南说明如何贡献翻译。

## 支持的语言

| Language | Locale | Status |
|----------|--------|--------|
| English | `en` | 已完成（源文档） |
| Chinese (Simplified) | `zh` | 接受贡献中 |
| Japanese | `ja` | 接受贡献中 |
| Korean | `ko` | 接受贡献中 |

## 翻译机制

SimpleTuner 使用 Zensical 的**后缀式 i18n**。翻译与原文件并排存放，通过语言后缀区分：

```
documentation/
├── index.md          # English (default)
├── index.zh.md       # Chinese translation
├── index.ja.md       # Japanese translation
├── INSTALL.md        # English
├── INSTALL.zh.md     # Chinese translation
└── ...
```

## 贡献翻译

### 1. 找到要翻译的页面

优先从高影响页面开始：

- `index.md` - 首页
- `INSTALL.md` - 安装指南
- `QUICKSTART.md` - 快速开始
- `quickstart/FLUX.md` - 热门模型指南

### 2. 创建翻译文件

复制英文文件并添加语言后缀：

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. 翻译内容

- 保持所有 Markdown 格式不变
- 代码块保持英文（命令、配置示例）
- 翻译正文、标题与描述
- 技术术语保持一致（参考下方术语表）

### 4. 提交 PR

- 一次 PR 只包含一个文件也可以
- 标题示例：`docs(i18n): Add Chinese translation for INSTALL.md`
- 在 PR 描述中注明语言

## 翻译指南

### 需要翻译的内容

- 标题与正文
- UI 标签与说明
- 图片的 alt 文本
- 提示框标题（Note、Warning 等）

### 需要保持英文的内容

- 代码块与命令
- 文件路径与文件名
- 配置键与取值
- API 端点与参数
- 技术标识符（模型名等）

### 术语表

请在翻译中统一以下术语：

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

## 本地构建

在本地测试你的翻译：

```bash
# Install dependencies
pip install zensical

# Serve with hot reload
zensical serve

# Open http://localhost:8000 and switch languages
```

## 有问题吗？

- 使用 `documentation` 与 `i18n` 标签创建 issue
- 加入 [Discord](https://discord.gg/JGkSwEbjRb) 参与讨论
