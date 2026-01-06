# SimpleTuner ドキュメントの翻訳

SimpleTuner ドキュメントの翻訳にご協力いただきありがとうございます。このガイドでは翻訳の貢献方法を説明します。

## 対応言語

| Language | Locale | Status |
|----------|--------|--------|
| English | `en` | 完了（ソース） |
| Chinese (Simplified) | `zh` | 貢献受付中 |
| Japanese | `ja` | 貢献受付中 |
| Korean | `ko` | 貢献受付中 |

## 翻訳の仕組み

SimpleTuner は [mkdocs-static-i18n](https://github.com/ultrabug/mkdocs-static-i18n) を使った**接尾辞ベースの i18n**を採用しています。翻訳は元ファイルと同じ場所に、言語サフィックス付きで保存します。

```
documentation/
├── index.md          # English (default)
├── index.zh.md       # Chinese translation
├── index.ja.md       # Japanese translation
├── INSTALL.md        # English
├── INSTALL.zh.md     # Chinese translation
└── ...
```

## 翻訳への貢献方法

### 1. 翻訳するページを探す

まずは影響度の高いページから:

- `index.md` - ホームページ
- `INSTALL.md` - インストールガイド
- `QUICKSTART.md` - クイックスタートガイド
- `quickstart/FLUX.md` - 人気モデルガイド

### 2. 翻訳ファイルを作成する

英語ファイルをコピーし、言語サフィックスを付けます:

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. 内容を翻訳する

- Markdown の書式はそのまま維持する
- コードブロックは英語のまま（コマンドや設定例）
- 文章、見出し、説明を翻訳する
- 技術用語は一貫性を保つ（下記の用語集を参照）

### 4. プルリクエストを送る

- 1 PR に 1 ファイルでも問題ありません
- タイトル例: `docs(i18n): Add Chinese translation for INSTALL.md`
- PR の説明に言語を記載する

## 翻訳ガイドライン

### 翻訳するもの

- 見出しと本文
- UI ラベルと説明
- 画像の alt テキスト
- 注記タイトル（Note, Warning など）

### 英語のままにするもの

- コードブロックとコマンド
- ファイルパスとファイル名
- 設定キーと値
- API エンドポイントとパラメータ
- 技術識別子（モデル名など）

### 用語集

翻訳で用語を統一してください:

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

## ローカルビルド

ローカルで翻訳をテストします:

```bash
# Install dependencies
pip install mkdocs mkdocs-material mkdocs-static-i18n pymdown-extensions

# Serve with hot reload
mkdocs serve

# Open http://localhost:8000 and switch languages
```

## 質問がある場合

- `documentation` と `i18n` ラベルで issue を開いてください
- 議論は [Discord](https://discord.gg/JGkSwEbjRb) に参加してください
