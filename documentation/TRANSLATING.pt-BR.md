# Traduzindo a documentacao do SimpleTuner

Obrigado por ajudar a traduzir a documentacao do SimpleTuner! Este guia explica como contribuir com traducoes.

## Idiomas suportados

| Idioma | Locale | Status |
|----------|--------|--------|
| Ingles | `en` | Completo (fonte) |
| Chines (Simplificado) | `zh` | Aceitando contribuicoes |
| Japones | `ja` | Aceitando contribuicoes |
| Coreano | `ko` | Aceitando contribuicoes |

## Como a traducao funciona

O SimpleTuner usa **i18n baseado em sufixo** com [mkdocs-static-i18n](https://github.com/ultrabug/mkdocs-static-i18n). As traducoes sao armazenadas ao lado dos arquivos originais com um sufixo de idioma:

```
documentation/
├── index.md          # Ingles (padrao)
├── index.zh.md       # Traducao chinesa
├── index.ja.md       # Traducao japonesa
├── INSTALL.md        # Ingles
├── INSTALL.zh.md     # Traducao chinesa
└── ...
```

## Contribuindo com uma traducao

### 1. Encontre uma pagina para traduzir

Comece com paginas de alto impacto:

- `index.md` - Pagina inicial
- `INSTALL.md` - Guia de instalacao
- `QUICKSTART.md` - Guia de inicio rapido
- `quickstart/FLUX.md` - Guia de modelo popular

### 2. Crie o arquivo de traducao

Copie o arquivo em ingles e adicione o sufixo de idioma:

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. Traduza o conteudo

- Mantenha toda a formatacao Markdown intacta
- Mantenha blocos de codigo em ingles (comandos, exemplos de configuracao)
- Traduza prosa, titulos e descricoes
- Mantenha termos tecnicos consistentes (veja o glossario abaixo)

### 4. Envie um pull request

- Um arquivo por PR esta ok
- Titulo: `docs(i18n): Add Chinese translation for INSTALL.md`
- Inclua o idioma na descricao do PR

## Diretrizes de traducao

### O que traduzir

- Titulos e prosa
- Rotulos e descricoes de UI
- Texto alternativo de imagens
- Titulos de admonitions (Note, Warning, etc.)

### O que manter em ingles

- Blocos de codigo e comandos
- Caminhos de arquivo e nomes de arquivo
- Chaves e valores de configuracao
- Endpoints e parametros de API
- Identificadores tecnicos (nomes de modelos, etc.)

### Glossario

Mantenha estes termos consistentes nas traducoes:

| Ingles | 中文 | 日本語 | 한국어 |
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

## Construindo localmente

Teste suas traducoes localmente:

```bash
# Instalar dependencias
pip install mkdocs mkdocs-material mkdocs-static-i18n pymdown-extensions

# Servir com hot reload
mkdocs serve

# Abra http://localhost:8000 e alterne idiomas
```

## Duvidas?

- Abra uma issue com as labels `documentation` e `i18n`
- Entre no nosso [Discord](https://discord.gg/JGkSwEbjRb) para discussao
