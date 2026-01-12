# Traducir la documentación de SimpleTuner

¡Gracias por ayudar a traducir la documentación de SimpleTuner! Esta guía explica cómo contribuir con traducciones.

## Idiomas soportados

| Idioma | Locale | Estado |
|----------|--------|--------|
| English | `en` | Completo (fuente) |
| Chinese (Simplified) | `zh` | Se aceptan contribuciones |
| Japanese | `ja` | Se aceptan contribuciones |
| Korean | `ko` | Se aceptan contribuciones |

## Cómo funciona la traducción

SimpleTuner usa **i18n basado en sufijos** con [mkdocs-static-i18n](https://github.com/ultrabug/mkdocs-static-i18n). Las traducciones se almacenan junto a los archivos originales con un sufijo de idioma:

```
documentation/
├── index.md          # English (default)
├── index.zh.md       # Chinese translation
├── index.ja.md       # Japanese translation
├── INSTALL.md        # English
├── INSTALL.zh.md     # Chinese translation
└── ...
```

## Contribuir con una traducción

### 1. Encuentra una página para traducir

Comienza con páginas de alto impacto:

- `index.md` - Página de inicio
- `INSTALL.md` - Guía de instalación
- `QUICKSTART.md` - Guía de inicio rápido
- `quickstart/FLUX.md` - Guía de modelo popular

### 2. Crea el archivo de traducción

Copia el archivo en inglés y agrega el sufijo de idioma:

```bash
cp documentation/INSTALL.md documentation/INSTALL.zh.md
```

### 3. Traduce el contenido

- Mantén todo el formato Markdown intacto
- Mantén bloques de código en inglés (comandos, ejemplos de configuración)
- Traduce prosa, encabezados y descripciones
- Mantén términos técnicos consistentes (ver glosario abajo)

### 4. Envía un pull request

- Un archivo por PR está bien
- Título: `docs(i18n): Add Chinese translation for INSTALL.md`
- Incluye el idioma en la descripción del PR

## Guías de traducción

### Qué traducir

- Encabezados y prosa
- Etiquetas y descripciones de UI
- Texto alternativo de imágenes
- Títulos de admoniciones (Note, Warning, etc.)

### Qué mantener en inglés

- Bloques de código y comandos
- Rutas de archivos y nombres de archivo
- Claves y valores de configuración
- Endpoints y parámetros de API
- Identificadores técnicos (nombres de modelos, etc.)

### Glosario

Mantén estos términos consistentes entre traducciones:

| Inglés | 中文 | 日本語 | 한국어 |
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

## Construir localmente

Prueba tus traducciones localmente:

```bash
# Instalar dependencias
pip install mkdocs mkdocs-material mkdocs-static-i18n pymdown-extensions

# Servir con hot reload
mkdocs serve

# Abrir http://localhost:8000 y cambiar idiomas
```

## ¿Preguntas?

- Abre un issue con las etiquetas `documentation` e `i18n`
- Únete a nuestro [Discord](https://discord.gg/JGkSwEbjRb) para discutir
