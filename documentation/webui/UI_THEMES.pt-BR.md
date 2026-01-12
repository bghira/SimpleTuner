# Temas de UI do SimpleTuner

## Introdução

A WebUI do SimpleTuner suporta temas personalizados que podem alterar cores, adicionar imagens de fundo e incluir sons personalizados. Os temas podem ser instalados via pacotes pip ou colocados em uma pasta local.

## Temas Integrados

O SimpleTuner inclui três temas integrados:

- **Dark** - O tema escuro padrão com acentos roxos
- **Tron** - Um tema experimental em ciano neon
- **Light** - Um tema bege inspirado no Windows 98

## Instalando Temas

### Método 1: Pacote Pip (Recomendado)

Instale pacotes de temas diretamente com pip:

```bash
pip install simpletuner-theme-seutema
```

Temas instalados via pip são automaticamente descobertos através de entry points do Python.

### Método 2: Pasta de Tema Local

Coloque os temas em `~/.simpletuner/themes/seutema/`:

```
~/.simpletuner/themes/seutema/
├── theme.json    # Manifesto do tema (obrigatório)
├── theme.css     # Substituições CSS (obrigatório)
└── assets/       # Imagens e sons opcionais
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## Trocando Temas

1. Abra **Configurações de UI** na WebUI
2. Encontre o seletor de **Tema**
3. Clique em **Atualizar** para descobrir temas recém-instalados
4. Selecione o tema desejado

## Criando Temas Personalizados

### Usando o Template

A maneira mais fácil de criar um tema personalizado é fazer fork do template oficial:

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### Estrutura do Tema

```
simpletuner-theme-template/
├── setup.py                 # Configuração do pacote com entry_points
├── build_theme.py           # Script de build para gerar CSS
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # Classe do tema
        ├── tokens.yaml      # Tokens de design (edite este!)
        ├── theme.css        # CSS gerado
        └── assets/          # Assets opcionais do tema
            ├── images/
            └── sounds/
```

### Editando Tokens de Design

Edite `tokens.yaml` para personalizar cores:

```yaml
colors:
  primary:
    purple: "#sua-cor-primaria"
  accent:
    blue: "#sua-cor-de-acento"
  dark:
    bg: "#sua-cor-de-fundo"
  text:
    primary: "#sua-cor-de-texto"
```

Depois reconstrua o CSS:

```bash
python build_theme.py
```

### Adicionando Assets Personalizados

#### Imagens

Coloque imagens em `assets/images/` e declare na sua classe de tema:

```python
@classmethod
def get_assets(cls) -> Dict:
    return {
        "images": {
            "sidebar-bg": "assets/images/sidebar-bg.png",
            "logo": "assets/images/logo.svg",
        },
        "sounds": {},
    }
```

Referencie imagens no seu CSS:

```css
.sidebar {
    background-image: url('/api/themes/meutema/assets/images/sidebar-bg');
}
```

#### Sons

Coloque sons em `assets/sounds/` e declare:

```python
@classmethod
def get_assets(cls) -> Dict:
    return {
        "images": {},
        "sounds": {
            "success": "assets/sounds/success.wav",
            "error": "assets/sounds/error.wav",
        },
    }
```

Formatos suportados:
- **Imagens**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.ico`
- **Sons**: `.wav`, `.mp3`, `.ogg`, `.m4a`

### Registro de Entry Point

Para temas instaláveis via pip, registre via `setup.py`:

```python
setup(
    name="simpletuner-theme-meutema",
    # ...
    entry_points={
        "simpletuner.themes": [
            "meutema = meu_pacote_tema:MeuTema",
        ],
    },
)
```

Sua classe de tema deve ter:

```python
class MeuTema:
    id = "meutema"
    name = "Meu Tema"
    description = "Um tema personalizado"
    author = "Seu Nome"

    @classmethod
    def get_css_path(cls) -> Path:
        return Path(__file__).parent / "theme.css"

    @classmethod
    def get_assets(cls) -> Dict:
        return {"images": {}, "sounds": {}}
```

### Manifesto de Tema Local

Para temas locais, crie `theme.json`:

```json
{
    "id": "meutema",
    "name": "Meu Tema",
    "description": "Um tema personalizado",
    "author": "Seu Nome",
    "version": "1.0.0",
    "assets": {
        "images": {
            "sidebar-bg": "assets/images/sidebar-bg.png"
        },
        "sounds": {
            "success": "assets/sounds/success.wav"
        }
    }
}
```

## Referência de Variáveis CSS

Temas sobrescrevem propriedades personalizadas CSS. Variáveis principais incluem:

### Cores

```css
:root {
    /* Primária */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* Fundos */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* Texto */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* Semântico */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

Veja `static/css/tokens.css` no código fonte do SimpleTuner para a lista completa.

## Segurança

Assets de tema são validados por segurança:

- Nomes de assets devem ser alfanuméricos (com hífens/underscores)
- Tentativas de path traversal (`../`) são bloqueadas
- Apenas extensões de arquivo na lista branca são permitidas
- Assets devem ser declarados no manifesto do tema

## Solução de Problemas

### Tema não aparece

1. Clique em **Atualizar** no seletor de temas
2. Verifique se seu pacote de tema está instalado: `pip list | grep theme`
3. Para temas locais, verifique se `theme.json` é um JSON válido

### CSS não está sendo aplicado

1. Verifique o console do navegador por erros 404
2. Verifique se `theme.css` existe e é CSS válido
3. Certifique-se de que as variáveis CSS usam os nomes corretos de `tokens.css`

### Assets não carregam

1. Verifique se os assets estão declarados em `get_assets()` ou `theme.json`
2. Verifique se as extensões de arquivo estão na lista permitida
3. Certifique-se de que os arquivos de assets existem nos caminhos declarados
