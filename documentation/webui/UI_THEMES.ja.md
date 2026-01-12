# SimpleTuner UI テーマ

## はじめに

SimpleTuner WebUI はカスタムテーマをサポートしており、色の変更、背景画像の追加、カスタムサウンドの追加が可能です。テーマは pip パッケージ経由でインストールするか、ローカルフォルダに配置できます。

## 組み込みテーマ

SimpleTuner には2つの組み込みテーマが含まれています：

- **Dark** - 紫のアクセントを持つデフォルトのダークテーマ
- **Tron** - 実験的なネオンシアンテーマ

## テーマのインストール

### 方法 1：Pip パッケージ（推奨）

pip で直接テーマパッケージをインストール：

```bash
pip install simpletuner-theme-yourtheme
```

pip でインストールされたテーマは Python エントリーポイント経由で自動的に検出されます。

### 方法 2：ローカルテーマフォルダ

テーマを `~/.simpletuner/themes/yourtheme/` に配置：

```
~/.simpletuner/themes/yourtheme/
├── theme.json    # テーママニフェスト（必須）
├── theme.css     # CSS オーバーライド（必須）
└── assets/       # オプションの画像とサウンド
    ├── images/
    │   └── sidebar-bg.png
    └── sounds/
        └── success.wav
```

## テーマの切り替え

1. WebUI で **UI 設定** を開く
2. **テーマ** セレクターを見つける
3. **更新** をクリックして新しくインストールされたテーマを検出
4. 希望のテーマを選択

## カスタムテーマの作成

### テンプレートの使用

カスタムテーマを作成する最も簡単な方法は、公式テンプレートをフォークすることです：

```bash
git clone https://github.com/bghira/simpletuner-theme-template
cd simpletuner-theme-template
```

### テーマ構造

```
simpletuner-theme-template/
├── setup.py                 # entry_points を含むパッケージ設定
├── build_theme.py           # CSS 生成用ビルドスクリプト
└── src/
    └── simpletuner_theme_template/
        ├── __init__.py      # テーマクラス
        ├── tokens.yaml      # デザイントークン（これを編集！）
        ├── theme.css        # 生成された CSS
        └── assets/          # オプションのテーマアセット
            ├── images/
            └── sounds/
```

### デザイントークンの編集

`tokens.yaml` を編集して色をカスタマイズ：

```yaml
colors:
  primary:
    purple: "#your-primary-color"
  accent:
    blue: "#your-accent-color"
  dark:
    bg: "#your-background-color"
  text:
    primary: "#your-text-color"
```

その後 CSS を再ビルド：

```bash
python build_theme.py
```

### カスタムアセットの追加

#### 画像

画像を `assets/images/` に配置し、テーマクラスで宣言：

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

CSS で画像を参照：

```css
.sidebar {
    background-image: url('/api/themes/mytheme/assets/images/sidebar-bg');
}
```

#### サウンド

サウンドを `assets/sounds/` に配置して宣言：

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

サポートされる形式：
- **画像**：`.png`、`.jpg`、`.jpeg`、`.gif`、`.svg`、`.webp`、`.ico`
- **サウンド**：`.wav`、`.mp3`、`.ogg`、`.m4a`

### エントリーポイント登録

pip でインストール可能なテーマの場合、`setup.py` で登録：

```python
setup(
    name="simpletuner-theme-mytheme",
    # ...
    entry_points={
        "simpletuner.themes": [
            "mytheme = my_theme_package:MyTheme",
        ],
    },
)
```

テーマクラスには以下が必要です：

```python
class MyTheme:
    id = "mytheme"
    name = "My Theme"
    description = "A custom theme"
    author = "Your Name"

    @classmethod
    def get_css_path(cls) -> Path:
        return Path(__file__).parent / "theme.css"

    @classmethod
    def get_assets(cls) -> Dict:
        return {"images": {}, "sounds": {}}
```

### ローカルテーママニフェスト

ローカルテーマの場合、`theme.json` を作成：

```json
{
    "id": "mytheme",
    "name": "My Theme",
    "description": "A custom theme",
    "author": "Your Name",
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

## CSS 変数リファレンス

テーマは CSS カスタムプロパティをオーバーライドします。主要な変数：

### カラー

```css
:root {
    /* プライマリ */
    --colors-primary-purple: #667eea;
    --colors-primary-gradient: linear-gradient(...);

    /* 背景 */
    --colors-dark-bg: #0a0a0a;
    --colors-dark-sidebar: #0f0f0f;
    --colors-card-bg: rgba(255, 255, 255, 0.04);

    /* テキスト */
    --colors-text-primary: #ffffff;
    --colors-text-secondary: #94a3b8;

    /* セマンティック */
    --colors-semantic-success: #22c55e;
    --colors-semantic-error: #ef4444;
}
```

完全なリストは SimpleTuner ソースの `static/css/tokens.css` を参照してください。

## セキュリティ

テーマアセットはセキュリティのために検証されます：

- アセット名は英数字（ハイフン/アンダースコア可）である必要があります
- パストラバーサル試行（`../`）はブロックされます
- ホワイトリストのファイル拡張子のみ許可されます
- アセットはテーママニフェストで宣言する必要があります

## トラブルシューティング

### テーマが表示されない

1. テーマセレクターで **更新** をクリック
2. テーマパッケージがインストールされているか確認：`pip list | grep theme`
3. ローカルテーマの場合、`theme.json` が有効な JSON か確認

### CSS が適用されない

1. ブラウザコンソールで 404 エラーを確認
2. `theme.css` が存在し、有効な CSS であることを確認
3. CSS 変数が `tokens.css` の正しい名前を使用していることを確認

### アセットが読み込まれない

1. アセットが `get_assets()` または `theme.json` で宣言されているか確認
2. ファイル拡張子が許可リストにあるか確認
3. アセットファイルが宣言されたパスに存在することを確認
