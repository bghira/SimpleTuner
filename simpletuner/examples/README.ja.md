# 学習サンプル

サンプルは `model_family/model_flavour/vram/recipe` の構成です。モデルファミリー、フレーバー、`24g`・`48g`・`80g` などの VRAM 枠の順で選びます。VRAM の `default` は特定の容量を指定せず、フレーバーの `default` はモデル設定の既定値を使います。`8x80g` は各 80 GB の GPU が 8 台必要です。

名前付きレシピで LoRA、LoKr、ControlNet、AnyFlow、条件入力の各設定を区別します。共有データセット設定、プロンプト集、検証スクリプトは `_shared/` にあります。

## サンプルの一覧と実行

SimpleTuner のインストール後、一覧を確認し、完全な相対パスを `example=` に指定します。

```bash
simpletuner examples list
simpletuner train example=flux3_image/klein/24g/peft-lora
```

## コピーして編集

コピー先には分類フォルダー構成が維持され、参照される JSON ファイルはレシピフォルダーへコピーされます。設定とデータセット設定を編集し、その環境で学習します。

```bash
simpletuner examples copy flux3_image/klein/24g/peft-lora --dest config
# config/flux3_image/klein/24g/peft-lora/config.json
cd config/flux3_image/klein/24g/peft-lora
simpletuner train
```

WebUI の環境作成画面とセットアップウィザードでは、先にモデルファミリーを選びます。そのファミリーのサンプルだけが表示されます。ファミリーを変更すると以前のサンプル選択は解除されます。

レシピは出発点です。モデルガイドを確認し、データとハードウェアに合わせてバッチサイズ、解像度、学習期間を調整してください。

## Parquet データの段階的な学習

PixArt Sigma のレシピをコピーし、データローダーのキャプチャバケット、マニフェスト名、キャッシュのパスを編集してください。認証情報は AWS の環境変数で渡し、S3 互換エンドポイントには `AWS_ENDPOINT_URL` を指定します。R2 では Webshart エントリーの `aws_region_name` を `auto` に設定します。

```bash
simpletuner examples copy pixart/600M-512/default/progressive-webshart-full --dest config
```

この例では 600M-512 モデルを 128、256、512 ピクセルのクロップなしの段階で微調整します。テキストは現在のバッチでエンコードし、シャード単位でサンプリングします。全体をシャッフルするには `webshart.sampling_policy: global` を指定します。初回の学習を開始する前に段階の区切りを決め、チェックポイントから再開する場合も同じ設定を使います。
