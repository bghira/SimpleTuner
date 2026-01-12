# 公開（Publishing）プロバイダー

SimpleTuner は `--publishing_config` を通じて学習出力を複数の宛先に公開できます。Hugging Face へのアップロードは引き続き `--push_to_hub` で制御され、`publishing_config` はその他のプロバイダー向けに追加的に実行されます（メインプロセスで検証完了後に実行）。

## 設定形式
- インライン JSON（`--publishing_config='[{"provider": "s3", ...}]'`）、SDK から渡す Python dict、または JSON ファイルのパスを受け付けます。
- 値はリストに正規化され、`--webhook_config` と同様の挙動です。
- 各エントリには `provider` キーが必要です。`base_path` を指定するとリモート宛先内のパスにプレフィックスを付けられます。URI を返せない設定の場合、プロバイダーは一度だけ警告ログを出します。

## デフォルトの成果物

公開は、実行時の `output_dir`（フォルダ/ファイル）をディレクトリ名を基準にアップロードします。メタデータには現在のジョブ ID と検証種別が含まれ、下流で URI を実行に紐付けられます。

## プロバイダー

利用するプロバイダーのオプション依存関係をプロジェクトの `.venv` にインストールしてください。

### S3 互換 / Backblaze B2（S3 API）
- プロバイダー: `s3` または `backblaze_b2`
- 依存関係: `pip install boto3`
- 例：
```json
[
  {
    "provider": "s3",
    "bucket": "simpletuner-models",
    "region": "us-east-1",
    "access_key": "AKIA...",
    "secret_key": "SECRET",
    "base_path": "runs/2024",
    "endpoint_url": "https://s3.us-west-004.backblazeb2.com",
    "public_base_url": "https://cdn.example.com/models"
  }
]
```

⚠️ **セキュリティ注意**: 資格情報をバージョン管理にコミットしないでください。本番環境では環境変数の置換またはシークレット管理を使用してください。

### Azure Blob Storage
- プロバイダー: `azure_blob`（別名 `azure`）
- 依存関係: `pip install azure-storage-blob`
- 例：
```json
[
  {
    "provider": "azure_blob",
    "connection_string": "DefaultEndpointsProtocol=....",
    "container": "simpletuner",
    "base_path": "models/latest"
  }
]
```

### Dropbox
- プロバイダー: `dropbox`
- 依存関係: `pip install dropbox`
- 例：
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
大きなファイルは自動的にアップロードセッションでストリーミングされ、許可されている場合は共有リンクが作成されます。許可されていない場合は `dropbox://` のパスが記録されます。

## CLI 使用例
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```

SimpleTuner をプログラムから呼び出す場合、`publishing_config` に list/dict を渡せば正規化されます。
