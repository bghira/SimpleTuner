# APIトレーニングチュートリアル

## はじめに

このガイドでは、セットアップとデータセット管理はコマンドラインで行いながら、**HTTP API経由で完全に**SimpleTunerのトレーニングジョブを実行する方法を説明します。他のチュートリアルの構造に沿っていますが、WebUIのオンボーディングはスキップします。以下の内容を学びます:

- 統合サーバーのインストールと起動
- OpenAPIスキーマの検出とダウンロード
- RESTコールを使用した環境の作成と更新
- `/api/training`を介したトレーニングジョブの検証、起動、監視
- 2つの実証済み設定への分岐: PixArt Sigma 900Mフルファインチューンと、Flux Kontext LyCORIS LoRA実行

## 前提条件

- Python 3.10–3.13、Git、および`pip`
- 仮想環境にインストールされたSimpleTuner（`pip install 'simpletuner[cuda]'`またはプラットフォームに合わせたバリアント）
  - CUDA 13 / Blackwell users (NVIDIA B-series GPUs): `pip install 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130`
- 必要なHugging Faceリポジトリへのアクセス（ゲート付きモデルをプルする前に`huggingface-cli login`）
- キャプション付きでローカルにステージングされたデータセット（PixArt用のキャプションテキストファイル、Kontext用のペアになった編集/参照フォルダ）
- `curl`と`jq`を使用できるシェル

## サーバーの起動 {#start-the-server}

SimpleTunerのチェックアウト先（またはパッケージがインストールされている環境）から:

```bash
simpletuner server --port 8001
```

APIは`http://localhost:8001`で利用できます。以下のコマンドを別のターミナルで実行している間、サーバーは実行したままにしてください。

> **ヒント:** トレーニングの準備ができている既存の設定環境がある場合、`--env`を付けてサーバーを起動すると、サーバーが完全にロードされた後に自動的にトレーニングが開始されます:
>
> ```bash
> simpletuner server --port 8001 --env my-training-config
> ```
>
> これにより起動時に設定が検証され、サーバーの準備が完了した直後にトレーニングが開始されます—無人または自動デプロイメントに便利です。`--env`オプションは`simpletuner train --env`と同じように機能します。

### 設定とデプロイメント

本番環境では、バインドアドレスとポートを設定できます:

| オプション | 環境変数 | デフォルト | 説明 |
|--------|---------------------|---------|-------------|
| `--host` | `SIMPLETUNER_HOST` | `0.0.0.0` | サーバーをバインドするアドレス（リバースプロキシの背後では`127.0.0.1`を使用） |
| `--port` | `SIMPLETUNER_PORT` | `8001` | サーバーをバインドするポート |

<details>
<summary><b>本番デプロイメントオプション（TLS、リバースプロキシ、Systemd、Docker）</b></summary>

本番デプロイメントでは、TLS終端のためにリバースプロキシを使用することを推奨します。

#### Nginx設定

```nginx
server {
    listen 443 ssl http2;
    server_name training.example.com;

    # TLS configuration (example using Let's Encrypt paths)
    ssl_certificate /etc/letsencrypt/live/training.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/training.example.com/privkey.pem;

    # WebSocket support for SSE streaming (Critical for real-time logs)
    location /api/training/stream {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        # SSE-specific settings
        proxy_buffering off;
        proxy_read_timeout 86400s;
    }

    # Main application
    location / {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        # Large file uploads for datasets
        client_max_body_size 10G;
        proxy_request_buffering off;
    }
}
```

#### Caddy設定

```caddyfile
training.example.com {
    reverse_proxy 127.0.0.1:8001 {
        # SSE streaming support
        flush_interval -1
    }
    # Large file uploads
    request_body {
        max_size 10GB
    }
}
```

#### systemdサービス

```ini
[Unit]
Description=SimpleTuner Training Server
After=network.target

[Service]
Type=simple
User=trainer
WorkingDirectory=/home/trainer/simpletuner-workspace
Environment="SIMPLETUNER_HOST=127.0.0.1"
Environment="SIMPLETUNER_PORT=8001"
ExecStart=/home/trainer/simpletuner-workspace/.venv/bin/simpletuner server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

#### TraefikとDocker Compose

```yaml
version: '3.8'
services:
  simpletuner:
    image: ghcr.io/bghira/simpletuner:latest
    command: simpletuner server --host 0.0.0.0 --port 8001
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.simpletuner.rule=Host(`training.example.com`)"
      - "traefik.http.services.simpletuner.loadbalancer.server.port=8001"
```
</details>

## 認証

SimpleTunerはマルチユーザー認証をサポートしています。初回起動時には、管理者アカウントを作成する必要があります。

### 初回セットアップ

セットアップが必要かどうかを確認します:

```bash
curl -s http://localhost:8001/api/cloud/auth/setup/status | jq
```

`needs_setup`が`true`の場合、最初の管理者を作成します:

```bash
curl -s -X POST http://localhost:8001/api/cloud/auth/setup/first-admin \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "admin@example.com",
    "username": "admin",
    "password": "your-secure-password"
  }'
```

### APIキー

スクリプトによるアクセスのために、ログイン後にAPIキーを生成します:

```bash
# まずログイン（セッションクッキーを保存）
curl -s -X POST http://localhost:8001/api/cloud/auth/login \
  -H 'Content-Type: application/json' \
  -c cookies.txt \
  -d '{"username": "admin", "password": "your-secure-password"}'

# APIキーを作成
curl -s -X POST http://localhost:8001/api/cloud/auth/api-keys \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{"name": "automation-key"}' | jq
```

返されたキー（`st_`接頭辞付き）を後続のリクエストで使用します:

```bash
curl -s http://localhost:8001/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

### ユーザー管理

管理者はAPIまたはWebUIの**ユーザー管理**ページから追加のユーザーを作成できます:

```bash
# 新しいユーザーを作成（管理者セッションが必要）
curl -s -X POST http://localhost:8001/api/users \
  -H 'Content-Type: application/json' \
  -b cookies.txt \
  -d '{
    "email": "researcher@example.com",
    "username": "researcher",
    "password": "their-password",
    "level_names": ["researcher"]
  }'
```

> **注意:** パブリック登録はデフォルトで無効になっています。管理者は**ユーザー管理 → 登録**タブで有効にできますが、プライベートデプロイメントでは無効のままにすることを推奨します。

## APIの探索

FastAPIはインタラクティブなドキュメントとOpenAPIスキーマを提供します:

```bash
# FastAPI Swagger UI
python -m webbrowser http://localhost:8001/docs

# ReDocビュー
python -m webbrowser http://localhost:8001/redoc

# ローカル検査用にスキーマをダウンロード
curl -o openapi.json http://localhost:8001/openapi.json
jq '.info' openapi.json
```

このチュートリアルで使用されるすべてのエンドポイントは、`configurations`および`training`タグの下に記載されています。

## 高速パス: 環境なしで実行

**設定/環境管理を完全にスキップ**したい場合、トレーニングエンドポイントに完全なCLIスタイルのペイロードを直接投稿することで、1回限りのトレーニング実行を発行できます:

1. データセットを記述するデータローダーJSONを作成または再利用します。トレーナーは`--data_backend_config`で参照されるパスのみを必要とします。

   ```bash
   cat <<'JSON' > config/multidatabackend-once.json
   [
     {
       "id": "demo-images",
       "type": "local",
       "dataset_type": "image",
       "instance_data_dir": "/data/datasets/demo",
       "caption_strategy": "textfile",
       "resolution": 1024,
       "resolution_type": "pixel_area"
     },
     {
       "id": "demo-text-embeds",
       "type": "local",
       "dataset_type": "text_embeds",
       "default": true,
       "cache_dir": "/data/cache/text/demo"
     }
   ]
   JSON
   ```

2. インライン設定を検証します。すべての必須CLI引数（`--model_family`、`--model_type`、`--pretrained_model_name_or_path`、`--output_dir`、`--data_backend_config`、および`--num_train_epochs`または`--max_train_steps`のいずれか）を提供します:

   ```bash
   curl -s -X POST http://localhost:8001/api/training/validate \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0
   ```

   緑色の「Configuration Valid」スニペットは、トレーナーがペイロードを受け入れることを確認します。

3. **同じ**フォームフィールドでトレーニングを起動します（`--seed`や`--validation_prompt`などのオーバーライドを追加できます）:

   ```bash
   curl -s -X POST http://localhost:8001/api/training/start \
     -F __active_tab__=model \
     -F --model_family=pixart_sigma \
     -F --model_type=full \
     -F --model_flavour=900M-1024-v0.6 \
     -F --pretrained_model_name_or_path=terminusresearch/pixart-900m-1024-ft-v0.6 \
     -F --output_dir=/workspace/output/inline-demo \
     -F --data_backend_config=config/multidatabackend-once.json \
     -F --train_batch_size=1 \
     -F --learning_rate=0.0001 \
     -F --max_train_steps=200 \
     -F --num_train_epochs=0 \
     -F --validation_prompt='test shot of <token>'
   ```

サーバーは提出された設定をデフォルトと自動的にマージし、解決された設定をアクティブファイルに書き込み、トレーニングを開始します。再利用可能な環境が必要な場合、残りのセクションではより完全なワークフローをカバーします—任意のモデルファミリーに同じアプローチを再利用できます。

### アドホック実行の監視

このガイドの後半で使用される同じステータスエンドポイントを通じて進行状況を追跡できます:

- `GET /api/training/status`をポーリングして、高レベルの状態、アクティブなジョブID、起動段階情報を取得します。
- `GET /api/training/events?since_index=N`で増分ログを取得するか、`/api/training/events/stream`のWebSocketを介してストリーミングします。

プッシュスタイルの更新には、フォームフィールドと一緒にWebhook設定を提供します:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --model_family=pixart_sigma \
  ... \
  -F --webhook_config='[{"webhook_type":"raw","callback_url":"https://example.com/simpletuner","log_level":"info","ssl_no_verify":false}]' \
  -F --webhook_reporting_interval=10
```

ペイロードは文字列としてJSONシリアライズされている必要があります。サーバーはジョブライフサイクルの更新を`callback_url`にポストします。サポートされているフィールドについては、`documentation/OPTIONS.md`の`--webhook_config`の説明、またはサンプル`config/webhooks.json`テンプレートを参照してください。

<details>
<summary><b>リバースプロキシのWebhook設定</b></summary>

HTTPSを使用したリバースプロキシを使用する場合、Webhook URLはパブリックエンドポイントである必要があります:

1.  **パブリックサーバー:** `https://training.example.com/webhook/callback`を使用
2.  **トンネリング:** ローカル開発にはngrokまたはcloudflaredを使用

**リアルタイムログ（SSE）のトラブルシューティング:**
`GET /api/training/events`は機能するがストリームがハングする場合:
*   **Nginx:** `proxy_buffering off;`を確認し、`proxy_read_timeout`を高く設定（例: 86400s）。
*   **CloudFlare:** 長時間接続を終了します。CloudFlare Tunnelを使用するか、ストリームエンドポイントのプロキシをバイパスします。
</details>

### 手動検証のトリガー

スケジュールされた検証間隔の**間**に評価パスを強制したい場合は、新しいエンドポイントを呼び出します:

```bash
curl -s -X POST http://localhost:8001/api/training/validation/run
```

- サーバーはアクティブな`job_id`でレスポンスします。
- トレーナーは次の勾配同期の直後に発動する検証実行をキューに入れます（現在のマイクロバッチを中断しません）。
- 実行は設定された検証プロンプト/設定を再利用するため、結果の画像は通常のイベント/ログストリームに表示されます。
- 組み込みパイプラインではなく外部実行可能ファイルに検証をオフロードするには、設定（またはペイロード）で`--validation_method=external-script`を設定し、`--validation_external_script`をスクリプトに指定します。プレースホルダーを使用してトレーニングコンテキストをスクリプトに渡すことができます: `{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}`、`{remote_checkpoint_path}`（検証の場合は空）、および任意の`validation_*`設定値（例: `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`）。スクリプトをファイア・アンド・フォーゲットでトレーニングをブロックせずに実行したい場合は、`--validation_external_background`を有効にします。
- チェックポイントがローカルに書き込まれた直後（アップロードがバックグラウンドで実行中でも）に自動化をトリガーしたいですか？ `--post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'`を設定します。検証フックと同じプレースホルダーを使用します。このフックでは`{remote_checkpoint_path}`は空に解決されます。
- SimpleTunerの組み込みアップロードを維持し、結果のリモートURLを独自のツールに渡すことを好みますか？ 代わりに`--post_upload_script`を設定します。これは公開プロバイダー/Hugging Face Hubアップロードごとに1回、`{remote_checkpoint_path}`（バックエンドによって提供される場合）と同じコンテキストプレースホルダーで発動します。SimpleTunerはスクリプトからの結果を取り込まないため、アーティファクト/メトリクスを自分でトラッカーにログします。
  - 例: `--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'`（`notify.sh`はトラッカーAPIを呼び出します）。
  - 動作サンプル:
    - `simpletuner/examples/external-validation/replicate_post_upload.py`は`{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}`を使用してReplicate推論をトリガーします。
    - `simpletuner/examples/external-validation/wavespeed_post_upload.py`は同じプレースホルダーを使用してWaveSpeed推論をトリガーし、完了をポーリングします。
    - `simpletuner/examples/external-validation/fal_post_upload.py`はfal.ai Flux LoRA推論をトリガーします（`FAL_KEY`と`flux`を含む`model_family`が必要）。
    - `simpletuner/examples/external-validation/use_second_gpu.py`はアップロードを必要とせずに別のGPUでFlux LoRA推論を実行します。

アクティブなジョブがない場合、エンドポイントはHTTP 400を返すため、リトライをスクリプト化する際は最初に`/api/training/status`を確認してください。

### 手動チェックポイントのトリガー

次のスケジュールされたチェックポイントを待たずに、現在のモデル状態を即座に永続化するには、以下を実行します:

```bash
curl -s -X POST http://localhost:8001/api/training/checkpoint/run
```

- サーバーはアクティブな`job_id`でレスポンスします。
- トレーナーは次の勾配同期後に、スケジュールされたチェックポイントと同じ設定（アップロードルール、ローリング保持など）を使用してチェックポイントを保存します。
- ローリングクリーンアップとWebhook通知は、スケジュールされたチェックポイントとまったく同じように動作します。

検証と同様に、トレーニングジョブが実行中でない場合、エンドポイントはHTTP 400を返します。

### 検証プレビューのストリーム

Tiny AutoEncoder（または同等の）フックを公開するモデルは、画像/ビデオがまだサンプリング中に**ステップごとの検証プレビュー**を発行できます。ペイロードにCLIフラグを追加して機能を有効にします:

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=validation \
  -F --validation_preview=true \
  -F --validation_preview_steps=4 \
  -F --validation_num_inference_steps=20 \
  …その他のフィールド…
```

- `--validation_preview`（デフォルトは`false`）はプレビューデコーダーのロックを解除します。
- `--validation_preview_steps`は中間フレームを発行する頻度を決定します。上記の例では、ステップ1、5、9、13、17、20でイベントを受信します（最初のステップは常に発行され、その後4ステップごと）。

各プレビューは`validation.image`イベントとして公開されます（`simpletuner/helpers/training/validation.py:899-929`を参照）。rawウェブフック、`GET /api/training/events`、または`/api/training/events/stream`のSSEストリームを介してこれらを消費できます。典型的なペイロードは次のようになります:

```json
{
  "type": "validation.image",
  "title": "Validation (step 5/20): night bench",
  "body": "night bench shot of <token>",
  "data": {
    "step": 5,
    "timestep": 563.0,
    "resolution": [1024, 1024],
    "validation_type": "intermediary",
    "prompt": "night bench shot of <token>",
    "step_label": "5/20"
  },
  "images": [
    {"src": "data:image/png;base64,...", "mime_type": "image/png"}
  ]
}
```

ビデオ対応モデルは代わりに`videos`配列を添付します（`mime_type: image/gif`のGIFデータURI）。これらのイベントはほぼリアルタイムでストリーミングされるため、ダッシュボードに直接表示したり、rawウェブフックバックエンドを介してSlack/Discordに送信したりできます。

## 一般的なAPIワークフロー

1. **環境を作成** – `POST /api/configs/environments`
2. **データローダーファイルを入力** – 生成された`config/<env>/multidatabackend.json`を編集
3. **トレーニングハイパーパラメータを更新** – `PUT /api/configs/<env>`
4. **環境を有効化** – `POST /api/configs/<env>/activate`
5. **トレーニングパラメータを検証** – `POST /api/training/validate`
6. **トレーニングを起動** – `POST /api/training/start`
7. **ジョブを監視または停止** – `/api/training/status`、`/api/training/events`、`/api/training/stop`、`/api/training/cancel`

以下の各例はこのフローに従います。

## オプション: API経由でデータセットをアップロード（ローカルバックエンド） {#optional-upload-datasets-over-the-api-local-backends}

データセットがSimpleTunerを実行しているマシンにまだない場合、データローダーを配線する前にHTTP経由でプッシュできます。アップロードエンドポイントは設定された`datasets_dir`（WebUIオンボーディング中に設定）を尊重し、ローカルファイルシステム向けです:

1. データセットルート下に**ターゲットフォルダーを作成**:

   ```bash
   DATASETS_DIR=${DATASETS_DIR:-/workspace/simpletuner/datasets}
   curl -s -X POST http://localhost:8001/api/datasets/folders \
     -F parent_path="$DATASETS_DIR" \
     -F folder_name="pixart-upload"
   ```

2. **ファイルまたはZIPをアップロード**（画像とオプションの`.txt/.jsonl/.csv`メタデータが受け入れられます）:

   ```bash
   # zipをアップロード（サーバー上で自動的に展開）
   curl -s -X POST http://localhost:8001/api/datasets/upload/zip \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F file=@/path/to/dataset.zip

   # または個別のファイルをアップロード
   curl -s -X POST http://localhost:8001/api/datasets/upload \
     -F target_path="$DATASETS_DIR/pixart-upload" \
     -F files[]=@image001.png \
     -F files[]=@image001.txt
   ```

> **アップロードのトラブルシューティング:** リバースプロキシを使用している場合に大きなアップロードが「Entity Too Large」エラーで失敗する場合は、本文サイズ制限を増やしていることを確認してください（例: Nginxの`client_max_body_size 10G;`またはCaddyの`request_body { max_size 10GB }`）。

アップロード完了後、`multidatabackend.json`エントリを新しいフォルダー（例: `"/data/datasets/pixart-upload"`）に指定します。

## 例: PixArt Sigma 900Mフルファインチューン

### 1. REST経由で環境を作成

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "pixart-api-demo",
        "model_family": "pixart_sigma",
        "model_flavour": "900M-1024-v0.6",
        "model_type": "full",
        "description": "PixArt 900M API-driven training"
      }
```

これにより`config/pixart-api-demo/`とスターター`multidatabackend.json`が作成されます。

### 2. データセットを配線

データローダーファイルを編集します（パスを実際のデータセット/キャッシュの場所に置き換えます）:

```bash
cat <<'JSON' > config/pixart-api-demo/multidatabackend.json
[
  {
    "id": "pixart-camera",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/pseudo-camera-10k",
    "caption_strategy": "filename",
    "resolution": 1.0,
    "resolution_type": "area",
    "minimum_image_size": 0.25,
    "maximum_image_size": 1.0,
    "target_downsample_size": 1.0,
    "cache_dir_vae": "/data/cache/vae/pixart/pseudo-camera-10k",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square",
    "metadata_backend": "discovery"
  },
  {
    "id": "pixart-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/pixart/pseudo-camera-10k",
    "write_batch_size": 128
  }
]
JSON
```

### 3. API経由でハイパーパラメータを更新

現在の設定を取得し、オーバーライドをマージして、結果を戻します:

```bash
curl -s http://localhost:8001/api/configs/pixart-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/pixart900m",
      "--train_batch_size": 2,
      "--gradient_accumulation_steps": 2,
      "--learning_rate": 0.0001,
      "--optimizer": "adamw_bf16",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 500,
      "--max_train_steps": 1800,
      "--num_train_epochs": 0,
      "--validation_prompt": "a studio portrait of <token> wearing a leather jacket",
      "--validation_guidance": 3.8,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 28,
      "--cache_dir_vae": "/data/cache/vae/pixart",
      "--seed": 1337,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "bf16",
      "--dataloader_prefetch": true,
      "--report_to": "none",
      "--checkpoints_total_limit": 4,
      "--validation_seed": 12345,
      "--data_backend_config": "pixart-api-demo/multidatabackend.json"
    }' > /tmp/pixart-config.json

jq '{
      "name": "pixart-api-demo",
      "description": "PixArt 900M full tune (API)",
      "tags": ["pixart", "api"],
      "config": .
    }' /tmp/pixart-config.json > /tmp/pixart-update.json

curl -s -X PUT http://localhost:8001/api/configs/pixart-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/pixart-update.json
```

### 4. 環境を有効化

```bash
curl -s -X POST http://localhost:8001/api/configs/pixart-api-demo/activate
```

### 5. 起動前に検証

`validate`はフォームエンコードされたデータを消費します。少なくとも、`num_train_epochs`または`max_train_steps`の1つが0であることを確認します:

```bash
curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

成功ブロック（`Configuration Valid`）は、トレーナーがマージされた設定を受け入れることを意味します。

### 6. トレーニング開始

```bash
curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

レスポンスにはジョブIDが含まれます。トレーニングはステップ3で保存されたパラメータで実行されます。

### 7. 監視と停止

```bash
# 粗いステータスをクエリ
curl -s http://localhost:8001/api/training/status | jq

# 増分ログイベントをストリーム
curl -s 'http://localhost:8001/api/training/events?since_index=0' | jq

# キャンセルまたは停止
curl -s -X POST http://localhost:8001/api/training/stop
curl -s -X POST http://localhost:8001/api/training/cancel -F job_id=<JOB_ID>
```

PixArt注意事項:

- 選択した`train_batch_size * gradient_accumulation_steps`に対して十分な大きさのデータセットを保持します
- ミラーが必要な場合は`HF_ENDPOINT`を設定し、`terminusresearch/pixart-900m-1024-ft-v0.6`をダウンロードする前に認証します
- プロンプトに応じて`--validation_guidance`を3.6から4.4の間で調整します

## 例: Flux Kontext LyCORIS LoRA

KontextはパイプラインのほとんどをFlux Devと共有しますが、ペアになった編集/参照画像が必要です。

### 1. 環境をプロビジョニング

```bash
curl -s -X POST http://localhost:8001/api/configs/environments \
  -H 'Content-Type: application/json' \
  -d
```json
{
        "name": "kontext-api-demo",
        "model_family": "flux",
        "model_flavour": "kontext",
        "model_type": "lora",
        "lora_type": "lycoris",
        "description": "Flux Kontext LoRA via API"
      }
```

### 2. ペアデータローダーを記述

Kontextは編集/参照ペアとテキスト埋め込みキャッシュが必要です:

```bash
cat <<'JSON' > config/kontext-api-demo/multidatabackend.json
[
  {
    "id": "kontext-edit",
    "type": "local",
    "dataset_type": "image",
    "instance_data_dir": "/data/datasets/kontext/edit",
    "conditioning_data": ["kontext-reference"],
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "caption_strategy": "textfile",
    "minimum_image_size": 768,
    "maximum_image_size": 1536,
    "target_downsample_size": 1024,
    "cache_dir_vae": "/data/cache/vae/kontext/edit",
    "crop": true,
    "crop_style": "random",
    "crop_aspect": "square"
  },
  {
    "id": "kontext-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/datasets/kontext/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "/data/cache/vae/kontext/reference"
  },
  {
    "id": "kontext-text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "/data/cache/text/kontext"
  }
]
JSON
```

編集フォルダーと参照フォルダー間でファイル名が一致することを確認してください。SimpleTunerは名前に基づいて埋め込みをステッチします。

### 3. Kontext固有のハイパーパラメータを適用

```bash
curl -s http://localhost:8001/api/configs/kontext-api-demo \
  | jq '.config + {
      "--output_dir": "/workspace/output/kontext",
      "--train_batch_size": 1,
      "--gradient_accumulation_steps": 4,
      "--learning_rate": 0.00001,
      "--optimizer": "optimi-lion",
      "--lr_scheduler": "cosine",
      "--lr_warmup_steps": 200,
      "--max_train_steps": 12000,
      "--num_train_epochs": 0,
      "--validation_prompt": "a cinematic 1024px product photo of <token>",
      "--validation_guidance": 2.5,
      "--validation_resolution": "1024x1024",
      "--validation_num_inference_steps": 30,
      "--cache_dir_vae": "/data/cache/vae/kontext",
      "--seed": 777,
      "--resume_from_checkpoint": "latest",
      "--base_model_precision": "int8-quanto",
      "--dataloader_prefetch": true,
      "--report_to": "wandb",
      "--lora_rank": 16,
      "--lora_dropout": 0.05,
      "--conditioning_multidataset_sampling": "combined",
      "--clip_skip": 2,
      "--data_backend_config": "kontext-api-demo/multidatabackend.json"
    }' > /tmp/kontext-config.json

jq '{
      "name": "kontext-api-demo",
      "description": "Flux Kontext LyCORIS (API)",
      "tags": ["flux", "kontext", "api"],
      "config": .
    }' /tmp/kontext-config.json > /tmp/kontext-update.json

curl -s -X PUT http://localhost:8001/api/configs/kontext-api-demo \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/kontext-update.json
```

### 4. 有効化、検証、起動

```bash
curl -s -X POST http://localhost:8001/api/configs/kontext-api-demo/activate

curl -s -X POST http://localhost:8001/api/training/validate \
  -F __active_tab__=model \
  -F --num_train_epochs=0

curl -s -X POST http://localhost:8001/api/training/start \
  -F __active_tab__=model \
  -F --num_train_epochs=0
```

Kontextのヒント:

- `conditioning_type=reference_strict`はクロップを整列させます。データセットのアスペクト比が異なる場合は`reference_loose`に切り替えます
- 1024pxで24GB VRAM内に収めるには`int8-quanto`に量子化します。フル精度にはHopper/BlackwellクラスのGPUが必要です
- マルチノード実行の場合、サーバーを起動する前に`--accelerate_config`または`CUDA_VISIBLE_DEVICES`を設定します

## GPU対応キューイングでローカルジョブを送信

マルチGPUマシンで実行する場合、GPU割り当て認識でキューAPI経由でローカルトレーニングジョブを送信できます。必要なGPUが利用できない場合、ジョブはキューに入れられます。

### GPU可用性の確認

```bash
curl -s "http://localhost:8001/api/system/status?include_allocation=true" | jq '.gpu_allocation'
```

レスポンスは利用可能なGPUを示します:

```json
{
  "allocated_gpus": [0, 1],
  "available_gpus": [2, 3],
  "running_local_jobs": 1,
  "devices": [
    {"index": 0, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
    {"index": 1, "name": "A100", "memory_gb": 40, "allocated": true, "job_id": "abc123"},
    {"index": 2, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null},
    {"index": 3, "name": "A100", "memory_gb": 40, "allocated": false, "job_id": null}
  ]
}
```

ローカルGPU情報を含むキュー統計も取得できます:

```bash
curl -s http://localhost:8001/api/queue/stats | jq '.local'
```

### ローカルジョブの送信

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "no_wait": false,
    "any_gpu": false
  }'
```

オプション:

| オプション | デフォルト | 説明 |
|--------|---------|-------------|
| `config_name` | 必須 | 実行するトレーニング環境の名前 |
| `no_wait` | false | trueの場合、GPUが利用できないときに即座に拒否 |
| `any_gpu` | false | trueの場合、設定されたデバイスIDの代わりに利用可能な任意のGPUを使用 |

レスポンス:

```json
{
  "success": true,
  "job_id": "abc123",
  "status": "running",
  "allocated_gpus": [0, 1],
  "queue_position": null
}
```

`status`フィールドは結果を示します:

- `running` - 割り当てられたGPUでジョブが即座に開始
- `queued` - ジョブがキューに入り、GPUが利用可能になると開始
- `rejected` - GPUが利用できず、`no_wait`がtrueだった

### ローカル同時実行制限の設定

管理者はキュー同時実行エンドポイントを介して、使用できるローカルジョブとGPUの数を制限できます:

```bash
# 現在の制限を取得
curl -s http://localhost:8001/api/queue/stats | jq '{local_gpu_max_concurrent, local_job_max_concurrent}'

# 制限を更新（クラウド制限と一緒に）
curl -s -X POST http://localhost:8001/api/queue/concurrency \
  -H 'Content-Type: application/json' \
  -d '{
    "local_gpu_max_concurrent": 6,
    "local_job_max_concurrent": 2
  }'
```

無制限のGPU使用には`local_gpu_max_concurrent`を`null`に設定します。

### CLI代替

同じ機能はCLI経由でも利用できます:

```bash
# デフォルトのキューイング動作で送信
simpletuner jobs submit my-config

# GPUが利用できない場合は拒否
simpletuner jobs submit my-config --no-wait

# 利用可能な任意のGPUを使用
simpletuner jobs submit my-config --any-gpu

# 何が起こるかをプレビュー（ドライラン）
simpletuner jobs submit my-config --dry-run
```

## リモートワーカーにジョブをディスパッチ

ワーカーとして登録されたリモートGPUマシンがある場合（[ワーカーオーケストレーション](../experimental/server/WORKERS.md)を参照）、キューAPI経由でジョブをディスパッチできます。

### 利用可能なワーカーの確認

```bash
curl -s http://localhost:8001/api/admin/workers | jq '.workers[] | {name, status, gpu_name, gpu_count}'
```

### 特定のターゲットへの送信

```bash
# リモートワーカーを優先、ローカルGPUにフォールバック（デフォルト）
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "auto"
  }'

# リモートワーカーのみに強制ディスパッチ
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'

# オーケストレーターのローカルGPUのみで実行
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "local"
  }'
```

### ラベルによるワーカーの選択

ワーカーはフィルタリング用のラベルを持つことができます（例: GPUタイプ、場所、チーム）:

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "config_name": "my-training-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*", "team": "nlp"}
  }'
```

ラベルはglobパターンをサポートします（`*`は任意の文字に一致）。

## 便利なエンドポイント一覧

- `GET /api/configs/` – 環境をリスト（トレーニング設定には`?config_type=model`を渡す）
- `GET /api/configs/examples` – 同梱されているテンプレートを列挙
- `POST /api/configs/{name}/dataloader` – デフォルトが必要な場合にデータローダーファイルを再生成
- `GET /api/training/status` – 高レベルの状態、アクティブな`job_id`、起動段階情報
- `GET /api/training/events?since_index=N` – 増分トレーナーログストリーム
- `POST /api/training/checkpoints` – アクティブなジョブの出力ディレクトリのチェックポイントをリスト
- `GET /api/system/status?include_allocation=true` – GPU割り当て情報を含むシステムメトリクス
- `GET /api/queue/stats` – ローカルGPU割り当てを含むキュー統計
- `POST /api/queue/submit` – GPU対応キューイングでローカルまたはワーカージョブを送信
- `POST /api/queue/concurrency` – クラウドおよびローカル同時実行制限を更新
- `GET /api/admin/workers` – 登録されたワーカーとそのステータスをリスト

## 次のステップ

- `documentation/OPTIONS.md`で特定のオプション定義を探索
- これらのRESTコールを`jq`/`yq`またはPythonクライアントと組み合わせて自動化
- リアルタイムダッシュボード用に`/api/training/events/stream`でWebSocketをフック
- エクスポートされた設定（`GET /api/configs/<env>/export`）を再利用して、動作するセットアップをバージョン管理
- **クラウドGPUでトレーニングを実行**するにはReplicateを使用—[クラウドトレーニングチュートリアル](../experimental/cloud/TUTORIAL.md)を参照

これらのパターンを使用すると、実証済みのCLIセットアッププロセスに依存しながら、WebUIに触れることなくSimpleTunerトレーニングを完全にスクリプト化できます。
