# ワーカーオーケストレーション

SimpleTuner のワーカーオーケストレーションにより、学習ジョブを複数の GPU マシンへ分散できます。ワーカーは中央のオーケストレーターに登録し、リアルタイムでジョブを受け取り、状態を報告します。

## 概要

オーケストレーター/ワーカー構成により、次が可能になります：

- **分散学習** - GPU を持つ任意のマシンでジョブを実行
- **自動検出** - ワーカーが GPU 能力を自己登録
- **リアルタイム配布** - SSE（Server-Sent Events）でジョブ配信
- **混在フリート** - クラウドのエフェメラルワーカーとオンプレ常駐ワーカーを組み合わせ
- **耐障害性** - 孤立したジョブは自動的に再キューされる

## ワーカー種別

| 種別 | ライフサイクル | 用途 |
|------|-----------|----------|
| **Ephemeral** | ジョブ完了後に停止 | クラウドのスポットインスタンス（RunPod, Vast.ai） |
| **Persistent** | ジョブ間もオンライン | オンプレ GPU、予約インスタンス |

## クイックスタート

### 1. オーケストレーターを起動

中央マシンで SimpleTuner サーバーを起動します：

```bash
simpletuner server --host 0.0.0.0 --port 8001
```

本番環境では SSL を有効化：

```bash
simpletuner server --host 0.0.0.0 --port 8001 --ssl
```

### 2. ワーカートークンを作成

**Web UI:** Administration → Workers → Create Worker

**API:**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "gpu-worker-1",
    "worker_type": "persistent",
    "labels": {"location": "datacenter-a", "gpu_type": "a100"}
  }'
```

レスポンスにはトークンが含まれます（1 回のみ表示）：

```json
{
  "worker_id": "w_abc123",
  "token": "wt_xxxxxxxxxxxx",
  "name": "gpu-worker-1"
}
```

### 3. ワーカーを起動

GPU マシン側で実行：

```bash
simpletuner worker \
  --orchestrator-url https://orchestrator.example.com:8001 \
  --worker-token wt_xxxxxxxxxxxx \
  --name gpu-worker-1 \
  --persistent
```

または環境変数で設定：

```bash
export SIMPLETUNER_ORCHESTRATOR_URL=https://orchestrator.example.com:8001
export SIMPLETUNER_WORKER_TOKEN=wt_xxxxxxxxxxxx
export SIMPLETUNER_WORKER_NAME=gpu-worker-1
export SIMPLETUNER_WORKER_PERSISTENT=true

simpletuner worker
```

ワーカーは次を実行します：

1. オーケストレーターに接続
2. GPU 能力を報告（自動検出）
3. ジョブ配信ループに入る
4. 30 秒ごとにハートビート送信

### 4. ワーカーにジョブを送信

**Web UI:** 学習を設定後、**Train in Cloud** → **Worker** を選択

**API:**

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-training-config",
    "target": "worker"
  }'
```

ターゲット指定：

| ターゲット | 挙動 |
|--------|----------|
| `worker` | リモートワーカーにのみ配布 |
| `local` | オーケストレーターの GPU で実行 |
| `auto` | ワーカー優先、無ければローカル |

## CLI リファレンス

```
simpletuner worker [OPTIONS]

OPTIONS:
  --orchestrator-url URL   Orchestrator の URL（または SIMPLETUNER_ORCHESTRATOR_URL）
  --worker-token TOKEN     認証トークン（または SIMPLETUNER_WORKER_TOKEN）
  --name NAME              ワーカー名（デフォルト: ホスト名）
  --persistent             ジョブ間もオンライン（デフォルト: ephemeral）
  -v, --verbose            デバッグログ有効化
```

### Ephemeral と Persistent

**Ephemeral（デフォルト）：**
- 1 ジョブ完了後に停止
- 分単位課金のスポットインスタンスに最適
- オーケストレーターは 1 時間後にオフラインの ephemeral ワーカーをクリーンアップ

**Persistent（`--persistent`）：**
- 新しいジョブを待機して常時オンライン
- 接続が切れても自動再接続
- オンプレ GPU や予約インスタンス向け

## ワーカーライフサイクル

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  CONNECTING │ ──▶ │    IDLE     │ ──▶ │    BUSY     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  DRAINING   │     │   OFFLINE   │
                    └─────────────┘     └─────────────┘
```

| ステータス | 説明 |
|--------|-------------|
| `CONNECTING` | 接続中 |
| `IDLE` | ジョブ待機中 |
| `BUSY` | ジョブ実行中 |
| `DRAINING` | 現ジョブを完了して停止準備 |
| `OFFLINE` | 切断（ハートビートタイムアウト） |

## ヘルスモニタリング

オーケストレーターはワーカーの健全性を監視します：

- **ハートビート間隔:** 30 秒（ワーカー → オーケストレーター）
- **タイムアウトしきい値:** 120 秒無通信でオフライン
- **ヘルスチェックループ:** オーケストレーター側で 60 秒ごとに実行

### 障害時の動作

**ジョブ実行中にワーカーがオフライン：**

1. ハートビートタイムアウト後にジョブを失敗扱い
2. リトライが残っていれば（デフォルト: 3）再キュー
3. 次の空きワーカーがジョブを取得

**オーケストレーター再起動：**

1. ワーカーが自動再接続
2. 進行中ジョブを報告
3. オーケストレーターが状態を整合して再開

## GPU マッチング

ワーカーは登録時に GPU 能力を報告します：

```json
{
  "gpu_count": 2,
  "gpu_name": "NVIDIA A100-SXM4-80GB",
  "gpu_vram_gb": 80,
  "accelerator_type": "cuda"
}
```

ジョブは GPU 要件を指定できます：

```bash
curl -s -X POST http://localhost:8001/api/queue/submit \
  -H "Content-Type: application/json" \
  -d '{
    "config_name": "my-config",
    "target": "worker",
    "worker_labels": {"gpu_type": "a100*"}
  }'
```

スケジューラは次に基づきマッチングします：

1. GPU 数の要件
2. ラベルマッチング（glob パターン対応）
3. ワーカーの空き状況（IDLE）

## ラベル

ラベルにより柔軟なワーカー選択が可能です：

**ワーカー作成時にラベルを付与：**

```bash
curl -s -X POST http://localhost:8001/api/admin/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "worker-1",
    "labels": {
      "location": "us-west",
      "gpu_type": "a100",
      "team": "nlp"
    }
  }'
```

**ラベルでワーカーを選択：**

```bash
# team=nlp のワーカーをマッチ
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"team": "nlp"}}'

# gpu_type が "a100" で始まるワーカーをマッチ
curl -s -X POST http://localhost:8001/api/queue/submit \
  -d '{"config_name": "my-config", "worker_labels": {"gpu_type": "a100*"}}'
```

## 管理操作

### ワーカー一覧

```bash
curl -s http://localhost:8001/api/admin/workers | jq
```

レスポンス：

```json
{
  "workers": [
    {
      "id": "w_abc123",
      "name": "gpu-worker-1",
      "status": "idle",
      "worker_type": "persistent",
      "gpu_count": 2,
      "gpu_name": "A100",
      "labels": {"location": "datacenter-a"},
      "last_heartbeat": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### ワーカーのドレイン

実行中ジョブを完了して、新規配布を停止します：

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/drain
```

ワーカーの挙動：

1. 実行中ジョブを完了
2. DRAINING 状態へ
3. 新規ジョブを拒否
4. ジョブ完了後に切断（ephemeral）または draining 状態のまま（persistent）

### トークンのローテーション

ワーカーの認証トークンを再生成：

```bash
curl -s -X POST http://localhost:8001/api/admin/workers/w_abc123/token
```

古いトークンは即時無効化されます。ワーカー側も新しいトークンに更新してください。

### ワーカー削除

```bash
curl -s -X DELETE http://localhost:8001/api/admin/workers/w_abc123
```

オフライン状態のワーカーのみ削除できます。

## セキュリティ

### トークン認証

- ワーカーは `X-Worker-Token` ヘッダーで認証
- トークンは保存前に SHA-256 ハッシュ化
- トークンは作成後にオーケストレーター外へ出ません
- 定期的にトークンをローテーション

### ネットワークセキュリティ

本番環境では：

1. `--ssl` を使うか、リバースプロキシで TLS を終端
2. ワーカー登録を信頼ネットワークに限定
3. `/api/workers/*` へのアクセスをファイアウォールで制限

### 監査ログ

すべてのワーカー操作はログに記録されます：

- 登録試行（成功/失敗）
- ジョブ配布イベント
- ステータス遷移
- トークンローテーション
- 管理操作

ログの確認は [Audit Guide](AUDIT.md) を参照してください。

## トラブルシューティング

### ワーカーが接続できない

**"Connection refused"**
- オーケストレーターの URL とポートを確認
- ファイアウォールの受信許可を確認
- `--host 0.0.0.0` で起動しているか確認

**"Invalid token"**
- トークンがローテーションされている可能性
- トークン文字列の空白を確認

**"SSL certificate verify failed"**
- 自己署名証明書では `--ssl-no-verify` を使用（開発用）
- もしくは CA 証明書を信頼ストアに追加

### ワーカーが予期せずオフラインになる

**ハートビートタイムアウト（120 秒）**
- ワーカーとオーケストレーターのネットワーク安定性を確認
- ワーカーのリソース枯渇（CPU/メモリ）を確認
- 不安定なネットワークなら `SIMPLETUNER_HEARTBEAT_TIMEOUT` を増やす

**プロセスクラッシュ**
- ワーカーログに Python 例外がないか確認
- GPU ドライバの動作確認（`nvidia-smi`）
- 学習用ディスク容量を確認

### ジョブがワーカーに配布されない

**アイドルワーカーがいない**
- 管理パネルでステータス確認
- ワーカーが接続済みで IDLE か確認
- ジョブとワーカーのラベルが一致しているか確認

**GPU 要件が合わない**
- ジョブがワーカーの GPU 数を超えている
- 学習設定の `--num_processes` を調整

## API リファレンス

### ワーカー向けエンドポイント（Worker → Orchestrator）

| エンドポイント | メソッド | 説明 |
|----------|--------|-------------|
| `/api/workers/register` | POST | 登録と能力報告 |
| `/api/workers/stream` | GET | ジョブ配信の SSE ストリーム |
| `/api/workers/heartbeat` | POST | 定期 keepalive |
| `/api/workers/job/{id}/status` | POST | ジョブ進捗の報告 |
| `/api/workers/disconnect` | POST | 正常終了通知 |

### 管理エンドポイント（`admin.workers` 権限が必要）

| エンドポイント | メソッド | 説明 |
|----------|--------|-------------|
| `/api/admin/workers` | GET | 全ワーカー一覧 |
| `/api/admin/workers` | POST | ワーカートークン作成 |
| `/api/admin/workers/{id}` | DELETE | ワーカー削除 |
| `/api/admin/workers/{id}/drain` | POST | ワーカーをドレイン |
| `/api/admin/workers/{id}/token` | POST | トークンローテーション |

## 関連資料

- [Enterprise Guide](ENTERPRISE.md) - SSO、クォータ、承認ワークフロー
- [Job Queue](../../JOB_QUEUE.md) - キューのスケジューリングと優先度
- [Cloud Training](../cloud/README.md) - クラウドプロバイダー連携
- [API Tutorial](../../api/TUTORIAL.md) - REST API によるローカル学習
