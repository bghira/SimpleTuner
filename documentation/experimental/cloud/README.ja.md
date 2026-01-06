# クラウド学習システム

> **ステータス:** Experimental
>
> **利用可能:** Web UI（Cloud タブ）

SimpleTuner のクラウド学習システムは、独自のインフラを構築せずにクラウド GPU プロバイダ上で学習ジョブを実行できます。複数プロバイダを追加できるようプラガブルに設計されています。

## 概要

クラウド学習システムが提供するもの:

- **統合ジョブ追跡** - ローカルとクラウドの学習ジョブを一元管理
- **自動データ梱包** - ローカルデータセットを自動でパッケージ化してアップロード
- **結果配布** - HuggingFace、S3、ローカルダウンロードへ送信可能
- **コスト追跡** - プロバイダ別の支出を監視し、上限を設定
- **設定スナップショット** - git で学習設定をバージョン管理（任意）

## 重要コンセプト

クラウド学習を使う前に、以下の 3 点を理解してください。

### 1. データの扱い

クラウドジョブを送信すると:

1. **データセットが梱包される** - ローカルデータセット（`type: "local"`）が zip 化され、概要が表示されます
2. **プロバイダへアップロード** - 同意後、zip がクラウドプロバイダへ直接送信されます
3. **学習実行** - モデルは必要なダウンロード後にクラウド GPU で学習されます
4. **データ削除** - 学習後、アップロードデータはプロバイダ側から削除され、モデルが配布されます

**セキュリティ注記:**
- API トークンはローカルから外に出ません
- 機微なファイル（.env、.git、資格情報）は自動除外
- 各ジョブ送信前にアップロード内容を確認し同意

### 2. 学習済みモデルの受け取り

学習結果のモデルは受け取り先が必要です。以下のいずれかを設定します:

| 宛先 | 設定 | 最適用途 |
|-------------|-------|----------|
| **HuggingFace Hub** | `HF_TOKEN` を設定、Publishing タブで有効化 | 共有・簡単アクセス |
| **ローカルダウンロード** | Webhook URL を設定し ngrok などで公開 | プライバシー・ローカル運用 |
| **S3 ストレージ** | Publishing タブでエンドポイント設定 | チーム共有・アーカイブ |

[Receiving Trained Models](TUTORIAL.md#receiving-trained-models) を参照して段階的に設定してください。

### 3. コストモデル

Replicate は GPU 時間を秒課金します:

| ハードウェア | VRAM | コスト | 典型 LoRA（2000 steps） |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**課金開始** は学習開始時、**課金停止** は完了または失敗時です。

**自己防衛:**
- Cloud 設定で支出上限を設定
- 送信前にコスト見積もりを表示
- 実行中ジョブはいつでもキャンセル可能（使用分のみ課金）

価格と上限は [Costs](REPLICATE.md#costs) を参照してください。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web UI (Cloud Tab)                       │
├─────────────────────────────────────────────────────────────────┤
│  Job List  │  Metrics/Charts  │  Actions/Config  │  Job Details │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Cloud API Routes  │
                    │   /api/cloud/*      │
                    └──────────┬──────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼────────┐   ┌────────▼────────┐   ┌────────▼────────┐
│    JobStore     │   │ Upload Service  │   │    Provider     │
│  (Persistence)  │   │ (Data Packaging)│   │   Clients       │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                     │
                             ┌───────────────────────┤
                             │                       │
                   ┌─────────▼─────────┐   ┌─────────▼─────────┐
                   │     Replicate     │   │  SimpleTuner.io   │
                   │    Cog Client     │   │   (Coming Soon)   │
                   └───────────────────┘   └───────────────────┘
```

## 対応プロバイダ

| プロバイダ | ステータス | 機能 |
|----------|--------|----------|
| [Replicate](REPLICATE.md) | Supported | コスト追跡、ライブログ、Webhook |
| [Worker Orchestration](../server/WORKERS.md) | Supported | 自前分散ワーカー、任意 GPU |
| SimpleTuner.io | Coming Soon | SimpleTuner チームによるマネージド学習 |

### Worker Orchestration

複数マシンの分散学習は [Worker Orchestration Guide](../server/WORKERS.md) を参照してください。ワーカーは以下で動作します:

- オンプレ GPU サーバ
- クラウド VM（任意プロバイダ）
- スポットインスタンス（RunPod、Vast.ai、Lambda Labs）

ワーカーは SimpleTuner オーケストレータに登録され、自動でジョブを受け取ります。

## データフロー

### ジョブ送信

1. **設定準備** - 学習設定をシリアライズ
2. **データ梱包** - ローカルデータセット（`type: "local"`）を zip 化
3. **アップロード** - zip を Replicate のファイルホスティングに送信
4. **送信** - クラウドプロバイダへジョブ送信
5. **追跡** - ジョブ状態をポーリングしリアルタイム更新

### 結果受け取り

結果は次の方法で配信されます:

1. **HuggingFace Hub** - 学習済みモデルを HuggingFace にプッシュ
2. **S3 互換ストレージ** - 任意の S3 エンドポイントへアップロード（AWS、MinIO など）
3. **ローカルダウンロード** - SimpleTuner の内蔵 S3 互換エンドポイントでローカル受信

## データプライバシーと同意

クラウドジョブ送信時、SimpleTuner は以下をアップロードする場合があります:

- **学習データセット** - `type: "local"` の画像/ファイル
- **設定** - 学習パラメータ（学習率、モデル設定など）
- **キャプション/メタデータ** - データセットに紐づくテキストファイル

データはクラウドプロバイダ（例: Replicate のファイルホスティング）へ直接アップロードされ、SimpleTuner のサーバを経由しません。

### 同意設定

Cloud タブでアップロード挙動を設定できます:

| 設定 | 挙動 |
|---------|----------|
| **Always Ask** | 各アップロード前に確認ダイアログを表示 |
| **Always Allow** | 信頼されたワークフローでは確認を省略 |
| **Never Upload** | クラウド学習を無効化（ローカルのみ） |

## ローカル S3 エンドポイント

SimpleTuner には学習済みモデル受信用の S3 互換エンドポイントがあります:

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

ファイルは既定で `~/.simpletuner/cloud_outputs/` に保存されます。

資格情報は手動設定できます。設定しない場合、各ジョブごとに一時的な資格情報が自動生成されるため、この方法が推奨です。

これにより「ダウンロードのみ」モードが可能になります:
1. ローカル SimpleTuner への webhook URL を設定
2. SimpleTuner が S3 公開設定を自動構成
3. 学習済みモデルがローカルにアップロードされる

**注記:** クラウドプロバイダが到達できるよう、ngrok、cloudflared などでローカル SimpleTuner を公開する必要があります。

## 新しいプロバイダの追加

クラウドシステムは拡張性を想定しています。新しいプロバイダを追加するには:

1. `CloudTrainerService` を実装するクライアントクラスを作成:

```python
from .base import CloudTrainerService, CloudJobInfo, CloudJobStatus

class NewProviderClient(CloudTrainerService):
    @property
    def provider_name(self) -> str:
        return "new_provider"

    @property
    def supports_cost_tracking(self) -> bool:
        return True  # or False

    @property
    def supports_live_logs(self) -> bool:
        return True  # or False

    async def validate_credentials(self) -> Dict[str, Any]:
        # Validate API key and return user info
        ...

    async def list_jobs(self, limit: int = 50) -> List[CloudJobInfo]:
        # List recent jobs from the provider
        ...

    async def run_job(self, config, dataloader, ...) -> CloudJobInfo:
        # Submit a new training job
        ...

    async def cancel_job(self, job_id: str) -> bool:
        # Cancel a running job
        ...

    async def get_job_logs(self, job_id: str) -> str:
        # Fetch logs for a job
        ...

    async def get_job_status(self, job_id: str) -> CloudJobInfo:
        # Get current status of a job
        ...
```

2. クラウドルートにプロバイダを登録
3. 新プロバイダ用の UI タブを追加

## ファイルと場所

| パス | 説明 |
|------|-------------|
| `~/.simpletuner/cloud/` | クラウド関連状態とジョブ履歴 |
| `~/.simpletuner/cloud/job_history.json` | 統合ジョブ追跡データベース |
| `~/.simpletuner/cloud/provider_configs/` | プロバイダ別設定 |
| `~/.simpletuner/cloud_outputs/` | ローカル S3 エンドポイントの保存先 |

## トラブルシューティング

### "REPLICATE_API_TOKEN not set"

SimpleTuner 起動前に環境変数を設定:

```bash
export REPLICATE_API_TOKEN="r8_..."
simpletuner --webui
```

### データアップロード失敗

- インターネット接続を確認
- データセットのパスが存在するか確認
- ブラウザコンソールのエラー確認
- クラウドプロバイダのクレジットと権限を確認

### Webhook が結果を受け取らない

- ローカルインスタンスが公開されているか確認
- Webhook URL が正しいか確認
- ファイアウォールの受信設定を確認

## 現在の制限

クラウド学習システムは **単発ジョブ** を想定しています。以下は現時点で未対応です:

### ワークフロー/パイプライン（DAG）

ジョブ依存や多段ワークフロー（あるジョブの出力が別ジョブの入力になる）は未対応です。各ジョブは独立し自己完結しています。

**ワークフローが必要な場合:**
- 外部オーケストレーションツール（Airflow、Prefect、Dagster）を使用
- REST API を使ってパイプラインからジョブを連鎖
- Airflow 統合例は [ENTERPRISE.md](../server/ENTERPRISE.md#external-orchestration-airflow) を参照

### 学習再開

中断・失敗・早期停止した学習の再開は組み込みで未対応です。ジョブが失敗またはキャンセルされた場合:
- 最初から再送信が必要
- クラウドストレージから自動でチェックポイント復旧しません

**回避策:**
- HuggingFace Hub への頻繁なプッシュ（`--push_checkpoints_to_hub`）で中間チェックポイントを保存
- 出力をダウンロードして新ジョブの初期状態として再アップロードする独自管理
- 重要な長時間ジョブは小さな学習セグメントに分割

これらの制限は将来のリリースで改善される可能性があります。

## 参照

### クラウド学習

- [Cloud Training Tutorial](TUTORIAL.md) - 導入ガイド
- [Replicate Integration](REPLICATE.md) - Replicate 設定
- [Job Queue](../../JOB_QUEUE.md) - ジョブスケジューリングと同時実行
- [Operations Guide](OPERATIONS_TUTORIAL.md) - 本番デプロイ

### マルチユーザー機能（ローカル/クラウド共通）

- [Enterprise Guide](../server/ENTERPRISE.md) - SSO、承認、ガバナンス
- [External Authentication](../server/EXTERNAL_AUTH.md) - OIDC と LDAP 設定
- [Audit Logging](../server/AUDIT.md) - セキュリティイベント監査

### 一般

- [Local API Tutorial](../../api/TUTORIAL.md) - REST API でのローカル学習
- [Datasets Documentation](../../DATALOADER.md) - dataloader 設定理解
