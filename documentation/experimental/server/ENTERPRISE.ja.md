# エンタープライズガイド

このドキュメントでは、認証、承認ワークフロー、クォータ管理を備えたマルチユーザー環境で SimpleTuner を運用する方法を説明します。

## 1. デプロイとインフラ

### 設定方法

ほとんどのエンタープライズ機能は **Web UI**（Administration パネル）または **REST API** から設定できます。一部のインフラ設定は、設定ファイルまたは環境変数が必要です。

| 機能 | Web UI | API | 設定ファイル |
|---------|--------|-----|-------------|
| OIDC/LDAP プロバイダー | ✓ | ✓ | ✓ |
| ユーザーとロール | ✓ | ✓ | |
| 承認ルール | ✓ | ✓ | |
| クォータ | ✓ | ✓ | |
| 通知 | ✓ | ✓ | |
| ネットワークバイパス（信頼プロキシ） | | | ✓ |
| バックグラウンドのジョブポーリング | | | ✓ |
| TLS 設定 | | | ✓ |

**設定ファイル**（`simpletuner-enterprise.yaml` または `.json`）は、起動時に必要なインフラ設定にのみ必要です。SimpleTuner は次の順に検索します：

1. `$SIMPLETUNER_ENTERPRISE_CONFIG`（環境変数）
2. `./simpletuner-enterprise.yaml`（カレントディレクトリ）
3. `~/.config/simpletuner/enterprise.yaml`
4. `/etc/simpletuner/enterprise.yaml`

このファイルは `${VAR}` 構文による環境変数の展開に対応します。

### クイックスタートチェックリスト

1.  **SimpleTuner を起動**: `simpletuner server`（ローカル利用なら `--webui`）
2.  **UI で設定**: Administration パネルでユーザー、SSO、クォータを設定
3.  **ヘルスチェック**（本番向け）：
    *   ライブネス: `GET /api/cloud/health/live`（200 OK）
    *   レディネス: `GET /api/cloud/health/ready`（200 OK）
    *   ディープチェック: `GET /api/cloud/health`（プロバイダー接続を含む）

### ネットワークセキュリティと認証バイパス

<details>
<summary>信頼プロキシと内部ネットワークバイパスの設定（設定ファイルが必要）</summary>

企業環境（VPN、プライベート VPC）では、内部トラフィックを信頼したり、認証をゲートウェイに委譲したい場合があります。

**simpletuner-enterprise.yaml:**

```yaml
network:
  # ロードバランサのヘッダーを信頼（例: AWS ALB, Nginx）
  trust_proxy_headers: true
  trusted_proxies:
    - "10.0.0.0/8"
    - "192.168.0.0/16"

  # 任意: 特定の内部サブネットはログインをバイパス
  bypass_auth_for_internal: true
  internal_networks:
    - "10.10.0.0/16"  # VPN クライアント

auth:
  # ヘルスチェックは常に認証不要
  bypass_paths:
    - "/health"
    - "/api/cloud/health"
    - "/api/cloud/metrics/prometheus"
```

</details>

### ロードバランサと TLS 設定

SimpleTuner は TLS 終端のために上流のリバースプロキシを想定します。

<details>
<summary>nginx リバースプロキシ例</summary>

```nginx
server {
    listen 443 ssl http2;
    server_name trainer.internal;

    ssl_certificate /etc/ssl/certs/simpletuner.crt;
    ssl_certificate_key /etc/ssl/private/simpletuner.key;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket 対応（リアルタイムログ/SSE）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

</details>

### 可観測性（Prometheus & Logging）

**メトリクス：**
`GET /api/cloud/metrics/prometheus` をスクレイプして運用指標を取得します。
*   `simpletuner_jobs_active`: 現在のキュー深さ
*   `simpletuner_cost_total_usd`: 支出トラッキング
*   `simpletuner_uptime_seconds`: 稼働時間

**ログ：**
`SIMPLETUNER_LOG_FORMAT=json` を設定すると Splunk/Datadog/ELK で扱いやすくなります。

<details>
<summary>データ保持期間の設定</summary>

コンプライアンス要件に合わせて保持期間を環境変数で設定します：

| 変数 | デフォルト | 説明 |
|----------|---------|-------------|
| `SIMPLETUNER_JOB_RETENTION_DAYS` | 90 | 完了ジョブの保持日数 |
| `SIMPLETUNER_AUDIT_RETENTION_DAYS` | 90 | 監査ログの保持日数 |

```bash
# SOC 2 / HIPAA: 1 年保持
export SIMPLETUNER_JOB_RETENTION_DAYS=365
export SIMPLETUNER_AUDIT_RETENTION_DAYS=365

# 自動クリーンアップを無効化（手動管理）
export SIMPLETUNER_JOB_RETENTION_DAYS=0
```

`0` にすると自動クリーンアップが無効になります。クリーンアップは毎日実行されます。

</details>

---


## 2. アイデンティティとアクセス管理（SSO）

SimpleTuner は Okta、Azure AD、Keycloak、Active Directory 向けに OIDC（OpenID Connect）と LDAP をサポートします。

### プロバイダー設定

**Web UI:** **Administration → Auth** に移動して追加/設定します。

**API:** curl 例は [API Cookbook](#4-api-cookbook) を参照してください。

<details>
<summary>設定ファイルからの設定（IaC/GitOps 向け）</summary>

`simpletuner-enterprise.yaml` に追加します：

```yaml
oidc:
  enabled: true
  provider: "okta"  # または "azure_ad", "google"

  client_id: "0oa1234567890abcdef"
  client_secret: "${OIDC_CLIENT_SECRET}"
  issuer_url: "https://your-org.okta.com/oauth2/default"

  scopes: ["openid", "email", "profile", "groups"]

  # ID プロバイダーのグループを SimpleTuner のロールへマップ
  role_mapping:
    claim: "groups"
    admin_groups: ["ML-Platform-Admins"]
    user_groups: ["ML-Researchers"]
```

</details>

<details>
<summary>ワーカー間での OAuth State 検証</summary>

マルチワーカー構成（ロードバランサ配下の複数 Gunicorn ワーカーなど）で OIDC 認証を行う場合、OAuth state の検証は全ワーカー間で共有される必要があります。SimpleTuner は OAuth state を DB に保存することで自動的に対応します。

**仕組み：**

1. **State 生成**: OIDC ログイン開始時に暗号学的にランダムな state トークンを生成し、プロバイダー名、リダイレクト URI、10 分の有効期限と共に DB に保存します。

2. **State 検証**: コールバックが別ワーカーに到達しても、state を DB から取得し、原子的に消費します（1 回限り）。

3. **クリーンアップ**: 期限切れの state は通常運用の中で自動削除されます。

追加の設定は不要です。OAuth state はジョブ/ユーザーと同じ DB に保存されます。

**"Invalid OAuth state" が出る場合：**
1. ログイン開始から 10 分以内にコールバックが来ているか確認
2. すべてのワーカーが同じ DB を使用しているか確認
3. DB の書き込み権限を確認
4. ログ内に "Failed to store OAuth state" がないか確認

</details>

### ユーザー管理とロール

SimpleTuner は階層的なロールシステムを採用しています。ユーザーは `GET/POST /api/users` で管理できます。

| ロール | 優先度 | 説明 |
|------|----------|-------------|
| **Viewer** | 10 | ジョブ履歴とログの閲覧のみ。 |
| **Researcher** | 20 | 標準アクセス。ジョブ送信と自分の API キー管理が可能。 |
| **Lead** | 30 | 保留中ジョブの承認とチームの使用状況閲覧。 |
| **Admin** | 100 | ユーザー管理やルール設定を含むフル権限。 |

---


## 3. ガバナンスと運用

### 承認ワークフロー

特定条件で承認が必要になるようにし、コストやリソース使用を制御します。ルールは送信時に評価されます。

**フロー：**
1.  ユーザーがジョブを送信 → ステータスが `pending_approval` になる
2.  Lead が `GET /api/approvals/requests` を確認
3.  Lead が `POST /.../approve` または `reject`
4.  ジョブは自動的にキューへ進むかキャンセルされる

<details>
<summary>承認ルールエンジン</summary>

ルールエンジンは、ジョブ送信を設定済みルールに照らして評価します。ルールは優先順に処理され、最初に一致したルールが承認要件をトリガーします。

**利用可能な条件：**

| 条件 | 説明 |
|-----------|-------------|
| `cost_exceeds` | 予測コストがしきい値（USD）を超えると発火 |
| `hardware_type` | ハードウェア種別をマッチ（glob パターン） |
| `daily_jobs_exceed` | ユーザーの 1 日ジョブ数がしきい値超過で発火 |
| `first_job` | ユーザーの初回ジョブで発火 |
| `config_pattern` | 設定名パターンに一致 |
| `provider` | 特定プロバイダー名に一致 |

**例: $50 を超えるジョブは承認必須**

```bash
curl -X POST http://localhost:8080/api/approvals/rules \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "high_cost",
    "condition": "cost_exceeds",
    "threshold": "50",
    "required_approver_level": "lead",
    "exempt_levels": ["admin"]
  }'
```

`exempt_levels` で特定ユーザーの承認を免除できます。また `applies_to_provider`/`applies_to_level` でルールの適用範囲を限定できます。

</details>

<details>
<summary>メールベース承認（IMAP ワークフロー）</summary>

メールでの承認が望ましいチーム向けに、SimpleTuner は IMAP IDLE を用いた承認をサポートします。

**動作概要：**
1. ジョブ送信で承認が必要になる
2. 承認者へ通知メールが送信される（固有トークン付き）
3. IMAP ハンドラが IDLE で受信箱を監視
4. 承認者が "approve"/"reject"（または `yes`、`lgtm`、`+1` など）で返信
5. システムが返信を解析して承認を処理

**Administration → Notifications** または API で設定します。返信トークンは 24 時間で期限切れ、かつ一度きりです。

</details>

### ジョブキューと同時実行

スケジューラはリソースを公平に利用できるよう管理します。詳細は [専用ドキュメント](../../JOB_QUEUE.md) を参照してください。

*   **優先度:** Admin > Lead > Researcher > Viewer
*   **同時実行:** 全体およびユーザー単位で制限
    *   UI で更新: **Cloud タブ → Job Queue パネル**（管理者のみ）
    *   API で更新: `POST /api/queue/concurrency` に `{"max_concurrent": 10, "user_max_concurrent": 3}`

### ジョブ状態ポーリング（Webhook 不要）

Webhook を公開できない環境向けに、SimpleTuner はバックグラウンドポーラーを提供します。

`simpletuner-enterprise.yaml` に追加：

```yaml
background:
  job_status_polling:
    enabled: true
    interval_seconds: 30
```

このサービスは 30 秒ごとにプロバイダー API を問い合わせ、ローカル DB を更新し、SSE 経由で UI にリアルタイム反映します。

### API キーローテーション

クラウドプロバイダー資格情報の安全な管理については、**API Cookbook** と [Cloud Training](../cloud/README.md) の詳細を参照してください。

---


## 4. API Cookbook

<details>
<summary>OIDC/LDAP 設定例</summary>

**Keycloak（OIDC）：**
```bash
curl -X POST http://localhost:8080/api/cloud/external-auth/providers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "keycloak",
    "provider_type": "oidc",
    "enabled": true,
    "config": {
      "issuer": "https://keycloak.example.com/realms/ml-training",
      "client_id": "simpletuner",
      "client_secret": "your-client-secret",
      "scopes": ["openid", "email", "profile", "roles"],
      "roles_claim": "realm_access.roles"
    }
  }'
```

**LDAP / Active Directory：**
```bash
curl -X POST http://localhost:8080/api/cloud/external-auth/providers \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "name": "corporate-ad",
    "provider_type": "ldap",
    "enabled": true,
    "level_mapping": {
      "CN=ML-Admins,OU=Groups,DC=corp,DC=com": ["admin"]
    },
    "config": {
      "server": "ldaps://ldap.corp.com:636",
      "base_dn": "DC=corp,DC=com",
      "bind_dn": "CN=svc-simpletuner,OU=Service Accounts,DC=corp,DC=com",
      "bind_password": "service-account-password",
      "user_search_filter": "(sAMAccountName={username})",
      "use_ssl": true
    }
  }'
```

</details>

<details>
<summary>ユーザー管理の例</summary>

**Researcher を作成：**
```bash
curl -X POST http://localhost:8080/api/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "email": "researcher@company.com",
    "username": "jsmith",
    "password": "secure_password_123",
    "level_names": ["researcher"]
  }'
```

**カスタム権限の付与：**
```bash
curl -X POST http://localhost:8080/api/users/123/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"permission_name": "admin.approve", "granted": true}'
```

</details>

<details>
<summary>資格情報管理</summary>

SimpleTuner には API 資格情報の追跡、ローテーション、監査のためのライフサイクル管理が含まれます。

**資格情報の解決順:** ジョブ送信時、ユーザー単位の資格情報が優先され、無い場合はグローバル資格情報（環境変数）にフォールバックします。

| シナリオ | ユーザー単位 | グローバル | 動作 |
|----------|--------|--------|----------|
| **共有組織キー** | ❌ | ✅ | 全ユーザーが組織の API キーを共有 |
| **BYOK** | ✅ | ❌ | 各ユーザーが自分のキーを提供 |
| **ハイブリッド** | 一部 | ✅ | キーがあるユーザーはそれを使用、他はグローバル |

**ローテーション:** **Admin > Auth** → ユーザー → **Manage Credentials** → **Rotate**。90 日以上更新がない資格情報は警告バッジが表示されます。

</details>

#### 外部オーケストレーション {#external-orchestration-airflow}

<details>
<summary>Airflow の例</summary>

```python
def submit_and_wait(job_config, provider="replicate", **context):
    resp = requests.post(
        f"http://localhost:8080/api/cloud/{provider}/submit",
        json=job_config,
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    job_id = resp.json()["job_id"]

    while True:
        status = requests.get(f"http://localhost:8080/api/cloud/jobs/{job_id}")
        state = status.json()["status"]
        if state in ("completed", "failed", "cancelled"):
            return status.json()
        time.sleep(30)
```

</details>

---


## 5. トラブルシューティング

**ヘルスチェック失敗**
*   `503 Service Unavailable`: DB 接続を確認。
*   `Degraded`: オプションコンポーネント（クラウド API など）が未接続/未設定の可能性。

**認証の問題**
*   **OIDC リダイレクトループ:** `issuer_url` が完全一致しているか確認（末尾スラッシュに注意）。
*   **内部認証バイパス:** ログに "Auth bypassed for IP..." が出ているか確認。ロードバランサが正しい `X-Real-IP` を渡しているか確認。

**ジョブ更新が止まる**
*   Webhook が使えない場合、`simpletuner-enterprise.yaml` の **Job Status Polling** を有効化。
*   `GET /api/cloud/metrics/prometheus` の `simpletuner_jobs_active` を確認。

**メトリクスが出ない**
*   Prometheus が `/api/cloud/metrics/prometheus` をスクレイプしているか確認（`/metrics` だけではありません）。

---


## 6. 組織とチームクォータ

SimpleTuner は階層的な組織/チーム構造と天井型クォータ（ceiling）をサポートします。

### 階層

```
Organization (quota ceiling)
    └── Team (quota ceiling, bounded by org)
         └── User (limit, bounded by team and org)
```

### 天井モデル

クォータは天井モデルで運用されます：
- **組織クォータ**: すべてのメンバーに対する絶対上限
- **チームクォータ**: チームメンバーの上限（組織上限を超えられない）
- **ユーザー/レベルクォータ**: 個別制限（チーム/組織上限に従う）

**例：**
- 組織上限: 同時ジョブ 100
- チーム上限: 同時ジョブ 20
- ユーザー制限: 同時ジョブ 50 → **実効は 20**（チーム上限が優先）

**適用ルール：**
- チームクォータは設定時に検証: 組織上限を超える設定は HTTP 400
- ユーザークォータは実行時に検証: 有効上限はユーザー/チーム/組織の最小値
- 組織上限を下げても既存チーム上限は自動調整されない（管理者が手動更新）

<details>
<summary>API 例</summary>

**組織を作成：**
```bash
curl -X POST http://localhost:8080/api/orgs \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "ML Research", "slug": "ml-research"}'
```

**組織クォータ上限を設定：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/quotas \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"quota_type": "concurrent_jobs", "limit_value": 100, "action": "block"}'
```

**チームを作成：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"name": "NLP Team", "slug": "nlp"}'
```

**ユーザーをチームに追加：**
```bash
curl -X POST http://localhost:8080/api/orgs/1/teams/1/members \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"user_id": 123, "role": "member"}'
```

</details>

### クォータとコスト上限のアクション

クォータやコスト上限に達すると、設定された `action` に応じて動作が変わります：

| アクション | 挙動 |
|--------|----------|
| `warn` | 警告を出してジョブを実行 |
| `block` | ジョブ送信を拒否 |
| `notify` | ジョブ実行は続行し、管理者に通知 |

<details>
<summary>コスト上限の設定</summary>

コスト上限はプロバイダーごとに **Cloud タブ → Settings** または API で設定できます：

```bash
curl -X PUT http://localhost:8080/api/cloud/providers/<provider>/config \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "cost_limit_enabled": true,
    "cost_limit_amount": 500.00,
    "cost_limit_period": "monthly",
    "cost_limit_action": "warn"
  }'
```

ステータス確認: `GET /api/cloud/metrics/cost-limit-status`

</details>

---


## 7. 制限事項

### ワークフロー/パイプラインジョブ（DAG）

SimpleTuner はジョブ依存関係や、あるジョブの出力が次のジョブに渡るような多段ワークフローをサポートしません。各クラウドジョブは独立しています。

**推奨:** Airflow、Prefect、Dagster などの外部オーケストレーションツールで REST API を通じて連携してください。上の [Airflow 例](#external-orchestration-airflow) を参照。

### 学習ランの再開

中断/失敗/キャンセルされた学習の再開機能はありません。クラウドジョブはチェックポイントから自動復旧しません。

**回避策：**
- `--push_checkpoints_to_hub` を有効化して中間状態を保存
- 出力をダウンロードし、それを新しいジョブの開始点として使う
- 重要なワークロードでは長時間ジョブを分割

<details>
<summary>UI 機能リファレンス</summary>

| 機能 | UI の場所 | API |
|---------|-------------|-----|
| 組織とチーム | Administration → Orgs | `/api/orgs` |
| クォータ | Administration → Quotas | `/api/orgs/{id}/quotas` |
| OIDC/LDAP | Administration → Auth | `/api/cloud/external-auth/providers` |
| ユーザー | Administration → Users | `/api/users` |
| 監査ログ | Sidebar → Audit Log | `/api/audit` |
| キュー | Cloud タブ → Job Queue | `/api/queue/concurrency` |
| 承認 | Administration → Approvals | `/api/approvals/requests` |

認証未設定（シングルユーザー）または管理者権限の場合に Administration セクションが表示されます。

</details>

<details>
<summary>エンタープライズのオンボーディングフロー</summary>

Admin パネルには、認証 → 組織 → チーム → クォータ → 資格情報の順に設定するガイド付きオンボーディングがあります。

| ステップ | 機能 |
|------|---------|
| 1 | 認証（OIDC/LDAP） |
| 2 | 組織 |
| 3 | チーム |
| 4 | クォータ |
| 5 | 資格情報 |

各ステップは完了/スキップ可能で、状態はブラウザの localStorage に保存されます。

</details>

---


## 8. 通知システム

SimpleTuner には、ジョブ状態、承認、クォータ、システムイベント向けのマルチチャネル通知システムがあります。

| チャネル | 用途 |
|---------|----------|
| **Email** | 承認依頼、ジョブ完了（SMTP/IMAP） |
| **Webhook** | CI/CD 連携（JSON + HMAC 署名） |
| **Slack** | チーム通知（Incoming Webhook） |

**Administration → Notifications** または API で設定します。

<details>
<summary>イベント種別</summary>

| カテゴリ | イベント |
|----------|--------|
| Approval | `approval.required`, `approval.granted`, `approval.rejected`, `approval.expired` |
| Job | `job.submitted`, `job.started`, `job.completed`, `job.failed`, `job.cancelled` |
| Quota | `quota.warning`, `quota.exceeded`, `cost.warning`, `cost.exceeded` |
| System | `system.provider_error`, `system.provider_degraded`, `system.webhook_failure` |
| Auth | `auth.login_failure`, `auth.new_device` |

</details>

<details>
<summary>チャネル設定例</summary>

**Email:**
```bash
curl -X POST http://localhost:8080/api/cloud/notifications/channels \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "channel_type": "email",
    "name": "Primary Email",
    "smtp_host": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_use_tls": true
  }'
```

**Slack:**
```bash
curl -X POST http://localhost:8080/api/cloud/notifications/channels \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{
    "channel_type": "slack",
    "name": "Training Alerts",
    "webhook_url": "https://hooks.slack.com/services/T00/B00/xxxx"
  }'
```

**Webhook:** ペイロードは HMAC-SHA256 で署名（`X-SimpleTuner-Signature` ヘッダー）。

</details>

---


## 9. リソースルール

リソースルールは、設定、ハードウェア種別、出力パスに対して glob パターンで細かなアクセス制御を行います。

| タイプ | パターン例 |
|------|----------------|
| `config` | `team-x-*`, `production-*` |
| `hardware` | `gpu-a100*`, `*-80gb` |
| `provider` | `replicate`, `runpod` |

ルールは **allow/deny** と「最も許可的なルールが優先」のロジックで処理されます。**Administration → Rules** から設定します。

<details>
<summary>ルール例</summary>

**チーム分離:** Researcher は "team-x-" で始まる設定のみ利用可
```
Level: researcher
Rules:
  - config: "team-x-*" → allow
  - config: "*" → deny
```

**ハードウェア制限:** Researcher は T4/V100 のみ、Lead は全て可
```
Level: researcher → hardware: "gpu-t4*" allow, "gpu-v100*" allow
Level: lead → hardware: "*" allow
```

</details>

---


## 10. 権限マトリクス

<details>
<summary>完全な権限マトリクス</summary>

### ジョブ権限

| 権限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `job.submit` | | ✓ | ✓ | ✓ |
| `job.view.own` | ✓ | ✓ | ✓ | ✓ |
| `job.view.all` | | | ✓ | ✓ |
| `job.cancel.own` | | ✓ | ✓ | ✓ |
| `job.cancel.all` | | | | ✓ |
| `job.priority.high` | | | ✓ | ✓ |
| `job.bypass.queue` | | | | ✓ |
| `job.bypass.approval` | | | | ✓ |

### 設定権限

| 権限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `config.view` | ✓ | ✓ | ✓ | ✓ |
| `config.create` | | ✓ | ✓ | ✓ |
| `config.edit.own` | | ✓ | ✓ | ✓ |
| `config.edit.all` | | | | ✓ |

### 管理権限

| 権限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `admin.users` | | | | ✓ |
| `admin.approve` | | | ✓ | ✓ |
| `admin.audit` | | | ✓ | ✓ |
| `admin.config` | | | | ✓ |
| `queue.approve` | | | ✓ | ✓ |
| `queue.manage` | | | | ✓ |

### 組織/チーム権限

| 権限 | Viewer | Researcher | Lead | Admin |
|------------|:------:|:----------:|:----:|:-----:|
| `org.view` | | | ✓ | ✓ |
| `org.create` | | | | ✓ |
| `team.view` | | | ✓ | ✓ |
| `team.create` | | | ✓ | ✓ |
| `team.manage.members` | | | ✓ | ✓ |

</details>

**権限オーバーライド:** 個別ユーザーに対して **Administration → Users → Permission Overrides** で付与/拒否できます。
