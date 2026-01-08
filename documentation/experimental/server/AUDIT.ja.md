# 監査ログ

SimpleTuner の監査ログシステムは、セキュリティに関連するイベントの改ざん検出可能な記録を提供します。管理操作、認証イベント、ジョブ操作はすべて暗号学的なチェーン検証付きで記録されます。

## 概要

監査ログには次が含まれます：
- **認証イベント**: ログイン試行（成功/失敗）、ログアウト、セッション期限切れ
- **ユーザー管理**: ユーザー作成、更新、削除、権限変更
- **API キー操作**: キー作成、失効、使用
- **資格情報管理**: プロバイダー資格情報の変更
- **ジョブ操作**: 送信、キャンセル、承認

## 監査ログへのアクセス

### Web UI

管理パネルの **Audit** タブで、フィルタ付きで監査エントリを閲覧できます。

### CLI

```bash
# 最近の監査エントリを一覧
simpletuner auth audit list

# イベント種別でフィルタ
simpletuner auth audit list --event-type auth.login.failed

# ユーザーでフィルタ
simpletuner auth audit user 123

# セキュリティイベントのみ
simpletuner auth audit security

# 統計情報
simpletuner auth audit stats

# チェーン整合性の検証
simpletuner auth audit verify
```

### API エンドポイント

すべてのエンドポイントは `admin.audit` 権限が必要です。

| メソッド | エンドポイント | 説明 |
|--------|----------|-------------|
| GET | `/api/audit` | フィルタ付きの監査エントリ一覧 |
| GET | `/api/audit/stats` | 監査統計の取得 |
| GET | `/api/audit/types` | 利用可能なイベント種別一覧 |
| GET | `/api/audit/verify` | チェーン整合性の検証 |
| GET | `/api/audit/user/{user_id}` | 指定ユーザーのエントリ取得 |
| GET | `/api/audit/security` | セキュリティ関連イベント取得 |

## イベント種別

### 認証イベント

| イベント | 説明 |
|-------|-------------|
| `auth.login.success` | ログイン成功 |
| `auth.login.failed` | ログイン失敗 |
| `auth.logout` | ログアウト |
| `auth.session.expired` | セッション期限切れ |
| `auth.api_key.used` | API キーが使用された |

### ユーザー管理イベント

| イベント | 説明 |
|-------|-------------|
| `user.created` | 新規ユーザー作成 |
| `user.updated` | ユーザー情報更新 |
| `user.deleted` | ユーザー削除 |
| `user.password.changed` | パスワード変更 |
| `user.level.changed` | レベル/ロール変更 |
| `user.permission.changed` | 権限変更 |

### API キーイベント

| イベント | 説明 |
|-------|-------------|
| `api_key.created` | 新しい API キー作成 |
| `api_key.revoked` | API キー失効 |

### 資格情報イベント

| イベント | 説明 |
|-------|-------------|
| `credential.created` | プロバイダー資格情報を追加 |
| `credential.deleted` | プロバイダー資格情報を削除 |
| `credential.used` | 資格情報が使用された |

### ジョブイベント

| イベント | 説明 |
|-------|-------------|
| `job.submitted` | ジョブがキューに送信された |
| `job.cancelled` | ジョブがキャンセルされた |
| `job.approved` | ジョブ承認が付与された |
| `job.rejected` | ジョブ承認が拒否された |

## クエリパラメータ

監査エントリ一覧では次でフィルタできます：

| パラメータ | 型 | 説明 |
|-----------|------|-------------|
| `event_type` | string | イベント種別でフィルタ |
| `actor_id` | int | 操作者ユーザー ID でフィルタ |
| `target_type` | string | 対象リソース種別でフィルタ |
| `target_id` | string | 対象リソース ID でフィルタ |
| `since` | ISO date | 開始タイムスタンプ |
| `until` | ISO date | 終了タイムスタンプ |
| `limit` | int | 最大件数（1-500、デフォルト 50） |
| `offset` | int | ページネーションのオフセット |

## チェーン整合性

各監査エントリには以下が含まれます：
- 内容の暗号学的ハッシュ
- 前エントリのハッシュ参照
- 単調時計によるタイムスタンプ

これによりハッシュチェーンが形成され、改ざんの検出が可能になります。整合性の確認には検証エンドポイントまたは CLI を使用します：

```bash
# チェーン全体を検証
simpletuner auth audit verify

# 特定範囲を検証
simpletuner auth audit verify --start-id 100 --end-id 200
```

検証では以下をチェックします：
1. 各エントリのハッシュが内容と一致する
2. 各エントリが前エントリのハッシュを正しく参照している
3. シーケンスに欠落がない

## 保持期間

監査ログは SimpleTuner のデータベースに保存されます。保持期間は環境に合わせて設定してください：

```bash
# 保持期間（日数）を設定する環境変数
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

古いエントリは、コンプライアンス要件に従ってアーカイブまたは削除できます。

## セキュリティ上の注意

- 監査ログは追記専用で、API から変更や削除はできません
- ログ閲覧には `admin.audit` 権限が必要です
- 失敗したログイン試行は IP アドレス付きで記録されます
- 本番環境では SIEM への転送を検討してください
