# 外部認証（OIDC & LDAP）

SimpleTuner は外部 ID プロバイダーによるシングルサインオン（SSO）をサポートします。OIDC プロバイダー（Keycloak、Auth0、Okta、Azure AD、Google）または LDAP/Active Directory を利用できます。

## 概要

外部認証の利点：
- **シングルサインオン**: 既存の社内資格情報で認証
- **自動プロビジョニング**: 初回ログイン時にユーザーを自動作成
- **レベルマッピング**: 外部グループ/ロールを SimpleTuner のアクセスレベルに対応付け
- **複数プロバイダー**: 複数のプロバイダーを同時に設定可能

## OIDC の設定

### 対応プロバイダー

- Keycloak
- Auth0
- Okta
- Azure Active Directory
- Google Workspace
- OpenID Connect 準拠プロバイダー

OpenLDAP + Dex、Keycloak、Auth0 で動作確認済みです。

### Web UI での設定

1. **Administration > Manage Users > Auth Providers** に移動
2. **Add Provider** をクリックして **OIDC Provider** を選択
3. プロバイダー情報を入力：
   - **Provider Name**: 一意の識別子（例: `corporate-sso`）
   - **Type**: OIDC
   - **Issuer URL**: OIDC ディスカバリー URL（例: `https://auth.example.com/.well-known/openid-configuration`）
   - **Client ID**: ID プロバイダーで取得した値
   - **Client Secret**: ID プロバイダーで取得した値
   - **Scopes**: 通常 `openid profile email`

### API での設定

```bash
curl -X POST http://localhost:8001/api/cloud/external-auth/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "corporate-sso",
    "provider_type": "oidc",
    "enabled": true,
    "auto_create_users": true,
    "default_levels": ["researcher"],
    "config": {
      "client_id": "your-client-id",
      "client_secret": "your-client-secret",
      "discovery_url": "https://auth.example.com/.well-known/openid-configuration",
      "scopes": ["openid", "profile", "email"]
    }
  }'
```

### OIDC コールバック URL

ID プロバイダーには次のコールバック URL を設定してください：

```
https://your-simpletuner-host/api/cloud/external-auth/oidc/{provider-name}/callback
```

例えばプロバイダー名が `corporate-sso` の場合：
```
https://simpletuner.example.com/api/cloud/external-auth/oidc/corporate-sso/callback
```

### OIDC ログインフロー

1. ログインページで「Login with SSO」をクリック
2. ブラウザが ID プロバイダーにリダイレクト
3. 社内資格情報で認証
4. ID プロバイダーが SimpleTuner のコールバックにリダイレクト
5. SimpleTuner がローカルユーザーを作成/更新してセッション開始

## LDAP の設定

### 対応構成

- Microsoft Active Directory
- OpenLDAP
- 389 Directory Server
- LDAPv3 準拠サーバー

FreeIPA と OpenLDAP で動作確認済みです。

### Web UI での設定

1. **Administration > Manage Users > Auth Providers** に移動
2. **Add Provider** をクリックして **LDAP / Active Directory** を選択
3. LDAP 詳細を入力：
   - **Provider Name**: 一意の識別子（例: `company-ldap`）
   - **Type**: LDAP
   - **Server URL**: LDAP サーバー（例: `ldap://ldap.example.com:389` または `ldaps://ldap.example.com:636`）
   - **Bind DN**: クエリ用サービスアカウント DN
   - **Bind Password**: サービスアカウントのパスワード
   - **User Base DN**: ユーザー検索のベース DN
   - **User Filter**: ユーザー検索の LDAP フィルタ
   - **Group Base DN**: グループ検索のベース DN（任意）
   - **Group Filter**: グループ検索の LDAP フィルタ（任意）

### API での設定

```bash
curl -X POST http://localhost:8001/api/cloud/external-auth/providers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "company-ldap",
    "provider_type": "ldap",
    "enabled": true,
    "auto_create_users": true,
    "default_levels": ["researcher"],
    "level_mapping": {
      "CN=SimpleTuner-Admins,OU=Groups,DC=example,DC=com": ["admin"],
      "CN=SimpleTuner-Users,OU=Groups,DC=example,DC=com": ["researcher"]
    },
    "config": {
      "server_url": "ldap://ldap.example.com:389",
      "use_tls": false,
      "bind_dn": "CN=simpletuner-svc,OU=Services,DC=example,DC=com",
      "bind_password": "service-account-password",
      "user_base_dn": "OU=Users,DC=example,DC=com",
      "user_filter": "(sAMAccountName={username})",
      "group_base_dn": "OU=Groups,DC=example,DC=com",
      "group_filter": "(member={user_dn})",
      "username_attribute": "sAMAccountName",
      "email_attribute": "mail",
      "display_name_attribute": "displayName"
    }
  }'
```

### LDAP ログイン

LDAP ログインはユーザー名/パスワード方式です。ユーザーは SimpleTuner のログイン画面で LDAP 資格情報を入力します。

```bash
# API ログイン
curl -X POST http://localhost:8001/api/cloud/external-auth/ldap/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jsmith",
    "password": "ldap-password"
  }'
```

## レベルマッピング

外部グループ/ロールを SimpleTuner のアクセスレベルに対応付けます：

```json
{
  "level_mapping": {
    "admin": ["admin"],
    "power-users": ["advanced"],
    "default": ["researcher"]
  }
}
```

OIDC は `groups` または `roles` クレームの値を使用します。
LDAP は LDAP グループ DN を使用します。

## ユーザープロビジョニング

### 自動プロビジョニング

`auto_create_users` が有効（デフォルト）だと、初回ログイン時にユーザーが自動作成されます：

```json
{
  "auto_create_users": true,
  "default_levels": ["researcher"]
}
```

### 手動プロビジョニング

`auto_create_users: false` にすると、ユーザーは事前に SimpleTuner 上に作成されている必要があります。

## API エンドポイント

| メソッド | エンドポイント | 説明 |
|--------|----------|-------------|
| GET | `/api/cloud/external-auth/providers` | 設定済みプロバイダー一覧 |
| POST | `/api/cloud/external-auth/providers` | プロバイダー作成 |
| PATCH | `/api/cloud/external-auth/providers/{name}` | プロバイダー更新 |
| DELETE | `/api/cloud/external-auth/providers/{name}` | プロバイダー削除 |
| GET | `/api/cloud/external-auth/providers/{name}/test` | プロバイダー接続テスト |
| GET | `/api/cloud/external-auth/oidc/{provider}/start` | OIDC フロー開始 |
| GET | `/api/cloud/external-auth/oidc/callback` | OIDC コールバック |
| POST | `/api/cloud/external-auth/ldap/login` | LDAP ログイン |
| GET | `/api/cloud/external-auth/available` | 利用可能プロバイダー一覧（公開） |

## CLI コマンド

```bash
# プロバイダー一覧（管理者権限が必要）
simpletuner auth users auth-status

# 外部認証は Web UI または API で管理します
# 専用 CLI コマンドはまだありません
```

## アーキテクチャメモ

### OAuth State の永続化

OIDC 認証では CSRF 対策のため `state` パラメータの検証が必要です。SimpleTuner はインメモリではなくデータベースに OAuth state を保存し、次に対応します。

- **マルチワーカー構成**: ロードバランサー環境でコールバックが別ワーカーに到達する
- **サーバー再起動**: 認証ウィンドウ中の state を保持
- **水平スケール**: スティッキーセッション不要

state トークンは 10 分で期限切れとなり、検証成功後に削除されます。

## セキュリティ上の注意

### OIDC

- クライアントシークレットは安全に保管（環境変数またはシークレット管理）
- コールバック URL は必ず HTTPS を使用
- `state` パラメータの検証で CSRF を防止（DB バックの state で自動対応）
- ID プロバイダー側で許可リダイレクト URI を制限することを推奨

### LDAP

- 本番では LDAPS（636）または StartTLS を使用
- 最小権限のサービスアカウントを利用
- サービスアカウントの資格情報を定期的にローテーション
- 高トラフィック環境では接続プールを検討

### 一般

- 外部認証はローカルのパスワードポリシーをバイパスします
- 外部認証ユーザーは SimpleTuner でパスワードをリセットできません
- 認証試行（失敗含む）は監査ログに記録されます
- コンプライアンス要件に応じてセッションタイムアウトを検討

## トラブルシューティング

### OIDC の問題

**"Invalid or expired state"**
- state トークンは 10 分で期限切れ
- SimpleTuner と ID プロバイダー間の時刻同期を確認

**"Discovery URL unreachable"**
- ID プロバイダーへのネットワーク接続を確認
- HTTPS のアウトバウンドが許可されているか確認

**"Invalid client_id or client_secret"**
- ID プロバイダーの資格情報を再確認
- 設定に余計な空白が入っていないか確認

### LDAP の問題

**"Connection refused"**
- サーバー URL とポートを確認
- LDAP サービスが稼働しているか確認
- ファイアウォールを確認

**"Invalid credentials"**
- bind DN の形式を確認（完全 DN が必要）
- bind パスワードを確認
- `ldapsearch` でテスト

**"User not found"**
- user_base_dn を確認
- user_filter の構文を確認
- ユーザーが存在し無効化されていないか確認

**"No groups returned"**
- group_base_dn を確認
- group_filter の構文を確認
- グループメンバーシップ属性が正しいか確認

## 設定例

### Keycloak

```json
{
  "name": "keycloak",
  "provider_type": "oidc",
  "config": {
    "client_id": "simpletuner",
    "client_secret": "your-secret",
    "discovery_url": "https://keycloak.example.com/realms/myrealm/.well-known/openid-configuration",
    "scopes": ["openid", "profile", "email", "groups"]
  }
}
```

### Active Directory

```json
{
  "name": "active-directory",
  "provider_type": "ldap",
  "config": {
    "server_url": "ldaps://dc01.example.com:636",
    "use_tls": true,
    "bind_dn": "CN=simpletuner,OU=Service Accounts,DC=example,DC=com",
    "bind_password": "service-password",
    "user_base_dn": "OU=Users,DC=example,DC=com",
    "user_filter": "(sAMAccountName={username})",
    "group_base_dn": "OU=Groups,DC=example,DC=com",
    "group_filter": "(member={user_dn})",
    "username_attribute": "sAMAccountName",
    "email_attribute": "mail",
    "display_name_attribute": "displayName"
  },
  "level_mapping": {
    "CN=ST-Admins,OU=Groups,DC=example,DC=com": ["admin"],
    "CN=ST-Advanced,OU=Groups,DC=example,DC=com": ["advanced"],
    "CN=ST-Users,OU=Groups,DC=example,DC=com": ["researcher"]
  }
}
```

### Azure AD（OIDC）

```json
{
  "name": "azure-ad",
  "provider_type": "oidc",
  "config": {
    "client_id": "your-app-id",
    "client_secret": "your-client-secret",
    "discovery_url": "https://login.microsoftonline.com/your-tenant-id/v2.0/.well-known/openid-configuration",
    "scopes": ["openid", "profile", "email"]
  }
}
```
