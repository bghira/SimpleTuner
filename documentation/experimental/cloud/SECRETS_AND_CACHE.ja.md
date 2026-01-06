# シークレットプロバイダーとキャッシュシステム

このドキュメントでは、クラウド学習向けの SimpleTuner のシークレット管理システムとキャッシュ基盤について説明します。

## 概要

SimpleTuner は複数のプロバイダーを階層的に連結したシークレット管理システムと、パフォーマンス最適化のためのインメモリキャッシュ層を使用します。

## シークレットプロバイダー

`SecretsManager` クラスは複数のシークレットプロバイダーを優先順位付きでチェーンします。シークレット取得時は、値が見つかるまで順番にチェックされます。

### 優先順位

1. **環境変数**（最優先）
2. **ファイルベースのシークレット**
3. **AWS Secrets Manager**
4. **HashiCorp Vault**（最下位）

この順序により、環境変数が常に他のソースを上書きできるため、次の用途に便利です。
- ローカル開発での上書き
- シークレット注入を行うコンテナ環境
- CI/CD パイプライン

### よく使うシークレットキー

<details>
<summary>シークレットキー定数</summary>

```python
REPLICATE_API_TOKEN  # Replicate API 認証
HF_TOKEN             # HuggingFace Hub トークン
CLOUD_WEBHOOK_SECRET # Webhook HMAC 検証用シークレット
```
</details>

---

## プロバイダー設定

### 1. 環境変数（デフォルト）

環境変数プロバイダーは常に利用可能で、他のプロバイダーより優先されます。

**キーの正規化ルール：**
- キーは大文字化
- ハイフン（`-`）をアンダースコア（`_`）へ変換
- ドット（`.`）をアンダースコア（`_`）へ変換

**任意のプレフィックス：**

<details>
<summary>Python 設定例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import EnvironmentSecretProvider

# プレフィックスなし
provider = EnvironmentSecretProvider()
# キー "replicate-api-token" -> REPLICATE_API_TOKEN

# プレフィックスあり
provider = EnvironmentSecretProvider(prefix="SIMPLETUNER")
# キー "replicate-api-token" -> SIMPLETUNER_REPLICATE_API_TOKEN
```
</details>

**使用例：**

<details>
<summary>環境変数でシークレットを設定</summary>

```bash
# 環境変数でシークレットを設定
export REPLICATE_API_TOKEN="r8_your_token_here"
export HF_TOKEN="hf_your_token_here"
export CLOUD_WEBHOOK_SECRET="your_webhook_secret"
```
</details>

### 2. ファイルベースのシークレット

シークレットは、権限制限を行った JSON または YAML ファイルに保存できます。

**デフォルトの場所（順にチェック）：**
- `~/.simpletuner/secrets.json`
- `~/.simpletuner/secrets.yaml`
- `~/.simpletuner/secrets.yml`

<details>
<summary>ファイル形式の例（JSON/YAML）</summary>

**ファイル形式（JSON）：**

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```

**ファイル形式（YAML）：**

```yaml
REPLICATE_API_TOKEN: r8_your_token_here
HF_TOKEN: hf_your_token_here
CLOUD_WEBHOOK_SECRET: your_webhook_secret
```
</details>

<details>
<summary>手動設定</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import FileSecretProvider

# カスタムファイルパスを使用
provider = FileSecretProvider(file_path="/path/to/secrets.json")
```
</details>

**セキュリティ注意:** API でシークレットを保存する場合、ファイルは `chmod 0o600`（所有者のみ読み書き）で作成されます。

### 3. AWS Secrets Manager

本番環境では、AWS Secrets Manager による安全で集中管理されたシークレット保管が利用できます。

<details>
<summary>環境変数</summary>

```bash
# 必須: AWS Secrets Manager のシークレット名
export SIMPLETUNER_AWS_SECRET_NAME="simpletuner/production"

# 任意: AWS リージョン（未設定の場合はデフォルト）
export AWS_DEFAULT_REGION="us-west-2"
```
</details>

<details>
<summary>AWS シークレット形式</summary>

AWS Secrets Manager のシークレットは JSON オブジェクトである必要があります：

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```
</details>

<details>
<summary>手動設定</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import AWSSecretsManagerProvider

provider = AWSSecretsManagerProvider(
    secret_name="simpletuner/production",
    region_name="us-west-2"
)
```
</details>

<details>
<summary>必要な AWS IAM 権限</summary>

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT:secret:simpletuner/*"
    }
  ]
}
```
</details>

### 4. HashiCorp Vault

HashiCorp Vault でシークレット管理を行う企業向けの設定です。

**必要な依存関係：**

```bash
pip install hvac
```

<details>
<summary>環境変数</summary>

```bash
# 必須: Vault サーバー URL
export VAULT_ADDR="https://vault.example.com:8200"

# 必須: Vault 認証トークン
export VAULT_TOKEN="s.your_vault_token"

# 任意: シークレットのパス（デフォルト: "simpletuner"）
export SIMPLETUNER_VAULT_PATH="simpletuner"
```
</details>

<details>
<summary>手動設定</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import HashiCorpVaultProvider

provider = HashiCorpVaultProvider(
    url="https://vault.example.com:8200",
    token="s.your_vault_token",
    path="simpletuner",
    mount_point="secret"  # KV マウントポイント（デフォルト: "secret"）
)
```
</details>

<details>
<summary>Vault KV 設定</summary>

```bash
# KV v2（推奨）
vault kv put secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"

# KV v1
vault write secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"
```

プロバイダーはまず KV v2 を試し、失敗した場合に v1 へフォールバックします。
</details>

---

## シークレットマネージャーの利用

### 基本的な使い方

<details>
<summary>シークレット取得</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import get_secrets_manager

# シングルトンを取得
secrets = get_secrets_manager()

# 任意のシークレットを取得
api_token = secrets.get("REPLICATE_API_TOKEN")
api_token = secrets.get("REPLICATE_API_TOKEN", default="fallback_value")

# 便利メソッド
replicate_token = secrets.get_replicate_token()
hf_token = secrets.get_hf_token()
webhook_secret = secrets.get_webhook_secret()
```
</details>

### シークレットの保存（ファイルプロバイダーのみ）

<details>
<summary>シークレットの設定と削除</summary>

```python
secrets = get_secrets_manager()

# シークレットを保存（~/.simpletuner/secrets.json へ書き込み）
secrets.set_secret("MY_CUSTOM_KEY", "my_value")

# Replicate トークンの便利メソッド
secrets.set_replicate_token("r8_new_token")

# シークレットを削除
secrets.delete_secret("MY_CUSTOM_KEY")
```
</details>

### カスタムプロバイダー設定

<details>
<summary>カスタム設定の例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

# シングルトンをリセット
SecretsManager.reset()

# カスタム設定で作成
secrets = SecretsManager(config={
    "file_path": "/custom/path/secrets.json",
    "aws_secret_name": "myapp/secrets",
    "aws_region": "eu-west-1",
    "vault_url": "https://vault.internal:8200",
    "vault_token": "s.token",
    "vault_path": "myapp/simpletuner"
})
```
</details>

---

## キャッシュシステム

SimpleTuner はインメモリの TTL（Time-To-Live）キャッシュを使用し、頻繁にアクセスされるデータの DB クエリを減らし、応答性を向上させます。

### TTLCache の特徴

- **`RLock` によるスレッド安全な操作**
- **キーごとの TTL** とデフォルト TTL の設定
- **期限切れエントリの自動クリーンアップ**
- **容量超過時の LRU 退避**
- **プレフィックス無効化**による一括クリア

### キャッシュ設定

<details>
<summary>TTLCache の初期化</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[str](
    default_ttl=300.0,      # デフォルト TTL 5分
    max_size=1000,          # 退避前の最大エントリ数
    cleanup_interval=60.0   # 60秒ごとに期限切れをクリーンアップ
)
```
</details>

### グローバルキャッシュインスタンス

SimpleTuner は 2 つのグローバルキャッシュを保持します：

| キャッシュ | デフォルト TTL | 最大サイズ | 目的 |
|-------|-------------|----------|---------|
| Provider Config Cache | 300秒（5分） | 100 | プロバイダー設定、Webhook URL、コスト制限 |
| User Permission Cache | 60秒（1分） | 500 | ユーザー権限（セキュリティのため短め） |

<details>
<summary>グローバルキャッシュへのアクセス</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    get_user_permission_cache
)

# グローバルキャッシュへアクセス
provider_cache = get_provider_config_cache()
user_cache = get_user_permission_cache()
```
</details>

### 基本キャッシュ操作

<details>
<summary>キャッシュ操作リファレンス</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[dict](default_ttl=120.0)

# 値を設定（デフォルト TTL）
cache.set("my_key", {"data": "value"})

# カスタム TTL で設定
cache.set("my_key", {"data": "value"}, ttl=60.0)

# 値を取得（期限切れ/未登録は None）
value = cache.get("my_key")

# 未登録時に計算
value = cache.get_or_set("my_key", lambda: compute_value())

# 非同期版
value = await cache.get_or_set_async("my_key", async_compute_value)

# 特定キーを削除
cache.delete("my_key")

# 全エントリをクリア
count = cache.clear()

# キャッシュ統計を取得
stats = cache.stats()
# {"size": 42, "max_size": 1000, "expired_count": 3, "default_ttl": 120.0}
```
</details>

---

## プロバイダーメタデータのキャッシュ

プロバイダー設定（Webhook URL、コスト制限、ハードウェア情報）は DB 負荷を減らすためにキャッシュされます。

### キャッシュキー形式

```
provider:{provider_name}:config
```

例: `provider:replicate:config`

### キャッシュの挙動

**読み取り時（`get_provider_config`）：**
1. `provider:{name}:config` をキャッシュで確認
2. 期限切れでなければ返却
3. 無ければ DB から読み込み
4. 5 分 TTL でキャッシュへ保存
5. 値を返却

**書き込み時（`save_provider_config`）：**
1. DB に書き込み
2. `provider:{name}:*` を無効化（プレフィックス無効化）

### キャッシュ無効化

<details>
<summary>無効化の例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    invalidate_provider_cache,
    invalidate_user_cache
)

# プロバイダーのキャッシュを全て無効化
invalidate_provider_cache("replicate")

# ユーザーのキャッシュを全て無効化
invalidate_user_cache(user_id=123)
```
</details>

### プレフィックス無効化

<details>
<summary>プレフィックス無効化の例</summary>

```python
cache = get_provider_config_cache()

# "provider:replicate:" で始まるキーを全て無効化
count = cache.invalidate_prefix("provider:replicate:")
```
</details>

---

## ハードウェア情報のキャッシュ

ハードウェア価格情報は別途キャッシュされ、手動で無効化できます。

### キャッシュの挙動

<details>
<summary>ハードウェア情報キャッシュ操作</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
    get_hardware_info,
    get_hardware_info_async,
    clear_hardware_info_cache,
    set_hardware_info
)

# ハードウェア情報を取得（キャッシュまたはデフォルトを使用）
info = get_hardware_info()
info = await get_hardware_info_async(store)

# キャッシュをクリアして再読み込みを強制
clear_hardware_info_cache()

# ハードウェア情報を手動設定（キャッシュ更新）
set_hardware_info({
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975}
})
```
</details>

### デフォルトのハードウェア価格

<details>
<summary>デフォルト値</summary>

```python
DEFAULT_HARDWARE_INFO = {
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
    "gpu-a100-large": {"name": "A100 (80GB)", "cost_per_second": 0.001400},
}
```
</details>

---

## デコレータベースのキャッシュ

関数結果のキャッシュには `@cached` デコレータを使用します：

<details>
<summary>cached デコレータの使用例</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import cached

@cached(ttl=120, key_prefix="mymodule")
def get_user(user_id: int) -> dict:
    # 高コストな DB クエリ
    return {"id": user_id, "name": "..."}

@cached(ttl=60)
async def fetch_external_data() -> dict:
    # 非同期 API 呼び出し
    return await api_client.get("/data")

# キャッシュキーは関数名と引数から自動生成:
# get_user(123) は "mymodule:get_user:123"

# 手動無効化のために内部キャッシュへアクセス
get_user._cache.clear()
```
</details>

---

## 資格情報の暗号化

SimpleTuner は API トークンなどの機密情報を Fernet 対称暗号で暗号化して保存します。ファイル権限に加えてセキュリティ層を追加します。

### 仕組み

1. **鍵導出**: PBKDF2-HMAC-SHA256（100,000 回）でマスターシークレットを導出
2. **暗号化**: Fernet（AES-128-CBC + HMAC）で暗号化
3. **保存**: 暗号化された値を base64 で DB に保存

### 鍵管理

暗号化鍵は次の優先順で取得します：

| 優先度 | ソース | 用途 |
|----------|--------|----------|
| 1 | `SIMPLETUNER_CREDENTIAL_KEY` 環境変数 | 本番環境、コンテナ環境 |
| 2 | `~/.simpletuner/credential.key` ファイル | ローカル開発、永続キー |
| 3 | 自動生成 | 初回セットアップ（ファイルに保存） |

<details>
<summary>環境変数でキーを設定</summary>

```bash
# 安全なキーを生成
export SIMPLETUNER_CREDENTIAL_KEY=$(openssl rand -base64 32)
```
</details>

**キーファイルの場所：**

```
~/.simpletuner/credential.key
```

キーファイルは `chmod 0600`（所有者のみ読み書き）で作成されます。

### 使用方法

資格情報の暗号化は、UI または API を介してプロバイダーの API トークンを保存する際に自動で行われます。直接呼び出す必要はありませんが、次の関数も利用できます：

<details>
<summary>暗号化/復号関数</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.auth.credential_encryption import (
    encrypt_credential,
    decrypt_credential,
)

# トークンを暗号化
encrypted = encrypt_credential("r8_my_api_token")

# 復号
plaintext = decrypt_credential(encrypted)
```
</details>

### 必要条件

資格情報の暗号化には `cryptography` パッケージが必要です：

```bash
pip install cryptography
```

未インストールの場合、資格情報は平文で保存されます（ログに警告）。

### キーのローテーション

暗号鍵をローテーションする手順：

1. 既存の資格情報をエクスポート（旧キーで復号されます）
2. 新しいキーを環境変数またはキーファイルに設定
3. サーバーを再起動
4. UI から資格情報を再入力（新しいキーで暗号化されます）

**注意:** 資格情報を再入力せずにキーを変更すると、既存の暗号化データが復号できなくなります。

### バックアップ時の注意

SimpleTuner をバックアップする場合：

- **含める:** `~/.simpletuner/credential.key`（または `SIMPLETUNER_CREDENTIAL_KEY`）
- **含める:** 暗号化資格情報を含むデータベース
- 復旧には両方が必要です

---

## ベストプラクティス

### シークレット管理

1. **CI/CD では環境変数を使う** - 常に利用可能で最優先
2. **ローカル開発はファイルベース** - 管理が容易で再起動後も保持
3. **本番は AWS/Vault を使用** - 集中管理、監査、ローテーション
4. **シークレットをバージョン管理に含めない** - `secrets.json` を `.gitignore` に追加

### キャッシュ調整

1. **プロバイダー設定（TTL 5 分）** - 変更頻度が低いので長めで問題なし
2. **ユーザー権限（TTL 1 分）** - 変更反映を早めるため短め
3. **書き込み時に無効化** - データ更新時は必ずキャッシュ無効化
4. **統計を監視** - `cache.stats()` でヒット率や期限切れ数を確認

### セキュリティ面

1. **ファイル権限** - シークレットファイルは `0o600`（所有者のみ）
2. **メモリキャッシュ** - 取得後はメモリにキャッシュされる
3. **機密キャッシュのクリア** - 必要に応じて `secrets.clear_cache()` を呼び出す
4. **監査ログ** - プロバイダー設定変更はすべて監査ログに記録

---

## トラブルシューティング

### シークレットが見つからない

<details>
<summary>プロバイダーの利用可否を確認</summary>

```python
secrets = get_secrets_manager()

# 利用可能なプロバイダーを確認
for provider in secrets._providers:
    print(f"{provider.__class__.__name__}: {provider.is_available()}")
```
</details>

### キャッシュが更新されない

<details>
<summary>強制的に無効化</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    invalidate_provider_cache
)

# 方法 1: 特定プロバイダーを無効化
invalidate_provider_cache("replicate")

# 方法 2: キャッシュ全体をクリア
cache = get_provider_config_cache()
cache.clear()
```
</details>

### Vault 接続の問題

<details>
<summary>Vault への接続確認</summary>

```bash
# Vault の状態を確認
curl -s $VAULT_ADDR/v1/sys/health

# トークンが有効か確認
curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_ADDR/v1/auth/token/lookup-self
```
</details>

### AWS Secrets Manager の問題

<details>
<summary>AWS 資格情報と権限を確認</summary>

```bash
# 現在のアイデンティティを確認
aws sts get-caller-identity

# シークレットアクセスをテスト
aws secretsmanager get-secret-value --secret-id $SIMPLETUNER_AWS_SECRET_NAME
```
</details>
