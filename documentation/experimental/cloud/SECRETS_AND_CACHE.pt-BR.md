# Provedores de segredos e sistema de cache

Este documento cobre o sistema de gerenciamento de segredos do SimpleTuner e a infraestrutura de cache para treinamento em nuvem.

## Visao geral

O SimpleTuner usa um sistema de gerenciamento de segredos em camadas com varios backends de provedor e uma camada de cache em memoria para otimizar o desempenho.

## Provedores de segredos

A classe `SecretsManager` encadeia varios provedores de segredo em ordem de prioridade. Ao recuperar um segredo, os provedores sao verificados sequencialmente ate encontrar um valor.

### Ordem de prioridade dos provedores

1. **Variaveis de ambiente** (maior prioridade)
2. **Segredos baseados em arquivo**
3. **AWS Secrets Manager**
4. **HashiCorp Vault** (menor prioridade)

Essa ordenacao permite que variaveis de ambiente sempre sobrescrevam outras fontes, o que e util para:
- Overrides de desenvolvimento local
- Deploys em containers com segredos injetados
- Pipelines CI/CD

### Chaves de segredo bem conhecidas

<details>
<summary>Constantes de chave de segredo</summary>

```python
REPLICATE_API_TOKEN  # Replicate API authentication
HF_TOKEN             # HuggingFace Hub token
CLOUD_WEBHOOK_SECRET # Webhook HMAC validation secret
```
</details>

---

## Configuracao de provedores

### 1. Variaveis de ambiente (padrao)

O provedor de ambiente esta sempre disponivel e tem precedencia sobre todos os outros provedores.

**Regras de normalizacao de chaves:**
- Chaves sao convertidas para maiusculas
- Hifens (`-`) sao convertidos para underscores (`_`)
- Pontos (`.`) sao convertidos para underscores (`_`)

**Prefixo opcional:**

<details>
<summary>Exemplo de configuracao em Python</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import EnvironmentSecretProvider

# Without prefix
provider = EnvironmentSecretProvider()
# Key "replicate-api-token" -> REPLICATE_API_TOKEN

# With prefix
provider = EnvironmentSecretProvider(prefix="SIMPLETUNER")
# Key "replicate-api-token" -> SIMPLETUNER_REPLICATE_API_TOKEN
```
</details>

**Uso:**

<details>
<summary>Definindo segredos via ambiente</summary>

```bash
# Set secrets via environment
export REPLICATE_API_TOKEN="r8_your_token_here"
export HF_TOKEN="hf_your_token_here"
export CLOUD_WEBHOOK_SECRET="your_webhook_secret"
```
</details>

### 2. Segredos baseados em arquivo

Segredos podem ser armazenados em um arquivo JSON ou YAML com permissoes restritivas.

**Locais padrao (verificados em ordem):**
- `~/.simpletuner/secrets.json`
- `~/.simpletuner/secrets.yaml`
- `~/.simpletuner/secrets.yml`

<details>
<summary>Exemplos de formato de arquivo (JSON/YAML)</summary>

**Formato do arquivo (JSON):**

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```

**Formato do arquivo (YAML):**

```yaml
REPLICATE_API_TOKEN: r8_your_token_here
HF_TOKEN: hf_your_token_here
CLOUD_WEBHOOK_SECRET: your_webhook_secret
```
</details>

<details>
<summary>Configuracao manual</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import FileSecretProvider

# Use a custom file path
provider = FileSecretProvider(file_path="/path/to/secrets.json")
```
</details>

**Nota de seguranca:** Ao salvar segredos via API, o arquivo e criado com `chmod 0o600` (apenas leitura/escrita para o dono).

### 3. AWS Secrets Manager

Para deploys em producao, o AWS Secrets Manager fornece armazenamento seguro e centralizado de segredos.

<details>
<summary>Variaveis de ambiente</summary>

```bash
# Required: Name of the secret in AWS Secrets Manager
export SIMPLETUNER_AWS_SECRET_NAME="simpletuner/production"

# Optional: AWS region (uses default if not set)
export AWS_DEFAULT_REGION="us-west-2"
```
</details>

<details>
<summary>Formato do segredo AWS</summary>

O segredo no AWS Secrets Manager deve ser um objeto JSON:

```json
{
  "REPLICATE_API_TOKEN": "r8_your_token_here",
  "HF_TOKEN": "hf_your_token_here",
  "CLOUD_WEBHOOK_SECRET": "your_webhook_secret"
}
```
</details>

<details>
<summary>Configuracao manual</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import AWSSecretsManagerProvider

provider = AWSSecretsManagerProvider(
    secret_name="simpletuner/production",
    region_name="us-west-2"
)
```
</details>

<details>
<summary>Permissoes AWS IAM necessarias</summary>

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

Para empresas que usam HashiCorp Vault para gerenciamento de segredos.

**Dependencias necessarias:**

```bash
pip install hvac
```

<details>
<summary>Variaveis de ambiente</summary>

```bash
# Required: Vault server URL
export VAULT_ADDR="https://vault.example.com:8200"
```
</details>
# Required: Vault authentication token
export VAULT_TOKEN="s.your_vault_token"

# Optional: Path to secrets (default: "simpletuner")
export SIMPLETUNER_VAULT_PATH="simpletuner"
```
</details>

<details>
<summary>Configuracao manual</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import HashiCorpVaultProvider

provider = HashiCorpVaultProvider(
    url="https://vault.example.com:8200",
    token="s.your_vault_token",
    path="simpletuner",
    mount_point="secret"  # KV mount point (default: "secret")
)
```
</details>

<details>
<summary>Configuracao KV do Vault</summary>

```bash
# KV v2 (recommended)
vault kv put secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"

# KV v1
vault write secret/simpletuner \
    REPLICATE_API_TOKEN="r8_your_token" \
    HF_TOKEN="hf_your_token"
```

O provedor tenta KV v2 primeiro, caindo para v1 se falhar.
</details>

---

## Usando o Secrets Manager

### Uso basico

<details>
<summary>Obtendo segredos</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import get_secrets_manager

# Get the singleton instance
secrets = get_secrets_manager()

# Get any secret
api_token = secrets.get("REPLICATE_API_TOKEN")
api_token = secrets.get("REPLICATE_API_TOKEN", default="fallback_value")

# Convenience methods
replicate_token = secrets.get_replicate_token()
hf_token = secrets.get_hf_token()
webhook_secret = secrets.get_webhook_secret()
```
</details>

### Salvando segredos (apenas provedor de arquivo)

<details>
<summary>Definindo e deletando segredos</summary>

```python
secrets = get_secrets_manager()

# Save a secret (writes to ~/.simpletuner/secrets.json)
secrets.set_secret("MY_CUSTOM_KEY", "my_value")

# Convenience method for Replicate token
secrets.set_replicate_token("r8_new_token")

# Delete a secret
secrets.delete_secret("MY_CUSTOM_KEY")
```
</details>

### Configuracao de provedor customizada

<details>
<summary>Exemplo de configuracao customizada</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.secrets import SecretsManager

# Reset singleton for custom configuration
SecretsManager.reset()

# Create with custom config
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

## Sistema de cache

O SimpleTuner usa um cache TTL (Time-To-Live) em memoria para reduzir consultas ao banco e melhorar tempos de resposta para dados acessados com frequencia.

### Recursos do TTLCache

- **Operacoes thread-safe** usando `RLock`
- **TTL por chave** com defaults configuraveis
- **Limpeza automatica** de entradas expiradas
- **Eviccao LRU** ao atingir capacidade
- **Invalidacao por prefixo** para limpeza em massa

### Configuracao do cache

<details>
<summary>Inicializacao do TTLCache</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[str](
    default_ttl=300.0,      # 5 minutes default TTL
    max_size=1000,          # Maximum entries before eviction
    cleanup_interval=60.0   # Cleanup expired entries every 60s
)
```
</details>

### Instancias globais de cache

O SimpleTuner mantem duas instancias globais de cache:

| Cache | TTL padrao | Tamanho maximo | Proposito |
|-------|-------------|----------|---------|
| Cache de config do provedor | 300s (5 min) | 100 | Config de provedor, URLs de webhook, limites de custo |
| Cache de permissao de usuario | 60s (1 min) | 500 | Permissoes de usuario (TTL menor por seguranca) |

<details>
<summary>Acessando caches globais</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    get_user_permission_cache
)

# Access global caches
provider_cache = get_provider_config_cache()
user_cache = get_user_permission_cache()
```
</details>

### Operacoes basicas de cache

<details>
<summary>Referencia de operacoes de cache</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import TTLCache

cache = TTLCache[dict](default_ttl=120.0)

# Set a value (uses default TTL)
cache.set("my_key", {"data": "value"})

# Set with custom TTL
cache.set("my_key", {"data": "value"}, ttl=60.0)

# Get a value (returns None if expired or missing)
value = cache.get("my_key")

# Get or compute if missing
value = cache.get_or_set("my_key", lambda: compute_value())

# Async version
value = await cache.get_or_set_async("my_key", async_compute_value)

# Delete a specific key
cache.delete("my_key")

# Clear all entries
count = cache.clear()

# Get cache statistics
stats = cache.stats()
# {"size": 42, "max_size": 1000, "expired_count": 3, "default_ttl": 120.0}
```
</details>

---

## Cache de metadados do provedor

A configuracao do provedor (URLs de webhook, limites de custo, info de hardware) e armazenada em cache para reduzir carga no banco.

### Formato da chave de cache

```
provider:{provider_name}:config
```

Exemplo: `provider:replicate:config`

### Comportamento do cache

**Na leitura (`get_provider_config`):**
1. Verifica o cache para `provider:{name}:config`
2. Se encontrado e nao expirado, retorna o valor em cache
3. Se ausente/expirado, carrega do banco
4. Armazena o resultado em cache com TTL de 5 minutos
5. Retorna o valor

**Na escrita (`save_provider_config`):**
1. Escreve no banco
2. Invalida o cache para `provider:{name}:*` (invalidacao por prefixo)

### Invalidacao de cache

<details>
<summary>Exemplos de invalidacao</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    invalidate_provider_cache,
    invalidate_user_cache
)

# Invalidate all cached config for a provider
invalidate_provider_cache("replicate")

# Invalidate all cached data for a user
invalidate_user_cache(user_id=123)
```
</details>

### Invalidacao baseada em prefixo

<details>
<summary>Exemplo de invalidacao por prefixo</summary>

```python
cache = get_provider_config_cache()

# Invalidate all keys starting with "provider:replicate:"
count = cache.invalidate_prefix("provider:replicate:")
```
</details>

---

## Cache de informacoes de hardware

Informacoes de precos de hardware sao armazenadas em cache separadamente com invalidacao manual.

### Comportamento do cache

<details>
<summary>Operacoes do cache de info de hardware</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.replicate_client import (
    get_hardware_info,
    get_hardware_info_async,
    clear_hardware_info_cache,
    set_hardware_info
)

# Get hardware info (uses cache or defaults)
info = get_hardware_info()
info = await get_hardware_info_async(store)

# Clear cache to force reload from config
clear_hardware_info_cache()

# Manually set hardware info (updates cache)
set_hardware_info({
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975}
})
```
</details>

### Precificacao padrao de hardware

<details>
<summary>Valores padrao de info de hardware</summary>

```python
DEFAULT_HARDWARE_INFO = {
    "gpu-l40s": {"name": "L40S (48GB)", "cost_per_second": 0.000975},
    "gpu-a100-large": {"name": "A100 (80GB)", "cost_per_second": 0.001400},
}
```
</details>

---

## Cache com decorators

Para cachear resultados de funcoes, use o decorator `@cached`:

<details>
<summary>Uso do decorator cached</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import cached

@cached(ttl=120, key_prefix="mymodule")
def get_user(user_id: int) -> dict:
    # Expensive database query
    return {"id": user_id, "name": "..."}

@cached(ttl=60)
async def fetch_external_data() -> dict:
    # Async API call
    return await api_client.get("/data")

# Cache key is auto-generated from function name and arguments:
# "mymodule:get_user:123" for get_user(123)

# Access the underlying cache for manual invalidation
get_user._cache.clear()
```
</details>

---

## Criptografia de credenciais

O SimpleTuner criptografa credenciais sensiveis (como tokens de API) em repouso usando criptografia simetrica Fernet. Isso fornece uma camada adicional de seguranca alem das permissoes de arquivo.

### Como funciona

1. **Derivacao de chave**: Um segredo mestre e derivado usando PBKDF2-HMAC-SHA256 (100.000 iteracoes)
2. **Criptografia**: Credenciais sao criptografadas com Fernet (AES-128-CBC com HMAC)
3. **Armazenamento**: Valores criptografados sao armazenados base64-encoded no banco de dados

### Gerenciamento de chaves

A chave de criptografia e obtida na seguinte ordem de prioridade:

| Prioridade | Fonte | Caso de uso |
|----------|--------|----------|
| 1 | `SIMPLETUNER_CREDENTIAL_KEY` env var | Deploys em producao, containers |
| 2 | Arquivo `~/.simpletuner/credential.key` | Desenvolvimento local, chave persistente |
| 3 | Auto-gerada | Primeira configuracao (salva no arquivo) |

<details>
<summary>Definindo a chave via ambiente</summary>

```bash
# Generate a secure key
export SIMPLETUNER_CREDENTIAL_KEY=$(openssl rand -base64 32)
```
</details>

**Local do arquivo de chave:**

```
~/.simpletuner/credential.key
```

O arquivo de chave e criado com `chmod 0600` (apenas leitura/escrita para o dono).

### Uso

A criptografia de credenciais e automatica ao armazenar tokens de API do provedor via UI ou API. Voce nao precisa chamar essas funcoes diretamente, mas elas estao disponiveis:

<details>
<summary>Funcoes de criptografia/descriptografia</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.auth.credential_encryption import (
    encrypt_credential,
    decrypt_credential,
)

# Encrypt a token
encrypted = encrypt_credential("r8_my_api_token")

# Decrypt it back
plaintext = decrypt_credential(encrypted)
```
</details>

### Requisitos

A criptografia de credenciais requer o pacote `cryptography`:

```bash
pip install cryptography
```

Se nao estiver instalado, credenciais sao armazenadas em texto plano (com um aviso nos logs).

### Rotacao de chaves

Para rotacionar a chave de criptografia:

1. Exporte credenciais existentes (serao descriptografadas com a chave antiga)
2. Defina a nova chave no ambiente ou arquivo de chave
3. Reinicie o servidor
4. Reinsira as credenciais via UI (serao criptografadas com a nova chave)

**Aviso:** Mudar a chave sem reinsere as credenciais tornara credenciais criptografadas existentes ilegiveis.

### Consideracoes de backup

Ao fazer backup do SimpleTuner:

- **Inclua**: `~/.simpletuner/credential.key` (ou anote seu `SIMPLETUNER_CREDENTIAL_KEY`)
- **Inclua**: O banco contendo credenciais criptografadas
- Ambos sao necessarios para recuperar credenciais criptografadas

---

## Boas praticas

### Gerenciamento de segredos

1. **Use variaveis de ambiente para CI/CD** - Estao sempre disponiveis e tem precedencia
2. **Use segredos em arquivo para dev local** - Facil de gerenciar, persistem entre reinicios
3. **Use AWS/Vault para producao** - Segredos centralizados, auditaveis e rotacionaveis
4. **Nunca comite segredos no controle de versao** - Adicione `secrets.json` ao `.gitignore`

### Ajuste de cache

1. **Config do provedor (TTL 5 min)** - Configuracoes raramente mudam; TTL maior e ok
2. **Permissoes de usuario (TTL 1 min)** - TTL menor garante propagacao rapida
3. **Invalidar na escrita** - Sempre invalide o cache ao atualizar os dados
4. **Monitorar stats de cache** - Use `cache.stats()` para verificar hit rates e expirados

### Consideracoes de seguranca

1. **Permissoes de arquivo** - Arquivos de segredo sao criados com `0o600` (somente dono)
2. **Cache em memoria** - Segredos sao cacheados em memoria apos primeira recuperacao
3. **Limpar caches sensiveis** - Chame `secrets.clear_cache()` se necessario por seguranca
4. **Audit logging** - Todas as alteracoes de config de provedor sao registradas no audit trail

---

## Solucao de problemas

### Segredo nao encontrado

<details>
<summary>Verifique disponibilidade dos provedores</summary>

```python
secrets = get_secrets_manager()

# Check which providers are available
for provider in secrets._providers:
    print(f"{provider.__class__.__name__}: {provider.is_available()}")
```
</details>

### Cache nao atualiza

<details>
<summary>Force invalidacao de cache</summary>

```python
from simpletuner.simpletuner_sdk.server.services.cloud.cache import (
    get_provider_config_cache,
    invalidate_provider_cache
)

# Option 1: Invalidate specific provider
invalidate_provider_cache("replicate")

# Option 2: Clear entire cache
cache = get_provider_config_cache()
cache.clear()
```
</details>

### Problemas de conexao com Vault

<details>
<summary>Verifique se o Vault esta acessivel</summary>

```bash
# Check Vault status
curl -s $VAULT_ADDR/v1/sys/health

# Verify token is valid
curl -s -H "X-Vault-Token: $VAULT_TOKEN" $VAULT_ADDR/v1/auth/token/lookup-self
```
</details>

### Problemas com AWS Secrets Manager

<details>
<summary>Verifique credenciais e permissoes AWS</summary>

```bash
# Check current identity
aws sts get-caller-identity

# Test secret access
aws secretsmanager get-secret-value --secret-id $SIMPLETUNER_AWS_SECRET_NAME
```
</details>
