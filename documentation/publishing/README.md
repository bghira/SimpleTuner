# Publishing Providers

SimpleTuner can now publish training outputs to multiple destinations through `--publishing_config`. Hugging Face uploads remain controlled by `--push_to_hub`; `publishing_config` is additive for other providers and runs after validation completes on the main process.

## Config formats
- Accepts inline JSON (`--publishing_config='[{"provider": "s3", ...}]'`), a Python dict passed through the SDK, or a path to a JSON file.
- Values normalise to a list, matching how `--webhook_config` behaves.
- Each entry requires a `provider` key. Optional `base_path` prefixes paths inside the remote destination. If your config cannot return a URI, the provider logs a one-time warning when queried.

## Default artifact
Publishing uploads the run’s `output_dir` (folders and files) using the baseline name of the directory. Metadata includes the current job id and validation type so downstream consumers can tie a URI back to the run.

## Providers
Install optional dependencies inside the project `.venv` when you use a provider.

### S3-compatible and Backblaze B2 (S3 API)
- Provider: `s3` or `backblaze_b2`
- Dependency: `pip install boto3`
- Example:
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

⚠️ **Security Note**: Never commit credentials to version control. Use environment variable substitution or a secrets manager for production deployments.

### Azure Blob Storage
- Provider: `azure_blob` (alias `azure`)
- Dependency: `pip install azure-storage-blob`
- Example:
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
- Provider: `dropbox`
- Dependency: `pip install dropbox`
- Example:
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
Large files stream in upload sessions automatically; shared links are created when permitted, otherwise a `dropbox://` path is recorded.

## CLI usage
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```
If you are calling SimpleTuner programmatically, pass a list/dict to `publishing_config` and it will be normalised for you.
