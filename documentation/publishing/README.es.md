# Proveedores de publicación

SimpleTuner ahora puede publicar salidas de entrenamiento a múltiples destinos mediante `--publishing_config`. Las subidas a Hugging Face siguen controladas por `--push_to_hub`; `publishing_config` es aditivo para otros proveedores y se ejecuta después de que la validación finalice en el proceso principal.

## Formatos de configuración
- Acepta JSON en línea (`--publishing_config='[{"provider": "s3", ...}]'`), un dict de Python pasado a través del SDK, o una ruta a un archivo JSON.
- Los valores se normalizan a una lista, igual que `--webhook_config`.
- Cada entrada requiere una clave `provider`. El `base_path` opcional antepone rutas dentro del destino remoto. Si tu configuración no puede devolver un URI, el proveedor registra una advertencia única cuando se consulta.

## Artefacto por defecto
La publicación sube el `output_dir` de la ejecución (carpetas y archivos) usando el nombre base del directorio. Los metadatos incluyen el id del trabajo actual y el tipo de validación para que los consumidores posteriores puedan vincular un URI con la ejecución.

## Proveedores
Instala dependencias opcionales dentro del `.venv` del proyecto cuando uses un proveedor.

### S3 compatible y Backblaze B2 (API S3)
- Proveedor: `s3` o `backblaze_b2`
- Dependencia: `pip install boto3`
- Ejemplo:
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

⚠️ **Nota de seguridad**: Nunca commits credenciales al control de versiones. Usa sustitución de variables de entorno o un gestor de secretos para despliegues de producción.

### Azure Blob Storage
- Proveedor: `azure_blob` (alias `azure`)
- Dependencia: `pip install azure-storage-blob`
- Ejemplo:
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
- Proveedor: `dropbox`
- Dependencia: `pip install dropbox`
- Ejemplo:
```json
[
  {
    "provider": "dropbox",
    "token": "sl.12345",
    "base_path": "/SimpleTuner/runs"
  }
]
```
Los archivos grandes se transmiten automáticamente en sesiones de subida; se crean enlaces compartidos cuando se permite, de lo contrario se registra una ruta `dropbox://`.

## Uso en CLI
```
simpletuner-train \
  --publishing_config=config/publishing.json \
  --push_to_hub=true \
  ...
```
Si llamas a SimpleTuner programáticamente, pasa una lista/dict a `publishing_config` y se normalizará automáticamente.
