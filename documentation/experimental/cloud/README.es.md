# Sistema de entrenamiento en la nube

> **Estado:** Experimental
>
> **Disponible en:** Web UI (pestaña Cloud)

El sistema de entrenamiento en la nube de SimpleTuner te permite ejecutar trabajos de entrenamiento en proveedores de GPU en la nube sin configurar infraestructura propia. El sistema está diseñado para ser extensible, permitiendo añadir múltiples proveedores con el tiempo.

## Visión general

El sistema de entrenamiento en la nube ofrece:

- **Seguimiento unificado de trabajos** - Rastrea trabajos locales y en la nube en un solo lugar
- **Empaquetado automático de datos** - Datasets locales se empaquetan y suben automáticamente
- **Entrega de resultados** - Los modelos entrenados pueden enviarse a HuggingFace, S3 o descargarse localmente
- **Seguimiento de costos** - Monitorea gasto y costos por proveedor con límites de gasto
- **Snapshots de configuración** - Opcionalmente versiona configs de entrenamiento con git

## Conceptos clave

Antes de usar entrenamiento en la nube, entiende estas tres cosas:

### 1. Qué pasa con tus datos

Cuando envías un trabajo en la nube:

1. **Se empaquetan los datasets** - Los datasets locales (`type: "local"`) se comprimen y verás un resumen
2. **Se suben al proveedor** - El zip va directamente al proveedor en la nube después de consentir
3. **Se ejecuta el entrenamiento** - Puede que el modelo tenga que descargarse antes de entrenar con tus muestras en GPUs de la nube
4. **Se eliminan los datos** - Tras el entrenamiento, los datos subidos se eliminan de los servidores del proveedor y se entrega tu modelo

**Notas de seguridad:**
- Tu token de API nunca sale de tu máquina
- Archivos sensibles (.env, .git, credenciales) se excluyen automáticamente
- Revisas y consientes subidas antes de cada trabajo

### 2. Cómo recibes los modelos entrenados

El entrenamiento produce un modelo que necesita un destino. Configura uno de:

| Destino | Configuración | Mejor para |
|-------------|-------|----------|
| **HuggingFace Hub** | Configura la variable de entorno `HF_TOKEN`, habilita en la pestaña Publishing | Compartir modelos, acceso fácil |
| **Descarga local** | Configura URL de webhook, expón el servidor vía ngrok | Privacidad, flujos locales |
| **Almacenamiento S3** | Configura endpoint en la pestaña Publishing | Acceso del equipo, archivo |

Consulta [Recibir modelos entrenados](TUTORIAL.md#receiving-trained-models) para la configuración paso a paso.

### 3. Modelo de costos

Replicate cobra por segundo de tiempo de GPU:

| Hardware | VRAM | Costo | LoRA típica (2000 pasos) |
|----------|------|------|---------------------------|
| L40S | 48GB | ~$3.50/hr | $5-15 |

**La facturación inicia** cuando comienza el entrenamiento y **se detiene** cuando termina o falla.

**Protégete:**
- Configura un límite de gasto en Cloud settings
- Se muestran estimaciones de costo antes de cada envío
- Cancela trabajos en ejecución en cualquier momento (pagas el tiempo usado)

Consulta [Costos](REPLICATE.md#costs) para precios y límites.

## Arquitectura

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

## Proveedores compatibles

| Proveedor | Estado | Funciones |
|----------|--------|----------|
| [Replicate](REPLICATE.md) | Compatible | Seguimiento de costos, logs en vivo, webhooks |
| [Orquestación de workers](../server/WORKERS.md) | Compatible | Workers distribuidos autoalojados, cualquier GPU |
| SimpleTuner.io | Próximamente | Servicio gestionado de entrenamiento por el equipo de SimpleTuner |

### Orquestación de workers

Para entrenamiento distribuido autoalojado en múltiples máquinas, consulta la [Guía de orquestación de workers](../server/WORKERS.md). Los workers pueden ejecutarse en:

- Servidores GPU on-premise
- VMs en la nube (cualquier proveedor)
- Instancias spot (RunPod, Vast.ai, Lambda Labs)

Los workers se registran con tu orquestador SimpleTuner y reciben trabajos automáticamente.

## Flujo de datos

### Enviar un trabajo

1. **Preparación de config** - Tu configuración de entrenamiento se serializa
2. **Empaquetado de datos** - Los datasets locales (con `type: "local"`) se comprimen
3. **Subida** - El zip se sube al hosting de archivos de Replicate
4. **Envío** - El trabajo se envía al proveedor en la nube
5. **Seguimiento** - El estado del trabajo se consulta y actualiza en tiempo real

### Recepción de resultados

Los resultados pueden entregarse vía:

1. **HuggingFace Hub** - Subir el modelo entrenado a tu cuenta de HuggingFace
2. **Almacenamiento compatible con S3** - Subir a cualquier endpoint S3 (AWS, MinIO, etc.)
3. **Descarga local** - SimpleTuner incluye un endpoint compatible con S3 para recibir subidas localmente

## Privacidad de datos y consentimiento

Cuando envías un trabajo en la nube, SimpleTuner puede subir:

- **Datasets de entrenamiento** - Imágenes/archivos de datasets con `type: "local"`
- **Configuración** - Tus parámetros de entrenamiento (tasa de aprendizaje, ajustes de modelo, etc.)
- **Captions/metadata** - Cualquier archivo de texto asociado a tus datasets

Los datos se suben directamente al proveedor de la nube (p. ej., hosting de archivos de Replicate). No pasan por los servidores de SimpleTuner.

### Ajustes de consentimiento

En la pestaña Cloud, puedes configurar el comportamiento de subida de datos:

| Ajuste | Comportamiento |
|---------|----------|
| **Always Ask** | Muestra un diálogo de confirmación listando datasets antes de cada subida |
| **Always Allow** | Omite confirmación para flujos confiables |
| **Never Upload** | Desactiva entrenamiento en la nube (solo local) |

## Endpoint S3 local

SimpleTuner incluye un endpoint compatible con S3 para recibir modelos entrenados:

```
PUT /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}/{key}
GET /api/cloud/storage/{bucket}  (list objects)
GET /api/cloud/storage  (list buckets)
```

Los archivos se guardan en `~/.simpletuner/cloud_outputs/` por defecto.

Puedes configurar las credenciales manualmente; si no lo haces, se generan credenciales efímeras automáticamente para cada trabajo de entrenamiento, lo cual es el enfoque recomendado.

Esto permite un modo de "solo descarga" donde:
1. Configuras una URL de webhook apuntando a tu instancia local de SimpleTuner
2. SimpleTuner autoconfigura los ajustes de publicación S3
3. Los modelos entrenados se suben de vuelta a tu máquina

**Nota:** Necesitarás exponer tu instancia local de SimpleTuner vía ngrok, cloudflared o similar para que el proveedor en la nube pueda alcanzarla.

## Añadir nuevos proveedores

El sistema en la nube está diseñado para ser extensible. Para añadir un nuevo proveedor:

1. Crea una nueva clase cliente implementando `CloudTrainerService`:

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

2. Registra el proveedor en las rutas de cloud
3. Añade elementos de UI para la nueva pestaña del proveedor

## Archivos y ubicaciones

| Ruta | Descripción |
|------|-------------|
| `~/.simpletuner/cloud/` | Estado relacionado con la nube e historial de trabajos |
| `~/.simpletuner/cloud/job_history.json` | Base de datos unificada de seguimiento de trabajos |
| `~/.simpletuner/cloud/provider_configs/` | Configuración por proveedor |
| `~/.simpletuner/cloud_outputs/` | Almacenamiento del endpoint S3 local |

## Solución de problemas

### "REPLICATE_API_TOKEN not set"

Configura la variable de entorno antes de iniciar SimpleTuner:

```bash
export REPLICATE_API_TOKEN="r8_..."
simpletuner --webui
```

### Falla la subida de datos

- Comprueba tu conexión a Internet
- Verifica que existan las rutas de dataset
- Busca errores en la consola del navegador
- Asegúrate de tener créditos y permisos adecuados en el proveedor de la nube

### El webhook no recibe resultados

- Asegúrate de que tu instancia local sea accesible públicamente
- Comprueba que la URL del webhook sea correcta
- Verifica que las reglas de firewall permitan conexiones entrantes

## Limitaciones actuales

El sistema de entrenamiento en la nube está diseñado para **trabajos de entrenamiento de un solo disparo**. Las siguientes funciones no se soportan actualmente:

### Trabajos de flujo/pipeline (DAGs)

SimpleTuner no soporta dependencias entre trabajos o flujos multi-paso donde la salida de un trabajo alimenta otro. Cada trabajo es independiente y autocontenido.

**Si necesitas flujos:**
- Usa herramientas de orquestación externas (Airflow, Prefect, Dagster)
- Encadena trabajos vía la API REST desde tu pipeline
- Consulta [ENTERPRISE.md](../server/ENTERPRISE.md#external-orchestration-airflow) para un ejemplo de integración con Airflow

### Reanudar ejecuciones de entrenamiento

No hay soporte integrado para reanudar una ejecución de entrenamiento que fue interrumpida, falló o se detuvo temprano. Si un trabajo falla o se cancela:
- Debes reenviarlo desde el inicio
- No hay recuperación automática de checkpoints desde almacenamiento en la nube

**Workarounds:**
- Configura subidas frecuentes a HuggingFace Hub (`--push_checkpoints_to_hub`) para guardar checkpoints intermedios
- Implementa tu propia gestión de checkpoints descargando salidas y re-subiéndolas como punto de partida para un nuevo trabajo
- Para trabajos críticos de larga duración, considera dividir en segmentos de entrenamiento más pequeños

Estas limitaciones pueden abordarse en futuras versiones.

## Ver también

### Entrenamiento en la nube

- [Tutorial de entrenamiento en la nube](TUTORIAL.md) - Guía de inicio
- [Integración con Replicate](REPLICATE.md) - Configuración del proveedor Replicate
- [Cola de trabajos](../../JOB_QUEUE.md) - Programación y concurrencia
- [Guía de operaciones](OPERATIONS_TUTORIAL.md) - Despliegue en producción

### Funciones multiusuario (aplica a local y nube)

- [Guía Enterprise](../server/ENTERPRISE.md) - SSO, aprobaciones y gobernanza
- [Autenticación externa](../server/EXTERNAL_AUTH.md) - Configuración OIDC y LDAP
- [Registro de auditoría](../server/AUDIT.md) - Logging de eventos de seguridad

### General

- [Tutorial de API local](../../api/TUTORIAL.md) - Entrenamiento local vía API REST
- [Documentación de datasets](../../DATALOADER.md) - Entender configs del dataloader
