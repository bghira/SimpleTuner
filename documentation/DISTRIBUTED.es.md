# Entrenamiento distribuido (multi-nodo)

Este documento contiene notas* sobre cómo configurar un clúster 4-way de 8xH100 para usar con SimpleTuner.

> *Esta guía no contiene instrucciones completas de instalación de extremo a extremo. En su lugar, sirven como consideraciones al seguir el documento [INSTALL](INSTALL.md) o una de las [guías de quickstart](QUICKSTART.md).

## Backend de almacenamiento

El entrenamiento multi-nodo requiere por defecto el uso de almacenamiento compartido entre nodos para el `output_dir`


### Ejemplo NFS en Ubuntu

Un ejemplo básico de almacenamiento para comenzar.

#### En el nodo "master" que escribirá los checkpoints

**1. Instalar paquetes de servidor NFS**

```bash
sudo apt update
sudo apt install nfs-kernel-server
```

**2. Configurar el export de NFS**

Edita el archivo de exports de NFS para compartir el directorio:

```bash
sudo nano /etc/exports
```

Agrega la siguiente línea al final del archivo (reemplaza `slave_ip` con la IP real de tu máquina slave):

```
/home/ubuntu/simpletuner/output slave_ip(rw,sync,no_subtree_check)
```

*Si quieres permitir múltiples slaves o toda una subred, puedes usar:*

```
/home/ubuntu/simpletuner/output subnet_ip/24(rw,sync,no_subtree_check)
```

**3. Exportar el directorio compartido**

```bash
sudo exportfs -a
```

**4. Reiniciar el servidor NFS**

```bash
sudo systemctl restart nfs-kernel-server
```

**5. Verificar el estado del servidor NFS**

```bash
sudo systemctl status nfs-kernel-server
```

---

#### En los nodos slave que envían el optimizador y otros estados

**1. Instalar paquetes de cliente NFS**

```bash
sudo apt update
sudo apt install nfs-common
```

**2. Crear el directorio de montaje**

Asegúrate de que el directorio existe (ya debería existir según tu configuración):

```bash
sudo mkdir -p /home/ubuntu/simpletuner/output
```

*Nota:* Si el directorio contiene datos, haz un respaldo, ya que montar ocultará los contenidos existentes.

**3. Montar el share NFS**

Monta el directorio compartido del master en el directorio local del slave (reemplaza `master_ip` con la IP del master):

```bash
sudo mount master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output
```

**4. Verificar el montaje**

Comprueba que el montaje sea exitoso:

```bash
mount | grep /home/ubuntu/simpletuner/output
```

**5. Probar acceso de escritura**

Crea un archivo de prueba para asegurar que tienes permisos de escritura:

```bash
touch /home/ubuntu/simpletuner/output/test_file_from_slave.txt
```

Luego, revisa en la máquina master si el archivo aparece en `/home/ubuntu/simpletuner/output`.

**6. Hacer persistente el montaje**

Para asegurar que el montaje persista entre reinicios, agrégalo al archivo `/etc/fstab`:

```bash
sudo nano /etc/fstab
```

Agrega la siguiente línea al final:

```
master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output nfs defaults 0 0
```

---

#### **Consideraciones adicionales:**

- **Permisos de usuario:** Asegúrate de que el usuario `ubuntu` tenga el mismo UID y GID en ambas máquinas para que los permisos de archivos sean consistentes. Puedes comprobar UIDs con `id ubuntu`.

- **Ajustes de firewall:** Si tienes un firewall habilitado, asegúrate de permitir tráfico NFS. En la máquina master:

  ```bash
  sudo ufw allow from slave_ip to any port nfs
  ```

- **Sincronizar relojes:** Es buena práctica tener los relojes de ambos sistemas sincronizados, especialmente en setups distribuidos. Usa `ntp` o `systemd-timesyncd`.

- **Probar checkpoints de DeepSpeed:** Ejecuta un trabajo pequeño de DeepSpeed para confirmar que los checkpoints se escriben correctamente en el directorio del master.


## Configuración del dataloader

Los datasets muy grandes pueden ser un desafío para gestionar eficientemente. SimpleTuner fragmentará automáticamente los datasets entre nodos y distribuirá el preprocesamiento en cada GPU disponible del clúster, mientras usa colas asíncronas e hilos para mantener el throughput.

### Tamaño de dataset para entrenamiento multi-GPU

Al entrenar con múltiples GPUs o nodos, tu dataset debe contener suficientes muestras para satisfacer el **tamaño efectivo de batch**:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**Cálculos de ejemplo:**

| Configuración | Cálculo | Tamaño efectivo de batch |
|--------------|---------|--------------------------|
| 1 nodo, 8 GPUs, batch_size=4, grad_accum=1 | 4 × 8 × 1 | 32 muestras |
| 2 nodos, 16 GPUs, batch_size=8, grad_accum=2 | 8 × 16 × 2 | 256 muestras |
| 4 nodos, 32 GPUs, batch_size=8, grad_accum=1 | 8 × 32 × 1 | 256 muestras |

Cada bucket de relación de aspecto en tu dataset debe contener al menos ese número de muestras (considerando `repeats`) o el entrenamiento fallará con un mensaje de error detallado.

#### Soluciones para datasets pequeños

Si tu dataset es más pequeño que el tamaño efectivo de batch:

1. **Reduce el batch size** - Baja `train_batch_size` para reducir requerimientos de memoria
2. **Reduce el número de GPUs** - Entrena con menos GPUs (aunque esto ralentiza el entrenamiento)
3. **Aumenta repeats** - Configura `repeats` en tu [configuración de dataloader](DATALOADER.md#repeats)
4. **Habilita sobresuscripción automática** - Usa `--allow_dataset_oversubscription` para ajustar repeats automáticamente

El flag `--allow_dataset_oversubscription` (documentado en [OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription)) calculará y aplicará automáticamente los repeats mínimos requeridos para tu configuración, haciéndolo ideal para prototipos o experimentos con datasets pequeños.

### Escaneo/descubrimiento lento de imágenes

El backend **discovery** actualmente restringe la recolección de datos de buckets de aspecto a un solo nodo. Esto puede tomar un tiempo **extremadamente** largo con datasets muy grandes, ya que cada imagen debe leerse desde el almacenamiento para obtener su geometría.

Para evitar este problema, debe usarse el [metadata_backend parquet](DATALOADER.md#parquet-caption-strategy-json-lines-datasets), permitiéndote preprocesar tus datos de cualquier forma accesible para ti. Como se describe en la sección enlazada, la tabla parquet contiene las columnas `filename`, `width`, `height` y `caption` para ayudar a ordenar rápidamente y de forma eficiente los datos en sus buckets correspondientes.


### Espacio de almacenamiento

Los datasets enormes, especialmente al usar el encoder de texto T5-XXL, consumirán cantidades enormes de espacio para los datos originales, los image embeds y los text embeds.

#### Almacenamiento en la nube

Usando un proveedor como Cloudflare R2, se pueden generar datasets extremadamente grandes con costos de almacenamiento muy bajos.

Consulta la [guía de configuración del dataloader](DATALOADER.md#local-cache-with-cloud-dataset) para un ejemplo de cómo configurar el tipo `aws` en `multidatabackend.json`

- Los datos de imagen pueden almacenarse localmente o vía S3
  - Si las imágenes están en S3, la velocidad de preprocesamiento se reduce según el ancho de banda de red
  - Si las imágenes se almacenan localmente, esto no aprovecha el throughput de NVMe durante el **entrenamiento**
- Los image embeds y text embeds pueden almacenarse por separado en almacenamiento local o en la nube
  - Colocar embeds en almacenamiento en la nube reduce muy poco la tasa de entrenamiento, ya que se obtienen en paralelo

Idealmente, todas las imágenes y todos los embeds se mantienen simplemente en un bucket de almacenamiento en la nube. Esto simplifica enormemente el riesgo de problemas durante el preprocesamiento y la reanudación del entrenamiento.

#### Codificación VAE bajo demanda

Para datasets grandes donde almacenar latentes VAE cacheados es impráctico por restricciones de almacenamiento o acceso lento a almacenamiento compartido, puedes usar `--vae_cache_disable`. Esto desactiva la caché VAE por completo, forzando al VAE a codificar imágenes al vuelo durante el entrenamiento.

Esto aumenta la carga de cómputo de GPU pero reduce significativamente los requisitos de almacenamiento y el I/O de red para latentes cacheados.

#### Preservar cachés de escaneo del filesystem

Si tus datasets son tan grandes que escanear imágenes nuevas se convierte en un cuello de botella, agregar `preserve_data_backend_cache=true` a cada entrada de configuración del dataloader evitará que el backend se escanee buscando nuevas imágenes.

**Nota** que entonces deberías usar el tipo de backend `image_embeds` ([más información aquí](DATALOADER.md#local-cache-with-cloud-dataset)) para permitir que estas listas de caché vivan por separado en caso de que tu trabajo de preprocesamiento se interrumpa. Esto evitará que la **lista de imágenes** se vuelva a escanear al inicio.

#### Compresión de datos

La compresión de datos debe habilitarse añadiendo lo siguiente a `config.json`:

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

Esto usará gzip en línea para reducir la cantidad de espacio en disco redundante consumido por los text embeds e image embeds bastante grandes.

## Configurar vía 🤗 Accelerate

Al usar `accelerate config` (`/home/user/.cache/huggingface/accelerate/default_config.yaml`) para desplegar SimpleTuner, estas opciones tendrán prioridad sobre el contenido de `config/config.env`

Un ejemplo de default_config de Accelerate que no incluye DeepSpeed:

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

Este documento no entra en tanto detalle como la [página dedicada](DEEPSPEED.md).

Al optimizar el entrenamiento con DeepSpeed para multi-nodo, usar el nivel ZeRO más bajo posible es **esencial**.

Por ejemplo, una GPU NVIDIA de 80G puede entrenar Flux con ZeRO nivel 1 offload, minimizando sustancialmente el overhead.

Agregando las siguientes líneas

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### Optimización con torch compile

Para rendimiento extra (con el inconveniente de problemas de compatibilidad) puedes habilitar torch compile agregando las siguientes líneas en el yaml de cada nodo:

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## Rendimiento esperado

- 4 nodos H100 SXM5 conectados vía red local
- 1TB de memoria por nodo
- Caché de entrenamiento transmitida desde backend de datos compartido compatible con S3 (Cloudflare R2) en la misma región
- Batch size de **8** por acelerador, y **sin** pasos de acumulación de gradiente
  - El tamaño efectivo total de batch es **256**
- Resolución a 1024px con data bucketing habilitado
- **Velocidad**: 15 segundos por paso con datos 1024x1024 al entrenar Flux.1-dev (12B) full-rank

Batch sizes más bajos, menor resolución, y habilitar torch compile pueden llevar la velocidad a **iteraciones por segundo**:

- Reducir resolución a 512px y deshabilitar data bucketing (solo recortes cuadrados)
- Cambiar DeepSpeed de AdamW a optimizador Lion fused
- Habilitar torch compile con max-autotune
- **Velocidad**: 2 iteraciones por segundo

## Monitoreo de salud de GPU

SimpleTuner incluye monitoreo automático de salud de GPU para detectar fallos de hardware tempranamente, lo cual es especialmente importante en entrenamiento distribuido donde un fallo de una sola GPU puede desperdiciar tiempo de cómputo y dinero en todo el clúster.

### Circuit Breaker de GPU

El **circuit breaker de GPU** está siempre habilitado y monitorea:

- **Errores ECC** - Detecta errores de memoria incorregibles (importante para GPUs A100/H100)
- **Temperatura** - Alerta al acercarse al umbral de apagado térmico
- **Throttling** - Detecta ralentización de hardware por problemas térmicos o de potencia
- **Errores CUDA** - Captura errores de runtime durante el entrenamiento

Cuando se detecta un fallo de GPU:

1. Se emite un webhook `gpu.fault` (si los webhooks están configurados)
2. El circuito se abre para prevenir más entrenamiento en hardware defectuoso
3. El entrenamiento termina limpiamente para que los orquestadores puedan terminar la instancia

### Configuración de webhooks

Configura webhooks en tu `config.json` para recibir alertas de fallos de GPU:

```json
{
  "--webhook_config": "config/webhooks.json"
}
```

Ejemplo de `webhooks.json` para alertas de Discord:

```json
{
  "webhook_url": "https://discord.com/api/webhooks/...",
  "webhook_type": "discord"
}
```

### Consideraciones multi-nodo

En entrenamiento multi-nodo:

- Cada nodo ejecuta su propio monitor de salud de GPU
- Un fallo de GPU en cualquier nodo disparará un webhook desde ese nodo
- El trabajo de entrenamiento fallará en todos los nodos debido al fallo de comunicación distribuida
- Los orquestadores deben monitorear fallos desde cualquier nodo en el clúster

Ver [Infraestructura de resiliencia](experimental/cloud/RESILIENCE.md#circuit-breaker-de-gpu) para formato detallado del payload del webhook y acceso programático.

## Advertencias del entrenamiento distribuido

- Cada nodo debe tener el mismo número de aceleradores disponibles
- Solo LoRA/LyCORIS puede cuantizarse, por lo que el entrenamiento distribuido de modelos completos requiere DeepSpeed
- Esta es una operación de muy alto costo, y tamaños de batch altos pueden ralentizar más de lo deseado, requiriendo escalar el número de GPUs en el clúster. Debe considerarse un balance cuidadoso del presupuesto.
- (DeepSpeed) Las validaciones podrían necesitar deshabilitarse al entrenar con DeepSpeed ZeRO 3
- (DeepSpeed) El guardado de modelos termina creando copias fragmentadas extrañas cuando se guarda con ZeRO nivel 3, pero los niveles 1 y 2 funcionan como se espera
- (DeepSpeed) Se vuelve requerido usar optimizadores basados en CPU de DeepSpeed ya que gestiona el sharding y offload de los estados del optimizador.
