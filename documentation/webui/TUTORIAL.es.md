# Tutorial del WebUI de SimpleTuner

## Introducción

Este tutorial te ayudará a comenzar con la interfaz web de SimpleTuner.

## Instalación de requisitos

Para sistemas Ubuntu, comienza instalando los paquetes requeridos:

```bash
apt -y install python3.12-venv python3.12-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # if you're using DeepSpeed
apt -y install ffmpeg # if training video models
```

## Crear un directorio de workspace

Un workspace contiene tus configuraciones, modelos de salida, imágenes de validación y, potencialmente, tus datasets.

En Vast o proveedores similares, puedes usar el directorio `/workspace/simpletuner`:

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

Si prefieres crearla en tu directorio home:
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## Instalar SimpleTuner en tu workspace

Crea un entorno virtual para instalar dependencias:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
```

### Dependencias específicas de CUDA

Los usuarios de NVIDIA deberán usar los extras de CUDA para obtener todas las dependencias correctas:

```bash
pip install -e 'simpletuner[cuda]'
# or, if you've cloned via git:
# pip install -e '.[cuda]'
```

Hay otros extras para usuarios en hardware Apple y ROCm; consulta las [instrucciones de instalación](../INSTALL.md).

## Iniciar el servidor

Para iniciar el servidor con SSL en el puerto 8080:

```bash
# for DeepSpeed, we'll need CUDA_HOME pointing to the correct location
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

Ahora, visita https://localhost:8080 en tu navegador.

Es posible que necesites reenviar el puerto por SSH, por ejemplo:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **Tip:** Si tienes un entorno de configuración existente (p. ej., de uso previo del CLI), puedes iniciar el servidor con `--env` para comenzar automáticamente el entrenamiento una vez que el servidor esté listo:
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> Esto equivale a iniciar el servidor y luego hacer clic manualmente en "Start Training" en el WebUI, pero permite un arranque sin supervisión.

## Configuración inicial: crear una cuenta admin

En el primer inicio, SimpleTuner requiere que crees una cuenta de administrador. Cuando visites el WebUI por primera vez, verás una pantalla de configuración que te pedirá crear el primer usuario admin.

Ingresa tu correo, nombre de usuario y una contraseña segura. Esta cuenta tendrá privilegios administrativos completos.

### Gestión de usuarios

Después de la configuración, puedes gestionar usuarios desde la página **Manage Users** (accesible desde la barra lateral al iniciar sesión como admin):

- **Users tab**: Crear, editar y eliminar cuentas de usuario. Asignar niveles de permiso (viewer, researcher, lead, admin).
- **Levels tab**: Definir niveles de permiso personalizados con control de acceso granular.
- **Auth Providers tab**: Configurar autenticación externa (OIDC, LDAP) para inicio de sesión único.
- **Registration tab**: Controlar si los nuevos usuarios pueden auto-registrarse (deshabilitado por defecto).

### API keys para automatización

Los usuarios pueden generar API keys para acceso mediante scripts desde su perfil o el panel admin. Las API keys usan el prefijo `st_` y pueden usarse con el encabezado `X-API-Key`:

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **Nota:** Para despliegues privados/internos, mantén el registro público deshabilitado y crea cuentas manualmente desde el panel admin.

## Usar el WebUI

### Pasos de onboarding

Una vez que cargues la página, se te harán preguntas de onboarding para configurar tu entorno.

#### Directorio de configuración

Se introduce el valor de configuración especial `configs_dir` para apuntar a una carpeta que contiene todas tus configuraciones de SimpleTuner, que se recomienda ordenar en subdirectorios: **el Web UI lo hará por ti**:

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="788" height="465" alt="image" src="https://github.com/user-attachments/assets/656aa287-3b59-476d-ac45-6ede325fe858" />

##### Migrar desde uso en línea de comandos

Si antes usabas SimpleTuner sin WebUI, puedes apuntar a tu carpeta config/ existente y todos tus entornos se auto-descubrirán.

Para usuarios nuevos, la ubicación predeterminada de tus configs y datasets será `~/.simpletuner/` y se recomienda mover tus datasets a un lugar con más espacio:

<img width="775" height="454" alt="image" src="https://github.com/user-attachments/assets/39238810-da26-4bde-8fc9-1002251f778a" />


#### Selección y configuración de (multi-)GPU

Después de configurar las rutas predeterminadas, llegarás a un paso donde se puede configurar multi-GPU (en la imagen, un Macbook)

<img width="755" height="646" alt="image" src="https://github.com/user-attachments/assets/de43a09d-06a7-45c0-8111-7a0b014499c8" />

Si tienes varias GPUs y quieres usar solo la segunda, aquí es donde puedes hacerlo.

> **Nota para usuarios multi-GPU:** Al entrenar con múltiples GPUs, los requisitos de tamaño del dataset aumentan proporcionalmente. El tamaño de batch efectivo se calcula como `train_batch_size × num_gpus × gradient_accumulation_steps`. Si tu dataset es más pequeño que este valor, tendrás que aumentar el ajuste `repeats` en la configuración del dataset o habilitar la opción `--allow_dataset_oversubscription` en la configuración avanzada. Consulta la [sección de tamaño de batch](#consideraciones-de-tamaño-de-batch-multi-gpu) más abajo para más detalles.

#### Crear tu primer entorno de entrenamiento

Si no se encontraron configuraciones previas en tu `configs_dir`, se te pedirá crear **tu primer entorno de entrenamiento**:

<img width="750" height="1381" alt="image" src="https://github.com/user-attachments/assets/4a3ee88f-c70f-416c-ae5d-6593deb9ca35" />

Usa **Bootstrap From Example** para seleccionar un ejemplo de configuración como base, o simplemente ingresa un nombre descriptivo y crea un entorno aleatorio si prefieres usar un asistente de configuración.

### Cambiar entre entornos de entrenamiento

Si tenías entornos de configuración preexistentes, aparecerán en este menú desplegable.

De lo contrario, la opción que creamos durante el onboarding ya estará seleccionada y activa.

<img width="965" height="449" alt="image" src="https://github.com/user-attachments/assets/d8c73cef-ecbb-4229-ad54-9ccd55f8175a" />

Usa **Manage Configs** para ir a la pestaña `Environment`, donde encontrarás una lista de tus entornos, configuraciones de dataloader y otras.

### Asistente de configuración

He trabajado para ofrecer un asistente de configuración completo que te ayudará a configurar algunos de los ajustes más importantes con un arranque directo y sin rodeos.

<img width="394" height="286" alt="image" src="https://github.com/user-attachments/assets/21e99854-1d75-4ba9-8be6-15e715d77f4e" />

En el menú de navegación superior izquierdo, el botón Wizard te llevará a un diálogo de selección:

<img width="1186" height="1756" alt="image" src="https://github.com/user-attachments/assets/f6d4ac57-e3f6-4060-a4d3-b7f0829d7350" />

Luego se ofrecen todas las variantes de modelos integradas. Cada variante prehabilitará ajustes necesarios como Attention Masking o límites de tokens extendidos.

#### Opciones de modelo LoRA

Si deseas entrenar un LoRA, podrás configurar aquí las opciones de cuantización del modelo.

En general, salvo que estés entrenando un modelo tipo Stable Diffusion, se recomienda int8-quanto porque no afectará la calidad y permite tamaños de batch más altos.

Algunos modelos pequeños como Cosmos2, Sana y PixArt no se llevan bien con la cuantización.

<img width="1106" height="1464" alt="image" src="https://github.com/user-attachments/assets/0284d987-6060-4692-934a-0905ef2d5ca1" />

#### Entrenamiento full-rank

Se desaconseja el entrenamiento full-rank, ya que generalmente tarda más y cuesta más recursos que un LoRA/LyCORIS con el mismo dataset.

Sin embargo, si deseas entrenar un checkpoint completo, puedes configurar aquí las etapas DeepSpeed ZeRO, que serán necesarias para modelos más grandes como Auraflow, Flux y otros más grandes.

FSDP2 es compatible, pero no configurable en este asistente. Simplemente deja DeepSpeed deshabilitado y configura FSDP2 manualmente después si deseas usarlo.

<img width="1097" height="1278" alt="image" src="https://github.com/user-attachments/assets/60475318-facd-4da1-a2a1-67cecff18e04" />


#### ¿Cuánto tiempo deseas entrenar?

Tendrás que decidir si deseas medir el tiempo de entrenamiento en épocas o pasos. Al final, es prácticamente lo mismo, aunque algunas personas desarrollan una preferencia por uno u otro.

<img width="1136" height="1091" alt="image" src="https://github.com/user-attachments/assets/9146cdcd-f277-45e5-92cb-f74f23039d51" />

#### Compartir tu modelo vía Hugging Face Hub

Opcionalmente, puedes publicar tus checkpoints finales *y* intermedios en [Hugging Face Hub](https://hf.co), pero necesitarás una cuenta; puedes iniciar sesión en el hub vía el asistente o la pestaña Publishing. En cualquier caso, siempre puedes cambiar de opinión y habilitarlo o deshabilitarlo.

Si seleccionas publicar tu modelo, recuerda elegir `Private repo` si no quieres que tu modelo sea accesible al público en general.

<img width="1090" height="859" alt="image" src="https://github.com/user-attachments/assets/d1f86b6b-b0d5-4caa-b3ff-6bd106928094" />

#### Validaciones del modelo

Si quieres que el trainer genere imágenes periódicamente, puedes configurar un único prompt de validación en este punto del asistente. Una biblioteca de prompts múltiples se puede configurar en la pestaña `Validations & Output` después de completar el asistente.

¿Quieres externalizar la validación a tu propio script o servicio? Cambia el **Validation Method** a `external-script` en la pestaña de validación después del asistente y proporciona `--validation_external_script`. Puedes pasar contexto de entrenamiento al script con placeholders como `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}` y cualquier valor de configuración `validation_*` (p. ej., `validation_num_inference_steps`, `validation_guidance`, `validation_noise_scheduler`). Habilita `--validation_external_background` para lanzar el script sin bloquear el entrenamiento.

¿Necesitas un hook en el momento en que un checkpoint se guarda en disco? Usa `--post_checkpoint_script` para ejecutar un script justo después de cada guardado (antes de que comiencen las subidas). Acepta los mismos placeholders, dejando `{remote_checkpoint_path}` vacío.

Si quieres mantener los proveedores de publicación integrados de SimpleTuner (o subidas a Hugging Face Hub) pero aún disparar tu propia automatización con la URL remota, usa `--post_upload_script` en su lugar. Se ejecuta una vez por subida con placeholders `{remote_checkpoint_path}`, `{local_checkpoint_path}`, `{global_step}`, `{tracker_run_name}`, `{tracker_project_name}`, `{model_family}`, `{huggingface_path}`. SimpleTuner no captura la salida del script: emite cualquier actualización del tracker directamente desde tu script.

Ejemplo de hook:

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

Donde `notify.sh` publica la URL en tu API web de seguimiento. Siéntete libre de adaptar a Slack, dashboards personalizados o cualquier otra integración.

Ejemplo funcional: `simpletuner/examples/external-validation/replicate_post_upload.py` demuestra el uso de `{remote_checkpoint_path}`, `{model_family}`, `{model_type}`, `{lora_type}` y `{huggingface_path}` para activar una inferencia de Replicate después de las subidas.

Otro ejemplo: `simpletuner/examples/external-validation/wavespeed_post_upload.py` llama la API de WaveSpeed y hace polling del resultado usando los mismos placeholders.

Ejemplo enfocado en Flux: `simpletuner/examples/external-validation/fal_post_upload.py` llama el endpoint de Flux LoRA en fal.ai; requiere `FAL_KEY` y solo se ejecuta cuando `model_family` incluye `flux`.

Ejemplo de GPU local: `simpletuner/examples/external-validation/use_second_gpu.py` ejecuta inferencia de Flux LoRA en otra GPU (por defecto `cuda:1`) y puede usarse incluso cuando no ocurren subidas.

<img width="1101" height="1357" alt="image" src="https://github.com/user-attachments/assets/97bdd3f1-b54c-4087-b4d5-05da8b271751" />

#### Registro de estadísticas de entrenamiento

SimpleTuner admite múltiples APIs destino si deseas enviar tus estadísticas de entrenamiento a una.

Nota: Ninguno de tus datos personales, logs de entrenamiento, captions o datos se envía **jamás** a los desarrolladores del proyecto SimpleTuner. El control de tus datos está en **tus** manos.

<img width="1099" height="1067" alt="image" src="https://github.com/user-attachments/assets/c9be9a20-12ad-402a-9605-66ba5771e630" />

#### Configuración de datasets

En este punto, puedes decidir si conservar algún dataset existente o crear una nueva configuración (dejando las demás intactas) mediante el Asistente de Creación de Datasets, que aparecerá al hacer clic.

<img width="1103" height="877" alt="image" src="https://github.com/user-attachments/assets/3d3cc391-52ed-422e-a4a1-676ca342df10" />

##### Asistente de datasets

Si elegiste crear un nuevo dataset, verás el siguiente asistente, que te guiará para agregar un dataset local o en la nube.

<img width="1110" height="857" alt="image" src="https://github.com/user-attachments/assets/3719e0f5-774e-461d-be02-902e08a679f6" />

<img width="1082" height="1255" alt="image" src="https://github.com/user-attachments/assets/ac38a3de-364a-447f-a734-cab2bdd5338d" />

Para un dataset local, podrás usar el botón **Browse directories** para acceder a un modal de navegador de datasets.

<img width="1201" height="1160" alt="image" src="https://github.com/user-attachments/assets/66a333d0-30fa-45d1-a5b2-1e859d789677" />

Si configuraste correctamente el directorio de datasets durante el onboarding, verás tu contenido aquí.

Haz clic en el directorio que deseas agregar y luego en **Select Directory**.

<img width="907" height="709" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

Después de esto, se te guiará para configurar valores de resolución y recorte.

**NOTA**: SimpleTuner no *hace upscaling* de imágenes, así que asegúrate de que sean al menos tan grandes como tu resolución configurada.

Cuando llegues al paso para configurar tus captions, **considera cuidadosamente** qué opción es la correcta.

Si solo quieres usar una palabra gatillo, esa sería la opción **Instance Prompt**.

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### Opcional: subir un dataset desde tu navegador

Si tus imágenes y captions aún no están en la máquina, el asistente de datasets ahora incluye un botón **Upload** junto a **Browse directories**. Puedes:

- Crear una subcarpeta nueva en tu directorio de datasets configurado y subir archivos individuales o un ZIP (se aceptan imágenes más metadatos .txt/.jsonl/.csv).
- Dejar que SimpleTuner extraiga el ZIP en esa carpeta (dimensionado para backends locales; archivos muy grandes son rechazados).
- Seleccionar de inmediato la carpeta recién cargada en el navegador y continuar el asistente sin salir de la UI.

#### Tasa de aprendizaje, tamaño de batch y optimizador

Una vez que completes el asistente de datasets (o si decides conservar tus datasets existentes), se te ofrecerán presets para optimizador/tasa de aprendizaje y tamaño de batch.

Estos son solo puntos de partida que ayudan a los nuevos usuarios a tomar mejores decisiones en sus primeras ejecuciones. Para usuarios experimentados, usa **Manual configuration** para control total.

**NOTA**: Si planeas usar DeepSpeed más adelante, la elección del optimizador no importa demasiado aquí.

##### Consideraciones de tamaño de batch multi-GPU

Al entrenar con múltiples GPUs, ten en cuenta que tu dataset debe soportar el **tamaño de batch efectivo**:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

Si tu dataset es más pequeño que este valor, SimpleTuner lanzará un error con guía específica. Puedes:
- Reducir el tamaño de batch
- Aumentar el valor `repeats` en la configuración del dataset
- Habilitar **Allow Dataset Oversubscription** en la configuración avanzada para ajustar automáticamente los repeats

Consulta [DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing) para más detalles sobre el dimensionamiento de datasets.

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### Presets de optimización de memoria

Para una configuración más fácil en hardware de consumo, cada modelo incluye presets personalizados que permiten seleccionar ahorro de memoria ligero, balanceado o agresivo.

En la sección **Memory Optimisation** de la pestaña **Training**, encontrarás el botón **Load Presets**:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

Que abre esta interfaz:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### Revisar y guardar

Si estás conforme con todos los valores seleccionados, haz clic en **Finish** para terminar el asistente.

Luego verás tu nuevo entorno seleccionado activamente y listo para entrenar.

En la mayoría de los casos, estos ajustes serán todo lo que necesitas configurar. Puede que quieras agregar datasets extra o ajustar otros parámetros.

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

En la página **Environment**, verás el trabajo de entrenamiento recién configurado y botones para descargar o duplicar la configuración si deseas usarla como plantilla.

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**NOTA**: El entorno **Default** es especial y no se recomienda usarlo como entorno de entrenamiento general; su configuración puede combinarse automáticamente en cualquier entorno que habilite la opción **Use environment defaults**:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
