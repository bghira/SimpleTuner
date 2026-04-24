# Tutorial del WebUI de SimpleTuner

## Introducción

Este tutorial te ayudará a comenzar con la interfaz web de SimpleTuner.

## Instalación de requisitos

Para sistemas Ubuntu, comienza instalando los paquetes requeridos:

```bash
apt -y install python3.13-venv python3.13-dev
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
python3.13 -m venv .venv
. .venv/bin/activate
```

### Dependencias específicas de CUDA

Los usuarios de NVIDIA deberán usar los extras de CUDA para obtener todas las dependencias correctas:

```bash
pip install -e 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install -e 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
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

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/4be22081-f13d-4aed-a87c-2313ddefc8a4" />

##### Migrar desde uso en línea de comandos

Si antes usabas SimpleTuner sin WebUI, puedes apuntar a tu carpeta config/ existente y todos tus entornos se auto-descubrirán.

Para usuarios nuevos, la ubicación predeterminada de tus configs y datasets será `~/.simpletuner/` y se recomienda mover tus datasets a un lugar con más espacio:

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/c5b3ab53-654e-4a9b-8e2d-7951f11619ef" />


#### Selección y configuración de (multi-)GPU

Después de configurar las rutas predeterminadas, llegarás a un paso donde se puede configurar multi-GPU (en la imagen, un sistema NVIDIA)

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/61d5a7bc-0a02-4a0a-8df0-207cce4b7bc1" />

Si tienes varias GPUs y quieres usar solo la segunda, aquí es donde puedes hacerlo.

> **Nota para usuarios multi-GPU:** Al entrenar con múltiples GPUs, los requisitos de tamaño del dataset aumentan proporcionalmente. El tamaño de batch efectivo se calcula como `train_batch_size × num_gpus × gradient_accumulation_steps`. Si tu dataset es más pequeño que este valor, tendrás que aumentar el ajuste `repeats` en la configuración del dataset o habilitar la opción `--allow_dataset_oversubscription` en la configuración avanzada. Consulta la [sección de tamaño de batch](#consideraciones-de-tamaño-de-batch-multi-gpu) más abajo para más detalles.

#### Crear tu primer entorno de entrenamiento

Si no se encontraron configuraciones previas en tu `configs_dir`, se te pedirá crear **tu primer entorno de entrenamiento**:

<img width="500" height="640" alt="image" src="https://github.com/user-attachments/assets/2110287a-16fd-4f87-893b-86d2a555a10f" />

Usa **Bootstrap From Example** para seleccionar un ejemplo de configuración como base, o simplemente ingresa un nombre descriptivo y crea un entorno aleatorio si prefieres usar un asistente de configuración.

### Cambiar entre entornos de entrenamiento

Si tenías entornos de configuración preexistentes, aparecerán en este menú desplegable.

De lo contrario, la opción que creamos durante el onboarding ya estará seleccionada y activa.

<img width="448" height="225" alt="image" src="https://github.com/user-attachments/assets/66fef6a9-2040-47fd-b22d-918470677992" />

Usa **Manage Configs** para ir a la pestaña `Environment`, donde encontrarás una lista de tus entornos, configuraciones de dataloader y otras.

### Asistente de configuración

He trabajado para ofrecer un asistente de configuración completo que te ayudará a configurar algunos de los ajustes más importantes con un arranque directo y sin rodeos.

<img width="470" height="358" alt="image" src="https://github.com/user-attachments/assets/e4bf1a4e-716c-4101-b753-e9e24bb42d8a" />

En el menú de navegación superior izquierdo, el botón Wizard te llevará a un diálogo de selección:

<img width="448" height="440" alt="image" src="https://github.com/user-attachments/assets/68324fa8-3ca9-45b1-b947-1e7738fd1d8c" />

Luego se ofrecen todas las variantes de modelos integradas. Cada variante prehabilitará ajustes necesarios como Attention Masking o límites de tokens extendidos.

#### Opciones de modelo LoRA

Si deseas entrenar un LoRA, podrás configurar aquí las opciones de cuantización del modelo.

En general, salvo que estés entrenando un modelo tipo Stable Diffusion, se recomienda int8-quanto porque no afectará la calidad y permite tamaños de batch más altos.

Algunos modelos pequeños como Cosmos2, Sana y PixArt no se llevan bien con la cuantización.

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/c2e721f2-b4da-4cd0-84fd-7ac81993e87c" />

#### Entrenamiento full-rank

Se desaconseja el entrenamiento full-rank, ya que generalmente tarda más y cuesta más recursos que un LoRA/LyCORIS con el mismo dataset.

Sin embargo, si deseas entrenar un checkpoint completo, puedes configurar aquí las etapas DeepSpeed ZeRO, que serán necesarias para modelos más grandes como Auraflow, Flux y otros más grandes.

FSDP2 es compatible, pero no configurable en este asistente. Simplemente deja DeepSpeed deshabilitado y configura FSDP2 manualmente después si deseas usarlo.

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/88438f1c-b0a2-4249-afd0-7878aa1abada" />


#### ¿Cuánto tiempo deseas entrenar?

Tendrás que decidir si deseas medir el tiempo de entrenamiento en épocas o pasos. Al final, es prácticamente lo mismo, aunque algunas personas desarrollan una preferencia por uno u otro.

<img width="508" height="475" alt="image" src="https://github.com/user-attachments/assets/dcb54279-0ce7-4c66-a9ab-4dd26f87278c" />

#### Compartir tu modelo vía Hugging Face Hub

Opcionalmente, puedes publicar tus checkpoints finales *y* intermedios en [Hugging Face Hub](https://hf.co), pero necesitarás una cuenta; puedes iniciar sesión en el hub vía el asistente o la pestaña Publishing. En cualquier caso, siempre puedes cambiar de opinión y habilitarlo o deshabilitarlo.

Si seleccionas publicar tu modelo, recuerda elegir `Private repo` si no quieres que tu modelo sea accesible al público en general.

<img width="508" height="370" alt="image" src="https://github.com/user-attachments/assets/8d2d282b-e66f-48a8-a40e-4e4ecc2d280b" />

#### Frecuencia de checkpoints

Durante el entrenamiento, tu modelo se guardará periódicamente en disco. Conservar más checkpoints requiere más espacio en disco.

Los checkpoints permiten reanudar el entrenamiento más adelante sin repetir todos los pasos. Conservar varios checkpoints te permite probar distintas versiones del modelo y quedarte con la que mejor funcione.

Se recomienda guardar un checkpoint cada 10%, aunque depende de la cantidad de datos con la que entrenes. Con un dataset pequeño, conviene guardar con frecuencia para asegurarte de no estar sobreajustando.

Los datasets extremadamente grandes se benefician de intervalos de checkpoint más largos para evitar perder tiempo escribiéndolos en disco.

<img width="508" height="485" alt="image" src="https://github.com/user-attachments/assets/c7b1cd0b-a1b9-47ec-87f9-1ecac2e0841a" />

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

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/e699ba57-526b-4f60-9e8c-0ba410761c9f" />

#### Registro de estadísticas de entrenamiento

SimpleTuner admite múltiples APIs destino si deseas enviar tus estadísticas de entrenamiento a una.

Nota: Ninguno de tus datos personales, logs de entrenamiento, captions o datos se envía **jamás** a los desarrolladores del proyecto SimpleTuner. El control de tus datos está en **tus** manos.

<img width="508" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8d15c5-456f-4637-af7e-c2f5f31cb968" />

#### Configuración de datasets

En este punto, puedes decidir si conservar algún dataset existente o crear una nueva configuración (dejando las demás intactas) mediante el Asistente de Creación de Datasets, que aparecerá al hacer clic.

<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/b5a7f883-e180-4662-b84c-fff609c6b1df" />

##### Asistente de datasets

Si elegiste crear un nuevo dataset, verás el siguiente asistente, que te guiará para agregar un dataset local o en la nube.

<img width="508" height="332" alt="image" src="https://github.com/user-attachments/assets/c523930b-563e-4b5d-b104-8e7ce4658b2c" />

<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/c263f58e-fd85-437e-811a-967b94e309fd" />

Para un dataset local, podrás usar el botón **Browse directories** para acceder a un modal de navegador de datasets.

<img width="396" height="576" alt="image" src="https://github.com/user-attachments/assets/14c51685-3559-4d16-be59-ed4b0959ca32" />

Si configuraste correctamente el directorio de datasets durante el onboarding, verás tu contenido aquí.

Haz clic en el directorio que deseas agregar y luego en **Select Directory**.

<img width="454" height="356" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

Después de esto, se te guiará para configurar valores de resolución y recorte.

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
