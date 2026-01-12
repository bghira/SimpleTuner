# Guía rápida de Flux[dev] / Flux[schnell]

![image](https://github.com/user-attachments/assets/6409d790-3bb4-457c-a4b4-a51a45fc91d1)

En este ejemplo, entrenaremos un LoRA Flux.1 Krea.

## Requisitos de hardware

Flux requiere mucha **RAM del sistema** además de memoria de GPU. Solo cuantizar el modelo al inicio requiere alrededor de 50GB de memoria del sistema. Si tarda demasiado tiempo, podrías necesitar evaluar las capacidades de tu hardware y si se requieren cambios.

Cuando entrenas cada componente de un LoRA de rango 16 (MLP, proyecciones, bloques multimodales), termina usando:

- un poco más de 30G de VRAM cuando no se cuantiza el modelo base
- un poco más de 18G de VRAM al cuantizar a int8 + pesos base/LoRA en bf16
- un poco más de 13G de VRAM al cuantizar a int4 + pesos base/LoRA en bf16
- un poco más de 9G de VRAM al cuantizar a NF4 + pesos base/LoRA en bf16
- un poco más de 9G de VRAM al cuantizar a int2 + pesos base/LoRA en bf16

Necesitarás:

- **el mínimo absoluto** es una sola **3080 10G**
- **un mínimo realista** es una sola 3090 o V100
- **idealmente** varias 4090, A6000, L40S o mejor

Por suerte, estas están disponibles a través de proveedores como [LambdaLabs](https://lambdalabs.com) que ofrece las tarifas más bajas disponibles y clústeres localizados para entrenamiento multinodo.

**A diferencia de otros modelos, las GPUs de Apple actualmente no funcionan para entrenar Flux.**


## Requisitos previos

Asegúrate de tener python instalado; SimpleTuner funciona bien con 3.10 a 3.12.

Puedes comprobarlo ejecutando:

```bash
python --version
```

Si no tienes python 3.12 instalado en Ubuntu, puedes intentar lo siguiente:

```bash
apt -y install python3.12 python3.12-venv
```

### Dependencias de la imagen de contenedor

Para Vast, RunPod y TensorDock (entre otros), lo siguiente funcionará en una imagen CUDA 12.2-12.8 para habilitar la compilación de extensiones CUDA:

```bash
apt -y install nvidia-cuda-toolkit
```

## Instalación

Instala SimpleTuner vía pip:

```bash
pip install 'simpletuner[cuda]'
```

Para instalación manual o entorno de desarrollo, consulta la [documentación de instalación](../INSTALL.md).

### Pasos adicionales para AMD ROCm

Lo siguiente debe ejecutarse para que una AMD MI300X sea utilizable:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install .
popd
```

## Configuración del entorno

### Método de interfaz web

La WebUI de SimpleTuner hace que la configuración sea bastante sencilla. Para ejecutar el servidor:

```bash
simpletuner server
```

Esto creará un servidor web en el puerto 8001 por defecto, al que puedes acceder visitando http://localhost:8001.

### Método manual / línea de comandos

Para ejecutar SimpleTuner mediante herramientas de línea de comandos, necesitarás configurar un archivo de configuración, los directorios del dataset y del modelo, y un archivo de configuración del dataloader.

#### Archivo de configuración

Un script experimental, `configure.py`, puede permitirte omitir por completo esta sección mediante una configuración interactiva paso a paso. Incluye funciones de seguridad que ayudan a evitar errores comunes.

**Nota:** Esto no configura tu dataloader. Aún tendrás que hacerlo manualmente más adelante.

Para ejecutarlo:

```bash
simpletuner configure
```

> ⚠️ Para usuarios ubicados en países donde Hugging Face Hub no es fácilmente accesible, debes agregar `HF_ENDPOINT=https://hf-mirror.com` a tu `~/.bashrc` o `~/.zshrc` dependiendo de cuál `$SHELL` use tu sistema.

Si prefieres configurarlo manualmente:

Copia `config/config.json.example` a `config/config.json`:

```bash
cp config/config.json.example config/config.json
```

Allí, probablemente necesitarás modificar las siguientes variables:

- `model_type` - Configúralo en `lora`.
- `model_family` - Configúralo en `flux`.
- `model_flavour` - esto es `krea` por defecto, pero puede configurarse en `dev` para entrenar el lanzamiento original FLUX.1-Dev.
  - `krea` - El modelo FLUX.1-Krea [dev] por defecto, una variante de pesos abiertos de Krea 1, un modelo propietario en colaboración entre BFL y Krea.ai
  - `dev` - Sabor de modelo Dev, el anterior valor predeterminado
  - `schnell` - Sabor de modelo Schnell; la guía rápida configura automáticamente el calendario de ruido rápido y el stack de LoRA asistente
  - `kontext` - Entrenamiento Kontext (ver [esta guía](../quickstart/FLUX_KONTEXT.md) para orientación específica)
  - `fluxbooru` - Un modelo des-destilado (requiere CFG) basado en FLUX.1-Dev llamado [FluxBooru](https://hf.co/terminusresearch/fluxbooru-v0.3), creado por el grupo de investigación terminus
  - `libreflux` - Un modelo des-destilado basado en FLUX.1-Schnell que requiere enmascaramiento de atención en las entradas del codificador de texto T5
- `offload_during_startup` - Configúralo en `true` si te quedas sin memoria durante las codificaciones del VAE.
- `pretrained_model_name_or_path` - Configúralo en `black-forest-labs/FLUX.1-dev`.
- `pretrained_vae_model_name_or_path` - Configúralo en `black-forest-labs/FLUX.1-dev`.
  - Ten en cuenta que necesitarás iniciar sesión en Huggingface y tener acceso para descargar este modelo. Más adelante en este tutorial veremos cómo iniciar sesión en Huggingface.
- `output_dir` - Configúralo al directorio donde quieres guardar tus checkpoints y las imágenes de validación. Se recomienda usar una ruta completa aquí.
- `train_batch_size` - esto debe mantenerse en 1, especialmente si tienes un dataset muy pequeño.
- `validation_resolution` - Como Flux es un modelo de 1024px, puedes configurarlo en `1024x1024`.
  - Además, Flux fue ajustado en buckets multi-aspecto, y otras resoluciones pueden especificarse separándolas con comas: `1024x1024,1280x768,2048x2048`
- `validation_guidance` - Usa el valor que suelas seleccionar en inferencia para Flux.
- `validation_guidance_real` - Usa >1.0 para usar CFG en inferencia de Flux. Ralentiza las validaciones, pero produce mejores resultados. Funciona mejor con un `VALIDATION_NEGATIVE_PROMPT` vacío.
- `validation_num_inference_steps` - Usa alrededor de 20 para ahorrar tiempo y aún ver calidad decente. Flux no es muy diverso y más pasos podrían solo perder tiempo.
- `--lora_rank=4` si quieres reducir sustancialmente el tamaño del LoRA entrenado. Esto puede ayudar con el uso de VRAM.
- Las ejecuciones LoRA de Schnell usan el calendario rápido automáticamente mediante los valores por defecto de la guía rápida; no se necesitan flags adicionales.

- `gradient_accumulation_steps` - La guía previa era evitar esto con entrenamiento bf16 ya que degradaría el modelo. Pruebas adicionales mostraron que esto no necesariamente es el caso para Flux.
  - Esta opción hace que los pasos de actualización se acumulen durante varios pasos. Esto aumentará el tiempo de entrenamiento linealmente; un valor de 2 hará que tu ejecución de entrenamiento sea la mitad de rápida y tome el doble de tiempo.
- `optimizer` - A los principiantes se les recomienda quedarse con adamw_bf16, aunque optimi-lion y optimi-stableadamw también son buenas opciones.
- `mixed_precision` - Los principiantes deberían mantener esto en `bf16`.
- `gradient_checkpointing` - configúralo en true prácticamente en todas las situaciones y en todos los dispositivos.
- `gradient_checkpointing_interval` - esto podría configurarse en un valor de 2 o superior en GPUs más grandes para hacer checkpoint solo cada _n_ bloques. Un valor de 2 haría checkpoint de la mitad de los bloques, y 3 de un tercio.

### Funciones experimentales avanzadas

<details>
<summary>Mostrar detalles experimentales avanzados</summary>


SimpleTuner incluye funciones experimentales que pueden mejorar significativamente la estabilidad y el rendimiento del entrenamiento.

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** reduce el sesgo de exposición y mejora la calidad de la salida al permitir que el modelo genere sus propias entradas durante el entrenamiento.

> ⚠️ Estas funciones aumentan la sobrecarga computacional del entrenamiento.

</details>

### Offloading de memoria (opcional)

Flux admite offloading de módulos agrupados mediante diffusers v0.33+. Esto reduce drásticamente la presión de VRAM cuando el cuello de botella son los pesos del transformer. Puedes habilitarlo agregando los siguientes flags a `TRAINER_EXTRA_ARGS` (o en la página Hardware de la WebUI):

```bash
--enable_group_offload \
--group_offload_type block_level \
--group_offload_blocks_per_group 1 \
--group_offload_use_stream \
# optional: spill offloaded weights to disk instead of RAM
# --group_offload_to_disk_path /fast-ssd/simpletuner-offload
```

- `--group_offload_use_stream` solo es efectivo en dispositivos CUDA; SimpleTuner desactiva automáticamente los streams en ROCm, MPS y backends CPU.
- No combines esto con `--enable_model_cpu_offload` — las dos estrategias son mutuamente excluyentes.
- Cuando uses `--group_offload_to_disk_path`, prefiere un SSD/NVMe local rápido.

#### Prompts de validación

Dentro de `config/config.json` está el "prompt de validación principal", que suele ser el instance_prompt principal en el que estás entrenando para tu único sujeto o estilo. Además, se puede crear un archivo JSON que contiene prompts adicionales para ejecutar durante las validaciones.

El archivo de ejemplo `config/user_prompt_library.json.example` contiene el siguiente formato:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "nickname": "the prompt goes here",
  "another_nickname": "another prompt goes here"
}
```
</details>

Los apodos son el nombre de archivo de la validación, así que mantenlos cortos y compatibles con tu sistema de archivos.

Para indicar al entrenador esta librería de prompts, añádela a TRAINER_EXTRA_ARGS agregando una nueva línea al final de `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
  "--user_prompt_library": "config/user_prompt_library.json",
```
</details>

Un conjunto de prompts diverso ayudará a determinar si el modelo colapsa a medida que entrena. En este ejemplo, la palabra `<token>` debe reemplazarse por el nombre de tu sujeto (instance_prompt).

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "anime_<token>": "a breathtaking anime-style portrait of <token>, capturing her essence with vibrant colors and expressive features",
    "chef_<token>": "a high-quality, detailed photograph of <token> as a sous-chef, immersed in the art of culinary creation",
    "just_<token>": "a lifelike and intimate portrait of <token>, showcasing her unique personality and charm",
    "cinematic_<token>": "a cinematic, visually stunning photo of <token>, emphasizing her dramatic and captivating presence",
    "elegant_<token>": "an elegant and timeless portrait of <token>, exuding grace and sophistication",
    "adventurous_<token>": "a dynamic and adventurous photo of <token>, captured in an exciting, action-filled moment",
    "mysterious_<token>": "a mysterious and enigmatic portrait of <token>, shrouded in shadows and intrigue",
    "vintage_<token>": "a vintage-style portrait of <token>, evoking the charm and nostalgia of a bygone era",
    "artistic_<token>": "an artistic and abstract representation of <token>, blending creativity with visual storytelling",
    "futuristic_<token>": "a futuristic and cutting-edge portrayal of <token>, set against a backdrop of advanced technology",
    "woman": "a beautifully crafted portrait of a woman, highlighting her natural beauty and unique features",
    "man": "a powerful and striking portrait of a man, capturing his strength and character",
    "boy": "a playful and spirited portrait of a boy, capturing youthful energy and innocence",
    "girl": "a charming and vibrant portrait of a girl, emphasizing her bright personality and joy",
    "family": "a heartwarming and cohesive family portrait, showcasing the bonds and connections between loved ones"
}
```
</details>

> ℹ️ Flux es un modelo de flow-matching y los prompts más cortos que tengan fuertes similitudes darán prácticamente la misma imagen producida por el modelo. Asegúrate de usar prompts más largos y descriptivos.

#### Seguimiento de puntuaciones CLIP

Si deseas habilitar evaluaciones para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/CLIP_SCORES.md) para información sobre cómo configurar e interpretar las puntuaciones CLIP.

# Pérdida de evaluación estable

Si deseas usar pérdida MSE estable para puntuar el rendimiento del modelo, consulta [este documento](../evaluation/EVAL_LOSS.md) para información sobre cómo configurar e interpretar la pérdida de evaluación.

#### Vistas previas de validación

SimpleTuner admite la transmisión de vistas previas de validación intermedias durante la generación usando modelos Tiny AutoEncoder. Esto te permite ver las imágenes de validación generándose paso a paso en tiempo real mediante callbacks de webhook.

Para habilitarlo:
<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "validation_preview": true,
  "validation_preview_steps": 1
}
```
</details>

**Requisitos:**
- Configuración de webhook
- Validación habilitada

Configura `validation_preview_steps` en un valor más alto (p. ej., 3 o 5) para reducir la sobrecarga del Tiny AutoEncoder. Con `validation_num_inference_steps=20` y `validation_preview_steps=5`, recibirás imágenes de vista previa en los pasos 5, 10, 15 y 20.

#### Desplazamiento del calendario temporal de Flux

Los modelos de flow-matching como Flux y SD3 tienen una propiedad llamada "shift" que permite desplazar la parte entrenada del calendario de timesteps usando un simple valor decimal.

##### Valores predeterminados

Por defecto, no se aplica desplazamiento de calendario a flux, lo que da una forma de campana sigmoide a la distribución de muestreo de timesteps. Es poco probable que sea el enfoque ideal para Flux, pero resulta en mayor aprendizaje en menos tiempo que el auto-shift.

##### Auto-shift

Un enfoque comúnmente recomendado es seguir varios trabajos recientes y habilitar el shift de timesteps dependiente de la resolución, `--flow_schedule_auto_shift`, que usa valores de shift más altos para imágenes grandes y valores más bajos para imágenes pequeñas. Esto da resultados de entrenamiento estables pero potencialmente mediocres.

##### Especificación manual

(_Gracias a General Awareness de Discord por los siguientes ejemplos_)

Al usar un valor `--flow_schedule_shift` de 0.1 (muy bajo), solo se ven afectados los detalles finos de la imagen:
![image](https://github.com/user-attachments/assets/991ca0ad-e25a-4b13-a3d6-b4f2de1fe982)

Al usar un valor `--flow_schedule_shift` de 4.0 (muy alto), se ven afectados los grandes rasgos compositivos y potencialmente el espacio de color del modelo:
![image](https://github.com/user-attachments/assets/857a1f8a-07ab-4b75-8e6a-eecff616a28d)

#### Entrenamiento con modelo cuantizado

Probado en sistemas Apple y NVIDIA, Hugging Face Optimum-Quanto puede usarse para reducir la precisión y los requisitos de VRAM, entrenando Flux con solo 16GB.

Para usuarios de `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "no_change",
  "text_encoder_2_precision": "no_change",
  "lora_rank": 16,
  "max_grad_norm": 1.0,
  "base_model_default_dtype": "bf16"
```
</details>

##### Ajustes específicos de LoRA (no LyCORIS)

```bash
# When training 'mmdit', we find very stable training that makes the model take longer to learn.
# When training 'all', we can easily shift the model distribution, but it is more prone to forgetting and benefits from high quality data.
# When training 'all+ffs', all attention layers are trained in addition to the feed-forward which can help with adapting the model objective for the LoRA.
# - This mode has been reported to lack portability, and platforms such as ComfyUI might not be able to load the LoRA.
# The option to train only the 'context' blocks is offered as well, but its impact is unknown, and is offered as an experimental choice.
# - An extension to this mode, 'context+ffs' is also available, which is useful for pretraining new tokens into a LoRA before continuing finetuning it via `--init_lora`.
# Other options include 'tiny' and 'nano' which train just 1 or 2 layers.
"--flux_lora_target": "all",

# If you want to use LoftQ initialisation, you can't use Quanto to quantise the base model.
# This possibly offers better/faster convergence, but only works on NVIDIA devices and requires Bits n Bytes and is incompatible with Quanto.
# Other options are 'default', 'gaussian' (difficult), and untested options: 'olora' and 'pissa'.
"--lora_init_type": "loftq",
```

#### Consideraciones del dataset

> ⚠️ La calidad de imagen para el entrenamiento es más importante para Flux que para la mayoría de los otros modelos, ya que absorberá los artefactos en tus imágenes _primero_, y luego aprenderá el concepto/sujeto.

Es crucial contar con un dataset sustancial para entrenar tu modelo. Hay limitaciones en el tamaño del dataset, y debes asegurarte de que sea lo suficientemente grande para entrenar tu modelo de forma efectiva. Ten en cuenta que el tamaño mínimo del dataset es `train_batch_size * gradient_accumulation_steps` y también mayor que `vae_batch_size`. El dataset no será utilizable si es demasiado pequeño.

> ℹ️ Con pocas imágenes, podrías ver el mensaje **no images detected in dataset** - aumentar el valor de `repeats` superará esta limitación.

Dependiendo del dataset que tengas, necesitarás configurar el directorio del dataset y el archivo de configuración del dataloader de manera diferente. En este ejemplo, usaremos [pseudo-camera-10k](https://huggingface.co/datasets/bghira/pseudo-camera-10k) como dataset.

Crea un documento `--data_backend_config` (`config/multidatabackend.json`) que contenga esto:

<details>
<summary>Ver ejemplo de config</summary>

```json
[
  {
    "id": "pseudo-camera-10k-flux",
    "type": "local",
    "crop": true,
    "crop_aspect": "square",
    "crop_style": "center",
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/pseudo-camera-10k",
    "instance_data_dir": "datasets/pseudo-camera-10k",
    "disabled": false,
    "skip_file_discovery": "",
    "caption_strategy": "filename",
    "metadata_backend": "discovery",
    "repeats": 0,
    "is_regularisation_data": true
  },
  {
    "id": "dreambooth-subject",
    "type": "local",
    "crop": false,
    "resolution": 1024,
    "minimum_image_size": 1024,
    "maximum_image_size": 1024,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "dreambooth-subject-512",
    "type": "local",
    "crop": false,
    "resolution": 512,
    "minimum_image_size": 512,
    "maximum_image_size": 512,
    "target_downsample_size": 512,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/flux/dreambooth-subject-512",
    "instance_data_dir": "datasets/dreambooth-subject",
    "caption_strategy": "instanceprompt",
    "instance_prompt": "the name of your subject goes here",
    "metadata_backend": "discovery",
    "repeats": 1000
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/flux",
    "disabled": false,
    "write_batch_size": 128
  }
]
```
</details>

> Consulta las opciones y requisitos de caption_strategy en [DATALOADER.md](../DATALOADER.md#caption_strategy).

> ℹ️ Se admite ejecutar datasets de 512px y 1024px de forma concurrente, y podría resultar en mejor convergencia para Flux.

Luego, crea un directorio `datasets`:

```bash
mkdir -p datasets
pushd datasets
    huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k --local-dir=pseudo-camera-10k
    mkdir dreambooth-subject
    # place your images into dreambooth-subject/ now
popd
```

Esto descargará alrededor de 10k muestras de fotografías a tu directorio `datasets/pseudo-camera-10k`, que se creará automáticamente.

Tus imágenes de Dreambooth deben ir en el directorio `datasets/dreambooth-subject`.

#### Iniciar sesión en WandB y Huggingface Hub

Querrás iniciar sesión en WandB y HF Hub antes de empezar el entrenamiento, especialmente si usas `--push_to_hub` y `--report_to=wandb`.

Si vas a subir elementos a un repositorio Git LFS manualmente, también deberías ejecutar `git config --global credential.helper store`

Ejecuta los siguientes comandos:

```bash
wandb login
```

y

```bash
huggingface-cli login
```

Sigue las instrucciones para iniciar sesión en ambos servicios.

### Ejecutar el entrenamiento

Desde el directorio de SimpleTuner, tienes varias opciones para iniciar el entrenamiento:

**Opción 1 (Recomendada - instalación con pip):**

```bash
pip install 'simpletuner[cuda]'
simpletuner train
```

**Opción 2 (Método de git clone):**

```bash
simpletuner train
```

**Opción 3 (Método heredado - aún funciona):**

```bash
./train.sh
```

Esto iniciará el caché a disco de text embeds y salidas del VAE.

Para más información, consulta los documentos de [dataloader](../DATALOADER.md) y [tutorial](../TUTORIAL.md).

**Nota:** No está claro si el entrenamiento en buckets multi-aspecto funciona correctamente para Flux en este momento. Se recomienda usar `crop_style=random` y `crop_aspect=square`.

## Configuración multi-GPU

SimpleTuner incluye **detección automática de GPU** mediante la WebUI. Durante el onboarding, configurarás:

- **Modo Auto**: usa automáticamente todas las GPUs detectadas con ajustes óptimos
- **Modo Manual**: selecciona GPUs específicas o configura un número de procesos personalizado
- **Modo Deshabilitado**: entrenamiento con una sola GPU

La WebUI detecta tu hardware y configura `--num_processes` y `CUDA_VISIBLE_DEVICES` automáticamente.

Para configuración manual o setups avanzados, consulta la [sección de entrenamiento multi-GPU](../INSTALL.md#multiple-gpu-training) en la guía de instalación.

## Consejos de inferencia

### LoRAs entrenados con CFG (flux_guidance_value > 1)

En ComfyUI, tendrás que pasar Flux por otro nodo llamado AdaptiveGuider. Uno de los miembros de nuestra comunidad ha proporcionado un nodo modificado aquí:

(**enlaces externos**) [IdiotSandwichTheThird/ComfyUI-Adaptive-Guidan...](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps) y su flujo de trabajo de ejemplo [aquí](https://github.com/IdiotSandwichTheThird/ComfyUI-Adaptive-Guidance-with-disabled-init-steps/blob/master/ExampleWorkflow.json)

### LoRA destilado con CFG (flux_guidance_scale == 1)

Inferir el LoRA destilado con CFG es tan sencillo como usar un guidance_scale más bajo alrededor del valor con el que fue entrenado.

## Notas y consejos de solución de problemas

### Configuración de VRAM más baja

Actualmente, el uso de VRAM más bajo (9090M) puede alcanzarse con:

- OS: Ubuntu Linux 24
- GPU: Un solo dispositivo NVIDIA CUDA (10G, 12G)
- Memoria del sistema: alrededor de 50G de memoria del sistema
- Precisión del modelo base: `nf4-bnb`
- Optimizador: Lion 8Bit Paged, `bnb-lion8bit-paged`
- Resolución: 512px
  - 1024px requiere >= 12G de VRAM
- Tamaño de lote: 1, cero pasos de acumulación de gradiente
- DeepSpeed: deshabilitado / sin configurar
- PyTorch: 2.6 Nightly (build del 29 de septiembre)
- Usar `--quantize_via=cpu` para evitar error outOfMemory durante el arranque en tarjetas <=16G.
- Con `--attention_mechanism=sageattention` para reducir aún más la VRAM en 0.1GB y mejorar la velocidad de generación de imágenes de validación en el entrenamiento.
- Asegúrate de habilitar `--gradient_checkpointing` o nada evitará que haga OOM

**NOTA**: El pre-caché de embeddings del VAE y salidas del codificador de texto puede usar más memoria y aún así producir OOM. Si eso ocurre, la cuantización del codificador de texto y el tiling del VAE pueden habilitarse vía `--vae_enable_tiling=true`. Se puede ahorrar más memoria al inicio con `--offload_during_startup=true`.

La velocidad fue de aproximadamente 1.4 iteraciones por segundo en una 4090.

### SageAttention

Al usar `--attention_mechanism=sageattention`, la inferencia puede acelerarse en tiempo de validación.

**Nota**: Esto no es compatible con _todas_ las configuraciones de modelo, pero vale la pena probarlo.

### Entrenamiento cuantizado con NF4

En términos simples, NF4 es una representación de 4 bits _aprox_ del modelo, lo que significa que el entrenamiento tiene serias preocupaciones de estabilidad que abordar.

En pruebas tempranas, se cumple lo siguiente:

- El optimizador Lion provoca colapso del modelo pero usa menos VRAM; las variantes de AdamW ayudan a mantenerlo estable; bnb-adamw8bit, adamw_bf16 son excelentes opciones
  - AdEMAMix no fue bien, pero los ajustes no se exploraron
- `--max_grad_norm=0.01` ayuda aún más a reducir la ruptura del modelo al evitar cambios enormes en muy poco tiempo
- NF4, AdamW8bit y un tamaño de lote más alto ayudan a superar los problemas de estabilidad, a costa de más tiempo de entrenamiento o VRAM usada
- Subir la resolución de 512px a 1024px ralentiza el entrenamiento de, por ejemplo, 1.4 segundos por paso a 3.5 segundos por paso (tamaño de lote 1, 4090)
- Todo lo que es difícil de entrenar en int8 o bf16 se vuelve más difícil en NF4
- Es menos compatible con opciones como SageAttention

NF4 no funciona con torch.compile, así que cualquier velocidad que obtengas es la que hay.

Si la VRAM no es una preocupación (p. ej., 48G o más), entonces int8 con torch.compile es tu mejor opción más rápida.

### Pérdida con máscara

Si estás entrenando un sujeto o estilo y te gustaría enmascarar uno u otro, consulta la sección de [entrenamiento con pérdida enmascarada](../DREAMBOOTH.md#masked-loss) de la guía de Dreambooth.

### Entrenamiento TREAD

> ⚠️ **Experimental**: TREAD es una función recién implementada. Aunque es funcional, las configuraciones óptimas aún se están explorando.

[TREAD](../TREAD.md) (paper) significa **T**oken **R**outing for **E**fficient **A**rchitecture-agnostic **D**iffusion. Es un método que puede acelerar el entrenamiento de Flux al enrutar tokens de forma inteligente a través de capas del transformer. La aceleración es proporcional a cuántos tokens eliminas.

#### Configuración rápida

Agrega esto a tu `config.json`:

<details>
<summary>Ver ejemplo de config</summary>

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": -2
      }
    ]
  }
}
```
</details>

Esta configuración:

- Mantiene solo el 50% de los tokens de imagen durante las capas 2 hasta la penúltima
- Los tokens de texto nunca se eliminan
- Acelera el entrenamiento en ~25% con impacto mínimo en la calidad

#### Puntos clave

- **Soporte de arquitectura limitado** - TREAD solo está implementado para modelos Flux y Wan
- **Mejor a altas resoluciones** - Mayores aceleraciones a 1024x1024+ debido a la complejidad O(n²) de la atención
- **Compatible con pérdida enmascarada** - Las regiones enmascaradas se conservan automáticamente (pero esto reduce la aceleración)
- **Funciona con cuantización** - Puede combinarse con entrenamiento int8/int4/NF4
- **Espera un pico de pérdida inicial** - Al iniciar entrenamiento LoRA/LoKr, la pérdida será más alta al principio pero se corrige rápido

#### Consejos de ajuste

- **Conservador (enfocado en calidad)**: Usa `selection_ratio` de 0.3-0.5
- **Agresivo (enfocado en velocidad)**: Usa `selection_ratio` de 0.6-0.8
- **Evita capas tempranas/tardías**: No enrutes en las capas 0-1 ni en la capa final
- **Para entrenamiento LoRA**: Podrías ver ligeras desaceleraciones - experimenta con distintas configs
- **Mayor resolución = mayor aceleración**: Más beneficioso en 1024px y arriba

#### Comportamiento conocido

- Cuantos más tokens se eliminen (mayor `selection_ratio`), más rápido el entrenamiento pero mayor pérdida inicial
- El entrenamiento LoRA/LoKr muestra un pico de pérdida inicial que se corrige rápidamente a medida que la red se adapta
- Algunas configuraciones LoRA pueden entrenar un poco más lento - las configs óptimas aún se exploran
- La implementación de RoPE (rotary position embedding) es funcional pero puede no ser 100% correcta

Para opciones de configuración detalladas y solución de problemas, consulta la [documentación completa de TREAD](../TREAD.md).

### Classifier-free guidance

#### Problema

El modelo Dev llega destilado con guidance desde el inicio, lo que significa que realiza una trayectoria muy directa hacia las salidas del modelo maestro. Esto se hace mediante un vector de guidance que se alimenta al modelo en entrenamiento e inferencia: el valor de este vector afecta mucho qué tipo de LoRA resultante terminas obteniendo:

#### Solución

- Un valor de 1.0 (**el predeterminado**) preservará la destilación inicial hecha al modelo Dev
  - Este es el modo más compatible
  - La inferencia es igual de rápida que el modelo original
  - La destilación por flow-matching reduce la creatividad y la variabilidad de salida del modelo, como con el Flux Dev original (todo mantiene la misma composición/estética)
- Un valor más alto (probado alrededor de 3.5-4.5) reintroducirá el objetivo CFG en el modelo
  - Esto requiere que el pipeline de inferencia tenga soporte para CFG
  - La inferencia es 50% más lenta y 0% aumento de VRAM **o** alrededor de 20% más lenta y 20% más VRAM debido a la inferencia CFG en batch
  - Sin embargo, este estilo de entrenamiento mejora la creatividad y la variabilidad de salida del modelo, lo cual puede ser necesario para ciertas tareas de entrenamiento

Podemos reintroducir parcialmente la destilación en un modelo des-destilado continuando el ajuste con un valor de vector de 1.0. Nunca se recuperará por completo, pero al menos será más utilizable.

#### Advertencias

- Esto tiene el impacto final de **o bien**:
  - Aumentar la latencia de inferencia 2x cuando calculamos la salida incondicional de forma secuencial, p. ej. con dos forward pass separados
  - Aumentar el consumo de VRAM equivalente a usar `num_images_per_prompt=2` y recibir dos imágenes en inferencia, acompañado por el mismo porcentaje de ralentización.
    - Esto suele ser una ralentización menos extrema que el cálculo secuencial, pero el uso de VRAM podría ser demasiado para la mayoría del hardware de entrenamiento de consumo.
    - Este método no está _actualmente_ integrado en SimpleTuner, pero el trabajo continúa.
- Los flujos de inferencia para ComfyUI u otras aplicaciones (p. ej., AUTOMATIC1111) tendrán que modificarse para habilitar también CFG "real", lo cual puede no ser posible actualmente de fábrica.

### Cuantización

- Se requiere una cuantización mínima de 8 bits para que una tarjeta de 16G entrene este modelo
  - En bfloat16/float16, un LoRA de rango 1 se sitúa en poco más de 30GB de uso de memoria
- Cuantizar el modelo a 8 bits no perjudica el entrenamiento
  - Permite aumentar tamaños de lote y posiblemente obtener un mejor resultado
  - Se comporta igual que el entrenamiento de precisión completa - fp32 no hará tu modelo mejor que bf16+int8.
- **int8** tiene aceleración por hardware y soporte de `torch.compile()` en hardware NVIDIA más nuevo (3090 o mejor)
- **nf4-bnb** baja los requisitos de VRAM a 9GB, encajando en una tarjeta de 10G (con soporte bfloat16)
- Al cargar el LoRA en ComfyUI más tarde, **debes** usar la misma precisión del modelo base con la que entrenaste tu LoRA.
- **int4** depende de kernels bf16 personalizados, y no funcionará si tu tarjeta no soporta bfloat16

### Bloqueos

- Si obtienes SIGKILL después de que se descarguen los codificadores de texto, esto significa que no tienes suficiente memoria del sistema para cuantizar Flux.
  - Prueba cargar con `--base_model_precision=bf16` pero si eso no funciona, quizá necesites más memoria.
  - Prueba `--quantize_via=accelerator` para usar la GPU en su lugar

### Schnell

- Si entrenas un LyCORIS LoKr en Dev, **generalmente** funciona muy bien en Schnell con solo 4 pasos después.
  - El entrenamiento directo en Schnell realmente necesita un poco más de tiempo en el horno; actualmente, los resultados no se ven bien

> ℹ️ Al fusionar Schnell con Dev de cualquier manera, la licencia de Dev toma el control y pasa a ser no comercial. Esto no debería importar para la mayoría de usuarios, pero vale la pena señalarlo.

### Tasas de aprendizaje

#### LoRA (--lora_type=standard)

- LoRA tiene un rendimiento global peor que LoKr para datasets más grandes
- Se ha reportado que Flux LoRA entrena de forma similar a LoRAs de SD 1.5
- Sin embargo, un modelo tan grande como 12B ha rendido empíricamente mejor con **tasas de aprendizaje más bajas.**
  - LoRA a 1e-3 podría quemarlo. LoRA a 1e-5 casi no hace nada.
- Rangos tan grandes como 64 a 128 pueden ser indeseables en un modelo 12B debido a dificultades generales que escalan con el tamaño del modelo base.
  - Prueba una red más pequeña primero (rango 1, rango 4) y ve subiendo: entrenan más rápido y podrían hacer todo lo que necesitas.
  - Si te resulta excesivamente difícil entrenar tu concepto en el modelo, podrías necesitar un rango más alto y más datos de regularización.
- Otros modelos diffusion transformer como PixArt y SD3 se benefician mucho de `--max_grad_norm` y SimpleTuner mantiene un valor bastante alto por defecto en Flux.
  - Un valor más bajo evitaría que el modelo se desmorone demasiado pronto, pero también puede hacer muy difícil aprender conceptos nuevos que se alejan mucho de la distribución de datos del modelo base. El modelo podría quedarse atascado y nunca mejorar.

#### LoKr (--lora_type=lycoris)

- Tasas de aprendizaje más altas son mejores para LoKr (`1e-3` con AdamW, `2e-4` con Lion)
- Otros algoritmos necesitan más exploración.
- Configurar `is_regularisation_data` en esos datasets puede ayudar a preservar / evitar bleed y mejorar la calidad del modelo final.
  - Esto se comporta distinto de la "preservación de pérdida previa", que es conocida por duplicar los tamaños de lote y no mejorar mucho el resultado
  - La implementación de datos de regularización de SimpleTuner ofrece una manera eficiente de preservar el modelo base

### Artefactos de imagen

Flux absorberá inmediatamente los artefactos de imagen malos. Así son las cosas: una ejecución final de entrenamiento solo con datos de alta calidad puede ser necesaria para corregirlo al final.

Cuando haces estas cosas (entre otras), pueden comenzar a aparecer artefactos de cuadrícula **en** las muestras:

- Sobreentrenar con datos de baja calidad
- Usar una tasa de aprendizaje demasiado alta
- Sobreentrenamiento (en general) de una red de baja capacidad con demasiadas imágenes
- Subentrenamiento (también) de una red de alta capacidad con muy pocas imágenes
- Usar relaciones de aspecto extrañas o tamaños de datos de entrenamiento

### Aspect bucketing

- Entrenar demasiado tiempo en recortes cuadrados probablemente no dañará demasiado este modelo. Dale con todo, es excelente y confiable.
- Por otro lado, usar los buckets de aspecto natural de tu dataset podría sesgar demasiado esas formas en inferencia.
  - Esto podría ser una cualidad deseable, ya que evita que estilos dependientes de aspecto, como lo cinematográfico, se filtren a otras resoluciones demasiado.
  - Sin embargo, si buscas mejorar resultados por igual en muchos buckets de aspecto, quizá tengas que experimentar con `crop_aspect=random`, lo cual tiene sus propias desventajas.
- Mezclar configuraciones de dataset definiendo tu dataset de imágenes múltiples veces ha dado resultados muy buenos y un modelo bien generalizado.

### Entrenar modelos Flux ajustados personalizados

Algunos modelos Flux ajustados en Hugging Face Hub (como Dev2Pro) carecen de la estructura completa de directorios, lo que requiere configurar estas opciones específicas.

Asegúrate de configurar estas opciones `flux_guidance_value`, `validation_guidance_real` y `flux_attention_masked_training` según lo hizo el creador, si esa información está disponible.

<details>
<summary>Ver ejemplo de config</summary>

```json
{
    "model_family": "flux",
    "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_model_name_or_path": "ashen0209/Flux-Dev2Pro",
    "pretrained_vae_model_name_or_path": "black-forest-labs/FLUX.1-dev",
    "pretrained_transformer_subfolder": "none",
}
```
</details>
