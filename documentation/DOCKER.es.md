# Docker para SimpleTuner

Esta configuración de Docker proporciona un entorno integral para ejecutar SimpleTuner en varias plataformas, incluyendo Runpod, Vast.ai y otros hosts compatibles con Docker. Está optimizada para facilidad de uso y robustez, integrando herramientas y librerías esenciales para proyectos de machine learning.

## Funcionalidades del contenedor

- **Imagen base con CUDA**: Construida a partir de `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` para soportar aplicaciones aceleradas por GPU.
- **Herramientas de desarrollo**: Incluye Git, SSH y varias utilidades como `tmux`, `vim`, `htop`.
- **Python y librerías**: Incluye Python 3.10 y SimpleTuner preinstalado vía pip.
- **Integración con Huggingface y WandB**: Preconfigurado para integración fluida con Huggingface Hub y WandB, facilitando el intercambio de modelos y el seguimiento de experimentos.

## Primeros pasos

### Soporte de Windows OS vía WSL (Experimental)

La siguiente guía se probó en una distro WSL2 con Dockerengine instalado.


### 1. Construir el contenedor

Clona el repositorio y navega al directorio que contiene el Dockerfile. Construye la imagen de Docker usando:

```bash
docker build -t simpletuner .
```

### 2. Ejecutar el contenedor

Para ejecutar el contenedor con soporte GPU, ejecuta:

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

Este comando configura el contenedor con acceso a GPU y mapea el puerto SSH para conectividad externa.

### 3. Variables de entorno

Para facilitar la integración con herramientas externas, el contenedor soporta variables de entorno para tokens de Huggingface y WandB. Pásalas en tiempo de ejecución de la siguiente manera:

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. Volúmenes de datos

Para almacenamiento persistente y compartir datos entre el host y el contenedor, monta un volumen de datos:

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. Acceso SSH

El acceso SSH al contenedor está configurado por defecto. Asegúrate de proporcionar tu clave pública SSH mediante la variable de entorno correspondiente (`SSH_PUBLIC_KEY` para Vast.ai o `PUBLIC_KEY` para Runpod).

### 6. Usar SimpleTuner

SimpleTuner está preinstalado y listo para usar. Puedes ejecutar comandos de entrenamiento directamente:

```bash
simpletuner configure
simpletuner train
```

Para configuración y setup, consulta la [documentación de instalación](INSTALL.md) y las [guías de quickstart](QUICKSTART.md).

## Configuración adicional

### Scripts y configuraciones personalizadas

Si deseas añadir scripts de inicio personalizados o modificar configuraciones, extiende el script de entrada (`docker-start.sh`) para ajustarlo a tus necesidades.

Si alguna capacidad no puede lograrse mediante esta configuración, por favor abre un issue.

### Docker Compose

Para usuarios que prefieren `docker-compose.yaml`, esta plantilla se proporciona para que la extiendas y la personalices según tus necesidades.

Una vez desplegado el stack, puedes conectarte al contenedor y empezar a operar como se menciona en los pasos anteriores.

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ ¡Por favor, ten cuidado al manejar tus tokens de WandB y Hugging Face! Se recomienda no commitarlos ni siquiera a un repositorio privado para evitar filtraciones. Para casos de uso en producción, se recomienda almacenamiento de gestión de claves, pero está fuera del alcance de esta guía.
---

## Solución de problemas

### Desajuste de versión de CUDA

**Síntoma**: La aplicación no utiliza la GPU, o aparecen errores relacionados con librerías CUDA al intentar ejecutar tareas aceleradas por GPU.

**Causa**: Este problema puede ocurrir si la versión de CUDA instalada dentro del contenedor Docker no coincide con la versión del driver CUDA disponible en la máquina host.

**Solución**:
1. **Verifica la versión del driver CUDA en el host**: Determina la versión del driver CUDA instalado en la máquina host ejecutando:
   ```bash
   nvidia-smi
   ```
   Este comando mostrará la versión de CUDA en la parte superior derecha de la salida.

2. **Ajusta la versión de CUDA del contenedor**: Asegúrate de que la versión del toolkit CUDA en tu imagen Docker sea compatible con el driver CUDA del host. NVIDIA suele permitir compatibilidad hacia adelante, pero consulta la matriz de compatibilidad específica en el sitio de NVIDIA.

3. **Reconstruye la imagen**: Si es necesario, modifica la imagen base en el Dockerfile para que coincida con el driver CUDA del host. Por ejemplo, si tu host corre CUDA 11.2 y el contenedor está configurado para CUDA 11.8, quizá necesites cambiar a una imagen base adecuada:
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   Tras modificar el Dockerfile, reconstruye la imagen Docker.

### Problemas de conexión SSH

**Síntoma**: No se puede conectar al contenedor vía SSH.

**Causa**: Configuración incorrecta de claves SSH o el servicio SSH no inicia correctamente.

**Solución**:
1. **Verifica la configuración SSH**: Asegúrate de que la clave pública SSH se haya agregado correctamente en `~/.ssh/authorized_keys` dentro del contenedor. También verifica que el servicio SSH esté en ejecución entrando al contenedor y ejecutando:
   ```bash
   service ssh status
   ```
2. **Puertos expuestos**: Confirma que el puerto SSH (22) esté expuesto y mapeado correctamente al iniciar el contenedor, como se muestra en las instrucciones de ejecución:
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### Consejos generales

- **Logs y salida**: Revisa los logs y la salida del contenedor para mensajes de error o advertencia que puedan aportar más contexto.
- **Documentación y foros**: Consulta la documentación de Docker y NVIDIA CUDA para consejos de troubleshooting más detallados. Foros de la comunidad y issue trackers relacionados con el software o dependencias específicas también pueden ser recursos valiosos.
