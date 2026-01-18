# SimpleTuner 💹

> ℹ️ No se envían datos a terceros salvo a través de los flags opt-in `report_to`, `push_to_hub` o webhooks que deben configurarse manualmente.

**SimpleTuner** está orientado a la simplicidad, con un enfoque en que el código sea fácil de entender. Esta base de código sirve como un ejercicio académico compartido y se agradecen las contribuciones.

Si quieres unirte a nuestra comunidad, nos puedes encontrar [en Discord](https://discord.gg/JGkSwEbjRb) a través de Terminus Research Group.
Si tienes alguna pregunta, no dudes en contactarnos allí.

<img width="1944" height="1657" alt="image" src="https://github.com/user-attachments/assets/af3a24ec-7347-4ddf-8edf-99818a246de1" />


## Tabla de contenidos

- [Filosofía de diseño](#filosofía-de-diseño)
- [Tutorial](#tutorial)
- [Características](#características)
  - [Características principales de entrenamiento](#características-principales-de-entrenamiento)
  - [Compatibilidad de arquitectura de modelos](#compatibilidad-de-arquitectura-de-modelos)
  - [Técnicas avanzadas de entrenamiento](#técnicas-avanzadas-de-entrenamiento)
  - [Características específicas del modelo](#características-específicas-del-modelo)
  - [Guías de inicio rápido](#guías-de-inicio-rápido)
- [Requisitos de hardware](#requisitos-de-hardware)
- [Kit de herramientas](#kit-de-herramientas)
- [Instalación](#instalación)
- [Solución de problemas](#solución-de-problemas)

## Filosofía de diseño

- **Simplicidad**: Apuntar a buenos valores predeterminados para la mayoría de los casos de uso, de modo que se requiera menos ajuste.
- **Versatilidad**: Diseñado para manejar un amplio rango de cantidades de imágenes, desde pequeños datasets hasta colecciones extensas.
- **Funciones de vanguardia**: Solo incorpora funciones con eficacia comprobada, evitando agregar opciones sin probar.

## Tutorial

Por favor, explora completamente este README antes de iniciar el [nuevo tutorial de Web UI](/documentation/webui/TUTORIAL.md) o el [tutorial clásico de línea de comandos](/documentation/TUTORIAL.md), ya que este documento contiene información vital que podrías necesitar primero.

Para un inicio rápido configurado manualmente sin leer toda la documentación ni usar interfaces web, puedes usar la guía de [Quick Start](/documentation/QUICKSTART.md).

Para sistemas con memoria limitada, consulta el [documento de DeepSpeed](/documentation/DEEPSPEED.md), que explica cómo usar 🤗Accelerate para configurar DeepSpeed de Microsoft para el offload del estado del optimizador. Para sharding basado en DTensor y paralelismo de contexto, lee la [guía de FSDP2](/documentation/FSDP2.md), que cubre el nuevo flujo FullyShardedDataParallel v2 dentro de SimpleTuner.

Para entrenamiento distribuido multi-nodo, [esta guía](/documentation/DISTRIBUTED.md) ayudará a ajustar las configuraciones de las guías INSTALL y Quickstart para que sean adecuadas para entrenamiento multi-nodo y optimizadas para datasets de imágenes con miles de millones de muestras.

---

## Características

SimpleTuner ofrece soporte de entrenamiento integral en múltiples arquitecturas de modelos de difusión con disponibilidad consistente de funciones:

### Características principales de entrenamiento

- **Web UI fácil de usar** - Gestiona todo el ciclo de vida de entrenamiento desde un panel moderno
- **Entrenamiento multimodal** - Pipeline unificado para modelos generativos de **Imagen, Video y Audio**
- **Entrenamiento multi-GPU** - Entrenamiento distribuido en múltiples GPUs con optimización automática
- **Caché avanzada** - Embeddings de imagen, video, audio y captions en caché en disco para entrenamientos más rápidos
- **Aspect bucketing** - Soporte para tamaños y relaciones de aspecto variadas en imagen/video
- **Concept sliders** - Targeting compatible con sliders para LoRA/LyCORIS/full (vía LyCORIS `full`) con muestreo positivo/negativo/neutral y fuerza por prompt; ver la [guía de Slider LoRA](/documentation/SLIDER_LORA.md)
- **Optimización de memoria** - La mayoría de los modelos entrenables en GPU de 24G, muchos en 16G con optimizaciones
- **Integración con DeepSpeed y FSDP2** - Entrena modelos grandes en GPUs más pequeñas con sharding de optim/grad/param, atención paralela de contexto, gradient checkpointing y offload del estado del optimizador
- **Entrenamiento en S3** - Entrena directamente desde almacenamiento en la nube (Cloudflare R2, Wasabi S3)
- **Soporte EMA** - Pesos de media móvil exponencial para mayor estabilidad y calidad
- **Trackers de experimentos personalizados** - Coloca un `accelerate.GeneralTracker` en `simpletuner/custom-trackers` y usa `--report_to=custom-tracker --custom_tracker=<name>`

### Funciones multiusuario y empresariales

SimpleTuner incluye una plataforma completa de entrenamiento multiusuario con funciones de nivel empresarial - **gratuita y de código abierto, para siempre**.

- **Orquestación de workers** - Registra workers de GPU distribuidos que se conectan automáticamente a un panel central y reciben trabajos vía SSE; soporta workers efímeros (lanzados en la nube) y persistentes (siempre activos); ver la [Guía de orquestación de workers](/documentation/experimental/server/WORKERS.md)
- **Integración SSO** - Autenticación con LDAP/Active Directory u OIDC (Okta, Azure AD, Keycloak, Google); ver la [Guía de autenticación externa](/documentation/experimental/server/EXTERNAL_AUTH.md)
- **Control de acceso basado en roles** - Cuatro roles por defecto (Viewer, Researcher, Lead, Admin) con 17+ permisos granulares; define reglas de recursos con patrones glob para restringir configs, hardware o proveedores por equipo
- **Organizaciones y equipos** - Estructura multi-tenant jerárquica con cuotas basadas en topes; los límites de org definen máximos absolutos, los de equipo operan dentro de los límites de la org
- **Cuotas y límites de gasto** - Aplica topes de costo (diario/mensual), límites de concurrencia de trabajos y límites de tasa de envío a nivel de org, equipo o usuario; acciones: bloquear, advertir o requerir aprobación
- **Cola de trabajos con prioridades** - Cinco niveles de prioridad (Low → Critical) con planificación de fair-share entre equipos, prevención de starvation para trabajos con esperas largas y overrides de prioridad de admin
- **Flujos de aprobación** - Reglas configurables disparan aprobaciones para trabajos que exceden umbrales de costo, usuarios nuevos o solicitudes de hardware específicas; aprueba vía UI, API o respuesta por email
- **Notificaciones por email** - Integración SMTP/IMAP para estado de trabajos, solicitudes de aprobación, alertas de cuota y notificaciones de finalización
- **API keys y permisos con alcance** - Genera API keys con expiración y alcance limitado para pipelines CI/CD
- **Audit logging** - Registra todas las acciones de usuarios con verificación de cadena para cumplimiento; ver la [Guía de auditoría](/documentation/experimental/server/AUDIT.md)

Para detalles de despliegue, consulta la [Guía Enterprise](/documentation/experimental/server/ENTERPRISE.md).

### Compatibilidad de arquitectura de modelos

| Modelo | Parámetros | PEFT LoRA | Lycoris | Full-Rank | ControlNet | Cuantización | Flow Matching | Codificadores de texto |
|--------|------------|-----------|---------|-----------|------------|--------------|---------------|------------------------|
| **Stable Diffusion XL** | 3.5B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L/G |
| **Stable Diffusion 3** | 2B-8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L/G + T5-XXL |
| **Flux.1** | 12B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL |
| **Flux.2** | 32B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | Mistral-3 Small |
| **ACE-Step** | 3.5B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HeartMuLa** | 3B | ✓ | ✓ | ✓* | ✗ | int8 | ✗ | Ninguno |
| **Chroma 1** | 8.9B | ✓ | ✓ | ✓* | ✗ | int8/fp8/nf4 | ✓ | T5-XXL |
| **Auraflow** | 6.8B | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | UMT5-XXL |
| **PixArt Sigma** | 0.6B-0.9B | ✗ | ✓ | ✓ | ✓ | int8 | ✗ | T5-XXL |
| **Sana** | 0.6B-4.8B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2-2B |
| **Lumina2** | 2B | ✓ | ✓ | ✓ | ✗ | int8 | ✓ | Gemma2 |
| **Kwai Kolors** | 5B | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ | ChatGLM-6B |
| **LTX Video** | 5B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **LTX Video 2** | 19B | ✓ | ✓ | ✓* | ✗ | int8/fp8 | ✓ | Gemma3 |
| **Wan Video** | 1.3B-14B | ✓ | ✓ | ✓* | ✗ | int8 | ✓ | UMT5 |
| **HiDream** | 17B (8.5B MoE) | ✓ | ✓ | ✓* | ✓ | int8/fp8/nf4 | ✓ | CLIP-L + T5-XXL + Llama |
| **Cosmos2** | 2B-14B | ✗ | ✓ | ✓ | ✗ | int8 | ✓ | T5-XXL |
| **OmniGen** | 3.8B | ✓ | ✓ | ✓ | ✗ | int8/fp8 | ✓ | T5-XXL |
| **Qwen Image** | 20B | ✓ | ✓ | ✓* | ✗ | int8/nf4 (req.) | ✓ | T5-XXL |
| **SD 1.x/2.x (Legacy)** | 0.9B | ✓ | ✓ | ✓ | ✓ | int8/nf4 | ✗ | CLIP-L |

*✓ = Compatible, ✗ = No compatible, * = Requiere DeepSpeed para entrenamiento full-rank*

### Técnicas avanzadas de entrenamiento

- **TREAD** - Dropout por token para modelos transformer, incluido el entrenamiento de Kontext
- **Entrenamiento con pérdida enmascarada** - Mejor convergencia con guía de segmentación/profundidad
- **Regularización de prior** - Mayor estabilidad de entrenamiento para consistencia de personajes
- **Gradient checkpointing** - Intervalos configurables para optimización de memoria/velocidad
- **Funciones de pérdida** - L2, Huber, Smooth L1 con soporte de scheduling
- **Ponderación SNR** - Ponderación Min-SNR gamma para mejorar la dinámica de entrenamiento
- **Group offloading** - Staging de módulos por grupo a CPU/disco en Diffusers v0.33+ con streams CUDA opcionales
- **Barridos de adaptadores de validación** - Adjunta temporalmente adaptadores LoRA (individuales o presets JSON) durante la validación para medir renders solo de adaptador o comparativos sin tocar el training loop
- **Hooks de validación externos** - Sustituye el pipeline de validación integrado o los pasos post-upload por tus scripts, para ejecutar checks en otra GPU o reenviar artefactos a cualquier proveedor cloud que elijas ([detalles](/documentation/OPTIONS.md#validation_method))
- **Regularización CREPA** - Alineación de representación entre frames para video DiTs ([guía](/documentation/experimental/VIDEO_CREPA.md))
- **Formatos de I/O de LoRA** - Carga/guarda LoRAs PEFT en layout estándar de Diffusers o claves estilo ComfyUI `diffusion_model.*` (Flux/Flux2/Lumina2/Z-Image detectan automáticamente entradas ComfyUI)

### Características específicas del modelo

- **Flux Kontext** - Conditioning de edición y entrenamiento image-to-image para modelos Flux
- **PixArt two-stage** - Soporte de pipeline eDiff para PixArt Sigma
- **Modelos de flow matching** - Scheduling avanzado con distribuciones beta/uniforme
- **HiDream MoE** - Aumento de pérdida de puerta Mixture of Experts
- **Entrenamiento con T5 enmascarado** - Mayor detalle fino para Flux y modelos compatibles
- **Fusión QKV** - Optimizaciones de memoria y velocidad (Flux, Lumina2)
- **Integración TREAD** - Enrutamiento selectivo de tokens para la mayoría de modelos
- **Wan 2.x I2V** - Presets de etapa alta/baja más un fallback de time-embedding 2.1 (ver quickstart de Wan)
- **Classifier-free guidance** - Reintroducción opcional de CFG para modelos destilados

### Guías de inicio rápido

Hay guías detalladas de inicio rápido disponibles para todos los modelos soportados:

- **[Guía de TwinFlow Few-Step (RCGM)](/documentation/distillation/TWINFLOW.md)** - Habilita pérdida auxiliar RCGM para generación de pocos pasos/un paso (modelos de flow o difusión vía diff2flow)
- **[Guía de Flux.1](/documentation/quickstart/FLUX.md)** - Incluye soporte de edición Kontext y fusión QKV
- **[Guía de Flux.2](/documentation/quickstart/FLUX2.md)** - **NUEVO**. Último y enorme modelo Flux con codificador de texto Mistral-3
- **[Guía de Z-Image](/documentation/quickstart/ZIMAGE.md)** - LoRA Base/Turbo con adaptador asistente + aceleración TREAD
- **[Guía de ACE-Step](/documentation/quickstart/ACE_STEP.md)** - **NUEVO**. Entrenamiento de modelo de generación de audio (text-to-music)
- **[Guía de HeartMuLa](/documentation/quickstart/HEARTMULA.md)** - **NUEVO**. Entrenamiento de modelo de audio autoregresivo (text-to-audio)
- **[Guía de Chroma](/documentation/quickstart/CHROMA.md)** - Transformer de flow-matching de Lodestone con schedules específicos de Chroma
- **[Guía de Stable Diffusion 3](/documentation/quickstart/SD3.md)** - Entrenamiento full y LoRA con ControlNet
- **[Guía de Stable Diffusion XL](/documentation/quickstart/SDXL.md)** - Pipeline completo de entrenamiento SDXL
- **[Guía de Auraflow](/documentation/quickstart/AURAFLOW.md)** - Entrenamiento de modelos de flow-matching
- **[Guía de PixArt Sigma](/documentation/quickstart/SIGMA.md)** - Modelo DiT con soporte de dos etapas
- **[Guía de Sana](/documentation/quickstart/SANA.md)** - Modelo ligero de flow-matching
- **[Guía de Lumina2](/documentation/quickstart/LUMINA2.md)** - Modelo de flow-matching de 2B parámetros
- **[Guía de Kwai Kolors](/documentation/quickstart/KOLORS.md)** - Basado en SDXL con encoder ChatGLM
- **[Guía de LongCat-Video](/documentation/quickstart/LONGCAT_VIDEO.md)** - Flow-matching text-to-video e image-to-video con Qwen-2.5-VL
- **[Guía de LongCat-Video Edit](/documentation/quickstart/LONGCAT_VIDEO_EDIT.md)** - Variante de conditioning-first (image-to-video)
- **[Guía de LongCat-Image](/documentation/quickstart/LONGCAT_IMAGE.md)** - Modelo bilingüe de flow-matching 6B con encoder Qwen-2.5-VL
- **[Guía de LongCat-Image Edit](/documentation/quickstart/LONGCAT_EDIT.md)** - Variante de edición de imagen que requiere latents de referencia
- **[Guía de LTX Video](/documentation/quickstart/LTXVIDEO.md)** - Entrenamiento de difusión de video
- **[Guía de Hunyuan Video 1.5](/documentation/quickstart/HUNYUANVIDEO.md)** - Flow-matching T2V/I2V de 8.3B con etapas de SR
- **[Guía de Wan Video](/documentation/quickstart/WAN.md)** - Flow-matching de video con soporte TREAD
- **[Guía de HiDream](/documentation/quickstart/HIDREAM.md)** - Modelo MoE con funciones avanzadas
- **[Guía de Cosmos2](/documentation/quickstart/COSMOS2IMAGE.md)** - Generación de imagen multimodal
- **[Guía de OmniGen](/documentation/quickstart/OMNIGEN.md)** - Modelo unificado de generación de imagen
- **[Guía de Qwen Image](/documentation/quickstart/QWEN_IMAGE.md)** - Entrenamiento a gran escala de 20B parámetros
- **[Guía de Stable Cascade Stage C](/quickstart/STABLE_CASCADE_C.md)** - LoRAs de prior con validación combinada prior+decoder
- **[Guía de Kandinsky 5.0 Image](/documentation/quickstart/KANDINSKY5_IMAGE.md)** - Generación de imagen con Qwen2.5-VL + Flux VAE
- **[Guía de Kandinsky 5.0 Video](/documentation/quickstart/KANDINSKY5_VIDEO.md)** - Generación de video con HunyuanVideo VAE

---

## Requisitos de hardware

### Requisitos generales

- **NVIDIA**: RTX 3080+ recomendado (probado hasta H200)
- **AMD**: 7900 XTX 24GB y MI300X verificados (uso de memoria más alto que NVIDIA)
- **Apple**: M3 Max+ con 24GB+ de memoria unificada para entrenamiento LoRA

### Guías de memoria por tamaño de modelo

- **Modelos grandes (12B+)**: A100-80G para full-rank, 24G+ para LoRA/Lycoris
- **Modelos medianos (2B-8B)**: 16G+ para LoRA, 40G+ para entrenamiento full-rank
- **Modelos pequeños (<2B)**: 12G+ suficiente para la mayoría de tipos de entrenamiento

**Nota**: La cuantización (int8/fp8/nf4) reduce significativamente los requisitos de memoria. Consulta las [guías de inicio rápido](#guías-de-inicio-rápido) para requisitos específicos por modelo.

## Instalación

SimpleTuner se puede instalar vía pip para la mayoría de los usuarios:

```bash
# Base installation (CPU-only PyTorch)
pip install simpletuner

# CUDA users (NVIDIA GPUs)
pip install 'simpletuner[cuda]'

# ROCm users (AMD GPUs)
pip install 'simpletuner[rocm]'

# Apple Silicon users (M1/M2/M3/M4 Macs)
pip install 'simpletuner[apple]'
```

Para instalación manual o setup de desarrollo, consulta la [documentación de instalación](/documentation/INSTALL.md).

## Solución de problemas

Habilita logs de depuración para más detalle agregando `export SIMPLETUNER_LOG_LEVEL=DEBUG` a tu archivo de entorno (`config/config.env`).

Para análisis de rendimiento del loop de entrenamiento, configurar `SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG` agrega marcas de tiempo que destacan problemas en tu configuración.

Para una lista completa de opciones disponibles, consulta [esta documentación](/documentation/OPTIONS.md).
