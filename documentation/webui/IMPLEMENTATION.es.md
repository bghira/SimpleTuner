# Detalles de implementación del WebUI de SimpleTuner

## Resumen de diseño

La API de SimpleTuner está construida con flexibilidad en mente.

En modo `trainer`, se abre un único puerto para integración de FastAPI con otros servicios.
Con el modo `unified`, se abre un puerto adicional para que el WebUI reciba eventos de callback desde un proceso `trainer` remoto.

## Framework web

El WebUI se construye y se sirve usando FastAPI;

- Alpine.js se usa para componentes reactivos
- HTMX se usa para carga dinámica de contenido e interactividad
- FastAPI con Starlette y SSE-Starlette se usan para servir una API centrada en datos con eventos enviados por el servidor (SSE) para actualizaciones en tiempo real
- Jinja2 se usa para plantillas HTML

Alpine fue elegido por su simplicidad y facilidad de integración: mantiene NodeJS fuera del stack, lo que facilita desplegar y mantener.

HTMX tiene una sintaxis simple y ligera que combina bien con Alpine. Ofrece capacidades extensas para carga dinámica de contenido, manejo de formularios e interactividad sin necesidad de un framework frontend completo.

He elegido Starlette y SSE-Starlette porque quería mantener al mínimo la duplicación de código; se requirió bastante refactorización para pasar de una dispersión ad-hoc de código procedural a un enfoque más declarativo.

### Flujo de datos

Históricamente, la app FastAPI también actuaba como un “service worker” dentro de clústeres de trabajos: el trainer arrancaba, exponía una superficie de callback estrecha y los orquestadores remotos transmitían actualizaciones de estado por HTTP. El WebUI simplemente reutiliza ese mismo bus de callbacks. En modo unified ejecutamos tanto el trainer como la interfaz en el mismo proceso, mientras que los despliegues solo con trainer pueden seguir enviando eventos a `/callbacks` y dejar que una instancia separada de WebUI los consuma por SSE. No se necesita una cola ni un broker nuevo: aprovechamos la infraestructura que ya se envía con los despliegues headless.

## Arquitectura de backend

La UI del trainer se apoya en el SDK principal que ahora expone servicios bien definidos en lugar de helpers procedurales sueltos. FastAPI sigue terminando cada solicitud, pero la mayoría de las rutas son delegadores delgados hacia objetos de servicio. Esto mantiene la capa HTTP simple y maximiza la reutilización para el CLI, el asistente de configuración y futuras APIs.

### Manejadores de rutas

`simpletuner/simpletuner_sdk/server/routes/web.py` conecta la superficie `/web/trainer`. Solo hay dos endpoints interesantes:

- `trainer_page` – renderiza el chrome externo (navegación, selector de configuración, lista de pestañas). Pide metadatos a `TabService` y pasa todo al template `trainer_htmx.html`.
- `render_tab` – un objetivo genérico de HTMX. Cada botón de pestaña llama a este endpoint con el nombre de la pestaña; la ruta resuelve el layout correspondiente mediante `TabService.render_tab` y devuelve el fragmento HTML.

El resto del router HTTP vive bajo `simpletuner/simpletuner_sdk/server/routes/` y sigue el mismo patrón: la lógica de negocio está en un módulo de servicio, la ruta extrae parámetros, llama al servicio y convierte el resultado en JSON o HTML.

### TabService

`TabService` es el orquestador central del formulario de entrenamiento. Define:

- metadatos estáticos para cada pestaña (título, icono, template, hook de contexto opcional)
- `render_tab()` que
  1. obtiene la configuración de la pestaña (template, descripción)
  2. pide a `FieldService` el paquete de campos de la pestaña/sección
  3. inyecta cualquier contexto específico de la pestaña (lista de datasets, inventario de GPU, estado de onboarding)
  4. devuelve un render Jinja de `form_tab.html`, `datasets_tab.html`, etc.

Al llevar esta lógica a una clase podemos reutilizar exactamente el mismo render para HTMX, el wizard de CLI y tests. Nada en el template accede a estado global: todo se suministra explícitamente por contexto.

### FieldService y FieldRegistry

`FieldService` convierte entradas del registry en diccionarios listos para templates. Responsabilidades:

- filtrar campos por contexto de plataforma/modelo (p. ej., ocultar controles solo CUDA en máquinas MPS)
- evaluar reglas de dependencia (`FieldDependency`) para que la UI pueda deshabilitar u ocultar controles (por ejemplo, extras de Dynamo quedan en gris hasta seleccionar un backend)
- enriquecer campos con hints, opciones dinámicas, formato de visualización y clases de columna

Delega el catálogo bruto de campos a `FieldRegistry`, que es una lista declarativa bajo `simpletuner/simpletuner_sdk/server/services/field_registry`. Cada `ConfigField` describe nombres de flags de CLI, reglas de validación, orden de importancia, metadatos de dependencia y copia UI por defecto. Este arreglo permite que otras capas (parser de CLI, API, generador de documentación) compartan la misma fuente de verdad mientras la presentan en su propio formato.

### Persistencia de estado y onboarding

El WebUI almacena preferencias ligeras vía `WebUIStateStore`. Lee valores por defecto desde `$SIMPLETUNER_WEB_UI_CONFIG` (o una ruta XDG) y expone:

- tema, raíz de datasets, valores por defecto de output dir
- estado del checklist de onboarding por función
- overrides de Accelerate en caché (solo claves en whitelist como `--num_processes`, `--dynamo_backend`)

Esos valores se inyectan en la página durante el render inicial de `/web/trainer` para que los stores de Alpine se inicialicen sin viajes extra.

### Interacción HTMX + Alpine

Cada panel de ajustes es solo un fragmento de HTML con `x-data` para el comportamiento de Alpine. Los botones de pestaña disparan GETs de HTMX contra `/web/trainer/tabs/{tab}`; el servidor responde con el formulario renderizado y Alpine mantiene el estado del componente existente. Un pequeño helper (`trainer-form.js`) re-reproduce cambios de valores guardados para que el usuario no pierda ediciones en curso al cambiar de pestaña.

Las actualizaciones del servidor (estado de entrenamiento, telemetría de GPU) fluyen por endpoints SSE (`sse_manager.js`) y depositan datos en stores de Alpine que alimentan toasts, barras de progreso y banners del sistema.

### Hoja de referencia de layout de archivos

- `templates/` – plantillas Jinja; `partials/form_field.html` renderiza controles individuales. `partials/form_field_htmx.html` es la variante compatible con HTMX usada cuando un wizard necesita binding bidireccional.
- `static/js/modules/` – scripts de componentes Alpine (formulario del trainer, inventario de hardware, navegador de datasets).
- `static/js/services/` – helpers compartidos (evaluación de dependencias, SSE manager, event bus).
- `simpletuner/simpletuner_sdk/server/services/` – capa de servicios backend (fields, tabs, configs, datasets, maintenance, events).

En conjunto esto mantiene el WebUI sin estado en el servidor, con partes con estado (datos de formulario, toasts) viviendo en el navegador. El backend se mantiene en transformaciones de datos puras, lo que facilita las pruebas y evita problemas de hilos cuando el trainer y el servidor web corren en el mismo proceso.
