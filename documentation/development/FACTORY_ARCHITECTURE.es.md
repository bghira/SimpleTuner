# Arquitectura de la fábrica de backends de datos

## Visión general

La Fábrica de Backends de Datos de SimpleTuner implementa una arquitectura modular que sigue principios SOLID para mantenibilidad, testabilidad y rendimiento.

## Diagrama de arquitectura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Factory Registry                                     │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │   Performance       │  │     Backend         │  │    Configuration    │  │
│  │    Tracking         │  │    Management       │  │     Loading         │  │
│  │                     │  │                     │  │                     │  │
│  │ • Memory usage      │  │ • text_embed_backends│ │ • JSON parsing      │  │
│  │ • Initialization    │  │ • image_embed_backends│ │ • Variable filling  │  │
│  │ • Build times       │  │ • data_backends     │  │ • Dependency sort   │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Configuration Classes                                  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendConfig  │  │  ImageBackendConfig │  │ TextEmbedBackendConfig│ │
│  │                     │  │                     │  │                     │  │
│  │ • Common fields     │  │ • Image datasets    │  │ • Text embedding    │  │
│  │ • Shared validation │  │ • Video datasets    │  │   datasets          │  │
│  │ • Base defaults     │  │ • Conditioning      │  │ • Simple validation │  │
│  │                     │  │ • Extensive config  │  │                     │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                       │                                      │
│                        ┌─────────────────────┐                             │
│                        │ImageEmbedBackendConfig│                             │
│                        │                     │                             │
│                        │ • Image embedding   │                             │
│                        │   datasets          │                             │
│                        │ • VAE cache config  │                             │
│                        └─────────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Builder Classes                                     │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │  BaseBackendBuilder │  │  LocalBackendBuilder│  │  AwsBackendBuilder  │  │
│  │                     │  │                     │  │                     │  │
│  │ • Metadata creation │  │ • LocalDataBackend  │  │ • S3DataBackend     │  │
│  │ • Cache management  │  │ • File system       │  │ • AWS S3 storage    │  │
│  │ • Common build flow │  │   operations        │  │ • Connection pools  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │  CsvBackendBuilder  │  │HuggingfaceBackendBuilder│                        │
│  │                     │  │                     │                          │
│  │ • CSVDataBackend    │  │ • HuggingfaceDatasetsBackend│                    │
│  │ • CSV file handling │  │ • Streaming datasets│                          │
│  │ • URL downloads     │  │ • Filter functions  │                          │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Runtime Components                                   │
│                                                                              │
│  ┌─────────────────────┐  ┌─────────────────────┐                          │
│  │   BatchFetcher      │  │ DataLoader Iterator │                          │
│  │                     │  │                     │                          │
│  │ • Background        │  │ • Multi-backend     │                          │
│  │   prefetching       │  │   sampling          │                          │
│  │ • Queue management  │  │ • Weighted selection│                          │
│  │ • Thread safety     │  │ • Exhaustion handling│                         │
│  └─────────────────────┘  └─────────────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Descripción de componentes

### 1. Registro de fábrica (`factory.py`)

El orquestador central que coordina toda la creación y gestión de backends.

**Responsabilidades clave:**
- Seguimiento de rendimiento con métricas de memoria y tiempo
- Gestión del ciclo de vida de backends
- Procesamiento y validación de configuración
- Coordinación de backends de embeddings de texto/imagen

**Funciones de rendimiento:**
- Seguimiento de uso de memoria en tiempo real
- Métricas de tiempo por etapa
- Monitoreo de memoria pico
- Logging para análisis

### 2. Clases de configuración (`config/`)

Gestión de configuración con tipos seguros usando dataclasses con herencia.

#### BaseBackendConfig (`config/base.py`)
- Base abstracta con campos comunes y validación
- Resolución compartida, estrategia de caption y validación de tamaño de imagen
- Proporciona plantilla para configuraciones especializadas

#### ImageBackendConfig (`config/image.py`)
- Configuración para datasets de imagen, video, conditioning y eval
- Maneja recorte complejo, relación de aspecto y ajustes específicos del backend
- Validación extensa para todos los casos de uso

#### TextEmbedBackendConfig (`config/text_embed.py`)
- Configuración simplificada para datasets de embeddings de texto
- Enfoque en gestión de caché y conectividad del backend

#### ImageEmbedBackendConfig (`config/image_embed.py`)
- Configuración para datasets de embeddings de imagen VAE
- Maneja directorios de caché y parámetros de procesamiento

### 3. Clases constructoras (`builders/`)

Implementación del patrón factory para crear instancias de backend.

#### BaseBackendBuilder (`builders/base.py`)
- Builder abstracto con lógica compartida de creación de backend de metadatos
- Gestión de caché y manejo de eliminación
- Patrón de método plantilla para flujo de build consistente

#### Builders especializados
- **LocalBackendBuilder**: Crea instancias `LocalDataBackend` para acceso al sistema de archivos
- **AwsBackendBuilder**: Crea instancias `S3DataBackend` con pooling de conexiones
- **CsvBackendBuilder**: Crea instancias `CSVDataBackend` para datasets basados en CSV
- **HuggingfaceBackendBuilder**: Crea instancias `HuggingfaceDatasetsBackend` con soporte de streaming

### 4. Componentes de ejecución (`runtime/`)

Componentes optimizados para operaciones en tiempo de entrenamiento.

#### BatchFetcher (`runtime/batch_fetcher.py`)
- Prefetch en segundo plano de batches de entrenamiento
- Gestión de cola segura para hilos
- Tamaño de cola configurable y monitoreo de rendimiento

#### DataLoader Iterator (`runtime/dataloader_iterator.py`)
- Muestreo ponderado a través de múltiples backends
- Manejo automático de agotamiento y repeticiones de dataset
- Soporte para estrategias de muestreo uniform y auto-weighting

## Flujo de datos

### Flujo de inicialización
```
1. Factory Registry Creation
   ├── Performance tracking initialization
   ├── Component registry setup
   └── Memory tracking start

2. Configuration Loading
   ├── JSON parsing and validation
   ├── Variable substitution
   └── Ordenamiento de dependencias

3. Configuración de backends
   ├── Backends de embeddings de texto
   ├── Backends de embeddings de imagen
   └── Backends de datos principales

4. Preparación de runtime
   ├── Creación del backend de metadatos
   ├── Configuración de dataset y sampler
   └── Inicialización de caché
```

### Flujo de entrenamiento
```
1. Solicitud de batch
   ├── Prefetching de BatchFetcher
   ├── Selección ponderada de backends
   └── Gestión de cola

2. Carga de datos
   ├── Acceso a datos específico del backend
   ├── Búsqueda de metadatos
   └── Composición del batch

3. Procesamiento
   ├── Búsqueda de embeddings de texto
   ├── Acceso a caché VAE
   └── Preparación final del batch
```

## Decisiones de diseño y justificación

### 1. Arquitectura modular
**Decisión**: Dividir la fábrica monolítica en componentes especializados
**Justificación**:
- Mejora la testabilidad al aislar responsabilidades
- Permite el desarrollo independiente de funciones
- Reduce la carga cognitiva al trabajar en funcionalidad específica
- Sigue el principio de responsabilidad única

### 2. Clases de configuración con herencia
**Decisión**: Usar dataclasses con jerarquía de herencia
**Justificación**:
- Seguridad de tipos y soporte del IDE
- Lógica de validación compartida sin duplicación
- Separación clara entre tipos de backend
- Detección de errores en tiempo de compilación

### 3. Patrón factory para builders
**Decisión**: Usar patrón factory para la creación de backends
**Justificación**:
- Lógica de creación centralizada
- Extensión fácil para nuevos tipos de backend
- Flujo de creación consistente en todos los backends
- Separación de configuración e instanciación

### 4. Diseño con rendimiento primero
**Decisión**: Seguimiento y monitoreo de rendimiento integrados
**Justificación**:
- Crítico para cargas de entrenamiento a gran escala
- Permite identificar optimizaciones
- Proporciona datos de análisis de rendimiento
- Capacidades de monitoreo en tiempo real

## Mejoras de rendimiento logradas

### Optimización de memoria
- Uso mínimo de memoria pico
- Carga perezosa de componentes
- Gestión eficiente de caché
- Patrones optimizados de recolección de basura

### Velocidad de inicialización
- Inicialización rápida de backends
- Procesamiento de configuración en paralelo
- Resolución de dependencias optimizada
- Operaciones redundantes mínimas

### Mantenibilidad del código
- Baja complejidad de funciones
- Tamaño máximo de función: 50 líneas
- Separación clara de responsabilidades
- Cobertura de tipos

## Enfoque de desarrollo

### Fases de implementación
- **Fase 1**: Diseño de arquitectura central e implementación de componentes
- **Fase 2**: Pruebas y validación
- **Fase 3**: Optimización y monitoreo de rendimiento
- **Fase 4**: Despliegue en producción y mantenimiento

### Estrategia de pruebas
- Pruebas end-to-end con cargas reales
- Benchmarking y monitoreo de rendimiento
- Validación de casos límite
- Pruebas de integración continua

## Métricas de calidad de código

### Organización de archivos
- Tamaño máximo de archivo: 500 líneas
- Tamaño promedio de función: 25 líneas
- Límites claros de módulos
- Convenciones de nombres consistentes

### Cobertura de tipos
- 100% de type hints en APIs públicas
- Cobertura de docstrings
- Cumplimiento de análisis estático
- Diseño amigable con IDE

### Estrategia de pruebas
- Pruebas unitarias para cada componente
- Pruebas de integración para flujos de trabajo
- Pruebas de regresión de rendimiento
- Validación de casos límite

## Mejoras futuras

### Funcionalidades planificadas
1. **Soporte async**: Operaciones de backend no bloqueantes
2. **Arquitectura de plugins**: Extensiones de backend de terceros
3. **Caché avanzada**: Jerarquía de caché multinivel
4. **Dashboard de monitoreo**: Visualización de rendimiento en tiempo real

### Puntos de extensión
- Nuevos tipos de backend vía registro de builders
- Clases de configuración personalizadas
- Estrategias de muestreo conectables
- Métricas de rendimiento personalizadas

## Conclusión

La Data Backend Factory proporciona una base para una gestión escalable y mantenible de backends de datos en SimpleTuner. La arquitectura modular habilita mejoras futuras mientras ofrece buen rendimiento, calidad de código y experiencia de desarrollo.

La arquitectura aborda la complejidad de gestionar múltiples fuentes de datos mientras proporciona funcionalidad robusta y patrones claros para extensión y personalización.
