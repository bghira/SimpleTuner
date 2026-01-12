# Documentación de entrenamiento TREAD

> ⚠️ **Función experimental**: El soporte TREAD en SimpleTuner es una implementación reciente. Aunque es funcional, aún se exploran configuraciones óptimas y algunos comportamientos pueden cambiar en futuras versiones.

## Resumen

TREAD (Token Routing for Efficient Architecture-agnostic Diffusion Training) es un método de aceleración de entrenamiento que acelera el entrenamiento de modelos de difusión al enrutar tokens de forma inteligente a través de capas transformer. Al procesar selectivamente solo los tokens más importantes en ciertas capas, TREAD puede reducir significativamente los costos computacionales manteniendo la calidad del modelo.

Basado en la investigación de [Krause et al. (2025)](https://arxiv.org/abs/2501.04765), TREAD logra aceleraciones mediante:
- Seleccionar dinámicamente qué tokens se procesan en cada capa transformer
- Mantener el flujo de gradientes en todos los tokens vía skip connections
- Usar decisiones de enrutamiento basadas en importancia

La aceleración es directamente proporcional al `selection_ratio`: cuanto más cercano a 1.0, más tokens se descartan y más rápido es el entrenamiento.

## Cómo funciona TREAD

### Concepto principal

Durante el entrenamiento, TREAD:
1. **Enruta tokens** - Para capas transformer específicas, selecciona un subconjunto de tokens para procesar según su importancia
2. **Procesa el subconjunto** - Solo los tokens seleccionados pasan por las operaciones costosas de atención y MLP
3. **Restaura la secuencia completa** - Tras procesar, la secuencia completa se restaura con gradientes fluyendo a todos los tokens

### Selección de tokens

Los tokens se seleccionan según su norma L1 (score de importancia), con aleatorización opcional para exploración:
- Tokens de mayor importancia tienen más probabilidad de mantenerse
- Una mezcla de selección por importancia y aleatoria evita overfitting a patrones específicos
- Las máscaras de force-keep pueden asegurar que ciertos tokens (como regiones enmascaradas) nunca se descarten

## Configuración

### Configuración básica

Para habilitar entrenamiento TREAD en SimpleTuner, agrega lo siguiente a tu configuración:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### Configuración de rutas

Cada ruta define una ventana donde el enrutamiento de tokens está activo:
- `selection_ratio`: Fracción de tokens a descartar (0.5 = mantener 50% de tokens)
- `start_layer_idx`: Primera capa donde inicia el enrutamiento (0-indexed)
- `end_layer_idx`: Última capa donde el enrutamiento está activo

Se soportan índices negativos: `-1` se refiere a la última capa.

### Ejemplo avanzado

Múltiples ventanas de enrutamiento con distintos ratios de selección:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## Compatibilidad

### Modelos soportados
- **FLUX Dev/Kontext, Wan, AuraFlow, PixArt y SD3** - Actualmente las únicas familias soportadas
- Soporte futuro planificado para otros transformers de difusión

### Funciona bien con
- **Entrenamiento con pérdida enmascarada** - TREAD preserva automáticamente regiones enmascaradas cuando se combina con condicionamiento mask/segmentation
- **Entrenamiento multi-GPU** - Compatible con setups de entrenamiento distribuido
- **Entrenamiento cuantizado** - Puede usarse con cuantización int8/int4/NF4

### Limitaciones
- Solo activo durante entrenamiento (no en inferencia)
- Requiere cálculo de gradientes (no funciona en modo eval)
- Implementación actualmente específica de FLUX y Wan, no disponible para Lumina2 u otras arquitecturas aún

## Consideraciones de rendimiento

### Beneficios de velocidad
- La aceleración es proporcional a `selection_ratio` (más cerca de 1.0 = más tokens descartados = entrenamiento más rápido)
- **Las mayores mejoras ocurren con entradas de video largas y resoluciones más altas** debido a la complejidad O(n²) de atención
- Típicamente 20-40% de aceleración, pero los resultados varían según la configuración
- Con entrenamiento de pérdida enmascarada, la aceleración se reduce ya que tokens enmascarados no pueden descartarse

### Trade-offs de calidad
- **Mayor descarte de tokens lleva a mayor pérdida inicial** al iniciar entrenamiento LoRA/LoKr
- La pérdida suele corregirse rápido y las imágenes se normalizan pronto salvo que se use un selection ratio alto
  - Esto puede ser la red ajustándose a menos tokens en capas intermedias
- Ratios conservadores (0.1-0.25) suelen mantener calidad
- Ratios agresivos (>0.35) definitivamente impactarán la convergencia

### Consideraciones específicas de LoRA
- El rendimiento puede depender de los datos: configuraciones óptimas necesitan más exploración
- El pico de pérdida inicial es más notable con LoRA/LoKr que con fine-tuning completo

### Configuraciones recomendadas

Para equilibrio velocidad/calidad:
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

Para máxima velocidad (espera un gran pico de pérdida):
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

Para entrenamiento de alta resolución (1024px+):
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## Detalles técnicos

### Implementación del router

El router TREAD (clase `TREADRouter`) maneja:
- Cálculo de importancia de tokens vía norma L1
- Generación de permutaciones para enrutamiento eficiente
- Restauración de tokens preservando gradientes

### Integración con atención

TREAD modifica los embeddings posicionales rotatorios (RoPE) para coincidir con la secuencia enrutada:
- Los tokens de texto mantienen posiciones originales
- Los tokens de imagen usan posiciones mezcladas/recortadas
- Asegura consistencia posicional durante el enrutamiento
- **Nota**: La implementación de RoPE para FLUX puede no ser 100% correcta pero parece funcionar en la práctica

### Compatibilidad con pérdida enmascarada

Al usar entrenamiento de pérdida enmascarada:
- Los tokens dentro de la máscara se preservan automáticamente
- Evita que se descarte señal de entrenamiento importante
- Activado vía `conditioning_type` en ["mask", "segmentation"]
- **Nota**: Esto reduce la aceleración porque se deben procesar más tokens

## Problemas conocidos y limitaciones

### Estado de implementación
- **Función experimental** - El soporte TREAD es reciente y puede tener problemas no descubiertos
- **Manejo de RoPE** - La implementación de embeddings posicionales rotatorios puede no ser perfectamente correcta
- **Pruebas limitadas** - Las configuraciones óptimas no se han explorado extensivamente

### Comportamiento de entrenamiento
- **Pico de pérdida inicial** - Al iniciar LoRA/LoKr con TREAD, espera una pérdida inicial más alta que se corrige rápidamente
- **Rendimiento LoRA** - Algunas configuraciones pueden mostrar ligeras ralentizaciones con entrenamiento LoRA
- **Sensibilidad a configuración** - El rendimiento depende mucho de la configuración de enrutamiento

### Bugs conocidos (arreglados)
- El entrenamiento de pérdida enmascarada estaba roto en versiones anteriores pero se corrigió con verificación adecuada del modelo (`kontext` guard)

## Troubleshooting

### Problemas comunes

**"TREAD training requires you to configure the routes"**
- Asegúrate de que `tread_config` incluya un arreglo `routes`
- Cada ruta necesita `selection_ratio`, `start_layer_idx` y `end_layer_idx`

**Entrenamiento más lento de lo esperado**
- Verifica que las rutas cubran rangos de capas relevantes
- Considera ratios de selección más agresivos
- Comprueba que gradient checkpointing no esté en conflicto
- Para entrenamiento LoRA, se espera algo de ralentización: prueba distintas configuraciones

**Pérdida inicial alta con LoRA/LoKr**
- Esto es comportamiento esperado: la red necesita adaptarse a menos tokens
- La pérdida suele corregirse en unos cientos de pasos
- Si la pérdida no mejora, reduce `selection_ratio` (mantén más tokens)

**Degradación de calidad**
- Reduce los ratios de selección (mantén más tokens)
- Evita enrutamiento en capas tempranas (0-2) o finales
- Asegura suficientes datos de entrenamiento para la mayor eficiencia

## Ejemplos prácticos

### Entrenamiento de alta resolución (1024px+)
Para máximo beneficio al entrenar a altas resoluciones:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### Fine-tuning LoRA
Configuración conservadora para minimizar el pico de pérdida inicial:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### Entrenamiento con pérdida enmascarada
Cuando se entrena con máscaras, los tokens en regiones enmascaradas se preservan:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
Nota: La aceleración real será menor de lo que sugiere 0.7 debido a la preservación forzada de tokens.

## Trabajo futuro

Como el soporte TREAD en SimpleTuner es reciente, hay varias áreas de mejora:

- **Optimización de configuración** - Más pruebas para encontrar configuraciones óptimas para diferentes casos de uso
- **Rendimiento LoRA** - Investigación de por qué algunas configuraciones LoRA muestran ralentizaciones
- **Implementación de RoPE** - Refinamiento del manejo de embeddings posicionales rotatorios para mayor corrección
- **Soporte extendido de modelos** - Implementación para otras arquitecturas de transformers de difusión más allá de Flux
- **Configuración automatizada** - Herramientas para determinar automáticamente el enrutamiento óptimo según modelo y dataset

Las contribuciones de la comunidad y resultados de pruebas son bienvenidos para ayudar a mejorar el soporte TREAD.

## Referencias

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [Documentación de SimpleTuner Flux](quickstart/FLUX.md#tread-training)

## Cita

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Björn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```
