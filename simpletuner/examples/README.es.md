# Ejemplos de entrenamiento

Los ejemplos se organizan como `model_family/model_flavour/vram/recipe`. Elige la familia, su variante y un nivel de memoria como `24g`, `48g` u `80g`. El nivel de VRAM `default` no especifica una capacidad; la variante `default` utiliza el valor predeterminado del modelo. Un nivel como `8x80g` requiere ocho GPU de 80 GB cada una.

Las carpetas con nombre separan las recetas LoRA, LoKr, ControlNet, AnyFlow y las variantes de condicionamiento. Los planes de datos, bibliotecas de prompts y scripts de validación compartidos están en `_shared/`.

## Listar y ejecutar ejemplos

Después de instalar SimpleTuner, lista las recetas y pasa la ruta relativa completa a `example=`:

```bash
simpletuner examples list
simpletuner train example=flux3_image/klein/24g/peft-lora
```

## Copiar y personalizar

El comando conserva las carpetas del catálogo dentro del destino y copia los archivos JSON referenciados en la carpeta de la receta. Edita la configuración y el plan de datos antes de entrenar desde ese entorno:

```bash
simpletuner examples copy flux3_image/klein/24g/peft-lora --dest config
# config/flux3_image/klein/24g/peft-lora/config.json
cd config/flux3_image/klein/24g/peft-lora
simpletuner train
```

En el creador de entornos y el asistente de configuración de la WebUI, selecciona primero la familia. El selector solo muestra recetas de esa familia. Cambiarla borra la selección anterior.

Estas recetas son puntos de partida. Consulta la guía del modelo y adapta el lote, la resolución y la duración a tus datos y hardware.
