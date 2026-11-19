# Arquitectura de la fábrica de backends de datos

## Construcción del dataset

La fábrica valida los identificadores, expande los datasets generados y ordena las fuentes de metadatos antes que sus dependientes. Los constructores abren los proveedores de origen y de metadatos. `DatasetRuntime` vincula los proveedores, la configuración resuelta, los datasets relacionados y las cachés de artefactos. `StateTracker` registra esos objetos por ID.

```text
FactoryRegistry → builders → DatasetRuntime
                              ├─ BaseDataBackend
                              ├─ MetadataBackend
                              ├─ TextEmbeddingCache / VAECache
                              └─ training dataloader
```

## Contratos de los proveedores

El proveedor de origen lee archivos u objetos. El proveedor de metadatos publica las muestras aceptadas y la geometría de recorte. `describe()` informa de recuentos y revisiones disponibles, `read_page()` devuelve una página acotada y `read_samples()` lee los campos solicitados en orden y conserva los duplicados. Las referencias de muestra pertenecen al esquema de identidad del proveedor.

Usa `iter_preparation_batches()` e `iter_preparation_text_records()` para preparar artefactos. Los proveedores asignan muestras aceptadas a los trabajadores con independencia de las repeticiones y del relleno del entrenamiento. Los artefactos pueden usar otro almacén. La codificación de texto en línea procesa el lote actual.

El descubrimiento Local, AWS y CSV comparten metadatos y planificación heredados. Hugging Face puede dividir las transformaciones preparadas por shard de Arrow. Parquet nativo de Webshart utiliza una vista de membresía indexada, transformaciones por shard para los recortes compatibles y un cursor de época reconstruible.

## Entrenamiento y reanudación

El entrenador solicita los lotes programados al runtime. El mezclador selecciona el dataset y el precargador puede adelantarse. El entrenador confirma la continuación después de consumir un lote. El checkpoint guarda la posición consumida y el estado del mezclador. La reanudación nativa requiere la misma vista aceptada, opciones de muestreo y configuración distribuida.

## Verificación

Al crear un proveedor, prueba en el runtime la estabilidad de las referencias, la proyección en orden, la cobertura de preparación, la cancelación y la reapertura. Si ofrece muestreo nativo, prueba la reproducción del cursor. Publica una revisión mientras otro consumidor conserva la anterior. Para un artefacto nuevo, comprueba que preparación y entrenamiento usan las mismas claves.

```bash
.venv/bin/python -m unittest -v -f tests.test_dataset_runtime tests.test_webshart_parquet_backend tests.test_vae_cache_traversal
```
