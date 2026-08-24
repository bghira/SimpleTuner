# Transforms de Clonacion de Voz

Los transforms de clonacion de voz son una funcion experimental planificada para datasets de audio. Sirven para expandir un conjunto de entrenamiento transfiriendo una identidad vocal a canciones, stems o interpretaciones adicionales antes de que empiece el entrenamiento principal del modelo.

El objetivo no es convertir SimpleTuner en una estacion separada de conversion de voz. El objetivo es reducir el entrelazado del dataset de audio. Si una voz solo aparece en un estilo estrecho, una LoRA puede aprender "esta voz dentro de este arreglo" en vez de la identidad vocal en si. Un split expandido con voice cloning puede poner la misma identidad vocal en arreglos, captions, letras y estructuras musicales mas variadas.

Esta funcion esta pensada solo para datasets de audio.

!!! warning "Consentimiento y derechos"
    Usa este workflow solo con voces y grabaciones que tienes permiso para usar. La identidad vocal es dato biometrico y creativo sensible. El transform puede crear audio derivado que suena como una persona real, asi que permisos, licencias y divulgacion importan.

## ELI5

Imagina que tienes seis grabaciones de una persona cantando, pero todas son de la misma banda y del mismo genero. Si entrenas solo con esas canciones, el modelo puede aprender que la voz, el tono de guitarra, la bateria, el rango de tempo y la estructura de cancion son una sola cosa.

Los transforms de clonacion de voz intentan separar esas ideas:

1. Aprenden un pequeno modelo de conversion de voz a partir de ejemplos de la voz objetivo.
2. Toman un conjunto mas amplio de canciones o vocal stems.
3. Sustituyen el timbre vocal de origen por el timbre objetivo.
4. Mantienen las nuevas captions y letras alineadas con el audio generado.
5. Agregan el audio generado como otro split normal de entrenamiento.

Asi el modelo principal ve la voz objetivo en mas contextos, no solo memorizando el dataset estrecho original.

## Para Que Sirve

Usalo cuando:

- tienes grabaciones autorizadas de la voz objetivo
- la identidad objetivo esta demasiado ligada a un genero, banda, produccion o estructura de cancion
- los trigger words solo funcionan dentro del dominio original
- dos o mas voces en un dataset se mezclan en una voz promedio
- quieres LoRAs separadas para identidades vocales separadas
- quieres que SimpleTuner prepare el split expandido dentro del mismo setup de entrenamiento

Evitalo cuando:

- ya tienes un dataset grande, variado y limpio de la misma voz
- el audio fuente de expansion es de baja calidad o no coincide con las captions
- necesitas publicar resultados y no tienes derechos claros
- el modelo generativo base no aprende la identidad ni con ejemplos directos limpios

## Como Entra En El Entrenamiento

La clonacion de voz es un transform de preparacion de datos, no un dataset de conditioning.

`conditioning_data` es para entradas auxiliares pareadas que permanecen unidas a una muestra primaria durante el entrenamiento, como imagenes de referencia o mapas de conditioning generados.

La clonacion de voz debe vivir en una lista `data_transforms` a nivel de dataset. El transform materializa nuevos archivos de audio, captions y letras opcionales, y registra el resultado como otro dataset primario `audio`. Despues, el dataloader normal lo ve como cualquier otro split de entrenamiento.

Forma de pseudo config:

```text
audio dataset:
    id: target-voice
    dataset_type: audio
    data_transforms:
        - task: identity_transfer
          source: expansion-audio-backend
          target: generated-audio-backend
          method: rvc
          audio_mode: separate_convert_remix
```

Comportamiento de inicio en pseudocodigo:

```text
for each audio dataset:
    for each data transform:
        if task is identity_transfer:
            prepare or reuse the target voice-conversion model
            prepare or reuse generated audio
            append generated audio as a normal train split

continue with normal metadata discovery, bucketing, caching, and training
```

## Transferencia de Identidad Estilo RVC

La primera implementacion prevista es conversion de voz estilo RVC.

En este contexto, el "modelo RVC" es especifico de la voz. Se entrena desde el dataset de identidad objetivo. El indice de recuperacion tambien es especifico de la voz y se construye desde features de esa misma voz. Componentes preentrenados amplios, como features de contenido, extraccion de pitch o modelos de separacion, son infraestructura reutilizable; el modelo de conversion y el indice son artifacts especificos del cantante o hablante.

SimpleTuner deberia poder:

1. Reutilizar un modelo de conversion de voz y un indice provistos.
2. Entrenar el modelo de conversion si no se proporciona ninguno.
3. Construir el indice de recuperacion desde los datos de la voz objetivo.
4. Cachear modelo, indice y audio generado bajo el directorio de salida del entrenamiento.
5. Reutilizar artifacts cacheados al inicio cuando los datos y settings no cambiaron.
6. Opcionalmente reutilizar o publicar el modelo de conversion de voz mediante un repositorio de modelo en el Hub.

## Comportamiento Por Defecto

Los defaults planificados son conservadores:

| Setting | Default | Por que |
| --- | --- | --- |
| `task` | `identity_transfer` | Identifica explicitamente el transform. |
| `method` | `rvc` | Primer backend de transferencia vocal soportado. |
| `train_if_missing` | `true` | SimpleTuner debe poder bootstrapear el modelo vocal desde el dataset objetivo. |
| `force_retrain` | `false` | Reutiliza un modelo cacheado valido cuando sea posible. |
| `build_index` | `true` | Retrieval suele mejorar estabilidad de identidad y reducir leakage. |
| `hub_model_id` | sin definir | No se usa cache remoto de modelo vocal sin opt-in del usuario. |
| `reuse_from_hub` | `true` cuando `hub_model_id` esta definido | Revisa el Hub antes de gastar tiempo entrenando un modelo bajo demanda. |
| `push_to_hub` | `false` | Subir un modelo vocal debe ser explicito porque el artifact representa una identidad vocal. |
| `audio_mode` | `separate_convert_remix` para canciones completas, `vocal_only` para vocal stems | Las mezclas completas necesitan separacion; los stems no. |
| `separation_method` | `demucs` cuando se necesita separacion | Demucs es el stem separator default esperado. |
| tipo del split generado | dataset primario `audio` | Los datos generados entrenan como audio normal, no como conditioning. |
| ubicacion de cache | dentro de `output_dir` | Mantiene artifacts ligados al entrenamiento y reutilizables al reiniciar. |
| captions | copia captions fuente salvo configuracion diferente | El nuevo split debe preservar letras y contexto de arreglo. |

Si se proporciona un modelo de conversion existente, SimpleTuner debe usarlo y solo entrenar uno nuevo cuando se pida explicitamente o falten artifacts necesarios.

## Cache en el Hub

Un modelo de conversion de voz puede ser lo bastante caro como para que entrenarlo bajo demanda repetidamente sea una trampa. Por eso el transform debe soportar un cache opcional en el Hub para el modelo vocal y el indice de retrieval.

El orden seguro de busqueda es:

```text
if local voice-conversion cache matches:
    reuse local model and index
else if hub_model_id is configured and reuse_from_hub is enabled:
    check the Hub repository
    download only if it has a SimpleTuner voice-transform manifest
    reuse only if the manifest matches this transform
else if train_if_missing is enabled:
    train the voice-conversion model
    build the retrieval index
    cache locally
    push to hub only when push_to_hub is true
else:
    stop and ask for a model path or a reusable cache
```

El repositorio del Hub debe usar un layout especifico de SimpleTuner, no una coleccion suelta de archivos:

```text
voice_transform/
    manifest.json
    model.pth
    index.index
    README.md
```

El manifest es el contrato. Debe registrar el fingerprint del dataset de identidad objetivo, settings de entrenamiento RVC, settings del indice, sample rate esperado, versiones de herramientas y version del formato voice-transform de SimpleTuner. SimpleTuner no debe reutilizar un artifact del Hub si no tiene este manifest o si el manifest no coincide con el transform actual. Eso evita aplicar silenciosamente el modelo vocal equivocado a un dataset nuevo.

Publicar debe ser opt-in. Una pseudo config razonable:

```text
identity_transfer:
    method: rvc
    model:
        train_if_missing: true
        hub_model_id: org/target-voice-rvc
        reuse_from_hub: true
        push_to_hub: true
        private: true
```

Para identidades privadas, mantén el repositorio del Hub privado salvo permiso explicito para publicar el modelo vocal. El audio generado y los artifacts del modelo pueden tener derechos diferentes, asi que trata sus settings de upload por separado.

## Configuracion en WebUI

El entrenamiento del modelo RVC debe poder configurarse desde WebUI, no solo con JSON crudo del dataloader.

La forma esperada en WebUI es un editor de transforms dentro del dataset de audio:

```text
Audio dataset
    Data transforms
        Add transform: Identity transfer
            Method: RVC
            Audio mode: vocal_only / separate_convert_remix / full_mix_convert
            Train RVC model if missing: on
            Force retrain: off
            Build retrieval index: on
            Hub model id: optional
            Reuse from Hub: on when Hub model id is set
            Push RVC model to Hub: off by default
            Hub repo privacy: private by default
            Caption rules: copy, append, remove
```

La WebUI debe hacer obvios los dos setups comunes:

- **Ya tienes vocal stems:** elige `vocal_only`, deja Demucs desactivado y escribe vocal stems generados.
- **Tienes canciones completas:** elige `separate_convert_remix`, usa separacion con Demucs, convierte solo el vocal stem y remezcla con los stems instrumentales originales.

La interfaz debe mostrar que el audio generado se convierte en otro split primario de entrenamiento de audio. No debe presentar identity transfer como `conditioning_data`, porque eso implicaria comportamiento de conditioning pareado durante el entrenamiento.

## Comportamiento Distribuido al Inicio

Cuando SimpleTuner inicia con varios ranks data-parallel, el startup de voice cloning debe usar las GPUs disponibles en vez de hacer que rank 0 haga todo el trabajo.

Hay dos fases distribuidas separadas:

1. **Entrenamiento del modelo RVC:** si `train_if_missing=true`, no hay cache local coincidente y no hay artifact coincidente en el Hub, el loop de entrenamiento RVC debe ejecutarse con DDP cuando `world_size > 1`. Cada rank debe recibir batches distintos de la voz objetivo mediante el patron normal de distributed sampler.
2. **Preparacion del audio generado:** las entradas fuente de expansion deben dividirse por rank, parecido a TextEmbedCache y VAECache. Cada rank separa, convierte y escribe solo su shard; luego todos los ranks sincronizan antes de continuar metadata discovery.

Pseudocodigo:

```text
if world_size > 1:
    if RVC model must be trained:
        train RVC with DDP across all ranks
        save final model and index once

    split expansion inputs by global rank
    each rank generates its own audio shard
    barrier
    rank 0 writes or verifies the combined manifest
    barrier
else:
    train and generate serially
```

Solo un proceso debe publicar el modelo vocal final en el Hub. Lo mismo aplica a updates finales del manifest. Los outputs generados por rank pueden escribirse independientemente si los nombres son deterministicos y no se superponen.

Esto evita desperdiciar tiempo de GPU en sistemas multi-GPU y mantiene el startup alineado con el modelo existente de preparacion de cache de SimpleTuner.

## Logs del Entrenamiento RVC

El entrenamiento RVC en startup no debe crear runs de TensorBoard o WandB todavia. Esos loggers se configuran para el trabajo principal de entrenamiento de SimpleTuner, y reutilizarlos para un trabajo anidado de conversion de voz exigiria nombres de run, rutas, reglas de resume y politicas de artifact adicionales.

La etapa RVC aun puede reportar stats utiles mediante el logger nativo de entrenamiento de SimpleTuner:

```text
output_dir/
    logs/
        rvc/
            training_stats.jsonl
            summary.json
```

Stats locales utiles incluyen loss de entrenamiento RVC, pitch loss si esta habilitado, reconstruction o discriminator loss cuando aplique, samples procesados, tiempo transcurrido, DDP world size, motivo de cache hit o miss, y si el modelo final vino de cache local, cache del Hub o entrenamiento bajo demanda.

Estos stats son solo locales salvo que una implementacion futura agregue explicitamente integracion con logger externo para RVC transforms.

## Elegir `audio_mode`

### `vocal_only`

Usalo cuando tu dataset de expansion ya esta preprocesado como vocal stems limpios.

```text
source vocal stem -> RVC conversion -> generated vocal stem
```

Gotchas:

- No ejecutes Demucs de nuevo sobre stems limpios sin una razon clara.
- Las captions deben describir voces y letras, no un arreglo completo de banda, salvo que vayas a remezclar despues.
- Si el modelo principal espera canciones completas, los datos vocal-only pueden ensenar una distribucion distinta.

### `separate_convert_remix`

Usalo cuando el dataset de expansion contiene canciones completas mezcladas.

```text
source full song
    -> Demucs separates vocals and instrumental stems
    -> RVC converts the vocal stem
    -> converted vocal is remixed with the original instrumental stems
    -> generated full song is added to training
```

Este es el modo preferido para expansion de canciones completas porque evita convertir bateria, bajo, guitarras, sala y artifacts de master como si fueran parte de la voz.

Gotchas:

- La separacion de stems puede dejar bleed, artifacts o problemas de fase.
- Si el vocal stem es debil, reverberante o enterrado, la voz convertida puede volverse inestable.
- El loudness del remix importa. Un split generado consistentemente mas alto o bajo puede sesgar el entrenamiento.
- Las captions deben describir el resultado remixado final, no solo la cancion fuente.

### `full_mix_convert`

Usalo solo para pruebas rapidas.

```text
source full song -> RVC conversion over the whole mix -> generated full song
```

Es rapido, pero normalmente tiene menor calidad. Puede arrastrar instrumentos por el conversor de voz y ensenar artifacts no deseados a la LoRA final.

## Captions y Letras

El split generado debe tener captions que coincidan con el audio generado.

Un buen default:

```text
copy source caption
remove source-vocal identity words when configured
append target-vocal identity or style words when configured
copy lyrics sidecar when lyrics still match
```

Para letras, copiar suele ser correcto cuando la interpretacion fuente y la convertida usan las mismas palabras. No es correcto cuando el transform cambia la cancion, edita secciones, quita voces o usa una fuente sin letra.

Copiar captions a ciegas puede estar mal. Si la caption fuente dice "female pop vocal" y la salida convertida tiene timbre masculino de rock, la caption debe ajustarse. El transform debe soportar reglas simples de append/remove; reescritura avanzada de captions puede venir despues.

## Cache y Reuso

El transform debe escribir dos tipos de cache:

```text
voice-conversion cache:
    model checkpoint
    retrieval index
    manifest

generated audio cache:
    generated audio files
    captions
    lyrics, when available
    manifest
```

El manifest debe registrar fingerprint del dataset de identidad, settings del transform, fingerprint de los datos fuente de expansion y versiones de herramientas. Si coinciden, el inicio puede reutilizar artifacts existentes. Si cambian, SimpleTuner debe regenerar solo la etapa afectada.

## Consejos Practicos

- Mantén una voz objetivo por LoRA cuando importa el control de identidad.
- Prefiere ejemplos vocales limpios y secos para entrenar el modelo de conversion.
- Evita duetos salvo que el objetivo sea aprender la mezcla del dueto.
- Usa canciones de expansion con variedad de tempo, tonalidad, genero, dinamica y fraseo.
- Varia captions para que los tokens de identidad no queden pegados a un solo arreglo.
- Revisa audio generado antes de entrenamientos largos.

## Fallos Comunes

| Sintoma | Causa probable |
| --- | --- |
| La LoRA solo funciona en un genero | La identidad vocal sigue entrelazada con captions de arreglo o datos fuente. |
| El split generado suena hueco o con fase rara | Artifacts de separacion/remix en procesamiento de canciones completas. |
| Los instrumentos suenan convertidos como voz | Se uso `full_mix_convert` cuando hacia falta separacion. |
| La identidad vocal es debil | Faltan datos objetivo mas limpios, mas datos o mejor indice. |
| Las captions no controlan la voz | Las captions aun mencionan la voz fuente u omiten la identidad objetivo. |
| El modelo principal aprende artifacts | El audio generado tiene baja calidad o demasiado peso en el mix de entrenamiento. |

## Relacion Con Datos de Regularizacion

Los datos generados por identity transfer no son datos de regularizacion por defecto.

Los datos de regularizacion suelen ensenar a la LoRA a preservar el comportamiento del modelo base. Los datos de identity transfer ensenan una voz objetivo en mas contextos. Demasiada regularizacion con pocos datos directos de identidad puede debilitar tokens de identidad. Demasiados datos generados pueden ensenar artifacts de conversion.

Tratalos como controles separados:

- dataset objetivo directo: senal de identidad mas fuerte
- dataset generado por identity transfer: mayor cobertura de contexto y estilo
- dataset de regularizacion: preservacion del modelo base

## Estado

Esta pagina describe el comportamiento previsto para un workflow experimental `data_transforms`. La restriccion importante de diseno es que identity transfer sea una funcion de audio de primera clase en SimpleTuner: entrenar o reutilizar el modelo de voice conversion, construir o reutilizar el indice, generar el split expandido, cachear resultados y continuar al entrenamiento normal sin exigir una segunda etapa manual de preprocessing.
