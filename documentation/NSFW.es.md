# Comprobaciones del Clasificador NSFW

SimpleTuner incluye comprobaciones opcionales de clasificador que pueden rechazar muestras durante el preprocesamiento del caché VAE. Esta función es una herramienta local de filtrado. No es asesoramiento legal, un sistema de cumplimiento ni una garantía de que un dataset sea legal o aceptable para un uso concreto.

## Tu responsabilidad

Tú eres responsable de decidir si tu dataset, ejecución de entrenamiento, salida del modelo y planes de publicación o distribución cumplen las reglas que se aplican a ti.

Esas reglas pueden incluir requisitos locales, regionales, nacionales y específicos de plataformas. Pueden depender del consentimiento, edad, derechos de imagen, privacidad, derechos de publicidad, reglas de obscenidad, políticas laborales o institucionales, y de si el resultado representa o suplanta a una persona real. Las leyes también cambian con el tiempo y varían según la jurisdicción.

SimpleTuner no decidirá esto por ti. No te avisará de que tu política está incompleta, no comprobará si tus umbrales coinciden con la ley ni confirmará que una salida del modelo sea segura para publicar. Si tienes dudas, busca asesoramiento legal cualificado para tu jurisdicción y caso de uso.

## Privacidad

Las comprobaciones del clasificador NSFW se ejecutan localmente en la máquina que ejecuta SimpleTuner.

- Esta función no envía muestras del dataset a una API de moderación de terceros.
- Los resultados del clasificador no se reenvían a terceros.
- La opción de telemetría de entrenamiento `--report_to` no recibe resultados del clasificador NSFW.
- Los reportes se almacenan localmente en la instancia, en el directorio de caché VAE, como `nsfw_classifier_report_rank*.json`.

El único comportamiento de red esperable es la carga normal de modelos desde Hugging Face si los pesos del clasificador aún no están en tu caché local de modelos. Una vez que el modelo está disponible localmente, la clasificación se ejecuta en la instancia.

## Comportamiento opt-in

La función está deshabilitada de forma predeterminada. Habilítala con:

```bash
--enable_nsfw_check=true
```

Las comprobaciones solo se aplican a muestras sin caché que el caché VAE está a punto de procesar. Los cachés VAE existentes se consideran confiables, y `skip_file_discovery=vae` omite la aplicación porque SimpleTuner asume que ya preparaste el caché bajo tu propia política.

Los datasets de evaluación no se escanean.

## Clasificadores compatibles

SimpleTuner admite modelos estándar de clasificación de imágenes de Hugging Face Transformers mediante `AutoImageProcessor` y `AutoModelForImageClassification`.

Los modelos predeterminados son:

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

Puedes proporcionar tu propia lista CSV:

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

SimpleTuner no habilita `trust_remote_code` para estos clasificadores y no añade `timm` como dependencia para esta función. Los modelos que requieren código personalizado o backends que no sean Transformers no están soportados por este escáner.

## Uso no NSFW

A pesar de los nombres de las opciones, este mecanismo no está limitado al filtrado de contenido sexual. Puede usarse para otras comprobaciones binarias o por puntuación de etiquetas si el clasificador emite etiquetas y puntuaciones reconocibles que se asignan claramente a las pistas unsafe/safe que SimpleTuner espera.

Ejemplos podrían incluir rechazar muestras con una categoría visual prohibida, contenido sensible para una marca u otra política local de dataset. Aun así, eres responsable de validar que las etiquetas, umbrales y ajustes de voto del clasificador coincidan con tu política.

## Contexto legal

El contenido sexual adulto no es automáticamente ilegal en todas partes, y SimpleTuner no prohíbe automáticamente el entrenamiento de modelos NSFW. Eso no significa que un dataset, salida o despliegue concreto sea legal.

Áreas de alto riesgo incluyen:

- Contenido que involucra menores o personas que aparentan ser menores. En Estados Unidos, el FBI Internet Crime Complaint Center afirma que el material de abuso sexual infantil creado por IA generativa y herramientas similares es ilegal.
- Imágenes íntimas no consensuadas, explotación sexual, acoso, chantaje o distribución sin permiso.
- Salidas que suplantan, recrean o representan de forma engañosa a una persona real, especialmente con fines sexuales, fraudulentos o dañinos para la reputación. La FTC ha destacado riesgos de suplantación con IA y fraude mediante deepfakes.
- Reglas de divulgación y transparencia de deepfakes. Por ejemplo, el Artículo 50 del EU AI Act incluye obligaciones de transparencia para ciertos contenidos de imagen, audio o video generados o manipulados por IA que constituyen deepfakes.
- Reglas contractuales o de plataforma, incluidas licencias de datasets, políticas de proveedores de hosting, reglas laborales, reglas de procesadores de pago y términos de distribución de modelos.

Trata el clasificador como un control dentro de tu propio proceso de revisión, no como el proceso de revisión completo.

## Opciones relacionadas

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

Consulta [DATALOADER.es.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.es.md#nsfw-classifier-checks-during-vae-caching) para detalles de la integración con el caché VAE.

## Referencias

- [FBI IC3: Child Sexual Abuse Material Created by Generative AI and Similar Online Tools is Illegal](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC: Proposed protections to combat AI impersonation of individuals](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
