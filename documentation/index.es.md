# SimpleTuner

**SimpleTuner** es un toolkit de ajuste fino de modelos de difusión multimodal, enfocado en la simplicidad y la facilidad de comprensión.

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __Primeros pasos__

    ---

    Instala SimpleTuner y entrena tu primer modelo en minutos

    [:octicons-arrow-right-24: Instalación](INSTALL.md)

-   :material-cog:{ .lg .middle } __Web UI__

    ---

    Configura y ejecuta entrenamiento mediante una interfaz web elegante

    [:octicons-arrow-right-24: Tutorial de Web UI](webui/TUTORIAL.md)

-   :material-api:{ .lg .middle } __API REST__

    ---

    Automatiza flujos de entrenamiento con la API HTTP

    [:octicons-arrow-right-24: Tutorial de API](api/TUTORIAL.md)

-   :material-cloud:{ .lg .middle } __Entrenamiento en la nube__

    ---

    Ejecuta entrenamiento en Replicate o workers distribuidos

    [:octicons-arrow-right-24: Entrenamiento en la nube](experimental/cloud/README.md)

-   :material-account-group:{ .lg .middle } __Multiusuario__

    ---

    Funcionalidades empresariales: SSO, cuotas, RBAC, orquestación de workers

    [:octicons-arrow-right-24: Guía empresarial](experimental/server/ENTERPRISE.md)

-   :material-book-open-variant:{ .lg .middle } __Guías de modelos__

    ---

    Guías paso a paso para Flux, SD3, SDXL, modelos de video y más

    [:octicons-arrow-right-24: Guías de modelos](quickstart/index.md)

</div>

## Funcionalidades

- **Entrenamiento multimodal** - Modelos de generación de imágenes, video y audio
- **Web UI y API** - Entrena vía navegador o automatiza con REST
- **Orquestación de workers** - Distribuye trabajos entre máquinas con GPU
- **Listo para empresas** - LDAP/OIDC SSO, RBAC, cuotas, registro de auditoría
- **Integración en la nube** - Replicate, workers autoalojados
- **Optimización de memoria** - DeepSpeed, FSDP2, cuantización

## Modelos compatibles

| Tipo | Modelos |
|------|--------|
| **Imagen** | Flux.1/2, SD3, SDXL, Chroma, Auraflow, PixArt, Sana, Lumina2, HiDream y más |
| **Video** | Wan, LTX Video, Hunyuan Video, Kandinsky 5, LongCat |
| **Audio** | ACE-Step |

Consulta [Guías de modelos](quickstart/index.md) para documentación completa.

## Comunidad

- [Discord](https://discord.gg/JGkSwEbjRb) - Terminus Research Group
- [GitHub Issues](https://github.com/bghira/SimpleTuner/issues) - Reportes de errores y solicitudes de funciones

## Licencia

SimpleTuner es software de código abierto.
