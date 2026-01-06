# Funcionalidades de servidor y multiusuario

Este directorio contiene documentación de las funciones del lado del servidor de SimpleTuner que aplican tanto a despliegues de entrenamiento locales como en la nube.

## Contenido

- [Orquestación de workers](WORKERS.md) - Registro distribuido de workers, despacho de trabajos y gestión de flota de GPU
- [Guía empresarial](ENTERPRISE.md) - Despliegue multiusuario, SSO, aprobaciones, cuotas y gobernanza
- [Autenticación externa](EXTERNAL_AUTH.md) - Configuración de proveedores de identidad OIDC y LDAP
- [Registro de auditoría](AUDIT.md) - Registro de eventos de seguridad con verificación de cadena

## Cuándo usar estos documentos

Estas funciones son relevantes cuando:

- Se distribuye el entrenamiento en varias máquinas con GPU mediante orquestación de workers
- Se ejecuta SimpleTuner como un servicio compartido para varios usuarios
- Se integra con proveedores de identidad corporativos (Okta, Azure AD, Keycloak, LDAP)
- Se requieren flujos de aprobación para el envío de trabajos
- Se registran acciones de usuarios por cumplimiento o seguridad
- Se gestionan cuotas de equipo y límites de recursos

Para documentación específica de la nube (Replicate, colas de trabajos, webhooks), consulta [Entrenamiento en la nube](../cloud/README.md).
