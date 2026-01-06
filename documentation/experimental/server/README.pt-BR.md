# Servidor e recursos multi-usuario

Este diretorio contem documentacao para os recursos do lado do servidor do SimpleTuner que se aplicam tanto a deploys locais quanto em nuvem.

## Conteudo

- [Orquestracao de Workers](WORKERS.md) - Registro de workers distribuidos, despacho de jobs e gerenciamento de frota de GPUs
- [Guia Enterprise](ENTERPRISE.md) - Deploy multi-usuario, SSO, aprovacoes, cotas e governanca
- [Autenticacao Externa](EXTERNAL_AUTH.md) - Configuracao de provedores de identidade OIDC e LDAP
- [Audit Logging](AUDIT.md) - Logging de eventos de seguranca com verificacao de cadeia

## Quando usar estes docs

Esses recursos sao relevantes quando:

- Distribuir treinamento entre multiplas maquinas GPU com orquestracao de workers
- Rodar o SimpleTuner como um servico compartilhado para varios usuarios
- Integrar com provedores de identidade corporativos (Okta, Azure AD, Keycloak, LDAP)
- Exigir workflows de aprovacao para envio de jobs
- Rastrear acoes de usuarios para compliance ou seguranca
- Gerenciar cotas de equipe e limites de recursos

Para documentacao especifica de nuvem (Replicate, filas de jobs, webhooks), veja [Cloud Training](../cloud/README.md).
