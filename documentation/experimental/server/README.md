# Server & Multi-User Features

This directory contains documentation for SimpleTuner's server-side features that apply to both local and cloud training deployments.

## Contents

- [Enterprise Guide](ENTERPRISE.md) - Multi-user deployment, SSO, approvals, quotas, and governance
- [External Authentication](EXTERNAL_AUTH.md) - OIDC and LDAP identity provider setup
- [Audit Logging](AUDIT.md) - Security event logging with chain verification

## When to Use These Docs

These features are relevant when:

- Running SimpleTuner as a shared service for multiple users
- Integrating with corporate identity providers (Okta, Azure AD, Keycloak, LDAP)
- Requiring approval workflows for job submission
- Tracking user actions for compliance or security
- Managing team quotas and resource limits

For cloud-specific documentation (Replicate, job queues, webhooks), see [Cloud Training](../cloud/README.md).
