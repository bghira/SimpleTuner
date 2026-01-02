# Audit Logging

SimpleTuner's audit logging system provides a tamper-evident record of security-relevant events. All administrative actions, authentication events, and job operations are logged with cryptographic chain verification.

## Overview

The audit log captures:
- **Authentication events**: Login attempts (success/failure), logouts, session expirations
- **User management**: User creation, updates, deletions, permission changes
- **API key operations**: Key creation, revocation, usage
- **Credential management**: Provider credential changes
- **Job operations**: Submissions, cancellations, approvals

## Accessing Audit Logs

### Web UI

Navigate to the **Audit** tab in the admin panel to browse audit entries with filtering options.

### CLI

```bash
# List recent audit entries
simpletuner auth audit list

# Filter by event type
simpletuner auth audit list --event-type auth.login.failed

# Filter by user
simpletuner auth audit user 123

# View security events only
simpletuner auth audit security

# Get statistics
simpletuner auth audit stats

# Verify chain integrity
simpletuner auth audit verify
```

### API Endpoints

All endpoints require the `admin.audit` permission.

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/audit` | List audit entries with filters |
| GET | `/api/audit/stats` | Get audit statistics |
| GET | `/api/audit/types` | List available event types |
| GET | `/api/audit/verify` | Verify chain integrity |
| GET | `/api/audit/user/{user_id}` | Get entries for a user |
| GET | `/api/audit/security` | Get security-related events |

## Event Types

### Authentication Events

| Event | Description |
|-------|-------------|
| `auth.login.success` | Successful login |
| `auth.login.failed` | Failed login attempt |
| `auth.logout` | User logged out |
| `auth.session.expired` | Session expired |
| `auth.api_key.used` | API key was used |

### User Management Events

| Event | Description |
|-------|-------------|
| `user.created` | New user created |
| `user.updated` | User details updated |
| `user.deleted` | User deleted |
| `user.level.changed` | User level/role changed |
| `user.permission.changed` | User permission changed |

### API Key Events

| Event | Description |
|-------|-------------|
| `api_key.created` | New API key created |
| `api_key.revoked` | API key revoked |

### Credential Events

| Event | Description |
|-------|-------------|
| `credential.created` | Provider credential added |
| `credential.deleted` | Provider credential removed |
| `credential.used` | Credential was used |

### Job Events

| Event | Description |
|-------|-------------|
| `job.submitted` | Job submitted to queue |
| `job.cancelled` | Job was cancelled |
| `job.approved` | Job approval granted |
| `job.rejected` | Job approval denied |

## Query Parameters

When listing audit entries, you can filter by:

| Parameter | Type | Description |
|-----------|------|-------------|
| `event_type` | string | Filter by event type |
| `actor_id` | int | Filter by user who performed action |
| `target_type` | string | Filter by target resource type |
| `target_id` | string | Filter by target resource ID |
| `since` | ISO date | Start timestamp |
| `until` | ISO date | End timestamp |
| `limit` | int | Max entries (1-500, default 50) |
| `offset` | int | Pagination offset |

## Chain Integrity

Each audit entry includes:
- A cryptographic hash of its content
- A reference to the previous entry's hash
- Timestamp from a monotonic clock

This creates a hash chain that makes tampering detectable. Use the verify endpoint or CLI command to check integrity:

```bash
# Verify entire chain
simpletuner auth audit verify

# Verify specific range
simpletuner auth audit verify --start-id 100 --end-id 200
```

The verification checks:
1. Each entry's hash matches its content
2. Each entry correctly references the previous entry's hash
3. No gaps in the sequence

## Retention

Audit logs are stored in the SimpleTuner database. Configure retention in your deployment:

```bash
# Environment variable for retention period (days)
SIMPLETUNER_AUDIT_RETENTION_DAYS=365
```

Older entries can be archived or purged according to your compliance requirements.

## Security Considerations

- Audit logs are append-only; entries cannot be modified or deleted through the API
- The `admin.audit` permission is required to view logs
- Failed login attempts are logged with IP addresses for security monitoring
- Consider forwarding audit logs to a SIEM for production deployments
