/**
 * Admin Panel - Audit Log Module
 *
 * Handles audit log loading, filtering, and chain verification.
 */

window.adminAuditMethods = {
    loadAuditEntriesIfNeeded() {
        if (!this.audit.loaded) {
            this.loadAuditEntries();
            this.loadAuditStats();
            this.loadEventTypes();
        }
    },

    async loadAuditEntries() {
        this.audit.loading = true;
        try {
            const params = new URLSearchParams();
            params.set('limit', String(this.audit.limit));
            params.set('offset', String(this.audit.offset));
            if (this.audit.eventTypeFilter) params.set('event_type', this.audit.eventTypeFilter);
            if (this.audit.actorSearch) params.set('actor_id', this.audit.actorSearch);
            if (this.audit.sinceDate) params.set('since', this.audit.sinceDate);
            if (this.audit.untilDate) params.set('until', this.audit.untilDate);
            if (this.audit.securityOnly) params.set('security_only', 'true');

            const response = await fetch(`/api/audit?${params}`);
            if (response.ok) {
                const data = await response.json();
                this.audit.entries = data.entries || [];
                this.audit.hasMore = data.has_more || false;
                this.audit.loaded = true;
            }
        } catch (error) {
            console.error('Failed to load audit entries:', error);
        } finally {
            this.audit.loading = false;
        }
    },

    async loadAuditStats() {
        try {
            const response = await fetch('/api/audit/stats');
            if (response.ok) {
                this.audit.stats = await response.json();
            }
        } catch (error) {
            console.error('Failed to load audit stats:', error);
        }
    },

    async loadEventTypes() {
        try {
            const response = await fetch('/api/audit/types');
            if (response.ok) {
                const data = await response.json();
                this.audit.eventTypes = data.event_types || [];
            }
        } catch (error) {
            console.error('Failed to load event types:', error);
        }
    },

    async verifyAuditChain() {
        this.audit.verifying = true;
        this.audit.verifyResult = null;
        try {
            const response = await fetch('/api/audit/verify');
            if (response.ok) {
                this.audit.verifyResult = await response.json();
            } else {
                const errorData = await response.json().catch(() => ({}));
                this.audit.verifyResult = {
                    valid: false,
                    error: errorData.detail || 'Verification request failed',
                };
            }
        } catch (error) {
            console.error('Failed to verify audit chain:', error);
            this.audit.verifyResult = { valid: false, error: 'Network error' };
        } finally {
            this.audit.verifying = false;
        }
    },

    applyAuditFilters() {
        this.audit.offset = 0;
        this.loadAuditEntries();
    },

    clearAuditFilters() {
        this.audit.eventTypeFilter = '';
        this.audit.actorSearch = '';
        this.audit.sinceDate = '';
        this.audit.untilDate = '';
        this.audit.securityOnly = false;
        this.audit.offset = 0;
        this.loadAuditEntries();
    },

    toggleSecurityOnly() {
        this.audit.securityOnly = !this.audit.securityOnly;
        this.applyAuditFilters();
    },

    auditNextPage() {
        if (this.audit.hasMore) {
            this.audit.offset += this.audit.limit;
            this.loadAuditEntries();
        }
    },

    auditPrevPage() {
        if (this.audit.offset > 0) {
            this.audit.offset = Math.max(0, this.audit.offset - this.audit.limit);
            this.loadAuditEntries();
        }
    },

    formatAuditTimestamp(ts) {
        return window.UIHelpers?.formatDateTime(ts, { seconds: true }) || '';
    },

    getEventTypeBadgeClass(eventType) {
        return window.UIHelpers?.getEventTypeBadgeClass(eventType) || 'bg-secondary';
    },

    getAuditDetails(entry) {
        if (!entry.details) return '';
        if (typeof entry.details === 'string') return entry.details;
        return JSON.stringify(entry.details).substring(0, 100);
    },
};
