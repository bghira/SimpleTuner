/**
 * Audit Log Alpine.js Component
 *
 * Provides audit log viewing, filtering, chain verification,
 * and export functionality.
 */

if (!window.auditLogComponent) {
    window.auditLogComponent = function() {
        return {
            // State
            loaded: false,
            loading: false,
            entries: [],
            eventTypes: [],
            stats: {
                total_entries: 0,
                last_24h: 0,
                security_events_24h: 0,
                failed_logins_24h: 0,
                first_entry: null,
                last_entry: null,
            },

            // Filters
            filters: {
                event_type: '',
                actor: '',
                target_id: '',
                since: '',
                until: '',
            },
            securityOnly: false,

            // Pagination
            limit: 50,
            offset: 0,
            hasMore: false,

            // Chain verification
            chainVerified: null,
            verifyResult: null,
            verifying: false,

            // UI state (using HintMixin for hero CTA)
            ...(window.HintMixin?.createSingleHint({
                useApi: true,
                hintKey: 'audit_hero',
            }) || { heroDismissed: false, loadHeroCTAState() {}, dismissHeroCTA() {}, restoreHeroCTA() {} }),
            selectedEntry: null,

            // Export configuration
            exportConfig: {
                format: 'json',
                webhook_url: '',
                auth_token: '',
                security_only: false,
            },

            // Modal instances
            detailsModalInstance: null,
            exportModalInstance: null,

            async init() {
                // Wait for auth before making any API calls
                const canProceed = await window.waitForAuthReady();
                if (!canProceed) {
                    return;
                }

                // Load hero CTA dismissed state
                await this.loadHeroCTAState();
                // Load event types
                await this.loadEventTypes();
                // Load stats
                await this.loadStats();
                // Load initial entries
                await this.loadEntries();
                // Load export config
                await this.loadExportConfig();

                this.loaded = true;

                // Initialize Bootstrap modals when DOM is ready
                this.$nextTick(() => {
                    if (typeof bootstrap !== 'undefined') {
                        const detailsModalEl = document.getElementById('auditEntryDetailsModal');
                        if (detailsModalEl) {
                            this.detailsModalInstance = new bootstrap.Modal(detailsModalEl);
                        }
                        const exportModalEl = document.getElementById('auditExportOptionsModal');
                        if (exportModalEl) {
                            this.exportModalInstance = new bootstrap.Modal(exportModalEl);
                        }
                    }
                });
            },

            // --- Data Loading ---

            async loadEntries() {
                this.loading = true;
                try {
                    const params = new URLSearchParams();
                    params.set('limit', this.limit);
                    params.set('offset', this.offset);

                    if (this.filters.event_type) params.set('event_type', this.filters.event_type);
                    if (this.filters.actor) params.set('actor_id', this.filters.actor);
                    if (this.filters.target_id) params.set('target_id', this.filters.target_id);
                    if (this.filters.since) params.set('since', this.filters.since);
                    if (this.filters.until) params.set('until', this.filters.until);

                    const endpoint = this.securityOnly ? '/api/audit/security' : '/api/audit';
                    const response = await fetch(`${endpoint}?${params}`);

                    if (response.ok) {
                        const data = await response.json();
                        this.entries = data.entries || [];
                        this.hasMore = data.has_more || (this.entries.length === this.limit);
                    } else {
                        console.error('Failed to load audit entries:', response.status);
                        window.showToast?.('Failed to load audit entries', 'error');
                    }
                } catch (error) {
                    console.error('Failed to load audit entries:', error);
                    window.showToast?.('Failed to load audit entries', 'error');
                } finally {
                    this.loading = false;
                }
            },

            async loadStats() {
                try {
                    const response = await fetch('/api/audit/stats');
                    if (response.ok) {
                        const data = await response.json();
                        this.stats = {
                            total_entries: data.total_entries || 0,
                            last_24h: data.last_24h || 0,
                            security_events_24h: (data.by_type?.AUTH_LOGIN_FAILED || 0) +
                                                 (data.by_type?.PERMISSION_DENIED || 0) +
                                                 (data.by_type?.RATE_LIMITED || 0) +
                                                 (data.by_type?.SUSPICIOUS_ACTIVITY || 0),
                            failed_logins_24h: data.by_type?.AUTH_LOGIN_FAILED || 0,
                            first_entry: data.first_entry,
                            last_entry: data.last_entry,
                        };
                    }
                } catch (error) {
                    console.warn('Failed to load audit stats:', error);
                }
            },

            async loadEventTypes() {
                try {
                    const response = await fetch('/api/audit/types');
                    if (response.ok) {
                        const data = await response.json();
                        this.eventTypes = data.event_types || [];
                    }
                } catch (error) {
                    console.warn('Failed to load event types:', error);
                }
            },

            // Hero CTA methods provided by HintMixin spread above

            // --- Filtering ---

            applyFilters() {
                this.offset = 0;
                this.loadEntries();
            },

            clearFilters() {
                this.filters = {
                    event_type: '',
                    actor: '',
                    target_id: '',
                    since: '',
                    until: '',
                };
                this.securityOnly = false;
                this.offset = 0;
                this.loadEntries();
            },

            hasActiveFilters() {
                return this.filters.event_type ||
                       this.filters.actor ||
                       this.filters.target_id ||
                       this.filters.since ||
                       this.filters.until ||
                       this.securityOnly;
            },

            toggleSecurityOnly() {
                this.securityOnly = !this.securityOnly;
                this.offset = 0;
                this.loadEntries();
            },

            // --- Pagination ---

            nextPage() {
                this.offset += this.limit;
                this.loadEntries();
            },

            prevPage() {
                this.offset = Math.max(0, this.offset - this.limit);
                this.loadEntries();
            },

            // --- Chain Verification ---

            async verifyChain() {
                this.verifying = true;
                this.verifyResult = null;
                try {
                    const response = await fetch('/api/audit/verify');
                    if (response.ok) {
                        this.verifyResult = await response.json();
                        this.chainVerified = this.verifyResult.valid;
                        if (this.verifyResult.valid) {
                            window.showToast?.('Audit chain integrity verified', 'success');
                        } else {
                            window.showToast?.('Audit chain integrity check failed', 'error');
                        }
                    } else {
                        window.showToast?.('Failed to verify audit chain', 'error');
                    }
                } catch (error) {
                    console.error('Failed to verify audit chain:', error);
                    window.showToast?.('Failed to verify audit chain', 'error');
                } finally {
                    this.verifying = false;
                }
            },

            // --- Entry Details ---

            showEntryDetails(entry) {
                this.selectedEntry = entry;
                // Lazy-initialize modal if not already done (handles HTMX tab loads)
                if (!this.detailsModalInstance && typeof bootstrap !== 'undefined') {
                    const detailsModalEl = document.getElementById('auditEntryDetailsModal');
                    if (detailsModalEl) {
                        this.detailsModalInstance = new bootstrap.Modal(detailsModalEl);
                    }
                }
                if (this.detailsModalInstance) {
                    this.detailsModalInstance.show();
                }
            },

            // --- Export ---

            async exportLogs(format) {
                try {
                    const params = new URLSearchParams();
                    params.set('format', format);
                    params.set('limit', 10000); // Export up to 10k entries

                    if (this.filters.event_type) params.set('event_type', this.filters.event_type);
                    if (this.filters.since) params.set('since', this.filters.since);
                    if (this.filters.until) params.set('until', this.filters.until);
                    if (this.securityOnly) params.set('security_only', 'true');

                    const endpoint = this.securityOnly ? '/api/audit/security' : '/api/audit';
                    const response = await fetch(`${endpoint}?${params}`);

                    if (response.ok) {
                        const data = await response.json();
                        const entries = data.entries || [];

                        let content, mimeType, filename;
                        const timestamp = new Date().toISOString().slice(0, 19).replace(/[:-]/g, '');

                        if (format === 'csv') {
                            content = this.entriesToCSV(entries);
                            mimeType = 'text/csv';
                            filename = `audit_log_${timestamp}.csv`;
                        } else {
                            content = JSON.stringify(entries, null, 2);
                            mimeType = 'application/json';
                            filename = `audit_log_${timestamp}.json`;
                        }

                        // Create and download file
                        const blob = new Blob([content], { type: mimeType });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = filename;
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        URL.revokeObjectURL(url);

                        window.showToast?.(`Exported ${entries.length} entries`, 'success');
                    }
                } catch (error) {
                    console.error('Failed to export logs:', error);
                    window.showToast?.('Failed to export logs', 'error');
                }
            },

            entriesToCSV(entries) {
                if (!entries.length) return '';

                const headers = ['id', 'timestamp', 'event_type', 'actor_id', 'actor_username', 'actor_ip', 'action', 'target_type', 'target_id', 'details'];
                const rows = entries.map(e => [
                    e.id,
                    e.timestamp,
                    e.event_type,
                    e.actor_id || '',
                    e.actor_username || '',
                    e.actor_ip || '',
                    `"${(e.action || '').replace(/"/g, '""')}"`,
                    e.target_type || '',
                    e.target_id || '',
                    `"${JSON.stringify(e.details || {}).replace(/"/g, '""')}"`,
                ]);

                return [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
            },

            async loadExportConfig() {
                try {
                    const response = await fetch('/api/audit/export-config');
                    if (response.ok) {
                        const config = await response.json();
                        this.exportConfig = {
                            format: config.format || 'json',
                            webhook_url: config.webhook_url || '',
                            auth_token: config.auth_token || '',
                            security_only: config.security_only || false,
                        };
                    }
                } catch (error) {
                    console.error('Failed to load export config:', error);
                }
            },

            showExportModal() {
                // Lazy-initialize modal if not already done (handles HTMX tab loads)
                if (!this.exportModalInstance && typeof bootstrap !== 'undefined') {
                    const exportModalEl = document.getElementById('auditExportOptionsModal');
                    if (exportModalEl) {
                        this.exportModalInstance = new bootstrap.Modal(exportModalEl);
                    }
                }
                if (this.exportModalInstance) {
                    this.exportModalInstance.show();
                }
            },

            async saveExportConfig() {
                try {
                    const response = await fetch('/api/audit/export-config', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.exportConfig),
                    });

                    if (response.ok) {
                        window.showToast?.('Export configuration saved', 'success');
                        if (this.exportModalInstance) {
                            this.exportModalInstance.hide();
                        }
                    } else {
                        window.showToast?.('Failed to save export configuration', 'error');
                    }
                } catch (error) {
                    console.error('Failed to save export config:', error);
                    window.showToast?.('Failed to save export configuration', 'error');
                }
            },

            // --- Formatting Helpers ---

            formatTimestamp(ts) {
                return window.UIHelpers?.formatTimestamp(ts) || '-';
            },

            formatDate(ts) {
                return window.UIHelpers?.formatDate(ts) || '-';
            },

            formatEventType(eventType) {
                if (!eventType) return '-';
                return eventType.replace(/_/g, ' ').toLowerCase()
                    .replace(/\b\w/g, c => c.toUpperCase());
            },

            formatDetails(details) {
                if (!details) return '';
                if (typeof details === 'string') return details;
                return JSON.stringify(details, null, 2);
            },

            getEventTypeBadgeClass(eventType) {
                return window.UIHelpers?.getEventTypeBadgeClass(eventType) || 'bg-secondary';
            },

            isSecurityEvent(entry) {
                const securityTypes = ['AUTH_LOGIN_FAILED', 'PERMISSION_DENIED', 'RATE_LIMITED', 'SUSPICIOUS_ACTIVITY'];
                return securityTypes.includes(entry.event_type);
            },

            isCriticalEvent(entry) {
                const criticalTypes = ['SUSPICIOUS_ACTIVITY'];
                return criticalTypes.includes(entry.event_type);
            },
        };
    };
}
