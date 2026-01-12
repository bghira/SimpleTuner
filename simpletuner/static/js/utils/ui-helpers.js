/**
 * UI Helpers - Centralized Badge and Icon Utilities
 *
 * Consolidates duplicate badge class and icon mapping functions
 * from various components into a single reusable module.
 */

window.UIHelpers = {
    // ==================== Error Handling ====================

    /**
     * Extract error message from API response data
     * Handles both string errors and Pydantic validation error arrays
     * @param {object} data - Response JSON data
     * @param {string} fallback - Fallback message if no error found
     * @returns {string} Human-readable error message
     */
    extractErrorMessage(data, fallback = 'An error occurred') {
        if (!data) return fallback;
        if (typeof data.detail === 'string') return data.detail;
        if (Array.isArray(data.detail)) {
            return data.detail.map(e => e.msg || e.message || String(e)).join(', ');
        }
        if (data.message) return data.message;
        if (data.error) return data.error;
        return fallback;
    },

    // ==================== Badge Classes ====================

    /**
     * Get badge class for event types (audit logs, notifications)
     * @param {string} eventType - Event type like 'AUTH_LOGIN_FAILED', 'JOB_CREATED', etc.
     * @returns {string} Bootstrap badge class
     */
    getEventTypeBadgeClass(eventType) {
        if (!eventType) return 'bg-secondary';
        const type = eventType.toLowerCase();
        if (type.includes('fail') || type.includes('denied') || type.includes('error')) return 'bg-danger';
        if (type.includes('warn') || type.includes('security') || type.includes('suspicious')) return 'bg-warning text-dark';
        if (type.includes('create') || type.includes('success') || type.includes('login')) return 'bg-success';
        if (type.includes('delete') || type.includes('remove')) return 'bg-danger';
        if (type.includes('update') || type.includes('modify')) return 'bg-info';
        if (type.includes('rate')) return 'bg-warning text-dark';
        return 'bg-secondary';
    },

    /**
     * Get badge class for user roles
     * @param {string} role - Role name like 'admin', 'lead', 'member'
     * @returns {string} Bootstrap badge class
     */
    getRoleBadgeClass(role) {
        switch (role) {
            case 'admin': return 'bg-danger';
            case 'lead': return 'bg-warning text-dark';
            default: return 'bg-secondary';
        }
    },

    /**
     * Get badge class for notification channel types
     * @param {string} type - Channel type: 'email', 'slack', 'webhook'
     * @returns {string} Bootstrap badge class
     */
    getChannelTypeBadgeClass(type) {
        const classes = {
            'email': 'bg-info',
            'slack': 'bg-warning text-dark',
            'webhook': 'bg-secondary',
        };
        return classes[type] || 'bg-secondary';
    },

    /**
     * Get badge class for severity levels
     * @param {string} severity - Severity: 'debug', 'info', 'warning', 'error', 'critical'
     * @returns {string} Bootstrap badge class
     */
    getSeverityBadgeClass(severity) {
        const classes = {
            debug: 'bg-secondary',
            info: 'bg-info',
            warning: 'bg-warning text-dark',
            error: 'bg-danger',
            critical: 'bg-dark',
        };
        return classes[severity] || 'bg-secondary';
    },

    /**
     * Get badge class for resource types
     * @param {string} resourceType - Type: 'config', 'hardware', 'provider', 'output_path'
     * @returns {string} Bootstrap badge class
     */
    getResourceTypeBadgeClass(resourceType) {
        const classes = {
            'config': 'bg-primary',
            'hardware': 'bg-warning text-dark',
            'provider': 'bg-info',
            'output_path': 'bg-secondary',
        };
        return classes[resourceType] || 'bg-secondary';
    },

    /**
     * Get badge class for allow/deny actions
     * @param {string} action - Action: 'allow' or 'deny'
     * @returns {string} Bootstrap badge class
     */
    getActionBadgeClass(action) {
        return action === 'allow' ? 'bg-success' : 'bg-danger';
    },

    /**
     * Get badge class for quota actions
     * @param {string} action - Action: 'block', 'warn', 'require_approval'
     * @returns {string} Bootstrap badge class
     */
    getQuotaActionBadgeClass(action) {
        const key = (action || '').toLowerCase();
        const classes = {
            'block': 'bg-danger',
            'warn': 'bg-warning text-dark',
            'require_approval': 'bg-info',
        };
        return classes[key] || 'bg-secondary';
    },

    // ==================== Icons ====================

    /**
     * Get FontAwesome icon class for notification channel types
     * @param {string} type - Channel type: 'email', 'slack', 'webhook'
     * @returns {string} FontAwesome icon class
     */
    getChannelIcon(type) {
        switch (type) {
            case 'email': return 'fas fa-envelope';
            case 'webhook': return 'fas fa-plug';
            case 'slack': return 'fab fa-slack';
            default: return 'fas fa-bell';
        }
    },

    /**
     * Get FontAwesome icon class for notification delivery status
     * @param {string} status - Status: 'delivered', 'failed', 'pending'
     * @returns {string} FontAwesome icon class (without 'fas' prefix)
     */
    getStatusIcon(status) {
        switch (status) {
            case 'delivered': return 'fa-check';
            case 'failed': return 'fa-times';
            case 'pending': return 'fa-clock';
            default: return 'fa-question';
        }
    },

    /**
     * Get full FontAwesome icon class with color for severity levels
     * @param {string} severity - Severity level
     * @returns {string} Full FontAwesome class with Bootstrap color
     */
    getSeverityIcon(severity) {
        const icons = {
            'success': 'fas fa-check-circle text-success',
            'info': 'fas fa-info-circle text-info',
            'warning': 'fas fa-exclamation-triangle text-warning',
            'error': 'fas fa-exclamation-circle text-danger',
            'danger': 'fas fa-exclamation-circle text-danger',
            'critical': 'fas fa-times-circle text-danger',
            'debug': 'fas fa-bug text-secondary',
            'secondary': 'fas fa-info-circle text-secondary',
        };
        return icons[severity] || 'fas fa-info-circle text-secondary';
    },

    /**
     * Get full FontAwesome icon class with color for event types
     * @param {string} eventType - Event type
     * @returns {string} Full FontAwesome class with Bootstrap color
     */
    getEventTypeIcon(eventType) {
        const icons = {
            'validation': 'fas fa-check-double text-success',
            'validation_complete': 'fas fa-check-double text-success',
            'checkpoint': 'fas fa-save text-primary',
            'progress': 'fas fa-chart-line text-info',
            'debug': 'fas fa-bug text-muted',
            'error': 'fas fa-exclamation-triangle text-danger',
            'notification': 'fas fa-bell text-muted',
        };
        return icons[eventType] || 'fas fa-info-circle text-secondary';
    },

    // ==================== Formatters ====================

    /**
     * Format event type for display (snake_case to Title Case)
     * @param {string} type - Event type like 'job_created'
     * @returns {string} Formatted string like 'Job Created'
     */
    formatEventType(type) {
        if (!type) return '';
        return type.replace(/\./g, ' ').replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    },

    // ==================== Date/Time Formatters ====================

    /**
     * Format a date as relative time (e.g., "5m ago", "2h ago", "3d ago")
     * Falls back to absolute date for older dates.
     * @param {string|Date} dateInput - ISO string or Date object
     * @param {Object} options - Options
     * @param {boolean} options.compact - Use compact format (default: false)
     * @param {number} options.relativeDays - Days before switching to absolute (default: 7)
     * @returns {string} Formatted relative or absolute date
     */
    formatRelativeTime(dateInput, options = {}) {
        if (!dateInput) return options.fallback || '';
        const { compact = false, relativeDays = 7 } = options;

        try {
            const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
            if (isNaN(date.getTime())) return options.fallback || '';

            const now = new Date();
            const diffMs = now - date;
            const diffSec = Math.floor(diffMs / 1000);
            const diffMin = Math.floor(diffSec / 60);
            const diffHr = Math.floor(diffMin / 60);
            const diffDay = Math.floor(diffHr / 24);

            // Relative formatting
            if (diffDay < relativeDays) {
                if (diffMin < 1) return compact ? 'now' : 'Just now';
                if (diffMin < 60) return `${diffMin}m ago`;
                if (diffHr < 24) return `${diffHr}h ago`;
                return `${diffDay}d ago`;
            }

            // Absolute formatting for older dates
            return date.toLocaleDateString();
        } catch {
            return options.fallback || String(dateInput);
        }
    },

    /**
     * Format a date with both date and time
     * @param {string|Date} dateInput - ISO string or Date object
     * @param {Object} options - Options
     * @param {boolean} options.seconds - Include seconds (default: false)
     * @returns {string} Formatted date and time
     */
    formatDateTime(dateInput, options = {}) {
        if (!dateInput) return options.fallback || '--';
        const { seconds = false } = options;

        try {
            const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
            if (isNaN(date.getTime())) return options.fallback || '--';

            const timeOpts = { hour: '2-digit', minute: '2-digit' };
            if (seconds) timeOpts.second = '2-digit';

            return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], timeOpts);
        } catch {
            return options.fallback || String(dateInput);
        }
    },

    /**
     * Format a date only (no time)
     * @param {string|Date} dateInput - ISO string or Date object
     * @param {Object} options - Options
     * @param {boolean} options.short - Use short format (default: false)
     * @returns {string} Formatted date
     */
    formatDate(dateInput, options = {}) {
        if (!dateInput) return options.fallback || '--';
        const { short = false } = options;

        try {
            const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
            if (isNaN(date.getTime())) return options.fallback || '--';

            if (short) {
                return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
            }
            return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
        } catch {
            return options.fallback || String(dateInput);
        }
    },

    /**
     * Format a timestamp for audit logs/events (compact with seconds)
     * @param {string|Date} dateInput - ISO string or Date object
     * @returns {string} Formatted timestamp
     */
    formatTimestamp(dateInput) {
        if (!dateInput) return '-';

        try {
            const date = dateInput instanceof Date ? dateInput : new Date(dateInput);
            if (isNaN(date.getTime())) return '-';

            return date.toLocaleString(undefined, {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
            });
        } catch {
            return '-';
        }
    },

    /**
     * Format a duration in seconds to human-readable format
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration like "2h 30m" or "45m"
     */
    formatDuration(seconds) {
        if (!seconds && seconds !== 0) return '--';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        if (hrs > 0) return `${hrs}h ${mins}m`;
        if (mins > 0) return `${mins}m`;
        return '< 1m';
    },

    /**
     * Format an estimated wait time
     * @param {number} seconds - Wait time in seconds
     * @returns {string} Formatted wait time with "~" prefix
     */
    formatWaitTime(seconds) {
        if (seconds === null || seconds === undefined) return 'N/A';
        if (seconds < 60) return '< 1 min';
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        if (hrs > 0) return `~${hrs}h ${mins}m`;
        return `~${mins} min`;
    },

    // ==================== Navigation Utilities ====================

    /**
     * Navigate to a specific tab by its data-tab attribute
     * @param {string} tabName - The value of the data-tab attribute
     * @returns {boolean} True if tab was found and clicked
     */
    navigateToTab(tabName) {
        const tabElement = document.querySelector(`[data-tab="${tabName}"]`);
        if (tabElement) {
            tabElement.click();
            return true;
        }
        return false;
    },

    /**
     * Navigate to a tab and optionally close a modal/panel
     * @param {string} tabName - The value of the data-tab attribute
     * @param {Function} closeCallback - Optional callback to close current modal/panel
     */
    navigateToTabAndClose(tabName, closeCallback) {
        if (closeCallback && typeof closeCallback === 'function') {
            closeCallback();
        }
        this.navigateToTab(tabName);
    },
};
