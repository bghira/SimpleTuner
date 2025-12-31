/**
 * Admin Panel - Hints & Utilities Module
 *
 * Handles dismissable hints and current user loading.
 */

window.adminHintMethods = {
    async loadHints() {
        this.hintsLoading = true;
        try {
            const response = await fetch('/api/cloud/hints');
            if (response.ok) {
                const data = await response.json();
                const dismissed = data.dismissed_hints || [];
                // Admin-specific hints
                this.hints.overview = !dismissed.includes('admin_overview');
                this.hints.users = !dismissed.includes('admin_users');
                this.hints.levels = !dismissed.includes('admin_levels');
                this.hints.rules = !dismissed.includes('admin_rules');
                this.hints.quotas = !dismissed.includes('admin_quotas');
                this.hints.approvals = !dismissed.includes('admin_approvals');
                this.hints.auth = !dismissed.includes('admin_auth');
                this.hints.orgs = !dismissed.includes('admin_orgs');
                this.hints.notifications = !dismissed.includes('admin_notifications');
            }
        } catch (error) {
            console.warn('Failed to load hints:', error);
        } finally {
            this.hintsLoading = false;
        }
    },

    async dismissHint(hintName) {
        this.hints[hintName] = false;
        try {
            await fetch(`/api/cloud/hints/dismiss/admin_${hintName}`, {
                method: 'POST',
            });
        } catch (error) {
            console.warn('Failed to dismiss hint:', error);
        }
    },

    async showHint(hintName) {
        this.hints[hintName] = true;
        try {
            await fetch(`/api/cloud/hints/show/admin_${hintName}`, {
                method: 'POST',
            });
        } catch (error) {
            console.warn('Failed to show hint:', error);
        }
    },

    // NOTE: anyHintsDismissed getter moved to index.js final return object

    restoreAllHints() {
        Object.keys(this.hints).forEach(k => this.showHint(k));
    },

    async loadCurrentUser() {
        try {
            const response = await fetch('/api/users/me');
            if (response.ok) {
                this.currentUser = await response.json();
            }
        } catch (error) {
            console.warn('Failed to load current user:', error);
        }
    },
};
