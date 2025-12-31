/**
 * Admin Panel - Quotas Management Module
 *
 * Handles quota CRUD and user quota status lookup.
 */

window.adminQuotaMethods = {
    async loadQuotas() {
        this.quotasLoading = true;
        try {
            const response = await fetch('/api/quotas');
            if (response.ok) {
                const data = await response.json();
                this.quotas = data.quotas || [];
            }
        } catch (error) {
            console.error('Failed to load quotas:', error);
        } finally {
            this.quotasLoading = false;
        }
    },

    async lookupUserQuotas() {
        if (!this.quotaLookupUserId) {
            this.quotaLookupResult = null;
            return;
        }

        this.quotaLookupLoading = true;
        this.quotaLookupResult = null;

        try {
            const response = await fetch(`/api/quotas/user/${this.quotaLookupUserId}`);
            if (response.ok) {
                this.quotaLookupResult = await response.json();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to load user quotas', 'error');
            }
        } catch (error) {
            console.error('Failed to lookup user quotas:', error);
            if (window.showToast) window.showToast('Failed to load user quotas', 'error');
        } finally {
            this.quotaLookupLoading = false;
        }
    },

    getQuotaLookupUserName() {
        if (!this.quotaLookupUserId || !this.users) return '';
        const user = this.users.find(u => u.id == this.quotaLookupUserId);
        return user ? (user.display_name || user.username) : `User #${this.quotaLookupUserId}`;
    },

    formatQuotaValue(value, quotaType) {
        if (value == null) return '-';
        const type = (quotaType || '').toLowerCase();
        if (type.includes('cost')) {
            return '$' + value.toFixed(2);
        }
        return Math.round(value).toString();
    },

    showCreateQuotaForm() {
        this.editingQuota = null;
        this.quotaForm = {
            quota_type: 'COST_MONTHLY',
            target_type: 'global',
            target_id: null,
            limit_value: 0,
            action: 'WARN',
        };
        this.quotaFormOpen = true;
    },

    showEditQuotaForm(quota) {
        this.editingQuota = quota;
        this.quotaForm = {
            quota_type: quota.quota_type,
            target_type: quota.target_type,
            target_id: quota.target_id,
            limit_value: quota.limit_value,
            action: quota.action,
        };
        this.quotaFormOpen = true;
    },

    async saveQuota() {
        this.saving = true;
        try {
            let response;
            if (this.editingQuota) {
                response = await fetch(`/api/quotas/${this.editingQuota.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.quotaForm),
                });
            } else {
                response = await fetch('/api/quotas', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.quotaForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingQuota ? 'Quota updated' : 'Quota created', 'success');
                }
                this.quotaFormOpen = false;
                await this.loadQuotas();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save quota', 'error');
            }
        } catch (error) {
            console.error('Failed to save quota:', error);
            if (window.showToast) window.showToast('Failed to save quota', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteQuota(quota) {
        if (!confirm('Delete this quota?')) return;

        try {
            const response = await fetch(`/api/quotas/${quota.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Quota deleted', 'success');
                await this.loadQuotas();
            } else {
                if (window.showToast) window.showToast('Failed to delete quota', 'error');
            }
        } catch (error) {
            console.error('Failed to delete quota:', error);
            if (window.showToast) window.showToast('Failed to delete quota', 'error');
        }
    },

    getQuotaTypeName(quotaType) {
        const key = (quotaType || '').toLowerCase();
        const names = {
            'cost_monthly': 'Monthly Cost',
            'cost_daily': 'Daily Cost',
            'concurrent_jobs': 'Concurrent Jobs',
            'jobs_per_day': 'Jobs/Day',
            'jobs_per_hour': 'Jobs/Hour',
        };
        return names[key] || quotaType;
    },

    getQuotaActionClass(action) {
        return window.UIHelpers?.getQuotaActionBadgeClass(action) || 'bg-secondary';
    },
};
