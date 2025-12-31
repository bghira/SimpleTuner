/**
 * Cloud Dashboard - Provider Management Module
 *
 * Handles provider loading, configuration, API key validation, and token management.
 */

window.cloudProviderMethods = {
    async loadProviders() {
        this.providersLoading = true;
        try {
            const response = await fetch('/api/cloud/providers');
            if (response.ok) {
                const data = await response.json();
                this.providers = data.providers || [];
            }
        } catch (error) {
            console.error('Failed to load providers:', error);
        } finally {
            this.providersLoading = false;
            this.loadProviderConfig();
        }
    },

    async loadProviderConfig() {
        if (!this.activeProvider) return;
        try {
            const response = await fetch(`/api/cloud/providers/${this.activeProvider}/config`);
            if (response.ok) {
                const data = await response.json();
                this.providerConfig = data || {};
                this.webhookUrl = data.webhook_url || '';
                this.loadCostLimitStatus();
            }
        } catch (error) {
            console.error('Failed to load provider config:', error);
        }
    },

    async loadCostLimitStatus() {
        this.costLimit.loading = true;
        try {
            const response = await fetch('/api/cloud/cost-limit/status');
            if (response.ok) {
                this.costLimit.status = await response.json();
            }
        } catch (error) {
            console.error('Failed to load cost limit status:', error);
        } finally {
            this.costLimit.loading = false;
        }
    },

    async updateCostLimitSetting(field, value) {
        this.costLimit.saving = true;
        try {
            // Map UI field names to provider config field names
            const fieldMapping = {
                enabled: 'cost_limit_enabled',
                limit_amount: 'cost_limit_amount',
                period: 'cost_limit_period',
                action: 'cost_limit_action',
            };

            const configField = fieldMapping[field] || `cost_limit_${field}`;
            const updates = {};
            updates[configField] = value;

            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(updates),
            });

            if (response.ok) {
                await this.loadCostLimitStatus();
                if (window.showToast) {
                    window.showToast('Cost limit settings updated', 'success');
                }
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to update', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to update cost limit setting:', error);
            if (window.showToast) {
                window.showToast('Failed to update cost limit', 'error');
            }
        } finally {
            this.costLimit.saving = false;
        }
    },

    async loadVersions() {
        this.versionsLoading = true;
        this.versionsError = null;
        try {
            const response = await fetch('/api/cloud/providers/replicate/versions');
            if (response.ok) {
                const data = await response.json();
                this.versions = data.versions || [];
            } else {
                this.versionsError = 'Failed to load versions';
            }
        } catch (error) {
            this.versionsError = 'Network error loading versions';
            console.error('Failed to load versions:', error);
        } finally {
            this.versionsLoading = false;
        }
    },

    async saveVersionOverride(version) {
        try {
            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    version_override: version || null,
                }),
            });

            if (response.ok) {
                await this.loadProviderConfig();
                if (window.showToast) {
                    window.showToast(version ? 'Version override saved' : 'Using latest version', 'success');
                }
            } else {
                if (window.showToast) {
                    window.showToast('Failed to save version', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to save version:', error);
            if (window.showToast) {
                window.showToast('Failed to save version', 'error');
            }
        }
    },

    async validateApiKey() {
        if (this.apiKeyState.loading) return;

        const provider = this.getActiveProvider();
        if (!provider?.configured) {
            this.apiKeyState.valid = false;
            return;
        }

        this.apiKeyState.loading = true;
        this.apiKeyState.error = null;

        try {
            const response = await fetch('/api/cloud/providers/replicate/validate');
            if (response.ok) {
                const data = await response.json();
                this.apiKeyState.valid = data.valid;
                this.apiKeyState.userInfo = data.username || null;
            } else {
                this.apiKeyState.valid = false;
                this.apiKeyState.error = 'API key validation failed';
            }
        } catch (error) {
            this.apiKeyState.valid = false;
            this.apiKeyState.error = 'Network error';
            console.error('Failed to validate API key:', error);
        } finally {
            this.apiKeyState.loading = false;
        }
    },

    async saveReplicateToken() {
        if (!this.tokenInput.trim()) {
            this.tokenError = 'API token is required';
            return;
        }

        this.tokenSaving = true;
        this.tokenError = null;
        this.tokenSuccess = false;

        try {
            const response = await fetch('/api/cloud/providers/replicate/token', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_token: this.tokenInput.trim() }),
            });

            if (response.ok) {
                this.tokenSuccess = true;
                this.tokenInput = '';
                await this.loadProviders();
                await this.validateApiKey();
                await this.loadCurrentUser();
                if (window.showToast) {
                    window.showToast('Replicate API token saved successfully', 'success');
                }
            } else {
                const data = await response.json();
                this.tokenError = data.detail || 'Failed to save token';
            }
        } catch (error) {
            this.tokenError = 'Network error saving token';
            console.error('Failed to save token:', error);
        } finally {
            this.tokenSaving = false;
        }
    },

    async deleteReplicateToken() {
        if (!confirm('Are you sure you want to remove the Replicate API token? This will disconnect your account.')) {
            return;
        }

        try {
            const response = await fetch('/api/cloud/providers/replicate/token', {
                method: 'DELETE',
            });

            if (response.ok) {
                this.apiKeyState.valid = false;
                this.apiKeyState.userInfo = null;
                await this.loadProviders();
                if (window.showToast) {
                    window.showToast('API token removed', 'success');
                }
            } else {
                if (window.showToast) {
                    window.showToast('Failed to remove token', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to delete token:', error);
            if (window.showToast) {
                window.showToast('Failed to remove token', 'error');
            }
        }
    },

    async loadCurrentUser() {
        try {
            const response = await fetch('/api/users/me');
            if (response.ok) {
                const data = await response.json();
                this.currentUser = data.user;
                this.hasAdminAccess = data.user?.is_admin || false;
            }
        } catch (error) {
            console.warn('Failed to load current user:', error);
        }
    },

    async loadPublishingStatus() {
        this.publishingStatus.loading = true;
        try {
            const response = await fetch('/api/cloud/publishing-status');
            if (response.ok) {
                const data = await response.json();
                Object.assign(this.publishingStatus, data);
            }
        } catch (error) {
            console.error('Failed to load publishing status:', error);
        } finally {
            this.publishingStatus.loading = false;
        }
    },

    async loadAvailableConfigs() {
        try {
            const response = await fetch('/api/configs');
            if (response.ok) {
                const data = await response.json();
                this.availableConfigs = data.configs || [];
            }
        } catch (error) {
            console.error('Failed to load available configs:', error);
        }
    },

    async loadPendingApprovals() {
        if (this.pendingApprovals.loading) return;
        this.pendingApprovals.loading = true;
        try {
            const response = await fetch('/api/queue?status=blocked&limit=100');
            if (response.ok) {
                const data = await response.json();
                this.pendingApprovals.count = (data.entries || []).length;
                this.pendingApprovals.lastLoaded = new Date();
            }
        } catch (error) {
            console.warn('Failed to load pending approvals:', error);
        } finally {
            this.pendingApprovals.loading = false;
        }
    },
};
