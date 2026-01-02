/**
 * Admin Panel - External Auth Providers Module
 *
 * Handles OIDC and LDAP provider configuration.
 */

window.adminAuthProviderMethods = {
    async loadAuthProviders() {
        this.authProvidersLoading = true;
        try {
            const response = await fetch('/api/auth/external/providers');
            if (response.ok) {
                const data = await response.json();
                this.authProviders = data.providers || [];
            }
        } catch (error) {
            console.error('Failed to load auth providers:', error);
        } finally {
            this.authProvidersLoading = false;
        }
    },

    showCreateAuthProviderForm(type = 'oidc') {
        this.editingAuthProvider = null;
        this.authProviderForm = {
            provider_type: type,
            name: '',
            enabled: true,
            config: type === 'oidc' ? {
                issuer_url: '',
                client_id: '',
                client_secret: '',
                scopes: 'openid email profile',
                auto_provision: true,
                default_level: 'researcher',
            } : {
                server_url: '',
                bind_dn: '',
                bind_password: '',
                base_dn: '',
                user_filter: '(uid={username})',
                auto_provision: true,
                default_level: 'researcher',
            },
        };
        this.authProviderFormOpen = true;
    },

    showEditAuthProviderForm(provider) {
        this.editingAuthProvider = provider;
        this.authProviderForm = {
            provider_type: provider.provider_type,
            name: provider.name,
            enabled: provider.enabled,
            config: { ...(provider.config || {}) },
        };
        this.authProviderFormOpen = true;
    },

    async saveAuthProvider() {
        this.saving = true;
        try {
            let response;
            if (this.editingAuthProvider) {
                response = await fetch(`/api/auth/external/providers/${this.editingAuthProvider.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.authProviderForm),
                });
            } else {
                response = await fetch('/api/auth/external/providers', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.authProviderForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingAuthProvider ? 'Provider updated' : 'Provider created', 'success');
                }
                this.authProviderFormOpen = false;
                await this.loadAuthProviders();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save provider', 'error');
            }
        } catch (error) {
            console.error('Failed to save auth provider:', error);
            if (window.showToast) window.showToast('Failed to save provider', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteAuthProvider(provider) {
        if (!confirm(`Delete auth provider "${provider.name}"? Users who logged in via this provider will not be able to log in again.`)) return;

        try {
            const response = await fetch(`/api/auth/external/providers/${provider.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Provider deleted', 'success');
                await this.loadAuthProviders();
            } else {
                if (window.showToast) window.showToast('Failed to delete provider', 'error');
            }
        } catch (error) {
            console.error('Failed to delete auth provider:', error);
            if (window.showToast) window.showToast('Failed to delete provider', 'error');
        }
    },

    async testAuthProvider(provider) {
        try {
            const response = await fetch(`/api/auth/external/providers/${provider.name}/test`, {
                method: 'POST',
            });
            if (response.ok) {
                const data = await response.json();
                if (data.success) {
                    if (window.showToast) window.showToast(`Connection successful (${data.latency_ms}ms)`, 'success');
                } else {
                    if (window.showToast) window.showToast(`Connection failed: ${data.error}`, 'error');
                }
            } else {
                if (window.showToast) window.showToast('Test request failed', 'error');
            }
        } catch (error) {
            console.error('Failed to test auth provider:', error);
            if (window.showToast) window.showToast('Test failed', 'error');
        }
    },

    getAuthProviderIcon(type) {
        return type === 'oidc' ? 'fa-openid' : 'fa-sitemap';
    },
};
