/**
 * External Authentication Provider Configuration Component
 *
 * Manages OIDC and LDAP provider configuration.
 */

window.externalAuthComponent = function() {
    return {
        // State
        loading: false,
        saving: false,
        deleting: false,
        providers: [],
        testingProvider: null,

        // Modal state
        editingProvider: null,
        deletingProvider: null,
        providerModal: null,
        deleteModal: null,

        // Form state
        providerForm: {
            name: '',
            provider_type: 'oidc',
            enabled: true,
            auto_create_users: true,
            default_levels: ['researcher'],
            level_mapping: {},
            config: {},
        },

        // Level mapping form state
        newMappingGroup: '',
        newMappingLevel: 'researcher',

        init() {
            this.loadProviders();

            this.$nextTick(() => {
                if (typeof bootstrap !== 'undefined') {
                    const modalEl = document.getElementById('externalAuthProviderModal');
                    if (modalEl) {
                        this.providerModal = new bootstrap.Modal(modalEl);
                    }
                    const deleteEl = document.getElementById('deleteProviderModal');
                    if (deleteEl) {
                        this.deleteModal = new bootstrap.Modal(deleteEl);
                    }
                }
            });
        },

        async loadProviders() {
            this.loading = true;
            try {
                const response = await fetch('/api/cloud/external-auth/providers');
                if (response.ok) {
                    const data = await response.json();
                    this.providers = data.providers || [];
                }
            } catch (error) {
                console.error('Failed to load providers:', error);
            } finally {
                this.loading = false;
            }
        },

        showProviderModal(defaultType) {
            this.editingProvider = null;
            this.resetProviderForm();
            if (defaultType) {
                this.providerForm.provider_type = defaultType;
            }
            if (this.providerModal) {
                this.providerModal.show();
            }
        },

        editProvider(provider) {
            this.editingProvider = provider;
            this.providerForm = {
                name: provider.name,
                provider_type: provider.type,
                enabled: provider.enabled,
                auto_create_users: provider.auto_create_users,
                default_levels: provider.default_levels || ['researcher'],
                level_mapping: provider.level_mapping || {},
                config: provider.config || {},
            };
            if (this.providerModal) {
                this.providerModal.show();
            }
        },

        resetProviderForm() {
            this.providerForm = {
                name: '',
                provider_type: 'oidc',
                enabled: true,
                auto_create_users: true,
                default_levels: ['researcher'],
                level_mapping: {},
                config: {},
            };
        },

        async saveProvider() {
            this.saving = true;
            try {
                const payload = {
                    name: this.providerForm.name,
                    provider_type: this.providerForm.provider_type,
                    enabled: this.providerForm.enabled,
                    auto_create_users: this.providerForm.auto_create_users,
                    default_levels: this.providerForm.default_levels,
                    level_mapping: this.providerForm.level_mapping,
                    config: this.providerForm.config,
                };

                let response;
                if (this.editingProvider) {
                    response = await fetch(`/api/cloud/external-auth/providers/${this.editingProvider.name}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                } else {
                    response = await fetch('/api/cloud/external-auth/providers', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                }

                if (response.ok) {
                    if (this.providerModal) {
                        this.providerModal.hide();
                    }
                    await this.loadProviders();
                    if (window.showToast) {
                        window.showToast(
                            this.editingProvider ? 'Provider updated' : 'Provider created',
                            'success'
                        );
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to save provider');
                }
            } catch (error) {
                console.error('Failed to save provider:', error);
                if (window.showToast) {
                    window.showToast(error.message || 'Failed to save provider', 'error');
                }
            } finally {
                this.saving = false;
            }
        },

        confirmDeleteProvider(provider) {
            this.deletingProvider = provider;
            if (this.deleteModal) {
                this.deleteModal.show();
            }
        },

        async deleteProvider() {
            if (!this.deletingProvider) return;

            this.deleting = true;
            try {
                const response = await fetch(`/api/cloud/external-auth/providers/${this.deletingProvider.name}`, {
                    method: 'DELETE',
                });

                if (response.ok) {
                    if (this.deleteModal) {
                        this.deleteModal.hide();
                    }
                    await this.loadProviders();
                    if (window.showToast) {
                        window.showToast('Provider deleted', 'success');
                    }
                } else {
                    throw new Error('Failed to delete provider');
                }
            } catch (error) {
                console.error('Failed to delete provider:', error);
                if (window.showToast) {
                    window.showToast('Failed to delete provider', 'error');
                }
            } finally {
                this.deleting = false;
                this.deletingProvider = null;
            }
        },

        async toggleProvider(provider) {
            try {
                const response = await fetch(`/api/cloud/external-auth/providers/${provider.name}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: provider.name,
                        provider_type: provider.type,
                        enabled: !provider.enabled,
                        auto_create_users: provider.auto_create_users,
                        default_levels: provider.default_levels || ['researcher'],
                        level_mapping: provider.level_mapping || {},
                        config: provider.config || {},
                    }),
                });

                if (response.ok) {
                    await this.loadProviders();
                    if (window.showToast) {
                        window.showToast(
                            provider.enabled ? 'Provider disabled' : 'Provider enabled',
                            'success'
                        );
                    }
                }
            } catch (error) {
                console.error('Failed to toggle provider:', error);
            }
        },

        async testProvider(providerName) {
            this.testingProvider = providerName;
            try {
                const response = await fetch(`/api/cloud/external-auth/providers/${providerName}/test`, {
                    method: 'POST',
                });

                const data = await response.json();
                if (data.success) {
                    if (window.showToast) {
                        window.showToast(`Connection successful${data.latency_ms ? ' (' + data.latency_ms.toFixed(0) + 'ms)' : ''}`, 'success');
                    }
                } else {
                    if (window.showToast) {
                        window.showToast(`Connection failed: ${data.error || 'Unknown error'}`, 'error');
                    }
                }
            } catch (error) {
                console.error('Failed to test provider:', error);
                if (window.showToast) {
                    window.showToast('Failed to test connection', 'error');
                }
            } finally {
                this.testingProvider = null;
            }
        },

        addLevelMapping() {
            const group = this.newMappingGroup.trim();
            if (!group) return;
            if (!this.providerForm.level_mapping) {
                this.providerForm.level_mapping = {};
            }
            this.providerForm.level_mapping[group] = this.newMappingLevel;
            this.newMappingGroup = '';
            this.newMappingLevel = 'researcher';
        },

        removeLevelMapping(groupDn) {
            if (this.providerForm.level_mapping && this.providerForm.level_mapping[groupDn]) {
                delete this.providerForm.level_mapping[groupDn];
            }
        },
    };
};
