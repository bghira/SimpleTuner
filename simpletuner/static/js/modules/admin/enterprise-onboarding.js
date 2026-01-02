/**
 * Admin Panel - Enterprise Onboarding Module
 *
 * Handles progressive disclosure CTA for enterprise features setup.
 */

window.adminEnterpriseOnboardingMethods = {
    loadEnterpriseOnboardingState() {
        const stored = localStorage.getItem('admin_enterprise_onboarding');
        if (stored) {
            try {
                Object.assign(this.enterpriseOnboarding, JSON.parse(stored));
            } catch (e) {
                console.warn('Failed to parse enterprise onboarding state');
            }
        }
    },

    saveEnterpriseOnboardingState() {
        localStorage.setItem('admin_enterprise_onboarding', JSON.stringify(this.enterpriseOnboarding));
    },

    shouldShowEnterpriseOnboarding() {
        if (this.enterpriseOnboarding.all_skipped) return false;
        // Show if any step is incomplete and not skipped
        return !this.enterpriseOnboarding.auth_configured && !this.enterpriseOnboarding.auth_skipped;
    },

    startAuthProviderSetup(type) {
        // Set the provider type to show the inline form
        this.quickAuthForm.provider_type = type;
    },

    skipEnterpriseStep(step) {
        this.enterpriseOnboarding[`${step}_skipped`] = true;
        this.saveEnterpriseOnboardingState();
    },

    skipEnterpriseOnboarding() {
        this.enterpriseOnboarding.all_skipped = true;
        this.saveEnterpriseOnboardingState();
    },

    slugify(text) {
        return text.toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/[\s_-]+/g, '-')
            .replace(/^-+|-+$/g, '');
    },

    async createQuickOrg() {
        if (!this.quickOrgForm.name.trim()) {
            if (window.showToast) window.showToast('Organization name is required', 'error');
            return;
        }

        this.quickOrgForm.saving = true;
        try {
            const payload = {
                name: this.quickOrgForm.name,
                slug: this.quickOrgForm.slug || this.slugify(this.quickOrgForm.name),
                description: '',
                is_active: true,
            };

            const response = await fetch('/api/orgs', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                const data = await response.json();
                this.enterpriseOnboarding.org_created = true;
                this.enterpriseOnboarding.created_org_id = data.id;
                this.enterpriseOnboarding.created_org_name = data.name;
                this.saveEnterpriseOnboardingState();
                await this.loadOrganizations();
                if (window.showToast) window.showToast('Organization created', 'success');
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to create organization', 'error');
            }
        } catch (error) {
            console.error('Failed to create quick org:', error);
            if (window.showToast) window.showToast('Failed to create organization', 'error');
        } finally {
            this.quickOrgForm.saving = false;
        }
    },

    async createQuickTeam() {
        if (!this.quickTeamForm.name.trim()) {
            if (window.showToast) window.showToast('Team name is required', 'error');
            return;
        }

        if (!this.enterpriseOnboarding.created_org_id) {
            if (window.showToast) window.showToast('Create an organization first', 'error');
            return;
        }

        this.quickTeamForm.saving = true;
        try {
            const payload = {
                name: this.quickTeamForm.name,
                slug: this.quickTeamForm.slug || this.slugify(this.quickTeamForm.name),
                description: '',
                is_active: true,
            };

            const response = await fetch(`/api/orgs/${this.enterpriseOnboarding.created_org_id}/teams`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                this.enterpriseOnboarding.team_created = true;
                this.saveEnterpriseOnboardingState();
                if (window.showToast) window.showToast('Team created', 'success');
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to create team', 'error');
            }
        } catch (error) {
            console.error('Failed to create quick team:', error);
            if (window.showToast) window.showToast('Failed to create team', 'error');
        } finally {
            this.quickTeamForm.saving = false;
        }
    },

    async saveQuickQuotas() {
        this.quickQuotaForm.saving = true;
        try {
            const quotas = [];
            if (this.quickQuotaForm.monthly_cost_limit > 0) {
                quotas.push({
                    quota_type: 'COST_MONTHLY',
                    target_type: 'global',
                    limit_value: this.quickQuotaForm.monthly_cost_limit,
                    action: 'REQUIRE_APPROVAL',
                });
            }
            if (this.quickQuotaForm.concurrent_jobs > 0) {
                quotas.push({
                    quota_type: 'CONCURRENT_JOBS',
                    target_type: 'global',
                    limit_value: this.quickQuotaForm.concurrent_jobs,
                    action: 'BLOCK',
                });
            }

            let allSucceeded = true;
            for (const quota of quotas) {
                const response = await fetch('/api/quotas', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(quota),
                });
                if (!response.ok) {
                    allSucceeded = false;
                    const data = await response.json().catch(() => ({}));
                    console.error('Failed to create quota:', quota.quota_type, data.detail || response.status);
                }
            }

            if (!allSucceeded) {
                if (window.showToast) window.showToast('Some quotas failed to save', 'warning');
            }

            this.enterpriseOnboarding.quotas_configured = true;
            this.saveEnterpriseOnboardingState();
            await this.loadQuotas();
            if (window.showToast) window.showToast('Quotas configured', 'success');
        } catch (error) {
            console.error('Failed to save quick quotas:', error);
            if (window.showToast) window.showToast('Failed to save quotas', 'error');
        } finally {
            this.quickQuotaForm.saving = false;
        }
    },

    markAuthProviderConfigured() {
        this.enterpriseOnboarding.auth_configured = true;
        this.saveEnterpriseOnboardingState();
    },

    async saveQuickCredentialSettings() {
        this.quickCredentialForm.saving = true;
        try {
            const response = await fetch('/api/cloud/settings/credentials', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    stale_threshold_days: this.quickCredentialForm.stale_days,
                    early_warning_enabled: this.quickCredentialForm.early_warning,
                    early_warning_percent: this.quickCredentialForm.warning_percent,
                }),
            });

            if (response.ok) {
                this.enterpriseOnboarding.credentials_configured = true;
                this.saveEnterpriseOnboardingState();
                if (window.showToast) window.showToast('Credential settings saved', 'success');
            } else {
                if (window.showToast) window.showToast('Failed to save settings', 'error');
            }
        } catch (error) {
            console.error('Failed to save credential settings:', error);
            if (window.showToast) window.showToast('Failed to save settings', 'error');
        } finally {
            this.quickCredentialForm.saving = false;
        }
    },

    async saveCredentialSecuritySkipped() {
        this.enterpriseOnboarding.credentials_skipped = true;
        this.saveEnterpriseOnboardingState();
    },

    async loadCredentialSecuritySettings() {
        try {
            const response = await fetch('/api/cloud/settings/credentials');
            if (response.ok) {
                const data = await response.json();
                this.staleCredentialsThreshold = data.stale_threshold_days || 90;
                this.credentialEarlyWarningEnabled = data.early_warning_enabled || false;
                this.credentialEarlyWarningPercent = data.early_warning_percent || 75;
                // Also populate credentialSettings for Auth tab UI
                this.credentialSettings = {
                    stale_threshold_days: data.stale_threshold_days || 90,
                    early_warning_enabled: data.early_warning_enabled || false,
                    early_warning_percent: data.early_warning_percent || 75,
                };
            }
        } catch (error) {
            console.warn('Failed to load credential settings:', error);
        }
    },

    getEarlyWarningDays() {
        return Math.floor(this.staleCredentialsThreshold * (this.credentialEarlyWarningPercent / 100));
    },

    async saveQuickAuthProvider() {
        this.quickAuthForm.saving = true;
        this.quickAuthForm.error = null;

        try {
            const providerType = this.quickAuthForm.provider_type;
            const config = providerType === 'oidc' ? {
                issuer_url: this.quickAuthForm.oidc.issuer_url,
                client_id: this.quickAuthForm.oidc.client_id,
                client_secret: this.quickAuthForm.oidc.client_secret,
                scopes: this.quickAuthForm.oidc.scopes,
                auto_provision: this.quickAuthForm.oidc.auto_provision,
                default_level: this.quickAuthForm.oidc.default_level,
            } : {
                server_url: this.quickAuthForm.ldap.server_url,
                bind_dn: this.quickAuthForm.ldap.bind_dn,
                bind_password: this.quickAuthForm.ldap.bind_password,
                base_dn: this.quickAuthForm.ldap.base_dn,
                user_filter: this.quickAuthForm.ldap.user_filter,
                auto_provision: this.quickAuthForm.ldap.auto_provision,
                default_level: this.quickAuthForm.ldap.default_level,
            };

            const response = await fetch('/api/auth/external/providers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider_type: providerType,
                    name: this.quickAuthForm.name,
                    enabled: true,
                    config: config,
                }),
            });

            if (response.ok) {
                this.enterpriseOnboarding.auth_configured = true;
                this.saveEnterpriseOnboardingState();
                await this.loadAuthProviders();
                if (window.showToast) {
                    window.showToast(`${providerType.toUpperCase()} provider connected`, 'success');
                }
                // Reset form
                this.quickAuthForm.provider_type = '';
                this.quickAuthForm.name = '';
            } else {
                const data = await response.json().catch(() => ({}));
                this.quickAuthForm.error = window.UIHelpers?.extractErrorMessage(data, 'Failed to create provider') || data.detail || 'Failed to create provider';
                if (window.showToast) {
                    window.showToast(this.quickAuthForm.error, 'error');
                }
            }
        } catch (error) {
            console.error('Failed to save auth provider:', error);
            this.quickAuthForm.error = 'Connection error';
            if (window.showToast) {
                window.showToast('Failed to connect provider', 'error');
            }
        } finally {
            this.quickAuthForm.saving = false;
        }
    },
};
