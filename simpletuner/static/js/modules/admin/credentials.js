/**
 * Admin Panel - Credential Management Module
 *
 * Handles user credential viewing, rotation, and stale checks.
 */

window.adminCredentialMethods = {
    async openCredentialsModal(user) {
        this.credentialsUser = user;
        this.credentialsModalOpen = true;
        this.credentialsLoading = true;
        this.userCredentials = [];
        await this.loadUserCredentials(user.id);
    },

    closeCredentialsModal() {
        this.credentialsModalOpen = false;
        this.credentialsUser = null;
        this.userCredentials = [];
    },

    async loadUserCredentials(userId) {
        this.credentialsLoading = true;
        try {
            const response = await fetch(`/api/users/${userId}/credentials`);
            if (response.ok) {
                const data = await response.json();
                this.userCredentials = data.credentials || [];
            }
        } catch (error) {
            console.error('Failed to load user credentials:', error);
            if (window.showToast) window.showToast('Failed to load credentials', 'error');
        } finally {
            this.credentialsLoading = false;
        }
    },

    confirmRotateCredential(credential) {
        this.rotatingCredential = credential;
        this.rotateConfirmOpen = true;
    },

    cancelRotate() {
        this.rotateConfirmOpen = false;
        this.rotatingCredential = null;
    },

    async rotateCredential() {
        if (!this.rotatingCredential || !this.credentialsUser) return;
        this.rotating = true;
        try {
            const response = await fetch(`/api/users/${this.credentialsUser.id}/credentials/rotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: this.rotatingCredential.provider,
                    reason: 'Manual rotation from admin panel',
                }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Credential rotated', 'success');
                this.rotateConfirmOpen = false;
                this.rotatingCredential = null;
                await this.loadUserCredentials(this.credentialsUser.id);
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to rotate', 'error');
            }
        } catch (error) {
            console.error('Failed to rotate credential:', error);
            if (window.showToast) window.showToast('Failed to rotate credential', 'error');
        } finally {
            this.rotating = false;
        }
    },

    async rotateAllCredentials() {
        if (!this.credentialsUser) return;
        if (!confirm('Rotate ALL credentials for this user? This will invalidate all current tokens.')) return;

        this.rotating = true;
        try {
            const response = await fetch(`/api/users/${this.credentialsUser.id}/credentials/rotate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    reason: 'Bulk rotation from admin panel',
                }),
            });
            if (response.ok) {
                const data = await response.json();
                if (window.showToast) window.showToast(`Rotated ${data.rotated_count} credential(s)`, 'success');
                await this.loadUserCredentials(this.credentialsUser.id);
            } else {
                if (window.showToast) window.showToast('Failed to rotate credentials', 'error');
            }
        } catch (error) {
            console.error('Failed to rotate all credentials:', error);
            if (window.showToast) window.showToast('Failed to rotate credentials', 'error');
        } finally {
            this.rotating = false;
        }
    },

    async checkAllUsersStaleCredentials() {
        this.bulkStaleCheckLoading = true;
        try {
            const response = await fetch('/api/users/credentials/check-stale', {
                method: 'POST',
            });
            if (response.ok) {
                const data = await response.json();
                // Update user list with stale info
                for (const user of this.users) {
                    const staleInfo = data.users_with_stale?.find(u => u.user_id === user.id);
                    if (staleInfo) {
                        user.has_stale_credentials = true;
                        user.stale_credential_count = staleInfo.stale_count;
                    } else {
                        user.has_stale_credentials = false;
                        user.stale_credential_count = 0;
                    }
                }
                if (window.showToast) {
                    window.showToast(`Found ${data.total_stale || 0} stale credential(s)`, data.total_stale > 0 ? 'warning' : 'success');
                }
            }
        } catch (error) {
            console.error('Failed to check stale credentials:', error);
            if (window.showToast) window.showToast('Failed to check credentials', 'error');
        } finally {
            this.bulkStaleCheckLoading = false;
        }
    },

    getCredentialAge(credential) {
        if (!credential.updated_at) return 'Unknown';
        const updated = new Date(credential.updated_at);
        const now = new Date();
        const days = Math.floor((now - updated) / (1000 * 60 * 60 * 24));
        if (days === 0) return 'Today';
        if (days === 1) return 'Yesterday';
        if (days < 30) return `${days} days ago`;
        if (days < 365) return `${Math.floor(days / 30)} month(s) ago`;
        return `${Math.floor(days / 365)} year(s) ago`;
    },

    isCredentialApproachingExpiry(credential) {
        if (!credential.updated_at || !this.staleCredentialsThreshold) return false;
        const updated = new Date(credential.updated_at);
        const now = new Date();
        const days = Math.floor((now - updated) / (1000 * 60 * 60 * 24));
        const warningThreshold = this.staleCredentialsThreshold * (this.credentialEarlyWarningPercent / 100);
        return days >= warningThreshold && days < this.staleCredentialsThreshold;
    },

    // Credential security settings
    async loadCredentialSettings() {
        try {
            const response = await fetch('/api/cloud/settings/credentials');
            if (response.ok) {
                const data = await response.json();
                this.credentialSettings = {
                    stale_threshold_days: data.stale_threshold_days || 90,
                    early_warning_enabled: data.early_warning_enabled || false,
                    early_warning_percent: data.early_warning_percent || 75,
                };
            }
        } catch (error) {
            console.error('Failed to load credential settings:', error);
        }
    },

    async saveCredentialSettings() {
        this.savingCredentialSettings = true;
        try {
            const response = await fetch('/api/cloud/settings/credentials', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.credentialSettings),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Credential settings saved', 'success');
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save settings', 'error');
            }
        } catch (error) {
            console.error('Failed to save credential settings:', error);
            if (window.showToast) window.showToast('Failed to save settings', 'error');
        } finally {
            this.savingCredentialSettings = false;
        }
    },
};
