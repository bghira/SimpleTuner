/**
 * Admin Panel - Registration Settings Module
 *
 * Handles public registration settings (enable/disable, default level).
 */

window.adminRegistrationMethods = {
    async loadRegistrationSettings() {
        this.registrationLoading = true;
        try {
            const response = await fetch('/api/cloud/settings/registration');
            if (response.ok) {
                const data = await response.json();
                this.registrationSettings = {
                    enabled: data.enabled || false,
                    default_level: data.default_level || 'researcher',
                    email_configured: data.email_configured || false,
                    email_required_warning: data.email_required_warning || null,
                };
            }
        } catch (error) {
            console.error('Failed to load registration settings:', error);
        } finally {
            this.registrationLoading = false;
        }
    },

    async saveRegistrationSettings() {
        try {
            const response = await fetch('/api/cloud/settings/registration', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    enabled: this.registrationSettings.enabled,
                    default_level: this.registrationSettings.default_level,
                }),
            });
            if (response.ok) {
                const data = await response.json();
                // Update with any warnings from server
                this.registrationSettings.email_configured = data.email_configured;
                this.registrationSettings.email_required_warning = data.email_required_warning;

                const status = this.registrationSettings.enabled ? 'enabled' : 'disabled';
                if (window.showToast) window.showToast(`Public registration ${status}`, 'success');
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save settings', 'error');
            }
        } catch (error) {
            console.error('Failed to save registration settings:', error);
            if (window.showToast) window.showToast('Failed to save settings', 'error');
        }
    },
};
