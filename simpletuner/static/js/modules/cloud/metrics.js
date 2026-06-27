/**
 * Cloud Dashboard - Metrics Module
 *
 * Handles metrics loading, period selection, and billing refresh.
 */

window.cloudMetricsMethods = {
    async loadMetrics() {
        if (this.metricsLoading) return;
        this.metricsLoading = true;

        try {
            const params = new URLSearchParams();
            params.set('days', String(this.metricsPeriod));
            params.set('provider', this.activeProvider);

            const response = await fetch(`/api/metrics?${params}`);
            if (response.ok) {
                const data = await response.json();
                Object.assign(this.metrics, data);
            }
        } catch (error) {
            console.error('Failed to load metrics:', error);
        } finally {
            this.metricsLoading = false;
        }
    },

    setMetricsPeriod(days) {
        this.metricsPeriod = days;
        this.loadMetrics();
    },

    async refreshBilling() {
        if (this.billingState.loading) return;
        this.billingState.loading = true;
        this.billingState.error = null;
        try {
            const response = await fetch('/api/cloud/billing/refresh', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                if (data.credit_balance !== undefined) {
                    this.metrics.credit_balance = data.credit_balance;
                }
                this.billingState.fetched = true;
                if (window.showToast) {
                    window.showToast('Billing info refreshed', 'success');
                }
            } else {
                this.billingState.error = 'Failed to refresh billing';
            }
        } catch (error) {
            console.error('Failed to refresh billing:', error);
            this.billingState.error = 'Network error';
        } finally {
            this.billingState.loading = false;
        }
    },

    async saveWebhookConfig() {
        this.configSaving = true;
        const webhookUrl = typeof this.webhookUrl === 'string' ? this.webhookUrl.trim() : '';
        try {
            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ webhook_url: webhookUrl }),
            });

            let data = {};
            try {
                data = await response.json();
            } catch (_) {}

            if (!response.ok) {
                throw new Error(data.detail || 'Failed to save webhook config');
            }

            const savedUrl = (data.config && data.config.webhook_url) || data.webhook_url || webhookUrl || '';
            this.savedWebhookUrl = savedUrl;
            this.webhookUrl = savedUrl;
            if (this.publishingStatus) {
                this.publishingStatus.local_upload_available = savedUrl.length > 0;
                if (!savedUrl) {
                    this.publishingStatus.local_upload_dir = null;
                }
            }

            if (window.showToast) {
                window.showToast('Webhook configuration saved', 'success');
            }
            return true;
        } catch (error) {
            if (window.showToast) {
                window.showToast(error.message || 'Failed to save webhook config', 'error');
            }
            return false;
        } finally {
            this.configSaving = false;
        }
    },

    async testWebhook() {
        if (!this.webhookUrl) return;

        this.webhookTesting = true;
        this.webhookTestMode = 'internal';
        this.webhookTestResult = null;

        try {
            // Build the full webhook endpoint URL
            const fullUrl = this.webhookUrl.replace(/\/$/, '') + '/api/cloud/webhook/replicate';

            const response = await fetch('/api/cloud/webhook/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    webhook_url: fullUrl,
                    provider: this.activeProvider || 'replicate',
                    method: 'direct',  // Force internal/direct test
                }),
            });

            if (response.ok) {
                this.webhookTestResult = await response.json();
            } else {
                const data = await response.json();
                this.webhookTestResult = {
                    success: false,
                    error: data.detail || 'Test request failed',
                };
            }
        } catch (error) {
            this.webhookTestResult = {
                success: false,
                error: 'Network error: ' + error.message,
            };
        } finally {
            this.webhookTesting = false;
        }
    },

    async testWebhookExternal() {
        if (!this.webhookUrl) return;

        // Check if API key is configured before attempting external test
        if (!this.apiKeyState.valid) {
            this.webhookTestResult = {
                success: false,
                error: 'API key required for external test. Configure your Replicate API key first.',
            };
            return;
        }

        this.webhookTesting = true;
        this.webhookTestMode = 'external';
        this.webhookTestResult = null;

        try {
            // Build the full webhook endpoint URL
            const fullUrl = this.webhookUrl.replace(/\/$/, '') + '/api/cloud/webhook/replicate';

            const response = await fetch('/api/cloud/webhook/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    webhook_url: fullUrl,
                    provider: this.activeProvider || 'replicate',
                    method: 'replicate_cog',  // Force external test via Replicate infrastructure
                }),
            });

            if (response.ok) {
                this.webhookTestResult = await response.json();
            } else {
                const data = await response.json();
                this.webhookTestResult = {
                    success: false,
                    error: data.detail || 'External test request failed',
                };
            }
        } catch (error) {
            this.webhookTestResult = {
                success: false,
                error: 'Network error: ' + error.message,
            };
        } finally {
            this.webhookTesting = false;
        }
    },
};
