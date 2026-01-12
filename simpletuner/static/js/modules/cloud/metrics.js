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
        try {
            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ webhook_url: this.webhookUrl || null }),
            });
            if (response.ok && window.showToast) {
                window.showToast('Webhook configuration saved', 'success');
            }
        } catch (error) {
            if (window.showToast) {
                window.showToast('Failed to save webhook config', 'error');
            }
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
