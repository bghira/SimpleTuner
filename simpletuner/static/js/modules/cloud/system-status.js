/**
 * Cloud Dashboard - System Status Module
 *
 * Handles system status, health checks, and Prometheus metrics.
 */

window.cloudSystemMethods = {
    async loadSystemStatus() {
        this.systemStatus.loading = true;
        try {
            const response = await fetch('/api/cloud/system-status');
            if (response.ok) {
                const data = await response.json();
                this.systemStatus.operational = data.operational;
                this.systemStatus.ongoing_incidents = data.ongoing_incidents || [];
                this.systemStatus.in_progress_maintenances = data.in_progress_maintenances || [];
                this.systemStatus.scheduled_maintenances = data.scheduled_maintenances || [];
                this.systemStatus.status_page_url = data.status_page_url || null;
                this.systemStatus.error = null;
            } else {
                this.systemStatus.error = 'Failed to load status';
            }
        } catch (error) {
            this.systemStatus.error = 'Network error';
            console.error('Failed to load system status:', error);
        } finally {
            this.systemStatus.loading = false;
        }
    },

    async loadHealthCheck(deep = false) {
        this.healthCheck.loading = true;
        this.healthCheck.error = null;
        try {
            const url = deep ? '/api/cloud/health?deep=true' : '/api/cloud/health';
            const response = await fetch(url);
            if (response.ok) {
                const data = await response.json();
                this.healthCheck.status = data.status;
                this.healthCheck.uptime_seconds = data.uptime_seconds;
                this.healthCheck.components = data.components || [];
                this.healthCheck.timestamp = data.timestamp;
                this.healthCheck.lastChecked = new Date();
            } else {
                this.healthCheck.error = 'Health check failed';
                this.healthCheck.status = 'unhealthy';
            }
        } catch (error) {
            this.healthCheck.error = 'Network error';
            this.healthCheck.status = 'unreachable';
            console.error('Failed to load health check:', error);
        } finally {
            this.healthCheck.loading = false;
        }
    },

    async runDeepHealthCheck() {
        await this.loadHealthCheck(true);
        if (this.healthCheck.status === 'healthy' && window.showToast) {
            window.showToast('All systems healthy', 'success');
        } else if (this.healthCheck.status !== 'healthy' && window.showToast) {
            window.showToast('Some components may be unhealthy', 'warning');
        }
    },

    startHealthCheckAutoRefresh() {
        this.stopHealthCheckAutoRefresh();
        // Refresh health check every 60 seconds
        this.healthCheck.autoRefreshInterval = setInterval(() => {
            this.loadHealthCheck();
        }, 60000);
    },

    stopHealthCheckAutoRefresh() {
        if (this.healthCheck.autoRefreshInterval) {
            clearInterval(this.healthCheck.autoRefreshInterval);
            this.healthCheck.autoRefreshInterval = null;
        }
    },

    async loadPrometheusMetrics() {
        this.prometheusMetrics.loading = true;
        this.prometheusMetrics.error = null;
        try {
            const response = await fetch('/api/metrics/prometheus');
            if (response.ok) {
                const text = await response.text();
                this.prometheusMetrics.raw = text;
                this.parsePrometheusMetrics(text);
            } else {
                this.prometheusMetrics.error = 'Failed to load metrics';
            }
        } catch (error) {
            this.prometheusMetrics.error = 'Network error';
            console.error('Failed to load Prometheus metrics:', error);
        } finally {
            this.prometheusMetrics.loading = false;
        }
    },

    parsePrometheusMetrics(text) {
        const lines = text.split('\n');
        const parsed = this.prometheusMetrics.parsed;

        for (const line of lines) {
            if (line.startsWith('#') || !line.trim()) continue;

            const match = line.match(/^(\w+)(?:\{[^}]*\})?\s+(.+)$/);
            if (!match) continue;

            const [, name, value] = match;
            const numValue = parseFloat(value);

            if (name === 'simpletuner_uptime_seconds') {
                parsed.uptime = numValue;
            } else if (name === 'simpletuner_cloud_jobs_total') {
                parsed.totalJobs = numValue;
            } else if (name === 'simpletuner_cloud_cost_total_usd') {
                parsed.totalCost = numValue;
            } else if (name === 'simpletuner_cloud_jobs_active') {
                parsed.activeJobs = numValue;
            }
        }
    },

    formatUptime(seconds) {
        if (!seconds) return '--';
        const days = Math.floor(seconds / 86400);
        const hours = Math.floor((seconds % 86400) / 3600);
        const mins = Math.floor((seconds % 3600) / 60);

        if (days > 0) {
            return `${days}d ${hours}h`;
        }
        if (hours > 0) {
            return `${hours}h ${mins}m`;
        }
        return `${mins}m`;
    },
};
