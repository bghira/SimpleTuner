/**
 * Metrics configuration Alpine.js component
 *
 * Handles Prometheus metrics export configuration and preview.
 */

function metricsComponent(initialSettings = {}) {
    return {
        // Configuration state
        prometheusEnabled: initialSettings.prometheus_enabled || false,
        selectedCategories: initialSettings.prometheus_categories || ['jobs', 'http'],
        tensorboardEnabled: initialSettings.tensorboard_enabled || false,
        dismissedHints: initialSettings.dismissed_hints || [],

        // UI state
        heroDismissed: false,
        loadingPreview: false,
        previewContent: '',
        previewError: null,
        previewMetricCount: 0,
        expandedCategory: null,
        testingScrape: false,
        scrapeResult: null,

        // Data from API
        categories: [],
        templates: [],

        // Circuit breaker state
        circuitBreakers: [],
        circuitBreakersLoading: false,

        // Initialization
        async init() {
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }

            // Check if hero should be shown
            this.heroDismissed = this.dismissedHints.includes('hero');

            // Load categories, templates, and circuit breaker status
            await Promise.all([
                this.loadCategoriesAndTemplates(),
                this.loadCircuitBreakers(),
            ]);

            // Load initial preview if enabled
            if (this.prometheusEnabled) {
                await this.refreshPreview();
            }
        },

        async loadCircuitBreakers() {
            this.circuitBreakersLoading = true;
            try {
                const response = await fetch('/api/metrics/circuit-breakers');
                if (response.ok) {
                    const data = await response.json();
                    this.circuitBreakers = data.circuit_breakers || [];
                }
            } catch (error) {
                console.error('Failed to load circuit breakers:', error);
            } finally {
                this.circuitBreakersLoading = false;
            }
        },

        getCircuitBreakerIcon(state) {
            switch (state) {
                case 'closed': return 'fas fa-check-circle text-success';
                case 'half_open': return 'fas fa-exclamation-circle text-warning';
                case 'open': return 'fas fa-times-circle text-danger';
                default: return 'fas fa-question-circle text-muted';
            }
        },

        getCircuitBreakerLabel(state) {
            switch (state) {
                case 'closed': return 'Healthy';
                case 'half_open': return 'Recovering';
                case 'open': return 'Open';
                default: return 'Unknown';
            }
        },

        async loadCategoriesAndTemplates() {
            try {
                const [categoriesRes, templatesRes] = await Promise.all([
                    fetch('/api/metrics/config/categories'),
                    fetch('/api/metrics/config/templates'),
                ]);

                if (categoriesRes.ok) {
                    const data = await categoriesRes.json();
                    this.categories = data.categories || [];
                }

                if (templatesRes.ok) {
                    const data = await templatesRes.json();
                    this.templates = data.templates || [];
                }
            } catch (error) {
                console.error('Failed to load metrics metadata:', error);
            }
        },

        async saveConfig() {
            try {
                const response = await fetch('/api/metrics/config', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prometheus_enabled: this.prometheusEnabled,
                        prometheus_categories: this.selectedCategories,
                    }),
                });

                if (!response.ok) {
                    throw new Error('Failed to save configuration');
                }

                // Refresh preview after save
                if (this.prometheusEnabled) {
                    await this.refreshPreview();
                }
            } catch (error) {
                console.error('Failed to save metrics config:', error);
            }
        },

        async applyTemplate(templateId) {
            try {
                const response = await fetch(`/api/metrics/config/apply-template/${templateId}`, {
                    method: 'POST',
                });

                if (!response.ok) {
                    throw new Error('Failed to apply template');
                }

                const data = await response.json();
                this.prometheusEnabled = data.prometheus_enabled;
                this.selectedCategories = data.prometheus_categories;

                // Dismiss hero after applying template
                if (!this.heroDismissed) {
                    await this.dismissHero();
                }

                // Refresh preview
                await this.refreshPreview();
            } catch (error) {
                console.error('Failed to apply template:', error);
            }
        },

        isTemplateActive(templateId) {
            const template = this.templates.find(t => t.id === templateId);
            if (!template) return false;

            // Check if current selection matches template
            if (template.categories.length !== this.selectedCategories.length) return false;
            return template.categories.every(c => this.selectedCategories.includes(c));
        },

        async refreshPreview() {
            this.loadingPreview = true;
            this.previewError = null;

            try {
                const response = await fetch('/api/metrics/config/preview', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.selectedCategories),
                });

                if (!response.ok) {
                    throw new Error('Failed to fetch preview');
                }

                const data = await response.json();
                this.previewContent = data.content;
                this.previewMetricCount = data.metric_count;
            } catch (error) {
                console.error('Failed to refresh preview:', error);
                this.previewError = error.message || 'Failed to load preview';
                this.previewContent = '';
                this.previewMetricCount = 0;
            } finally {
                this.loadingPreview = false;
            }
        },

        async testScrape() {
            this.testingScrape = true;
            this.scrapeResult = null;

            try {
                const startTime = performance.now();
                const response = await fetch('/api/metrics/prometheus');
                const latencyMs = Math.round(performance.now() - startTime);

                if (response.ok) {
                    const text = await response.text();
                    const metricCount = (text.match(/^[a-z_]+\{/gm) || []).length;
                    const sizeBytes = new Blob([text]).size;

                    this.scrapeResult = {
                        success: true,
                        latency_ms: latencyMs,
                        metric_count: metricCount,
                        size_bytes: sizeBytes,
                    };
                    if (window.showToast) {
                        window.showToast(`Scrape successful: ${metricCount} metrics in ${latencyMs}ms`, 'success');
                    }
                } else {
                    this.scrapeResult = {
                        success: false,
                        error: `HTTP ${response.status}: ${response.statusText}`,
                        latency_ms: latencyMs,
                    };
                    if (window.showToast) {
                        window.showToast(`Scrape failed: ${response.statusText}`, 'error');
                    }
                }
            } catch (error) {
                console.error('Failed to test scrape:', error);
                this.scrapeResult = {
                    success: false,
                    error: error.message || 'Network error',
                };
                if (window.showToast) {
                    window.showToast('Scrape failed: Network error', 'error');
                }
            } finally {
                this.testingScrape = false;
            }
        },

        copyEndpointUrl() {
            const url = `${window.location.origin}/api/metrics/prometheus`;
            navigator.clipboard.writeText(url).then(() => {
                // Show toast
                const toast = this.$refs.copyToast;
                if (toast && typeof bootstrap !== 'undefined') {
                    const bsToast = new bootstrap.Toast(toast, { delay: 2000 });
                    bsToast.show();
                }
            }).catch(err => {
                console.error('Failed to copy URL:', err);
            });
        },

        async dismissHero() {
            this.heroDismissed = true;

            try {
                await fetch('/api/metrics/config/dismiss-hint/hero', {
                    method: 'POST',
                });
            } catch (error) {
                console.error('Failed to dismiss hero hint:', error);
            }
        },

        toggleCategory(categoryId) {
            const index = this.selectedCategories.indexOf(categoryId);
            if (index === -1) {
                this.selectedCategories.push(categoryId);
            } else {
                this.selectedCategories.splice(index, 1);
            }
            this.saveConfig();
        },

        toggleCategoryExpand(categoryId) {
            this.expandedCategory = this.expandedCategory === categoryId ? null : categoryId;
        },
    };
}

// Register component globally
if (typeof window !== 'undefined') {
    window.metricsComponent = metricsComponent;
}
