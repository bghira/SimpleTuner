(function (global) {
    'use strict';

    function trainingMetricsState() {
        return {
            activeMetricsSection: 'training',
            trainingRuns: [],
            selectedTrainingEnvironment: '',
            trainingRunData: null,
            trainingMetricsLoading: false,
            trainingMetricsError: null,
            selectedTrainingMetrics: [],
            selectedTrainingMediaStep: null,
            selectedTrainingXAxis: 'step',
            trainingMediaLightbox: null,
            trainingChartHover: null,
            _trainingMetricsChart: null,
            _trainingTimestepChart: null,
            _trainingMetricsTimer: null,
            trainingMetricsPollMs: 5000,

            async loadTrainingRuns(options = {}) {
                const { silent = false } = options;
                if (!silent) {
                    this.trainingMetricsLoading = true;
                    this.trainingMetricsError = null;
                } else if (this.trainingMetricsLoading) {
                    return;
                }
                try {
                    const response = await fetch('/api/metrics/training/runs');
                    if (!response.ok) throw new Error(`Unable to load training runs (HTTP ${response.status})`);
                    const data = await response.json();
                    this.trainingRuns = Array.isArray(data.runs) ? data.runs : [];
                    if (!this.trainingRuns.length) {
                        this.selectedTrainingEnvironment = '';
                        this.trainingRunData = null;
                        return;
                    }
                    const selectedStillExists = this.trainingRuns.some(
                        (run) => run.environment === this.selectedTrainingEnvironment,
                    );
                    if (!selectedStillExists) this.selectedTrainingEnvironment = this.trainingRuns[0].environment;
                    await this.loadTrainingRun({ silent, preserveSelections: silent });
                } catch (error) {
                    this.trainingMetricsError = error.message || 'Unable to load training runs';
                } finally {
                    if (!silent) this.trainingMetricsLoading = false;
                }
            },

            async selectTrainingRun(environment) {
                if (!environment || environment === this.selectedTrainingEnvironment) return;
                this.selectedTrainingEnvironment = environment;
                await this.loadTrainingRun();
            },

            async loadTrainingRun(options = {}) {
                if (!this.selectedTrainingEnvironment) return;
                const { silent = false, preserveSelections = false } = options;
                if (!silent) {
                    this.trainingMetricsLoading = true;
                    this.trainingMetricsError = null;
                } else if (this.trainingMetricsLoading) {
                    return;
                }
                try {
                    const environment = encodeURIComponent(this.selectedTrainingEnvironment);
                    const response = await fetch(`/api/metrics/training/runs/${environment}?max_points=4000`);
                    if (!response.ok) throw new Error(`Unable to load run metrics (HTTP ${response.status})`);
                    this.trainingRunData = await response.json();
                    const available = this.trainingRunData.available_metrics || [];
                    this.syncSelectedTrainingMetrics(available, preserveSelections);
                    this.selectDefaultTrainingMedia({ preserveSelection: preserveSelections });
                    this.$nextTick(() => {
                        this.renderTrainingMetricsChart();
                        this.renderTimestepDistributionChart();
                    });
                } catch (error) {
                    this.trainingMetricsError = error.message || 'Unable to load run metrics';
                    this.trainingRunData = null;
                } finally {
                    if (!silent) this.trainingMetricsLoading = false;
                }
            },

            setMetricsSection(section) {
                this.activeMetricsSection = section;
                if (section === 'training') {
                    this.startTrainingMetricsPolling();
                    this.$nextTick(() => {
                        this.renderTrainingMetricsChart();
                        this.renderTimestepDistributionChart();
                    });
                } else {
                    this.stopTrainingMetricsPolling();
                }
                if (section === 'system') this.$nextTick(() => this.renderActiveGpuCharts());
            },

            syncSelectedTrainingMetrics(available, preserveSelections = false) {
                if (preserveSelections) {
                    const retained = this.selectedTrainingMetrics.filter((metric) => available.includes(metric));
                    if (retained.length) {
                        this.selectedTrainingMetrics = retained.slice(0, 8);
                        return;
                    }
                }
                this.selectedTrainingMetrics = global.TrainingMetricsCharts.defaultMetricNames(available);
            },

            setTrainingXAxis(mode) {
                this.selectedTrainingXAxis = mode === 'minutes' ? 'minutes' : 'step';
                this.renderTrainingMetricsChart();
            },

            toggleTrainingMetric(metric) {
                const index = this.selectedTrainingMetrics.indexOf(metric);
                if (index >= 0) {
                    this.selectedTrainingMetrics.splice(index, 1);
                } else if (this.selectedTrainingMetrics.length < 8) {
                    this.selectedTrainingMetrics.push(metric);
                }
                this.renderTrainingMetricsChart();
            },

            renderTrainingMetricsChart() {
                const canvas = this.$refs.trainingMetricsCanvas;
                if (!canvas || !this.trainingRunData || !global.TrainingMetricsCharts) return;
                if (!this._trainingMetricsChart) {
                    this._trainingMetricsChart = new global.TrainingMetricsCharts.TrainingMetricsChart(canvas, {
                        onHover: (value) => {
                            this.trainingChartHover = value;
                        },
                    });
                }
                this._trainingMetricsChart.setData(
                    this.trainingRunData.records || [],
                    this.selectedTrainingMetrics,
                    { xAxisMode: this.selectedTrainingXAxis },
                );
            },

            renderTimestepDistributionChart() {
                const canvas = this.$refs.timestepDistributionCanvas;
                if (!canvas || !this.trainingRunData || !global.TrainingMetricsCharts) return;
                if (!this._trainingTimestepChart) {
                    this._trainingTimestepChart = new global.TrainingMetricsCharts.TimestepDistributionChart(canvas);
                }
                this._trainingTimestepChart.setData(this.trainingRunData.timesteps || []);
            },

            latestTrainingMetrics() {
                if (!this.trainingRunData) return [];
                const latest = global.TrainingMetricsCharts.latestMetricValues(this.trainingRunData.records || []);
                return this.selectedTrainingMetrics.map((name) => ({ name, value: latest[name] }));
            },

            formatTrainingMetric(value) {
                return global.TrainingMetricsCharts.formatMetricValue(value);
            },

            trainingMediaSteps() {
                if (!this.trainingRunData) return [];
                return Array.from(
                    new Set((this.trainingRunData.media || []).map((item) => item.step)),
                ).sort((left, right) => left - right);
            },

            selectDefaultTrainingMedia(options = {}) {
                const { preserveSelection = false } = options;
                const steps = this.trainingMediaSteps();
                if (preserveSelection && steps.includes(this.selectedTrainingMediaStep)) return;
                this.selectedTrainingMediaStep = steps.length ? steps[steps.length - 1] : null;
            },

            selectedTrainingMedia() {
                if (!this.trainingRunData) return [];
                return (this.trainingRunData.media || [])
                    .filter((item) => item.step === this.selectedTrainingMediaStep)
                    .sort((left, right) => {
                        const labelOrder = String(left.label || '').localeCompare(String(right.label || ''));
                        return labelOrder || Number(left.index || 0) - Number(right.index || 0);
                    });
            },

            selectedTrainingMediaGroups() {
                const groups = new Map();
                this.selectedTrainingMedia().forEach((item) => {
                    const label = item.label || 'Validation';
                    if (!groups.has(label)) groups.set(label, []);
                    groups.get(label).push(item);
                });
                return Array.from(groups, ([label, items]) => ({ label, items }));
            },

            selectedTrainingImages() {
                return this.selectedTrainingMedia()
                    .filter((item) => item.type === 'image' && item.url)
                    .map((item) => ({
                        item,
                        src: item.url,
                        caption: [
                            item.label,
                            `step ${Number(item.step).toLocaleString()}`,
                            item.resolution,
                            `output ${Number(item.index || 0) + 1}`,
                        ].filter(Boolean).join(' · '),
                    }));
            },

            openTrainingMediaLightbox(item) {
                if (!item || item.type !== 'image') return;
                const images = this.selectedTrainingImages();
                const index = Math.max(0, images.findIndex((image) => image.item.path === item.path));
                this.trainingMediaLightbox = { images, index, actualSize: false };
            },

            closeTrainingMediaLightbox() {
                this.trainingMediaLightbox = null;
            },

            setTrainingLightboxIndex(index) {
                if (!this.trainingMediaLightbox) return;
                const maxIndex = this.trainingMediaLightbox.images.length - 1;
                this.trainingMediaLightbox.index = Math.max(0, Math.min(maxIndex, index));
            },

            toggleTrainingLightboxSize() {
                if (!this.trainingMediaLightbox) return;
                this.trainingMediaLightbox.actualSize = !this.trainingMediaLightbox.actualSize;
            },

            trainingReportUrl() {
                if (!this.selectedTrainingEnvironment) return '#';
                return `/api/metrics/training/runs/${encodeURIComponent(this.selectedTrainingEnvironment)}/report`;
            },

            destroyTrainingMetrics() {
                this.stopTrainingMetricsPolling();
                if (this._trainingMetricsChart) {
                    this._trainingMetricsChart.destroy();
                    this._trainingMetricsChart = null;
                }
                if (this._trainingTimestepChart) {
                    this._trainingTimestepChart.destroy();
                    this._trainingTimestepChart = null;
                }
            },

            startTrainingMetricsPolling() {
                if (this._trainingMetricsTimer) return;
                this._trainingMetricsTimer = window.setInterval(() => {
                    if (this.activeMetricsSection !== 'training') return;
                    if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return;
                    const status = this.trainingRunData && this.trainingRunData.run && this.trainingRunData.run.status;
                    if (status !== 'running') return;
                    this.loadTrainingRuns({ silent: true });
                }, this.trainingMetricsPollMs);
            },

            stopTrainingMetricsPolling() {
                if (!this._trainingMetricsTimer) return;
                window.clearInterval(this._trainingMetricsTimer);
                this._trainingMetricsTimer = null;
            },
        };
    }

    global.trainingMetricsState = trainingMetricsState;
})(typeof window !== 'undefined' ? window : globalThis);
