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
            selectedTrainingMediaLabel: '',
            selectedTrainingMediaStep: null,
            trainingChartHover: null,
            _trainingMetricsChart: null,

            async loadTrainingRuns() {
                this.trainingMetricsLoading = true;
                this.trainingMetricsError = null;
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
                    await this.loadTrainingRun();
                } catch (error) {
                    this.trainingMetricsError = error.message || 'Unable to load training runs';
                } finally {
                    this.trainingMetricsLoading = false;
                }
            },

            async selectTrainingRun(environment) {
                if (!environment || environment === this.selectedTrainingEnvironment) return;
                this.selectedTrainingEnvironment = environment;
                await this.loadTrainingRun();
            },

            async loadTrainingRun() {
                if (!this.selectedTrainingEnvironment) return;
                this.trainingMetricsLoading = true;
                this.trainingMetricsError = null;
                try {
                    const environment = encodeURIComponent(this.selectedTrainingEnvironment);
                    const response = await fetch(`/api/metrics/training/runs/${environment}?max_points=4000`);
                    if (!response.ok) throw new Error(`Unable to load run metrics (HTTP ${response.status})`);
                    this.trainingRunData = await response.json();
                    const available = this.trainingRunData.available_metrics || [];
                    this.selectedTrainingMetrics = global.TrainingMetricsCharts.defaultMetricNames(available);
                    this.selectDefaultTrainingMedia();
                    this.$nextTick(() => this.renderTrainingMetricsChart());
                } catch (error) {
                    this.trainingMetricsError = error.message || 'Unable to load run metrics';
                    this.trainingRunData = null;
                } finally {
                    this.trainingMetricsLoading = false;
                }
            },

            setMetricsSection(section) {
                this.activeMetricsSection = section;
                if (section === 'training') this.$nextTick(() => this.renderTrainingMetricsChart());
                if (section === 'system') this.$nextTick(() => this.renderActiveGpuCharts());
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
                );
            },

            latestTrainingMetrics() {
                if (!this.trainingRunData) return [];
                const latest = global.TrainingMetricsCharts.latestMetricValues(this.trainingRunData.records || []);
                return this.selectedTrainingMetrics.map((name) => ({ name, value: latest[name] }));
            },

            formatTrainingMetric(value) {
                return global.TrainingMetricsCharts.formatMetricValue(value);
            },

            trainingMediaLabels() {
                if (!this.trainingRunData) return [];
                return Array.from(new Set((this.trainingRunData.media || []).map((item) => item.label))).sort();
            },

            trainingMediaSteps() {
                if (!this.trainingRunData || !this.selectedTrainingMediaLabel) return [];
                return Array.from(
                    new Set(
                        (this.trainingRunData.media || [])
                            .filter((item) => item.label === this.selectedTrainingMediaLabel)
                            .map((item) => item.step),
                    ),
                ).sort((left, right) => left - right);
            },

            selectDefaultTrainingMedia() {
                const labels = this.trainingMediaLabels();
                this.selectedTrainingMediaLabel = labels[0] || '';
                const steps = this.trainingMediaSteps();
                this.selectedTrainingMediaStep = steps.length ? steps[steps.length - 1] : null;
            },

            selectTrainingMediaLabel(label) {
                this.selectedTrainingMediaLabel = label;
                const steps = this.trainingMediaSteps();
                this.selectedTrainingMediaStep = steps.length ? steps[steps.length - 1] : null;
            },

            selectedTrainingMedia() {
                if (!this.trainingRunData) return [];
                return (this.trainingRunData.media || [])
                    .filter(
                        (item) =>
                            item.label === this.selectedTrainingMediaLabel && item.step === this.selectedTrainingMediaStep,
                    )
                    .sort((left, right) => left.index - right.index);
            },

            trainingReportUrl() {
                if (!this.selectedTrainingEnvironment) return '#';
                return `/api/metrics/training/runs/${encodeURIComponent(this.selectedTrainingEnvironment)}/report`;
            },

            destroyTrainingMetrics() {
                if (this._trainingMetricsChart) {
                    this._trainingMetricsChart.destroy();
                    this._trainingMetricsChart = null;
                }
            },
        };
    }

    global.trainingMetricsState = trainingMetricsState;
})(typeof window !== 'undefined' ? window : globalThis);
