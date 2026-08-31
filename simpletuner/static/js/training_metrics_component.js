(function (global) {
    'use strict';

    const MAX_CHART_METRICS = 8;

    function chartId() {
        if (global.crypto && typeof global.crypto.randomUUID === 'function') {
            return global.crypto.randomUUID();
        }
        return `chart-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    }

    function cleanLayoutName(name) {
        const value = String(name || '').trim();
        return value || 'Default';
    }

    function uniqueMetrics(metrics, available) {
        const allowed = new Set(available || []);
        const selected = [];
        (metrics || []).forEach((metric) => {
            if (typeof metric !== 'string') return;
            if (allowed.size && !allowed.has(metric)) return;
            if (!selected.includes(metric)) selected.push(metric);
        });
        return selected.slice(0, MAX_CHART_METRICS);
    }

    function normalizeSmoothing(value) {
        if (global.TrainingMetricsCharts && typeof global.TrainingMetricsCharts.normalizeSmoothing === 'function') {
            return global.TrainingMetricsCharts.normalizeSmoothing(value);
        }
        const parsed = Number(value);
        if (!Number.isFinite(parsed)) return 0;
        return Math.max(0, Math.min(0.99, parsed));
    }

    function makeScalarChart(name, metrics, id = null, metricSearch = '', smoothing = 0) {
        return {
            id: id || chartId(),
            kind: 'scalar',
            name: String(name || 'Training metrics').trim() || 'Training metrics',
            metrics: Array.isArray(metrics) ? metrics.slice(0, MAX_CHART_METRICS) : [],
            metricSearch,
            smoothing: normalizeSmoothing(smoothing),
        };
    }

    function normalizeCharts(charts, available) {
        const normalized = [];
        (charts || []).forEach((chart, index) => {
            if (!chart || chart.kind === 'timestep') return;
            normalized.push(makeScalarChart(
                chart.name || `Chart ${index + 1}`,
                uniqueMetrics(chart.metrics, available),
                typeof chart.id === 'string' && chart.id.trim() ? chart.id : null,
                typeof chart.metricSearch === 'string' ? chart.metricSearch : '',
                chart.smoothing,
            ));
        });
        return normalized;
    }

    function defaultCharts(available) {
        const metrics = global.TrainingMetricsCharts
            ? global.TrainingMetricsCharts.defaultMetricNames(available || [], 1)
            : [];
        return [makeScalarChart('Training metrics', metrics)];
    }

    function defaultLayoutState() {
        return {
            templates: [],
            environments: {},
        };
    }

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
            selectedTrainingLayoutName: 'Default',
            trainingLayoutNameInput: 'Default',
            trainingLayoutModalOpen: false,
            trainingLayoutSaving: false,
            trainingLayoutError: null,
            trainingMetricLayouts: defaultLayoutState(),
            trainingMetricLayoutsLoaded: false,
            trainingCharts: [],
            showTrainingTimestepChart: false,
            trainingMediaPanelCollapsed: false,
            trainingChartHover: {},
            trainingChartSelectorOpen: null,
            trainingMediaLightbox: null,
            _trainingMetricsCharts: {},
            _trainingTimestepChart: null,
            _trainingMetricsTimer: null,
            _trainingLayoutSaveTimer: null,
            trainingMetricsPollMs: 5000,

            async loadTrainingMetricLayouts() {
                try {
                    const response = await fetch('/api/webui/ui-state/training-metrics');
                    if (!response.ok) throw new Error(`Unable to load training metrics layouts (HTTP ${response.status})`);
                    const payload = await response.json();
                    this.trainingMetricLayouts = this.normalizeTrainingMetricLayoutState(payload);
                } catch (error) {
                    console.warn('Failed to load training metrics layouts:', error);
                    this.trainingMetricLayouts = defaultLayoutState();
                } finally {
                    this.trainingMetricLayoutsLoaded = true;
                }
            },

            normalizeTrainingMetricLayoutState(payload) {
                const state = defaultLayoutState();
                if (!payload || typeof payload !== 'object') return state;
                if (Array.isArray(payload.templates)) {
                    payload.templates.forEach((template) => {
                        if (!template || typeof template !== 'object') return;
                        const name = cleanLayoutName(template.name);
                        const charts = normalizeCharts(template.charts || [], []);
                        state.templates.push({ name, charts });
                    });
                }
                if (payload.environments && typeof payload.environments === 'object') {
                    Object.entries(payload.environments).forEach(([environment, layout]) => {
                        if (!layout || typeof layout !== 'object') return;
                        state.environments[environment] = {
                            template: cleanLayoutName(layout.template),
                            charts: normalizeCharts(layout.charts || [], []),
                            xAxis: layout.xAxis === 'minutes' ? 'minutes' : 'step',
                            showTimestepChart: Boolean(layout.showTimestepChart),
                            mediaPanelCollapsed: Boolean(layout.mediaPanelCollapsed),
                        };
                    });
                }
                return state;
            },

            serializeTrainingMetricLayoutState() {
                return {
                    templates: (this.trainingMetricLayouts.templates || []).map((template) => ({
                        name: cleanLayoutName(template.name),
                        charts: normalizeCharts(template.charts || [], []).map((chart) => ({
                            kind: 'scalar',
                            name: chart.name,
                            metrics: chart.metrics,
                            smoothing: chart.smoothing,
                        })),
                    })),
                    environments: Object.fromEntries(
                        Object.entries(this.trainingMetricLayouts.environments || {}).map(([environment, layout]) => [
                            environment,
                            {
                                template: cleanLayoutName(layout.template),
                                xAxis: layout.xAxis === 'minutes' ? 'minutes' : 'step',
                                showTimestepChart: Boolean(layout.showTimestepChart),
                                mediaPanelCollapsed: Boolean(layout.mediaPanelCollapsed),
                                charts: normalizeCharts(layout.charts || [], []).map((chart) => ({
                                    kind: 'scalar',
                                    name: chart.name,
                                    metrics: chart.metrics,
                                    smoothing: chart.smoothing,
                                })),
                            },
                        ]),
                    ),
                };
            },

            async saveTrainingMetricLayoutState() {
                if (!this.trainingMetricLayoutsLoaded) return;
                this.trainingLayoutSaving = true;
                this.trainingLayoutError = null;
                try {
                    const response = await fetch('/api/webui/ui-state/training-metrics', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.serializeTrainingMetricLayoutState()),
                    });
                    if (!response.ok) throw new Error(`Unable to save training metrics layout (HTTP ${response.status})`);
                    this.trainingMetricLayouts = this.normalizeTrainingMetricLayoutState(await response.json());
                } catch (error) {
                    this.trainingLayoutError = error.message || 'Unable to save training metrics layout';
                } finally {
                    this.trainingLayoutSaving = false;
                }
            },

            queueTrainingMetricLayoutSave() {
                if (!this.trainingMetricLayoutsLoaded || !this.selectedTrainingEnvironment) return;
                this.rememberTrainingEnvironmentLayout();
                if (this._trainingLayoutSaveTimer) window.clearTimeout(this._trainingLayoutSaveTimer);
                this._trainingLayoutSaveTimer = window.setTimeout(() => {
                    this._trainingLayoutSaveTimer = null;
                    this.saveTrainingMetricLayoutState();
                }, 400);
            },

            rememberTrainingEnvironmentLayout() {
                if (!this.selectedTrainingEnvironment) return;
                this.trainingMetricLayouts.environments[this.selectedTrainingEnvironment] = {
                    template: cleanLayoutName(this.selectedTrainingLayoutName),
                    xAxis: this.selectedTrainingXAxis,
                    showTimestepChart: Boolean(this.showTrainingTimestepChart),
                    mediaPanelCollapsed: Boolean(this.trainingMediaPanelCollapsed),
                    charts: normalizeCharts(this.trainingCharts, this.trainingRunData?.available_metrics || []),
                };
            },

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
                    this.syncTrainingCharts(available, preserveSelections);
                    this.selectDefaultTrainingMedia({ preserveSelection: preserveSelections });
                    this.$nextTick(() => {
                        this.renderTrainingMetricsCharts();
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
                        this.renderTrainingMetricsCharts();
                        this.renderTimestepDistributionChart();
                    });
                } else {
                    this.stopTrainingMetricsPolling();
                }
                if (section === 'system') this.$nextTick(() => this.renderActiveGpuCharts());
            },

            trainingSavedLayouts() {
                const names = ['Default'];
                (this.trainingMetricLayouts.templates || []).forEach((template) => {
                    const name = cleanLayoutName(template.name);
                    if (!names.includes(name)) names.push(name);
                });
                return names;
            },

            getTrainingTemplate(name) {
                const normalized = cleanLayoutName(name);
                return (this.trainingMetricLayouts.templates || []).find(
                    (template) => cleanLayoutName(template.name) === normalized,
                );
            },

            syncTrainingCharts(available, preserveSelections = false) {
                if (preserveSelections && this.trainingCharts.length) {
                    this.trainingCharts = normalizeCharts(this.trainingCharts, available);
                    if (!this.trainingCharts.length) this.trainingCharts = defaultCharts(available);
                    this.syncLegacySelectedTrainingMetrics();
                    return;
                }

                const environmentLayout = this.trainingMetricLayouts.environments[this.selectedTrainingEnvironment];
                if (environmentLayout && Array.isArray(environmentLayout.charts) && environmentLayout.charts.length) {
                    this.selectedTrainingLayoutName = cleanLayoutName(environmentLayout.template);
                    this.trainingLayoutNameInput = this.selectedTrainingLayoutName;
                    this.selectedTrainingXAxis = environmentLayout.xAxis === 'minutes' ? 'minutes' : 'step';
                    this.showTrainingTimestepChart = Boolean(environmentLayout.showTimestepChart);
                    this.trainingMediaPanelCollapsed = Boolean(environmentLayout.mediaPanelCollapsed);
                    this.trainingCharts = normalizeCharts(environmentLayout.charts, available);
                    if (!this.trainingCharts.length) this.trainingCharts = defaultCharts(available);
                    this.syncLegacySelectedTrainingMetrics();
                    return;
                }

                const templateName = environmentLayout ? cleanLayoutName(environmentLayout.template) : 'Default';
                const template = this.getTrainingTemplate(templateName);
                this.selectedTrainingLayoutName = template ? template.name : 'Default';
                this.trainingLayoutNameInput = this.selectedTrainingLayoutName;
                this.selectedTrainingXAxis = environmentLayout?.xAxis === 'minutes' ? 'minutes' : 'step';
                this.showTrainingTimestepChart = Boolean(environmentLayout?.showTimestepChart);
                this.trainingMediaPanelCollapsed = Boolean(environmentLayout?.mediaPanelCollapsed);
                this.trainingCharts = template ? normalizeCharts(template.charts, available) : defaultCharts(available);
                if (!this.trainingCharts.length) this.trainingCharts = defaultCharts(available);
                this.syncLegacySelectedTrainingMetrics();
            },

            syncSelectedTrainingMetrics(available, preserveSelections = false) {
                this.syncTrainingCharts(available, preserveSelections);
            },

            syncLegacySelectedTrainingMetrics() {
                const firstChart = this.trainingCharts.find((chart) => chart.kind === 'scalar');
                this.selectedTrainingMetrics = firstChart ? firstChart.metrics.slice() : [];
            },

            selectTrainingLayout(name) {
                const layoutName = cleanLayoutName(name);
                const available = this.trainingRunData?.available_metrics || [];
                const template = this.getTrainingTemplate(layoutName);
                this.selectedTrainingLayoutName = template ? template.name : 'Default';
                this.trainingLayoutNameInput = this.selectedTrainingLayoutName;
                this.trainingCharts = template ? normalizeCharts(template.charts, available) : defaultCharts(available);
                if (!this.trainingCharts.length) this.trainingCharts = defaultCharts(available);
                this.syncLegacySelectedTrainingMetrics();
                this.trainingChartSelectorOpen = null;
                this.$nextTick(() => this.renderTrainingMetricsCharts());
                this.queueTrainingMetricLayoutSave();
            },

            async saveTrainingChartLayout(name) {
                const layoutName = cleanLayoutName(name || this.trainingLayoutNameInput || this.selectedTrainingLayoutName);
                const charts = normalizeCharts(this.trainingCharts, this.trainingRunData?.available_metrics || []);
                const existingIndex = (this.trainingMetricLayouts.templates || []).findIndex(
                    (template) => cleanLayoutName(template.name) === layoutName,
                );
                const template = { name: layoutName, charts };
                if (existingIndex >= 0) {
                    this.trainingMetricLayouts.templates.splice(existingIndex, 1, template);
                } else {
                    this.trainingMetricLayouts.templates.push(template);
                }
                this.selectedTrainingLayoutName = layoutName;
                this.trainingLayoutNameInput = layoutName;
                this.rememberTrainingEnvironmentLayout();
                await this.saveTrainingMetricLayoutState();
            },

            openTrainingLayoutModal() {
                this.trainingLayoutNameInput = cleanLayoutName(this.selectedTrainingLayoutName);
                this.trainingLayoutModalOpen = true;
            },

            closeTrainingLayoutModal() {
                if (this.trainingLayoutSaving) return;
                this.trainingLayoutModalOpen = false;
            },

            async confirmTrainingLayoutSave() {
                await this.saveTrainingChartLayout(this.trainingLayoutNameInput);
                if (!this.trainingLayoutError) this.trainingLayoutModalOpen = false;
            },

            setTrainingXAxis(mode) {
                this.selectedTrainingXAxis = mode === 'minutes' ? 'minutes' : 'step';
                this.renderTrainingMetricsCharts();
                this.queueTrainingMetricLayoutSave();
            },

            async toggleTrainingTimestepChart() {
                this.showTrainingTimestepChart = !this.showTrainingTimestepChart;
                this.$nextTick(() => this.renderTimestepDistributionChart());
                this.rememberTrainingEnvironmentLayout();
                if (this._trainingLayoutSaveTimer) {
                    window.clearTimeout(this._trainingLayoutSaveTimer);
                    this._trainingLayoutSaveTimer = null;
                }
                await this.saveTrainingMetricLayoutState();
            },

            async toggleTrainingMediaPanel() {
                this.trainingMediaPanelCollapsed = !this.trainingMediaPanelCollapsed;
                this.$nextTick(() => {
                    this.renderTrainingMetricsCharts();
                    this.renderTimestepDistributionChart();
                });
                this.rememberTrainingEnvironmentLayout();
                if (this._trainingLayoutSaveTimer) {
                    window.clearTimeout(this._trainingLayoutSaveTimer);
                    this._trainingLayoutSaveTimer = null;
                }
                await this.saveTrainingMetricLayoutState();
            },

            addTrainingChart() {
                this.trainingCharts.push(makeScalarChart(`Chart ${this.trainingCharts.length + 1}`, []));
                this.trainingChartSelectorOpen = this.trainingCharts[this.trainingCharts.length - 1].id;
                this.syncLegacySelectedTrainingMetrics();
                this.$nextTick(() => this.renderTrainingMetricsCharts());
                this.queueTrainingMetricLayoutSave();
            },

            removeTrainingChart(chartIdValue) {
                if (this.trainingCharts.length <= 1) return;
                this.trainingCharts = this.trainingCharts.filter((chart) => chart.id !== chartIdValue);
                if (this.trainingChartSelectorOpen === chartIdValue) this.trainingChartSelectorOpen = null;
                if (this._trainingMetricsCharts[chartIdValue]) {
                    this._trainingMetricsCharts[chartIdValue].destroy();
                    delete this._trainingMetricsCharts[chartIdValue];
                }
                this.syncLegacySelectedTrainingMetrics();
                this.$nextTick(() => this.renderTrainingMetricsCharts());
                this.queueTrainingMetricLayoutSave();
            },

            toggleTrainingChartSelector(chartIdValue) {
                this.trainingChartSelectorOpen = this.trainingChartSelectorOpen === chartIdValue ? null : chartIdValue;
            },

            filteredTrainingMetrics(chart) {
                const available = this.trainingRunData?.available_metrics || [];
                const query = String(chart.metricSearch || '').trim().toLowerCase();
                if (!query) return available;
                return available.filter((metric) => metric.toLowerCase().includes(query));
            },

            toggleTrainingMetric(metric, chartIdValue = null) {
                const chart = chartIdValue
                    ? this.trainingCharts.find((candidate) => candidate.id === chartIdValue)
                    : this.trainingCharts[0];
                if (!chart) return;
                const index = chart.metrics.indexOf(metric);
                if (index >= 0) {
                    chart.metrics.splice(index, 1);
                } else if (chart.metrics.length < MAX_CHART_METRICS) {
                    chart.metrics.push(metric);
                }
                this.syncLegacySelectedTrainingMetrics();
                this.renderTrainingMetricsCharts();
                this.queueTrainingMetricLayoutSave();
            },

            setTrainingChartSmoothing(chart, value) {
                if (!chart) return;
                chart.smoothing = normalizeSmoothing(value);
                this.renderTrainingMetricsCharts();
                this.queueTrainingMetricLayoutSave();
            },

            formatTrainingSmoothing(value) {
                const smoothing = normalizeSmoothing(value);
                if (!smoothing) return 'Off';
                return `${Math.round(smoothing * 100)}%`;
            },

            renderTrainingMetricsChart() {
                this.renderTrainingMetricsCharts();
            },

            renderTrainingMetricsCharts() {
                if (!this.trainingRunData || !global.TrainingMetricsCharts) return;
                const root = document.getElementById('metrics-tab-content') || document;
                const canvases = root.querySelectorAll('canvas[data-training-chart-id]');
                const liveChartIds = new Set();
                canvases.forEach((canvas) => {
                    const chartIdValue = canvas.dataset.trainingChartId;
                    const chart = this.trainingCharts.find((candidate) => candidate.id === chartIdValue);
                    if (!chart) return;
                    liveChartIds.add(chartIdValue);
                    if (!this._trainingMetricsCharts[chartIdValue]) {
                        this._trainingMetricsCharts[chartIdValue] = new global.TrainingMetricsCharts.TrainingMetricsChart(canvas, {
                            onHover: (value) => {
                                if (value) {
                                    this.trainingChartHover = { ...this.trainingChartHover, [chartIdValue]: value };
                                } else {
                                    const nextHover = { ...this.trainingChartHover };
                                    delete nextHover[chartIdValue];
                                    this.trainingChartHover = nextHover;
                                }
                            },
                        });
                    }
                    this._trainingMetricsCharts[chartIdValue].setData(
                        this.trainingRunData.records || [],
                        chart.metrics,
                        { xAxisMode: this.selectedTrainingXAxis, smoothing: chart.smoothing },
                    );
                });
                Object.keys(this._trainingMetricsCharts).forEach((chartIdValue) => {
                    if (liveChartIds.has(chartIdValue)) return;
                    this._trainingMetricsCharts[chartIdValue].destroy();
                    delete this._trainingMetricsCharts[chartIdValue];
                });
            },

            renderTimestepDistributionChart() {
                const canvas = this.$refs.timestepDistributionCanvas;
                if (!canvas || !this.trainingRunData || !global.TrainingMetricsCharts) return;
                if (!this._trainingTimestepChart) {
                    this._trainingTimestepChart = new global.TrainingMetricsCharts.TimestepDistributionChart(canvas);
                }
                this._trainingTimestepChart.setData(this.trainingRunData.timesteps || []);
            },

            trainingChartHoverLabel(chart) {
                const hover = this.trainingChartHover[chart.id];
                if (!hover) return '';
                if (hover.xAxisMode === 'minutes') {
                    return global.TrainingMetricsCharts.formatXAxisValue(hover.xValue, 'minutes');
                }
                return `Step ${Number(hover.record.step || 0).toLocaleString()}`;
            },

            trainingChartHoverValues(chart) {
                const hover = this.trainingChartHover[chart.id];
                if (!hover || !Array.isArray(hover.metrics)) return [];
                return hover.metrics
                    .filter((metric) => typeof metric.value === 'number' && Number.isFinite(metric.value))
                    .map((metric) => ({
                        ...metric,
                        formatted: this.formatTrainingMetric(metric.value),
                    }));
            },

            trainingChartLegend(chart) {
                const colors = global.TrainingMetricsCharts?.COLORS || [];
                return (chart?.metrics || []).map((metric, index) => ({
                    name: metric,
                    color: colors[index % colors.length] || '#38bdf8',
                }));
            },

            trainingChartTooltipStyle(chart) {
                const hover = this.trainingChartHover[chart.id];
                if (!hover) return '';
                const plottedValues = this.trainingChartHoverValues(chart);
                if (!plottedValues.length) return '';
                const tooltipWidth = 240;
                const chartWidth = Number(hover.width || 0);
                const minLeft = tooltipWidth / 2 + 8;
                const maxLeft = Math.max(minLeft, chartWidth - tooltipWidth / 2 - 8);
                const left = chartWidth
                    ? Math.min(Math.max(Number(hover.x || 0), minLeft), maxLeft)
                    : Number(hover.x || 0);
                const yValues = plottedValues
                    .map((metric) => Number(metric.y))
                    .filter((value) => Number.isFinite(value));
                const top = yValues.length ? Math.max(12, Math.min(...yValues)) : 24;
                const transform = top < 96 ? 'translate(-50%, 0.75rem)' : 'translate(-50%, calc(-100% - 0.75rem))';
                return `left: ${left}px; top: ${top}px; transform: ${transform};`;
            },

            latestTrainingMetrics(chart = null) {
                if (!this.trainingRunData) return [];
                const targetChart = chart || this.trainingCharts[0];
                const latest = global.TrainingMetricsCharts.latestMetricValues(this.trainingRunData.records || []);
                return (targetChart?.metrics || []).map((name) => ({ name, value: latest[name] }));
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

            trainingMediaStepIndex() {
                const steps = this.trainingMediaSteps();
                const index = steps.indexOf(this.selectedTrainingMediaStep);
                return index >= 0 ? index : Math.max(0, steps.length - 1);
            },

            setTrainingMediaStepIndex(index) {
                const steps = this.trainingMediaSteps();
                if (!steps.length) return;
                const parsed = Number.parseInt(index, 10);
                const clamped = Number.isFinite(parsed)
                    ? Math.max(0, Math.min(steps.length - 1, parsed))
                    : steps.length - 1;
                this.selectedTrainingMediaStep = steps[clamped];
            },

            selectedTrainingMediaStepLabel() {
                if (this.selectedTrainingMediaStep === null || typeof this.selectedTrainingMediaStep === 'undefined') {
                    return 'No step';
                }
                return `Step ${Number(this.selectedTrainingMediaStep).toLocaleString()}`;
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

            trainingImageCaption(item) {
                if (!item) return '';
                return [
                    item.label,
                    `step ${Number(item.step).toLocaleString()}`,
                    item.resolution,
                    `output ${Number(item.index || 0) + 1}`,
                ].filter(Boolean).join(' · ');
            },

            trainingImageEntry(item) {
                return {
                    item,
                    src: item.url,
                    caption: this.trainingImageCaption(item),
                };
            },

            selectedTrainingImages() {
                return this.selectedTrainingMedia()
                    .filter((item) => item.type === 'image' && item.url)
                    .map((item) => this.trainingImageEntry(item));
            },

            allTrainingImages() {
                if (!this.trainingRunData) return [];
                return (this.trainingRunData.media || [])
                    .filter((item) => item.type === 'image' && item.url)
                    .sort((left, right) => {
                        const stepOrder = Number(left.step || 0) - Number(right.step || 0);
                        if (stepOrder) return stepOrder;
                        const labelOrder = String(left.label || '').localeCompare(String(right.label || ''));
                        return labelOrder || Number(left.index || 0) - Number(right.index || 0);
                    })
                    .map((item) => this.trainingImageEntry(item));
            },

            openTrainingMediaLightbox(item) {
                if (!item || item.type !== 'image') return;
                this.trainingMediaLightbox = { path: item.path, actualSize: false };
            },

            closeTrainingMediaLightbox() {
                this.trainingMediaLightbox = null;
            },

            currentTrainingLightboxImage() {
                if (!this.trainingMediaLightbox) return null;
                return this.allTrainingImages().find(
                    (image) => image.item.path === this.trainingMediaLightbox.path,
                ) || null;
            },

            trainingLightboxSameStepImages() {
                const current = this.currentTrainingLightboxImage();
                if (!current) return [];
                return this.allTrainingImages().filter((image) => image.item.step === current.item.step);
            },

            trainingLightboxSameOutputImages() {
                const current = this.currentTrainingLightboxImage();
                if (!current) return [];
                return this.allTrainingImages().filter((image) => (
                    String(image.item.label || '') === String(current.item.label || '')
                    && Number(image.item.index || 0) === Number(current.item.index || 0)
                ));
            },

            trainingLightboxSameStepIndex() {
                const current = this.currentTrainingLightboxImage();
                if (!current) return -1;
                return this.trainingLightboxSameStepImages().findIndex((image) => image.item.path === current.item.path);
            },

            trainingLightboxSameOutputIndex() {
                const current = this.currentTrainingLightboxImage();
                if (!current) return -1;
                return this.trainingLightboxSameOutputImages().findIndex((image) => image.item.path === current.item.path);
            },

            trainingLightboxStepLabel() {
                const current = this.currentTrainingLightboxImage();
                if (!current) return 'No step';
                return `Step ${Number(current.item.step).toLocaleString()}`;
            },

            setTrainingLightboxImageOffset(offset) {
                if (!this.trainingMediaLightbox) return;
                const images = this.trainingLightboxSameStepImages();
                const currentIndex = this.trainingLightboxSameStepIndex();
                const next = images[currentIndex + offset];
                if (next) this.trainingMediaLightbox.path = next.item.path;
            },

            setTrainingLightboxStepIndex(index) {
                if (!this.trainingMediaLightbox) return;
                const images = this.trainingLightboxSameOutputImages();
                if (!images.length) return;
                const parsed = Number.parseInt(index, 10);
                const clamped = Number.isFinite(parsed)
                    ? Math.max(0, Math.min(images.length - 1, parsed))
                    : images.length - 1;
                const next = images[clamped];
                this.trainingMediaLightbox.path = next.item.path;
            },

            setTrainingLightboxStepOffset(offset) {
                if (!this.trainingMediaLightbox) return;
                const images = this.trainingLightboxSameOutputImages();
                const currentIndex = this.trainingLightboxSameOutputIndex();
                const next = images[currentIndex + offset];
                if (next) {
                    this.trainingMediaLightbox.path = next.item.path;
                }
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
                if (this._trainingLayoutSaveTimer) {
                    window.clearTimeout(this._trainingLayoutSaveTimer);
                    this._trainingLayoutSaveTimer = null;
                }
                Object.values(this._trainingMetricsCharts).forEach((chart) => chart.destroy());
                this._trainingMetricsCharts = {};
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
