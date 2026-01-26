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

        // GPU health state
        gpuHealth: [],
        gpuHealthLoading: false,
        activeGpuIndex: null,
        gpuHistory: {},
        gpuHistorySettings: {},
        gpuHistoryMaxPoints: 60,
        gpuHealthPollMs: 5000,
        _gpuHealthTimer: null,
        historyHover: {
            visible: false,
            left: '0px',
            top: '0px',
            lines: [],
            gpuIndex: null,
        },
        historyLatestLabel: {
            visible: false,
            left: '0px',
            top: '0px',
            lines: [],
            gpuIndex: null,
        },
        historyChartPadding: 8,

        // Initialization
        async init() {
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }

            // Check if hero should be shown
            this.heroDismissed = this.dismissedHints.includes('hero');

            // Load categories, templates, circuit breaker status, and GPU health
            await Promise.all([
                this.loadCategoriesAndTemplates(),
                this.loadCircuitBreakers(),
                this.loadGpuHealth(),
            ]);

            // Load initial preview if enabled
            if (this.prometheusEnabled) {
                await this.refreshPreview();
            }

            this.startGpuHealthPolling();
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

        async loadGpuHealth(options = {}) {
            const { silent = false } = options;
            if (!silent) {
                this.gpuHealthLoading = true;
            } else if (this.gpuHealthLoading) {
                return;
            }
            try {
                const response = await fetch('/api/metrics/gpu-health');
                if (response.ok) {
                    const data = await response.json();
                    this.gpuHealth = data.gpus || [];
                    this.syncGpuHealthState();
                }
            } catch (error) {
                console.error('Failed to load GPU health:', error);
            } finally {
                if (!silent) {
                    this.gpuHealthLoading = false;
                }
            }
        },

        syncGpuHealthState() {
            if (!Array.isArray(this.gpuHealth)) {
                return;
            }
            this.gpuHealth.forEach((gpu) => {
                this.ensureGpuHistoryState(gpu.index);
            });

            if (this.gpuHealth.length > 0) {
                const hasActive = this.gpuHealth.some((gpu) => gpu.index === this.activeGpuIndex);
                if (!hasActive) {
                    this.activeGpuIndex = this.gpuHealth[0].index;
                }
            } else {
                this.activeGpuIndex = null;
                this.historyLatestLabel = {
                    visible: false,
                    left: '0px',
                    top: '0px',
                    lines: [],
                    gpuIndex: null,
                };
            }

            this.recordGpuHistory();
            this.$nextTick(() => {
                this.renderActiveGpuCharts();
            });
        },

        startGpuHealthPolling() {
            if (this._gpuHealthTimer) {
                return;
            }
            this._gpuHealthTimer = window.setInterval(() => {
                if (typeof document !== 'undefined' && document.visibilityState === 'hidden') {
                    return;
                }
                this.loadGpuHealth({ silent: true });
            }, this.gpuHealthPollMs);
        },

        setActiveGpu(gpuIndex) {
            this.activeGpuIndex = gpuIndex;
            this.ensureGpuHistoryState(gpuIndex);
            this.$nextTick(() => {
                this.renderActiveGpuCharts();
            });
        },

        getActiveGpu() {
            if (!Array.isArray(this.gpuHealth) || this.gpuHealth.length === 0) {
                return null;
            }
            return this.gpuHealth.find((gpu) => gpu.index === this.activeGpuIndex) || this.gpuHealth[0];
        },

        ensureGpuHistoryState(gpuIndex) {
            if (gpuIndex === null || typeof gpuIndex === 'undefined') {
                return;
            }
            if (!this.gpuHistory[gpuIndex]) {
                this.gpuHistory[gpuIndex] = [];
            }
            if (!this.gpuHistorySettings[gpuIndex]) {
                this.gpuHistorySettings[gpuIndex] = { showTemp: true, showFan: true };
            }
        },

        recordGpuHistory() {
            const now = Date.now();
            this.gpuHealth.forEach((gpu) => {
                const gpuIndex = gpu.index;
                if (gpuIndex === null || typeof gpuIndex === 'undefined') {
                    return;
                }
                this.ensureGpuHistoryState(gpuIndex);
                const history = this.gpuHistory[gpuIndex];
                const temperature =
                    typeof gpu.temperature_celsius === 'number' && Number.isFinite(gpu.temperature_celsius)
                        ? gpu.temperature_celsius
                        : null;
                const fanSpeed =
                    typeof gpu.fan_speed_percent === 'number' && Number.isFinite(gpu.fan_speed_percent)
                        ? gpu.fan_speed_percent
                        : null;
                if (temperature === null && fanSpeed === null) {
                    return;
                }
                history.push({
                    timestamp: now,
                    temperature,
                    fanSpeed,
                });
                if (history.length > this.gpuHistoryMaxPoints) {
                    history.splice(0, history.length - this.gpuHistoryMaxPoints);
                }
            });
        },

        renderActiveGpuCharts() {
            const gpu = this.getActiveGpu();
            if (!gpu) {
                return;
            }
            this.renderVramChart(gpu);
            this.renderHistoryChart(gpu.index);
        },

        renderVramChart(gpu) {
            const canvas = this.$refs.vramCanvas;
            if (!canvas) {
                return;
            }
            const percent = this.getVramPercent(gpu);
            this.drawVramPie(canvas, percent);
        },

        renderHistoryChart(gpuIndex) {
            const canvas = this.$refs.historyCanvas;
            if (!canvas) {
                return;
            }
            const history = this.gpuHistory[gpuIndex] || [];
            const settings = this.gpuHistorySettings[gpuIndex] || { showTemp: true, showFan: true };
            this.drawHistoryLines(canvas, history, settings);
            this.updateHistoryLatestLabel(gpuIndex);
        },

        drawVramPie(canvas, percent) {
            const context = canvas.getContext('2d');
            if (!context) {
                return;
            }
            const wrapper = canvas.parentElement;
            const wrapperRect = wrapper ? wrapper.getBoundingClientRect() : null;
            const wrapperWidth = wrapperRect ? wrapperRect.width : 0;
            const wrapperHeight = wrapperRect ? wrapperRect.height : 0;
            const fallbackSize = canvas.clientWidth || 80;
            const size = wrapperWidth > 0 && wrapperHeight > 0
                ? Math.round(Math.min(wrapperWidth, wrapperHeight))
                : fallbackSize;
            const dpr = window.devicePixelRatio || 1;
            const targetWidth = size * dpr;
            const targetHeight = size * dpr;
            if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
                canvas.width = targetWidth;
                canvas.height = targetHeight;
            }
            const desiredStyle = `${size}px`;
            if (canvas.style.width !== desiredStyle || canvas.style.height !== desiredStyle) {
                canvas.style.width = desiredStyle;
                canvas.style.height = desiredStyle;
            }
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, size, size);

            const center = size / 2;
            const radius = Math.max(8, center - 6);
            const startAngle = -Math.PI / 2;

            context.lineWidth = 8;
            context.strokeStyle = 'rgba(148, 163, 184, 0.2)';
            context.beginPath();
            context.arc(center, center, radius, 0, Math.PI * 2);
            context.stroke();

            if (typeof percent === 'number' && Number.isFinite(percent)) {
                const clamped = Math.max(0, Math.min(100, percent));
                const endAngle = startAngle + (Math.PI * 2 * clamped) / 100;
                context.strokeStyle = this.getVramColor(clamped);
                context.lineCap = 'round';
                context.beginPath();
                context.arc(center, center, radius, startAngle, endAngle);
                context.stroke();
            }
        },

        drawHistoryLines(canvas, history, settings) {
            const context = canvas.getContext('2d');
            if (!context) {
                return;
            }
            const width = canvas.clientWidth || 240;
            const height = canvas.clientHeight || 120;
            const dpr = window.devicePixelRatio || 1;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, width, height);

            if (!history || history.length === 0 || (!settings.showTemp && !settings.showFan)) {
                return;
            }

            const padding = this.historyChartPadding;
            const chartWidth = Math.max(1, width - padding * 2);
            const chartHeight = Math.max(1, height - padding * 2);

            context.strokeStyle = 'rgba(148, 163, 184, 0.15)';
            context.lineWidth = 1;
            [0.25, 0.5, 0.75].forEach((ratio) => {
                const y = padding + chartHeight * ratio;
                context.beginPath();
                context.moveTo(padding, y);
                context.lineTo(padding + chartWidth, y);
                context.stroke();
            });

            const points = history.slice(-this.gpuHistoryMaxPoints);
            if (points.length === 1) {
                points.push(points[0]);
            }
            const step = points.length > 1 ? chartWidth / (points.length - 1) : 0;

            const tempValues = points
                .map((point) => point.temperature)
                .filter((value) => typeof value === 'number' && Number.isFinite(value));

            const tempScale = this.buildRange(tempValues, 10);
            const fanScale = { min: 0, max: 100 };

            if (settings.showTemp) {
                this.drawLineSeries(context, points, step, padding, chartHeight, tempScale, 'temperature', '#ef4444');
            }
            if (settings.showFan) {
                this.drawLineSeries(context, points, step, padding, chartHeight, fanScale, 'fanSpeed', '#06b6d4');
            }
        },

        drawLineSeries(context, points, step, padding, chartHeight, scale, key, color) {
            context.strokeStyle = color;
            context.lineWidth = 2;
            context.beginPath();
            let hasStarted = false;
            points.forEach((point, index) => {
                const value = point[key];
                if (typeof value !== 'number' || !Number.isFinite(value)) {
                    hasStarted = false;
                    return;
                }
                const x = padding + step * index;
                const y = padding + chartHeight - ((value - scale.min) / (scale.max - scale.min)) * chartHeight;
                if (!hasStarted) {
                    context.moveTo(x, y);
                    hasStarted = true;
                } else {
                    context.lineTo(x, y);
                }
            });
            if (hasStarted) {
                context.stroke();
            }
        },

        buildRange(values, padding = 0) {
            if (!values || values.length === 0) {
                return { min: 0, max: 100 };
            }
            let min = Math.min(...values);
            let max = Math.max(...values);
            if (min === max) {
                min -= padding;
                max += padding;
            }
            return {
                min: Math.max(0, min - padding),
                max: Math.max(min + padding, max + padding),
            };
        },

        clearHistoryHover() {
            if (!this.historyHover.visible) {
                return;
            }
            this.historyHover = {
                visible: false,
                left: '0px',
                top: '0px',
                lines: [],
                gpuIndex: null,
            };
        },

        updateHistoryLatestLabel(gpuIndex) {
            const canvas = this.$refs.historyCanvas;
            if (!canvas) {
                return;
            }
            const info = this.getHistoryLatestLabelInfo(gpuIndex, canvas);
            if (!info) {
                this.historyLatestLabel = {
                    visible: false,
                    left: '0px',
                    top: '0px',
                    lines: [],
                    gpuIndex: null,
                };
                return;
            }
            this.historyLatestLabel = {
                visible: true,
                left: `${info.left}px`,
                top: `${info.top}px`,
                lines: info.lines,
                gpuIndex,
            };
        },

        handleHistoryMouseMove(event, gpuIndex) {
            const canvas = event.target;
            if (!canvas) {
                return;
            }
            const info = this.getHistoryHoverInfo(gpuIndex, canvas, event.clientX, event.clientY);
            if (!info) {
                this.clearHistoryHover();
                return;
            }
            this.historyHover = {
                visible: true,
                left: `${info.left}px`,
                top: `${info.top}px`,
                lines: info.lines,
                gpuIndex,
            };
        },

        getHistoryLatestLabelInfo(gpuIndex, canvas) {
            const history = this.gpuHistory[gpuIndex] || [];
            const settings = this.gpuHistorySettings[gpuIndex] || { showTemp: true, showFan: true };
            if (!history.length || (!settings.showTemp && !settings.showFan)) {
                return null;
            }
            const rect = canvas.getBoundingClientRect();
            const width = rect.width || canvas.clientWidth || 240;
            const height = rect.height || canvas.clientHeight || 120;
            const padding = this.historyChartPadding;
            const chartWidth = Math.max(1, width - padding * 2);
            const chartHeight = Math.max(1, height - padding * 2);

            const points = history.slice(-this.gpuHistoryMaxPoints);
            const index = points.length - 1;
            const point = points[index];
            if (!point) {
                return null;
            }

            const tempValues = points
                .map((entry) => entry.temperature)
                .filter((value) => typeof value === 'number' && Number.isFinite(value));
            const tempScale = this.buildRange(tempValues, 10);
            const fanScale = { min: 0, max: 100 };

            const lines = [];
            let anchorY = padding + chartHeight / 2;
            if (settings.showTemp && typeof point.temperature === 'number' && Number.isFinite(point.temperature)) {
                lines.push({ text: `Temp ${Math.round(point.temperature)}°C`, color: '#ef4444' });
                anchorY = padding + chartHeight - ((point.temperature - tempScale.min) / (tempScale.max - tempScale.min)) * chartHeight;
            }
            if (settings.showFan && typeof point.fanSpeed === 'number' && Number.isFinite(point.fanSpeed)) {
                lines.push({ text: `Fan ${point.fanSpeed.toFixed(0)}%`, color: '#06b6d4' });
                if (!settings.showTemp || lines.length === 1) {
                    anchorY = padding + chartHeight - ((point.fanSpeed - fanScale.min) / (fanScale.max - fanScale.min)) * chartHeight;
                }
            }

            if (!lines.length) {
                return null;
            }

            const step = points.length > 1 ? chartWidth / (points.length - 1) : 0;
            const anchorX = padding + step * index;
            const tooltipWidth = 110;
            const tooltipHeight = 38;
            const preferRight = anchorX + tooltipWidth + 12 <= width;
            const left = preferRight
                ? Math.min(width - tooltipWidth, Math.max(4, anchorX + 8))
                : Math.max(4, anchorX - tooltipWidth - 8);
            const top = Math.min(height - tooltipHeight, Math.max(4, anchorY - 18));

            return { left, top, lines };
        },

        getHistoryHoverInfo(gpuIndex, canvas, clientX, clientY) {
            const history = this.gpuHistory[gpuIndex] || [];
            const settings = this.gpuHistorySettings[gpuIndex] || { showTemp: true, showFan: true };
            if (!history.length || (!settings.showTemp && !settings.showFan)) {
                return null;
            }
            const rect = canvas.getBoundingClientRect();
            const x = clientX - rect.left;
            const y = clientY - rect.top;
            const width = rect.width || canvas.clientWidth || 240;
            const height = rect.height || canvas.clientHeight || 120;
            const padding = this.historyChartPadding;
            const chartWidth = Math.max(1, width - padding * 2);
            const chartHeight = Math.max(1, height - padding * 2);

            const points = history.slice(-this.gpuHistoryMaxPoints);
            const step = points.length > 1 ? chartWidth / (points.length - 1) : 0;
            const rawIndex = step > 0 ? Math.round((x - padding) / step) : 0;
            const index = Math.min(points.length - 1, Math.max(0, rawIndex));
            const point = points[index];
            if (!point) {
                return null;
            }

            const tempValues = points
                .map((entry) => entry.temperature)
                .filter((value) => typeof value === 'number' && Number.isFinite(value));
            const tempScale = this.buildRange(tempValues, 10);
            const fanScale = { min: 0, max: 100 };

            const lines = [];
            let anchorY = padding + chartHeight / 2;
            if (settings.showTemp && typeof point.temperature === 'number' && Number.isFinite(point.temperature)) {
                lines.push({ text: `Temp ${Math.round(point.temperature)}°C`, color: '#ef4444' });
                anchorY = padding + chartHeight - ((point.temperature - tempScale.min) / (tempScale.max - tempScale.min)) * chartHeight;
            }
            if (settings.showFan && typeof point.fanSpeed === 'number' && Number.isFinite(point.fanSpeed)) {
                lines.push({ text: `Fan ${point.fanSpeed.toFixed(0)}%`, color: '#06b6d4' });
                if (!settings.showTemp || lines.length === 1) {
                    anchorY = padding + chartHeight - ((point.fanSpeed - fanScale.min) / (fanScale.max - fanScale.min)) * chartHeight;
                }
            }

            if (!lines.length) {
                return null;
            }

            const anchorX = padding + step * index;
            const tooltipWidth = 110;
            const tooltipHeight = 38;
            const left = Math.min(width - tooltipWidth, Math.max(4, anchorX + 10));
            const top = Math.min(height - tooltipHeight, Math.max(4, anchorY - 18));

            return { left, top, lines };
        },

        formatPercent(value, decimals = 0) {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                return '—';
            }
            const fixed = value.toFixed(decimals);
            return `${fixed}%`;
        },

        formatTemperature(value) {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                return '—';
            }
            return `${Math.round(value)}°C`;
        },

        formatBytes(value) {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                return '—';
            }
            if (value <= 0) {
                return '0 B';
            }
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            const exponent = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
            const scaled = value / 1024 ** exponent;
            const formatted = scaled >= 100 ? Math.round(scaled) : scaled.toFixed(1);
            return `${formatted} ${units[exponent]}`;
        },

        getVramPercent(gpu) {
            if (!gpu) {
                return null;
            }
            if (typeof gpu.memory_used_percent === 'number' && Number.isFinite(gpu.memory_used_percent)) {
                return gpu.memory_used_percent;
            }
            if (
                typeof gpu.memory_used_bytes === 'number' &&
                Number.isFinite(gpu.memory_used_bytes) &&
                typeof gpu.memory_total_bytes === 'number' &&
                Number.isFinite(gpu.memory_total_bytes) &&
                gpu.memory_total_bytes > 0
            ) {
                return (gpu.memory_used_bytes / gpu.memory_total_bytes) * 100;
            }
            return null;
        },

        getVramDetails(gpu) {
            if (!gpu) {
                return '—';
            }
            if (
                typeof gpu.memory_used_bytes !== 'number' ||
                !Number.isFinite(gpu.memory_used_bytes) ||
                typeof gpu.memory_total_bytes !== 'number' ||
                !Number.isFinite(gpu.memory_total_bytes)
            ) {
                return '—';
            }
            return `${this.formatBytes(gpu.memory_used_bytes)} / ${this.formatBytes(gpu.memory_total_bytes)}`;
        },

        getVramColor(percent) {
            if (percent >= 90) {
                return '#ef4444';
            }
            if (percent >= 75) {
                return '#f59e0b';
            }
            return '#10b981';
        },

        getUtilizationWidth(value) {
            if (typeof value !== 'number' || !Number.isFinite(value)) {
                return '0%';
            }
            const clamped = Math.max(0, Math.min(100, value));
            return `${clamped.toFixed(1)}%`;
        },

        getUtilizationClass(percent) {
            if (typeof percent !== 'number' || !Number.isFinite(percent)) {
                return 'util-low';
            }
            if (percent >= 80) {
                return 'util-high';
            }
            if (percent >= 40) {
                return 'util-medium';
            }
            return 'util-low';
        },

        getTempThresholdLabel(gpu) {
            const slowdown = gpu?.temperature_threshold_slowdown;
            const shutdown = gpu?.temperature_threshold_shutdown;
            if (
                typeof slowdown !== 'number' ||
                !Number.isFinite(slowdown) ||
                typeof shutdown !== 'number' ||
                !Number.isFinite(shutdown)
            ) {
                return '—';
            }
            return `Slowdown ${Math.round(slowdown)}°C · Shutdown ${Math.round(shutdown)}°C`;
        },

        getFanAnimationStyle(speedPercent) {
            if (typeof speedPercent !== 'number' || !Number.isFinite(speedPercent) || speedPercent <= 0) {
                return 'animation-duration: 0s;';
            }
            const clamped = Math.max(0, Math.min(100, speedPercent));
            const duration = Math.max(0.4, 4 - (clamped / 100) * 3.5);
            return `animation-duration: ${duration.toFixed(2)}s;`;
        },

        getTempClass(gpu) {
            const temp = gpu?.temperature_celsius;
            if (typeof temp !== 'number' || !Number.isFinite(temp)) {
                return 'temp-cool';
            }
            if (gpu.is_thermal_throttling) {
                return 'temp-critical';
            }
            if (gpu.temperature_threshold_slowdown && temp >= gpu.temperature_threshold_slowdown) {
                return 'temp-critical';
            }
            if (gpu.temperature_threshold_slowdown && temp >= gpu.temperature_threshold_slowdown - 10) {
                return 'temp-warning';
            }
            if (temp >= 75) {
                return 'temp-warm';
            }
            return 'temp-cool';
        },

        getHistorySeriesLabel(gpuIndex) {
            const settings = this.gpuHistorySettings[gpuIndex] || { showTemp: false, showFan: false };
            if (settings.showTemp && settings.showFan) {
                return 'Temp+Fan';
            }
            if (settings.showTemp) {
                return 'Temp';
            }
            if (settings.showFan) {
                return 'Fan';
            }
            return 'None';
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
