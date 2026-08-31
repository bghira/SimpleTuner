(function (global) {
    'use strict';

    const COLORS = ['#38bdf8', '#f59e0b', '#22c55e', '#f472b6', '#a78bfa', '#fb7185', '#2dd4bf', '#eab308'];
    const PREFERRED_METRICS = [
        'train_loss',
        'optimization_loss',
        'diffusion_loss',
        'loss/val/pooled',
        'loss/val',
        'iteration_step_time_seconds',
        'seconds_per_step',
        'learning_rate',
    ];

    function metricNames(records) {
        const names = new Set();
        (records || []).forEach((record) => {
            Object.keys(record.metrics || {}).forEach((name) => names.add(name));
        });
        return Array.from(names).sort((left, right) => left.localeCompare(right));
    }

    function defaultMetricNames(names, limit = 4) {
        const selected = [];
        PREFERRED_METRICS.forEach((preferred) => {
            names.forEach((name) => {
                if (selected.length >= limit) return;
                if (selected.includes(name)) return;
                if (name === preferred || name.startsWith(`${preferred}/`)) selected.push(name);
            });
        });
        for (const name of names) {
            if (selected.length >= limit) break;
            if (!selected.includes(name) && !name.includes('learning_rate')) selected.push(name);
        }
        return selected.slice(0, limit);
    }

    function latestMetricValues(records) {
        const latest = {};
        (records || []).forEach((record) => {
            Object.entries(record.metrics || {}).forEach(([name, value]) => {
                if (typeof value === 'number' && Number.isFinite(value)) latest[name] = value;
            });
        });
        return latest;
    }

    function formatMetricValue(value) {
        if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
        const absolute = Math.abs(value);
        if (absolute !== 0 && (absolute >= 10000 || absolute < 0.001)) return value.toExponential(3);
        return value.toLocaleString(undefined, { maximumFractionDigits: 6 });
    }

    function timestampMs(record) {
        if (!record || typeof record.timestamp !== 'string') return null;
        const timestamp = Date.parse(record.timestamp);
        return Number.isFinite(timestamp) ? timestamp : null;
    }

    function elapsedMinutes(record, startTimestamp) {
        const timestamp = timestampMs(record);
        if (timestamp === null || startTimestamp === null) return null;
        return (timestamp - startTimestamp) / 60000;
    }

    function formatXAxisValue(value, mode) {
        if (mode === 'minutes') return `${formatMetricValue(value)} min`;
        return Math.round(value).toLocaleString();
    }

    class TrainingMetricsChart {
        constructor(canvas, options = {}) {
            this.canvas = canvas;
            this.records = [];
            this.metrics = [];
            this.xAxisMode = options.xAxisMode === 'minutes' ? 'minutes' : 'step';
            this.options = options;
            this.geometry = null;
            this.hoverIndex = null;
            this._resizeHandler = () => this.draw();
            this._moveHandler = (event) => this._handlePointer(event);
            this._leaveHandler = () => {
                this.hoverIndex = null;
                this.draw();
                if (this.options.onHover) this.options.onHover(null);
            };
            global.addEventListener('resize', this._resizeHandler);
            canvas.addEventListener('pointermove', this._moveHandler);
            canvas.addEventListener('pointerleave', this._leaveHandler);
        }

        setData(records, metrics, options = {}) {
            this.records = Array.isArray(records) ? records : [];
            this.metrics = Array.isArray(metrics) ? metrics.slice(0, COLORS.length) : [];
            if (options.xAxisMode) this.xAxisMode = options.xAxisMode === 'minutes' ? 'minutes' : 'step';
            this.hoverIndex = null;
            this.draw();
        }

        destroy() {
            global.removeEventListener('resize', this._resizeHandler);
            this.canvas.removeEventListener('pointermove', this._moveHandler);
            this.canvas.removeEventListener('pointerleave', this._leaveHandler);
        }

        draw() {
            const canvas = this.canvas;
            const context = canvas.getContext('2d');
            if (!context) return;

            const width = Math.max(320, canvas.clientWidth || 800);
            const height = Math.max(220, canvas.clientHeight || 360);
            const dpr = global.devicePixelRatio || 1;
            if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
                canvas.width = Math.round(width * dpr);
                canvas.height = Math.round(height * dpr);
            }
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, width, height);

            const timestampValues = this.records.map(timestampMs).filter((value) => value !== null);
            const startTimestamp = timestampValues.length ? Math.min(...timestampValues) : null;
            const activeXAxisMode = this.xAxisMode === 'minutes' && startTimestamp !== null ? 'minutes' : 'step';
            const points = this.records
                .map((record) => ({
                    record,
                    xValue: activeXAxisMode === 'minutes'
                        ? elapsedMinutes(record, startTimestamp)
                        : Number(record.step),
                }))
                .filter((point) => Number.isFinite(point.xValue));
            const values = [];
            points.forEach((point) => {
                this.metrics.forEach((name) => {
                    const value = point.record.metrics && point.record.metrics[name];
                    if (typeof value === 'number' && Number.isFinite(value)) values.push(value);
                });
            });
            if (!points.length || !values.length || !this.metrics.length) {
                this._drawEmpty(context, width, height);
                this.geometry = null;
                return;
            }

            const padding = { left: 64, right: 20, top: 20, bottom: 42 };
            const plotWidth = width - padding.left - padding.right;
            const plotHeight = height - padding.top - padding.bottom;
            const xValues = points.map((point) => point.xValue);
            let minX = Math.min(...xValues);
            let maxX = Math.max(...xValues);
            let minValue = Math.min(...values);
            let maxValue = Math.max(...values);
            if (minX === maxX) maxX = minX + 1;
            if (minValue === maxValue) {
                const offset = Math.abs(minValue) * 0.05 || 1;
                minValue -= offset;
                maxValue += offset;
            }
            const valuePadding = (maxValue - minValue) * 0.08;
            minValue -= valuePadding;
            maxValue += valuePadding;

            const xForValue = (xValue) => padding.left + ((xValue - minX) / (maxX - minX)) * plotWidth;
            const yForValue = (value) => padding.top + (1 - (value - minValue) / (maxValue - minValue)) * plotHeight;
            this.geometry = { points, padding, plotWidth, plotHeight, minX, maxX, minValue, maxValue, xForValue, yForValue, xAxisMode: activeXAxisMode };

            this._drawGrid(context, width, height, this.geometry);
            this.metrics.forEach((name, metricIndex) => {
                context.strokeStyle = COLORS[metricIndex % COLORS.length];
                context.lineWidth = 2;
                context.lineJoin = 'round';
                context.beginPath();
                let started = false;
                points.forEach((point) => {
                    const value = point.record.metrics && point.record.metrics[name];
                    if (typeof value !== 'number' || !Number.isFinite(value)) {
                        started = false;
                        return;
                    }
                    const x = xForValue(point.xValue);
                    const y = yForValue(value);
                    if (!started) {
                        context.moveTo(x, y);
                        started = true;
                    } else {
                        context.lineTo(x, y);
                    }
                });
                context.stroke();
            });

            if (this.hoverIndex !== null && points[this.hoverIndex]) {
                const point = points[this.hoverIndex];
                const x = xForValue(point.xValue);
                context.strokeStyle = 'rgba(148, 163, 184, 0.65)';
                context.lineWidth = 1;
                context.beginPath();
                context.moveTo(x, padding.top);
                context.lineTo(x, padding.top + plotHeight);
                context.stroke();
                this.metrics.forEach((name, metricIndex) => {
                    const value = point.record.metrics && point.record.metrics[name];
                    if (typeof value !== 'number' || !Number.isFinite(value)) return;
                    context.fillStyle = COLORS[metricIndex % COLORS.length];
                    context.beginPath();
                    context.arc(x, yForValue(value), 4, 0, Math.PI * 2);
                    context.fill();
                });
            }
        }

        _drawEmpty(context, width, height) {
            context.fillStyle = '#94a3b8';
            context.font = '14px system-ui, sans-serif';
            context.textAlign = 'center';
            context.fillText('No scalar metrics selected', width / 2, height / 2);
        }

        _drawGrid(context, width, height, geometry) {
            const { padding, plotWidth, plotHeight, minX, maxX, minValue, maxValue, xAxisMode } = geometry;
            context.font = '12px system-ui, sans-serif';
            context.fillStyle = '#94a3b8';
            context.strokeStyle = 'rgba(148, 163, 184, 0.18)';
            context.lineWidth = 1;
            context.textAlign = 'right';
            context.textBaseline = 'middle';
            for (let index = 0; index <= 4; index += 1) {
                const ratio = index / 4;
                const y = padding.top + ratio * plotHeight;
                const value = maxValue - ratio * (maxValue - minValue);
                context.beginPath();
                context.moveTo(padding.left, y);
                context.lineTo(padding.left + plotWidth, y);
                context.stroke();
                context.fillText(formatMetricValue(value), padding.left - 10, y);
            }

            context.textAlign = 'center';
            context.textBaseline = 'top';
            for (let index = 0; index <= 4; index += 1) {
                const ratio = index / 4;
                const x = padding.left + ratio * plotWidth;
                const value = minX + ratio * (maxX - minX);
                context.fillText(formatXAxisValue(value, xAxisMode), x, height - padding.bottom + 12);
            }
        }

        _handlePointer(event) {
            if (!this.geometry) return;
            const rect = this.canvas.getBoundingClientRect();
            const x = event.clientX - rect.left;
            const { points, xForValue } = this.geometry;
            let nearestIndex = 0;
            let nearestDistance = Number.POSITIVE_INFINITY;
            points.forEach((record, index) => {
                const distance = Math.abs(xForValue(record.xValue) - x);
                if (distance < nearestDistance) {
                    nearestDistance = distance;
                    nearestIndex = index;
                }
            });
            if (this.hoverIndex !== nearestIndex) {
                this.hoverIndex = nearestIndex;
                this.draw();
            }
            if (this.options.onHover) {
                const record = points[nearestIndex];
                this.options.onHover({
                    record: record.record,
                    x: xForValue(record.xValue),
                    xValue: record.xValue,
                    xAxisMode: this.geometry.xAxisMode,
                    metrics: this.metrics.map((name, index) => ({
                        name,
                        value: record.record.metrics ? record.record.metrics[name] : undefined,
                        color: COLORS[index % COLORS.length],
                    })),
                });
            }
        }
    }

    class TimestepDistributionChart {
        constructor(canvas) {
            this.canvas = canvas;
            this.records = [];
            this._resizeHandler = () => this.draw();
            global.addEventListener('resize', this._resizeHandler);
        }

        setData(records) {
            this.records = Array.isArray(records) ? records : [];
            this.draw();
        }

        destroy() {
            global.removeEventListener('resize', this._resizeHandler);
        }

        draw() {
            const context = this.canvas.getContext('2d');
            if (!context) return;
            const width = Math.max(320, this.canvas.clientWidth || 800);
            const height = Math.max(220, this.canvas.clientHeight || 320);
            const dpr = global.devicePixelRatio || 1;
            this.canvas.width = Math.round(width * dpr);
            this.canvas.height = Math.round(height * dpr);
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, width, height);

            const points = [];
            this.records.forEach((record) => {
                const step = Number(record.step);
                if (!Number.isFinite(step) || !Array.isArray(record.timesteps)) return;
                record.timesteps.forEach((timestep) => {
                    const value = Number(timestep);
                    if (Number.isFinite(value)) points.push({ step, timestep: value });
                });
            });
            if (!points.length) {
                context.fillStyle = '#94a3b8';
                context.font = '14px system-ui, sans-serif';
                context.textAlign = 'center';
                context.fillText('No timestep samples recorded', width / 2, height / 2);
                return;
            }

            const padding = { left: 64, right: 20, top: 20, bottom: 42 };
            const plotWidth = width - padding.left - padding.right;
            const plotHeight = height - padding.top - padding.bottom;
            let minStep = Math.min(...points.map((point) => point.step));
            let maxStep = Math.max(...points.map((point) => point.step));
            let minTimestep = Math.min(...points.map((point) => point.timestep));
            let maxTimestep = Math.max(...points.map((point) => point.timestep));
            if (minStep === maxStep) maxStep = minStep + 1;
            if (minTimestep === maxTimestep) maxTimestep = minTimestep + 1;
            const xForStep = (step) => padding.left + ((step - minStep) / (maxStep - minStep)) * plotWidth;
            const yForTimestep = (timestep) =>
                padding.top + (1 - (timestep - minTimestep) / (maxTimestep - minTimestep)) * plotHeight;

            context.strokeStyle = 'rgba(148, 163, 184, 0.18)';
            context.lineWidth = 1;
            context.font = '12px system-ui, sans-serif';
            context.fillStyle = '#94a3b8';
            context.textAlign = 'right';
            context.textBaseline = 'middle';
            for (let index = 0; index <= 4; index += 1) {
                const ratio = index / 4;
                const y = padding.top + ratio * plotHeight;
                const value = maxTimestep - ratio * (maxTimestep - minTimestep);
                context.beginPath();
                context.moveTo(padding.left, y);
                context.lineTo(padding.left + plotWidth, y);
                context.stroke();
                context.fillText(formatMetricValue(value), padding.left - 10, y);
            }
            context.textAlign = 'center';
            context.textBaseline = 'top';
            for (let index = 0; index <= 4; index += 1) {
                const ratio = index / 4;
                const x = padding.left + ratio * plotWidth;
                const step = Math.round(minStep + ratio * (maxStep - minStep));
                context.fillText(step.toLocaleString(), x, height - padding.bottom + 12);
            }

            context.fillStyle = 'rgba(56, 189, 248, 0.45)';
            points.forEach((point) => {
                context.beginPath();
                context.arc(xForStep(point.step), yForTimestep(point.timestep), 2, 0, Math.PI * 2);
                context.fill();
            });
        }
    }

    global.TrainingMetricsCharts = {
        COLORS,
        TimestepDistributionChart,
        TrainingMetricsChart,
        defaultMetricNames,
        formatXAxisValue,
        formatMetricValue,
        latestMetricValues,
        metricNames,
    };
})(typeof window !== 'undefined' ? window : globalThis);
