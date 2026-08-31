global.fetch = jest.fn();

require('../../simpletuner/static/js/training_metrics_chart.js');
require('../../simpletuner/static/js/training_metrics_component.js');

describe('training metrics component', () => {
    let state;

    beforeEach(() => {
        fetch.mockReset();
        state = window.trainingMetricsState();
        state.$refs = {};
        state.$nextTick = (callback) => callback();
        state.renderTrainingMetricsChart = jest.fn();
        state.renderTrainingMetricsCharts = jest.fn();
        state.renderTimestepDistributionChart = jest.fn();
    });

    test('loads run summaries and selects the newest run', async () => {
        fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    runs: [
                        { environment: 'anima', run_name: 'Anima smoke' },
                        { environment: 'flux', run_name: 'Flux run' },
                    ],
                }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    run: { environment: 'anima', last_step: 20 },
                    available_metrics: ['loss', 'learning_rate'],
                    records: [{ step: 20, metrics: { loss: 0.5, learning_rate: 0.0001 } }],
                    media: [],
                }),
            });

        await state.loadTrainingRuns();

        expect(state.selectedTrainingEnvironment).toBe('anima');
        expect(state.trainingRunData.run.last_step).toBe(20);
        expect(state.trainingCharts).toHaveLength(1);
        expect(state.trainingCharts[0].name).toBe('Training metrics');
        expect(state.selectedTrainingMetrics).toEqual(['loss']);
        expect(fetch).toHaveBeenNthCalledWith(1, '/api/metrics/training/runs');
        expect(fetch).toHaveBeenNthCalledWith(2, '/api/metrics/training/runs/anima?max_points=4000');
    });

    test('groups validation media by prompt for the selected step', () => {
        state.trainingRunData = {
            media: [
                { label: 'portrait', step: 10, index: 0, path: 'a.webp' },
                { label: 'portrait', step: 20, index: 1, path: 'c.webp' },
                { label: 'portrait', step: 20, index: 0, path: 'b.webp' },
                { label: 'landscape', step: 20, index: 0, path: 'd.webp' },
            ],
        };

        state.selectDefaultTrainingMedia();

        expect(state.selectedTrainingMediaStep).toBe(20);
        expect(state.trainingMediaSteps()).toEqual([10, 20]);
        expect(state.selectedTrainingMedia().map((item) => item.path)).toEqual(['d.webp', 'b.webp', 'c.webp']);
        expect(state.selectedTrainingMediaGroups()).toEqual([
            { label: 'landscape', items: [{ label: 'landscape', step: 20, index: 0, path: 'd.webp' }] },
            {
                label: 'portrait',
                items: [
                    { label: 'portrait', step: 20, index: 0, path: 'b.webp' },
                    { label: 'portrait', step: 20, index: 1, path: 'c.webp' },
                ],
            },
        ]);
    });

    test('preserves metric selection while silently refreshing a running run', async () => {
        state.selectedTrainingEnvironment = 'anima';
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['train_loss'], metricSearch: '', smoothing: 0 }];
        state.selectedTrainingMetrics = ['train_loss'];
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                run: { environment: 'anima', status: 'running', last_step: 21 },
                available_metrics: ['learning_rate', 'train_loss', 'seconds_per_step'],
                records: [{ step: 21, metrics: { train_loss: 0.4, seconds_per_step: 1.2 } }],
                media: [],
            }),
        });

        await state.loadTrainingRun({ silent: true, preserveSelections: true });

        expect(state.trainingCharts[0].metrics).toEqual(['train_loss']);
        expect(state.selectedTrainingMetrics).toEqual(['train_loss']);
        expect(state.trainingMetricsLoading).toBe(false);
    });

    test('sets chart x-axis mode when rendering', () => {
        state.renderTrainingMetricsCharts.mockClear();

        state.setTrainingXAxis('minutes');

        expect(state.selectedTrainingXAxis).toBe('minutes');
        expect(state.renderTrainingMetricsCharts).toHaveBeenCalledTimes(1);
    });

    test('prefers one loss metric by default', () => {
        const selected = window.TrainingMetricsCharts.defaultMetricNames([
            'learning_rate',
            'loss/val/pooled',
            'seconds_per_step',
            'train_loss',
            'z_metric',
        ]);

        expect(selected).toEqual(['train_loss']);
    });

    test('limits the chart to eight selected metrics', () => {
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'], metricSearch: '' }];
        state.selectedTrainingMetrics = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

        state.toggleTrainingMetric('i');
        expect(state.selectedTrainingMetrics).toHaveLength(8);

        state.toggleTrainingMetric('a');
        state.toggleTrainingMetric('i');
        expect(state.selectedTrainingMetrics).toContain('i');
        expect(state.renderTrainingMetricsCharts).toHaveBeenCalledTimes(3);
    });

    test('applies saved environment chart layout', async () => {
        state.selectedTrainingEnvironment = 'anima';
        state.trainingMetricLayouts = {
            templates: [],
            environments: {
                anima: {
                    template: 'Loss detail',
                    xAxis: 'minutes',
                    showTimestepChart: true,
                    mediaPanelCollapsed: true,
                    charts: [
                        { kind: 'scalar', name: 'Loss', metrics: ['train_loss', 'loss/val'], smoothing: 0.25 },
                        { kind: 'scalar', name: 'GPU', metrics: ['system/gpu/0/memory_used_gb'], smoothing: 0.5 },
                    ],
                },
            },
        };
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                run: { environment: 'anima', status: 'running', last_step: 21 },
                available_metrics: ['train_loss', 'loss/val', 'system/gpu/0/memory_used_gb'],
                records: [{ step: 21, metrics: { train_loss: 0.4 } }],
                media: [],
            }),
        });

        await state.loadTrainingRun();

        expect(state.selectedTrainingLayoutName).toBe('Loss detail');
        expect(state.selectedTrainingXAxis).toBe('minutes');
        expect(state.showTrainingTimestepChart).toBe(true);
        expect(state.trainingMediaPanelCollapsed).toBe(true);
        expect(state.trainingCharts.map((chart) => chart.metrics)).toEqual([
            ['train_loss', 'loss/val'],
            ['system/gpu/0/memory_used_gb'],
        ]);
        expect(state.trainingCharts.map((chart) => chart.smoothing)).toEqual([0.25, 0.5]);
    });

    test('timestep chart expanded state is saved with the environment layout', () => {
        state.selectedTrainingEnvironment = 'anima';
        state.trainingMetricLayoutsLoaded = true;
        state.trainingRunData = { available_metrics: ['train_loss'] };
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['train_loss'], metricSearch: '', smoothing: 0 }];
        state.$nextTick = (callback) => callback();

        state.toggleTrainingTimestepChart();

        expect(state.showTrainingTimestepChart).toBe(true);
        expect(state.trainingMetricLayouts.environments.anima.showTimestepChart).toBe(true);
        expect(state.renderTimestepDistributionChart).toHaveBeenCalled();
    });

    test('validation media collapsed state is saved with the environment layout', async () => {
        state.selectedTrainingEnvironment = 'anima';
        state.trainingMetricLayoutsLoaded = true;
        state.trainingRunData = { available_metrics: ['train_loss'] };
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['train_loss'], metricSearch: '', smoothing: 0 }];
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                templates: [],
                environments: {
                    anima: {
                        template: 'Default',
                        xAxis: 'step',
                        showTimestepChart: false,
                        mediaPanelCollapsed: true,
                        charts: [{ kind: 'scalar', name: 'Loss', metrics: ['train_loss'], smoothing: 0 }],
                    },
                },
            }),
        });

        await state.toggleTrainingMediaPanel();

        expect(state.trainingMediaPanelCollapsed).toBe(true);
        expect(state.trainingMetricLayouts.environments.anima.mediaPanelCollapsed).toBe(true);
        expect(state.renderTrainingMetricsCharts).toHaveBeenCalled();
        expect(fetch).toHaveBeenCalledWith(
            '/api/webui/ui-state/training-metrics',
            expect.objectContaining({
                method: 'POST',
                body: expect.stringContaining('"mediaPanelCollapsed":true'),
            }),
        );
    });

    test('lightbox can switch between steps for the same validation output', () => {
        state.trainingRunData = {
            media: [
                { type: 'image', label: 'portrait', step: 10, index: 0, path: 'step-10-a.webp', url: '/a.webp' },
                { type: 'image', label: 'portrait', step: 10, index: 1, path: 'step-10-b.webp', url: '/b.webp' },
                { type: 'image', label: 'portrait', step: 20, index: 0, path: 'step-20-a.webp', url: '/c.webp' },
            ],
        };
        state.openTrainingMediaLightbox(state.trainingRunData.media[0]);

        state.setTrainingLightboxImageOffset(1);
        expect(state.currentTrainingLightboxImage().item.path).toBe('step-10-b.webp');

        state.setTrainingLightboxImageOffset(-1);
        state.setTrainingLightboxStepOffset(1);
        expect(state.currentTrainingLightboxImage().item.path).toBe('step-20-a.webp');
        expect(state.selectedTrainingMediaStep).toBe(20);

        state.setTrainingLightboxStepIndex(0);
        expect(state.currentTrainingLightboxImage().item.path).toBe('step-10-a.webp');
        expect(state.trainingLightboxStepLabel()).toBe('Step 10');
    });

    test('validation media step slider maps indices to recorded checkpoint steps', () => {
        state.trainingRunData = {
            media: [
                { type: 'image', step: 5, path: 'step-5.webp' },
                { type: 'image', step: 20, path: 'step-20.webp' },
                { type: 'image', step: 125, path: 'step-125.webp' },
            ],
        };
        state.selectDefaultTrainingMedia();

        expect(state.trainingMediaStepIndex()).toBe(2);
        expect(state.selectedTrainingMediaStepLabel()).toBe('Step 125');

        state.setTrainingMediaStepIndex(1);
        expect(state.selectedTrainingMediaStep).toBe(20);

        state.setTrainingMediaStepIndex(99);
        expect(state.selectedTrainingMediaStep).toBe(125);
    });

    test('save layout modal prompts for a layout name', async () => {
        state.selectedTrainingLayoutName = 'Loss detail';
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['train_loss'], metricSearch: '', smoothing: 0 }];
        state.trainingRunData = { available_metrics: ['train_loss'] };
        state.trainingMetricLayoutsLoaded = true;
        fetch.mockResolvedValueOnce({
            ok: true,
            json: async () => ({
                templates: [{ name: 'Loss detail', charts: [{ kind: 'scalar', name: 'Loss', metrics: ['train_loss'], smoothing: 0 }] }],
                environments: {},
            }),
        });

        state.openTrainingLayoutModal();
        expect(state.trainingLayoutModalOpen).toBe(true);
        expect(state.trainingLayoutNameInput).toBe('Loss detail');

        await state.confirmTrainingLayoutSave();
        expect(state.trainingLayoutModalOpen).toBe(false);
        expect(fetch).toHaveBeenCalledWith(
            '/api/webui/ui-state/training-metrics',
            expect.objectContaining({ method: 'POST' }),
        );
    });

    test('chart smoothing is clamped and rendered as a chart option', () => {
        state.trainingRunData = {
            records: [{ step: 1, metrics: { train_loss: 1 } }],
            available_metrics: ['train_loss'],
        };
        state.trainingCharts = [{ id: 'chart-a', kind: 'scalar', name: 'Loss', metrics: ['train_loss'], metricSearch: '', smoothing: 0 }];
        state.renderTrainingMetricsCharts.mockClear();

        state.setTrainingChartSmoothing(state.trainingCharts[0], 1.5);

        expect(state.trainingCharts[0].smoothing).toBe(0.99);
        expect(state.formatTrainingSmoothing(state.trainingCharts[0].smoothing)).toBe('99%');
        expect(state.renderTrainingMetricsCharts).toHaveBeenCalledTimes(1);
    });

    test('builds per-chart legend entries from selected metrics', () => {
        const chart = { metrics: ['train_loss', 'loss/val'] };

        expect(state.trainingChartLegend(chart)).toEqual([
            { name: 'train_loss', color: '#38bdf8' },
            { name: 'loss/val', color: '#f59e0b' },
        ]);
    });

    test('surfaces API failures without retaining stale run data', async () => {
        state.selectedTrainingEnvironment = 'anima';
        state.trainingRunData = { run: { environment: 'old' } };
        fetch.mockResolvedValueOnce({ ok: false, status: 500 });

        await state.loadTrainingRun();

        expect(state.trainingRunData).toBeNull();
        expect(state.trainingMetricsError).toContain('HTTP 500');
        expect(state.trainingMetricsLoading).toBe(false);
    });
});
