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
        expect(state.selectedTrainingMetrics).toEqual(['learning_rate', 'loss']);
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

        expect(state.selectedTrainingMetrics).toEqual(['train_loss']);
        expect(state.trainingMetricsLoading).toBe(false);
    });

    test('sets chart x-axis mode when rendering', () => {
        state.renderTrainingMetricsChart.mockClear();

        state.setTrainingXAxis('minutes');

        expect(state.selectedTrainingXAxis).toBe('minutes');
        expect(state.renderTrainingMetricsChart).toHaveBeenCalledTimes(1);
    });

    test('prefers validation loss and timing metrics by default', () => {
        const selected = window.TrainingMetricsCharts.defaultMetricNames([
            'learning_rate',
            'loss/val/pooled',
            'seconds_per_step',
            'train_loss',
            'z_metric',
        ]);

        expect(selected).toEqual(['train_loss', 'loss/val/pooled', 'seconds_per_step', 'learning_rate']);
    });

    test('limits the chart to eight selected metrics', () => {
        state.selectedTrainingMetrics = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

        state.toggleTrainingMetric('i');
        expect(state.selectedTrainingMetrics).toHaveLength(8);

        state.toggleTrainingMetric('a');
        state.toggleTrainingMetric('i');
        expect(state.selectedTrainingMetrics).toContain('i');
        expect(state.renderTrainingMetricsChart).toHaveBeenCalledTimes(3);
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
