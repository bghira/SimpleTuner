global.fetch = jest.fn();

require('../../simpletuner/static/js/training_metrics_chart.js');
require('../../simpletuner/static/js/training_metrics_component.js');

describe('training metrics component', () => {
    let state;

    beforeEach(() => {
        fetch.mockReset();
        state = window.trainingMetricsState();
        state.$nextTick = (callback) => callback();
        state.renderTrainingMetricsChart = jest.fn();
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

    test('filters validation media by prompt and step', () => {
        state.trainingRunData = {
            media: [
                { label: 'portrait', step: 10, index: 0, path: 'a.webp' },
                { label: 'portrait', step: 20, index: 1, path: 'c.webp' },
                { label: 'portrait', step: 20, index: 0, path: 'b.webp' },
                { label: 'landscape', step: 20, index: 0, path: 'd.webp' },
            ],
        };

        state.selectDefaultTrainingMedia();

        expect(state.selectedTrainingMediaLabel).toBe('landscape');
        expect(state.selectedTrainingMediaStep).toBe(20);
        state.selectTrainingMediaLabel('portrait');
        expect(state.trainingMediaSteps()).toEqual([10, 20]);
        expect(state.selectedTrainingMedia().map((item) => item.path)).toEqual(['b.webp', 'c.webp']);
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
