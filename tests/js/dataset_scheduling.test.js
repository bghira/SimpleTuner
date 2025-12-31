/**
 * Tests for dataset scheduling functionality.
 *
 * Covers _scheduleMode initialization based on start_epoch and start_step values.
 */

// Helper to create a minimal dataset runtime state initializer
// This mirrors the logic in dataloader-section-component.js
const initializeScheduleMode = (dataset) => {
    if (dataset._scheduleMode === undefined) {
        if (dataset.start_step !== undefined && dataset.start_step !== null && dataset.start_step > 0) {
            dataset._scheduleMode = 'step';
        } else if (dataset.start_epoch !== undefined && dataset.start_epoch !== null && dataset.start_epoch > 1) {
            dataset._scheduleMode = 'epoch';
        } else {
            dataset._scheduleMode = 'none';
        }
    }
    return dataset;
};

describe('Dataset Scheduling Mode Initialization', () => {
    describe('_scheduleMode detection', () => {
        test('defaults to none when no scheduling values set', () => {
            const dataset = { id: 'test', dataset_type: 'image' };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('defaults to none when start_epoch is 1', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 1 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('defaults to none when start_step is 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: 0 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('sets epoch mode when start_epoch > 1', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 5 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
        });

        test('sets step mode when start_step > 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: 100 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('step');
        });

        test('step mode takes precedence when both are set', () => {
            // If both are set, step mode should win since it's checked first
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 5, start_step: 100 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('step');
        });

        test('does not override existing _scheduleMode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 5, _scheduleMode: 'none' };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('handles null start_epoch as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: null };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('handles null start_step as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: null };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('handles undefined values correctly', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: undefined, start_step: undefined };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });
    });

    describe('edge cases', () => {
        test('handles start_epoch of 2 as epoch mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 2 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
        });

        test('handles start_step of 1 as step mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: 1 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('step');
        });

        test('handles negative start_epoch as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: -1 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('handles negative start_step as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: -10 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });

        test('handles float values for start_epoch', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 2.5 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
        });

        test('handles float values for start_step', () => {
            const dataset = { id: 'test', dataset_type: 'image', start_step: 50.5 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('step');
        });

        test('handles zero start_epoch as none mode', () => {
            // start_epoch of 0 or 1 should be treated as immediate
            const dataset = { id: 'test', dataset_type: 'image', start_epoch: 0 };
            initializeScheduleMode(dataset);

            expect(dataset._scheduleMode).toBe('none');
        });
    });

    describe('multiple datasets', () => {
        test('each dataset gets independent schedule mode', () => {
            const datasets = [
                { id: 'immediate', dataset_type: 'image' },
                { id: 'epoch-delayed', dataset_type: 'image', start_epoch: 3 },
                { id: 'step-delayed', dataset_type: 'image', start_step: 500 },
            ];

            datasets.forEach(initializeScheduleMode);

            expect(datasets[0]._scheduleMode).toBe('none');
            expect(datasets[1]._scheduleMode).toBe('epoch');
            expect(datasets[2]._scheduleMode).toBe('step');
        });
    });
});
