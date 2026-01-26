/**
 * Tests for dataset scheduling functionality.
 *
 * Covers _scheduleMode and _endScheduleMode initialization based on
 * start_epoch, start_step, end_epoch, and end_step values.
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

// Helper to initialize end schedule mode
// This mirrors the logic in dataloader-section-component.js
const initializeEndScheduleMode = (dataset) => {
    if (dataset._endScheduleMode === undefined) {
        if (dataset.end_step !== undefined && dataset.end_step !== null && dataset.end_step > 0) {
            dataset._endScheduleMode = 'step';
        } else if (dataset.end_epoch !== undefined && dataset.end_epoch !== null && dataset.end_epoch > 0) {
            dataset._endScheduleMode = 'epoch';
        } else {
            dataset._endScheduleMode = 'none';
        }
    }
    return dataset;
};

// Combined initializer for both start and end modes
const initializeAllScheduleModes = (dataset) => {
    initializeScheduleMode(dataset);
    initializeEndScheduleMode(dataset);
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

describe('Dataset End Scheduling Mode Initialization', () => {
    describe('_endScheduleMode detection', () => {
        test('defaults to none when no end values set', () => {
            const dataset = { id: 'test', dataset_type: 'image' };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('defaults to none when end_epoch is null', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: null };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('defaults to none when end_step is null', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: null };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('defaults to none when end_epoch is 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 0 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('defaults to none when end_step is 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: 0 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('sets epoch mode when end_epoch > 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 5 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('sets step mode when end_step > 0', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: 300 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('step');
        });

        test('step mode takes precedence when both are set', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 5, end_step: 300 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('step');
        });

        test('does not override existing _endScheduleMode', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 5, _endScheduleMode: 'none' };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('handles negative end_epoch as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: -1 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });

        test('handles negative end_step as none mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: -10 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('none');
        });
    });

    describe('edge cases', () => {
        test('handles end_epoch of 1 as epoch mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 1 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('handles end_step of 1 as step mode', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: 1 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('step');
        });

        test('handles float values for end_epoch', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_epoch: 3.5 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('handles float values for end_step', () => {
            const dataset = { id: 'test', dataset_type: 'image', end_step: 200.5 };
            initializeEndScheduleMode(dataset);

            expect(dataset._endScheduleMode).toBe('step');
        });
    });
});

describe('Combined Start and End Scheduling', () => {
    describe('curriculum learning scenarios', () => {
        test('low-res dataset: immediate start, ends at epoch 3', () => {
            const dataset = { id: 'lowres-512', dataset_type: 'image', end_epoch: 3 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('none');
            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('high-res dataset: starts at epoch 3, runs indefinitely', () => {
            const dataset = { id: 'highres-1024', dataset_type: 'image', start_epoch: 3 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
            expect(dataset._endScheduleMode).toBe('none');
        });

        test('step-based curriculum: low-res ends at step 300', () => {
            const dataset = { id: 'lowres-512', dataset_type: 'image', end_step: 300 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('none');
            expect(dataset._endScheduleMode).toBe('step');
        });

        test('step-based curriculum: high-res starts at step 300', () => {
            const dataset = { id: 'highres-1024', dataset_type: 'image', start_step: 300 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('step');
            expect(dataset._endScheduleMode).toBe('none');
        });

        test('mixed mode: start by epoch, end by step', () => {
            const dataset = { id: 'mixed', dataset_type: 'image', start_epoch: 2, end_step: 500 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
            expect(dataset._endScheduleMode).toBe('step');
        });

        test('mixed mode: start by step, end by epoch', () => {
            const dataset = { id: 'mixed', dataset_type: 'image', start_step: 100, end_epoch: 5 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('step');
            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('full range: starts at epoch 2, ends at epoch 5', () => {
            const dataset = { id: 'bounded', dataset_type: 'image', start_epoch: 2, end_epoch: 5 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('epoch');
            expect(dataset._endScheduleMode).toBe('epoch');
        });

        test('full range: starts at step 100, ends at step 300', () => {
            const dataset = { id: 'bounded', dataset_type: 'image', start_step: 100, end_step: 300 };
            initializeAllScheduleModes(dataset);

            expect(dataset._scheduleMode).toBe('step');
            expect(dataset._endScheduleMode).toBe('step');
        });
    });

    describe('multiple datasets for curriculum learning', () => {
        test('complete curriculum setup with handoff', () => {
            const datasets = [
                { id: 'lowres-512', dataset_type: 'image', end_step: 300 },
                { id: 'highres-1024', dataset_type: 'image', start_step: 300 },
            ];

            datasets.forEach(initializeAllScheduleModes);

            expect(datasets[0]._scheduleMode).toBe('none');
            expect(datasets[0]._endScheduleMode).toBe('step');
            expect(datasets[1]._scheduleMode).toBe('step');
            expect(datasets[1]._endScheduleMode).toBe('none');
        });

        test('epoch-based curriculum with three stages', () => {
            const datasets = [
                { id: 'stage1', dataset_type: 'image', end_epoch: 3 },
                { id: 'stage2', dataset_type: 'image', start_epoch: 3, end_epoch: 6 },
                { id: 'stage3', dataset_type: 'image', start_epoch: 6 },
            ];

            datasets.forEach(initializeAllScheduleModes);

            expect(datasets[0]._scheduleMode).toBe('none');
            expect(datasets[0]._endScheduleMode).toBe('epoch');
            expect(datasets[1]._scheduleMode).toBe('epoch');
            expect(datasets[1]._endScheduleMode).toBe('epoch');
            expect(datasets[2]._scheduleMode).toBe('epoch');
            expect(datasets[2]._endScheduleMode).toBe('none');
        });
    });
});
