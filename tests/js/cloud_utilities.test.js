/**
 * Tests for cloud utilities module.
 *
 * Tests format helpers, job status helpers, and computed properties.
 */

// Mock UIHelpers
window.UIHelpers = {
    formatDuration: (seconds) => {
        if (!seconds) return '--';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
    },
    formatWaitTime: (seconds) => {
        if (seconds === null || seconds === undefined) return 'N/A';
        if (seconds < 60) return '< 1 min';
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        if (hours > 0) {
            return `~${hours}h ${minutes}m`;
        }
        return `~${minutes} min`;
    },
};

// Load the utilities module
require('../../simpletuner/static/js/modules/cloud/utilities.js');

describe('cloudUtilityMethods', () => {
    let context;

    beforeEach(() => {
        // Create a fresh context with required state
        context = {
            jobs: [],
            statusFilter: null,
            jobSearchQuery: '',
            jobsSortOrder: 'desc',
            pollingStatus: {
                loading: false,
                active: false,
                preference: null,
            },
            wizardRequiresUpload: false,
            wizardTotalSteps: 3,
        };

        // Bind methods to context
        Object.keys(window.cloudUtilityMethods).forEach((key) => {
            if (typeof window.cloudUtilityMethods[key] === 'function') {
                context[key] = window.cloudUtilityMethods[key].bind(context);
            }
        });
    });

    describe('job status helpers', () => {
        test('jobIsQueued identifies queued states', () => {
            expect(context.jobIsQueued({ status: 'pending' })).toBe(true);
            expect(context.jobIsQueued({ status: 'queued' })).toBe(true);
            expect(context.jobIsQueued({ status: 'running' })).toBe(false);
            expect(context.jobIsQueued({ status: 'completed' })).toBe(false);
        });

        test('jobIsRunning identifies running state', () => {
            expect(context.jobIsRunning({ status: 'running' })).toBe(true);
            expect(context.jobIsRunning({ status: 'pending' })).toBe(false);
            expect(context.jobIsRunning({ status: 'completed' })).toBe(false);
        });

        test('jobIsFailed identifies failed state', () => {
            expect(context.jobIsFailed({ status: 'failed' })).toBe(true);
            expect(context.jobIsFailed({ status: 'completed' })).toBe(false);
            expect(context.jobIsFailed({ status: 'running' })).toBe(false);
        });

        test('jobIsTerminal identifies terminal states', () => {
            expect(context.jobIsTerminal({ status: 'completed' })).toBe(true);
            expect(context.jobIsTerminal({ status: 'failed' })).toBe(true);
            expect(context.jobIsTerminal({ status: 'cancelled' })).toBe(true);
            expect(context.jobIsTerminal({ status: 'running' })).toBe(false);
            expect(context.jobIsTerminal({ status: 'pending' })).toBe(false);
        });

        test('jobHasSnapshot checks for snapshot metadata', () => {
            expect(context.jobHasSnapshot({ metadata: { snapshot: { abbrev: 'abc123' } } })).toBeTruthy();
            expect(context.jobHasSnapshot({ metadata: {} })).toBeFalsy();
            expect(context.jobHasSnapshot({})).toBeFalsy();
        });

        test('jobDisplayName returns appropriate name', () => {
            // Prefer tracker_run_name
            expect(context.jobDisplayName({
                job_id: 'job-12345678',
                config_name: 'config',
                metadata: { tracker_run_name: 'my-run' },
            })).toBe('my-run');

            // Fallback to config_name
            expect(context.jobDisplayName({
                job_id: 'job-12345678',
                config_name: 'config',
            })).toBe('config');

            // Fallback to truncated job_id
            expect(context.jobDisplayName({
                job_id: 'job-12345678-abcd',
            })).toBe('job-1234');
        });
    });

    describe('formatDuration', () => {
        test('handles null/undefined', () => {
            expect(context.formatDuration(null)).toBe('--');
            expect(context.formatDuration(undefined)).toBe('--');
            expect(context.formatDuration(0)).toBe('--');
        });

        test('formats minutes only', () => {
            expect(context.formatDuration(60)).toBe('1m');
            expect(context.formatDuration(120)).toBe('2m');
            expect(context.formatDuration(1800)).toBe('30m');
        });

        test('formats hours and minutes', () => {
            expect(context.formatDuration(3600)).toBe('1h 0m');
            expect(context.formatDuration(3660)).toBe('1h 1m');
            expect(context.formatDuration(7200)).toBe('2h 0m');
            expect(context.formatDuration(5400)).toBe('1h 30m');
        });
    });

    describe('formatWaitTime', () => {
        test('handles null/undefined', () => {
            expect(context.formatWaitTime(null)).toBe('N/A');
            expect(context.formatWaitTime(undefined)).toBe('N/A');
        });

        test('formats less than a minute', () => {
            expect(context.formatWaitTime(30)).toBe('< 1 min');
            expect(context.formatWaitTime(59)).toBe('< 1 min');
        });

        test('formats minutes', () => {
            expect(context.formatWaitTime(60)).toBe('~1 min');
            expect(context.formatWaitTime(300)).toBe('~5 min');
            expect(context.formatWaitTime(600)).toBe('~10 min');
        });

        test('formats hours and minutes', () => {
            expect(context.formatWaitTime(3600)).toBe('~1h 0m');
            expect(context.formatWaitTime(5400)).toBe('~1h 30m');
        });
    });

    describe('formatCost', () => {
        test('handles null/undefined', () => {
            expect(context.formatCost(null)).toBe('--');
            expect(context.formatCost(undefined)).toBe('--');
        });

        test('formats costs with two decimal places', () => {
            expect(context.formatCost(0)).toBe('$0.00');
            expect(context.formatCost(1)).toBe('$1.00');
            expect(context.formatCost(10.5)).toBe('$10.50');
            expect(context.formatCost(99.99)).toBe('$99.99');
        });
    });

    describe('formatBytes', () => {
        test('handles zero/null', () => {
            expect(context.formatBytes(0)).toBe('0 B');
            expect(context.formatBytes(null)).toBe('0 B');
        });

        test('formats bytes', () => {
            expect(context.formatBytes(500)).toBe('500.0 B');
            expect(context.formatBytes(1023)).toBe('1023.0 B');
        });

        test('formats kilobytes', () => {
            expect(context.formatBytes(1024)).toBe('1.0 KB');
            expect(context.formatBytes(2048)).toBe('2.0 KB');
            expect(context.formatBytes(1536)).toBe('1.5 KB');
        });

        test('formats megabytes', () => {
            expect(context.formatBytes(1048576)).toBe('1.0 MB');
            expect(context.formatBytes(5242880)).toBe('5.0 MB');
        });

        test('formats gigabytes', () => {
            expect(context.formatBytes(1073741824)).toBe('1.0 GB');
            expect(context.formatBytes(2147483648)).toBe('2.0 GB');
        });
    });

    describe('getUploadStageTitle', () => {
        test('returns correct titles for known stages', () => {
            expect(context.getUploadStageTitle('scanning')).toBe('Scanning Files');
            expect(context.getUploadStageTitle('packaging')).toBe('Packaging Data');
            expect(context.getUploadStageTitle('uploading')).toBe('Uploading');
            expect(context.getUploadStageTitle('complete')).toBe('Complete');
            expect(context.getUploadStageTitle('error')).toBe('Error');
        });

        test('returns default for unknown stages', () => {
            expect(context.getUploadStageTitle('unknown')).toBe('Processing');
            expect(context.getUploadStageTitle('')).toBe('Processing');
        });
    });

    describe('statusColor', () => {
        test('returns correct colors for all statuses', () => {
            expect(context.statusColor('pending')).toBe('secondary');
            expect(context.statusColor('uploading')).toBe('info');
            expect(context.statusColor('queued')).toBe('info');
            expect(context.statusColor('running')).toBe('primary');
            expect(context.statusColor('completed')).toBe('success');
            expect(context.statusColor('failed')).toBe('danger');
            expect(context.statusColor('cancelled')).toBe('warning');
        });

        test('returns default for unknown status', () => {
            expect(context.statusColor('unknown')).toBe('secondary');
        });
    });

    describe('statusIcon', () => {
        test('returns correct icons for all statuses', () => {
            expect(context.statusIcon('pending')).toBe('fa-clock');
            expect(context.statusIcon('uploading')).toBe('fa-upload');
            expect(context.statusIcon('queued')).toBe('fa-hourglass-half');
            expect(context.statusIcon('running')).toBe('fa-play');
            expect(context.statusIcon('completed')).toBe('fa-check');
            expect(context.statusIcon('failed')).toBe('fa-times');
            expect(context.statusIcon('cancelled')).toBe('fa-ban');
        });

        test('returns default for unknown status', () => {
            expect(context.statusIcon('unknown')).toBe('fa-circle');
        });
    });

    describe('jobTypeIcon', () => {
        test('returns cloud icon for cloud jobs', () => {
            expect(context.jobTypeIcon('cloud')).toBe('fa-cloud');
        });

        test('returns desktop icon for local jobs', () => {
            expect(context.jobTypeIcon('local')).toBe('fa-desktop');
            expect(context.jobTypeIcon('other')).toBe('fa-desktop');
        });
    });

    describe('wizard step helpers', () => {
        test('wizardIsStep checks step equality', () => {
            expect(context.wizardIsStep(1, 1)).toBe(true);
            expect(context.wizardIsStep(2, 1)).toBe(false);
        });

        test('wizardIsStep handles "last" target', () => {
            context.wizardTotalSteps = 3;
            expect(context.wizardIsStep(3, 'last')).toBe(true);
            expect(context.wizardIsStep(2, 'last')).toBe(false);
        });

        test('wizardIsDataStep checks for data step with upload', () => {
            context.wizardRequiresUpload = true;
            expect(context.wizardIsDataStep(2)).toBe(true);
            expect(context.wizardIsDataStep(1)).toBe(false);

            context.wizardRequiresUpload = false;
            expect(context.wizardIsDataStep(2)).toBe(false);
        });

        test('wizardIsFinalStep checks for last step', () => {
            context.wizardTotalSteps = 4;
            expect(context.wizardIsFinalStep(4)).toBe(true);
            expect(context.wizardIsFinalStep(3)).toBe(false);
        });
    });
});

describe('cloudComputedProperties', () => {
    describe('activeJobs', () => {
        test('filters for non-terminal jobs', () => {
            const jobs = [
                { job_id: '1', status: 'pending' },
                { job_id: '2', status: 'running' },
                { job_id: '3', status: 'completed' },
                { job_id: '4', status: 'queued' },
                { job_id: '5', status: 'failed' },
            ];

            const context = { jobs };
            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'activeJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(3);
            expect(result.map((j) => j.job_id)).toEqual(['1', '2', '4']);
        });
    });

    describe('completedJobs', () => {
        test('filters for terminal jobs', () => {
            const jobs = [
                { job_id: '1', status: 'pending' },
                { job_id: '2', status: 'completed' },
                { job_id: '3', status: 'failed' },
                { job_id: '4', status: 'cancelled' },
            ];

            const context = { jobs };
            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'completedJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(3);
            expect(result.map((j) => j.job_id)).toEqual(['2', '3', '4']);
        });
    });

    describe('filteredJobs', () => {
        const jobs = [
            { job_id: 'job-abc', config_name: 'sdxl-train', status: 'completed', provider: 'replicate', created_at: '2024-01-01T10:00:00Z' },
            { job_id: 'job-def', config_name: 'flux-finetune', status: 'running', provider: 'replicate', created_at: '2024-01-02T10:00:00Z' },
            { job_id: 'job-ghi', config_name: 'sd15-lora', status: 'failed', provider: 'modal', created_at: '2024-01-03T10:00:00Z' },
        ];

        test('applies status filter', () => {
            const context = {
                jobs,
                statusFilter: 'running',
                jobSearchQuery: '',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(1);
            expect(result[0].job_id).toBe('job-def');
        });

        test('applies search query to job_id', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: 'abc',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(1);
            expect(result[0].job_id).toBe('job-abc');
        });

        test('applies search query to config_name', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: 'flux',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(1);
            expect(result[0].config_name).toBe('flux-finetune');
        });

        test('applies search query to provider', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: 'modal',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(1);
            expect(result[0].provider).toBe('modal');
        });

        test('sorts by date descending', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: '',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result[0].job_id).toBe('job-ghi'); // newest first
            expect(result[2].job_id).toBe('job-abc'); // oldest last
        });

        test('sorts by date ascending', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: '',
                jobsSortOrder: 'asc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result[0].job_id).toBe('job-abc'); // oldest first
            expect(result[2].job_id).toBe('job-ghi'); // newest last
        });

        test('search is case-insensitive', () => {
            const context = {
                jobs,
                statusFilter: null,
                jobSearchQuery: 'SDXL',
                jobsSortOrder: 'desc',
            };

            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'filteredJobs').get;
            const result = getter.call(context);

            expect(result).toHaveLength(1);
            expect(result[0].config_name).toBe('sdxl-train');
        });
    });

    describe('failedCount', () => {
        test('counts failed jobs', () => {
            const jobs = [
                { status: 'failed' },
                { status: 'completed' },
                { status: 'failed' },
                { status: 'running' },
            ];

            const context = { jobs };
            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'failedCount').get;
            const result = getter.call(context);

            expect(result).toBe(2);
        });

        test('returns 0 when no failed jobs', () => {
            const jobs = [
                { status: 'completed' },
                { status: 'running' },
            ];

            const context = { jobs };
            const getter = Object.getOwnPropertyDescriptor(window.cloudComputedProperties, 'failedCount').get;
            const result = getter.call(context);

            expect(result).toBe(0);
        });
    });
});
