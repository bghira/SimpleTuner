/**
 * Tests for cloudJobMethods.
 *
 * cloudJobMethods handles job CRUD, syncing, filtering, and selection.
 */

// Mock fetch
global.fetch = jest.fn();

// Mock confirm
global.confirm = jest.fn();

// Mock console
global.console = {
    ...console,
    error: jest.fn(),
    warn: jest.fn(),
};

// Mock showToast
global.showToast = jest.fn();
window.showToast = global.showToast;

// Load the module
require('../../simpletuner/static/js/modules/cloud/jobs.js');

describe('cloudJobMethods', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        // Create a fresh context with required state for each test
        context = {
            jobs: [],
            jobsLoading: false,
            jobsInitialized: false,
            syncing: false,
            selectedJob: null,
            activeProvider: 'replicate',
            loadMetrics: jest.fn(),
        };
    });

    describe('loadJobs', () => {
        test('sets jobsLoading during fetch', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            const loadPromise = window.cloudJobMethods.loadJobs.call(context);
            expect(context.jobsLoading).toBe(true);

            await loadPromise;
            expect(context.jobsLoading).toBe(false);
        });

        test('fetches jobs with correct parameters', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            await window.cloudJobMethods.loadJobs.call(context);

            expect(fetch).toHaveBeenCalledWith(
                expect.stringMatching(/\/api\/cloud\/jobs\?limit=100&provider=replicate/)
            );
        });

        test('populates jobs array on success', async () => {
            const mockJobs = [
                { job_id: 'job-1', status: 'running' },
                { job_id: 'job-2', status: 'completed' },
            ];

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: mockJobs }),
            });

            await window.cloudJobMethods.loadJobs.call(context);

            expect(context.jobs).toEqual(mockJobs);
            expect(context.jobsInitialized).toBe(true);
        });

        test('skips loading state update when syncActive is true', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            // Should not set jobsLoading to true when syncActive
            const loadPromise = window.cloudJobMethods.loadJobs.call(context, true);
            expect(context.jobsLoading).toBe(false);

            await loadPromise;
        });

        test('handles API error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await window.cloudJobMethods.loadJobs.call(context);

            expect(context.jobsLoading).toBe(false);
            expect(console.error).toHaveBeenCalledWith(
                'Failed to load jobs:',
                expect.any(Error)
            );
        });

        test('handles non-ok response', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
            });

            await window.cloudJobMethods.loadJobs.call(context);

            expect(context.jobs).toEqual([]);
            expect(context.jobsInitialized).toBe(false);
        });
    });

    describe('updateJobsInPlace', () => {
        test('updates existing jobs while preserving inline state', () => {
            context.jobs = [
                {
                    job_id: 'job-1',
                    status: 'running',
                    inline_stage: 'training',
                    inline_log: 'Step 100',
                    inline_progress: 50,
                },
            ];

            const newJobs = [
                { job_id: 'job-1', status: 'completed' },
            ];

            window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

            expect(context.jobs[0].status).toBe('completed');
            expect(context.jobs[0].inline_stage).toBe('training');
            expect(context.jobs[0].inline_log).toBe('Step 100');
            expect(context.jobs[0].inline_progress).toBe(50);
        });

        test('adds new jobs at the beginning', () => {
            context.jobs = [
                { job_id: 'job-1', status: 'completed' },
            ];

            const newJobs = [
                { job_id: 'job-1', status: 'completed' },
                { job_id: 'job-2', status: 'running' },
            ];

            window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

            expect(context.jobs.length).toBe(2);
            expect(context.jobs[0].job_id).toBe('job-2'); // New job at beginning
        });

        test('removes deleted jobs', () => {
            context.jobs = [
                { job_id: 'job-1', status: 'completed' },
                { job_id: 'job-2', status: 'failed' },
            ];

            const newJobs = [
                { job_id: 'job-1', status: 'completed' },
            ];

            window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

            expect(context.jobs.length).toBe(1);
            expect(context.jobs[0].job_id).toBe('job-1');
        });

        test('updates selectedJob if it still exists', () => {
            const selectedJob = { job_id: 'job-1', status: 'running' };
            context.jobs = [selectedJob];
            context.selectedJob = selectedJob;

            const newJobs = [
                { job_id: 'job-1', status: 'completed' },
            ];

            window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

            expect(context.selectedJob.status).toBe('completed');
        });

        test('clears selectedJob if it was deleted', () => {
            const selectedJob = { job_id: 'job-1', status: 'running' };
            context.jobs = [selectedJob];
            context.selectedJob = selectedJob;

            const newJobs = [];

            window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

            expect(context.selectedJob).toBeNull();
        });
    });

    describe('syncJobs', () => {
        test('prevents concurrent syncs', async () => {
            context.syncing = true;

            await window.cloudJobMethods.syncJobs.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sends POST request to sync endpoint', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ updated: 5 }),
            });
            // Mock loadJobs
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            await window.cloudJobMethods.syncJobs.call(context);

            expect(fetch).toHaveBeenCalledWith('/api/cloud/jobs/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: 'replicate' }),
            });
        });

        test('shows success toast with count', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ updated: 3 }),
            });
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            await window.cloudJobMethods.syncJobs.call(context);

            expect(window.showToast).toHaveBeenCalledWith('Synced 3 job(s)', 'success');
        });

        test('shows error toast on failure', async () => {
            fetch.mockResolvedValueOnce({ ok: false });

            await window.cloudJobMethods.syncJobs.call(context);

            expect(window.showToast).toHaveBeenCalledWith('Failed to sync jobs', 'error');
        });

        test('handles network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await window.cloudJobMethods.syncJobs.call(context);

            expect(window.showToast).toHaveBeenCalledWith('Failed to sync jobs', 'error');
            expect(context.syncing).toBe(false);
        });

        test('sets syncing flag during operation', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ updated: 0 }),
            });
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            const promise = window.cloudJobMethods.syncJobs.call(context);
            expect(context.syncing).toBe(true);

            await promise;
            expect(context.syncing).toBe(false);
        });
    });

    describe('cancelJob', () => {
        test('requires confirmation', async () => {
            confirm.mockReturnValueOnce(false);

            await window.cloudJobMethods.cancelJob.call(context, 'job-123');

            expect(confirm).toHaveBeenCalledWith('Are you sure you want to cancel this job?');
            expect(fetch).not.toHaveBeenCalled();
        });

        test('sends POST to cancel endpoint', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            await window.cloudJobMethods.cancelJob.call(context, 'job-123');

            expect(fetch).toHaveBeenCalledWith('/api/cloud/jobs/job-123/cancel', { method: 'POST' });
        });

        test('shows success toast and reloads jobs', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });

            // Mock loadJobs to avoid second fetch
            context.loadJobs = jest.fn();

            await window.cloudJobMethods.cancelJob.call(context, 'job-123');

            expect(window.showToast).toHaveBeenCalledWith('Job cancelled', 'success');
            expect(context.loadJobs).toHaveBeenCalled();
        });

        test('shows error from response on failure', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Job cannot be cancelled' }),
            });

            await window.cloudJobMethods.cancelJob.call(context, 'job-123');

            expect(window.showToast).toHaveBeenCalledWith('Job cannot be cancelled', 'error');
        });

        test('uses default error message when detail missing', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({}),
            });

            await window.cloudJobMethods.cancelJob.call(context, 'job-123');

            expect(window.showToast).toHaveBeenCalledWith('Failed to cancel job', 'error');
        });
    });

    describe('deleteJob', () => {
        test('requires confirmation', async () => {
            confirm.mockReturnValueOnce(false);

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(confirm).toHaveBeenCalledWith('Are you sure you want to remove this job from history?');
            expect(fetch).not.toHaveBeenCalled();
        });

        test('sends DELETE request', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ jobs: [] }),
            });

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(fetch).toHaveBeenCalledWith('/api/cloud/jobs/job-123', { method: 'DELETE' });
        });

        test('clears selectedJob if deleted job was selected', async () => {
            context.selectedJob = { job_id: 'job-123' };
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });

            // Mock loadJobs to avoid second fetch
            context.loadJobs = jest.fn();

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(context.selectedJob).toBeNull();
        });

        test('calls loadMetrics after delete', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });

            // Mock loadJobs to avoid second fetch
            context.loadJobs = jest.fn();

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(context.loadMetrics).toHaveBeenCalled();
        });

        test('shows success toast', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });

            // Mock loadJobs to avoid second fetch
            context.loadJobs = jest.fn();

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(window.showToast).toHaveBeenCalledWith('Job removed from history', 'success');
        });

        test('shows error on failure', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Permission denied' }),
            });

            await window.cloudJobMethods.deleteJob.call(context, 'job-123');

            expect(window.showToast).toHaveBeenCalledWith('Permission denied', 'error');
        });
    });

    describe('selectJob', () => {
        test('sets selectedJob', () => {
            const job = { job_id: 'job-123', status: 'running' };

            window.cloudJobMethods.selectJob.call(context, job);

            expect(context.selectedJob).toBe(job);
        });

        test('allows selecting null', () => {
            context.selectedJob = { job_id: 'job-123' };

            window.cloudJobMethods.selectJob.call(context, null);

            expect(context.selectedJob).toBeNull();
        });
    });
});

describe('cloudJobMethods edge cases', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        context = {
            jobs: [],
            jobsLoading: false,
            jobsInitialized: false,
            syncing: false,
            selectedJob: null,
            activeProvider: 'replicate',
            loadMetrics: jest.fn(),
        };
    });

    test('loadJobs handles empty jobs array from API', async () => {
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ jobs: [] }),
        });

        await window.cloudJobMethods.loadJobs.call(context);

        expect(context.jobs).toEqual([]);
        expect(context.jobsInitialized).toBe(true);
    });

    test('loadJobs handles missing jobs field in response', async () => {
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({}),
        });

        await window.cloudJobMethods.loadJobs.call(context);

        expect(context.jobs).toEqual([]);
    });

    test('updateJobsInPlace with no existing jobs adds all new jobs', () => {
        context.jobs = [];

        const newJobs = [
            { job_id: 'job-1' },
            { job_id: 'job-2' },
        ];

        window.cloudJobMethods.updateJobsInPlace.call(context, newJobs);

        expect(context.jobs.length).toBe(2);
    });

    test('syncJobs works without showToast', async () => {
        delete window.showToast;

        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ updated: 1 }),
        });
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ jobs: [] }),
        });

        // Should not throw
        await expect(
            window.cloudJobMethods.syncJobs.call(context)
        ).resolves.not.toThrow();

        window.showToast = global.showToast;
    });
});
