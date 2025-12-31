/**
 * Tests for cloud queue management module.
 *
 * Tests queue stats, concurrency settings, and queue admin operations.
 */

// Load the queue module
require('../../simpletuner/static/js/modules/cloud/queue.js');

describe('cloudQueueMethods', () => {
    let context;

    beforeEach(() => {
        // Reset fetch mock
        fetch.mockReset();
        fetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({}),
        });

        // Create a fresh context with required state
        context = {
            showQueuePanel: false,
            loadingQueueStats: false,
            savingQueueSettings: false,
            processingQueue: false,
            cleaningQueue: false,
            cleanupDays: 7,
            lastCleanupResult: null,
            hasAdminAccess: false,
            currentUser: { id: 'user-1' },
            queueStats: {
                pending: 0,
                running: 0,
                blocked: 0,
            },
            queueSettings: {
                max_concurrent: 5,
                user_max_concurrent: 2,
                team_max_concurrent: 10,
                enable_fair_share: false,
            },
            queuePendingJobs: [],
            queueBlockedJobs: [],
            approvalModal: {
                open: false,
                action: null,
                job: null,
                reason: '',
                notes: '',
                processing: false,
            },
            refreshQueueStats: jest.fn(),
            loadJobs: jest.fn(),
            loadQueuePendingJobs: jest.fn(),
            loadQueueBlockedJobs: jest.fn(),
            loadPendingApprovals: jest.fn(),
        };

        // Bind methods to context
        Object.keys(window.cloudQueueMethods).forEach((key) => {
            if (typeof window.cloudQueueMethods[key] === 'function') {
                context[key] = window.cloudQueueMethods[key].bind(context);
            }
        });
    });

    describe('toggleQueuePanel', () => {
        test('opens panel and triggers refresh methods', () => {
            context.showQueuePanel = false;

            // Replace bound methods with spies
            const refreshSpy = jest.fn();
            const pendingSpy = jest.fn();
            const blockedSpy = jest.fn();
            context.refreshQueueStats = refreshSpy;
            context.loadQueuePendingJobs = pendingSpy;
            context.loadQueueBlockedJobs = blockedSpy;

            // Re-bind toggleQueuePanel to use the spies
            context.toggleQueuePanel = window.cloudQueueMethods.toggleQueuePanel.bind(context);

            context.toggleQueuePanel();

            expect(context.showQueuePanel).toBe(true);
            expect(refreshSpy).toHaveBeenCalled();
            expect(pendingSpy).toHaveBeenCalled();
            expect(blockedSpy).toHaveBeenCalled();
        });

        test('closes panel without refresh', () => {
            context.showQueuePanel = true;
            const refreshSpy = jest.fn();
            context.refreshQueueStats = refreshSpy;
            context.toggleQueuePanel = window.cloudQueueMethods.toggleQueuePanel.bind(context);

            context.toggleQueuePanel();

            expect(context.showQueuePanel).toBe(false);
            expect(refreshSpy).not.toHaveBeenCalled();
        });
    });

    describe('refreshQueueStats', () => {
        test('fetches queue stats from API', async () => {
            const mockStats = {
                pending: 5,
                running: 2,
                blocked: 1,
                max_concurrent: 10,
                user_max_concurrent: 3,
                team_max_concurrent: 15,
                enable_fair_share: true,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStats),
            });

            await context.refreshQueueStats();

            expect(fetch).toHaveBeenCalledWith('/api/queue/stats');
            expect(context.queueStats.pending).toBe(5);
            expect(context.queueStats.running).toBe(2);
            expect(context.queueSettings.max_concurrent).toBe(10);
            expect(context.queueSettings.enable_fair_share).toBe(true);
        });

        test('handles API errors gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            // Should not throw
            await expect(context.refreshQueueStats()).resolves.not.toThrow();
            expect(context.loadingQueueStats).toBe(false);
        });

        test('prevents concurrent requests', async () => {
            context.loadingQueueStats = true;

            await context.refreshQueueStats();

            // Should not make another fetch call
            expect(fetch).not.toHaveBeenCalled();
        });
    });

    describe('loadQueuePendingJobs', () => {
        test('fetches pending jobs from API', async () => {
            const mockData = {
                entries: [
                    { job_id: 'job-1', status: 'pending' },
                    { job_id: 'job-2', status: 'pending' },
                ],
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockData),
            });

            await context.loadQueuePendingJobs();

            expect(fetch).toHaveBeenCalledWith('/api/queue?status=pending&limit=20');
            expect(context.queuePendingJobs).toHaveLength(2);
        });

        test('auto-shows queue panel for non-admin with pending jobs', async () => {
            context.hasAdminAccess = false;
            context.currentUser = { id: 'user-1' };
            context.showQueuePanel = false;

            const mockData = {
                entries: [
                    { job_id: 'job-1', user_id: 'user-1', status: 'pending' },
                ],
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockData),
            });

            await context.loadQueuePendingJobs();

            expect(context.showQueuePanel).toBe(true);
        });

        test('does not auto-show for admin', async () => {
            context.hasAdminAccess = true;
            context.showQueuePanel = false;

            const mockData = {
                entries: [
                    { job_id: 'job-1', user_id: 'user-1', status: 'pending' },
                ],
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockData),
            });

            await context.loadQueuePendingJobs();

            expect(context.showQueuePanel).toBe(false);
        });
    });

    describe('loadQueueBlockedJobs', () => {
        test('fetches blocked jobs from API', async () => {
            const mockData = {
                entries: [
                    { job_id: 'job-1', status: 'blocked' },
                ],
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockData),
            });

            await context.loadQueueBlockedJobs();

            expect(fetch).toHaveBeenCalledWith('/api/queue?status=blocked&limit=20');
            expect(context.queueBlockedJobs).toHaveLength(1);
        });
    });

    describe('updateConcurrencyLimits', () => {
        test('sends updated settings to API', async () => {
            context.queueSettings = {
                max_concurrent: 8,
                user_max_concurrent: 4,
                team_max_concurrent: 20,
                enable_fair_share: true,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(context.queueSettings),
            });

            await context.updateConcurrencyLimits();

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/concurrency',
                expect.objectContaining({
                    method: 'POST',
                    body: JSON.stringify({
                        max_concurrent: 8,
                        user_max_concurrent: 4,
                        team_max_concurrent: 20,
                        enable_fair_share: true,
                    }),
                })
            );
        });

        test('shows success toast on success', async () => {
            window.showToast = jest.fn();

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ max_concurrent: 10 }),
            });

            await context.updateConcurrencyLimits();

            expect(window.showToast).toHaveBeenCalledWith('Queue settings updated', 'success');

            delete window.showToast;
        });

        test('shows error toast on failure', async () => {
            window.showToast = jest.fn();

            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Invalid settings' }),
            });

            await context.updateConcurrencyLimits();

            expect(window.showToast).toHaveBeenCalledWith('Invalid settings', 'error');

            delete window.showToast;
        });
    });

    describe('processQueue', () => {
        test('triggers queue processing', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ dispatched: 3 }),
            });

            window.showToast = jest.fn();

            await context.processQueue();

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/process',
                expect.objectContaining({ method: 'POST' })
            );
            expect(window.showToast).toHaveBeenCalledWith('Dispatched 3 job(s)', 'success');

            delete window.showToast;
        });

        test('prevents concurrent processing', async () => {
            context.processingQueue = true;

            await context.processQueue();

            expect(fetch).not.toHaveBeenCalled();
        });
    });

    describe('cleanupQueue', () => {
        beforeEach(() => {
            // Mock confirm to return true
            global.confirm = jest.fn(() => true);
        });

        test('sends cleanup request with days parameter', async () => {
            context.cleanupDays = 14;

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ deleted: 5 }),
            });

            window.showToast = jest.fn();

            await context.cleanupQueue();

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/cleanup?days=14',
                expect.objectContaining({ method: 'POST' })
            );
            expect(context.lastCleanupResult).toEqual({ deleted: 5 });
            expect(window.showToast).toHaveBeenCalledWith('Cleaned up 5 entries', 'success');

            delete window.showToast;
        });

        test('requires confirmation', async () => {
            global.confirm = jest.fn(() => false);

            await context.cleanupQueue();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('prevents concurrent cleanup', async () => {
            context.cleaningQueue = true;

            await context.cleanupQueue();

            expect(fetch).not.toHaveBeenCalled();
        });
    });

    describe('cancelQueuedJob', () => {
        beforeEach(() => {
            global.confirm = jest.fn(() => true);
        });

        test('sends cancel request', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            window.showToast = jest.fn();

            await context.cancelQueuedJob('job-123');

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/job-123/cancel',
                expect.objectContaining({ method: 'POST' })
            );
            expect(window.showToast).toHaveBeenCalledWith('Job cancelled', 'success');

            delete window.showToast;
        });

        test('requires confirmation', async () => {
            global.confirm = jest.fn(() => false);

            await context.cancelQueuedJob('job-123');

            expect(fetch).not.toHaveBeenCalled();
        });
    });

    describe('approveQueuedJob', () => {
        test('sends approve request', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            window.showToast = jest.fn();

            await context.approveQueuedJob('job-123');

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/job-123/approve',
                expect.objectContaining({ method: 'POST' })
            );
            expect(window.showToast).toHaveBeenCalledWith('Job approved', 'success');

            delete window.showToast;
        });
    });

    describe('rejectQueuedJob', () => {
        test('sends reject request with reason', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            window.showToast = jest.fn();

            await context.rejectQueuedJob('job-123', 'Cost too high');

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/job-123/reject?reason=Cost%20too%20high',
                expect.objectContaining({ method: 'POST' })
            );
            expect(window.showToast).toHaveBeenCalledWith('Job rejected', 'success');

            delete window.showToast;
        });
    });

    describe('approval modal', () => {
        test('openApprovalModal sets up modal state', () => {
            const job = { job_id: 'job-456', config_name: 'test' };

            context.openApprovalModal('approve', job);

            expect(context.approvalModal.open).toBe(true);
            expect(context.approvalModal.action).toBe('approve');
            expect(context.approvalModal.job).toBe(job);
            expect(context.approvalModal.reason).toBe('');
        });

        test('closeApprovalModal resets modal state', () => {
            context.approvalModal.open = true;
            context.approvalModal.action = 'reject';
            context.approvalModal.job = { job_id: 'test' };
            context.approvalModal.reason = 'some reason';

            context.closeApprovalModal();

            expect(context.approvalModal.open).toBe(false);
            expect(context.approvalModal.action).toBeNull();
            expect(context.approvalModal.job).toBeNull();
            expect(context.approvalModal.reason).toBe('');
        });

        test('submitApprovalAction calls approve for approve action', async () => {
            context.approvalModal.job = { job_id: 'job-789' };
            context.approvalModal.action = 'approve';

            // Mock approveQueuedJob
            context.approveQueuedJob = jest.fn().mockResolvedValue();

            await context.submitApprovalAction();

            expect(context.approveQueuedJob).toHaveBeenCalledWith('job-789');
        });

        test('submitApprovalAction calls reject with reason for reject action', async () => {
            context.approvalModal.job = { job_id: 'job-789' };
            context.approvalModal.action = 'reject';
            context.approvalModal.reason = 'Too expensive';

            // Mock rejectQueuedJob
            context.rejectQueuedJob = jest.fn().mockResolvedValue();

            await context.submitApprovalAction();

            expect(context.rejectQueuedJob).toHaveBeenCalledWith('job-789', 'Too expensive');
        });

        test('submitApprovalAction requires reason for reject', async () => {
            context.approvalModal.job = { job_id: 'job-789' };
            context.approvalModal.action = 'reject';
            context.approvalModal.reason = '   '; // Empty after trim

            window.showToast = jest.fn();

            await context.submitApprovalAction();

            expect(window.showToast).toHaveBeenCalledWith('Rejection reason is required', 'error');

            delete window.showToast;
        });

        test('submitApprovalAction does nothing without job', async () => {
            context.approvalModal.job = null;
            context.approveQueuedJob = jest.fn();

            await context.submitApprovalAction();

            expect(context.approveQueuedJob).not.toHaveBeenCalled();
        });
    });
});

describe('Queue Position Calculation', () => {
    test('calculates user position in queue', () => {
        function calculateUserPosition(userId, pendingJobs) {
            const userJobs = pendingJobs.filter(j => j.user_id === userId);
            if (userJobs.length === 0) return null;

            // Find the earliest position of user's jobs
            const position = pendingJobs.findIndex(j => j.user_id === userId);
            return position + 1; // 1-indexed
        }

        const pendingJobs = [
            { job_id: 'job-1', user_id: 'user-a' },
            { job_id: 'job-2', user_id: 'user-b' },
            { job_id: 'job-3', user_id: 'user-a' },
            { job_id: 'job-4', user_id: 'user-c' },
        ];

        expect(calculateUserPosition('user-a', pendingJobs)).toBe(1);
        expect(calculateUserPosition('user-b', pendingJobs)).toBe(2);
        expect(calculateUserPosition('user-c', pendingJobs)).toBe(4);
        expect(calculateUserPosition('user-d', pendingJobs)).toBeNull();
    });

    test('estimates wait time based on position', () => {
        function estimateWaitTime(position, avgJobDuration, maxConcurrent) {
            if (!position) return null;

            // Rough estimate: position / concurrent * avg duration
            const batches = Math.ceil(position / maxConcurrent);
            return batches * avgJobDuration;
        }

        // 3rd position, 30 min avg, 2 concurrent = 2 batches * 30 = 60 min
        expect(estimateWaitTime(3, 1800, 2)).toBe(3600);

        // 5th position, 30 min avg, 5 concurrent = 1 batch * 30 = 30 min
        expect(estimateWaitTime(5, 1800, 5)).toBe(1800);

        // No position
        expect(estimateWaitTime(null, 1800, 5)).toBeNull();
    });
});
