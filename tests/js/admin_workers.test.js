/**
 * Tests for admin workers module.
 *
 * Tests GPU worker CRUD, status monitoring, and token management.
 */

// Mock UIHelpers
window.UIHelpers = {
    extractErrorMessage: (data, fallback) => {
        return data?.detail || fallback;
    },
};

// Mock window.showToast
window.showToast = jest.fn();

// Mock window.confirm
global.confirm = jest.fn(() => true);

// Mock navigator.clipboard
global.navigator.clipboard = {
    writeText: jest.fn(() => Promise.resolve()),
};

// Load the workers module
require('../../simpletuner/static/js/modules/admin/workers.js');

describe('adminWorkerMethods', () => {
    let context;

    beforeEach(() => {
        // Reset fetch mock
        fetch.mockClear();

        // Reset other mocks
        window.showToast.mockClear();
        global.confirm.mockClear();
        global.navigator.clipboard.writeText.mockClear();

        // Create a fresh context with required state
        context = {
            workers: [],
            workersLoading: false,
            workerStats: {
                total: 0,
                idle: 0,
                busy: 0,
                offline: 0,
            },
            workerForm: {
                name: '',
                worker_type: 'persistent',
                labels_str: '',
            },
            workerFormOpen: false,
            workerTokenModalOpen: false,
            workerToken: null,
            workerConnectionCommand: null,
            saving: false,
            error: null,
            deletingWorker: null,
            deleteWorkerOpen: false,
            workerRefreshInterval: null,
            workerUptimeTick: Date.now(),
            workerUptimeLoadedAt: null,
            workerUptimeInterval: null,
            activeTab: 'workers',
        };

        // Bind methods to context
        Object.keys(window.adminWorkerMethods).forEach((key) => {
            if (typeof window.adminWorkerMethods[key] === 'function') {
                context[key] = window.adminWorkerMethods[key].bind(context);
            }
        });
    });

    afterEach(() => {
        // Clean up any intervals
        if (context.workerRefreshInterval) {
            clearInterval(context.workerRefreshInterval);
        }
        if (context.workerUptimeInterval) {
            clearInterval(context.workerUptimeInterval);
        }
    });

    describe('loadWorkers', () => {
        test('fetches workers from API successfully', async () => {
            const mockWorkers = [
                { worker_id: '1', name: 'worker-1', status: 'idle' },
                { worker_id: '2', name: 'worker-2', status: 'busy' },
            ];

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ workers: mockWorkers }),
            });

            await context.loadWorkers();

            expect(fetch).toHaveBeenCalledWith('/api/admin/workers');
            expect(context.workers).toEqual(mockWorkers);
            expect(context.workersLoading).toBe(false);
        });

        test('sets loading state during request', async () => {
            let loadingDuringRequest = false;

            fetch.mockImplementationOnce(() => {
                loadingDuringRequest = context.workersLoading;
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ workers: [] }),
                });
            });

            await context.loadWorkers();

            expect(loadingDuringRequest).toBe(true);
            expect(context.workersLoading).toBe(false);
        });

        test('updates worker stats after loading', async () => {
            const mockWorkers = [
                { worker_id: '1', status: 'idle' },
                { worker_id: '2', status: 'busy' },
                { worker_id: '3', status: 'offline' },
            ];

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ workers: mockWorkers }),
            });

            await context.loadWorkers();

            expect(context.workerStats.total).toBe(3);
            expect(context.workerStats.idle).toBe(1);
            expect(context.workerStats.busy).toBe(1);
            expect(context.workerStats.offline).toBe(1);
        });

        test('handles empty workers array', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ workers: [] }),
            });

            await context.loadWorkers();

            expect(context.workers).toEqual([]);
            expect(context.workerStats.total).toBe(0);
        });

        test('handles missing workers property', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            await context.loadWorkers();

            expect(context.workers).toEqual([]);
        });

        test('handles failed response', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                statusText: 'Internal Server Error',
            });

            await context.loadWorkers();

            expect(window.showToast).toHaveBeenCalledWith('Failed to load workers', 'error');
            expect(context.workersLoading).toBe(false);
        });

        test('handles network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.loadWorkers();

            expect(window.showToast).toHaveBeenCalledWith('Failed to load workers', 'error');
            expect(context.workersLoading).toBe(false);
        });
    });

    describe('updateWorkerStats', () => {
        test('calculates stats correctly', () => {
            context.workers = [
                { status: 'idle' },
                { status: 'idle' },
                { status: 'busy' },
                { status: 'offline' },
                { status: 'draining' },
            ];

            context.updateWorkerStats();

            expect(context.workerStats.total).toBe(5);
            expect(context.workerStats.idle).toBe(2);
            expect(context.workerStats.busy).toBe(1);
            expect(context.workerStats.offline).toBe(1);
        });

        test('handles empty worker list', () => {
            context.workers = [];

            context.updateWorkerStats();

            expect(context.workerStats.total).toBe(0);
            expect(context.workerStats.idle).toBe(0);
            expect(context.workerStats.busy).toBe(0);
            expect(context.workerStats.offline).toBe(0);
        });

        test('handles workers with various statuses', () => {
            context.workers = [
                { status: 'connecting' },
                { status: 'draining' },
                { status: 'unknown' },
            ];

            context.updateWorkerStats();

            expect(context.workerStats.total).toBe(3);
            expect(context.workerStats.idle).toBe(0);
            expect(context.workerStats.busy).toBe(0);
            expect(context.workerStats.offline).toBe(0);
        });
    });

    describe('createWorker', () => {
        test('creates worker successfully', async () => {
            context.workerForm = {
                name: 'gpu-worker-1',
                worker_type: 'persistent',
                labels_str: 'gpu=nvidia,region=us-west',
            };

            const mockResponse = {
                token: 'test-token-123',
                connection_command: 'simpletuner worker --orchestrator-url http://localhost:8001 --worker-token test-token-123',
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            // Mock loadWorkers
            context.loadWorkers = jest.fn();

            await context.createWorker();

            expect(fetch).toHaveBeenCalledWith('/api/admin/workers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: 'gpu-worker-1',
                    worker_type: 'persistent',
                    labels: { gpu: 'nvidia', region: 'us-west' },
                }),
            });

            expect(context.workerToken).toBe('test-token-123');
            expect(context.workerConnectionCommand).toBe('simpletuner worker --orchestrator-url http://localhost:8001 --worker-token test-token-123');
            expect(context.workerFormOpen).toBe(false);
            expect(context.workerTokenModalOpen).toBe(true);
            expect(window.showToast).toHaveBeenCalledWith('Worker created successfully', 'success');
            expect(context.loadWorkers).toHaveBeenCalled();
        });

        test('parses labels correctly', async () => {
            context.workerForm = {
                name: 'test-worker',
                worker_type: 'persistent',
                labels_str: 'key1=value1, key2=value2',
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ token: 'test-token' }),
            });

            context.loadWorkers = jest.fn();

            await context.createWorker();

            const requestBody = JSON.parse(fetch.mock.calls[0][1].body);
            expect(requestBody.labels).toEqual({ key1: 'value1', key2: 'value2' });
        });

        test('handles missing connection command', async () => {
            context.workerForm.name = 'test-worker';

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ token: 'test-token-456' }),
            });

            context.loadWorkers = jest.fn();

            await context.createWorker();

            expect(context.workerConnectionCommand).toBe('# Configure your worker with token: test-token-456');
        });

        test('does not create worker without name', async () => {
            context.workerForm.name = '';

            await context.createWorker();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sets saving state during request', async () => {
            context.workerForm.name = 'test-worker';
            let savingDuringRequest = false;

            fetch.mockImplementationOnce(() => {
                savingDuringRequest = context.saving;
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ token: 'test' }),
                });
            });

            context.loadWorkers = jest.fn();

            await context.createWorker();

            expect(savingDuringRequest).toBe(true);
            expect(context.saving).toBe(false);
        });

        test('handles failed response', async () => {
            context.workerForm.name = 'test-worker';

            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Worker name already exists' }),
            });

            await context.createWorker();

            expect(context.error).toBe('Worker name already exists');
            expect(window.showToast).toHaveBeenCalledWith('Worker name already exists', 'error');
            expect(context.saving).toBe(false);
        });

        test('handles network error', async () => {
            context.workerForm.name = 'test-worker';

            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.createWorker();

            expect(context.error).toBe('Network error');
            expect(window.showToast).toHaveBeenCalledWith('Failed to create worker', 'error');
            expect(context.saving).toBe(false);
        });
    });

    describe('drainWorker', () => {
        const mockWorker = { worker_id: '123', name: 'worker-1' };

        test('drains worker successfully', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
            });

            context.loadWorkers = jest.fn();

            await context.drainWorker(mockWorker);

            expect(confirm).toHaveBeenCalledWith('Drain worker worker-1? It will finish the current job and stop accepting new ones.');
            expect(fetch).toHaveBeenCalledWith('/api/admin/workers/123/drain', {
                method: 'POST',
            });
            expect(window.showToast).toHaveBeenCalledWith('Worker draining initiated', 'success');
            expect(context.loadWorkers).toHaveBeenCalled();
        });

        test('does not drain if not confirmed', async () => {
            global.confirm.mockReturnValueOnce(false);

            await context.drainWorker(mockWorker);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('handles failed response', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Worker not found' }),
            });

            await context.drainWorker(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Worker not found', 'error');
        });

        test('handles failed response without detail', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({}),
            });

            await context.drainWorker(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Failed to drain worker', 'error');
        });

        test('handles network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.drainWorker(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Failed to drain worker', 'error');
        });
    });

    describe('rotateWorkerToken', () => {
        const mockWorker = { worker_id: '456', name: 'worker-2' };

        test('rotates token successfully', async () => {
            const mockResponse = {
                token: 'new-token-789',
                connection_command: 'simpletuner worker --orchestrator-url http://localhost:8001 --worker-token new-token-789',
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            await context.rotateWorkerToken(mockWorker);

            expect(confirm).toHaveBeenCalledWith('Rotate token for worker-2? The old token will be immediately invalidated.');
            expect(fetch).toHaveBeenCalledWith('/api/admin/workers/456/token', {
                method: 'POST',
            });
            expect(context.workerToken).toBe('new-token-789');
            expect(context.workerConnectionCommand).toBe('simpletuner worker --orchestrator-url http://localhost:8001 --worker-token new-token-789');
            expect(context.workerTokenModalOpen).toBe(true);
            expect(window.showToast).toHaveBeenCalledWith('Token rotated successfully', 'success');
        });

        test('handles missing connection command', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ token: 'new-token' }),
            });

            await context.rotateWorkerToken(mockWorker);

            expect(context.workerConnectionCommand).toBe('# Configure your worker with token: new-token');
        });

        test('does not rotate if not confirmed', async () => {
            global.confirm.mockReturnValueOnce(false);

            await context.rotateWorkerToken(mockWorker);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('handles failed response', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Unauthorized' }),
            });

            await context.rotateWorkerToken(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Unauthorized', 'error');
        });

        test('handles failed response without detail', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({}),
            });

            await context.rotateWorkerToken(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Failed to rotate token', 'error');
        });

        test('handles network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.rotateWorkerToken(mockWorker);

            expect(window.showToast).toHaveBeenCalledWith('Failed to rotate token', 'error');
        });
    });

    describe('deleteWorker', () => {
        const mockWorker = { worker_id: '789', name: 'worker-3' };

        test('deletes worker successfully', async () => {
            context.deletingWorker = mockWorker;

            fetch.mockResolvedValueOnce({
                ok: true,
            });

            context.loadWorkers = jest.fn();

            await context.deleteWorker();

            expect(fetch).toHaveBeenCalledWith('/api/admin/workers/789', {
                method: 'DELETE',
            });
            expect(window.showToast).toHaveBeenCalledWith('Worker removed', 'success');
            expect(context.deleteWorkerOpen).toBe(false);
            expect(context.deletingWorker).toBeNull();
            expect(context.loadWorkers).toHaveBeenCalled();
        });

        test('does not delete if no worker selected', async () => {
            context.deletingWorker = null;

            await context.deleteWorker();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sets saving state during request', async () => {
            context.deletingWorker = mockWorker;
            let savingDuringRequest = false;

            fetch.mockImplementationOnce(() => {
                savingDuringRequest = context.saving;
                return Promise.resolve({ ok: true });
            });

            context.loadWorkers = jest.fn();

            await context.deleteWorker();

            expect(savingDuringRequest).toBe(true);
            expect(context.saving).toBe(false);
        });

        test('handles failed response', async () => {
            context.deletingWorker = mockWorker;

            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Worker is busy' }),
            });

            await context.deleteWorker();

            expect(window.showToast).toHaveBeenCalledWith('Worker is busy', 'error');
            expect(context.saving).toBe(false);
        });

        test('handles failed response without detail', async () => {
            context.deletingWorker = mockWorker;

            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({}),
            });

            await context.deleteWorker();

            expect(window.showToast).toHaveBeenCalledWith('Failed to remove worker', 'error');
        });

        test('handles network error', async () => {
            context.deletingWorker = mockWorker;

            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.deleteWorker();

            expect(window.showToast).toHaveBeenCalledWith('Failed to remove worker', 'error');
            expect(context.saving).toBe(false);
        });
    });

    describe('getWorkerStatusBadgeClass', () => {
        test('returns correct class for idle status', () => {
            expect(context.getWorkerStatusBadgeClass('idle')).toBe('bg-success');
        });

        test('returns correct class for busy status', () => {
            expect(context.getWorkerStatusBadgeClass('busy')).toBe('bg-info');
        });

        test('returns correct class for offline status', () => {
            expect(context.getWorkerStatusBadgeClass('offline')).toBe('bg-secondary');
        });

        test('returns correct class for connecting status', () => {
            expect(context.getWorkerStatusBadgeClass('connecting')).toBe('bg-warning');
        });

        test('returns correct class for draining status', () => {
            expect(context.getWorkerStatusBadgeClass('draining')).toBe('bg-warning');
        });

        test('returns default class for unknown status', () => {
            expect(context.getWorkerStatusBadgeClass('unknown')).toBe('bg-secondary');
            expect(context.getWorkerStatusBadgeClass('invalid')).toBe('bg-secondary');
            expect(context.getWorkerStatusBadgeClass('')).toBe('bg-secondary');
        });
    });

    describe('formatWorkerTimeAgo', () => {
        test('returns "Never" for null timestamp', () => {
            expect(context.formatWorkerTimeAgo(null)).toBe('Never');
            expect(context.formatWorkerTimeAgo(undefined)).toBe('Never');
            expect(context.formatWorkerTimeAgo('')).toBe('Never');
        });

        test('returns "Just now" for recent timestamp', () => {
            const now = new Date();
            expect(context.formatWorkerTimeAgo(now.toISOString())).toBe('Just now');

            const thirtySecondsAgo = new Date(now.getTime() - 30 * 1000);
            expect(context.formatWorkerTimeAgo(thirtySecondsAgo.toISOString())).toBe('Just now');
        });

        test('returns minutes ago for timestamps within an hour', () => {
            const now = new Date();
            const fiveMinutesAgo = new Date(now.getTime() - 5 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(fiveMinutesAgo.toISOString())).toBe('5m ago');

            const thirtyMinutesAgo = new Date(now.getTime() - 30 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(thirtyMinutesAgo.toISOString())).toBe('30m ago');
        });

        test('returns hours ago for timestamps within a day', () => {
            const now = new Date();
            const twoHoursAgo = new Date(now.getTime() - 2 * 60 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(twoHoursAgo.toISOString())).toBe('2h ago');

            const twelveHoursAgo = new Date(now.getTime() - 12 * 60 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(twelveHoursAgo.toISOString())).toBe('12h ago');
        });

        test('returns days ago for older timestamps', () => {
            const now = new Date();
            const twoDaysAgo = new Date(now.getTime() - 2 * 24 * 60 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(twoDaysAgo.toISOString())).toBe('2d ago');

            const tenDaysAgo = new Date(now.getTime() - 10 * 24 * 60 * 60 * 1000);
            expect(context.formatWorkerTimeAgo(tenDaysAgo.toISOString())).toBe('10d ago');
        });
    });

    describe('parseWorkerLabels', () => {
        test('parses single label correctly', () => {
            const result = context.parseWorkerLabels('key=value');
            expect(result).toEqual({ key: 'value' });
        });

        test('parses multiple labels correctly', () => {
            const result = context.parseWorkerLabels('gpu=nvidia,region=us-west,tier=premium');
            expect(result).toEqual({
                gpu: 'nvidia',
                region: 'us-west',
                tier: 'premium',
            });
        });

        test('handles labels with spaces', () => {
            const result = context.parseWorkerLabels('key1 = value1 , key2 = value2');
            expect(result).toEqual({
                key1: 'value1',
                key2: 'value2',
            });
        });

        test('returns empty object for null input', () => {
            expect(context.parseWorkerLabels(null)).toEqual({});
            expect(context.parseWorkerLabels(undefined)).toEqual({});
            expect(context.parseWorkerLabels('')).toEqual({});
        });

        test('ignores invalid pairs', () => {
            const result = context.parseWorkerLabels('valid=yes,invalid,also=valid');
            expect(result).toEqual({
                valid: 'yes',
                also: 'valid',
            });
        });

        test('ignores pairs without value', () => {
            const result = context.parseWorkerLabels('key1=,key2=value2');
            expect(result).toEqual({
                key2: 'value2',
            });
        });

        test('ignores pairs without key', () => {
            const result = context.parseWorkerLabels('=value1,key2=value2');
            expect(result).toEqual({
                key2: 'value2',
            });
        });
    });

    describe('startWorkerAutoRefresh', () => {
        test('starts auto-refresh interval', () => {
            jest.useFakeTimers();
            context.loadWorkers = jest.fn();

            context.startWorkerAutoRefresh();

            expect(context.workerRefreshInterval).not.toBeNull();

            // Fast-forward 30 seconds
            jest.advanceTimersByTime(30000);

            expect(context.loadWorkers).toHaveBeenCalledTimes(1);

            jest.useRealTimers();
        });

        test('refreshes workers periodically when on workers tab', () => {
            jest.useFakeTimers();
            context.loadWorkers = jest.fn();
            context.activeTab = 'workers';

            context.startWorkerAutoRefresh();

            // Fast-forward 90 seconds (3 intervals)
            jest.advanceTimersByTime(90000);

            expect(context.loadWorkers).toHaveBeenCalledTimes(3);

            jest.useRealTimers();
        });

        test('does not refresh when on different tab', () => {
            jest.useFakeTimers();
            context.loadWorkers = jest.fn();
            context.activeTab = 'audit';

            context.startWorkerAutoRefresh();

            // Fast-forward 60 seconds
            jest.advanceTimersByTime(60000);

            expect(context.loadWorkers).not.toHaveBeenCalled();

            jest.useRealTimers();
        });

        test('clears existing interval before creating new one', () => {
            jest.useFakeTimers();
            context.loadWorkers = jest.fn();

            // Start first interval
            context.startWorkerAutoRefresh();
            const firstInterval = context.workerRefreshInterval;

            // Start second interval
            context.startWorkerAutoRefresh();
            const secondInterval = context.workerRefreshInterval;

            expect(firstInterval).not.toBe(secondInterval);

            // Should only have one active interval
            jest.advanceTimersByTime(30000);
            expect(context.loadWorkers).toHaveBeenCalledTimes(1);

            jest.useRealTimers();
        });
    });

    describe('stopWorkerAutoRefresh', () => {
        test('stops auto-refresh interval', () => {
            jest.useFakeTimers();
            context.loadWorkers = jest.fn();

            context.startWorkerAutoRefresh();
            context.stopWorkerAutoRefresh();

            expect(context.workerRefreshInterval).toBeNull();

            // Fast-forward time - should not call loadWorkers
            jest.advanceTimersByTime(60000);
            expect(context.loadWorkers).not.toHaveBeenCalled();

            jest.useRealTimers();
        });

        test('handles null interval gracefully', () => {
            context.workerRefreshInterval = null;

            expect(() => context.stopWorkerAutoRefresh()).not.toThrow();
            expect(context.workerRefreshInterval).toBeNull();
        });

        test('clears interval and sets to null', () => {
            jest.useFakeTimers();

            context.startWorkerAutoRefresh();
            expect(context.workerRefreshInterval).not.toBeNull();

            context.stopWorkerAutoRefresh();
            expect(context.workerRefreshInterval).toBeNull();

            jest.useRealTimers();
        });
    });

    describe('copyWorkerToken', () => {
        test('copies token to clipboard', async () => {
            context.workerToken = 'test-token-123';

            await context.copyWorkerToken();

            expect(navigator.clipboard.writeText).toHaveBeenCalledWith('test-token-123');
            expect(window.showToast).toHaveBeenCalledWith('Token copied to clipboard', 'success');
        });

        test('does nothing if no token', async () => {
            context.workerToken = null;

            await context.copyWorkerToken();

            expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
        });

        test('handles clipboard error', async () => {
            console.error.mockClear();
            context.workerToken = 'test-token';
            const error = new Error('Permission denied');
            navigator.clipboard.writeText.mockRejectedValueOnce(error);

            // Call the function and wait for the promise to settle
            context.copyWorkerToken();
            // Wait for all microtasks to complete
            await new Promise(resolve => setTimeout(resolve, 0));

            expect(console.error).toHaveBeenCalledWith('Failed to copy token:', error);
            expect(console.error).toHaveBeenCalledTimes(1);
        });
    });

    describe('copyWorkerCommand', () => {
        test('copies command to clipboard', async () => {
            context.workerConnectionCommand = 'worker connect --token test-token';

            await context.copyWorkerCommand();

            expect(navigator.clipboard.writeText).toHaveBeenCalledWith('worker connect --token test-token');
            expect(window.showToast).toHaveBeenCalledWith('Command copied to clipboard', 'success');
        });

        test('does nothing if no command', async () => {
            context.workerConnectionCommand = null;

            await context.copyWorkerCommand();

            expect(navigator.clipboard.writeText).not.toHaveBeenCalled();
        });

        test('handles clipboard error', async () => {
            console.error.mockClear();
            context.workerConnectionCommand = 'worker connect';
            const error = new Error('Permission denied');
            navigator.clipboard.writeText.mockRejectedValueOnce(error);

            // Call the function and wait for the promise to settle
            context.copyWorkerCommand();
            // Wait for all microtasks to complete
            await new Promise(resolve => setTimeout(resolve, 0));

            expect(console.error).toHaveBeenCalledWith('Failed to copy command:', error);
            expect(console.error).toHaveBeenCalledTimes(1);
        });
    });

    describe('showAddWorkerModal', () => {
        test('opens modal with default form values', () => {
            context.showAddWorkerModal();

            expect(context.workerFormOpen).toBe(true);
            expect(context.workerForm).toEqual({
                name: '',
                worker_type: 'persistent',
                labels_str: '',
            });
        });

        test('resets form when opening modal', () => {
            context.workerForm = {
                name: 'old-worker',
                worker_type: 'ephemeral',
                labels_str: 'old=labels',
            };

            context.showAddWorkerModal();

            expect(context.workerForm).toEqual({
                name: '',
                worker_type: 'persistent',
                labels_str: '',
            });
        });
    });

    describe('confirmDeleteWorker', () => {
        test('sets deleting worker and opens modal', () => {
            const mockWorker = { worker_id: '123', name: 'test-worker' };

            context.confirmDeleteWorker(mockWorker);

            expect(context.deletingWorker).toBe(mockWorker);
            expect(context.deleteWorkerOpen).toBe(true);
        });
    });
});

describe('workersPageComponent', () => {
    test('is defined as a function', () => {
        expect(typeof window.workersPageComponent).toBe('function');
    });

    test('returns an object with required state properties', () => {
        const component = window.workersPageComponent();

        expect(component.workers).toEqual([]);
        expect(component.workersLoading).toBe(false);
        expect(component.workerStats).toEqual({ total: 0, idle: 0, busy: 0, offline: 0 });
        expect(component.saving).toBe(false);
        expect(component.error).toBeNull();
    });

    test('returns an object with modal state properties', () => {
        const component = window.workersPageComponent();

        expect(component.workerFormOpen).toBe(false);
        expect(component.workerForm).toEqual({ name: '', worker_type: 'persistent', labels_str: '' });
        expect(component.workerTokenModalOpen).toBe(false);
        expect(component.workerToken).toBe('');
        expect(component.workerConnectionCommand).toBe('');
        expect(component.deleteWorkerOpen).toBe(false);
        expect(component.deletingWorker).toBeNull();
    });

    test('returns an object with auto-refresh state', () => {
        const component = window.workersPageComponent();

        expect(component.workerRefreshInterval).toBeNull();
    });

    test('includes init method', () => {
        const component = window.workersPageComponent();

        expect(typeof component.init).toBe('function');
    });

    test('includes destroy method', () => {
        const component = window.workersPageComponent();

        expect(typeof component.destroy).toBe('function');
    });

    test('includes all adminWorkerMethods', () => {
        const component = window.workersPageComponent();

        // Check that key methods from adminWorkerMethods are present
        expect(typeof component.loadWorkers).toBe('function');
        expect(typeof component.updateWorkerStats).toBe('function');
        expect(typeof component.createWorker).toBe('function');
        expect(typeof component.drainWorker).toBe('function');
        expect(typeof component.rotateWorkerToken).toBe('function');
        expect(typeof component.deleteWorker).toBe('function');
        expect(typeof component.getWorkerStatusBadgeClass).toBe('function');
        expect(typeof component.formatWorkerTimeAgo).toBe('function');
        expect(typeof component.parseWorkerLabels).toBe('function');
        expect(typeof component.showAddWorkerModal).toBe('function');
        expect(typeof component.confirmDeleteWorker).toBe('function');
        expect(typeof component.copyWorkerToken).toBe('function');
        expect(typeof component.copyWorkerCommand).toBe('function');
        expect(typeof component.startWorkerAutoRefresh).toBe('function');
        expect(typeof component.stopWorkerAutoRefresh).toBe('function');
    });

    test('init calls loadWorkers and startWorkerAutoRefresh', async () => {
        const component = window.workersPageComponent();

        // Mock the methods
        component.loadWorkers = jest.fn().mockResolvedValue();
        component.startWorkerAutoRefresh = jest.fn();

        await component.init();

        expect(component.loadWorkers).toHaveBeenCalled();
        expect(component.startWorkerAutoRefresh).toHaveBeenCalled();
    });

    test('destroy calls stopWorkerAutoRefresh', () => {
        const component = window.workersPageComponent();

        component.stopWorkerAutoRefresh = jest.fn();

        component.destroy();

        expect(component.stopWorkerAutoRefresh).toHaveBeenCalled();
    });

    test('each instance is independent', () => {
        const component1 = window.workersPageComponent();
        const component2 = window.workersPageComponent();

        component1.workers = [{ id: 1 }];
        component2.workers = [{ id: 2 }, { id: 3 }];

        expect(component1.workers).toHaveLength(1);
        expect(component2.workers).toHaveLength(2);
    });
});
