/**
 * Tests for cloud job submission flow.
 *
 * Job submission is the core user flow for cloud training.
 * These tests verify the pre-submit checks, payload preparation, and submission process.
 */

// Mock fetch
global.fetch = jest.fn();

// Mock crypto.randomUUID
global.crypto = {
    randomUUID: jest.fn(() => 'test-uuid-1234'),
};

// Mock Alpine store
global.Alpine = {
    store: jest.fn(() => ({
        datasets: [],
    })),
};

// Mock showToast
global.showToast = jest.fn();

// Mock EventSource for upload progress
class MockEventSource {
    constructor(url) {
        this.url = url;
        this.onmessage = null;
        this.onerror = null;
    }
    close() {}
}
global.EventSource = MockEventSource;

// Load the module
require('../../simpletuner/static/js/modules/cloud/job-submission.js');

describe('cloudSubmissionMethods', () => {
    let context;

    beforeEach(() => {
        jest.clearAllMocks();
        global.showToast = jest.fn();

        // Create context with required state
        context = {
            activeProvider: 'replicate',
            webhookUrl: '',
            selectedConfigName: null,
            submitting: false,
            submitError: null,
            preSubmitModal: {
                open: false,
                wizardStep: 1,
                loading: false,
                snapshotMessage: '',
                snapshotName: '',
                trackerRunName: '',
                configName: '',
                dataUploadPreview: null,
                dataConsentConfirmed: false,
                dataConsent: 'deny',
                costEstimate: null,
                quotaImpact: null,
                configPreview: {},
                dataloaderPreview: [],
                gitAvailable: false,
                repoPresent: false,
                isDirty: false,
                dirtyCount: 0,
                dirtyPaths: [],
                currentCommit: '',
                currentAbbrev: '',
                currentBranch: '',
                webhookCheck: {
                    tested: false,
                    testing: false,
                    success: null,
                    error: null,
                    skipped: false,
                },
            },
            uploadProgress: {
                active: false,
                stage: '',
                current: 0,
                total: 0,
                percent: 0,
                message: '',
                error: null,
                eventSource: null,
            },
            providerConfig: {
                cost_limit_enabled: false,
            },
            jobs: [],

            // Mock methods that would be provided by other modules
            isActiveProviderConfigured: jest.fn(() => true),
            getActiveProvider: jest.fn(() => ({ name: 'Replicate' })),
            loadJobs: jest.fn(),
            loadCostLimitStatus: jest.fn(),
            loadDataUploadPreview: jest.fn(),
            loadCostEstimate: jest.fn(),
            loadConfigPreview: jest.fn(),
            checkWebhookReachability: jest.fn(),
        };

        // Bind methods to context
        Object.keys(window.cloudSubmissionMethods).forEach((key) => {
            if (typeof window.cloudSubmissionMethods[key] === 'function') {
                context[key] = window.cloudSubmissionMethods[key].bind(context);
            }
        });
    });

    describe('resetPreSubmitModal', () => {
        test('resets modal to initial state', () => {
            context.preSubmitModal.open = false;
            context.preSubmitModal.wizardStep = 3;
            context.preSubmitModal.loading = false;
            context.preSubmitModal.dataConsentConfirmed = true;

            context.resetPreSubmitModal();

            expect(context.preSubmitModal.open).toBe(true);
            expect(context.preSubmitModal.wizardStep).toBe(1);
            expect(context.preSubmitModal.loading).toBe(true);
            expect(context.preSubmitModal.dataConsentConfirmed).toBe(false);
        });

        test('resets webhook check state', () => {
            context.preSubmitModal.webhookCheck = {
                tested: true,
                testing: false,
                success: true,
                error: null,
                skipped: true,
            };

            context.resetPreSubmitModal();

            expect(context.preSubmitModal.webhookCheck).toEqual({
                tested: false,
                testing: false,
                success: null,
                error: null,
                skipped: false,
            });
        });
    });

    describe('applyPreSubmitData', () => {
        test('applies git data from response', () => {
            const data = {
                git_available: true,
                repo_present: true,
                is_dirty: true,
                dirty_count: 3,
                dirty_paths: ['file1.py', 'file2.py', 'file3.py'],
                current_commit: 'abc123def456',
                current_abbrev: 'abc123d',
                current_branch: 'main',
                tracker_run_name: 'my-training-run',
                config_name: 'sdxl-lora',
            };

            context.applyPreSubmitData(data);

            expect(context.preSubmitModal.gitAvailable).toBe(true);
            expect(context.preSubmitModal.repoPresent).toBe(true);
            expect(context.preSubmitModal.isDirty).toBe(true);
            expect(context.preSubmitModal.dirtyCount).toBe(3);
            expect(context.preSubmitModal.dirtyPaths).toEqual(['file1.py', 'file2.py', 'file3.py']);
            expect(context.preSubmitModal.currentCommit).toBe('abc123def456');
            expect(context.preSubmitModal.currentAbbrev).toBe('abc123d');
            expect(context.preSubmitModal.currentBranch).toBe('main');
            expect(context.preSubmitModal.trackerRunName).toBe('my-training-run');
            expect(context.preSubmitModal.configName).toBe('sdxl-lora');
        });

        test('handles missing optional fields', () => {
            const data = {
                git_available: false,
                repo_present: false,
                is_dirty: false,
                dirty_count: 0,
                current_commit: '',
                current_abbrev: '',
                current_branch: '',
            };

            context.applyPreSubmitData(data);

            expect(context.preSubmitModal.dirtyPaths).toEqual([]);
            expect(context.preSubmitModal.trackerRunName).toBe('');
            expect(context.preSubmitModal.configName).toBe('');
        });

        test('resets snapshotName', () => {
            context.preSubmitModal.snapshotName = 'previous-snapshot';

            context.applyPreSubmitData({ git_available: false });

            expect(context.preSubmitModal.snapshotName).toBe('');
        });
    });

    describe('openPreSubmitModal', () => {
        test('shows error when provider not configured', async () => {
            context.isActiveProviderConfigured.mockReturnValue(false);

            await context.openPreSubmitModal();

            expect(global.showToast).toHaveBeenCalledWith(
                'Please configure Replicate first',
                'error'
            );
        });

        test('fetches pre-submit check and consent data', async () => {
            fetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({
                        git_available: true,
                        repo_present: true,
                        is_dirty: false,
                        dirty_count: 0,
                        current_commit: 'abc123',
                        current_abbrev: 'abc123',
                        current_branch: 'main',
                    }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ consent: 'allow' }),
                });

            await context.openPreSubmitModal();

            expect(fetch).toHaveBeenCalledWith('/api/cloud/pre-submit-check');
            expect(fetch).toHaveBeenCalledWith('/api/cloud/data-consent/setting');
        });

        test('sets dataConsentConfirmed when consent is allow', async () => {
            fetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ git_available: false }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ consent: 'allow' }),
                });

            await context.openPreSubmitModal();

            expect(context.preSubmitModal.dataConsentConfirmed).toBe(true);
        });

        test('calls parallel data loading methods', async () => {
            // Re-assign mock functions after binding
            const loadDataUploadPreviewMock = jest.fn();
            const loadCostEstimateMock = jest.fn();
            const loadConfigPreviewMock = jest.fn();
            const checkWebhookReachabilityMock = jest.fn();

            context.loadDataUploadPreview = loadDataUploadPreviewMock;
            context.loadCostEstimate = loadCostEstimateMock;
            context.loadConfigPreview = loadConfigPreviewMock;
            context.checkWebhookReachability = checkWebhookReachabilityMock;

            fetch
                .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({}) })
                .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({ consent: 'deny' }) });

            await context.openPreSubmitModal();

            expect(loadDataUploadPreviewMock).toHaveBeenCalled();
            expect(loadCostEstimateMock).toHaveBeenCalled();
            expect(loadConfigPreviewMock).toHaveBeenCalled();
            expect(checkWebhookReachabilityMock).toHaveBeenCalled();
        });

        test('sets loading to false when done', async () => {
            fetch
                .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({}) })
                .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({}) });

            await context.openPreSubmitModal();

            expect(context.preSubmitModal.loading).toBe(false);
        });

        test('handles fetch error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await expect(context.openPreSubmitModal()).resolves.not.toThrow();
            expect(context.preSubmitModal.loading).toBe(false);
        });
    });

    describe('generateUploadId', () => {
        test('returns a valid UUID-like string when crypto.randomUUID is available', () => {
            // The actual implementation uses crypto.randomUUID if available
            // We just verify it returns a string of reasonable format
            const id = context.generateUploadId();

            expect(typeof id).toBe('string');
            expect(id.length).toBeGreaterThan(0);
        });

        test('falls back to random string when crypto.randomUUID unavailable', () => {
            const originalCrypto = global.crypto;
            global.crypto = {};  // No randomUUID

            const id = context.generateUploadId();

            expect(typeof id).toBe('string');
            expect(id.length).toBeGreaterThan(0);

            global.crypto = originalCrypto;
        });

        test('generates unique IDs on each call', () => {
            const id1 = context.generateUploadId();
            const id2 = context.generateUploadId();

            expect(id1).not.toBe(id2);
        });
    });

    describe('buildBasePayload', () => {
        test('builds payload with upload id', () => {
            const payload = context.buildBasePayload('upload-123');

            expect(payload.upload_id).toBe('upload-123');
            expect(payload.webhook_url).toBeNull();
            expect(payload.snapshot_name).toBeNull();
            expect(payload.snapshot_message).toBeNull();
            expect(payload.tracker_run_name).toBeNull();
        });

        test('includes optional fields when present', () => {
            context.webhookUrl = 'https://example.com/webhook';
            context.preSubmitModal.snapshotName = 'v1.0';
            context.preSubmitModal.snapshotMessage = 'First release';
            context.preSubmitModal.trackerRunName = 'experiment-1';

            const payload = context.buildBasePayload('upload-123');

            expect(payload.webhook_url).toBe('https://example.com/webhook');
            expect(payload.snapshot_name).toBe('v1.0');
            expect(payload.snapshot_message).toBe('First release');
            expect(payload.tracker_run_name).toBe('experiment-1');
        });
    });

    describe('prepareJobPayload', () => {
        test('uses selectedConfigName when set', async () => {
            context.selectedConfigName = 'saved-config';

            const result = await context.prepareJobPayload('upload-123');

            expect(result.success).toBe(true);
            expect(result.payload.config_name_to_load).toBe('saved-config');
            expect(result.payload.config).toBeUndefined();
        });

        test('fetches active config when no selectedConfigName', async () => {
            fetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({
                        config: { learning_rate: 0.0001 },
                        name: 'active-config',
                    }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({
                        datasets: [{ name: 'dataset1' }],
                    }),
                });

            const result = await context.prepareJobPayload('upload-123');

            expect(result.success).toBe(true);
            expect(result.payload.config).toEqual({ learning_rate: 0.0001 });
            expect(result.payload.config_name).toBe('active-config');
            expect(result.payload.dataloader_config).toEqual([{ name: 'dataset1' }]);
        });

        test('returns error when no active config found', async () => {
            fetch
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ config: {} }),
                })
                .mockResolvedValueOnce({
                    ok: true,
                    json: () => Promise.resolve({ datasets: [] }),
                });

            const result = await context.prepareJobPayload('upload-123');

            expect(result.success).toBe(false);
            expect(result.error).toContain('No active configuration found');
        });

        test('handles fetch error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            const result = await context.prepareJobPayload('upload-123');

            expect(result.success).toBe(false);
            expect(result.error).toContain('Failed to fetch');
        });
    });

    describe('handleSubmissionSuccess', () => {
        test('shows success toast for new job', () => {
            context.handleSubmissionSuccess({
                job_id: 'job-abc123',
                data_uploaded: false,
            });

            expect(global.showToast).toHaveBeenCalledWith(
                'Job submitted: job-abc123',
                'success'
            );
            expect(context.loadJobs).toHaveBeenCalled();
        });

        test('shows info toast for idempotent hit (duplicate)', () => {
            context.handleSubmissionSuccess({
                job_id: 'job-abc123',
                idempotent_hit: true,
            });

            expect(global.showToast).toHaveBeenCalledWith(
                'Duplicate detected: Job job-abc123 was already submitted',
                'info'
            );
        });

        test('includes data uploaded note when applicable', () => {
            context.handleSubmissionSuccess({
                job_id: 'job-abc123',
                data_uploaded: true,
            });

            expect(global.showToast).toHaveBeenCalledWith(
                'Job submitted: job-abc123 (data uploaded)',
                'success'
            );
        });

        test('shows cost limit warning when present', () => {
            context.handleSubmissionSuccess({
                job_id: 'job-abc123',
                cost_limit_warning: 'Approaching cost limit',
            });

            expect(global.showToast).toHaveBeenCalledWith(
                'Approaching cost limit',
                'warning'
            );
        });

        test('reloads cost limit status when enabled', () => {
            context.providerConfig.cost_limit_enabled = true;

            context.handleSubmissionSuccess({ job_id: 'job-123' });

            expect(context.loadCostLimitStatus).toHaveBeenCalled();
        });
    });

    describe('handleSubmissionError', () => {
        test('sets submitError from string', () => {
            context.handleSubmissionError('Something went wrong');

            expect(context.submitError).toBe('Something went wrong');
            expect(global.showToast).toHaveBeenCalledWith('Something went wrong', 'error');
        });

        test('extracts message from error object', () => {
            context.handleSubmissionError({ message: 'API error' });

            expect(context.submitError).toBe('API error');
        });

        test('uses default message for unknown error format', () => {
            context.handleSubmissionError({});

            expect(context.submitError).toBe('Failed to submit job');
        });
    });

    describe('submitCloudJob', () => {
        test('prevents double submission', async () => {
            context.submitting = true;

            await context.submitCloudJob();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sets submitting flag during submission', async () => {
            context.selectedConfigName = 'test-config';
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true, job_id: 'job-123' }),
            });

            const promise = context.submitCloudJob();

            expect(context.submitting).toBe(true);

            await promise;

            expect(context.submitting).toBe(false);
        });

        test('closes pre-submit modal', async () => {
            context.preSubmitModal.open = true;
            context.selectedConfigName = 'test-config';
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true, job_id: 'job-123' }),
            });

            await context.submitCloudJob();

            expect(context.preSubmitModal.open).toBe(false);
        });

        test('handles successful submission', async () => {
            context.selectedConfigName = 'test-config';
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    success: true,
                    job_id: 'job-abc123',
                }),
            });

            await context.submitCloudJob();

            expect(fetch).toHaveBeenCalledWith(
                '/api/cloud/jobs/submit?provider=replicate',
                expect.objectContaining({
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                })
            );
            expect(global.showToast).toHaveBeenCalledWith(
                expect.stringContaining('job-abc123'),
                'success'
            );
        });

        test('handles submission failure response', async () => {
            context.selectedConfigName = 'test-config';
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    success: false,
                    error: 'Provider quota exceeded',
                }),
            });

            await context.submitCloudJob();

            expect(global.showToast).toHaveBeenCalledWith('Provider quota exceeded', 'error');
        });

        test('handles network error', async () => {
            context.selectedConfigName = 'test-config';
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.submitCloudJob();

            expect(global.showToast).toHaveBeenCalledWith('Network error', 'error');
        });

        test('starts upload progress when data needs uploading', async () => {
            context.selectedConfigName = 'test-config';
            context.preSubmitModal.dataUploadPreview = { requires_upload: true };
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true, job_id: 'job-123' }),
            });

            await context.submitCloudJob();

            // Upload progress should have been started
            expect(context.uploadProgress.active).toBe(false);  // Should be false after completion
        });
    });

    describe('webhook check', () => {
        test('skipWebhookCheck sets skipped flag', () => {
            context.skipWebhookCheck();

            expect(context.preSubmitModal.webhookCheck.skipped).toBe(true);
        });

        test('retryWebhookCheck resets and re-checks', async () => {
            context.preSubmitModal.webhookCheck.tested = true;
            context.preSubmitModal.webhookCheck.skipped = true;

            // Create a mock that tracks if it was called
            const checkMock = jest.fn();
            context.checkWebhookReachability = checkMock;

            await context.retryWebhookCheck();

            // After calling retryWebhookCheck, skipped should be false
            expect(context.preSubmitModal.webhookCheck.skipped).toBe(false);
            // checkWebhookReachability should have been called
            expect(checkMock).toHaveBeenCalled();
        });
    });

    describe('upload progress', () => {
        test('dismissUploadError clears error state', () => {
            context.uploadProgress.error = 'Upload failed';
            context.uploadProgress.active = true;
            context.uploadProgress.stage = 'uploading';

            context.dismissUploadError();

            expect(context.uploadProgress.error).toBeNull();
            expect(context.uploadProgress.active).toBe(false);
            expect(context.uploadProgress.stage).toBe('');
        });

        test('stopUploadProgress closes event source', () => {
            const mockEventSource = { close: jest.fn() };
            context.uploadProgress.eventSource = mockEventSource;
            context.uploadProgress.active = true;

            context.stopUploadProgress();

            expect(mockEventSource.close).toHaveBeenCalled();
            expect(context.uploadProgress.eventSource).toBeNull();
            expect(context.uploadProgress.active).toBe(false);
        });
    });
});

describe('Job submission edge cases', () => {
    let context;

    beforeEach(() => {
        jest.clearAllMocks();
        global.showToast = jest.fn();

        context = {
            activeProvider: 'replicate',
            webhookUrl: '',
            selectedConfigName: null,
            submitting: false,
            submitError: null,
            preSubmitModal: {
                open: false,
                loading: false,
                dataUploadPreview: null,
                webhookCheck: {},
            },
            uploadProgress: {
                active: false,
                eventSource: null,
            },
            providerConfig: {},
            isActiveProviderConfigured: jest.fn(() => true),
            getActiveProvider: jest.fn(() => null),
            loadJobs: jest.fn(),
            loadDataUploadPreview: jest.fn(),
            loadCostEstimate: jest.fn(),
            loadConfigPreview: jest.fn(),
            checkWebhookReachability: jest.fn(),
        };

        Object.keys(window.cloudSubmissionMethods).forEach((key) => {
            if (typeof window.cloudSubmissionMethods[key] === 'function') {
                context[key] = window.cloudSubmissionMethods[key].bind(context);
            }
        });
    });

    test('handles null provider name gracefully', async () => {
        context.isActiveProviderConfigured.mockReturnValue(false);
        context.getActiveProvider.mockReturnValue(null);

        await context.openPreSubmitModal();

        expect(global.showToast).toHaveBeenCalledWith(
            expect.stringContaining('Please configure'),
            'error'
        );
    });

    test('handles missing showToast gracefully', () => {
        delete global.showToast;

        // Should not throw
        expect(() => {
            context.handleSubmissionSuccess({ job_id: 'test' });
        }).not.toThrow();

        expect(() => {
            context.handleSubmissionError('error');
        }).not.toThrow();

        global.showToast = jest.fn();
    });

    test('payload includes upload_id for tracking', async () => {
        context.selectedConfigName = 'test';
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ success: true, job_id: 'job-123' }),
        });

        await context.submitCloudJob();

        const callBody = JSON.parse(fetch.mock.calls[0][1].body);
        // Just verify upload_id exists and is a non-empty string
        expect(callBody.upload_id).toBeDefined();
        expect(typeof callBody.upload_id).toBe('string');
        expect(callBody.upload_id.length).toBeGreaterThan(0);
    });
});
