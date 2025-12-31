/**
 * Tests for cloudProviderMethods.
 *
 * cloudProviderMethods handles provider loading, API key validation,
 * token management, and cost limit configuration.
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
require('../../simpletuner/static/js/modules/cloud/providers.js');

describe('cloudProviderMethods', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        context = {
            providers: [],
            providersLoading: false,
            activeProvider: 'replicate',
            providerConfig: {},
            webhookUrl: '',
            costLimit: {
                loading: false,
                saving: false,
                status: null,
            },
            versions: [],
            versionsLoading: false,
            versionsError: null,
            apiKeyState: {
                loading: false,
                valid: false,
                error: null,
                userInfo: null,
            },
            tokenInput: '',
            tokenSaving: false,
            tokenError: null,
            tokenSuccess: false,
            currentUser: null,
            hasAdminAccess: false,
            publishingStatus: {
                loading: false,
            },
            availableConfigs: [],
            pendingApprovals: {
                loading: false,
                count: 0,
                lastLoaded: null,
            },
            getActiveProvider: jest.fn(() => ({ configured: true })),
            loadProviderConfig: jest.fn(),
            loadCostLimitStatus: jest.fn(),
            loadProviders: jest.fn(),
            validateApiKey: jest.fn(),
            loadCurrentUser: jest.fn(),
        };
    });

    describe('loadProviders', () => {
        test('sets providersLoading during fetch', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ providers: [] }),
            });

            const promise = window.cloudProviderMethods.loadProviders.call(context);
            expect(context.providersLoading).toBe(true);

            await promise;
            expect(context.providersLoading).toBe(false);
        });

        test('populates providers from response', async () => {
            const mockProviders = [
                { name: 'replicate', configured: true },
                { name: 'runpod', configured: false },
            ];

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ providers: mockProviders }),
            });

            await window.cloudProviderMethods.loadProviders.call(context);

            expect(context.providers).toEqual(mockProviders);
        });

        test('calls loadProviderConfig after loading', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ providers: [] }),
            });

            // Use the real loadProviderConfig for this test
            const realLoadProviderConfig = window.cloudProviderMethods.loadProviderConfig;
            context.loadProviderConfig = jest.fn();

            await window.cloudProviderMethods.loadProviders.call(context);

            expect(context.loadProviderConfig).toHaveBeenCalled();
        });

        test('handles API error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await window.cloudProviderMethods.loadProviders.call(context);

            expect(context.providersLoading).toBe(false);
            expect(console.error).toHaveBeenCalled();
        });
    });

    describe('loadProviderConfig', () => {
        test('returns early if no activeProvider', async () => {
            context.activeProvider = null;

            await window.cloudProviderMethods.loadProviderConfig.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('fetches config for active provider', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    webhook_url: 'https://webhook.example.com',
                    version_override: 'v1.0.0',
                }),
            });

            // Use real loadCostLimitStatus for this test
            context.loadCostLimitStatus = jest.fn();

            await window.cloudProviderMethods.loadProviderConfig.call(context);

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config');
            expect(context.webhookUrl).toBe('https://webhook.example.com');
        });

        test('calls loadCostLimitStatus after loading config', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            context.loadCostLimitStatus = jest.fn();

            await window.cloudProviderMethods.loadProviderConfig.call(context);

            expect(context.loadCostLimitStatus).toHaveBeenCalled();
        });
    });

    describe('loadCostLimitStatus', () => {
        test('sets loading state during fetch', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ enabled: true }),
            });

            const promise = window.cloudProviderMethods.loadCostLimitStatus.call(context);
            expect(context.costLimit.loading).toBe(true);

            await promise;
            expect(context.costLimit.loading).toBe(false);
        });

        test('stores status from response', async () => {
            const mockStatus = {
                enabled: true,
                limit_amount: 100,
                period: 'monthly',
                current_spend: 45.50,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStatus),
            });

            await window.cloudProviderMethods.loadCostLimitStatus.call(context);

            expect(context.costLimit.status).toEqual(mockStatus);
        });
    });

    describe('updateCostLimitSetting', () => {
        test('maps UI field names to config field names', async () => {
            fetch.mockResolvedValueOnce({ ok: true });
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            context.loadCostLimitStatus = jest.fn();

            await window.cloudProviderMethods.updateCostLimitSetting.call(context, 'enabled', true);

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cost_limit_enabled: true }),
            });
        });

        test('shows success toast on update', async () => {
            fetch.mockResolvedValueOnce({ ok: true });

            context.loadCostLimitStatus = jest.fn();

            await window.cloudProviderMethods.updateCostLimitSetting.call(context, 'limit_amount', 50);

            expect(window.showToast).toHaveBeenCalledWith('Cost limit settings updated', 'success');
        });

        test('shows error toast on failure', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: false,
                json: () => Promise.resolve({ detail: 'Invalid value' }),
            }));

            await window.cloudProviderMethods.updateCostLimitSetting.call(context, 'limit_amount', -1);

            expect(window.showToast).toHaveBeenCalledWith('Invalid value', 'error');
        });

        test('sets saving flag during operation', async () => {
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadCostLimitStatus = jest.fn();

            const promise = window.cloudProviderMethods.updateCostLimitSetting.call(context, 'enabled', true);
            expect(context.costLimit.saving).toBe(true);

            await promise;
            expect(context.costLimit.saving).toBe(false);
        });
    });

    describe('loadVersions', () => {
        test('sets loading state', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ versions: [] }),
            });

            const promise = window.cloudProviderMethods.loadVersions.call(context);
            expect(context.versionsLoading).toBe(true);

            await promise;
            expect(context.versionsLoading).toBe(false);
        });

        test('populates versions from response', async () => {
            const mockVersions = ['v1.0.0', 'v1.1.0', 'v2.0.0'];

            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ versions: mockVersions }),
            }));

            context.versions = [];
            context.versionsLoading = false;
            context.versionsError = null;

            await window.cloudProviderMethods.loadVersions.call(context);

            expect(context.versions).toEqual(mockVersions);
        });

        test('sets error on failure', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({ ok: false }));

            context.versions = [];
            context.versionsLoading = false;
            context.versionsError = null;

            await window.cloudProviderMethods.loadVersions.call(context);

            expect(context.versionsError).toBe('Failed to load versions');
        });

        test('sets error on network failure', async () => {
            fetch.mockImplementationOnce(() => Promise.reject(new Error('Network error')));

            context.versions = [];
            context.versionsLoading = false;
            context.versionsError = null;

            await window.cloudProviderMethods.loadVersions.call(context);

            expect(context.versionsError).toBe('Network error loading versions');
        });
    });

    describe('saveVersionOverride', () => {
        test('sends version to config endpoint', async () => {
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviderConfig = jest.fn();

            await window.cloudProviderMethods.saveVersionOverride.call(context, 'v1.0.0');

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ version_override: 'v1.0.0' }),
            });
        });

        test('sends null when clearing version', async () => {
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviderConfig = jest.fn();

            await window.cloudProviderMethods.saveVersionOverride.call(context, '');

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ version_override: null }),
            });
        });

        test('shows appropriate toast for version or latest', async () => {
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviderConfig = jest.fn();

            await window.cloudProviderMethods.saveVersionOverride.call(context, 'v1.0.0');
            expect(window.showToast).toHaveBeenCalledWith('Version override saved', 'success');

            fetch.mockResolvedValueOnce({ ok: true });
            await window.cloudProviderMethods.saveVersionOverride.call(context, '');
            expect(window.showToast).toHaveBeenCalledWith('Using latest version', 'success');
        });
    });

    describe('validateApiKey', () => {
        test('returns early if already loading', async () => {
            context.apiKeyState.loading = true;

            await window.cloudProviderMethods.validateApiKey.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sets valid to false if provider not configured', async () => {
            context.getActiveProvider = jest.fn(() => ({ configured: false }));

            await window.cloudProviderMethods.validateApiKey.call(context);

            expect(context.apiKeyState.valid).toBe(false);
            expect(fetch).not.toHaveBeenCalled();
        });

        test('validates API key and stores result', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ valid: true, username: 'testuser' }),
            }));

            context.apiKeyState = {
                loading: false,
                valid: false,
                error: null,
                userInfo: null,
            };

            await window.cloudProviderMethods.validateApiKey.call(context);

            expect(context.apiKeyState.valid).toBe(true);
            expect(context.apiKeyState.userInfo).toBe('testuser');
        });

        test('sets error on validation failure', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({ ok: false }));

            context.apiKeyState = {
                loading: false,
                valid: false,
                error: null,
                userInfo: null,
            };

            await window.cloudProviderMethods.validateApiKey.call(context);

            expect(context.apiKeyState.valid).toBe(false);
            expect(context.apiKeyState.error).toBe('API key validation failed');
        });

        test('handles network error', async () => {
            fetch.mockImplementationOnce(() => Promise.reject(new Error('Network error')));

            context.apiKeyState = {
                loading: false,
                valid: false,
                error: null,
                userInfo: null,
            };

            await window.cloudProviderMethods.validateApiKey.call(context);

            expect(context.apiKeyState.valid).toBe(false);
            expect(context.apiKeyState.error).toBe('Network error');
        });
    });

    describe('saveReplicateToken', () => {
        test('requires non-empty token', async () => {
            context.tokenInput = '   ';

            await window.cloudProviderMethods.saveReplicateToken.call(context);

            expect(context.tokenError).toBe('API token is required');
            expect(fetch).not.toHaveBeenCalled();
        });

        test('sends token to API', async () => {
            context.tokenInput = 'r8_test_token';
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviders = jest.fn();
            context.validateApiKey = jest.fn();
            context.loadCurrentUser = jest.fn();

            await window.cloudProviderMethods.saveReplicateToken.call(context);

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/token', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ api_token: 'r8_test_token' }),
            });
        });

        test('clears input and shows success on save', async () => {
            context.tokenInput = 'r8_test_token';
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviders = jest.fn();
            context.validateApiKey = jest.fn();
            context.loadCurrentUser = jest.fn();

            await window.cloudProviderMethods.saveReplicateToken.call(context);

            expect(context.tokenInput).toBe('');
            expect(context.tokenSuccess).toBe(true);
            expect(window.showToast).toHaveBeenCalledWith('Replicate API token saved successfully', 'success');
        });

        test('sets error from response on failure', async () => {
            context.tokenInput = 'invalid_token';
            context.tokenSaving = false;
            context.tokenError = null;
            context.tokenSuccess = false;

            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: false,
                json: () => Promise.resolve({ detail: 'Invalid token format' }),
            }));

            await window.cloudProviderMethods.saveReplicateToken.call(context);

            expect(context.tokenError).toBe('Invalid token format');
        });
    });

    describe('deleteReplicateToken', () => {
        test('requires confirmation', async () => {
            confirm.mockReturnValueOnce(false);

            await window.cloudProviderMethods.deleteReplicateToken.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('sends DELETE request', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviders = jest.fn();

            await window.cloudProviderMethods.deleteReplicateToken.call(context);

            expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/token', {
                method: 'DELETE',
            });
        });

        test('clears API key state on success', async () => {
            confirm.mockReturnValueOnce(true);
            fetch.mockResolvedValueOnce({ ok: true });
            context.loadProviders = jest.fn();
            context.apiKeyState.valid = true;
            context.apiKeyState.userInfo = 'testuser';

            await window.cloudProviderMethods.deleteReplicateToken.call(context);

            expect(context.apiKeyState.valid).toBe(false);
            expect(context.apiKeyState.userInfo).toBeNull();
        });
    });

    describe('loadCurrentUser', () => {
        test('fetches and stores user info', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({
                    user: { id: 'user-1', name: 'Test User', is_admin: true },
                }),
            }));

            context.currentUser = null;
            context.hasAdminAccess = false;

            await window.cloudProviderMethods.loadCurrentUser.call(context);

            expect(context.currentUser).toEqual({ id: 'user-1', name: 'Test User', is_admin: true });
            expect(context.hasAdminAccess).toBe(true);
        });

        test('sets hasAdminAccess based on is_admin', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({
                    user: { id: 'user-1', is_admin: false },
                }),
            }));

            context.currentUser = null;
            context.hasAdminAccess = true;

            await window.cloudProviderMethods.loadCurrentUser.call(context);

            expect(context.hasAdminAccess).toBe(false);
        });
    });

    describe('loadAvailableConfigs', () => {
        test('fetches and stores configs', async () => {
            const mockConfigs = ['config1', 'config2', 'config3'];

            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ configs: mockConfigs }),
            }));

            context.availableConfigs = [];

            await window.cloudProviderMethods.loadAvailableConfigs.call(context);

            expect(context.availableConfigs).toEqual(mockConfigs);
        });

        test('handles missing configs field', async () => {
            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({}),
            }));

            context.availableConfigs = ['old'];

            await window.cloudProviderMethods.loadAvailableConfigs.call(context);

            expect(context.availableConfigs).toEqual([]);
        });
    });

    describe('loadPendingApprovals', () => {
        test('prevents concurrent loads', async () => {
            context.pendingApprovals.loading = true;

            await window.cloudProviderMethods.loadPendingApprovals.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('counts blocked queue entries', async () => {
            const mockEntries = [
                { id: 1 }, { id: 2 }, { id: 3 },
            ];

            fetch.mockImplementationOnce(() => Promise.resolve({
                ok: true,
                json: () => Promise.resolve({ entries: mockEntries }),
            }));

            context.pendingApprovals = {
                loading: false,
                count: 0,
                lastLoaded: null,
            };

            await window.cloudProviderMethods.loadPendingApprovals.call(context);

            expect(context.pendingApprovals.count).toBe(3);
            expect(context.pendingApprovals.lastLoaded).toBeInstanceOf(Date);
        });

        test('sets loading state', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ entries: [] }),
            });

            const promise = window.cloudProviderMethods.loadPendingApprovals.call(context);
            expect(context.pendingApprovals.loading).toBe(true);

            await promise;
            expect(context.pendingApprovals.loading).toBe(false);
        });
    });
});
