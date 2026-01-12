/**
 * Tests for ApiClient utility.
 *
 * ApiClient is the foundation for all API calls in the application.
 * It handles URL resolution across different deployment modes (local, split frontend/backend).
 */

// Store original values at module load
const ORIGINAL_LOCATION = global.location;
const ORIGINAL_SERVER_CONFIG = global.ServerConfig;

describe('ApiClient', () => {
    beforeAll(() => {
        // Clear any prior module cache before all tests
        delete require.cache[require.resolve('../../simpletuner/static/js/utils/api.js')];
    });

    beforeEach(() => {
        // Reset location before each test
        global.location = { origin: 'http://localhost:3000' };
        // Clear ServerConfig
        delete global.ServerConfig;

        // Reload the module fresh for each test to pick up new global.location
        delete require.cache[require.resolve('../../simpletuner/static/js/utils/api.js')];
        require('../../simpletuner/static/js/utils/api.js');
    });

    afterEach(() => {
        // Clean up
        delete global.ServerConfig;
    });

    afterAll(() => {
        // Restore original values after all tests
        global.location = ORIGINAL_LOCATION;
        global.ServerConfig = ORIGINAL_SERVER_CONFIG;
    });

    describe('normalizePath (via resolve)', () => {
        test('adds leading slash to paths without one', () => {
            const result = window.ApiClient.resolve('api/test');
            // Should end with the normalized path
            expect(result).toMatch(/\/api\/test$/);
            expect(result).not.toContain('//api');  // No double slash
        });

        test('preserves leading slash on paths that have one', () => {
            const result = window.ApiClient.resolve('/api/test');
            expect(result).toMatch(/\/api\/test$/);
        });

        test('handles empty string by returning root', () => {
            const result = window.ApiClient.resolve('');
            expect(result.endsWith('/')).toBe(true);
        });
    });

    describe('apiBaseUrl', () => {
        test('uses location.origin when no ServerConfig', () => {
            // Should match whatever location.origin is set to
            expect(window.ApiClient.apiBaseUrl).toBe(global.location.origin);
        });

        test('uses ServerConfig.apiBaseUrl when available', () => {
            global.ServerConfig = { apiBaseUrl: 'https://api.example.com' };
            expect(window.ApiClient.apiBaseUrl).toBe('https://api.example.com');
        });

        test('falls back to origin when ServerConfig.apiBaseUrl is empty', () => {
            global.ServerConfig = { apiBaseUrl: '' };
            expect(window.ApiClient.apiBaseUrl).toBe(global.location.origin);
        });
    });

    describe('callbackBaseUrl', () => {
        test('uses apiBaseUrl when no callbackUrl configured', () => {
            expect(window.ApiClient.callbackBaseUrl).toBe(global.location.origin);
        });

        test('uses ServerConfig.callbackUrl when available', () => {
            global.ServerConfig = { callbackUrl: 'https://callback.example.com' };
            expect(window.ApiClient.callbackBaseUrl).toBe('https://callback.example.com');
        });
    });

    describe('resolve', () => {
        test('resolves path with default options', () => {
            const result = window.ApiClient.resolve('/api/jobs');
            expect(result).toBe(`${global.location.origin}/api/jobs`);
        });

        test('uses forceApi option to always use apiBaseUrl', () => {
            global.ServerConfig = {
                apiBaseUrl: 'https://api.example.com',
                callbackUrl: 'https://callback.example.com',
            };

            const result = window.ApiClient.resolve('/webhook', { forceApi: true });
            expect(result).toBe('https://api.example.com/webhook');
        });

        test('uses forceCallback option to always use callbackBaseUrl', () => {
            global.ServerConfig = {
                apiBaseUrl: 'https://api.example.com',
                callbackUrl: 'https://callback.example.com',
            };

            const result = window.ApiClient.resolve('/notify', { forceCallback: true });
            expect(result).toBe('https://callback.example.com/notify');
        });

        test('uses ServerConfig.getEndpointUrl when available', () => {
            global.ServerConfig = {
                getEndpointUrl: jest.fn((path) => `https://custom.example.com${path}`),
            };

            const result = window.ApiClient.resolve('/api/test');
            expect(global.ServerConfig.getEndpointUrl).toHaveBeenCalledWith('/api/test');
            expect(result).toBe('https://custom.example.com/api/test');
        });

        test('forceApi takes precedence over getEndpointUrl', () => {
            global.ServerConfig = {
                apiBaseUrl: 'https://api.example.com',
                getEndpointUrl: jest.fn(),
            };

            window.ApiClient.resolve('/api/test', { forceApi: true });
            expect(global.ServerConfig.getEndpointUrl).not.toHaveBeenCalled();
        });

        test('forceCallback takes precedence over getEndpointUrl', () => {
            global.ServerConfig = {
                callbackUrl: 'https://callback.example.com',
                getEndpointUrl: jest.fn(),
            };

            window.ApiClient.resolve('/api/test', { forceCallback: true });
            expect(global.ServerConfig.getEndpointUrl).not.toHaveBeenCalled();
        });
    });

    describe('resolveWebsocket', () => {
        test('converts https to wss', () => {
            global.ServerConfig = { apiBaseUrl: 'https://api.example.com' };
            const result = window.ApiClient.resolveWebsocket('/ws/events');
            expect(result).toBe('wss://api.example.com/ws/events');
        });

        test('converts http to ws', () => {
            const result = window.ApiClient.resolveWebsocket('/ws/events');
            // Use dynamic origin to handle test isolation
            const expectedOrigin = global.location.origin.replace(/^http/, 'ws');
            expect(result).toBe(`${expectedOrigin}/ws/events`);
        });

        test('passes options through to resolve', () => {
            global.ServerConfig = {
                apiBaseUrl: 'https://api.example.com',
                callbackUrl: 'https://callback.example.com',
            };

            const result = window.ApiClient.resolveWebsocket('/ws', { forceCallback: true });
            expect(result).toBe('wss://callback.example.com/ws');
        });
    });

    describe('fetch', () => {
        beforeEach(() => {
            global.fetch = jest.fn().mockResolvedValue({ ok: true });
        });

        test('calls fetch with resolved URL', async () => {
            await window.ApiClient.fetch('/api/jobs');

            expect(global.fetch).toHaveBeenCalledWith(
                `${global.location.origin}/api/jobs`,
                {}
            );
        });

        test('passes fetch options through', async () => {
            const options = {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{"test": true}',
            };

            await window.ApiClient.fetch('/api/jobs', options);

            expect(global.fetch).toHaveBeenCalledWith(
                `${global.location.origin}/api/jobs`,
                options
            );
        });

        test('passes resolve options through', async () => {
            global.ServerConfig = { callbackUrl: 'https://callback.example.com' };

            await window.ApiClient.fetch('/notify', {}, { forceCallback: true });

            expect(global.fetch).toHaveBeenCalledWith(
                'https://callback.example.com/notify',
                {}
            );
        });
    });
});

describe('ApiClient edge cases', () => {
    beforeEach(() => {
        global.location = { origin: 'http://localhost:3000' };
        delete global.ServerConfig;
        delete require.cache[require.resolve('../../simpletuner/static/js/utils/api.js')];
        require('../../simpletuner/static/js/utils/api.js');
    });

    test('handles missing location gracefully', () => {
        // When location is undefined, getBaseOrigin returns ""
        delete global.location;
        // Should not throw, but will return empty base
        expect(() => window.ApiClient.apiBaseUrl).not.toThrow();
        // Result will be just "" when no location
        expect(window.ApiClient.apiBaseUrl).toBe('');
    });

    test('handles null path by normalizing to root', () => {
        // Null path normalizes to "/"
        const result = window.ApiClient.resolve(null);
        expect(result).toBe('http://localhost:3000/');
    });

    test('handles undefined path by normalizing to root', () => {
        const result = window.ApiClient.resolve(undefined);
        expect(result).toBe('http://localhost:3000/');
    });

    test('handles empty string path by normalizing to root', () => {
        const result = window.ApiClient.resolve('');
        expect(result).toBe('http://localhost:3000/');
    });

    test('handles numeric path by normalizing to root', () => {
        // Non-string input normalizes to "/"
        const result = window.ApiClient.resolve(123);
        expect(result).toBe('http://localhost:3000/');
    });
});
