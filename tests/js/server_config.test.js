const fs = require('fs');
const path = require('path');
const source = fs.readFileSync(path.join(__dirname, '../../simpletuner/static/js/server-config.js'), 'utf8');
const health = mode => ({ ok: true, json: async () => ({ mode }) });

describe('Server configuration', () => {
    beforeEach(() => {
        jest.spyOn(document, 'readyState', 'get').mockReturnValue('loading');
        jest.spyOn(document, 'addEventListener').mockImplementation(() => {});
        window.eval(source);
        window.ServerConfig.apiBaseUrl = 'http://localhost:8000';
        window.ServerConfig.callbackUrl = 'http://localhost:8000';
        AbortSignal.timeout = jest.fn(() => new AbortController().signal);
    });

    afterEach(() => jest.restoreAllMocks());

    test('tries the second callback port after the first connection fails', async () => {
        fetch = jest.fn().mockResolvedValueOnce(health('trainer'))
            .mockRejectedValueOnce(new TypeError('Connection refused'))
            .mockResolvedValueOnce(health('callback'));
        await window.ServerConfig.init();
        expect(fetch).toHaveBeenCalledTimes(3);
        expect(window.ServerConfig.callbackUrl).toBe('http://localhost:8001');
        expect(window.ServerConfig.isReady).toBe(true);
    });

    test('does not select a trainer as a callback server', async () => {
        fetch = jest.fn().mockResolvedValueOnce(health('trainer'))
            .mockResolvedValueOnce(health('trainer'))
            .mockResolvedValueOnce(health('unified'));
        await window.ServerConfig.init();
        expect(window.ServerConfig.callbackUrl).toBe('http://localhost:8001');
    });

    test.each([
        ['/api/models/wizard', 'api'],
        ['/models/flux/flavours', 'api'],
        ['/api/training/events/stream', 'api'],
        ['/api/jobs?name=/events', 'api'],
        ['/callback', 'callback'],
        ['/broadcast', 'callback'],
        ['/api/events?limit=10', 'callback'],
        ['/api/events/recent', 'callback'],
    ])('routes %s to the %s server', (endpoint, target) => {
        Object.assign(window.ServerConfig, { mode: 'trainer', apiBaseUrl: 'https://api.test', callbackUrl: 'https://callback.test' });
        expect(window.ServerConfig.getEndpointUrl(endpoint)).toBe(`https://${target}.test${endpoint}`);
    });

    test('unified mode keeps callback requests on the API server', () => {
        Object.assign(window.ServerConfig, { mode: 'unified', apiBaseUrl: 'https://api.test', callbackUrl: 'https://callback.test' });
        expect(window.ServerConfig.getEndpointUrl('/api/events')).toBe('https://api.test/api/events');
    });
});
