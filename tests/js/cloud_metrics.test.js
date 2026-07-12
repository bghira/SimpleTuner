/**
 * Tests for cloud metrics and webhook configuration methods.
 */

global.fetch = jest.fn();

global.console = {
    ...console,
    error: jest.fn(),
};

global.showToast = jest.fn();
window.showToast = global.showToast;

require('../../simpletuner/static/js/modules/cloud/metrics.js');

describe('cloudMetricsMethods webhook configuration', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        context = {
            webhookUrl: '',
            savedWebhookUrl: '',
            configSaving: false,
            publishingStatus: {
                local_upload_available: false,
                local_upload_dir: null,
            },
        };
    });

    test('saves valid webhook and marks local upload configured after success', async () => {
        context.webhookUrl = ' https://webhook.example.com ';
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({
                provider: 'replicate',
                config: { webhook_url: 'https://webhook.example.com' },
            }),
        });

        const result = await window.cloudMetricsMethods.saveWebhookConfig.call(context);

        expect(result).toBe(true);
        expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ webhook_url: 'https://webhook.example.com' }),
        });
        expect(context.savedWebhookUrl).toBe('https://webhook.example.com');
        expect(context.webhookUrl).toBe('https://webhook.example.com');
        expect(context.publishingStatus.local_upload_available).toBe(true);
        expect(window.showToast).toHaveBeenCalledWith('Webhook configuration saved', 'success');
    });

    test('keeps draft webhook visible when save fails validation', async () => {
        context.webhookUrl = 'h';
        context.savedWebhookUrl = '';
        fetch.mockResolvedValueOnce({
            ok: false,
            json: () => Promise.resolve({ detail: 'Invalid webhook URL: Invalid URL format' }),
        });

        const result = await window.cloudMetricsMethods.saveWebhookConfig.call(context);

        expect(result).toBe(false);
        expect(context.webhookUrl).toBe('h');
        expect(context.savedWebhookUrl).toBe('');
        expect(context.publishingStatus.local_upload_available).toBe(false);
        expect(window.showToast).toHaveBeenCalledWith('Invalid webhook URL: Invalid URL format', 'error');
    });

    test('sends empty string when clearing webhook configuration', async () => {
        context.webhookUrl = '';
        context.savedWebhookUrl = 'https://old-webhook.example.com';
        context.publishingStatus.local_upload_available = true;
        context.publishingStatus.local_upload_dir = '/tmp/outputs';
        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({
                provider: 'replicate',
                config: { webhook_url: null },
            }),
        });

        const result = await window.cloudMetricsMethods.saveWebhookConfig.call(context);

        expect(result).toBe(true);
        expect(fetch).toHaveBeenCalledWith('/api/cloud/providers/replicate/config', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ webhook_url: '' }),
        });
        expect(context.savedWebhookUrl).toBe('');
        expect(context.webhookUrl).toBe('');
        expect(context.publishingStatus.local_upload_available).toBe(false);
        expect(context.publishingStatus.local_upload_dir).toBeNull();
    });
});
