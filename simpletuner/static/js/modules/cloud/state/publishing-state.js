/**
 * Publishing & Config State Factory
 *
 * State for configuration, publishing, webhooks, and advanced settings.
 */

window.cloudPublishingStateFactory = function(initial) {
    const initialData = initial || {};
    return {
        availableConfigs: [],
        selectedConfigName: null,
        webhookUrl: initialData.webhook_url || '',
        webhookTesting: false,
        webhookTestMode: null,
        webhookTestResult: null,
        publishingStatus: {
            loading: false,
            hf_configured: false,
            hf_token_valid: false,
            hf_username: null,
            hub_model_id: null,
            push_to_hub: false,
            s3_configured: false,
            local_upload_available: false,
            local_upload_dir: null,
            message: null,
        },
        advancedConfig: {
            loading: false,
            saving: false,
            ssl_verify: true,
            ssl_ca_bundle: '',
            proxy_url: '',
            http_timeout: 30,
            webhook_ip_allowlist_enabled: false,
            webhook_allowed_ips: [],
            newIpEntry: '',
            ipValidationError: null,
            sslWarningAcknowledged: false,
        },
    };
};
