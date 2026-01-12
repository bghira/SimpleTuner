/**
 * Provider State Factory
 *
 * State for cloud provider configuration, API keys, and versions.
 */

window.cloudProviderStateFactory = function() {
    return {
        activeProvider: 'replicate',
        providersLoading: true,
        providers: [],
        providerConfig: {},
        configSaving: false,
        tokenInput: '',
        tokenSaving: false,
        tokenError: null,
        tokenSuccess: false,
        apiKeyState: {
            loading: false,
            valid: null,
            userInfo: null,
            error: null,
        },
        versions: [],
        versionsLoading: false,
        versionsError: null,
    };
};
