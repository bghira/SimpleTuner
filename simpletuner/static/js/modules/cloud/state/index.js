/**
 * Cloud Dashboard State Composition
 *
 * Composes all state factory modules into a single state object.
 * This file must be loaded after all state modules.
 *
 * Loading order:
 * 1. setup-state.js
 * 2. provider-state.js
 * 3. jobs-state.js
 * 4. metrics-state.js
 * 5. publishing-state.js
 * 6. system-status-state.js
 * 7. queue-state.js
 * 8. upload-state.js
 * 9. ui-state.js
 * 10. modals-state.js
 * 11. connection-state.js
 * 12. index.js (this file)
 */

window.cloudStateFactory = function(initialData) {
    const initial = initialData || {};

    // Compose all state modules
    return {
        // First-run setup state
        setupState: window.cloudSetupStateFactory(),

        // Provider configuration state
        ...window.cloudProviderStateFactory(),

        // Jobs list and filtering state
        ...window.cloudJobsStateFactory(),

        // Metrics and billing state
        ...window.cloudMetricsStateFactory(),

        // Publishing and config state (requires initial data for webhook_url)
        ...window.cloudPublishingStateFactory(initial),

        // System health and status state
        ...window.cloudSystemStatusStateFactory(),

        // Queue management state
        ...window.cloudQueueStateFactory(),

        // Upload and submission state
        ...window.cloudUploadStateFactory(),

        // UI toggles and display settings
        ...window.cloudUIStateFactory(),

        // Modal states
        ...window.cloudModalsStateFactory(),

        // Auth and connection state
        ...window.cloudConnectionStateFactory(),
    };
};
