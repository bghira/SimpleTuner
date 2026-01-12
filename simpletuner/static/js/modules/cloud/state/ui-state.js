/**
 * UI State Factory
 *
 * State for UI toggles, onboarding, hints, and display settings.
 */

window.cloudUIStateFactory = function() {
    return {
        showAdvancedSettings: false,
        showWebhookSecuritySettings: false,
        showSettingsPanel: false,
        hints: {
            dataloader_dismissed: false,
            git_dismissed: false,
        },
        onboarding: {
            data_understood: false,
            results_understood: false,
            cost_understood: false,
        },
        quickCostLimitEnabled: false,
        quickCostLimitAmount: 50,
        quickCostLimitPeriod: 'monthly',
        setupStatus: {
            datasetsConfigured: false,
            activeConfigExists: false,
            outputConfigured: false,
        },
        datasetStatus: {
            loading: false,
            configured: false,
            count: 0,
        },
        sheetDragging: false,
        sheetStartY: 0,
        sheetCurrentY: 0,
        sheetTransformStyle: '',
    };
};
