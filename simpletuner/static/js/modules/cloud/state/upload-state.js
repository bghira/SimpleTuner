/**
 * Upload & Submission State Factory
 *
 * State for job submission, upload progress, and pre-submit modal.
 */

window.cloudUploadStateFactory = function() {
    return {
        submitting: false,
        submitError: null,
        quickSubmitMode: localStorage.getItem('cloud_quick_submit_mode') === 'true',
        uploadProgress: {
            active: false,
            stage: '',
            current: 0,
            total: 0,
            percent: 0,
            message: '',
            error: null,
            estimatedTimeRemaining: null,
            eventSource: null,
        },
        preSubmitModal: {
            open: false,
            loading: false,
            wizardStep: 1,
            gitAvailable: false,
            repoPresent: false,
            isDirty: false,
            dirtyCount: 0,
            dirtyPaths: [],
            currentCommit: null,
            currentAbbrev: null,
            currentBranch: null,
            snapshotName: '',
            snapshotMessage: '',
            trackerRunName: '',
            configName: '',
            dataConsent: 'ask',
            dataUploadPreview: null,
            dataConsentConfirmed: false,
            costEstimate: null,
            quotaImpact: null,  // Quota impact from /api/quotas/cost-estimate
            configPreview: {},
            dataloaderPreview: [],
            webhookCheck: {
                tested: false,
                testing: false,
                success: null,
                error: null,
                skipped: false,
            },
        },
    };
};
