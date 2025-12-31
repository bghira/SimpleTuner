/**
 * Tests for cloudUploadStateFactory.
 *
 * cloudUploadStateFactory creates the state for job submission,
 * upload progress, and pre-submit modal.
 */

// Mock localStorage
const localStorageMock = (() => {
    let store = {};
    return {
        getItem: jest.fn((key) => store[key] || null),
        setItem: jest.fn((key, value) => {
            store[key] = value;
        }),
        removeItem: jest.fn((key) => {
            delete store[key];
        }),
        clear: () => {
            store = {};
        },
    };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Load the module
require('../../simpletuner/static/js/modules/cloud/state/upload-state.js');

describe('cloudUploadStateFactory', () => {
    beforeEach(() => {
        localStorageMock.clear();
        jest.clearAllMocks();
    });

    describe('initial state', () => {
        test('returns object with all required properties', () => {
            const state = window.cloudUploadStateFactory();

            expect(state).toHaveProperty('submitting');
            expect(state).toHaveProperty('submitError');
            expect(state).toHaveProperty('quickSubmitMode');
            expect(state).toHaveProperty('uploadProgress');
            expect(state).toHaveProperty('preSubmitModal');
        });

        test('submitting starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.submitting).toBe(false);
        });

        test('submitError starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.submitError).toBeNull();
        });

        test('quickSubmitMode reads from localStorage', () => {
            localStorageMock.getItem.mockReturnValueOnce('true');

            const state = window.cloudUploadStateFactory();

            expect(localStorageMock.getItem).toHaveBeenCalledWith('cloud_quick_submit_mode');
            expect(state.quickSubmitMode).toBe(true);
        });

        test('quickSubmitMode defaults to false when not in localStorage', () => {
            localStorageMock.getItem.mockReturnValueOnce(null);

            const state = window.cloudUploadStateFactory();

            expect(state.quickSubmitMode).toBe(false);
        });
    });

    describe('uploadProgress state', () => {
        test('has all required properties', () => {
            const state = window.cloudUploadStateFactory();
            const progress = state.uploadProgress;

            expect(progress).toHaveProperty('active');
            expect(progress).toHaveProperty('stage');
            expect(progress).toHaveProperty('current');
            expect(progress).toHaveProperty('total');
            expect(progress).toHaveProperty('percent');
            expect(progress).toHaveProperty('message');
            expect(progress).toHaveProperty('error');
            expect(progress).toHaveProperty('estimatedTimeRemaining');
            expect(progress).toHaveProperty('eventSource');
        });

        test('active starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.active).toBe(false);
        });

        test('stage starts empty', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.stage).toBe('');
        });

        test('current starts at 0', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.current).toBe(0);
        });

        test('total starts at 0', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.total).toBe(0);
        });

        test('percent starts at 0', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.percent).toBe(0);
        });

        test('message starts empty', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.message).toBe('');
        });

        test('error starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.error).toBeNull();
        });

        test('estimatedTimeRemaining starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.estimatedTimeRemaining).toBeNull();
        });

        test('eventSource starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.uploadProgress.eventSource).toBeNull();
        });
    });

    describe('preSubmitModal state', () => {
        test('has all required properties', () => {
            const state = window.cloudUploadStateFactory();
            const modal = state.preSubmitModal;

            expect(modal).toHaveProperty('open');
            expect(modal).toHaveProperty('loading');
            expect(modal).toHaveProperty('wizardStep');
            expect(modal).toHaveProperty('gitAvailable');
            expect(modal).toHaveProperty('repoPresent');
            expect(modal).toHaveProperty('isDirty');
            expect(modal).toHaveProperty('dirtyCount');
            expect(modal).toHaveProperty('dirtyPaths');
            expect(modal).toHaveProperty('currentCommit');
            expect(modal).toHaveProperty('currentAbbrev');
            expect(modal).toHaveProperty('currentBranch');
            expect(modal).toHaveProperty('snapshotName');
            expect(modal).toHaveProperty('snapshotMessage');
            expect(modal).toHaveProperty('trackerRunName');
            expect(modal).toHaveProperty('configName');
            expect(modal).toHaveProperty('dataConsent');
            expect(modal).toHaveProperty('dataUploadPreview');
            expect(modal).toHaveProperty('dataConsentConfirmed');
            expect(modal).toHaveProperty('costEstimate');
            expect(modal).toHaveProperty('quotaImpact');
            expect(modal).toHaveProperty('configPreview');
            expect(modal).toHaveProperty('dataloaderPreview');
            expect(modal).toHaveProperty('webhookCheck');
        });

        test('open starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.open).toBe(false);
        });

        test('loading starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.loading).toBe(false);
        });

        test('wizardStep starts at 1', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.wizardStep).toBe(1);
        });

        test('git-related properties start correctly', () => {
            const state = window.cloudUploadStateFactory();
            const modal = state.preSubmitModal;

            expect(modal.gitAvailable).toBe(false);
            expect(modal.repoPresent).toBe(false);
            expect(modal.isDirty).toBe(false);
            expect(modal.dirtyCount).toBe(0);
            expect(modal.dirtyPaths).toEqual([]);
            expect(modal.currentCommit).toBeNull();
            expect(modal.currentAbbrev).toBeNull();
            expect(modal.currentBranch).toBeNull();
        });

        test('snapshot properties start empty', () => {
            const state = window.cloudUploadStateFactory();
            const modal = state.preSubmitModal;

            expect(modal.snapshotName).toBe('');
            expect(modal.snapshotMessage).toBe('');
        });

        test('dataConsent starts as "ask"', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.dataConsent).toBe('ask');
        });

        test('dataConsentConfirmed starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.dataConsentConfirmed).toBe(false);
        });

        test('costEstimate starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.costEstimate).toBeNull();
        });

        test('quotaImpact starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.quotaImpact).toBeNull();
        });

        test('configPreview starts as empty object', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.configPreview).toEqual({});
        });

        test('dataloaderPreview starts as empty array', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.dataloaderPreview).toEqual([]);
        });
    });

    describe('webhookCheck state', () => {
        test('has all required properties', () => {
            const state = window.cloudUploadStateFactory();
            const webhook = state.preSubmitModal.webhookCheck;

            expect(webhook).toHaveProperty('tested');
            expect(webhook).toHaveProperty('testing');
            expect(webhook).toHaveProperty('success');
            expect(webhook).toHaveProperty('error');
            expect(webhook).toHaveProperty('skipped');
        });

        test('tested starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.webhookCheck.tested).toBe(false);
        });

        test('testing starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.webhookCheck.testing).toBe(false);
        });

        test('success starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.webhookCheck.success).toBeNull();
        });

        test('error starts as null', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.webhookCheck.error).toBeNull();
        });

        test('skipped starts as false', () => {
            const state = window.cloudUploadStateFactory();

            expect(state.preSubmitModal.webhookCheck.skipped).toBe(false);
        });
    });

    describe('factory behavior', () => {
        test('returns new instance each call', () => {
            const state1 = window.cloudUploadStateFactory();
            const state2 = window.cloudUploadStateFactory();

            expect(state1).not.toBe(state2);
        });

        test('instances are independent', () => {
            const state1 = window.cloudUploadStateFactory();
            const state2 = window.cloudUploadStateFactory();

            state1.submitting = true;
            state1.uploadProgress.active = true;
            state1.preSubmitModal.wizardStep = 5;

            expect(state2.submitting).toBe(false);
            expect(state2.uploadProgress.active).toBe(false);
            expect(state2.preSubmitModal.wizardStep).toBe(1);
        });

        test('deeply nested objects are independent', () => {
            const state1 = window.cloudUploadStateFactory();
            const state2 = window.cloudUploadStateFactory();

            state1.preSubmitModal.dirtyPaths.push('/path/to/file');
            state1.preSubmitModal.webhookCheck.tested = true;

            expect(state2.preSubmitModal.dirtyPaths).toEqual([]);
            expect(state2.preSubmitModal.webhookCheck.tested).toBe(false);
        });
    });
});

describe('cloudUploadStateFactory localStorage integration', () => {
    beforeEach(() => {
        localStorageMock.clear();
        jest.clearAllMocks();
    });

    test('quickSubmitMode is true when localStorage returns "true"', () => {
        localStorageMock.getItem.mockReturnValueOnce('true');

        const state = window.cloudUploadStateFactory();

        expect(state.quickSubmitMode).toBe(true);
    });

    test('quickSubmitMode is false when localStorage returns "false"', () => {
        localStorageMock.getItem.mockReturnValueOnce('false');

        const state = window.cloudUploadStateFactory();

        expect(state.quickSubmitMode).toBe(false);
    });

    test('quickSubmitMode is false when localStorage returns non-"true"', () => {
        localStorageMock.getItem.mockReturnValueOnce('something-else');

        const state = window.cloudUploadStateFactory();

        expect(state.quickSubmitMode).toBe(false);
    });
});
