/**
 * Tests for dataset wizard Alpine.js component.
 *
 * Covers the functionality previously tested via Selenium E2E tests:
 * - Component initialization and state
 * - New folder dialog management
 * - Upload modal management
 * - Caption modal management
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
        clear: jest.fn(() => {
            store = {};
        }),
    };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Mock Alpine store
global.Alpine = {
    data: jest.fn((name, factory) => {
        global.Alpine._components = global.Alpine._components || {};
        global.Alpine._components[name] = factory;
    }),
    store: jest.fn(() => ({
        modelContext: {},
        configValues: {},
    })),
    start: jest.fn(),
    _components: {},
};

// Mock $nextTick
const mockNextTick = jest.fn((cb) => setTimeout(cb, 0));

// Mock HintMixin (used by datasetWizardComponent for hero CTA)
global.window = global.window || {};
window.HintMixin = {
    createMultiHint: jest.fn(({ hintKeys }) => {
        const hints = {};
        hintKeys.forEach((key) => {
            hints[key] = true;
        });
        return {
            hints,
            hintsLoading: false,
            loadHints: jest.fn(),
            dismissHint: jest.fn((key) => {
                hints[key] = false;
            }),
            showHint: jest.fn((key) => {
                hints[key] = true;
            }),
            restoreAllHints: jest.fn(),
            anyHintsDismissed: function () {
                return hintKeys.some((key) => !this.hints[key]);
            },
            _saveHintsToStorage: jest.fn(),
        };
    }),
};

// Load the dataset wizard module
require('../../simpletuner/static/js/dataset-wizard.js');

describe('datasetWizardComponent', () => {
    let component;

    beforeEach(() => {
        jest.clearAllMocks();
        localStorageMock.clear();

        // Get the component factory from window
        const factory = window.datasetWizardComponent;
        expect(factory).toBeDefined();

        // Create component instance
        component = factory();

        // Add mock $nextTick and $refs
        component.$nextTick = mockNextTick;
        component.$refs = {
            newFolderInput: { focus: jest.fn() },
            fileInput: { value: '' },
            zipInput: { value: '' },
        };
    });

    describe('component initialization', () => {
        test('initializes with correct default state', () => {
            // Modal state
            expect(component.wizardOpen).toBe(false);
            expect(component.wizardStep).toBe(1);
            expect(component.saving).toBe(false);

            // New folder state
            expect(component.showNewFolderInput).toBe(false);
            expect(component.newFolderName).toBe('');
            expect(component.newFolderError).toBeNull();

            // Upload modal state
            expect(component.uploadModalOpen).toBe(false);
            expect(component.selectedUploadFiles).toEqual([]);
            expect(component.uploading).toBe(false);

            // Caption modal state
            expect(component.captionModalOpen).toBe(false);
            expect(component.captionStatus).toEqual({
                total_images: 0,
                with_caption: 0,
                without_caption: 0,
                coverage_ratio: 0,
            });
            expect(component.pendingCaptions).toEqual({});
        });

        test('has all required fields for Alpine state management', () => {
            const requiredFields = [
                'showNewFolderInput',
                'newFolderName',
                'newFolderError',
                'uploadModalOpen',
                'selectedUploadFiles',
                'captionModalOpen',
                'captionStatus',
                'pendingCaptions',
            ];

            requiredFields.forEach((field) => {
                expect(field in component).toBe(true);
                expect(typeof component[field]).not.toBe('undefined');
            });
        });
    });

    describe('new folder dialog', () => {
        test('openNewFolderDialog sets correct state', () => {
            component.openNewFolderDialog();

            expect(component.showNewFolderInput).toBe(true);
            expect(component.newFolderName).toBe('');
            expect(component.newFolderError).toBeNull();
        });

        test('openNewFolderDialog focuses input via $nextTick', () => {
            component.openNewFolderDialog();

            expect(mockNextTick).toHaveBeenCalled();
        });

        test('cancelNewFolder resets state', () => {
            // First open the dialog
            component.showNewFolderInput = true;
            component.newFolderName = 'test-folder';
            component.newFolderError = 'some error';

            // Cancel
            component.cancelNewFolder();

            expect(component.showNewFolderInput).toBe(false);
            expect(component.newFolderName).toBe('');
            expect(component.newFolderError).toBeNull();
        });
    });

    describe('upload modal', () => {
        test('openUploadModal sets correct state', () => {
            component.openUploadModal();

            expect(component.uploadModalOpen).toBe(true);
            expect(component.uploadTab).toBe('files');
            expect(component.selectedUploadFiles).toEqual([]);
            expect(component.uploading).toBe(false);
            expect(component.uploadProgress).toBe(0);
            expect(component.uploadResult).toBeNull();
        });

        test('closeUploadModal resets state when not uploading', () => {
            // Open first
            component.uploadModalOpen = true;
            component.selectedUploadFiles = [{ name: 'test.png' }];
            component.uploadResult = { success: true };

            // Close
            component.closeUploadModal();

            expect(component.uploadModalOpen).toBe(false);
            expect(component.selectedUploadFiles).toEqual([]);
            expect(component.uploadResult).toBeNull();
        });

        test('closeUploadModal does not close while uploading', () => {
            component.uploadModalOpen = true;
            component.uploading = true;

            component.closeUploadModal();

            // Should still be open
            expect(component.uploadModalOpen).toBe(true);
        });
    });

    describe('caption modal', () => {
        test('has caption modal state initialized', () => {
            expect(component.captionModalOpen).toBe(false);
            expect(component.captionStatus).toBeDefined();
            expect(component.captionStatus.total_images).toBe(0);
            expect(component.captionStatus.with_caption).toBe(0);
            expect(component.captionStatus.without_caption).toBe(0);
            expect(component.captionStatus.coverage_ratio).toBe(0);
        });

        test('pendingCaptions is initialized as empty object', () => {
            expect(component.pendingCaptions).toEqual({});
        });
    });

    describe('file browser state', () => {
        test('has file browser state initialized', () => {
            expect(component.fileBrowserOpen).toBe(false);
            expect(component.currentPath).toBe('');
            expect(component.directories).toEqual([]);
            expect(component.selectedPath).toBeNull();
            expect(component.loadingDirectories).toBe(false);
        });
    });

    describe('wizard flow state', () => {
        test('has wizard steps state', () => {
            expect(component.wizardStep).toBe(1);
            expect(component.wizardTitle).toBe('Create Dataset - Step 1');
        });

        test('has dataset queue state', () => {
            expect(component.datasetQueue).toEqual([]);
            expect(component.editingQueuedDataset).toBe(false);
            expect(component.editingIndex).toBe(-1);
        });
    });

    describe('conditioning configuration', () => {
        test('has conditioning state initialized', () => {
            expect(component.conditioningConfigured).toBe(false);
            expect(component.selectedConditioningType).toBe('');
            expect(component.conditioningParams).toBeDefined();
            expect(component.conditioningGenerators).toBeDefined();
            expect(Array.isArray(component.conditioningGenerators)).toBe(true);
        });
    });
});

describe('Dataset Wizard Modal State Transitions', () => {
    let component;

    beforeEach(() => {
        const factory = window.datasetWizardComponent;
        component = factory();
        component.$nextTick = jest.fn((cb) => cb());
        component.$refs = {
            newFolderInput: { focus: jest.fn() },
            fileInput: { value: '' },
            zipInput: { value: '' },
        };
    });

    test('opening new folder dialog from closed state', () => {
        expect(component.showNewFolderInput).toBe(false);

        component.openNewFolderDialog();

        expect(component.showNewFolderInput).toBe(true);
    });

    test('open then cancel new folder dialog', () => {
        component.openNewFolderDialog();
        expect(component.showNewFolderInput).toBe(true);

        component.cancelNewFolder();
        expect(component.showNewFolderInput).toBe(false);
    });

    test('open then close upload modal', () => {
        component.openUploadModal();
        expect(component.uploadModalOpen).toBe(true);

        component.closeUploadModal();
        expect(component.uploadModalOpen).toBe(false);
    });

    test('state isolation between modals', () => {
        // Open new folder
        component.openNewFolderDialog();
        expect(component.showNewFolderInput).toBe(true);
        expect(component.uploadModalOpen).toBe(false);

        // Cancel new folder
        component.cancelNewFolder();

        // Open upload
        component.openUploadModal();
        expect(component.showNewFolderInput).toBe(false);
        expect(component.uploadModalOpen).toBe(true);
    });
});

describe('Getter Reactivity (canProceed, stepDefinitions)', () => {
    let component;

    beforeEach(() => {
        const factory = window.datasetWizardComponent;
        component = factory();
        component.$nextTick = jest.fn((cb) => cb());
        component.$refs = {};
    });

    describe('canProceed getter', () => {
        test('canProceed is a reactive getter, not a static value', () => {
            // Verify canProceed is defined via getter (not static false)
            const descriptor = Object.getOwnPropertyDescriptor(component, 'canProceed');
            expect(descriptor).toBeDefined();
            expect(typeof descriptor.get).toBe('function');
        });

        test('canProceed returns falsy when dataset ID is empty on name step', () => {
            component.wizardStep = 1;
            component.currentDataset.id = '';

            // Returns '' (falsy) due to short-circuit evaluation, which works for :disabled="!canProceed"
            expect(component.canProceed).toBeFalsy();
        });

        test('canProceed returns false when dataset ID is only whitespace on name step', () => {
            component.wizardStep = 1;
            component.currentDataset.id = '   ';

            expect(component.canProceed).toBe(false);
        });

        test('canProceed returns true when dataset ID is provided on name step', () => {
            component.wizardStep = 1;
            component.currentDataset.id = 'my-dataset';

            expect(component.canProceed).toBe(true);
        });

        test('canProceed reacts to dataset ID changes', () => {
            component.wizardStep = 1;

            // Initially empty
            component.currentDataset.id = '';
            expect(component.canProceed).toBeFalsy();

            // User types dataset ID
            component.currentDataset.id = 'test-dataset';
            expect(component.canProceed).toBe(true);

            // User clears the field
            component.currentDataset.id = '';
            expect(component.canProceed).toBeFalsy();
        });

        test('canProceed checks selectedBackend on type step', () => {
            component.wizardStep = 2;

            // No backend selected
            component.selectedBackend = null;
            expect(component.canProceed).toBe(false);

            // Backend selected
            component.selectedBackend = 'local';
            expect(component.canProceed).toBe(true);
        });
    });

    describe('stepDefinitions getter', () => {
        test('stepDefinitions is a reactive getter', () => {
            const descriptor = Object.getOwnPropertyDescriptor(component, 'stepDefinitions');
            expect(descriptor).toBeDefined();
            expect(typeof descriptor.get).toBe('function');
        });

        test('stepDefinitions returns base steps', () => {
            const steps = component.stepDefinitions;

            expect(Array.isArray(steps)).toBe(true);
            expect(steps.length).toBeGreaterThanOrEqual(7);

            // Check required steps exist
            const stepIds = steps.map((s) => s.id);
            expect(stepIds).toContain('name');
            expect(stepIds).toContain('type');
            expect(stepIds).toContain('config');
            expect(stepIds).toContain('resolution');
            expect(stepIds).toContain('cropping');
            expect(stepIds).toContain('captions');
            expect(stepIds).toContain('review');
        });

        test('stepDefinitions includes text-embeds step when enabled', () => {
            component.separateTextEmbeds = false;
            let stepIds = component.stepDefinitions.map((s) => s.id);
            expect(stepIds).not.toContain('text-embeds');

            component.separateTextEmbeds = true;
            stepIds = component.stepDefinitions.map((s) => s.id);
            expect(stepIds).toContain('text-embeds');
        });

        test('stepDefinitions includes vae-cache step when enabled', () => {
            component.separateVaeCache = false;
            let stepIds = component.stepDefinitions.map((s) => s.id);
            expect(stepIds).not.toContain('vae-cache');

            component.separateVaeCache = true;
            stepIds = component.stepDefinitions.map((s) => s.id);
            expect(stepIds).toContain('vae-cache');
        });
    });

    describe('currentStepDef getter', () => {
        test('currentStepDef is a reactive getter', () => {
            const descriptor = Object.getOwnPropertyDescriptor(component, 'currentStepDef');
            expect(descriptor).toBeDefined();
            expect(typeof descriptor.get).toBe('function');
        });

        test('currentStepDef returns correct step for wizardStep', () => {
            component.wizardStep = 1;
            expect(component.currentStepDef.id).toBe('name');

            component.wizardStep = 2;
            expect(component.currentStepDef.id).toBe('type');

            component.wizardStep = 3;
            expect(component.currentStepDef.id).toBe('config');
        });
    });

    describe('totalSteps getter', () => {
        test('totalSteps is a reactive getter', () => {
            const descriptor = Object.getOwnPropertyDescriptor(component, 'totalSteps');
            expect(descriptor).toBeDefined();
            expect(typeof descriptor.get).toBe('function');
        });

        test('totalSteps matches stepDefinitions length', () => {
            expect(component.totalSteps).toBe(component.stepDefinitions.length);
        });

        test('totalSteps changes when optional steps are enabled', () => {
            const baseSteps = component.totalSteps;

            component.separateTextEmbeds = true;
            expect(component.totalSteps).toBe(baseSteps + 1);

            component.separateVaeCache = true;
            expect(component.totalSteps).toBe(baseSteps + 2);
        });
    });
});

describe('HintMixin Integration', () => {
    let component;

    beforeEach(() => {
        const factory = window.datasetWizardComponent;
        component = factory();
        component.$nextTick = jest.fn((cb) => cb());
    });

    test('component includes HintMixin properties', () => {
        expect(component.hints).toBeDefined();
        expect(component.hints.hero).toBe(true);
        expect(typeof component.loadHints).toBe('function');
        expect(typeof component.dismissHint).toBe('function');
        expect(typeof component.showHint).toBe('function');
    });

    test('component includes hero CTA helper methods', () => {
        expect(typeof component.showHeroCTA).toBe('function');
        expect(typeof component.dismissHeroCTA).toBe('function');
        expect(typeof component.restoreHeroCTA).toBe('function');
        expect(typeof component.launchWizardFromHero).toBe('function');
    });

    test('showHeroCTA returns hints.hero value', () => {
        component.hints.hero = true;
        expect(component.showHeroCTA()).toBe(true);

        component.hints.hero = false;
        expect(component.showHeroCTA()).toBe(false);
    });
});
