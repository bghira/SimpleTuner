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
