/**
 * Tests for dataloader section component audio capabilities.
 *
 * Covers the supportsAudioInputs and requiresS2VDatasets getters
 * which determine whether audio settings should be shown for video datasets.
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

// Default mock trainer store
const createMockTrainerStore = (overrides = {}) => ({
    modelContext: {
        capabilities: {},
        ...overrides.modelContext
    },
    configValues: {},
    ...overrides
});

// Setup Alpine mock
global.Alpine = {
    data: jest.fn((name, factory) => {
        global.Alpine._components = global.Alpine._components || {};
        global.Alpine._components[name] = factory;
    }),
    store: jest.fn(() => createMockTrainerStore()),
    start: jest.fn(),
    _components: {},
};

// Mock window utilities
global.window = global.window || {};
window.showToast = jest.fn();
window.ToastType = { SUCCESS: 'success', ERROR: 'error', WARNING: 'warning' };

// Load the dataloader section component module
require('../../simpletuner/static/js/dataloader-section-component.js');

// Helper to create component with specific model context
const createComponentWithContext = (modelContext) => {
    global.Alpine.store = jest.fn((storeName) => {
        if (storeName === 'trainer') {
            return {
                modelContext: modelContext,
                configValues: {}
            };
        }
        return null;
    });

    const factory = window.dataloaderSectionComponent;
    return factory();
};

describe('Dataloader Section Audio Capabilities', () => {
    describe('supportsAudioInputs getter', () => {
        test('returns false when trainer store is unavailable', () => {
            global.Alpine.store = jest.fn(() => null);
            const factory = window.dataloaderSectionComponent;
            const component = factory();
            expect(component.supportsAudioInputs).toBe(false);
        });

        test('returns false when modelContext has no audio capabilities', () => {
            const component = createComponentWithContext({
                capabilities: {}
            });
            expect(component.supportsAudioInputs).toBe(false);
        });

        test('returns true when capabilities.supports_audio_inputs is true', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: true
                }
            });
            expect(component.supportsAudioInputs).toBe(true);
        });

        test('returns true when modelContext.supportsAudioInputs is true', () => {
            const component = createComponentWithContext({
                supportsAudioInputs: true,
                capabilities: {}
            });
            expect(component.supportsAudioInputs).toBe(true);
        });

        test('returns false when supports_audio_inputs is explicitly false', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: false
                }
            });
            expect(component.supportsAudioInputs).toBe(false);
        });

        test('handles string "true" value correctly', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: 'true'
                }
            });
            expect(component.supportsAudioInputs).toBe(true);
        });

        test('handles numeric 1 value correctly', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: 1
                }
            });
            expect(component.supportsAudioInputs).toBe(true);
        });
    });

    describe('requiresS2VDatasets getter', () => {
        test('returns false when trainer store is unavailable', () => {
            global.Alpine.store = jest.fn(() => null);
            const factory = window.dataloaderSectionComponent;
            const component = factory();
            expect(component.requiresS2VDatasets).toBe(false);
        });

        test('returns false when modelContext has no S2V requirement', () => {
            const component = createComponentWithContext({
                capabilities: {}
            });
            expect(component.requiresS2VDatasets).toBe(false);
        });

        test('returns true when capabilities.requires_s2v_datasets is true', () => {
            const component = createComponentWithContext({
                capabilities: {
                    requires_s2v_datasets: true
                }
            });
            expect(component.requiresS2VDatasets).toBe(true);
        });

        test('returns true when modelContext.requiresS2VDatasets is true', () => {
            const component = createComponentWithContext({
                requiresS2VDatasets: true,
                capabilities: {}
            });
            expect(component.requiresS2VDatasets).toBe(true);
        });

        test('returns false when requires_s2v_datasets is explicitly false', () => {
            const component = createComponentWithContext({
                capabilities: {
                    requires_s2v_datasets: false
                }
            });
            expect(component.requiresS2VDatasets).toBe(false);
        });
    });

    describe('Audio capabilities combinations', () => {
        test('model with audio inputs and no S2V requirement (like LTX-2)', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: true,
                    requires_s2v_datasets: false
                }
            });

            expect(component.supportsAudioInputs).toBe(true);
            expect(component.requiresS2VDatasets).toBe(false);
        });

        test('S2V model requires audio datasets (like WanS2V)', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: true,
                    requires_s2v_datasets: true
                }
            });

            expect(component.supportsAudioInputs).toBe(true);
            expect(component.requiresS2VDatasets).toBe(true);
        });

        test('model without audio support (like SDXL)', () => {
            const component = createComponentWithContext({
                capabilities: {
                    supports_audio_inputs: false,
                    requires_s2v_datasets: false
                }
            });

            expect(component.supportsAudioInputs).toBe(false);
            expect(component.requiresS2VDatasets).toBe(false);
        });
    });

    describe('normalizeBoolean helper', () => {
        test('normalizes various truthy values', () => {
            const component = createComponentWithContext({});
            expect(component.normalizeBoolean(true)).toBe(true);
            expect(component.normalizeBoolean('true')).toBe(true);
            expect(component.normalizeBoolean(1)).toBe(true);
            expect(component.normalizeBoolean('1')).toBe(true);
        });

        test('normalizes various falsy values', () => {
            const component = createComponentWithContext({});
            expect(component.normalizeBoolean(false)).toBe(false);
            expect(component.normalizeBoolean('false')).toBe(false);
            expect(component.normalizeBoolean(0)).toBe(false);
            expect(component.normalizeBoolean('0')).toBe(false);
            expect(component.normalizeBoolean(null)).toBe(false);
            expect(component.normalizeBoolean(undefined)).toBe(false);
        });
    });
});
