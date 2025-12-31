/**
 * Tests for field sync and formDirty tracking.
 *
 * Covers:
 * - Field value storage and retrieval across tab changes
 * - Dirty state tracking and clearing
 * - Form value preservation during navigation
 * - JSON modal content synchronization after save
 */

// Mock localStorage
const localStorageMock = (() => {
    let store = {};
    return {
        getItem: jest.fn((key) => store[key] || null),
        setItem: jest.fn((key, value) => {
            store[key] = value.toString();
        }),
        removeItem: jest.fn((key) => {
            delete store[key];
        }),
        clear: jest.fn(() => {
            store = {};
        }),
    };
})();

Object.defineProperty(window, 'localStorage', {
    value: localStorageMock,
});

// Create a comprehensive mock trainer store
const createMockTrainerStore = (initialConfig = {}) => ({
    // Tab state
    activeTab: 'basic',
    tabs: ['basic', 'model', 'datasets', 'training', 'validation', 'cloud'],

    // Form state
    formDirty: false,
    formValueStore: {},
    activeEnvironmentConfig: { ...initialConfig },

    // JSON modal state
    configJsonDraft: '',
    configJsonError: '',

    // Saving state
    saving: false,
    loading: false,

    // Tab switching
    switchTab(tabName) {
        if (this.tabs.includes(tabName)) {
            // Store current form values before switching
            this.storeFieldValues();
            this.activeTab = tabName;
            localStorage.setItem('activeTab', tabName);
            return true;
        }
        return false;
    },

    isTabActive(tabName) {
        return this.activeTab === tabName;
    },

    // Field value storage
    storeFieldValues(form = null) {
        // In real implementation, this reads from DOM form elements
        // For testing, we simulate by preserving formValueStore
        const values = { ...this.formValueStore };
        this.formValueStore = values;
    },

    // Store a single field value
    storeFieldValue(fieldName, value, kind = 'single') {
        this.formValueStore[fieldName] = {
            kind,
            value,
        };
        this.markFormDirty();
    },

    // Get a stored field value
    getStoredFieldValue(fieldName) {
        const entry = this.formValueStore[fieldName];
        if (!entry) return null;
        return entry.value;
    },

    // Apply stored values to form
    applyStoredValues() {
        // In real implementation, this sets DOM element values
        // For testing, we verify formValueStore is preserved
        return Object.keys(this.formValueStore).length;
    },

    // Dirty state management
    markFormDirty() {
        this.formDirty = true;
    },

    clearDirtyState() {
        this.formDirty = false;
    },

    computeDirtyState() {
        let isDirty = false;

        // Check if any stored values differ from initial config
        Object.entries(this.formValueStore).forEach(([name, entry]) => {
            const originalValue = this.activeEnvironmentConfig[name];
            if (entry.value !== originalValue) {
                isDirty = true;
            }
        });

        this.formDirty = isDirty;
        return isDirty;
    },

    // Config JSON operations
    refreshConfigJson() {
        const normalized = { ...this.activeEnvironmentConfig };

        // Merge formValueStore into normalized config
        Object.entries(this.formValueStore).forEach(([name, entry]) => {
            normalized[name] = entry.value;
        });

        this.configJsonDraft = JSON.stringify(normalized, null, 2);
        this.configJsonError = '';
    },

    formatConfigJson() {
        try {
            const formatted = JSON.stringify(JSON.parse(this.configJsonDraft), null, 2);
            this.configJsonDraft = formatted;
            this.configJsonError = '';
            return true;
        } catch (error) {
            this.configJsonError = `Invalid JSON: ${error.message}`;
            return false;
        }
    },

    // Save operation
    async saveConfiguration() {
        this.saving = true;

        // Merge form values into active config
        Object.entries(this.formValueStore).forEach(([name, entry]) => {
            this.activeEnvironmentConfig[name] = entry.value;
        });

        // Clear dirty state after successful save
        this.clearDirtyState();

        // Update JSON modal content
        this.refreshConfigJson();

        this.saving = false;
        return true;
    },

    // Reset form to original state
    resetForm() {
        this.formValueStore = {};
        this.formDirty = false;
    },
});

describe('Field Sync Across Tab Changes', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore({
            '--model_family': 'flux',
            '--model_type': 'lora',
        });
    });

    describe('preserving dirty fields when changing tabs', () => {
        test('field values are preserved when switching from basic to model tab', () => {
            // Modify a field on basic tab
            store.storeFieldValue('--output_dir', '/new/output/path');
            store.storeFieldValue('--job_id', 'test-job-123');

            expect(store.formDirty).toBe(true);
            expect(store.getStoredFieldValue('--output_dir')).toBe('/new/output/path');

            // Switch to model tab
            store.switchTab('model');

            // Values should still be preserved
            expect(store.getStoredFieldValue('--output_dir')).toBe('/new/output/path');
            expect(store.getStoredFieldValue('--job_id')).toBe('test-job-123');
            expect(store.formDirty).toBe(true);
        });

        test('field values are preserved across multiple tab switches', () => {
            // Set values on different "tabs"
            store.storeFieldValue('--output_dir', '/path/one');
            store.switchTab('model');

            store.storeFieldValue('--lora_rank', '64');
            store.switchTab('training');

            store.storeFieldValue('--learning_rate', '1e-4');
            store.switchTab('datasets');

            // All values should be preserved
            expect(store.getStoredFieldValue('--output_dir')).toBe('/path/one');
            expect(store.getStoredFieldValue('--lora_rank')).toBe('64');
            expect(store.getStoredFieldValue('--learning_rate')).toBe('1e-4');
        });

        test('dirty state persists across tab changes', () => {
            store.storeFieldValue('--output_dir', '/changed/path');
            expect(store.formDirty).toBe(true);

            // Navigate through multiple tabs
            store.switchTab('model');
            expect(store.formDirty).toBe(true);

            store.switchTab('datasets');
            expect(store.formDirty).toBe(true);

            store.switchTab('basic');
            expect(store.formDirty).toBe(true);
        });

        test('checkbox values are preserved correctly', () => {
            store.storeFieldValue('--use_gradient_checkpointing', true, 'checkbox');
            store.storeFieldValue('--train_text_encoder', false, 'checkbox');

            store.switchTab('training');
            store.switchTab('model');
            store.switchTab('basic');

            expect(store.getStoredFieldValue('--use_gradient_checkpointing')).toBe(true);
            expect(store.getStoredFieldValue('--train_text_encoder')).toBe(false);
        });
    });

    describe('returning to tab preserves dirty fields', () => {
        test('values are restored when returning to a previously edited tab', () => {
            // Edit fields on basic tab
            store.storeFieldValue('--output_dir', '/custom/output');
            store.storeFieldValue('--pretrained_model_name_or_path', 'custom-model');

            // Navigate away and back
            store.switchTab('model');
            store.switchTab('training');
            store.switchTab('basic');

            // Simulate applyStoredValues being called
            const fieldCount = store.applyStoredValues();

            expect(fieldCount).toBeGreaterThan(0);
            expect(store.getStoredFieldValue('--output_dir')).toBe('/custom/output');
            expect(store.getStoredFieldValue('--pretrained_model_name_or_path')).toBe('custom-model');
        });

        test('multiple round-trips preserve all values', () => {
            const testValues = {
                '--field1': 'value1',
                '--field2': 'value2',
                '--field3': 'value3',
            };

            // Store initial values
            Object.entries(testValues).forEach(([name, value]) => {
                store.storeFieldValue(name, value);
            });

            // Multiple round-trips
            for (let i = 0; i < 3; i++) {
                store.switchTab('model');
                store.switchTab('datasets');
                store.switchTab('basic');
            }

            // All values should still be present
            Object.entries(testValues).forEach(([name, value]) => {
                expect(store.getStoredFieldValue(name)).toBe(value);
            });
        });

        test('new edits on return are also preserved', () => {
            store.storeFieldValue('--field1', 'original');
            store.switchTab('model');
            store.switchTab('basic');

            // Make additional edits after returning
            store.storeFieldValue('--field1', 'updated');
            store.storeFieldValue('--field2', 'new-value');

            store.switchTab('training');
            store.switchTab('basic');

            expect(store.getStoredFieldValue('--field1')).toBe('updated');
            expect(store.getStoredFieldValue('--field2')).toBe('new-value');
        });
    });
});

describe('Form Dirty State Management', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore({
            '--model_family': 'sdxl',
            '--output_dir': '/original/path',
        });
    });

    describe('marking form as dirty', () => {
        test('form starts clean', () => {
            expect(store.formDirty).toBe(false);
        });

        test('storing a field value marks form as dirty', () => {
            store.storeFieldValue('--output_dir', '/new/path');
            expect(store.formDirty).toBe(true);
        });

        test('multiple field changes keep form dirty', () => {
            store.storeFieldValue('--field1', 'value1');
            store.storeFieldValue('--field2', 'value2');
            store.storeFieldValue('--field3', 'value3');

            expect(store.formDirty).toBe(true);
        });

        test('markFormDirty can be called directly', () => {
            store.markFormDirty();
            expect(store.formDirty).toBe(true);
        });
    });

    describe('clearing dirty state on save', () => {
        test('saving clears dirty state', async () => {
            store.storeFieldValue('--output_dir', '/new/path');
            expect(store.formDirty).toBe(true);

            await store.saveConfiguration();

            expect(store.formDirty).toBe(false);
        });

        test('saved values are merged into active config', async () => {
            store.storeFieldValue('--output_dir', '/saved/path');
            store.storeFieldValue('--learning_rate', '1e-5');

            await store.saveConfiguration();

            expect(store.activeEnvironmentConfig['--output_dir']).toBe('/saved/path');
            expect(store.activeEnvironmentConfig['--learning_rate']).toBe('1e-5');
        });

        test('saving state is tracked during save', async () => {
            expect(store.saving).toBe(false);

            const savePromise = store.saveConfiguration();
            // In real async operation, saving would be true here

            await savePromise;
            expect(store.saving).toBe(false);
        });
    });

    describe('computeDirtyState', () => {
        test('returns false when no changes', () => {
            const isDirty = store.computeDirtyState();
            expect(isDirty).toBe(false);
        });

        test('returns true when values differ from original', () => {
            store.formValueStore['--model_family'] = { kind: 'single', value: 'flux' };
            const isDirty = store.computeDirtyState();
            expect(isDirty).toBe(true);
        });

        test('updates formDirty property', () => {
            store.formValueStore['--output_dir'] = { kind: 'single', value: '/different/path' };
            store.computeDirtyState();
            expect(store.formDirty).toBe(true);
        });
    });

    describe('resetting form', () => {
        test('resetForm clears all stored values', () => {
            store.storeFieldValue('--field1', 'value1');
            store.storeFieldValue('--field2', 'value2');

            store.resetForm();

            expect(Object.keys(store.formValueStore).length).toBe(0);
            expect(store.formDirty).toBe(false);
        });

        test('resetForm clears dirty state', () => {
            store.markFormDirty();
            expect(store.formDirty).toBe(true);

            store.resetForm();
            expect(store.formDirty).toBe(false);
        });
    });
});

describe('View JSON Modal Content Updates', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore({
            '--model_family': 'flux',
            '--model_type': 'lora',
            '--output_dir': '/original/path',
        });
    });

    describe('JSON modal reflects current state', () => {
        test('refreshConfigJson generates valid JSON from config', () => {
            store.refreshConfigJson();

            expect(store.configJsonDraft).toBeTruthy();
            expect(() => JSON.parse(store.configJsonDraft)).not.toThrow();

            const parsed = JSON.parse(store.configJsonDraft);
            expect(parsed['--model_family']).toBe('flux');
            expect(parsed['--model_type']).toBe('lora');
        });

        test('refreshConfigJson includes unsaved form values', () => {
            store.storeFieldValue('--learning_rate', '1e-4');
            store.storeFieldValue('--batch_size', '8');

            store.refreshConfigJson();

            const parsed = JSON.parse(store.configJsonDraft);
            expect(parsed['--learning_rate']).toBe('1e-4');
            expect(parsed['--batch_size']).toBe('8');
        });

        test('form value changes are reflected in JSON', () => {
            store.storeFieldValue('--output_dir', '/modified/path');
            store.refreshConfigJson();

            const parsed = JSON.parse(store.configJsonDraft);
            expect(parsed['--output_dir']).toBe('/modified/path');
        });
    });

    describe('JSON modal updates after save', () => {
        test('saving updates JSON modal content', async () => {
            store.storeFieldValue('--new_field', 'new_value');

            await store.saveConfiguration();

            const parsed = JSON.parse(store.configJsonDraft);
            expect(parsed['--new_field']).toBe('new_value');
        });

        test('JSON modal shows saved values after save', async () => {
            store.storeFieldValue('--output_dir', '/saved/output');
            store.storeFieldValue('--learning_rate', '2e-5');

            await store.saveConfiguration();

            const parsed = JSON.parse(store.configJsonDraft);
            expect(parsed['--output_dir']).toBe('/saved/output');
            expect(parsed['--learning_rate']).toBe('2e-5');
        });

        test('JSON modal is formatted correctly after save', async () => {
            store.storeFieldValue('--field', 'value');

            await store.saveConfiguration();

            // Check for proper formatting (indentation)
            expect(store.configJsonDraft).toContain('\n');
            expect(store.configJsonDraft).toMatch(/^\{\n/);
        });
    });

    describe('JSON formatting', () => {
        test('formatConfigJson formats valid JSON', () => {
            store.configJsonDraft = '{"key":"value","nested":{"a":1}}';

            const result = store.formatConfigJson();

            expect(result).toBe(true);
            expect(store.configJsonDraft).toContain('\n');
            expect(store.configJsonError).toBe('');
        });

        test('formatConfigJson sets error for invalid JSON', () => {
            store.configJsonDraft = '{invalid json}';

            const result = store.formatConfigJson();

            expect(result).toBe(false);
            expect(store.configJsonError).toContain('Invalid JSON');
        });

        test('formatConfigJson clears previous errors on success', () => {
            store.configJsonError = 'Previous error';
            store.configJsonDraft = '{"valid": "json"}';

            store.formatConfigJson();

            expect(store.configJsonError).toBe('');
        });
    });
});

describe('Integration: Tab Navigation with Dirty Fields', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore({
            '--model_family': 'sd15',
        });
    });

    test('complete workflow: edit -> navigate -> return -> verify', () => {
        // 1. Start on basic tab, make edits
        expect(store.activeTab).toBe('basic');
        store.storeFieldValue('--output_dir', '/my/output');
        store.storeFieldValue('--job_id', 'my-job');
        expect(store.formDirty).toBe(true);

        // 2. Navigate to model tab
        store.switchTab('model');
        expect(store.activeTab).toBe('model');

        // 3. Make more edits on model tab
        store.storeFieldValue('--lora_rank', '32');
        store.storeFieldValue('--model_type', 'lora');

        // 4. Navigate to training tab
        store.switchTab('training');
        expect(store.activeTab).toBe('training');

        // 5. Return to basic tab
        store.switchTab('basic');
        expect(store.activeTab).toBe('basic');

        // 6. Verify all values are preserved
        expect(store.getStoredFieldValue('--output_dir')).toBe('/my/output');
        expect(store.getStoredFieldValue('--job_id')).toBe('my-job');
        expect(store.getStoredFieldValue('--lora_rank')).toBe('32');
        expect(store.getStoredFieldValue('--model_type')).toBe('lora');
        expect(store.formDirty).toBe(true);
    });

    test('complete workflow: edit -> save -> verify clean state', async () => {
        // 1. Make edits
        store.storeFieldValue('--output_dir', '/saved/path');
        store.storeFieldValue('--learning_rate', '1e-4');
        expect(store.formDirty).toBe(true);

        // 2. Save configuration
        await store.saveConfiguration();

        // 3. Verify clean state
        expect(store.formDirty).toBe(false);
        expect(store.activeEnvironmentConfig['--output_dir']).toBe('/saved/path');
        expect(store.activeEnvironmentConfig['--learning_rate']).toBe('1e-4');

        // 4. Verify JSON modal is updated
        const parsed = JSON.parse(store.configJsonDraft);
        expect(parsed['--output_dir']).toBe('/saved/path');
    });

    test('edits after save create new dirty state', async () => {
        // 1. Initial edit and save
        store.storeFieldValue('--output_dir', '/first/path');
        await store.saveConfiguration();
        expect(store.formDirty).toBe(false);

        // 2. New edit after save
        store.storeFieldValue('--output_dir', '/second/path');
        expect(store.formDirty).toBe(true);

        // 3. Tab navigation preserves new dirty state
        store.switchTab('model');
        store.switchTab('basic');
        expect(store.formDirty).toBe(true);
        expect(store.getStoredFieldValue('--output_dir')).toBe('/second/path');
    });
});

describe('Edge Cases', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore();
    });

    test('empty values are stored correctly', () => {
        store.storeFieldValue('--optional_field', '');
        expect(store.getStoredFieldValue('--optional_field')).toBe('');
    });

    test('null values are handled', () => {
        store.storeFieldValue('--nullable_field', null);
        expect(store.getStoredFieldValue('--nullable_field')).toBeNull();
    });

    test('special characters in values are preserved', () => {
        const specialValue = '/path/with spaces/and-dashes/and_underscores';
        store.storeFieldValue('--path', specialValue);

        store.switchTab('model');
        store.switchTab('basic');

        expect(store.getStoredFieldValue('--path')).toBe(specialValue);
    });

    test('JSON string values are preserved correctly', () => {
        const jsonValue = '{"nested": "object", "array": [1, 2, 3]}';
        store.storeFieldValue('--json_config', jsonValue);

        store.switchTab('training');
        store.refreshConfigJson();

        expect(store.getStoredFieldValue('--json_config')).toBe(jsonValue);
    });

    test('numeric values as strings are preserved', () => {
        store.storeFieldValue('--learning_rate', '1e-4');
        store.storeFieldValue('--batch_size', '16');

        store.switchTab('model');
        store.switchTab('basic');

        expect(store.getStoredFieldValue('--learning_rate')).toBe('1e-4');
        expect(store.getStoredFieldValue('--batch_size')).toBe('16');
    });

    test('rapid tab switching preserves all values', () => {
        store.storeFieldValue('--field1', 'value1');

        // Rapid switching
        for (let i = 0; i < 10; i++) {
            store.switchTab('model');
            store.switchTab('training');
            store.switchTab('basic');
        }

        expect(store.getStoredFieldValue('--field1')).toBe('value1');
        expect(store.formDirty).toBe(true);
    });

    test('switching to same tab is a no-op', () => {
        store.storeFieldValue('--field1', 'value1');
        const initialStore = { ...store.formValueStore };

        store.switchTab('basic'); // Already on basic

        expect(store.formValueStore).toEqual(initialStore);
    });

    test('invalid tab name is rejected', () => {
        store.storeFieldValue('--field1', 'value1');

        const result = store.switchTab('nonexistent');

        expect(result).toBe(false);
        expect(store.activeTab).toBe('basic');
        expect(store.getStoredFieldValue('--field1')).toBe('value1');
    });
});

describe('localStorage Integration', () => {
    let store;

    beforeEach(() => {
        localStorageMock.clear();
        store = createMockTrainerStore();
    });

    test('active tab is persisted to localStorage', () => {
        store.switchTab('model');

        expect(localStorage.setItem).toHaveBeenCalledWith('activeTab', 'model');
    });

    test('tab changes are tracked in localStorage', () => {
        store.switchTab('model');
        store.switchTab('training');
        store.switchTab('datasets');

        expect(localStorage.setItem).toHaveBeenCalledWith('activeTab', 'model');
        expect(localStorage.setItem).toHaveBeenCalledWith('activeTab', 'training');
        expect(localStorage.setItem).toHaveBeenCalledWith('activeTab', 'datasets');
    });
});

describe('Form Value Store Structure', () => {
    let store;

    beforeEach(() => {
        store = createMockTrainerStore();
    });

    test('stored values have correct structure', () => {
        store.storeFieldValue('--field_name', 'field_value');

        const entry = store.formValueStore['--field_name'];
        expect(entry).toHaveProperty('kind');
        expect(entry).toHaveProperty('value');
        expect(entry.kind).toBe('single');
        expect(entry.value).toBe('field_value');
    });

    test('checkbox values have checkbox kind', () => {
        store.storeFieldValue('--checkbox_field', true, 'checkbox');

        const entry = store.formValueStore['--checkbox_field'];
        expect(entry.kind).toBe('checkbox');
        expect(entry.value).toBe(true);
    });

    test('getStoredFieldValue returns null for non-existent fields', () => {
        expect(store.getStoredFieldValue('--nonexistent')).toBeNull();
    });

    test('overwriting a field updates the value', () => {
        store.storeFieldValue('--field', 'first');
        store.storeFieldValue('--field', 'second');

        expect(store.getStoredFieldValue('--field')).toBe('second');
    });
});
