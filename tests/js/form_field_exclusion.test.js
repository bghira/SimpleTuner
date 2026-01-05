/**
 * Tests for UI-only form field exclusion.
 *
 * Covers:
 * - UI sentinel fields (ez_model_type, __active_tab__, __disabled_fields__) are excluded from config
 * - FormData serialization filters out non-CLI fields
 * - Alpine.js x-model fields without name attributes don't leak into form data
 *
 * Related bug: ez_model_type sentinel key leaked into config.json causing argparser errors on upgrade
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

/**
 * Simulates the excluded_fields logic from configs_service.py normalize_form_to_config
 */
const UI_ONLY_EXCLUDED_FIELDS = new Set([
    'configs_dir',
    '__active_tab__',
    '__disabled_fields__',
    'ez_model_type',
]);

/**
 * Simulates prefix-based exclusion patterns
 */
const EXCLUDED_PREFIXES = [
    'currentDataset.',
    '--currentDataset.',
    'datasets_page_',
    '--datasets_page_',
];

/**
 * Filter form data to remove UI-only fields (mirrors backend logic)
 */
function filterFormDataForConfig(formData) {
    const filtered = {};

    for (const [key, value] of Object.entries(formData)) {
        // Skip explicitly excluded fields
        if (UI_ONLY_EXCLUDED_FIELDS.has(key)) {
            continue;
        }

        // Skip prefix-matched fields
        const matchesPrefix = EXCLUDED_PREFIXES.some(prefix => key.startsWith(prefix));
        if (matchesPrefix) {
            continue;
        }

        filtered[key] = value;
    }

    return filtered;
}

/**
 * Simulate FormData collection from a form element
 */
function simulateFormDataCollection(formInputs) {
    const formData = {};
    for (const input of formInputs) {
        if (input.name) {
            formData[input.name] = input.value;
        }
    }
    return formData;
}

describe('UI-Only Field Exclusion', () => {
    beforeEach(() => {
        localStorageMock.clear();
    });

    describe('excluded fields set', () => {
        test('ez_model_type is in excluded fields', () => {
            expect(UI_ONLY_EXCLUDED_FIELDS.has('ez_model_type')).toBe(true);
        });

        test('__active_tab__ is in excluded fields', () => {
            expect(UI_ONLY_EXCLUDED_FIELDS.has('__active_tab__')).toBe(true);
        });

        test('__disabled_fields__ is in excluded fields', () => {
            expect(UI_ONLY_EXCLUDED_FIELDS.has('__disabled_fields__')).toBe(true);
        });

        test('configs_dir is in excluded fields', () => {
            expect(UI_ONLY_EXCLUDED_FIELDS.has('configs_dir')).toBe(true);
        });

        test('valid CLI args are not in excluded fields', () => {
            expect(UI_ONLY_EXCLUDED_FIELDS.has('--model_type')).toBe(false);
            expect(UI_ONLY_EXCLUDED_FIELDS.has('--output_dir')).toBe(false);
            expect(UI_ONLY_EXCLUDED_FIELDS.has('--lora_rank')).toBe(false);
        });
    });

    describe('filterFormDataForConfig', () => {
        test('filters out ez_model_type from form data', () => {
            const formData = {
                '--model_type': 'lora',
                '--output_dir': '/path/to/output',
                'ez_model_type': 'lora',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--model_type', 'lora');
            expect(filtered).toHaveProperty('--output_dir', '/path/to/output');
            expect(filtered).not.toHaveProperty('ez_model_type');
        });

        test('filters out __active_tab__ from form data', () => {
            const formData = {
                '--learning_rate': '1e-4',
                '__active_tab__': 'training',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--learning_rate', '1e-4');
            expect(filtered).not.toHaveProperty('__active_tab__');
        });

        test('filters out __disabled_fields__ from form data', () => {
            const formData = {
                '--num_train_epochs': '10',
                '__disabled_fields__': 'field1,field2,field3',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--num_train_epochs', '10');
            expect(filtered).not.toHaveProperty('__disabled_fields__');
        });

        test('filters out multiple UI-only fields at once', () => {
            const formData = {
                '--model_type': 'lora',
                '--output_dir': '/path/to/output',
                '--learning_rate': '1e-4',
                'ez_model_type': 'lora',
                '__active_tab__': 'model',
                '__disabled_fields__': 'some,fields',
                'configs_dir': '/path/to/configs',
            };

            const filtered = filterFormDataForConfig(formData);

            // Should keep CLI args
            expect(Object.keys(filtered)).toHaveLength(3);
            expect(filtered).toHaveProperty('--model_type');
            expect(filtered).toHaveProperty('--output_dir');
            expect(filtered).toHaveProperty('--learning_rate');

            // Should exclude UI-only fields
            expect(filtered).not.toHaveProperty('ez_model_type');
            expect(filtered).not.toHaveProperty('__active_tab__');
            expect(filtered).not.toHaveProperty('__disabled_fields__');
            expect(filtered).not.toHaveProperty('configs_dir');
        });

        test('filters out datasets_page_ prefixed fields', () => {
            const formData = {
                '--data_backend_config': 'datasets/config.json',
                'datasets_page_data_backend_config': 'some_value',
                '--datasets_page_other': 'should_be_excluded',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--data_backend_config');
            expect(filtered).not.toHaveProperty('datasets_page_data_backend_config');
            expect(filtered).not.toHaveProperty('--datasets_page_other');
        });

        test('filters out currentDataset. prefixed fields', () => {
            const formData = {
                '--resolution': '1024',
                'currentDataset.name': 'my_dataset',
                '--currentDataset.path': '/some/path',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--resolution');
            expect(filtered).not.toHaveProperty('currentDataset.name');
            expect(filtered).not.toHaveProperty('--currentDataset.path');
        });

        test('preserves all valid CLI arguments', () => {
            const formData = {
                '--model_type': 'lora',
                '--model_family': 'flux',
                '--pretrained_model_name_or_path': 'black-forest-labs/FLUX.1-dev',
                '--output_dir': '/output',
                '--lora_rank': '64',
                '--learning_rate': '1e-4',
                '--num_train_epochs': '10',
                '--train_batch_size': '1',
                '--gradient_accumulation_steps': '4',
                '--use_gradient_checkpointing': 'true',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(Object.keys(filtered)).toHaveLength(10);
            Object.entries(formData).forEach(([key, value]) => {
                expect(filtered).toHaveProperty(key, value);
            });
        });

        test('handles empty form data', () => {
            const formData = {};
            const filtered = filterFormDataForConfig(formData);
            expect(Object.keys(filtered)).toHaveLength(0);
        });

        test('handles form data with only excluded fields', () => {
            const formData = {
                'ez_model_type': 'lora',
                '__active_tab__': 'basic',
                '__disabled_fields__': 'field1',
            };

            const filtered = filterFormDataForConfig(formData);
            expect(Object.keys(filtered)).toHaveLength(0);
        });
    });

    describe('FormData collection simulation', () => {
        test('inputs without name attribute are not collected', () => {
            // Simulates radio buttons after fix: no name attribute
            const formInputs = [
                { name: '--model_type', value: 'lora' },
                { name: '', value: 'lora' },  // ez_model_type radio without name
                { value: 'full' },  // Another radio without name attribute
                { name: '--output_dir', value: '/output' },
            ];

            const formData = simulateFormDataCollection(formInputs);

            expect(formData).toHaveProperty('--model_type', 'lora');
            expect(formData).toHaveProperty('--output_dir', '/output');
            expect(formData).not.toHaveProperty('ez_model_type');
            expect(formData).not.toHaveProperty('');
            expect(Object.keys(formData)).toHaveLength(2);
        });

        test('inputs with name attribute are collected', () => {
            const formInputs = [
                { name: '--learning_rate', value: '1e-4' },
                { name: '--num_train_epochs', value: '10' },
                { name: '__active_tab__', value: 'training' },
            ];

            const formData = simulateFormDataCollection(formInputs);

            expect(Object.keys(formData)).toHaveLength(3);
            expect(formData).toHaveProperty('--learning_rate');
            expect(formData).toHaveProperty('--num_train_epochs');
            expect(formData).toHaveProperty('__active_tab__');
        });

        test('combined collection and filtering removes UI-only fields', () => {
            const formInputs = [
                { name: '--model_type', value: 'lora' },
                { name: '--output_dir', value: '/output' },
                { name: '__active_tab__', value: 'model' },
                { name: '__disabled_fields__', value: 'field1,field2' },
            ];

            const formData = simulateFormDataCollection(formInputs);
            const filtered = filterFormDataForConfig(formData);

            expect(Object.keys(filtered)).toHaveLength(2);
            expect(filtered).toHaveProperty('--model_type');
            expect(filtered).toHaveProperty('--output_dir');
        });
    });

    describe('EZ Mode wizard field handling', () => {
        test('ez_model_type should not appear in filtered config', () => {
            // Simulates what happens when EZ Mode wizard is visible and form is saved
            const formDataWithEzMode = {
                '--model_type': 'lora',
                '--model_family': 'flux',
                '--lora_rank': '64',
                'ez_model_type': 'lora',  // This would leak without the fix
                '__active_tab__': 'model',
            };

            const filtered = filterFormDataForConfig(formDataWithEzMode);

            expect(filtered).not.toHaveProperty('ez_model_type');
            expect(filtered).toHaveProperty('--model_type', 'lora');
        });

        test('model_type CLI arg is preserved while ez_model_type is filtered', () => {
            const formData = {
                '--model_type': 'full',
                'ez_model_type': 'full',
            };

            const filtered = filterFormDataForConfig(formData);

            expect(filtered).toHaveProperty('--model_type', 'full');
            expect(filtered).not.toHaveProperty('ez_model_type');
            expect(Object.keys(filtered)).toHaveLength(1);
        });
    });

    describe('regression prevention', () => {
        test('config saved from model tab with EZ mode should not include sentinel fields', () => {
            // This test simulates the exact bug scenario:
            // User is on model tab with EZ mode visible, saves config
            const modelTabFormData = {
                // Valid CLI args
                '--model_type': 'lora',
                '--model_family': 'flux',
                '--pretrained_model_name_or_path': 'black-forest-labs/FLUX.1-dev',
                '--lora_rank': '64',
                '--base_model_precision': 'int8-quanto',
                // UI-only fields that would cause argparser errors
                'ez_model_type': 'lora',
                '__active_tab__': 'model',
                '__disabled_fields__': 'some_field,another_field',
            };

            const configToSave = filterFormDataForConfig(modelTabFormData);

            // Verify no UI-only fields leaked through
            const uiOnlyKeys = Object.keys(configToSave).filter(
                key => !key.startsWith('--')
            );
            expect(uiOnlyKeys).toHaveLength(0);

            // Verify all CLI args are preserved
            expect(configToSave).toHaveProperty('--model_type');
            expect(configToSave).toHaveProperty('--model_family');
            expect(configToSave).toHaveProperty('--pretrained_model_name_or_path');
            expect(configToSave).toHaveProperty('--lora_rank');
            expect(configToSave).toHaveProperty('--base_model_precision');
        });

        test('upgrading config with ez_model_type should not cause errors', () => {
            // Simulates loading an old config that has ez_model_type
            const oldConfigWithSentinel = {
                '--model_type': 'lora',
                '--output_dir': '/output',
                'ez_model_type': 'lora',  // Leaked from old version
            };

            // The filter should strip it before passing to argparser
            const cleanedConfig = filterFormDataForConfig(oldConfigWithSentinel);

            expect(cleanedConfig).not.toHaveProperty('ez_model_type');
            expect(cleanedConfig).toHaveProperty('--model_type');
            expect(cleanedConfig).toHaveProperty('--output_dir');
        });
    });
});

describe('Form Input Name Attribute Requirements', () => {
    describe('Alpine.js x-model inputs', () => {
        test('x-model inputs should not need name attribute for data binding', () => {
            // This documents the expected behavior:
            // Inputs using Alpine x-model for data binding don't need name attributes
            // The name attribute is only needed for HTML form submission
            const alpineOnlyInput = {
                'x-model': 'model_type',
                value: 'lora',
                // No name attribute - correct for Alpine-only binding
            };

            // Should not be collected by FormData
            const formData = simulateFormDataCollection([alpineOnlyInput]);
            expect(Object.keys(formData)).toHaveLength(0);
        });

        test('inputs with both name and x-model get collected (the bug scenario)', () => {
            // This was the bug: having both name="ez_model_type" and x-model="model_type"
            const inputWithBoth = {
                name: 'ez_model_type',
                'x-model': 'model_type',
                value: 'lora',
            };

            const formData = simulateFormDataCollection([inputWithBoth]);

            // The name causes it to be collected
            expect(formData).toHaveProperty('ez_model_type', 'lora');

            // But it should be filtered out before saving
            const filtered = filterFormDataForConfig(formData);
            expect(filtered).not.toHaveProperty('ez_model_type');
        });
    });

    describe('hidden input fields', () => {
        test('__active_tab__ hidden input is filtered', () => {
            const hiddenInputs = [
                { name: '__active_tab__', value: 'training', type: 'hidden' },
                { name: '--learning_rate', value: '1e-4' },
            ];

            const formData = simulateFormDataCollection(hiddenInputs);
            const filtered = filterFormDataForConfig(formData);

            expect(filtered).not.toHaveProperty('__active_tab__');
            expect(filtered).toHaveProperty('--learning_rate');
        });

        test('__disabled_fields__ hidden input is filtered', () => {
            const hiddenInputs = [
                { name: '__disabled_fields__', value: 'field1,field2', type: 'hidden' },
                { name: '--num_train_epochs', value: '10' },
            ];

            const formData = simulateFormDataCollection(hiddenInputs);
            const filtered = filterFormDataForConfig(formData);

            expect(filtered).not.toHaveProperty('__disabled_fields__');
            expect(filtered).toHaveProperty('--num_train_epochs');
        });
    });
});
