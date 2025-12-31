/**
 * Tests for DependencyManager.
 *
 * DependencyManager handles field visibility based on interdependencies.
 * This is critical for the trainer configuration form where fields show/hide
 * based on other field values.
 */

// Mock fetch
global.fetch = jest.fn();

// Mock ApiClient
global.ApiClient = {
    fetch: jest.fn(),
};

// Load the module
require('../../simpletuner/static/js/services/dependency-manager.js');

describe('DependencyManager', () => {
    let manager;

    beforeEach(() => {
        // Get fresh instance
        manager = new window.dependencyManager.constructor();
        jest.clearAllMocks();
    });

    describe('initialization', () => {
        test('starts uninitialized', () => {
            expect(manager.initialized).toBe(false);
        });

        test('initialize sets initialized flag', async () => {
            await manager.initialize({
                fields: {},
            });
            expect(manager.initialized).toBe(true);
        });

        test('initialize with metadata loads it directly', async () => {
            const metadata = {
                fields: {
                    field1: { section: 'basic' },
                    field2: { section: 'advanced' },
                },
            };

            await manager.initialize(metadata);

            expect(manager.fieldMetadata.size).toBe(2);
            expect(manager.fieldMetadata.get('field1')).toEqual({ section: 'basic' });
        });

        test('initialize without metadata fetches from backend', async () => {
            const mockMetadata = {
                fields: {
                    test_field: { section: 'test' },
                },
            };

            global.ApiClient.fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockMetadata),
            });

            await manager.initialize(null);

            expect(global.ApiClient.fetch).toHaveBeenCalledWith('/api/fields/metadata');
            expect(manager.fieldMetadata.size).toBe(1);
        });

        test('handles fetch error gracefully', async () => {
            global.ApiClient.fetch.mockRejectedValueOnce(new Error('Network error'));

            // Should not throw
            await expect(manager.initialize(null)).resolves.not.toThrow();
        });
    });

    describe('loadMetadata', () => {
        test('clears existing data', () => {
            manager.dependencies.set('old', []);
            manager.observers.set('old', new Set());
            manager.fieldMetadata.set('old', {});

            manager.loadMetadata({ fields: {} });

            expect(manager.dependencies.size).toBe(0);
            expect(manager.observers.size).toBe(0);
            expect(manager.fieldMetadata.size).toBe(0);
        });

        test('builds dependency map from metadata', () => {
            const metadata = {
                fields: {
                    dependent_field: {
                        section: 'test',
                        dependencies: [
                            { field: 'trigger_field', operator: 'equals', value: true },
                        ],
                    },
                    trigger_field: {
                        section: 'test',
                    },
                },
            };

            manager.loadMetadata(metadata);

            expect(manager.dependencies.get('dependent_field')).toHaveLength(1);
            expect(manager.dependencies.get('dependent_field')[0].field).toBe('trigger_field');
        });

        test('builds observer map (reverse mapping)', () => {
            const metadata = {
                fields: {
                    field_a: {
                        dependencies: [{ field: 'field_b', operator: 'equals', value: 'x' }],
                    },
                    field_c: {
                        dependencies: [{ field: 'field_b', operator: 'equals', value: 'y' }],
                    },
                },
            };

            manager.loadMetadata(metadata);

            // field_b should observe both field_a and field_c
            const observers = manager.observers.get('field_b');
            expect(observers.size).toBe(2);
            expect(observers.has('field_a')).toBe(true);
            expect(observers.has('field_c')).toBe(true);
        });
    });

    describe('evaluateSingleDependency', () => {
        beforeEach(() => {
            manager.fieldStates.set('trigger', { value: 'test_value' });
            manager.fieldStates.set('number_field', { value: 10 });
            manager.fieldStates.set('bool_field', { value: true });
        });

        test('equals operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'equals',
                value: 'test_value',
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'equals',
                value: 'other_value',
            })).toBe(false);
        });

        test('not_equals operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'not_equals',
                value: 'other_value',
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'not_equals',
                value: 'test_value',
            })).toBe(false);
        });

        test('in operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'in',
                values: ['test_value', 'other'],
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'in',
                values: ['other', 'another'],
            })).toBe(false);
        });

        test('not_in operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'not_in',
                values: ['other', 'another'],
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'not_in',
                values: ['test_value', 'other'],
            })).toBe(false);
        });

        test('greater_than operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'greater_than',
                value: 5,
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'greater_than',
                value: 10,
            })).toBe(false);

            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'greater_than',
                value: 15,
            })).toBe(false);
        });

        test('less_than operator', () => {
            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'less_than',
                value: 15,
            })).toBe(true);

            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'less_than',
                value: 10,
            })).toBe(false);

            expect(manager.evaluateSingleDependency({
                field: 'number_field',
                operator: 'less_than',
                value: 5,
            })).toBe(false);
        });

        test('unknown operator returns true (fail open)', () => {
            expect(manager.evaluateSingleDependency({
                field: 'trigger',
                operator: 'unknown_op',
                value: 'x',
            })).toBe(true);
        });

        test('missing dependency field returns true', () => {
            expect(manager.evaluateSingleDependency({
                field: 'nonexistent',
                operator: 'equals',
                value: 'x',
            })).toBe(true);
        });
    });

    describe('evaluateDependencies', () => {
        beforeEach(() => {
            manager.fieldStates.set('field_a', { value: true });
            manager.fieldStates.set('field_b', { value: 'enabled' });
        });

        test('all dependencies must be satisfied (AND logic)', () => {
            const dependencies = [
                { field: 'field_a', operator: 'equals', value: true },
                { field: 'field_b', operator: 'equals', value: 'enabled' },
            ];

            expect(manager.evaluateDependencies(dependencies)).toBe(true);
        });

        test('fails if any dependency is not satisfied', () => {
            const dependencies = [
                { field: 'field_a', operator: 'equals', value: true },
                { field: 'field_b', operator: 'equals', value: 'disabled' },
            ];

            expect(manager.evaluateDependencies(dependencies)).toBe(false);
        });

        test('empty dependencies array returns true', () => {
            expect(manager.evaluateDependencies([])).toBe(true);
        });
    });

    describe('getFieldValue', () => {
        test('gets checkbox value correctly', () => {
            const checkbox = document.createElement('input');
            checkbox.type = 'checkbox';
            checkbox.checked = true;

            expect(manager.getFieldValue(checkbox)).toBe(true);

            checkbox.checked = false;
            expect(manager.getFieldValue(checkbox)).toBe(false);
        });

        test('gets number input value as number', () => {
            const number = document.createElement('input');
            number.type = 'number';
            number.value = '42';

            expect(manager.getFieldValue(number)).toBe(42);
        });

        test('handles invalid number input', () => {
            const number = document.createElement('input');
            number.type = 'number';
            number.value = '';

            expect(manager.getFieldValue(number)).toBe(0);
        });

        test('gets text input value as string', () => {
            const text = document.createElement('input');
            text.type = 'text';
            text.value = 'hello';

            expect(manager.getFieldValue(text)).toBe('hello');
        });

        test('gets select value', () => {
            const select = document.createElement('select');
            select.innerHTML = '<option value="opt1">Option 1</option><option value="opt2">Option 2</option>';
            select.value = 'opt2';

            expect(manager.getFieldValue(select)).toBe('opt2');
        });
    });

    describe('registerField', () => {
        let element;

        beforeEach(() => {
            element = document.createElement('input');
            element.type = 'text';
            element.value = 'initial';
        });

        test('stores initial state', () => {
            manager.registerField('test_field', element);

            const state = manager.fieldStates.get('test_field');
            expect(state.value).toBe('initial');
            expect(state.element).toBe(element);
            expect(state.visible).toBe(true);
            expect(state.enabled).toBe(true);
        });

        test('uses provided initial value', () => {
            manager.registerField('test_field', element, 'custom_value');

            const state = manager.fieldStates.get('test_field');
            expect(state.value).toBe('custom_value');
        });

        test('adds change listener', () => {
            manager.registerField('test_field', element);

            const changeEvent = new Event('change');
            element.value = 'new_value';
            element.dispatchEvent(changeEvent);

            const state = manager.fieldStates.get('test_field');
            expect(state.value).toBe('new_value');
        });
    });

    describe('fieldChanged', () => {
        beforeEach(() => {
            // Set up metadata with dependencies
            manager.loadMetadata({
                fields: {
                    trigger_field: { section: 'test' },
                    dependent_field: {
                        section: 'test',
                        dependencies: [
                            { field: 'trigger_field', operator: 'equals', value: 'show' },
                        ],
                    },
                },
            });

            // Set up DOM elements
            const triggerEl = document.createElement('input');
            triggerEl.name = 'trigger_field';
            const dependentEl = document.createElement('input');
            dependentEl.name = 'dependent_field';

            // Create container for dependent element
            const container = document.createElement('div');
            container.className = 'mb-3';
            container.appendChild(dependentEl);
            document.body.appendChild(container);

            manager.registerField('trigger_field', triggerEl, 'initial');
            manager.registerField('dependent_field', dependentEl, '');
        });

        afterEach(() => {
            document.body.innerHTML = '';
        });

        test('updates field state value', () => {
            manager.fieldChanged('trigger_field', 'new_value');

            const state = manager.fieldStates.get('trigger_field');
            expect(state.value).toBe('new_value');
        });

        test('dispatches fieldChanged custom event', () => {
            const listener = jest.fn();
            window.addEventListener('fieldChanged', listener);

            manager.fieldChanged('trigger_field', 'new_value');

            expect(listener).toHaveBeenCalled();
            expect(listener.mock.calls[0][0].detail).toEqual({
                field: 'trigger_field',
                value: 'new_value',
            });

            window.removeEventListener('fieldChanged', listener);
        });

        test('triggers dependency check on dependent fields', () => {
            // Initially trigger_field is 'initial', so dependent should be hidden
            // Then change to 'show', dependent should become visible

            manager.fieldChanged('trigger_field', 'show');

            const dependentState = manager.fieldStates.get('dependent_field');
            expect(dependentState.visible).toBe(true);
        });
    });

    describe('getContext', () => {
        test('returns all field values as object', () => {
            manager.fieldStates.set('field1', { value: 'a' });
            manager.fieldStates.set('field2', { value: 42 });
            manager.fieldStates.set('field3', { value: true });

            const context = manager.getContext();

            expect(context).toEqual({
                field1: 'a',
                field2: 42,
                field3: true,
            });
        });

        test('returns empty object when no fields', () => {
            expect(manager.getContext()).toEqual({});
        });
    });

    describe('getDependentFields', () => {
        test('returns array of dependent field names', () => {
            manager.observers.set('trigger', new Set(['dep1', 'dep2', 'dep3']));

            const dependents = manager.getDependentFields('trigger');

            expect(dependents).toHaveLength(3);
            expect(dependents).toContain('dep1');
            expect(dependents).toContain('dep2');
            expect(dependents).toContain('dep3');
        });

        test('returns empty array for field with no dependents', () => {
            expect(manager.getDependentFields('no_dependents')).toEqual([]);
        });
    });

    describe('isFieldVisible', () => {
        test('returns visibility state', () => {
            manager.fieldStates.set('visible_field', { visible: true });
            manager.fieldStates.set('hidden_field', { visible: false });

            expect(manager.isFieldVisible('visible_field')).toBe(true);
            expect(manager.isFieldVisible('hidden_field')).toBe(false);
        });

        test('returns false for unknown field', () => {
            expect(manager.isFieldVisible('unknown')).toBe(false);
        });
    });

    describe('reset', () => {
        test('clears field states', () => {
            manager.fieldStates.set('field1', { value: 'a' });
            manager.fieldStates.set('field2', { value: 'b' });

            manager.reset();

            // Field states should be cleared (and re-populated from DOM)
            // Since DOM is empty, should have no states
            expect(manager.fieldStates.size).toBe(0);
        });
    });
});

describe('DependencyManager real-world scenarios', () => {
    let manager;

    beforeEach(() => {
        manager = new window.dependencyManager.constructor();
        document.body.innerHTML = '';
    });

    afterEach(() => {
        document.body.innerHTML = '';
    });

    test('LoRA fields show only when training_type is lora', () => {
        manager.loadMetadata({
            fields: {
                training_type: { section: 'basic' },
                lora_rank: {
                    section: 'lora',
                    dependencies: [
                        { field: 'training_type', operator: 'equals', value: 'lora' },
                    ],
                },
                lora_alpha: {
                    section: 'lora',
                    dependencies: [
                        { field: 'training_type', operator: 'equals', value: 'lora' },
                    ],
                },
            },
        });

        // Create elements
        const typeEl = document.createElement('select');
        typeEl.name = 'training_type';

        const rankContainer = document.createElement('div');
        rankContainer.className = 'mb-3';
        const rankEl = document.createElement('input');
        rankEl.name = 'lora_rank';
        rankContainer.appendChild(rankEl);
        document.body.appendChild(rankContainer);

        const alphaContainer = document.createElement('div');
        alphaContainer.className = 'mb-3';
        const alphaEl = document.createElement('input');
        alphaEl.name = 'lora_alpha';
        alphaContainer.appendChild(alphaEl);
        document.body.appendChild(alphaContainer);

        manager.registerField('training_type', typeEl, 'full');
        manager.registerField('lora_rank', rankEl);
        manager.registerField('lora_alpha', alphaEl);

        // Initially hidden (training_type is 'full')
        expect(manager.isFieldVisible('lora_rank')).toBe(false);
        expect(manager.isFieldVisible('lora_alpha')).toBe(false);

        // Change to lora
        manager.fieldChanged('training_type', 'lora');

        expect(manager.isFieldVisible('lora_rank')).toBe(true);
        expect(manager.isFieldVisible('lora_alpha')).toBe(true);

        // Change back to full
        manager.fieldChanged('training_type', 'full');

        expect(manager.isFieldVisible('lora_rank')).toBe(false);
        expect(manager.isFieldVisible('lora_alpha')).toBe(false);
    });

    test('advanced options show when expert_mode is enabled', () => {
        manager.loadMetadata({
            fields: {
                expert_mode: { section: 'basic' },
                advanced_option: {
                    section: 'advanced',
                    dependencies: [
                        { field: 'expert_mode', operator: 'equals', value: true },
                    ],
                },
            },
        });

        const expertEl = document.createElement('input');
        expertEl.type = 'checkbox';
        expertEl.name = 'expert_mode';

        const advContainer = document.createElement('div');
        advContainer.className = 'mb-3';
        const advEl = document.createElement('input');
        advEl.name = 'advanced_option';
        advContainer.appendChild(advEl);
        document.body.appendChild(advContainer);

        manager.registerField('expert_mode', expertEl, false);
        manager.registerField('advanced_option', advEl);

        expect(manager.isFieldVisible('advanced_option')).toBe(false);

        manager.fieldChanged('expert_mode', true);
        expect(manager.isFieldVisible('advanced_option')).toBe(true);
    });

    test('numeric threshold dependencies', () => {
        manager.loadMetadata({
            fields: {
                learning_rate: { section: 'basic' },
                lr_warmup_warning: {
                    section: 'basic',
                    dependencies: [
                        { field: 'learning_rate', operator: 'greater_than', value: 0.001 },
                    ],
                },
            },
        });

        const lrEl = document.createElement('input');
        lrEl.type = 'number';
        lrEl.name = 'learning_rate';

        const warningContainer = document.createElement('div');
        warningContainer.className = 'mb-3';
        const warningEl = document.createElement('div');
        warningEl.id = 'lr_warmup_warning';
        warningContainer.appendChild(warningEl);
        document.body.appendChild(warningContainer);

        manager.registerField('learning_rate', lrEl, 0.0001);
        manager.registerField('lr_warmup_warning', warningEl);

        // Low LR - no warning
        expect(manager.isFieldVisible('lr_warmup_warning')).toBe(false);

        // High LR - show warning
        manager.fieldChanged('learning_rate', 0.01);
        expect(manager.isFieldVisible('lr_warmup_warning')).toBe(true);
    });
});
