/**
 * Dependency Manager for SimpleTuner Configuration Fields
 *
 * This service manages field interdependencies, handling visibility,
 * validation, and state updates based on field relationships.
 */

class DependencyManager {
    constructor() {
        this.dependencies = new Map(); // field -> dependency rules
        this.observers = new Map(); // field -> fields that depend on it
        this.fieldStates = new Map(); // current field values/states
        this.fieldMetadata = new Map(); // field metadata from backend
        this.initialized = false;
    }

    /**
     * Initialize the dependency manager with field metadata
     * @param {Object} metadata - Field metadata from backend
     */
    async initialize(metadata) {
        if (metadata) {
            this.loadMetadata(metadata);
        } else {
            // Fetch from backend if not provided
            await this.fetchFieldMetadata();
        }
        this.initialized = true;
    }

    /**
     * Fetch field metadata from backend
     */
    async fetchFieldMetadata() {
        try {
            const response = await fetch('/api/fields/metadata');
            if (!response.ok) {
                throw new Error('Failed to fetch field metadata');
            }
            const data = await response.json();
            this.loadMetadata(data);
        } catch (error) {
            console.error('Error fetching field metadata:', error);
        }
    }

    /**
     * Load metadata and build dependency maps
     * @param {Object} metadata - Field metadata object
     */
    loadMetadata(metadata) {
        // Clear existing data
        this.dependencies.clear();
        this.observers.clear();
        this.fieldMetadata.clear();

        // Process each field
        for (const [fieldName, fieldData] of Object.entries(metadata.fields)) {
            this.fieldMetadata.set(fieldName, fieldData);

            // Register dependencies
            if (fieldData.dependencies && fieldData.dependencies.length > 0) {
                this.dependencies.set(fieldName, fieldData.dependencies);

                // Build reverse mapping (observers)
                fieldData.dependencies.forEach(dep => {
                    if (!this.observers.has(dep.field)) {
                        this.observers.set(dep.field, new Set());
                    }
                    this.observers.get(dep.field).add(fieldName);
                });
            }
        }
    }

    /**
     * Register a field element with the dependency manager
     * @param {string} fieldName - Field name
     * @param {HTMLElement} element - Field element
     * @param {*} initialValue - Initial field value
     */
    registerField(fieldName, element, initialValue = null) {
        // Store initial state
        this.fieldStates.set(fieldName, {
            value: initialValue || this.getFieldValue(element),
            element: element,
            visible: true,
            enabled: true
        });

        // Add change listener
        element.addEventListener('change', (e) => {
            this.fieldChanged(fieldName, this.getFieldValue(element));
        });

        // Initial dependency check
        this.checkFieldDependencies(fieldName);
    }

    /**
     * Handle field value change
     * @param {string} fieldName - Changed field name
     * @param {*} newValue - New field value
     */
    fieldChanged(fieldName, newValue) {
        // Update field state
        const state = this.fieldStates.get(fieldName);
        if (state) {
            state.value = newValue;
        }

        // Notify dependent fields
        const dependents = this.observers.get(fieldName);
        if (dependents) {
            dependents.forEach(dependentField => {
                this.checkFieldDependencies(dependentField);
            });
        }

        // Emit change event for other systems
        window.dispatchEvent(new CustomEvent('fieldChanged', {
            detail: { field: fieldName, value: newValue }
        }));
    }

    /**
     * Check and update field based on its dependencies
     * @param {string} fieldName - Field to check
     */
    checkFieldDependencies(fieldName) {
        const dependencies = this.dependencies.get(fieldName);
        if (!dependencies || dependencies.length === 0) return;

        const state = this.fieldStates.get(fieldName);
        if (!state) return;

        // Check if all dependencies are satisfied
        const isVisible = this.evaluateDependencies(dependencies);

        // Update field visibility
        this.setFieldVisibility(fieldName, isVisible);
    }

    /**
     * Evaluate dependency rules
     * @param {Array} dependencies - Array of dependency rules
     * @returns {boolean} - Whether all dependencies are satisfied
     */
    evaluateDependencies(dependencies) {
        // All dependencies must be satisfied (AND logic)
        return dependencies.every(dep => this.evaluateSingleDependency(dep));
    }

    /**
     * Evaluate a single dependency rule
     * @param {Object} dep - Dependency rule
     * @returns {boolean} - Whether dependency is satisfied
     */
    evaluateSingleDependency(dep) {
        const depState = this.fieldStates.get(dep.field);
        if (!depState) return true; // If dependency field doesn't exist, assume satisfied

        const depValue = depState.value;

        switch (dep.operator) {
            case 'equals':
                return depValue === dep.value;

            case 'not_equals':
                return depValue !== dep.value;

            case 'in':
                return dep.values && dep.values.includes(depValue);

            case 'not_in':
                return dep.values && !dep.values.includes(depValue);

            case 'greater_than':
                return parseFloat(depValue) > parseFloat(dep.value);

            case 'less_than':
                return parseFloat(depValue) < parseFloat(dep.value);

            default:
                console.warn(`Unknown dependency operator: ${dep.operator}`);
                return true;
        }
    }

    /**
     * Set field visibility
     * @param {string} fieldName - Field name
     * @param {boolean} visible - Whether field should be visible
     */
    setFieldVisibility(fieldName, visible) {
        const state = this.fieldStates.get(fieldName);
        if (!state) return;

        state.visible = visible;

        // Hidden fields are used for proxy values in custom UIs; nothing to toggle visibly
        if (state.element.type === 'hidden') {
            state.element.disabled = !visible;
            return;
        }

        // Find the field container (usually the parent element with mb-3 class)
        let container = state.element.closest('.mb-3') ||
                       state.element.closest('.field-wrapper') ||
                       state.element.parentElement;

        if (container) {
            if (visible) {
                container.style.display = '';
                container.classList.remove('field-hidden');
                // Enable the field
                state.element.disabled = false;
            } else {
                container.style.display = 'none';
                container.classList.add('field-hidden');
                // Disable the field to prevent submission
                state.element.disabled = true;
            }
        }

        // For sections, check if all fields are hidden
        this.updateSectionVisibility(fieldName);
    }

    /**
     * Update section visibility based on field visibility
     * @param {string} fieldName - Field that changed
     */
    updateSectionVisibility(fieldName) {
        const metadata = this.fieldMetadata.get(fieldName);
        if (!metadata) return;

        const sectionId = `section-${metadata.section}`;
        const section = document.getElementById(sectionId);
        if (!section) return;

        // Check if any fields in this section are visible
        let hasVisibleFields = false;
        this.fieldMetadata.forEach((field, name) => {
            if (field.section === metadata.section) {
                const state = this.fieldStates.get(name);
                if (state && state.visible) {
                    hasVisibleFields = true;
                }
            }
        });

        // Update section visibility
        section.style.display = hasVisibleFields ? '' : 'none';
    }

    /**
     * Get field value from element
     * @param {HTMLElement} element - Field element
     * @returns {*} - Field value
     */
    getFieldValue(element) {
        if (element.type === 'checkbox') {
            return element.checked;
        } else if (element.type === 'number') {
            return parseFloat(element.value) || 0;
        } else {
            return element.value;
        }
    }

    /**
     * Get current context (all field values)
     * @returns {Object} - Current field values
     */
    getContext() {
        const context = {};
        this.fieldStates.forEach((state, fieldName) => {
            context[fieldName] = state.value;
        });
        return context;
    }

    /**
     * Batch update field values
     * @param {Object} values - Field values to update
     */
    updateFieldValues(values) {
        Object.entries(values).forEach(([fieldName, value]) => {
            const state = this.fieldStates.get(fieldName);
            if (state && state.element) {
                // Update element value
                if (state.element.type === 'checkbox') {
                    if (Array.isArray(value)) {
                        state.element.checked = value.includes(state.element.value || 'on');
                    } else {
                        state.element.checked = Boolean(value);
                    }
                } else {
                    state.element.value = value;
                }
                // Trigger change
                this.fieldChanged(fieldName, value);
            }
        });
    }

    /**
     * Get fields affected by a specific field
     * @param {string} fieldName - Field name
     * @returns {Array} - Array of dependent field names
     */
    getDependentFields(fieldName) {
        const dependents = this.observers.get(fieldName);
        return dependents ? Array.from(dependents) : [];
    }

    /**
     * Check if a field is currently visible
     * @param {string} fieldName - Field name
     * @returns {boolean} - Whether field is visible
     */
    isFieldVisible(fieldName) {
        const state = this.fieldStates.get(fieldName);
        return state ? state.visible : false;
    }

    /**
     * Get field metadata
     * @param {string} fieldName - Field name
     * @returns {Object} - Field metadata
     */
    getFieldMetadata(fieldName) {
        return this.fieldMetadata.get(fieldName);
    }

    /**
     * Initialize all fields on a page/tab
     * @param {HTMLElement} container - Container element
     */
    initializeFieldsInContainer(container) {
        // Find all form fields
        const fields = container.querySelectorAll('input, select, textarea');

        fields.forEach(element => {
            const fieldName = element.name?.replace('--', '') || element.id;
            if (fieldName && this.fieldMetadata.has(fieldName)) {
                this.registerField(fieldName, element);
            }
        });
    }

    /**
     * Reset dependency manager
     */
    reset() {
        this.fieldStates.clear();
        // Re-initialize visible fields
        document.querySelectorAll('input:not([type="hidden"]), select, textarea').forEach(element => {
            const fieldName = element.name?.replace('--', '') || element.id;
            if (fieldName && this.fieldMetadata.has(fieldName)) {
                this.registerField(fieldName, element);
            }
        });
    }
}

// Create singleton instance
window.dependencyManager = new DependencyManager();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DependencyManager;
}
