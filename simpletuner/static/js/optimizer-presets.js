(function(root, factory) {
    const api = factory();

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    }

    if (root) {
        root.SimpleTunerOptimizerPresets = api;
    }
})(typeof window !== 'undefined' ? window : globalThis, function() {
    'use strict';

    const PRESET_KEYS = Object.freeze(['aggressive', 'moderate', 'slow_safe']);
    const PRESETS = Object.freeze({
        aggressive: Object.freeze({
            key: 'aggressive',
            label: 'Aggressive',
            tagline: 'Fast convergence, higher risk of instability.',
            optimizer: 'optimi-lion',
            fullLearningRate: 1e-5,
            loraLearningRate: 1e-4,
            batchSize: 2,
            gradientAccumulation: 1
        }),
        moderate: Object.freeze({
            key: 'moderate',
            label: 'Moderate',
            tagline: 'Balanced stability with AdamW BF16.',
            optimizer: 'adamw_bf16',
            fullLearningRate: 1e-5,
            loraLearningRate: 1e-4,
            batchSize: 2,
            gradientAccumulation: 1
        }),
        slow_safe: Object.freeze({
            key: 'slow_safe',
            label: 'Slow & Safe',
            tagline: 'Lower risk, more VRAM due to larger batch size.',
            optimizer: 'adamw_bf16',
            fullLearningRate: 1e-5,
            loraLearningRate: 1e-4,
            batchSize: 4,
            gradientAccumulation: 2
        })
    });

    function clonePreset(preset) {
        return preset ? { ...preset } : null;
    }

    function getPresetDefinition(key) {
        return clonePreset(PRESETS[key]);
    }

    function getPresetKeys() {
        return [...PRESET_KEYS];
    }

    function getLearningRateForModelType(presetOrKey, modelType) {
        const preset = typeof presetOrKey === 'string' ? getPresetDefinition(presetOrKey) : presetOrKey;
        if (!preset) {
            return null;
        }

        return modelType === 'lora' ? preset.loraLearningRate : preset.fullLearningRate;
    }

    function getPresetValues(key, modelType) {
        const preset = getPresetDefinition(key);
        if (!preset) {
            return null;
        }

        return {
            learning_rate: getLearningRateForModelType(preset, modelType),
            optimizer: preset.optimizer,
            train_batch_size: preset.batchSize,
            gradient_accumulation_steps: preset.gradientAccumulation
        };
    }

    function formatLearningRate(value) {
        if (typeof value !== 'number' || Number.isNaN(value)) {
            return value;
        }
        if (value === 0) {
            return '0';
        }
        if ((value > 0 && value < 0.001) || value >= 1000) {
            return value.toExponential();
        }
        return value.toString();
    }

    function getDisplayPresets(modelType) {
        return PRESET_KEYS.map(key => {
            const preset = PRESETS[key];

            return {
                key: preset.key,
                label: preset.label,
                tagline: preset.tagline,
                optimizer: preset.optimizer,
                batchSize: preset.batchSize,
                gradientAccumulation: preset.gradientAccumulation,
                displayLearningRate: formatLearningRate(getLearningRateForModelType(preset, modelType)),
                loraLearningRate: formatLearningRate(preset.loraLearningRate),
                fullLearningRate: formatLearningRate(preset.fullLearningRate)
            };
        });
    }

    function resolveOptimizerSelection(preferredValue, optimizerChoices) {
        const choices = Array.isArray(optimizerChoices) ? optimizerChoices : [];

        if (preferredValue && choices.some(choice => choice.value === preferredValue)) {
            return preferredValue;
        }
        if (choices.length > 0) {
            return choices[0].value;
        }
        return preferredValue || 'adamw_bf16';
    }

    function numberMatches(value, expected) {
        const numericValue = typeof value === 'number' ? value : Number(value);
        return Number.isFinite(numericValue) && numericValue === expected;
    }

    function findMatchingPresetKey(values, modelType) {
        if (!values || typeof values !== 'object') {
            return null;
        }

        return PRESET_KEYS.find(key => {
            const presetValues = getPresetValues(key, modelType);
            return (
                numberMatches(values.learning_rate, presetValues.learning_rate) &&
                values.optimizer === presetValues.optimizer &&
                numberMatches(values.train_batch_size, presetValues.train_batch_size) &&
                numberMatches(values.gradient_accumulation_steps, presetValues.gradient_accumulation_steps)
            );
        }) || null;
    }

    const helpers = Object.freeze({
        getPresetDefinition,
        getPresetKeys,
        getLearningRateForModelType,
        getPresetValues,
        getDisplayPresets,
        resolveOptimizerSelection,
        findMatchingPresetKey,
        formatLearningRate
    });

    function getTrainerStore() {
        return typeof window !== 'undefined' && window.Alpine?.store
            ? window.Alpine.store('trainer')
            : null;
    }

    function readConfigValue(fieldName) {
        const trainerStore = getTrainerStore();
        const config = trainerStore?.activeEnvironmentConfig;
        if (!config) {
            return undefined;
        }
        return config[`--${fieldName}`] ?? config[fieldName];
    }

    function readDomValue(fieldName) {
        if (typeof document === 'undefined') {
            return undefined;
        }

        const el = document.getElementById(fieldName) || document.querySelector(`[name="${fieldName}"]`);
        if (!el) {
            return undefined;
        }

        if (el.type === 'checkbox') {
            return el.checked;
        }
        return el.value;
    }

    function readFieldValue(fieldName) {
        const configValue = readConfigValue(fieldName);
        if (configValue !== undefined && configValue !== null && configValue !== '') {
            return configValue;
        }
        return readDomValue(fieldName);
    }

    function writeFieldValue(fieldName, value) {
        const canonicalKey = `--${fieldName}`;
        const trainerStore = getTrainerStore();

        if (trainerStore) {
            if (typeof trainerStore.updateConfigValue === 'function') {
                trainerStore.updateConfigValue(canonicalKey, value);
            } else if (trainerStore.activeEnvironmentConfig) {
                trainerStore.activeEnvironmentConfig[canonicalKey] = value;
            }

            if (trainerStore.configValues) {
                trainerStore.configValues[canonicalKey] = value;
            }

            if (trainerStore.formValueStore) {
                trainerStore.formValueStore[canonicalKey] = { kind: 'single', value };
            }
        }

        if (typeof document !== 'undefined') {
            const el = document.getElementById(fieldName) || document.querySelector(`[name="${fieldName}"]`);
            if (el) {
                if (el.type === 'checkbox') {
                    el.checked = Boolean(value);
                } else {
                    el.value = value ?? '';
                }
                el.dispatchEvent(new Event('change', { bubbles: true }));
            }
        }
    }

    function getOptimizerChoicesFromDom() {
        if (typeof document === 'undefined') {
            return [];
        }

        const select = document.getElementById('optimizer');
        if (!select) {
            return [];
        }

        return Array.from(select.options)
            .filter(option => option.value !== '')
            .map(option => ({
                value: option.value,
                label: option.textContent.trim() || option.value
            }));
    }

    function optimizerPresetsComponent() {
        return {
            isOpen: false,
            selectedPreset: null,
            modelType: 'lora',
            optimizerChoices: [],
            _eventHandler: null,

            init() {
                window.__optimizerPresetsInstance = this;
                this._eventHandler = () => this.open();
                window.addEventListener('open-optimizer-presets', this._eventHandler);
            },

            destroy() {
                if (this._eventHandler) {
                    window.removeEventListener('open-optimizer-presets', this._eventHandler);
                }
                if (window.__optimizerPresetsInstance === this) {
                    window.__optimizerPresetsInstance = null;
                }
            },

            open() {
                this.modelType = this.getModelType();
                this.optimizerChoices = getOptimizerChoicesFromDom();
                this.selectedPreset = this.getCurrentPresetKey();
                this.isOpen = true;
            },

            close() {
                this.isOpen = false;
            },

            getModelType() {
                const modelType = readFieldValue('model_type');
                return modelType === 'full' ? 'full' : 'lora';
            },

            getCurrentPresetKey() {
                return findMatchingPresetKey({
                    learning_rate: readFieldValue('learning_rate'),
                    optimizer: readFieldValue('optimizer'),
                    train_batch_size: readFieldValue('train_batch_size'),
                    gradient_accumulation_steps: readFieldValue('gradient_accumulation_steps')
                }, this.modelType);
            },

            getOptimizerPresetCards() {
                return getDisplayPresets(this.modelType);
            },

            isOptimizerPresetSelected(presetKey) {
                return this.selectedPreset === presetKey;
            },

            selectOptimizerPreset(presetKey) {
                this.selectedPreset = presetKey;
            },

            applySelectedPreset() {
                if (!this.selectedPreset) {
                    window.showToast?.('Select an optimizer preset first.', 'warning');
                    return;
                }

                const values = getPresetValues(this.selectedPreset, this.modelType);
                if (!values) {
                    window.showToast?.('Unknown optimizer preset selected.', 'error');
                    return;
                }

                const optimizer = resolveOptimizerSelection(values.optimizer, this.optimizerChoices);
                writeFieldValue('learning_rate', values.learning_rate);
                writeFieldValue('optimizer', optimizer);
                writeFieldValue('train_batch_size', values.train_batch_size);
                writeFieldValue('gradient_accumulation_steps', values.gradient_accumulation_steps);

                const trainerStore = getTrainerStore();
                if (trainerStore) {
                    if (typeof trainerStore.applyStoredValues === 'function') {
                        trainerStore.applyStoredValues();
                    }
                    if (typeof trainerStore.markFormDirty === 'function') {
                        trainerStore.markFormDirty();
                    }
                }

                const preset = getPresetDefinition(this.selectedPreset);
                this.close();
                window.showToast?.(`${preset.label} optimizer preset applied`, 'success');
            }
        };
    }

    function injectOptimizerPresetsButton() {
        if (typeof document === 'undefined') {
            return;
        }

        const section = document.getElementById('section-optimizer_config');
        if (!section || section.querySelector('.optimizer-presets-btn')) {
            return;
        }

        const sectionTitle = section.querySelector('.section-title');
        if (!sectionTitle) {
            return;
        }

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm btn-outline-primary optimizer-presets-btn ms-2';
        btn.innerHTML = '<i class="fas fa-magic me-1"></i>Load Presets';
        btn.title = 'Load optimizer presets';
        btn.addEventListener('click', event => {
            event.preventDefault();
            event.stopPropagation();
            window.openOptimizerPresetsModal?.();
        });

        const badge = sectionTitle.querySelector('.badge');
        if (badge) {
            sectionTitle.insertBefore(btn, badge);
        } else {
            sectionTitle.appendChild(btn);
        }
    }

    function setupOptimizerPresetUi() {
        if (typeof window === 'undefined' || typeof document === 'undefined') {
            return;
        }

        if (window.Alpine) {
            window.Alpine.data('optimizerPresetsComponent', optimizerPresetsComponent);
        } else {
            document.addEventListener('alpine:init', () => {
                window.Alpine.data('optimizerPresetsComponent', optimizerPresetsComponent);
            });
        }

        window.openOptimizerPresetsModal = function() {
            const instance = window.__optimizerPresetsInstance;
            if (instance) {
                instance.open();
            } else {
                window.dispatchEvent(new CustomEvent('open-optimizer-presets'));
            }
        };

        const scheduleInjection = () => setTimeout(injectOptimizerPresetsButton, 50);
        const registerButtonInjection = () => {
            scheduleInjection();
            if (window.__optimizerPresetsButtonInjectionRegistered || !document.body) {
                return;
            }
            window.__optimizerPresetsButtonInjectionRegistered = true;
            document.body.addEventListener('htmx:afterSwap', event => {
                if (
                    event.detail.target.id === 'tab-content' ||
                    event.detail.target.id === 'training-tab-content' ||
                    event.detail.target.querySelector?.('#section-optimizer_config')
                ) {
                    scheduleInjection();
                }
            });

            document.body.addEventListener('htmx:afterSettle', () => {
                if (document.getElementById('section-optimizer_config')) {
                    scheduleInjection();
                }
            });
        };

        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', registerButtonInjection);
        } else {
            registerButtonInjection();
        }
    }

    setupOptimizerPresetUi();

    return helpers;
});
