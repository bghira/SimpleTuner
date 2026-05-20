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

    return Object.freeze({
        getPresetDefinition,
        getPresetKeys,
        getLearningRateForModelType,
        getPresetValues,
        getDisplayPresets,
        resolveOptimizerSelection,
        findMatchingPresetKey,
        formatLearningRate
    });
});
