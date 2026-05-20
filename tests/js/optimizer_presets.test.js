const OptimizerPresets = require('../../simpletuner/static/js/optimizer-presets.js');

describe('OptimizerPresets', () => {
    test('exposes wizard preset keys in display order', () => {
        expect(OptimizerPresets.getPresetKeys()).toEqual(['aggressive', 'moderate', 'slow_safe']);
    });

    test('returns preset values for LoRA and full model training', () => {
        expect(OptimizerPresets.getPresetValues('aggressive', 'lora')).toEqual({
            learning_rate: 1e-4,
            optimizer: 'optimi-lion',
            train_batch_size: 2,
            gradient_accumulation_steps: 1,
        });

        expect(OptimizerPresets.getPresetValues('aggressive', 'full')).toEqual({
            learning_rate: 1e-5,
            optimizer: 'optimi-lion',
            train_batch_size: 2,
            gradient_accumulation_steps: 1,
        });
    });

    test('formats display presets with current model type learning rate', () => {
        const [aggressive] = OptimizerPresets.getDisplayPresets('lora');

        expect(aggressive.displayLearningRate).toBe('1e-4');
        expect(aggressive.loraLearningRate).toBe('1e-4');
        expect(aggressive.fullLearningRate).toBe('1e-5');
    });

    test('matches a saved config to a preset', () => {
        const match = OptimizerPresets.findMatchingPresetKey({
            learning_rate: '0.0001',
            optimizer: 'adamw_bf16',
            train_batch_size: '4',
            gradient_accumulation_steps: '2',
        }, 'lora');

        expect(match).toBe('slow_safe');
    });

    test('resolves preferred optimizer when available and first choice otherwise', () => {
        const choices = [
            { value: 'adamw_bf16', label: 'adamw_bf16' },
            { value: 'optimi-lion', label: 'optimi-lion' },
        ];

        expect(OptimizerPresets.resolveOptimizerSelection('optimi-lion', choices)).toBe('optimi-lion');
        expect(OptimizerPresets.resolveOptimizerSelection('missing', choices)).toBe('adamw_bf16');
    });
});
