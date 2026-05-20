/**
 * Tests for the training configuration wizard component.
 */

const { trainingWizardComponent } = require('../../simpletuner/static/js/training-wizard.js');
const OptimizerPresets = require('../../simpletuner/static/js/optimizer-presets.js');

describe('trainingWizardComponent step titles', () => {
    let component;

    beforeEach(() => {
        component = trainingWizardComponent();
    });

    test('numbered static step titles are unique and sequential', () => {
        const numberedTitles = component.steps
            .map((step) => step.title.match(/Step (\d+):/))
            .filter(Boolean)
            .map((match) => Number(match[1]));

        expect(numberedTitles).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    });

    test('currentStepTitle follows visible step order when optional variant step is hidden', () => {
        component.answers.model_family = 'sdxl';
        component.answers.model_flavour = null;

        const visibleIds = component.visibleSteps.map((step) => step.id);
        expect(visibleIds).not.toContain('model-flavour');

        component.currentStepIndex = visibleIds.indexOf('publishing');

        expect(component.currentStepTitle).toBe('Training Configuration Wizard - Step 5: Publishing');
    });

    test('currentStepTitle follows visible step order when optional variant step is visible', () => {
        component.answers.model_family = 'flux';
        component.answers.model_flavour = 'dev';
        component.modelFlavours = ['dev', 'schnell'];

        const visibleIds = component.visibleSteps.map((step) => step.id);
        expect(visibleIds).toContain('model-flavour');

        component.currentStepIndex = visibleIds.indexOf('publishing');

        expect(component.currentStepTitle).toBe('Training Configuration Wizard - Step 6: Publishing');
    });
});

describe('trainingWizardComponent optimizer presets', () => {
    let component;

    beforeEach(() => {
        component = trainingWizardComponent();
        component.optimizerChoices = [
            { value: 'adamw_bf16', label: 'adamw_bf16' },
            { value: 'optimi-lion', label: 'optimi-lion' },
        ];
    });

    test('uses shared optimizer preset display values', () => {
        component.answers.model_type = 'lora';

        expect(component.getOptimizerPresetCards()).toEqual(OptimizerPresets.getDisplayPresets('lora'));
    });

    test('applies shared optimizer preset values without advancing by itself', () => {
        component.answers.model_type = 'full';

        const applied = component.applyOptimizerPreset('aggressive');

        expect(applied).toBe(true);
        expect(component.selectedPreset).toBe('aggressive');
        expect(component.answers.learning_rate).toBe(1e-5);
        expect(component.answers.optimizer).toBe('optimi-lion');
        expect(component.answers.train_batch_size).toBe(2);
        expect(component.answers.gradient_accumulation_steps).toBe(1);
    });
});
