/**
 * Tests for the training configuration wizard component.
 */

const { trainingWizardComponent } = require('../../simpletuner/static/js/training-wizard.js');

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
