/**
 * Tests to verify Easy Mode event handlers use .stop to prevent bubbling.
 *
 * Background: Easy Mode controls don't have `name` attributes, so they're not
 * included in FormData. When their events bubble to the form's handleFieldInput(),
 * checkFormDirty() runs and resets formDirty because FormData shows no changes.
 *
 * Solution: All Easy Mode @change and @input handlers must use .stop modifier
 * to prevent event bubbling.
 *
 * This test does static analysis of the template to catch regressions where
 * someone accidentally removes the .stop modifier.
 */

const fs = require('fs');
const path = require('path');

describe('Easy Mode Event Handler Static Analysis', () => {
    let formTabContent;

    beforeAll(() => {
        const templatePath = path.join(__dirname, '../../simpletuner/templates/form_tab.html');
        formTabContent = fs.readFileSync(templatePath, 'utf8');
    });

    describe('Easy Mode handlers must use .stop modifier', () => {
        // These are the Easy Mode handler patterns that MUST have .stop
        const ezModeHandlers = [
            // Model EZ Mode
            'onModelFamilyChange',
            'onModelFlavourChange',
            'onModelTypeChange',
            'onLoraRankSlider',
            'onGradientCheckpointingChange',
            'onRamtorchPresetChange',
            'onBlockSwapSlider',
            'onVaeTilingChange',
            'onVaeSlicingChange',
            'onVaeBatchSizeChange',
            'onBaseModelPrecisionChange',
            'onTextEncoderPrecisionChange',
            'onUseEmaChange',

            // Basic EZ Mode
            'onReportToChange',
            'onProjectNameChange',
            'onRunNameChange',
            'onOutputDirChange',
            'onResumeFromCheckpointChange',
            'onDataBackendConfigChange',
            'onOverrideDatasetConfigChange',
            'onDisableBucketPruningChange',
            'onAllowOversubscriptionChange',
            'onBatchSizeChange',
            'onResolutionChange',
            'onMinImageSizeChange',
            'onMaxImageSizeChange',
            'onTargetDownsampleChange',
            'onCaptionStrategyChange',

            // Training EZ Mode
            'onNumEpochsChange',
            'onMaxStepsChange',
            'onCheckpointStepIntervalChange',
            'onCheckpointEpochIntervalChange',
            'onCheckpointsTotalLimitChange',
            'onLearningRateChange',
            'onLrSchedulerChange',
            'onLrWarmupStepsChange',
            'onLrEndChange',
            'onOptimizerChange',
            'onMaxGradNormChange',
            'onFlowAutoShiftChange',
            'onFlowShiftChange',

            // Validation EZ Mode
            'onValidationStepIntervalChange',
            'onValidationEpochIntervalChange',
            'onValidationInferenceStepsChange',
            'onValidationPromptChange',
            'onValidationNegativePromptChange',
            'onUserPromptLibraryChange',
            'onValidationGuidanceChange',
            'onValidationRandomizeChange',
            'onValidationSeedChange',
        ];

        test.each(ezModeHandlers)('%s must use @change.stop or @input.stop', (handlerName) => {
            // Find all occurrences of this handler in the template
            const regex = new RegExp(`@(change|input)(\\.stop)?="${handlerName}`, 'g');
            const matches = [...formTabContent.matchAll(regex)];

            expect(matches.length).toBeGreaterThan(0);

            // Every match must have .stop
            for (const match of matches) {
                const hasStop = match[2] === '.stop';
                expect(hasStop).toBe(true);
            }
        });

        test('no Easy Mode handlers use @change without .stop', () => {
            // Find any @change="on... without .stop
            const badPatterns = formTabContent.match(/@change="on[A-Z][a-zA-Z]+Change/g) || [];
            expect(badPatterns).toEqual([]);
        });

        test('no Easy Mode handlers use @input without .stop', () => {
            // Find any @input="on... without .stop
            const badPatterns = formTabContent.match(/@input="on[A-Z][a-zA-Z]+/g) || [];
            expect(badPatterns).toEqual([]);
        });
    });

    describe('Event bubbling prevention rationale', () => {
        test('Easy Mode controls should NOT have name attributes', () => {
            // Easy Mode controls with name attributes would be included in FormData,
            // which could cause duplicate values or conflicts with the actual form fields.
            // Verify that EZ Mode specific controls don't have names.

            // Look for ez_ prefixed IDs that also have name attributes (shouldn't exist)
            const ezWithName = formTabContent.match(/id="ez_[^"]+"\s+[^>]*name="/g) || [];
            expect(ezWithName).toEqual([]);
        });

        test('updateFormField is the only path for Easy Mode to update dirty state', () => {
            // Every EZ Mode component should have an updateFormField method
            // that sets formDirty directly
            const updateFormFieldDefs = formTabContent.match(/updateFormField\s*\([^)]+\)\s*\{/g) || [];

            // Should have at least 4 (one per EZ Mode component: model, basic, training, validation)
            expect(updateFormFieldDefs.length).toBeGreaterThanOrEqual(4);
        });
    });
});
