/**
 * Training Setup Wizard Component
 * Guides users through training configuration step-by-step
 */

const FLUX_FLAVOURS = [
    { value: 'krea', label: 'Krea', description: 'Default FLUX.1-Krea dev model (open weights)' },
    { value: 'dev', label: 'Dev', description: 'Original FLUX.1-Dev (commercial use requires license)' },
    { value: 'schnell', label: 'Schnell', description: 'Fast inference variant' },
    { value: 'kontext', label: 'Kontext', description: 'Context-aware training' },
    { value: 'fluxbooru', label: 'FluxBooru', description: 'De-distilled variant for CFG' },
    { value: 'libreflux', label: 'LibreFlux', description: 'De-distilled Schnell variant' }
];

const SD3_FLAVOURS = [
    { value: 'medium', label: 'Medium', description: 'SD3 Medium model' },
    { value: '3.5-large', label: '3.5 Large', description: 'SD3.5 Large model' },
    { value: '3.5-medium', label: '3.5 Medium', description: 'SD3.5 Medium model' }
];

function trainingWizardComponent() {
    return {
        // State
        wizardOpen: false,
        currentStepIndex: 0,
        modelsLoading: true,
        modelsError: null,
        modelFlavours: [],
        loggingProviders: [],
        pendingDatasetPlan: null,  // Holds dataset plan from dataset wizard when using deferred commit
        answers: {
            model_family: null,
            model_flavour: null,
            model_type: 'lora',  // Default to LoRA
            push_to_hub: false,  // Default to keeping files local
            push_checkpoints_to_hub: null,
            checkpointing_steps: 100,  // Default to 100 steps
            enable_validations: true,
            validation_steps: 100,
            validation_prompt: '',
            report_to: 'none',
            learning_rate: null,
            optimizer: null
        },

        // Configuration
        modelFamilyOptions: [],

        // Step definitions
        steps: [
            {
                id: 'model-family',
                label: 'Model',
                title: 'Training Setup Wizard - Step 1: Model Family',
                required: true,
                validate: function() { return this.answers.model_family !== null; }
            },
            {
                id: 'model-flavour',
                label: 'Variant',
                title: 'Training Setup Wizard - Step 2: Model Variant',
                required: false,
                condition: function() { return this.needsModelFlavour(); },
                validate: function() { return !this.needsModelFlavour() || this.answers.model_flavour !== null; }
            },
            {
                id: 'training-type',
                label: 'Type',
                title: 'Training Setup Wizard - Step 3: Training Type',
                required: true,
                validate: function() { return this.answers.model_type !== null; }
            },
            {
                id: 'publishing',
                label: 'Publishing',
                title: 'Training Setup Wizard - Step 4: Publishing',
                required: false,
                validate: function() { return true; } // Always valid, can skip
            },
            {
                id: 'checkpoints',
                label: 'Checkpoints',
                title: 'Training Setup Wizard - Step 5: Checkpoints',
                required: true,
                validate: function() { return this.answers.checkpointing_steps > 0; }
            },
            {
                id: 'validations-enable',
                label: 'Validations',
                title: 'Training Setup Wizard - Step 6: Validations',
                required: true,
                validate: function() { return this.answers.enable_validations !== null; }
            },
            {
                id: 'validation-prompt',
                label: 'Prompt',
                title: 'Training Setup Wizard - Step 7: Validation Prompt',
                required: false,
                condition: function() { return this.answers.enable_validations === true; },
                validate: function() { return !this.answers.enable_validations || this.answers.validation_prompt.trim().length > 0; }
            },
            {
                id: 'logging',
                label: 'Logging',
                title: 'Training Setup Wizard - Step 8: Logging',
                required: false,
                validate: function() { return true; }
            },
            {
                id: 'dataset',
                label: 'Dataset',
                title: 'Training Setup Wizard - Step 9: Dataset',
                required: true,
                validate: function() { return this.hasExistingDataset || this.datasetConfigured; }
            },
            {
                id: 'advanced',
                label: 'Advanced',
                title: 'Training Setup Wizard - Step 10: Advanced Settings',
                required: false,
                validate: function() { return true; }
            },
            {
                id: 'review',
                label: 'Review',
                title: 'Training Setup Wizard - Review Configuration',
                required: true,
                validate: function() { return true; }
            }
        ],

        datasetConfigured: false,
        hasExistingDataset: false,

        // Computed properties
        get currentStep() {
            const visibleSteps = this.visibleSteps;
            const step = visibleSteps[this.currentStepIndex] || visibleSteps[0] || this.steps[0];
            return step;
        },

        get canProceed() {
            const result = this.currentStep && this.currentStep.validate.call(this);
            return result;
        },

        get visibleSteps() {
            const filtered = this.steps.filter(step => {
                try {
                    const shouldShow = !step.condition || step.condition.call(this);
                    return shouldShow;
                } catch (error) {
                    console.error('[TRAINING WIZARD] Error checking step condition for', step.id, error);
                    return true; // Show step if there's an error
                }
            });
            return filtered;
        },

        // Initialization
        async init() {
            console.log('[TRAINING WIZARD] Initializing...');

            try {
                // Load current configuration to pre-populate answers
                await this.loadCurrentConfig();

                // Check if dataset exists
                await this.checkDataset();

                console.log('[TRAINING WIZARD] Ready');
            } catch (error) {
                console.error('[TRAINING WIZARD] Init error:', error);
            }
        },

        async loadCurrentConfig() {
            try {
                // Get current config from trainer store
                const trainerStore = Alpine.store('trainer');
                if (trainerStore && trainerStore.configValues) {
                    const config = trainerStore.configValues;

                    // Pre-populate answers from config
                    if (config.model_family) this.answers.model_family = config.model_family;
                    if (config.model_flavour) this.answers.model_flavour = config.model_flavour;
                    if (config.model_type) this.answers.model_type = config.model_type;
                    if (config.push_to_hub !== undefined) {
                        this.answers.push_to_hub = config.push_to_hub === true || config.push_to_hub === 'true';
                    }
                    if (config.push_checkpoints_to_hub !== undefined) {
                        this.answers.push_checkpoints_to_hub = config.push_checkpoints_to_hub === true || config.push_checkpoints_to_hub === 'true';
                    }
                    if (config.checkpointing_steps) this.answers.checkpointing_steps = parseInt(config.checkpointing_steps);
                    if (config.validation_steps) this.answers.validation_steps = parseInt(config.validation_steps);
                    if (config.validation_prompt) this.answers.validation_prompt = config.validation_prompt;
                    if (config.report_to) this.answers.report_to = config.report_to;
                    if (config.disable_validations !== undefined) {
                        this.answers.enable_validations = !(config.disable_validations === true || config.disable_validations === 'true');
                    }

                    console.log('[TRAINING WIZARD] Loaded current config:', this.answers);
                }
            } catch (error) {
                console.error('[TRAINING WIZARD] Failed to load config:', error);
            }
        },

        async checkDataset() {
            try {
                const response = await fetch('/api/datasets/plan');
                if (response.ok) {
                    const data = await response.json();
                    this.hasExistingDataset = (data.datasets && data.datasets.length > 0);
                }
            } catch (error) {
                console.error('[TRAINING WIZARD] Failed to check dataset:', error);
            }
        },

        // Navigation methods
        async openWizard() {
            console.log('[TRAINING WIZARD] Opening wizard');
            this.wizardOpen = true;
            this.currentStepIndex = 0;

            // Load model families if not already loaded
            if (this.modelFamilyOptions.length === 0) {
                await this.loadModelFamilies();
            }

            // Load logging providers if not already loaded
            if (this.loggingProviders.length === 0) {
                await this.loadLoggingProviders();
            }

            // Re-check dataset on open
            this.checkDataset();
        },

        async loadModelFamilies() {
            console.log('[TRAINING WIZARD] Loading model families from API');
            this.modelsLoading = true;
            this.modelsError = null;

            try {
                const response = await fetch('/api/models/wizard');
                if (!response.ok) {
                    throw new Error(`Failed to load models: ${response.statusText}`);
                }

                const data = await response.json();
                console.log('[TRAINING WIZARD] Loaded models:', data);

                // Transform API response to match the expected format
                this.modelFamilyOptions = data.models.map(model => ({
                    value: model.family,
                    label: model.name,
                    description: model.description
                }));

                console.log('[TRAINING WIZARD] Transformed model options:', this.modelFamilyOptions);
            } catch (error) {
                console.error('[TRAINING WIZARD] Error loading models:', error);
                this.modelsError = error.message;
                // Fallback to empty array - user will see error message
                this.modelFamilyOptions = [];
            } finally {
                this.modelsLoading = false;
            }
        },

        closeWizard() {
            if (this.currentStepIndex > 0 && this.currentStepIndex < this.visibleSteps.length - 1) {
                if (!confirm('Exit wizard? Your progress will be lost.')) {
                    return;
                }
            }
            this.wizardOpen = false;
        },

        nextStep() {
            if (!this.canProceed) {
                window.showToast && window.showToast('Please complete this step before continuing', 'warning');
                return;
            }

            // Apply current answer to form
            this.applyAnswersToForm();

            // Move to next visible step
            const visibleSteps = this.visibleSteps;
            if (this.currentStepIndex < visibleSteps.length - 1) {
                this.currentStepIndex++;
            }
        },

        previousStep() {
            if (this.currentStepIndex > 0) {
                this.currentStepIndex--;
            }
        },

        skipStep() {
            if (this.currentStep.required) {
                window.showToast && window.showToast('This step is required', 'warning');
                return;
            }
            this.currentStep.skipped = true;
            this.nextStep();
        },

        goToStep(index) {
            // Only allow going back to completed steps
            if (index < this.currentStepIndex) {
                this.currentStepIndex = index;
            }
        },

        async finishWizard() {
            console.log('[TRAINING WIZARD] Finishing wizard...');

            // Apply all answers to form
            this.applyAnswersToForm();

            // Save the config to disk
            const trainerStore = Alpine.store('trainer');
            if (trainerStore && typeof trainerStore.saveConfig === 'function') {
                console.log('[TRAINING WIZARD] Saving configuration to disk...');
                try {
                    await trainerStore.saveConfig();
                    console.log('[TRAINING WIZARD] Configuration saved successfully');
                } catch (error) {
                    console.error('[TRAINING WIZARD] Config save failed:', error);
                    window.showToast && window.showToast('Failed to save configuration', 'error');
                    return; // Don't continue if config save failed
                }
            } else {
                console.warn('[TRAINING WIZARD] saveConfig method not available');
            }

            // If we have a pending dataset plan, save it now
            if (this.pendingDatasetPlan && this.pendingDatasetPlan.length > 0) {
                console.log('[TRAINING WIZARD] Committing pending dataset plan:', this.pendingDatasetPlan);

                try {
                    const response = await fetch('/api/datasets/plan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            datasets: this.pendingDatasetPlan,
                            createBackup: false
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        let errorMessage = 'Failed to save dataset plan';

                        if (error.detail?.message) {
                            errorMessage = error.detail.message;
                        }

                        console.error('[TRAINING WIZARD] Dataset plan save failed:', errorMessage);
                        window.showToast && window.showToast(`Dataset save failed: ${errorMessage}`, 'error');
                        return; // Don't close wizard if dataset save failed
                    }

                    console.log('[TRAINING WIZARD] Dataset plan saved successfully');

                    // Refresh the dataloader section to show the new datasets
                    const trainer = Alpine.store('trainer');
                    if (trainer && typeof trainer.loadDatasets === 'function') {
                        await trainer.loadDatasets();
                    }
                } catch (error) {
                    console.error('[TRAINING WIZARD] Error saving dataset plan:', error);
                    window.showToast && window.showToast('Failed to save datasets', 'error');
                    return; // Don't close wizard if there was an error
                }
            }

            // Close wizard
            this.wizardOpen = false;

            // Clear pending dataset plan
            this.pendingDatasetPlan = null;

            // Show success message
            window.showToast && window.showToast('Wizard complete! Configuration and datasets saved.', 'success');
        },

        // Answer management
        async selectAnswer(key, value) {
            this.answers[key] = value;
            console.log(`[TRAINING WIZARD] Answer set: ${key} = ${value}`);

            // If model family changed, fetch flavours and reset selection
            if (key === 'model_family') {
                this.modelFlavours = [];
                this.answers.model_flavour = null;
                await this.loadModelFlavours(value);
            }
        },

        async loadLoggingProviders() {
            console.log('[TRAINING WIZARD] Loading logging providers from field registry');

            try {
                const response = await fetch('/api/fields/field/report_to');
                if (!response.ok) {
                    console.warn('[TRAINING WIZARD] Could not load logging providers');
                    // Fallback to hardcoded options
                    this.loggingProviders = [
                        { value: 'none', label: 'None' },
                        { value: 'wandb', label: 'Weights & Biases' },
                        { value: 'tensorboard', label: 'TensorBoard' }
                    ];
                    return;
                }

                const data = await response.json();
                console.log('[TRAINING WIZARD] Loaded field data:', data);

                // Extract choices from field metadata
                if (data.choices && Array.isArray(data.choices)) {
                    this.loggingProviders = data.choices.map(choice => ({
                        value: choice.value,
                        label: choice.label
                    }));
                    console.log('[TRAINING WIZARD] Logging providers:', this.loggingProviders);
                } else {
                    // Fallback
                    this.loggingProviders = [
                        { value: 'none', label: 'None' },
                        { value: 'wandb', label: 'Weights & Biases' },
                        { value: 'tensorboard', label: 'TensorBoard' }
                    ];
                }
            } catch (error) {
                console.error('[TRAINING WIZARD] Error loading logging providers:', error);
                // Fallback to hardcoded options
                this.loggingProviders = [
                    { value: 'none', label: 'None' },
                    { value: 'wandb', label: 'Weights & Biases' },
                    { value: 'tensorboard', label: 'TensorBoard' }
                ];
            }
        },

        async loadModelFlavours(modelFamily) {
            if (!modelFamily) {
                this.modelFlavours = [];
                return;
            }

            console.log(`[TRAINING WIZARD] Loading flavours for ${modelFamily}`);

            try {
                // Fetch model details to get default flavour
                const detailsResponse = await fetch(`/api/models/${modelFamily}`);
                let defaultFlavour = null;

                if (detailsResponse.ok) {
                    const details = await detailsResponse.json();
                    defaultFlavour = details.default_flavour;
                    console.log(`[TRAINING WIZARD] Default flavour: ${defaultFlavour}`);
                }

                // Fetch available flavours
                const response = await fetch(`/api/models/${modelFamily}/flavours`);
                if (!response.ok) {
                    console.warn(`[TRAINING WIZARD] No flavours available for ${modelFamily}`);
                    this.modelFlavours = [];
                    return;
                }
                const data = await response.json();
                this.modelFlavours = data.flavours || [];
                console.log(`[TRAINING WIZARD] Loaded ${this.modelFlavours.length} flavours:`, this.modelFlavours);

                // Auto-select default flavour or the only flavour
                if (this.modelFlavours.length === 1) {
                    // Only one flavour, select it automatically
                    this.answers.model_flavour = this.modelFlavours[0];
                    console.log(`[TRAINING WIZARD] Auto-selected only flavour: ${this.modelFlavours[0]}`);
                } else if (defaultFlavour && this.modelFlavours.includes(defaultFlavour)) {
                    // Multiple flavours but we have a default, select it
                    this.answers.model_flavour = defaultFlavour;
                    console.log(`[TRAINING WIZARD] Auto-selected default flavour: ${defaultFlavour}`);
                }
            } catch (error) {
                console.error('[TRAINING WIZARD] Error loading flavours:', error);
                this.modelFlavours = [];
            }
        },

        applyAnswersToForm() {
            const trainerStore = Alpine.store('trainer');
            if (!trainerStore || !trainerStore.configValues) {
                console.warn('[TRAINING WIZARD] Trainer store not available');
                return;
            }

            // Apply each answer to config
            Object.entries(this.answers).forEach(([key, value]) => {
                if (value !== null && value !== undefined && key !== 'enable_validations') {
                    trainerStore.configValues[key] = value;
                    trainerStore.configValues[`--${key}`] = value;
                }
            });

            // Special handling for validations
            if (this.answers.enable_validations === false) {
                trainerStore.configValues.disable_validations = true;
                trainerStore.configValues['--disable_validations'] = true;
            } else {
                trainerStore.configValues.disable_validations = false;
                trainerStore.configValues['--disable_validations'] = false;
            }

            // Mark form as dirty
            if (trainerStore.markFormDirty) {
                trainerStore.markFormDirty();
            }

            console.log('[TRAINING WIZARD] Answers applied to form');
        },

        // Field navigation using existing search mechanism
        wizardNavigateToField(fieldName) {
            console.log(`[TRAINING WIZARD] Navigating to field: ${fieldName}`);

            // Use the existing field navigation from search
            const fieldResult = {
                name: fieldName,
                title: fieldName,
                context: {
                    tab: this.getTabForField(fieldName)
                }
            };

            // Find the search component instance and call selectField
            const searchEl = document.querySelector('[x-data*="searchComponent"]');
            if (searchEl && searchEl.__x && searchEl.__x.$data) {
                searchEl.__x.$data.selectField(fieldResult);
            } else {
                // Fallback: just switch to the tab
                this.wizardNavigateToTab(this.getTabForField(fieldName));
            }
        },

        wizardNavigateToFieldMulti(fieldNames) {
            // Navigate to first field
            if (fieldNames && fieldNames.length > 0) {
                this.wizardNavigateToField(fieldNames[0]);
            }
        },

        wizardNavigateToTab(tabName) {
            console.log(`[TRAINING WIZARD] Navigating to tab: ${tabName}`);

            const trainerStore = Alpine.store('trainer');
            if (trainerStore && trainerStore.switchTab) {
                trainerStore.switchTab(tabName);
            }
        },

        getTabForField(fieldName) {
            // Map fields to their tabs
            const fieldToTab = {
                'model_family': 'basic',
                'model_flavour': 'basic',
                'model_type': 'basic',
                'push_to_hub': 'publishing',
                'push_checkpoints_to_hub': 'publishing',
                'checkpointing_steps': 'checkpoints',
                'validation_steps': 'validations',
                'validation_prompt': 'validations',
                'disable_validations': 'validations',
                'report_to': 'publishing',
                'learning_rate': 'training',
                'optimizer': 'training'
            };

            return fieldToTab[fieldName] || 'basic';
        },

        // Model-specific helpers
        needsModelFlavour() {
            // Check if the current model family has flavours loaded
            return this.modelFlavours && this.modelFlavours.length > 0;
        },

        getModelFlavourOptions() {
            // Return hardcoded options for known model families with descriptions
            if (this.answers.model_family === 'flux') {
                return FLUX_FLAVOURS;
            }
            if (this.answers.model_family === 'sd3') {
                return SD3_FLAVOURS;
            }

            // For other families, use the fetched flavours
            if (this.modelFlavours && this.modelFlavours.length > 0) {
                return this.modelFlavours.map(f => ({
                    value: f,
                    label: f,
                    description: `${f} variant`
                }));
            }

            return [];
        },

        getHelpText(key, value) {
            // Provide contextual help based on quickstart docs
            if (key === 'model_family') {
                if (value === 'flux') {
                    return 'FLUX requires 16-30GB VRAM depending on quantization. Supports LoRA and full fine-tuning.';
                }
                if (value === 'wan') {
                    return 'Wan 2.1 is for video generation. Requires 12GB+ VRAM for 1.3B model.';
                }
                if (value === 'sd3') {
                    return 'Stable Diffusion 3 with improved architecture. Multiple model sizes available.';
                }
                if (value === 'sdxl') {
                    return 'Stable Diffusion XL 1.0 - high quality 1024px images.';
                }
            }
            return '';
        },

        getRecommendedSettings() {
            // Return recommended settings based on model
            if (this.answers.model_family === 'flux' && this.answers.model_type === 'lora') {
                return 'Learning rate: 1e-4, Optimizer: adamw_bf16';
            }
            if (this.answers.model_family === 'wan') {
                return 'Learning rate: 5e-5, Optimizer: optimi-lion';
            }
            return 'Learning rate: 1e-4, Optimizer: adamw_bf16';
        },

        // Dataset wizard integration
        async openDatasetWizard(createStandalone = false) {
            console.log('[TRAINING WIZARD] openDatasetWizard called, createStandalone:', createStandalone);

            // Close training wizard
            this.wizardOpen = false;

            // First, navigate to the datasets tab to ensure it's loaded
            const trainerStore = Alpine.store('trainer');
            console.log('[TRAINING WIZARD] trainerStore found:', !!trainerStore);

            if (trainerStore && typeof trainerStore.activateTab === 'function') {
                console.log('[TRAINING WIZARD] Activating datasets tab...');
                await trainerStore.activateTab('datasets');
                console.log('[TRAINING WIZARD] Datasets tab activated');
            }

            // Wait for the dataset wizard component to be initialized by Alpine
            const maxAttempts = 20; // 2 seconds max
            let attempts = 0;
            let datasetWizardEl = null;

            console.log('[TRAINING WIZARD] Starting to poll for dataset wizard component...');

            while (attempts < maxAttempts) {
                datasetWizardEl = document.querySelector('[x-data*="datasetWizardComponent"]');

                // Try to get Alpine component using Alpine's $data utility
                let alpineData = null;
                if (datasetWizardEl) {
                    try {
                        alpineData = Alpine.$data(datasetWizardEl);
                    } catch (e) {
                        // Alpine not initialized yet
                    }
                }

                console.log(`[TRAINING WIZARD] Attempt ${attempts + 1}: element found:`, !!datasetWizardEl, ', has Alpine data:', !!alpineData, ', has openWizard method:', !!(alpineData && typeof alpineData.openWizard === 'function'));

                if (datasetWizardEl && alpineData && typeof alpineData.openWizard === 'function') {
                    // Found and initialized!
                    console.log('[TRAINING WIZARD] Dataset wizard component found and initialized!');
                    break;
                }

                // Wait 100ms before trying again
                await new Promise(resolve => setTimeout(resolve, 100));
                attempts++;
            }

            // Now try to open the dataset wizard
            datasetWizardEl = document.querySelector('[x-data*="datasetWizardComponent"]');
            const datasetWizard = datasetWizardEl ? Alpine.$data(datasetWizardEl) : null;

            if (datasetWizard && typeof datasetWizard.openWizard === 'function') {
                // Set standalone config flag if needed
                if (createStandalone) {
                    datasetWizard.createNewConfig = true;
                }

                // Enable deferred commit mode - dataset wizard will prepare plan but not save
                datasetWizard.deferCommit = true;
                console.log('[TRAINING WIZARD] Enabled defer commit mode in dataset wizard');

                // Save current step index before opening dataset wizard
                const savedStepIndex = this.currentStepIndex;
                const savedAnswers = { ...this.answers };

                // Set up a watcher to detect when the dataset wizard closes
                const checkWizardClosed = setInterval(() => {
                    if (!datasetWizard.wizardOpen) {
                        clearInterval(checkWizardClosed);
                        console.log('[TRAINING WIZARD] Dataset wizard closed, reopening training wizard');

                        // Capture the dataset plan from the dataset wizard
                        if (datasetWizard.pendingDatasetPlan) {
                            this.pendingDatasetPlan = datasetWizard.pendingDatasetPlan;
                            console.log('[TRAINING WIZARD] Captured dataset plan:', this.pendingDatasetPlan);
                        }

                        // Mark dataset as configured
                        this.datasetConfigured = true;

                        // Reopen training wizard after a short delay, preserving state
                        setTimeout(() => {
                            // Restore answers (in case they were cleared)
                            Object.assign(this.answers, savedAnswers);
                            // Restore step index
                            this.currentStepIndex = savedStepIndex;
                            // Reopen wizard
                            this.wizardOpen = true;

                            console.log('[TRAINING WIZARD] Reopened at step', this.currentStepIndex, 'with preserved answers');
                        }, 300);
                    }
                }, 200);

                // Open dataset wizard
                console.log('[TRAINING WIZARD] Opening dataset wizard');
                datasetWizard.openWizard();
            } else {
                console.warn('[TRAINING WIZARD] Dataset wizard component not found after', attempts, 'attempts');
                console.warn('[TRAINING WIZARD] datasetWizardEl:', datasetWizardEl);
                console.warn('[TRAINING WIZARD] datasetWizard:', datasetWizard);
                window.showToast && window.showToast('Dataset wizard not available', 'warning');
            }
        }
    };
}

// Export to window
window.trainingWizardComponent = trainingWizardComponent;

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { trainingWizardComponent };
}

console.log('[TRAINING WIZARD] Component loaded');
