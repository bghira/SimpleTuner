/**
 * Training Configuration Wizard Component
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
        optimizerChoices: [],
        advancedMode: 'preset',
        selectedPreset: null,
        trackerFieldDefaults: {
            project: {
                placeholder: 'simpletuner',
                defaultValue: 'simpletuner'
            },
            run: {
                placeholder: 'simpletuner-testing',
                defaultValue: 'simpletuner-testing'
            }
        },
        trackerDefaultsLoaded: false,
        loraRankOptions: [1, 2, 4, 8, 16, 32, 64, 128, 256],
        quantizationFields: {
            base: null,
            textEncoders: [],
            quantizeVia: null
        },
        quantizationLoading: false,
        modelDetailsCache: {},
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
            optimizer: null,
            validation_resolution: '1024x1024',
            validation_num_inference_steps: 20,
            train_batch_size: 2,
            tracker_project_name: '',
            tracker_run_name: '',
            lora_rank: 16,
            base_model_precision: 'int8-quanto',
            text_encoder_1_precision: 'no_change',
            text_encoder_2_precision: 'no_change',
            text_encoder_3_precision: 'no_change',
            text_encoder_4_precision: 'no_change',
            quantize_via: 'accelerator',
            hub_model_id: '',
            model_card_note: '',
            model_card_safe_for_work: false,
            model_card_private: false,
            createNewDataset: false  // Track whether user chose to create new dataset
        },

        // Configuration
        modelFamilyOptions: [],

        // Step definitions
        steps: [
            {
                id: 'model-family',
                label: 'Model',
                title: 'Training Configuration Wizard - Step 1: Model Family',
                required: true,
                validate: function() { return this.answers.model_family !== null; }
            },
            {
                id: 'model-flavour',
                label: 'Variant',
                title: 'Training Configuration Wizard - Step 2: Model Variant',
                required: false,
                condition: function() { return this.needsModelFlavour(); },
                validate: function() { return !this.needsModelFlavour() || this.answers.model_flavour !== null; }
            },
            {
                id: 'training-type',
                label: 'Type',
                title: 'Training Configuration Wizard - Step 3: Training Type',
                required: true,
                validate: function() { return this.answers.model_type !== null; }
            },
            {
                id: 'publishing',
                label: 'Publishing',
                title: 'Training Configuration Wizard - Step 4: Publishing',
                required: false,
                validate: function() {
                    if (this.answers.push_to_hub) {
                        return typeof this.answers.hub_model_id === 'string' && this.answers.hub_model_id.trim().length > 0;
                    }
                    return true;
                }
            },
            {
                id: 'checkpoints',
                label: 'Checkpoints',
                title: 'Training Configuration Wizard - Step 5: Checkpoints',
                required: true,
                validate: function() { return this.answers.checkpointing_steps > 0; }
            },
            {
                id: 'validations-enable',
                label: 'Validations',
                title: 'Training Configuration Wizard - Step 6: Validations',
                required: true,
                validate: function() {
                    if (this.answers.enable_validations === null) {
                        return false;
                    }
                    if (this.answers.enable_validations) {
                        const stepsValid = typeof this.answers.validation_steps === 'number' && this.answers.validation_steps > 0;
                        const promptValid = this.answers.validation_prompt && this.answers.validation_prompt.trim().length > 0;
                        const resolutionValid = typeof this.answers.validation_resolution === 'string' && this.answers.validation_resolution.trim().length > 0;
                        const inferenceValid =
                            typeof this.answers.validation_num_inference_steps === 'number' &&
                            this.answers.validation_num_inference_steps > 0;
                        return stepsValid && promptValid && resolutionValid && inferenceValid;
                    }
                    return true;
                }
            },
            {
                id: 'logging',
                label: 'Logging',
                title: 'Training Configuration Wizard - Step 7: Logging',
                required: false,
                validate: function() { return true; }
            },
            {
                id: 'dataset',
                label: 'Dataset',
                title: 'Training Configuration Wizard - Step 8: Dataset',
                required: true,
                validate: function() { return this.hasExistingDataset || this.datasetConfigured; }
            },
            {
                id: 'advanced',
                label: 'Advanced',
                title: 'Training Configuration Wizard - Step 9: Advanced Settings',
                required: false,
                validate: function() {
                    if (this.advancedMode === 'manual') {
                        const lrValid = typeof this.answers.learning_rate === 'number' && this.answers.learning_rate > 0;
                        const optimizerValid = Boolean(this.answers.optimizer);
                        const batchValid = typeof this.answers.train_batch_size === 'number' && this.answers.train_batch_size > 0;
                        return lrValid && optimizerValid && batchValid;
                    }
                    return true;
                }
            },
            {
                id: 'review',
                label: 'Review',
                title: 'Training Configuration Wizard - Review Configuration',
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

        get validationAlignmentDetails() {
            const checkpointRaw = this.answers.checkpointing_steps;
            const validationRaw = this.answers.validation_steps;
            const checkpointingSteps = typeof checkpointRaw === 'number' ? checkpointRaw : parseInt(checkpointRaw);
            const validationSteps = typeof validationRaw === 'number' ? validationRaw : parseInt(validationRaw);

            const checkpointsValid = Number.isFinite(checkpointingSteps) && checkpointingSteps > 0;
            const validationsValid = Number.isFinite(validationSteps) && validationSteps > 0;

            if (!checkpointsValid || !validationsValid) {
                return {
                    hasData: false,
                    checkpointingSteps: checkpointsValid ? checkpointingSteps : null,
                    validationSteps: validationsValid ? validationSteps : null,
                    aligns: false,
                    example: null
                };
            }

            const aligns =
                checkpointingSteps === validationSteps ||
                checkpointingSteps % validationSteps === 0 ||
                validationSteps % checkpointingSteps === 0;

            let example = validationSteps;
            if (!aligns) {
                if (checkpointingSteps > validationSteps) {
                    example = checkpointingSteps;
                } else if (validationSteps > checkpointingSteps) {
                    example = checkpointingSteps;
                } else {
                    example = checkpointingSteps;
                }
            }

            return {
                hasData: true,
                checkpointingSteps,
                validationSteps,
                aligns,
                example
            };
        },

        get loggingEnabled() {
            return this.answers.report_to && this.answers.report_to !== 'none';
        },

        get trackerProjectDisplay() {
            const value = (this.answers.tracker_project_name || '').trim();
            if (value.length > 0) {
                return value;
            }
            return this.trackerFieldDefaults.project.defaultValue || 'simpletuner';
        },

        get trackerRunDisplay() {
            const value = (this.answers.tracker_run_name || '').trim();
            if (value.length > 0) {
                return value;
            }
            return 'Auto-generated';
        },

        get loggingSummary() {
            if (!this.loggingEnabled) {
                return 'Disabled';
            }
            return `${this.answers.report_to || 'none'} (Project: ${this.trackerProjectDisplay}, Run: ${this.trackerRunDisplay})`;
        },

        // Initialization
        async init() {
            console.log('[TRAINING WIZARD] Initializing...');

            try {
                // Load current configuration to pre-populate answers
                await this.loadCurrentConfig();

                // Check if dataset exists
                await this.checkDataset();

                if (this.answers.model_type === 'lora') {
                    await this.loadQuantizationOptions();
                }

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
                    if (config.validation_resolution !== undefined && config.validation_resolution !== null) {
                        this.answers.validation_resolution = String(config.validation_resolution);
                    }
                    if (config.validation_num_inference_steps !== undefined && config.validation_num_inference_steps !== null) {
                        const parsedSteps = parseInt(config.validation_num_inference_steps);
                        if (!Number.isNaN(parsedSteps)) {
                            this.answers.validation_num_inference_steps = parsedSteps;
                        }
                    }
                    if (config.report_to) this.answers.report_to = config.report_to;
                    if (config.disable_validations !== undefined) {
                        this.answers.enable_validations = !(config.disable_validations === true || config.disable_validations === 'true');
                    }
                    if (config.learning_rate !== undefined && config.learning_rate !== null) {
                        const parsedLr = parseFloat(config.learning_rate);
                        if (!Number.isNaN(parsedLr)) {
                            this.answers.learning_rate = parsedLr;
                        }
                    }
                    if (config.optimizer) {
                        this.answers.optimizer = config.optimizer;
                    }
                    if (config.train_batch_size) {
                        const parsedBatch = parseInt(config.train_batch_size);
                        if (!Number.isNaN(parsedBatch)) {
                            this.answers.train_batch_size = parsedBatch;
                        }
                    }
                    const projectName = config.tracker_project_name || config['--tracker_project_name'];
                    if (projectName) {
                        this.answers.tracker_project_name = projectName;
                    }
                    const runName = config.tracker_run_name || config['--tracker_run_name'];
                    if (runName) {
                        this.answers.tracker_run_name = runName;
                    }
                    const loraRankValue = config.lora_rank || config['--lora_rank'];
                    if (loraRankValue) {
                        const parsedRank = parseInt(loraRankValue);
                        if (!Number.isNaN(parsedRank)) {
                            this.answers.lora_rank = parsedRank;
                        }
                    }
                    if (config.base_model_precision) {
                        this.answers.base_model_precision = config.base_model_precision;
                    }
                    for (let i = 1; i <= 4; i++) {
                        const key = `text_encoder_${i}_precision`;
                        const value = config[key] || config[`--${key}`];
                        if (value) {
                            this.answers[key] = value;
                        }
                    }
                    if (config.quantize_via) {
                        this.answers.quantize_via = config.quantize_via;
                    }
                    const hubModelId = config.hub_model_id || config['--hub_model_id'];
                    if (hubModelId) {
                        this.answers.hub_model_id = hubModelId;
                    }
                    const modelCardNote = config.model_card_note || config['--model_card_note'];
                    if (modelCardNote) {
                        this.answers.model_card_note = modelCardNote;
                    }
                    const safeForWork = config.model_card_safe_for_work ?? config['--model_card_safe_for_work'];
                    if (safeForWork !== undefined) {
                        this.answers.model_card_safe_for_work = safeForWork === true || safeForWork === 'true' || safeForWork === '1';
                    }
                    const modelCardPrivate = config.model_card_private ?? config['--model_card_private'];
                    if (modelCardPrivate !== undefined) {
                        this.answers.model_card_private = modelCardPrivate === true || modelCardPrivate === 'true' || modelCardPrivate === '1';
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

            if (this.optimizerChoices.length === 0) {
                await this.loadOptimizerChoices();
            }

            if (!this.trackerDefaultsLoaded) {
                await this.loadTrackerDefaults();
            }

            this.selectedPreset = null;

            if (
                typeof this.answers.learning_rate === 'number' &&
                this.answers.learning_rate > 0 &&
                this.answers.optimizer &&
                typeof this.answers.train_batch_size === 'number' &&
                this.answers.train_batch_size > 0
            ) {
                this.advancedMode = 'manual';
            } else {
                this.advancedMode = 'preset';
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

            // Debug: Log activeEnvironmentConfig to confirm values are set
            const trainerStore = Alpine.store('trainer');
            if (trainerStore) {
                console.log('[TRAINING WIZARD] activeEnvironmentConfig after applying answers:',
                    JSON.stringify({
                        model_family: trainerStore.activeEnvironmentConfig?.model_family,
                        model_flavour: trainerStore.activeEnvironmentConfig?.model_flavour,
                        model_type: trainerStore.activeEnvironmentConfig?.model_type,
                        checkpointing_steps: trainerStore.activeEnvironmentConfig?.checkpointing_steps,
                        push_to_hub: trainerStore.activeEnvironmentConfig?.push_to_hub
                    }, null, 2)
                );
            }

            // Save the config to disk (bypass the dialog, use direct save)
            if (trainerStore && typeof trainerStore.doSaveConfig === 'function') {
                console.log('[TRAINING WIZARD] Saving configuration to disk...');
                try {
                    // Use doSaveConfig to bypass the dialog
                    const autoPreserve = trainerStore.autoPreserveEnabled ? trainerStore.autoPreserveEnabled() : false;
                    await trainerStore.doSaveConfig({
                        createBackup: false,
                        preserveDefaults: autoPreserve
                    });
                    console.log('[TRAINING WIZARD] Configuration saved successfully');
                } catch (error) {
                    console.error('[TRAINING WIZARD] Config save failed:', error);
                    window.showToast && window.showToast('Failed to save configuration', 'error');
                    return; // Don't continue if config save failed
                }
            } else {
                console.warn('[TRAINING WIZARD] doSaveConfig method not available');
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
                if (this.answers.model_type === 'lora') {
                    await this.loadQuantizationOptions();
                }
            }

            if (key === 'model_type') {
                if (value === 'lora') {
                    if (!this.answers.lora_rank || this.answers.lora_rank <= 0) {
                        this.answers.lora_rank = 16;
                    }
                    if (!this.answers.base_model_precision) {
                        this.answers.base_model_precision = 'int8-quanto';
                    }
                    await this.loadQuantizationOptions();
                } else {
                    this.answers.base_model_precision = 'no_change';
                    this.answers.quantize_via = 'accelerator';
                    for (let i = 1; i <= 4; i++) {
                        this.answers[`text_encoder_${i}_precision`] = 'no_change';
                    }
                    await this.loadQuantizationOptions();
                }
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

        async loadTrackerDefaults() {
            console.log('[TRAINING WIZARD] Loading tracking field defaults');

            try {
                const [projectResponse, runResponse] = await Promise.all([
                    fetch('/api/fields/field/tracker_project_name'),
                    fetch('/api/fields/field/tracker_run_name')
                ]);

                if (projectResponse.ok) {
                    const projectData = await projectResponse.json();
                    this.trackerFieldDefaults.project.placeholder =
                        projectData.placeholder || projectData.default_value || this.trackerFieldDefaults.project.placeholder;
                    this.trackerFieldDefaults.project.defaultValue =
                        projectData.default_value || this.trackerFieldDefaults.project.defaultValue;
                }

                if (runResponse.ok) {
                    const runData = await runResponse.json();
                    this.trackerFieldDefaults.run.placeholder =
                        runData.placeholder || runData.default_value || this.trackerFieldDefaults.run.placeholder;
                    this.trackerFieldDefaults.run.defaultValue =
                        runData.default_value || this.trackerFieldDefaults.run.defaultValue;
                }
            } catch (error) {
                console.warn('[TRAINING WIZARD] Unable to load tracking defaults:', error);
            } finally {
                this.trackerDefaultsLoaded = true;
            }
        },

        async fetchModelDetails(modelFamily) {
            if (!modelFamily) {
                return null;
            }
            if (this.modelDetailsCache[modelFamily]) {
                return this.modelDetailsCache[modelFamily];
            }

            try {
                const response = await fetch(`/api/models/${modelFamily}`);
                if (!response.ok) {
                    throw new Error(`Failed to load model details for ${modelFamily}`);
                }
                const data = await response.json();
                this.modelDetailsCache[modelFamily] = data;
                return data;
            } catch (error) {
                console.warn('[TRAINING WIZARD] Unable to load model details:', error);
                return null;
            }
        },

        async loadQuantizationOptions() {
            if (this.answers.model_type !== 'lora') {
                this.quantizationFields = { base: null, textEncoders: [], quantizeVia: null };
                return;
            }

            if (this.quantizationLoading) {
                return;
            }

            this.quantizationLoading = true;

            try {
                const params = new URLSearchParams({
                    include_advanced: 'true',
                    importance_level: 'experimental',
                    model_type: 'lora'
                });
                if (this.answers.model_family) {
                    params.set('model_family', this.answers.model_family);
                }

                const [fieldsResponse, modelDetails] = await Promise.all([
                    fetch(`/api/fields/tabs/model?${params.toString()}`),
                    this.fetchModelDetails(this.answers.model_family)
                ]);

                if (!fieldsResponse.ok) {
                    throw new Error(`Failed to load quantization fields: ${fieldsResponse.statusText}`);
                }

                const fieldData = await fieldsResponse.json();
                const sections = fieldData.fields || {};
                const flattened = Object.values(sections).flat();

                const findField = (name) => flattened.find(field => field.name === name);

                const baseField = findField('base_model_precision');
                const quantizeViaField = findField('quantize_via');
                const textEncoderFields = [];
                for (let i = 1; i <= 4; i++) {
                    const field = findField(`text_encoder_${i}_precision`);
                    if (field) {
                        textEncoderFields.push(field);
                    }
                }

                const supportsTextEncoder = Boolean(modelDetails?.attributes?.supports_text_encoder_training);
                const encoderConfigRaw = modelDetails?.attributes?.text_encoder_configuration;
                const encoderConfig = (encoderConfigRaw && typeof encoderConfigRaw === 'object') ? encoderConfigRaw : null;
                let encoderEntries = [];
                if (encoderConfig) {
                    encoderEntries = Object.keys(encoderConfig).map(key => ({
                        key,
                        name: encoderConfig[key]?.name || ''
                    }));
                }
                let resolvedEncoderCount = 0;
                if (encoderEntries.length > 0) {
                    resolvedEncoderCount = encoderEntries.length;
                } else if (supportsTextEncoder && textEncoderFields.length > 0) {
                    resolvedEncoderCount = 1;
                }

                const formatChoices = (choices) => {
                    if (!Array.isArray(choices)) {
                        return [];
                    }
                    return choices.map(choice => ({
                        value: choice.value,
                        label: choice.label || choice.value
                    }));
                };

                const baseInfo = baseField ? {
                    id: baseField.name,
                    label: baseField.ui_label || 'Base Model Precision',
                    choices: formatChoices(baseField.choices),
                    helpText: baseField.help_text || '',
                    tooltip: baseField.tooltip || '',
                    defaultValue: baseField.default_value
                } : null;

                const textEncoderInfo = textEncoderFields
                    .map((field, index) => {
                        const fieldId = field.name;
                        const currentValue = this.answers[fieldId];
                        const hasCustomValue = currentValue && currentValue !== 'no_change';

                        const choices = formatChoices(field.choices);
                        const entry = encoderEntries[index] || null;
                        const rawLabel = (field.ui_label || '').trim();
                        const entryName = entry && entry.name ? entry.name.trim() : '';
                        const isDefaultLabel = /^text encoder \d+ precision$/i.test(rawLabel);
                        const displayName = entryName || (!isDefaultLabel ? rawLabel : '');
                        const withinSupportedRange = index < resolvedEncoderCount;
                        const shouldShow =
                            withinSupportedRange ||
                            displayName.length > 0 ||
                            (!isDefaultLabel && rawLabel.length > 0) ||
                            hasCustomValue;
                        if (!shouldShow) {
                            return null;
                        }

                        const baseLabel = (() => {
                            if (displayName) {
                                return displayName.toLowerCase().includes('precision')
                                    ? displayName
                                    : `${displayName} Precision`;
                            }
                            if (rawLabel) {
                                return rawLabel;
                            }
                            return `Text Encoder ${index + 1} Precision`;
                        })();

                        const shortLabel = displayName || rawLabel || `Encoder ${index + 1}`;
                        const helpText = displayName
                            ? `Set quantisation precision for the ${displayName} text encoder.`
                            : (field.help_text || '');

                        return {
                            id: fieldId,
                            label: baseLabel,
                            shortLabel,
                            choices,
                            helpText: helpText || '',
                            tooltip: field.tooltip || ''
                        };
                    })
                    .filter(Boolean);

                const quantizeViaInfo = quantizeViaField ? {
                    id: quantizeViaField.name,
                    label: quantizeViaField.ui_label || 'Quantization Device',
                    choices: formatChoices(quantizeViaField.choices),
                    helpText: quantizeViaField.help_text || '',
                    tooltip: quantizeViaField.tooltip || '',
                    defaultValue: quantizeViaField.default_value
                } : null;

                this.quantizationFields = {
                    base: baseInfo,
                    textEncoders: textEncoderInfo,
                    quantizeVia: quantizeViaInfo
                };

                if (!baseInfo) {
                    this.answers.base_model_precision = 'no_change';
                }

                if (this.answers.model_type === 'lora' && baseInfo) {
                    const choiceValues = baseInfo.choices.map(choice => choice.value);
                    if (!this.answers.base_model_precision || !choiceValues.includes(this.answers.base_model_precision)) {
                        if (choiceValues.includes('int8-quanto')) {
                            this.answers.base_model_precision = 'int8-quanto';
                        } else if (baseInfo.defaultValue && choiceValues.includes(baseInfo.defaultValue)) {
                            this.answers.base_model_precision = baseInfo.defaultValue;
                        } else if (choiceValues.length > 0) {
                            this.answers.base_model_precision = choiceValues[0];
                        } else {
                            this.answers.base_model_precision = 'no_change';
                        }
                    }
                }

                if (this.answers.model_type === 'lora' && quantizeViaInfo) {
                    const choiceValues = quantizeViaInfo.choices.map(choice => choice.value);
                    if (!choiceValues.includes(this.answers.quantize_via)) {
                        if (choiceValues.includes('accelerator')) {
                            this.answers.quantize_via = 'accelerator';
                        } else if (quantizeViaInfo.defaultValue && choiceValues.includes(quantizeViaInfo.defaultValue)) {
                            this.answers.quantize_via = quantizeViaInfo.defaultValue;
                        } else if (choiceValues.length > 0) {
                            this.answers.quantize_via = choiceValues[0];
                        }
                    }
                }

                // Ensure text encoder defaults exist
                if (textEncoderInfo.length === 0) {
                    for (let i = 1; i <= 4; i++) {
                        this.answers[`text_encoder_${i}_precision`] = 'no_change';
                    }
                } else {
                    textEncoderInfo.forEach(field => {
                        const key = field.id;
                        const choiceValues = field.choices.map(choice => choice.value);
                        if (!choiceValues.includes(this.answers[key])) {
                            this.answers[key] = 'no_change';
                        }
                    });
                }
            } catch (error) {
                console.error('[TRAINING WIZARD] Failed to load quantization options:', error);
                this.quantizationFields = { base: null, textEncoders: [], quantizeVia: null };
            } finally {
                this.quantizationLoading = false;
            }
        },

        async loadOptimizerChoices() {
            console.log('[TRAINING WIZARD] Loading optimizer choices from field registry');

            try {
                const response = await fetch('/api/fields/field/optimizer');
                if (!response.ok) {
                    throw new Error(`Failed to load optimizer field: ${response.statusText}`);
                }

                const data = await response.json();
                if (Array.isArray(data.choices) && data.choices.length > 0) {
                    this.optimizerChoices = data.choices.map(choice => ({
                        value: choice.value,
                        label: choice.label || choice.value
                    }));
                    return;
                }

                console.warn('[TRAINING WIZARD] Optimizer field returned no choices, using fallback list');
                this.optimizerChoices = this.getOptimizerFallbackChoices();
            } catch (error) {
                console.error('[TRAINING WIZARD] Error loading optimizer choices:', error);
                this.optimizerChoices = this.getOptimizerFallbackChoices();
            }
        },

        getOptimizerFallbackChoices() {
            return [
                { value: 'adamw_bf16', label: 'adamw_bf16' },
                { value: 'adamw_schedulefree', label: 'adamw_schedulefree' },
                { value: 'optimi-lion', label: 'optimi-lion' }
            ];
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
            if (!trainerStore) {
                console.warn('[TRAINING WIZARD] Trainer store not available');
                return;
            }

            console.log('[TRAINING WIZARD] Applying answers to trainer store...', this.answers);

            // Update config values, formValueStore, AND activeEnvironmentConfig
            Object.entries(this.answers).forEach(([key, value]) => {
                if (key === 'lora_rank' && this.answers.model_type !== 'lora') {
                    return;
                }
                if (value !== null && value !== undefined && key !== 'enable_validations' && key !== 'createNewDataset') {
                    const fieldName = `--${key}`;

                    // 1. Update configValues
                    if (trainerStore.configValues) {
                        trainerStore.configValues[key] = value;
                        trainerStore.configValues[fieldName] = value;
                    }

                    // 2. Update formValueStore (used by ensureCompleteFormData)
                    if (trainerStore.formValueStore) {
                        trainerStore.formValueStore[fieldName] = {
                            value: value,
                            kind: typeof value === 'boolean' ? 'checkbox' : 'text'
                        };
                    }

                    // 3. Update activeEnvironmentConfig (this is what appendConfigValuesToFormData uses!)
                    if (trainerStore.activeEnvironmentConfig) {
                        trainerStore.activeEnvironmentConfig[key] = value;
                        trainerStore.activeEnvironmentConfig[fieldName] = value;
                    }

                    console.log(`[TRAINING WIZARD] Updated ${fieldName} = ${value}`);
                }
            });

            // Special handling for validations
            const disableValidations = this.answers.enable_validations === false;
            const fieldName = '--disable_validations';

            if (trainerStore.configValues) {
                trainerStore.configValues.disable_validations = disableValidations;
                trainerStore.configValues[fieldName] = disableValidations;
            }

            if (trainerStore.formValueStore) {
                trainerStore.formValueStore[fieldName] = {
                    value: disableValidations,
                    kind: 'checkbox'
                };
            }

            if (trainerStore.activeEnvironmentConfig) {
                trainerStore.activeEnvironmentConfig.disable_validations = disableValidations;
                trainerStore.activeEnvironmentConfig[fieldName] = disableValidations;
            }

            // Mark form as dirty
            if (trainerStore.markFormDirty) {
                trainerStore.markFormDirty();
            }

            console.log('[TRAINING WIZARD] Answers applied to all trainer store locations');
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

        activateManualAdvanced() {
            this.advancedMode = 'manual';
            this.selectedPreset = null;

            const fallback = this.getPresetDefinition('moderate') || {
                optimizer: 'adamw_bf16',
                fullLearningRate: 1e-5,
                loraLearningRate: 1e-4,
                batchSize: 2
            };

            if (!(typeof this.answers.learning_rate === 'number' && this.answers.learning_rate > 0)) {
                this.answers.learning_rate = this.getLearningRateForCurrentModel(fallback);
            }

            if (
                typeof this.answers.train_batch_size !== 'number' ||
                Number.isNaN(this.answers.train_batch_size) ||
                this.answers.train_batch_size <= 0
            ) {
                this.answers.train_batch_size = fallback.batchSize;
            }

            if (!this.answers.optimizer || !this.isOptimizerChoiceAvailable(this.answers.optimizer)) {
                this.answers.optimizer = this.resolveOptimizerSelection(fallback.optimizer);
            }
        },

        applyPreset(presetKey) {
            const preset = this.getPresetDefinition(presetKey);
            if (!preset) {
                console.warn('[TRAINING WIZARD] Unknown preset selected:', presetKey);
                return;
            }

            this.advancedMode = 'preset';
            this.selectedPreset = presetKey;

            this.answers.learning_rate = this.getLearningRateForCurrentModel(preset);
            this.answers.optimizer = this.resolveOptimizerSelection(preset.optimizer);
            this.answers.train_batch_size = preset.batchSize ?? 2;

            this.nextStep();
        },

        resolveOptimizerSelection(preferredValue) {
            if (preferredValue && this.isOptimizerChoiceAvailable(preferredValue)) {
                return preferredValue;
            }
            if (this.optimizerChoices.length > 0) {
                return this.optimizerChoices[0].value;
            }
            return preferredValue || 'adamw_bf16';
        },

        isOptimizerChoiceAvailable(value) {
            return this.optimizerChoices.some(choice => choice.value === value);
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
                'validation_resolution': 'validations',
                'validation_num_inference_steps': 'validations',
                'disable_validations': 'validations',
                'report_to': 'publishing',
                'tracker_project_name': 'basic',
                'tracker_run_name': 'basic',
                'hub_model_id': 'publishing',
                'model_card_note': 'publishing',
                'model_card_safe_for_work': 'publishing',
                'model_card_private': 'publishing',
                'learning_rate': 'training',
                'optimizer': 'training',
                'train_batch_size': 'basic',
                'lora_rank': 'model'
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

        getPresetDefinition(key) {
            const definitions = {
                aggressive: {
                    key: 'aggressive',
                    label: 'Aggressive',
                    tagline: 'Fast convergence, higher risk of instability.',
                    optimizer: 'optimi-lion',
                    fullLearningRate: 1e-5,
                    loraLearningRate: 1e-4,
                    batchSize: 2
                },
                moderate: {
                    key: 'moderate',
                    label: 'Moderate',
                    tagline: 'Balanced stability with AdamW BF16.',
                    optimizer: 'adamw_bf16',
                    fullLearningRate: 1e-5,
                    loraLearningRate: 1e-4,
                    batchSize: 2
                },
                slow_safe: {
                    key: 'slow_safe',
                    label: 'Slow & Safe',
                    tagline: 'Lower risk, more VRAM due to larger batch size.',
                    optimizer: 'adamw_bf16',
                    fullLearningRate: 1e-5,
                    loraLearningRate: 1e-4,
                    batchSize: 4
                }
            };

            return definitions[key] || null;
        },

        getAdvancedPresets() {
            const presetKeys = ['aggressive', 'moderate', 'slow_safe'];
            return presetKeys
                .map(key => {
                    const preset = this.getPresetDefinition(key);
                    if (!preset) {
                        return null;
                    }

                    const currentLr = this.formatLearningRate(this.getLearningRateForCurrentModel(preset));
                    const loraLr = this.formatLearningRate(preset.loraLearningRate);
                    const fullLr = this.formatLearningRate(preset.fullLearningRate);

                    return {
                        key: preset.key,
                        label: preset.label,
                        tagline: preset.tagline,
                        optimizer: preset.optimizer,
                        batchSize: preset.batchSize,
                        displayLearningRate: currentLr,
                        loraLearningRate: loraLr,
                        fullLearningRate: fullLr
                    };
                })
                .filter(Boolean);
        },

        getLearningRateForCurrentModel(preset) {
            if (!preset) {
                return typeof this.answers.learning_rate === 'number' && this.answers.learning_rate > 0
                    ? this.answers.learning_rate
                    : 1e-4;
            }
            const isLora = this.answers.model_type === 'lora';
            return isLora ? preset.loraLearningRate : preset.fullLearningRate;
        },

        getLoraRankIndex() {
            const current = parseInt(this.answers.lora_rank);
            const idx = this.loraRankOptions.indexOf(current);
            if (idx >= 0) {
                return idx;
            }
            const defaultIdx = this.loraRankOptions.indexOf(16);
            return defaultIdx >= 0 ? defaultIdx : 0;
        },

        setLoraRankFromIndex(index) {
            const idx = Math.min(Math.max(parseInt(index), 0), this.loraRankOptions.length - 1);
            const value = this.loraRankOptions[idx] || 16;
            this.answers.lora_rank = value;
        },

        formatLearningRate(value) {
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
        },

        getBaseQuantizationSummary() {
            if (this.answers.model_type !== 'lora') {
                return 'N/A';
            }
            if (!this.quantizationFields.base) {
                return 'Defaults';
            }
            const base = this.answers.base_model_precision || 'no_change';
            const via = this.answers.quantize_via || 'accelerator';
            return `${base} via ${via}`;
        },

        getTextEncoderPrecisionSummary() {
            if (this.answers.model_type !== 'lora') {
                return 'N/A';
            }
            if (!Array.isArray(this.quantizationFields.textEncoders) || this.quantizationFields.textEncoders.length === 0) {
                return 'Defaults';
            }
            const overrides = this.quantizationFields.textEncoders
                .map(field => {
                    const value = this.answers[field.id];
                    if (!value || value === 'no_change') {
                        return null;
                    }
                    const label = field.shortLabel || field.label || field.id;
                    return `${label}: ${value}`;
                })
                .filter(Boolean);
            return overrides.length > 0 ? overrides.join(', ') : 'Defaults';
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
