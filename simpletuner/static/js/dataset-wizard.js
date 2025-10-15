// Dataset Creation Wizard Component
(function() {
    'use strict';

    console.log('[WIZARD] dataset-wizard.js loading...');

    // Conditioning generator types reference
    const CONDITIONING_GENERATOR_TYPES = [
        { value: 'canny', label: 'Canny Edges', description: 'Detect edge maps using the Canny operator.' },
        { value: 'edges', label: 'Edges (OpenCV)', description: 'Generic edge detector (alias for Canny).' },
        { value: 'hed', label: 'HED Edge Detector', description: 'Holistically-Nested Edge Detection network.' },
        { value: 'lineart', label: 'Line Art', description: 'Line art extraction tuned for anime/illustrations.' },
        { value: 'depth', label: 'Depth (Automatic)', description: 'Depth map using default depth estimator.' },
        { value: 'depth_midas', label: 'Depth (MiDaS)', description: 'Depth estimation using MiDaS.' },
        { value: 'normal_map', label: 'Normal Map', description: 'Surface normals derived from depth.' },
        { value: 'random_masks', label: 'Random Masks', description: 'Randomly generated masks for inpainting.' },
        { value: 'jpeg_artifacts', label: 'JPEG Artifacts', description: 'Generates degraded JPEG control inputs.' },
        { value: 'superresolution', label: 'Super Resolution', description: 'Reconstructs high-res conditioning frames.' },
    ];

    window.datasetWizardComponent = function() {
        console.log('[WIZARD] datasetWizardComponent function called (Alpine initializing component)');
        return {
            // Modal state
            wizardOpen: false,
            wizardStep: 1,
            wizardTitle: 'Create Dataset - Step 1',
            saving: false,

            // Dataset blueprints from API
            blueprints: [],
            selectedBackend: null,
            selectedBlueprint: null,

            // Current dataset being configured
            currentDataset: {},
            datasetQueue: [],
            editingQueuedDataset: false,
            editingIndex: -1,

            // Config mode
            createNewConfig: false,
            deferCommit: false,  // If true, don't save to disk - return plan to caller instead
            pendingDatasetPlan: null,  // Holds the dataset plan when deferCommit is true
            existingDatasets: [],
            existingTextEmbeds: null,

            // Step-specific state
            datasetIdError: '',
            separateVaeCache: false,  // Whether to create separate VAE cache dataset
            separateTextEmbeds: false, // Whether to create separate text embeds dataset
            resolutionConfigSkipped: false,
            showAdvancedResolution: false,
            conditioningConfigured: false,
            selectedConditioningType: '',
            conditioningParams: {
                low_threshold: 100,
                high_threshold: 200
            },
            conditioningGenerators: CONDITIONING_GENERATOR_TYPES,
            newAspectBucket: null,

            // Separate cache dataset configs
            textEmbedsDataset: {
                id: 'text-embeds',
                cache_dir: 'cache/text'
            },
            vaeCacheDataset: {
                id: 'vae-cache',
                type: 'local',  // Backend type: local or aws
                // AWS-specific fields
                aws_bucket_name: '',
                aws_region_name: null,
                aws_endpoint_url: '',
                aws_access_key_id: '',
                aws_secret_access_key: ''
            },

            // File browser state
            fileBrowserOpen: false,
            fileBrowserFieldId: null,
            currentPath: '',
            parentPath: null,
            canGoUpValue: false,
            directories: [],
            selectedPath: null,
            selectedDirInfo: null,
            loadingDirectories: false,
            fileBrowserError: null,
            fileBrowserErrorType: null, // 'not_found' | 'permission' | 'other'

            // Model context (ControlNet detection and video model detection)
            get showConditioningStep() {
                const trainer = Alpine.store('trainer');
                if (!trainer) return false;
                const context = trainer.modelContext || {};
                return Boolean(context.controlnetEnabled || context.requiresConditioningDataset);
            },

            get isVideoModel() {
                const trainer = Alpine.store('trainer');
                if (!trainer) return false;
                const context = trainer.modelContext || {};
                return Boolean(context.isVideoModel || context.supportsVideo);
            },

            // Dynamic step configuration
            get stepDefinitions() {
                const steps = [
                    { id: 'name', number: 1, label: 'Name', fixed: true },
                    { id: 'type', number: 2, label: 'Type', fixed: true },
                    { id: 'config', number: 3, label: 'Config', fixed: true }
                ];

                // Insert Step 2.1 (text embeds) if enabled
                if (this.separateTextEmbeds) {
                    steps.push({ id: 'text-embeds', number: 3.1, label: 'Text Cache', fixed: false });
                }

                // Insert Step 2.2 (VAE cache) if enabled
                if (this.separateVaeCache) {
                    steps.push({ id: 'vae-cache', number: 3.2, label: 'VAE Cache', fixed: false });
                }

                // Add remaining fixed steps
                steps.push(
                    { id: 'resolution', number: 4, label: 'Resolution', fixed: true },
                    { id: 'cropping', number: 5, label: 'Cropping', fixed: true },
                    { id: 'captions', number: 6, label: 'Captions', fixed: true }
                );

                // Conditional conditioning step
                if (this.showConditioningStep) {
                    steps.push({ id: 'conditioning', number: 7, label: 'Conditioning', fixed: false });
                }

                // Final review step
                steps.push({ id: 'review', number: 8, label: 'Review', fixed: true });

                // Renumber steps sequentially
                steps.forEach((step, index) => {
                    step.displayNumber = index + 1;
                });

                return steps;
            },

            get totalSteps() {
                return this.stepDefinitions.length;
            },

            get currentStepDef() {
                return this.stepDefinitions[this.wizardStep - 1];
            },

            // Map logical step IDs to display numbers
            getStepNumber(stepId) {
                const step = this.stepDefinitions.find(s => s.id === stepId);
                return step ? step.displayNumber : null;
            },

            isStepCompleted(stepId) {
                const stepNum = this.getStepNumber(stepId);
                return stepNum !== null && this.wizardStep > stepNum;
            },

            isStepActive(stepId) {
                const stepNum = this.getStepNumber(stepId);
                return stepNum !== null && this.wizardStep === stepNum;
            },

            get canProceed() {
                if (!this.currentStepDef) return false;

                switch (this.currentStepDef.id) {
                    case 'name':
                        return this.currentDataset.id && this.currentDataset.id.trim().length > 0;
                    case 'type':
                        return this.selectedBackend !== null;
                    case 'config':
                        return this.validateRequiredFields();
                    case 'text-embeds':
                        return this.textEmbedsDataset.id && this.textEmbedsDataset.cache_dir;
                    case 'vae-cache':
                        if (!this.vaeCacheDataset.id || !this.vaeCacheDataset.type || !this.currentDataset.cache_dir_vae) {
                            return false;
                        }
                        // If AWS is selected, validate AWS fields
                        if (this.vaeCacheDataset.type === 'aws') {
                            return this.vaeCacheDataset.aws_bucket_name && this.vaeCacheDataset.aws_bucket_name.trim().length > 0;
                        }
                        return true;
                    default:
                        return true;
                }
            },

            get canGoUp() {
                return this.canGoUpValue;
            },

            async init() {
                console.log('[WIZARD] Component initializing...');
                this.currentDataset = this.getDefaultDataset();
                await this.loadBlueprints();
            },

            async loadExistingConfig() {
                try {
                    const response = await fetch('/api/datasets/plan');
                    if (response.ok) {
                        const data = await response.json();
                        this.existingDatasets = data.datasets || [];

                        // Find existing text_embeds dataset
                        this.existingTextEmbeds = this.existingDatasets.find(
                            d => d.dataset_type === 'text_embeds'
                        );

                        console.log('[WIZARD] Loaded existing config:', {
                            datasetCount: this.existingDatasets.length,
                            hasTextEmbeds: !!this.existingTextEmbeds
                        });
                    }
                } catch (error) {
                    console.error('[WIZARD] Failed to load existing config:', error);
                    // Non-fatal - wizard can still work without this
                }
            },

            async loadBlueprints() {
                try {
                    const response = await fetch('/api/datasets/blueprints');
                    const data = await response.json();
                    this.blueprints = data.blueprints || [];
                    console.log('[WIZARD] Loaded blueprints:', this.blueprints.length);
                } catch (error) {
                    console.error('Failed to load dataset blueprints:', error);
                    this.showToast('Failed to load dataset types', 'error');
                }
            },

            async openWizard() {
                console.log('[WIZARD] Opening wizard...');
                this.resetWizard();

                // Load existing datasets to check for text_embeds and validate IDs
                await this.loadExistingConfig();

                this.wizardOpen = true;
                console.log('[WIZARD] wizardOpen set to:', this.wizardOpen);

                // Debug: Check modal element after state change
                setTimeout(() => {
                    const modal = document.querySelector('.modal-backdrop');
                    console.log('[WIZARD] Modal element:', modal);
                    if (modal) {
                        console.log('[WIZARD] Modal display:', window.getComputedStyle(modal).display);
                        console.log('[WIZARD] Modal visibility:', window.getComputedStyle(modal).visibility);
                    }
                }, 100);
            },

            closeWizard() {
                if (this.datasetQueue.length > 0) {
                    if (!confirm('You have datasets in the queue. Are you sure you want to close without saving?')) {
                        return;
                    }
                }
                this.wizardOpen = false;
                this.resetWizard();
            },

            resetWizard() {
                this.wizardStep = 1;
                this.selectedBackend = null;
                this.selectedBlueprint = null;
                this.currentDataset = this.getDefaultDataset();
                this.datasetIdError = '';
                this.resolutionConfigSkipped = false;
                this.showAdvancedResolution = false;
                this.conditioningConfigured = false;
                this.selectedConditioningType = '';
                this.editingQueuedDataset = false;
                this.editingIndex = -1;
                this.updateWizardTitle();
            },

            // Deep clone helper to avoid shared object references
            deepClone(obj) {
                if (obj === null || typeof obj !== 'object') {
                    return obj;
                }
                // Try structured clone first (modern browsers), but fall back to JSON if it fails
                if (typeof structuredClone !== 'undefined') {
                    try {
                        return structuredClone(obj);
                    } catch (e) {
                        console.warn('[WIZARD] structuredClone failed, falling back to JSON:', e);
                    }
                }
                // Fallback to JSON round-trip (loses functions, but we don't have any in dataset objects)
                return JSON.parse(JSON.stringify(obj));
            },

            getDefaultDataset() {
                return {
                    id: '',
                    type: 'local',
                    dataset_type: 'image',
                    resolution: 1024,
                    resolution_type: 'pixel_area',
                    crop: false,
                    crop_style: 'random',
                    crop_aspect: 'square',
                    crop_aspect_buckets: [],
                    caption_strategy: 'textfile',
                    instance_prompt: '',
                    prepend_instance_prompt: false,
                    metadata_backend: 'discovery',
                    cache_dir_vae: 'cache/vae',
                    cache_dir_text: 'cache/text',
                    probability: 1,
                    repeats: 0,
                    parquet: {
                        path: '',
                        filename_column: 'id',
                        caption_column: 'caption',
                        width_column: '',
                        height_column: '',
                        fallback_caption_column: '',
                        identifier_includes_extension: false
                    }
                };
            },

            selectBackend(backendType) {
                this.selectedBackend = backendType;
                this.currentDataset.type = backendType;

                // Find the blueprint for this backend type
                this.selectedBlueprint = this.blueprints.find(b =>
                    b.backendType === backendType && b.datasetTypes.includes('image')
                );

                // Apply blueprint defaults
                if (this.selectedBlueprint && this.selectedBlueprint.defaults) {
                    this.currentDataset = {
                        ...this.currentDataset,
                        ...this.selectedBlueprint.defaults
                    };
                }
            },

            validateRequiredFields() {
                if (!this.selectedBlueprint) return false;
                const requiredFields = this.selectedBlueprint.fields.filter(f => f.required);
                for (const field of requiredFields) {
                    const value = this.currentDataset[field.id];
                    if (value === null || value === undefined || value === '') {
                        return false;
                    }
                }
                return true;
            },

            validateDatasetId() {
                const id = this.currentDataset.id.trim();
                if (!id) {
                    this.datasetIdError = 'Dataset ID is required';
                    return false;
                }

                // Check if ID already exists in queue
                const existsInQueue = this.datasetQueue.some((d, idx) =>
                    d.id === id && idx !== this.editingIndex
                );
                if (existsInQueue) {
                    this.datasetIdError = 'Dataset ID already exists in queue';
                    return false;
                }

                // Check if ID exists in existing datasets (when NOT editing and NOT creating new config)
                if (!this.editingQueuedDataset && !this.createNewConfig) {
                    const existsInExisting = this.existingDatasets.some(d => d.id === id);
                    if (existsInExisting) {
                        this.datasetIdError = 'Dataset ID already exists in current config (will be replaced)';
                        // This is a warning, not an error - allow it but warn user
                        // return false; // Uncomment to prevent replacement
                    }
                }

                // Check if ID contains invalid characters
                if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
                    this.datasetIdError = 'Dataset ID can only contain letters, numbers, hyphens, and underscores';
                    return false;
                }

                // Clear error if we got here and only had the replacement warning
                if (this.datasetIdError.includes('will be replaced')) {
                    // Keep the warning but allow proceeding
                    return true;
                }

                this.datasetIdError = '';
                return true;
            },

            handleStepSubmit() {
                // Validation before moving to next step
                if (this.wizardStep === 1) {
                    if (!this.validateDatasetId()) {
                        return;
                    }
                    // Update VAE cache directory to include dataset ID
                    this.updateVaeCacheDir();
                }

                this.wizardNextStep();
            },

            updateVaeCacheDir() {
                // Only update if it's still the default or empty
                if (!this.currentDataset.cache_dir_vae || this.currentDataset.cache_dir_vae === 'cache/vae') {
                    this.currentDataset.cache_dir_vae = `cache/vae/${this.currentDataset.id}`;
                }
                // Also update text cache dir if needed
                if (!this.currentDataset.cache_dir_text || this.currentDataset.cache_dir_text === 'cache/text') {
                    this.currentDataset.cache_dir_text = `cache/text/${this.currentDataset.id}`;
                }
            },

            wizardNextStep() {
                if (this.wizardStep < this.totalSteps) {
                    this.wizardStep++;
                }
                this.updateWizardTitle();
            },

            wizardPrevStep() {
                if (this.wizardStep > 1) {
                    this.wizardStep--;
                }
                this.updateWizardTitle();
            },

            goToStep(stepNumber) {
                // Allow clicking on completed steps to go back
                if (stepNumber < this.wizardStep && stepNumber >= 1 && stepNumber <= this.totalSteps) {
                    this.wizardStep = stepNumber;
                    this.updateWizardTitle();
                }
            },

            updateWizardTitle() {
                const stepDef = this.currentStepDef;
                if (stepDef) {
                    this.wizardTitle = `Create Dataset - Step ${stepDef.displayNumber}: ${stepDef.label}`;
                } else {
                    this.wizardTitle = 'Create Dataset';
                }
            },

            skipResolutionConfig() {
                this.resolutionConfigSkipped = true;
                // Set defaults
                this.currentDataset.resolution = 1024;
                this.currentDataset.resolution_type = 'pixel_area';
                // Advance to next step
                this.wizardNextStep();
            },

            skipConditioning() {
                this.conditioningConfigured = false;
                this.selectedConditioningType = '';
                this.wizardNextStep();
            },

            addAspectBucket() {
                if (this.newAspectBucket === null || this.newAspectBucket === '') {
                    return;
                }

                const value = parseFloat(this.newAspectBucket);
                if (isNaN(value) || value <= 0) {
                    this.showToast('Please enter a valid positive number', 'error');
                    return;
                }

                if (!this.currentDataset.crop_aspect_buckets) {
                    this.currentDataset.crop_aspect_buckets = [];
                }

                // Avoid duplicates
                if (!this.currentDataset.crop_aspect_buckets.includes(value)) {
                    this.currentDataset.crop_aspect_buckets.push(value);
                    this.newAspectBucket = null;
                } else {
                    this.showToast('This aspect bucket already exists', 'warning');
                }
            },

            removeAspectBucket(index) {
                if (this.currentDataset.crop_aspect_buckets) {
                    this.currentDataset.crop_aspect_buckets.splice(index, 1);
                }
            },

            addDatasetToQueue() {
                // Validate before adding
                if (!this.validateDatasetId()) {
                    this.showToast('Please fix validation errors', 'error');
                    return;
                }

                // Prepare the dataset with deep clone to avoid shared references
                const datasetToAdd = this.deepClone(this.currentDataset);

                // Clean up parquet config if not using parquet caption strategy
                if (datasetToAdd.caption_strategy !== 'parquet') {
                    delete datasetToAdd.parquet;
                }

                // Set references to separate cache datasets if enabled
                if (this.separateTextEmbeds && this.textEmbedsDataset.id) {
                    datasetToAdd.text_embeds = this.textEmbedsDataset.id;
                }
                if (this.separateVaeCache && this.vaeCacheDataset.id) {
                    datasetToAdd.image_embeds = this.vaeCacheDataset.id;
                }

                // If editing, remove any old conditioning datasets associated with this dataset
                if (this.editingQueuedDataset) {
                    const oldDataset = this.datasetQueue[this.editingIndex];
                    if (oldDataset && oldDataset.conditioning_data) {
                        // Find and remove the old conditioning dataset
                        const oldConditioningId = oldDataset.conditioning_data;
                        const conditioningIndex = this.datasetQueue.findIndex(d => d.id === oldConditioningId);
                        if (conditioningIndex !== -1 && conditioningIndex !== this.editingIndex) {
                            this.datasetQueue.splice(conditioningIndex, 1);
                            // Adjust editing index if we removed something before it
                            if (conditioningIndex < this.editingIndex) {
                                this.editingIndex--;
                            }
                        }
                    }
                }

                // Add conditioning if configured
                if (this.conditioningConfigured && this.selectedConditioningType) {
                    const conditioningId = `${datasetToAdd.id}-conditioning`;
                    datasetToAdd.conditioning_data = conditioningId;
                    datasetToAdd.conditioning = [{
                        type: this.selectedConditioningType,
                        params: this.selectedConditioningType === 'canny' ?
                            { ...this.conditioningParams } : {}
                    }];

                    // Check if conditioning dataset already exists (in case of ID reuse)
                    const existingConditioningIndex = this.datasetQueue.findIndex(d => d.id === conditioningId);
                    if (existingConditioningIndex === -1) {
                        // Add new conditioning dataset to queue
                        this.datasetQueue.push({
                            id: conditioningId,
                            type: 'local',
                            dataset_type: 'conditioning',
                            conditioning_type: 'controlnet'
                        });
                    }
                } else {
                    // Remove conditioning references if not configured
                    delete datasetToAdd.conditioning_data;
                    delete datasetToAdd.conditioning;
                }

                if (this.editingQueuedDataset) {
                    // Update existing dataset in queue
                    this.datasetQueue[this.editingIndex] = datasetToAdd;
                    this.showToast('Dataset updated in queue', 'success');
                } else {
                    // Add to queue
                    this.datasetQueue.push(datasetToAdd);
                    this.showToast('Dataset added to queue', 'success');
                }

                // Reset for next dataset
                this.resetWizard();
            },

            editQueuedDataset(index) {
                const dataset = this.datasetQueue[index];
                // Deep clone to avoid mutating the queue during editing
                this.currentDataset = this.deepClone(dataset);
                this.editingQueuedDataset = true;
                this.editingIndex = index;

                // Initialize parquet config if not present
                if (!this.currentDataset.parquet) {
                    this.currentDataset.parquet = {
                        path: '',
                        filename_column: 'id',
                        caption_column: 'caption',
                        width_column: '',
                        height_column: '',
                        fallback_caption_column: '',
                        identifier_includes_extension: false
                    };
                }

                // Set appropriate backend
                this.selectBackend(dataset.type);

                // Restore conditioning state if present
                if (dataset.conditioning_data && dataset.conditioning && dataset.conditioning.length > 0) {
                    this.conditioningConfigured = true;
                    this.selectedConditioningType = dataset.conditioning[0].type;
                    if (dataset.conditioning[0].params) {
                        this.conditioningParams = this.deepClone(dataset.conditioning[0].params);
                    }
                } else {
                    this.conditioningConfigured = false;
                    this.selectedConditioningType = '';
                }

                // Go to step 1 to allow editing
                this.wizardStep = 1;
                this.updateWizardTitle();
            },

            removeFromQueue(index) {
                if (confirm('Remove this dataset from the queue?')) {
                    const dataset = this.datasetQueue[index];

                    // If this dataset has conditioning, remove the conditioning dataset too
                    if (dataset.conditioning_data) {
                        const conditioningId = dataset.conditioning_data;
                        const conditioningIndex = this.datasetQueue.findIndex(d => d.id === conditioningId);
                        if (conditioningIndex !== -1 && conditioningIndex !== index) {
                            // Remove conditioning dataset first (to maintain indices)
                            if (conditioningIndex > index) {
                                this.datasetQueue.splice(conditioningIndex, 1);
                                this.datasetQueue.splice(index, 1);
                            } else {
                                this.datasetQueue.splice(index, 1);
                                this.datasetQueue.splice(conditioningIndex, 1);
                            }
                            this.showToast('Dataset and associated conditioning removed', 'info');
                            return;
                        }
                    }

                    // Just remove this dataset
                    this.datasetQueue.splice(index, 1);
                    this.showToast('Dataset removed from queue', 'info');
                }
            },

            async confirmAllDatasets() {
                console.log('[WIZARD] confirmAllDatasets called, queue length:', this.datasetQueue.length);

                // If we're on the review step with a current dataset that hasn't been added to queue yet,
                // automatically add it first
                const reviewStepNum = this.getStepNumber('review');
                if (this.wizardStep === reviewStepNum && this.currentDataset.id && !this.editingQueuedDataset) {
                    // Check if current dataset is already in queue
                    const alreadyInQueue = this.datasetQueue.some(d => d.id === this.currentDataset.id);
                    if (!alreadyInQueue) {
                        console.log('[WIZARD] Auto-adding current dataset to queue before save');
                        this.addDatasetToQueue();
                    }
                }

                if (this.datasetQueue.length === 0) {
                    this.showToast('No datasets to save', 'warning');
                    return;
                }

                this.saving = true;

                try {
                    // Prepare datasets to save
                    let datasetsToSave = [...this.datasetQueue];

                    // Add separate text embeds dataset if configured
                    if (this.separateTextEmbeds) {
                        const hasTextEmbeds = datasetsToSave.some(d => d.dataset_type === 'text_embeds');
                        if (!hasTextEmbeds) {
                            console.log('[WIZARD] Adding separate text_embeds dataset');
                            datasetsToSave.push({
                                id: this.textEmbedsDataset.id,
                                type: 'text_embeds',
                                dataset_type: 'text_embeds',
                                default: true,
                                cache_dir: this.textEmbedsDataset.cache_dir
                            });
                        }
                    }

                    // Add separate VAE cache dataset if configured
                    if (this.separateVaeCache) {
                        const hasImageEmbeds = datasetsToSave.some(d => d.dataset_type === 'image_embeds');
                        if (!hasImageEmbeds) {
                            console.log('[WIZARD] Adding separate image_embeds dataset for VAE cache');
                            const vaeCacheConfig = {
                                id: this.vaeCacheDataset.id,
                                type: this.vaeCacheDataset.type,
                                dataset_type: 'image_embeds',
                                default: true
                            };

                            // Add AWS-specific fields if AWS backend is selected
                            if (this.vaeCacheDataset.type === 'aws') {
                                vaeCacheConfig.aws_bucket_name = this.vaeCacheDataset.aws_bucket_name;
                                if (this.vaeCacheDataset.aws_region_name) {
                                    vaeCacheConfig.aws_region_name = this.vaeCacheDataset.aws_region_name;
                                }
                                if (this.vaeCacheDataset.aws_endpoint_url) {
                                    vaeCacheConfig.aws_endpoint_url = this.vaeCacheDataset.aws_endpoint_url;
                                }
                                if (this.vaeCacheDataset.aws_access_key_id) {
                                    vaeCacheConfig.aws_access_key_id = this.vaeCacheDataset.aws_access_key_id;
                                }
                                if (this.vaeCacheDataset.aws_secret_access_key) {
                                    vaeCacheConfig.aws_secret_access_key = this.vaeCacheDataset.aws_secret_access_key;
                                }
                            }

                            datasetsToSave.push(vaeCacheConfig);
                        }
                    }

                    // If creating new config, ensure text_embeds dataset is included
                    if (this.createNewConfig && !this.separateTextEmbeds) {
                        const hasTextEmbeds = datasetsToSave.some(d => d.dataset_type === 'text_embeds');
                        if (!hasTextEmbeds && this.currentDataset.cache_dir_text) {
                            console.log('[WIZARD] Auto-adding text_embeds dataset for new config');
                            datasetsToSave.push({
                                id: 'text-embeds',
                                type: 'text_embeds',
                                dataset_type: 'text_embeds',
                                default: true,
                                cache_dir: this.currentDataset.cache_dir_text
                            });
                        }
                    }

                    if (!this.createNewConfig) {
                        // Append mode: merge with existing datasets
                        console.log('[WIZARD] Append mode: merging with existing datasets');
                        datasetsToSave = this.mergeWithExistingDatasets(datasetsToSave);
                    }

                    // If deferCommit is true, store the plan and close without saving
                    if (this.deferCommit) {
                        console.log('[WIZARD] Defer commit mode: storing dataset plan without saving');
                        this.pendingDatasetPlan = datasetsToSave;
                        this.showToast(`Dataset plan prepared (${this.datasetQueue.length} dataset(s))`, 'success');

                        // Close wizard - the parent (training wizard) will handle the commit
                        this.wizardOpen = false;
                        // Don't reset wizard or clear queue yet - parent might need the data
                        return;
                    }

                    // Normal mode: save immediately
                    const response = await fetch('/api/datasets/plan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            datasets: datasetsToSave,
                            createBackup: false
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        let errorMessage = 'Failed to save datasets';

                        if (error.detail?.message) {
                            errorMessage = error.detail.message;
                        }

                        // If there are validation errors, show them
                        if (error.detail?.validations && Array.isArray(error.detail.validations)) {
                            const errorFields = error.detail.validations
                                .filter(v => v.level === 'error')
                                .map(v => `${v.field}: ${v.message}`)
                                .join('; ');
                            if (errorFields) {
                                errorMessage += ` (${errorFields})`;
                            }
                        }

                        throw new Error(errorMessage);
                    }

                    const result = await response.json();
                    this.showToast(`Successfully saved ${this.datasetQueue.length} dataset(s)`, 'success');

                    // Close wizard and reload datasets
                    this.wizardOpen = false;
                    this.resetWizard();
                    this.datasetQueue = [];

                    // Refresh the dataloader section
                    const trainer = Alpine.store('trainer');
                    if (trainer && typeof trainer.loadDatasets === 'function') {
                        await trainer.loadDatasets();
                    }

                } catch (error) {
                    console.error('Failed to save datasets:', error);
                    this.showToast(error.message || 'Failed to save datasets', 'error');
                    // DO NOT reset wizard on error - keep user on review step
                    // DO NOT close modal - let them fix the issue
                } finally {
                    this.saving = false;
                }
            },

            mergeWithExistingDatasets(newDatasets) {
                // Start with existing datasets
                const merged = [...this.existingDatasets];

                // Add new datasets, checking for ID conflicts
                for (const newDataset of newDatasets) {
                    const existingIndex = merged.findIndex(d => d.id === newDataset.id);
                    if (existingIndex >= 0) {
                        // Replace existing dataset with same ID
                        console.log(`[WIZARD] Replacing existing dataset: ${newDataset.id}`);
                        merged[existingIndex] = newDataset;
                    } else {
                        // Add new dataset
                        merged.push(newDataset);
                    }
                }

                return merged;
            },

            showToast(message, type = 'info') {
                if (window.showToast) {
                    window.showToast(message, type);
                } else {
                    console.log(`[${type.toUpperCase()}] ${message}`);
                }
            },

            // File browser methods
            async openFileBrowser(fieldId) {
                this.fileBrowserFieldId = fieldId;
                this.fileBrowserOpen = true;

                // Load initial directory - server will use configured datasets_dir if no path provided
                await this.loadDirectories(null);
            },

            async loadDirectories(path) {
                this.loadingDirectories = true;
                this.directories = [];
                this.selectedPath = null;
                this.selectedDirInfo = null;
                this.fileBrowserError = null;
                this.fileBrowserErrorType = null;

                try {
                    const url = path
                        ? `/api/datasets/browse?path=${encodeURIComponent(path)}`
                        : '/api/datasets/browse';

                    const response = await fetch(url);
                    if (!response.ok) {
                        const error = await response.json();

                        // Determine error type based on status code
                        if (response.status === 404) {
                            this.fileBrowserErrorType = 'not_found';
                            this.fileBrowserError = error.detail || 'Directory does not exist';
                        } else if (response.status === 403) {
                            this.fileBrowserErrorType = 'permission';
                            this.fileBrowserError = error.detail || 'Access denied';
                        } else {
                            this.fileBrowserErrorType = 'other';
                            this.fileBrowserError = error.detail || 'Failed to load directories';
                        }
                        return;
                    }

                    const data = await response.json();
                    this.directories = data.directories || [];
                    this.currentPath = data.currentPath;
                    this.parentPath = data.parentPath;
                    this.canGoUpValue = data.canGoUp;
                } catch (error) {
                    console.error('Failed to load directories:', error);
                    this.fileBrowserErrorType = 'other';
                    this.fileBrowserError = error.message || 'Failed to load directories';
                } finally {
                    this.loadingDirectories = false;
                }
            },

            async selectDirectory(path) {
                this.selectedPath = path;

                // Check if this directory has dataset metadata
                try {
                    const response = await fetch(`/api/datasets/detect?path=${encodeURIComponent(path)}`);
                    if (response.ok) {
                        this.selectedDirInfo = await response.json();
                    }
                } catch (error) {
                    console.error('Failed to detect dataset:', error);
                }
            },

            async navigateToDirectory(path) {
                await this.loadDirectories(path);
            },

            async goToParentDirectory() {
                if (this.parentPath) {
                    await this.loadDirectories(this.parentPath);
                }
            },

            confirmDirectory() {
                if (this.selectedPath && this.fileBrowserFieldId) {
                    this.currentDataset[this.fileBrowserFieldId] = this.selectedPath;
                    this.closeFileBrowser();
                }
            },

            importDatasetConfig() {
                if (!this.selectedDirInfo || !this.selectedDirInfo.config) return;

                const config = this.selectedDirInfo.config;

                // Import all relevant fields from the detected config
                if (config.resolution) this.currentDataset.resolution = config.resolution;
                if (config.resolution_type) this.currentDataset.resolution_type = config.resolution_type;
                if (config.caption_strategy) this.currentDataset.caption_strategy = config.caption_strategy;
                if (config.dataset_type) this.currentDataset.dataset_type = config.dataset_type;
                if (config.crop !== undefined) this.currentDataset.crop = config.crop;
                if (config.crop_aspect) this.currentDataset.crop_aspect = config.crop_aspect;
                if (config.crop_style) this.currentDataset.crop_style = config.crop_style;
                if (config.maximum_image_size) this.currentDataset.maximum_image_size = config.maximum_image_size;
                if (config.target_downsample_size) this.currentDataset.target_downsample_size = config.target_downsample_size;
                if (config.probability !== undefined) this.currentDataset.probability = config.probability;
                if (config.repeats !== undefined) this.currentDataset.repeats = config.repeats;

                // Import video config if present
                if (config.video) {
                    this.currentDataset.video = { ...config.video };
                }

                // Suggest using the detected dataset ID
                if (this.selectedDirInfo.datasetId && !this.currentDataset.id) {
                    this.currentDataset.id = this.selectedDirInfo.datasetId;
                }

                this.showToast('Dataset configuration imported successfully', 'success');
                this.confirmDirectory();
            },

            closeFileBrowser() {
                this.fileBrowserOpen = false;
                this.fileBrowserFieldId = null;
                this.currentPath = '';
                this.parentPath = null;
                this.canGoUpValue = false;
                this.directories = [];
                this.selectedPath = null;
                this.selectedDirInfo = null;
                this.fileBrowserError = null;
                this.fileBrowserErrorType = null;
            },

            openDatasetsDirConfig() {
                // Close the file browser and dataset wizard
                this.closeFileBrowser();
                this.closeWizard();

                // Trigger the onboarding modal for datasets_dir
                // This is handled by the webui-state component
                if (window.Alpine && Alpine.store('webuiState')) {
                    const webuiState = Alpine.store('webuiState');
                    if (webuiState.openOnboardingStepModal) {
                        webuiState.openOnboardingStepModal('default_datasets_dir');
                    }
                }
            },

            // New Configuration Modal State & Methods
            showNewConfigModal: false,
            newConfigName: '',
            creatingConfig: false,
            currentEnvironment: '',

            openNewConfigModal() {
                // Get the current environment from the trainer store
                const trainer = Alpine.store('trainer');
                if (trainer && trainer.activeEnvironment) {
                    this.currentEnvironment = trainer.activeEnvironment;
                } else {
                    // Fallback: Get from webui state
                    const webuiState = Alpine.store('webuiState');
                    if (webuiState && webuiState.defaults && webuiState.defaults.configs_dir) {
                        this.currentEnvironment = webuiState.defaults.configs_dir;
                    }
                }

                this.newConfigName = '';
                this.showNewConfigModal = true;
                document.body.classList.add('modal-open');
            },

            closeNewConfigModal() {
                this.showNewConfigModal = false;
                this.newConfigName = '';
                this.creatingConfig = false;
                document.body.classList.remove('modal-open');
            },

            async createNewDataloaderConfig() {
                // EXTENSIVE LOGGING to find the culprit
                const stack = new Error().stack;
                console.log('[NEW CONFIG] ============ FUNCTION CALLED ============');
                console.log('[NEW CONFIG] Stack trace:', stack);
                console.log('[NEW CONFIG] Current state - name:', this.newConfigName, 'creating:', this.creatingConfig);

                // Check if this is being called from Alpine event handler
                if (stack.includes('Alpine') || stack.includes('cdn.min.js')) {
                    console.log('[NEW CONFIG] ⚠️ CALLED FROM ALPINE.JS FRAMEWORK');
                }

                // Log all event listeners on the button
                const button = document.querySelector('#new-dataloader-config-modal button.btn-primary');
                if (button) {
                    console.log('[NEW CONFIG] Button element:', button);
                    console.log('[NEW CONFIG] Button onclick:', button.onclick);
                }

                // CRITICAL: Use timestamp-based guard to prevent infinite loops
                const now = Date.now();
                if (this._lastConfigCreationAttempt && (now - this._lastConfigCreationAttempt) < 2000) {
                    console.log('[NEW CONFIG] Blocked: too soon after last attempt (debounce)');
                    return;
                }

                // Guard against multiple simultaneous calls
                if (!this.newConfigName || !this.newConfigName.trim() || this.creatingConfig) {
                    console.log('[NEW CONFIG] Blocked: empty name or already creating');
                    return;
                }

                // Validate config name format
                const namePattern = /^[a-zA-Z0-9_-]+$/;
                if (!namePattern.test(this.newConfigName)) {
                    if (window.showToast) {
                        window.showToast('Invalid configuration name. Use only letters, numbers, hyphens, and underscores.', 'error');
                    }
                    return;
                }

                // Get the current environment name
                const trainer = Alpine.store('trainer');
                const environmentName = trainer && trainer.activeEnvironment;

                if (!environmentName) {
                    if (window.showToast) {
                        window.showToast('No environment selected. Please select an environment first.', 'error');
                    }
                    return;
                }

                console.log('[NEW CONFIG] Creating config:', this.newConfigName, 'for environment:', environmentName);

                // Set guards
                this._lastConfigCreationAttempt = now;
                this.creatingConfig = true;

                try {
                    // Use the proper endpoint to create a dataloader for the environment
                    // This creates an EMPTY multidatabackend config (no default datasets)
                    const response = await fetch(`/api/configs/${encodeURIComponent(environmentName)}/dataloader`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            path: this.newConfigName ? `${environmentName}/multidatabackend-${this.newConfigName}.json` : null,
                            include_defaults: false
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json().catch(() => ({ detail: 'Failed to create configuration' }));
                        throw new Error(error.detail || 'Failed to create configuration');
                    }

                    const result = await response.json();

                    // Show success message
                    if (window.showToast) {
                        window.showToast(`Configuration created successfully at ${result.dataloader?.path || this.newConfigName + '.json'}`, 'success');
                    }

                    // Close the modal
                    this.closeNewConfigModal();

                    // Reload the dataset plan to reflect the new config (but don't await to avoid blocking)
                    this.loadExistingConfig().catch(err => {
                        console.error('[NEW CONFIG] Failed to reload config:', err);
                    });

                } catch (error) {
                    console.error('[NEW CONFIG] Failed to create configuration:', error);
                    if (window.showToast) {
                        window.showToast(error.message || 'Failed to create configuration', 'error');
                    }
                    // Don't close modal on error so user can retry with different name
                } finally {
                    this.creatingConfig = false;
                }
            }
        };
    };

    console.log('[WIZARD] dataset-wizard.js loaded, window.datasetWizardComponent =', typeof window.datasetWizardComponent);
})();
