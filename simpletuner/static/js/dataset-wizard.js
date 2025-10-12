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

            // Step-specific state
            datasetIdError: '',
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

            // Model context (ControlNet detection)
            get showConditioningStep() {
                const trainer = Alpine.store('trainer');
                if (!trainer) return false;
                const context = trainer.modelContext || {};
                return Boolean(context.controlnetEnabled || context.requiresConditioningDataset);
            },

            get canProceed() {
                switch (this.wizardStep) {
                    case 1:
                        return this.currentDataset.id && this.currentDataset.id.trim().length > 0;
                    case 2:
                        return this.selectedBackend !== null;
                    case 3:
                        return this.validateRequiredFields();
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

            openWizard() {
                console.log('[WIZARD] Opening wizard...');
                this.resetWizard();
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
                // Use structured clone if available (modern browsers)
                if (typeof structuredClone !== 'undefined') {
                    return structuredClone(obj);
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

                // Check if ID contains invalid characters
                if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
                    this.datasetIdError = 'Dataset ID can only contain letters, numbers, hyphens, and underscores';
                    return false;
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
            },

            wizardNextStep() {
                // Skip conditioning step (7) if not applicable
                if (this.wizardStep === 6 && !this.showConditioningStep) {
                    this.wizardStep = 7; // Skip conditioning (step 7), go to review (step 7)
                } else {
                    this.wizardStep++;
                }
                this.updateWizardTitle();
            },

            wizardPrevStep() {
                // Skip conditioning step when going back if not applicable
                if (this.wizardStep === 7 && !this.showConditioningStep) {
                    this.wizardStep = 6; // Go back to captions (step 6)
                } else {
                    this.wizardStep--;
                }
                this.updateWizardTitle();
            },

            goToStep(step) {
                // Allow clicking on completed steps to go back
                if (step < this.wizardStep) {
                    this.wizardStep = step;
                    this.updateWizardTitle();
                }
            },

            updateWizardTitle() {
                const stepNames = ['Name', 'Type', 'Config', 'Resolution', 'Cropping', 'Captions', 'Conditioning', 'Review'];
                const stepNum = this.showConditioningStep ? this.wizardStep :
                    (this.wizardStep >= 7 ? this.wizardStep - 1 : this.wizardStep);
                const stepName = this.showConditioningStep ?
                    stepNames[this.wizardStep - 1] :
                    stepNames.filter((_, i) => i !== 6)[Math.min(stepNum - 1, 6)];
                this.wizardTitle = `Create Dataset - Step ${stepNum}: ${stepName}`;
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
                if (this.datasetQueue.length === 0) {
                    this.showToast('No datasets to save', 'warning');
                    return;
                }

                this.saving = true;

                try {
                    const response = await fetch('/api/datasets/plan', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            datasets: this.datasetQueue,
                            createBackup: false
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail?.message || 'Failed to save datasets');
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
                } finally {
                    this.saving = false;
                }
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
            }
        };
    };

    console.log('[WIZARD] dataset-wizard.js loaded, window.datasetWizardComponent =', typeof window.datasetWizardComponent);
})();
