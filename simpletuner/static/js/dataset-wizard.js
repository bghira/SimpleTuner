// Dataset Creation Wizard Component
(function() {
    'use strict';

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

            async init() {
                await this.loadBlueprints();
            },

            async loadBlueprints() {
                try {
                    const response = await fetch('/api/datasets/blueprints');
                    const data = await response.json();
                    this.blueprints = data.blueprints || [];
                } catch (error) {
                    console.error('Failed to load dataset blueprints:', error);
                    this.showToast('Failed to load dataset types', 'error');
                }
            },

            openWizard() {
                this.resetWizard();
                this.wizardOpen = true;
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

            getDefaultDataset() {
                return {
                    id: '',
                    type: 'local',
                    dataset_type: 'image',
                    resolution: 1024,
                    resolution_type: 'pixel_area',
                    caption_strategy: 'textfile',
                    metadata_backend: 'discovery',
                    cache_dir_vae: 'cache/vae',
                    probability: 1,
                    repeats: 0
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
                if (this.wizardStep === 1 && !this.validateDatasetId()) {
                    return;
                }

                this.wizardNextStep();
            },

            wizardNextStep() {
                // Skip conditioning step if not applicable
                if (this.wizardStep === 4 && !this.showConditioningStep) {
                    this.wizardStep = 5; // Go directly to review
                } else {
                    this.wizardStep++;
                }
                this.updateWizardTitle();
            },

            wizardPrevStep() {
                // Skip conditioning step when going back if not applicable
                if (this.wizardStep === 5 && !this.showConditioningStep) {
                    this.wizardStep = 4;
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
                const stepNames = ['Name', 'Type', 'Config', 'Resolution', 'Conditioning', 'Review'];
                const stepNum = this.showConditioningStep ? this.wizardStep :
                    (this.wizardStep >= 5 ? this.wizardStep - 1 : this.wizardStep);
                const stepName = this.showConditioningStep ?
                    stepNames[this.wizardStep - 1] :
                    stepNames.filter((_, i) => i !== 4)[Math.min(stepNum - 1, 4)];
                this.wizardTitle = `Create Dataset - Step ${stepNum}: ${stepName}`;
            },

            skipResolutionConfig() {
                this.resolutionConfigSkipped = true;
                // Set defaults
                this.currentDataset.resolution = 1024;
                this.currentDataset.resolution_type = 'pixel_area';
            },

            skipConditioning() {
                this.conditioningConfigured = false;
                this.selectedConditioningType = '';
                this.wizardNextStep();
            },

            addDatasetToQueue() {
                // Validate before adding
                if (!this.validateDatasetId()) {
                    this.showToast('Please fix validation errors', 'error');
                    return;
                }

                // Prepare the dataset
                const datasetToAdd = { ...this.currentDataset };

                // Add conditioning if configured
                if (this.conditioningConfigured && this.selectedConditioningType) {
                    const conditioningId = `${datasetToAdd.id}-conditioning`;
                    datasetToAdd.conditioning_data = conditioningId;
                    datasetToAdd.conditioning = [{
                        type: this.selectedConditioningType,
                        params: this.selectedConditioningType === 'canny' ?
                            { ...this.conditioningParams } : {}
                    }];

                    // Add conditioning dataset to queue
                    this.datasetQueue.push({
                        id: conditioningId,
                        type: 'local',
                        dataset_type: 'conditioning',
                        conditioning_type: 'controlnet'
                    });
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
                this.currentDataset = { ...dataset };
                this.editingQueuedDataset = true;
                this.editingIndex = index;

                // Set appropriate backend
                this.selectBackend(dataset.type);

                // Go to step 1 to allow editing
                this.wizardStep = 1;
                this.updateWizardTitle();
            },

            removeFromQueue(index) {
                if (confirm('Remove this dataset from the queue?')) {
                    this.datasetQueue.splice(index, 1);
                    this.showToast('Dataset removed from queue', 'info');
                }
            },

            async confirmAllDatasets() {
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
            }
        };
    };
})();
