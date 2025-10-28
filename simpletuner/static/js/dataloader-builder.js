// Dataloader configuration builder with full DATALOADER.md support
// Wrap in IIFE to prevent redeclaration
(function() {
    // Skip if already defined
    if (window.DataloaderBuilder) {
        return;
    }

    class DataloaderBuilder {
    constructor() {
        this.datasets = [];
        this.container = null;
        this.jsonEditor = null;
        this.fieldDependencies = this.setupFieldDependencies();
        this.conditioningGenerators = this.setupConditioningGenerators();
        this.init();
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initialize());
        } else {
            this.initialize();
        }
    }

    initialize() {
        this.container = document.getElementById('datasetList');
        this.jsonEditor = document.getElementById('dataloader_config');

        if (!this.container || !this.jsonEditor) {
            console.error('Required elements not found');
            return;
        }

        this.loadFromJSON();
        this.bindEvents();
        this.render();

        // If no datasets loaded, initialize with default
        if (this.datasets.length === 0) {
            this.loadDefaultDataset();
        }
    }

    loadDefaultDataset() {
        this.datasets = [{
            id: "my-dataset",
            dataset_type: "image",  // Required field for SimpleTuner
            type: "local",
            instance_data_dir: "/path/to/images",
            resolution: 1024,
            resolution_type: "pixel_area",
            caption_strategy: "textfile",
            metadata_backend: "discovery",
            cache_dir_vae: "cache/vae",
            minimum_image_size: 256,
            maximum_image_size: 4096,
            target_downsample_size: 1024,
            crop: true,
            crop_style: "center",
            crop_aspect: "square",
            hash_filenames: true
        }];
        this.render();
        this.syncToJSON();
    }

    setupFieldDependencies() {
        return {
            dataset_type: {
                image: {
                    show: ['.media-config', '.embed-references', '.cache-dir-vae'],
                    hide: ['.text-embed-filter', '.text-embeds-default', '.cache-dir']
                },
                video: {
                    show: ['.media-config', '.video-config', '.embed-references', '.cache-dir-vae'],
                    hide: ['.text-embed-filter', '.text-embeds-default', '.cache-dir', '.conditioning-config']
                },
                text_embeds: {
                    show: ['.text-embed-filter', '.text-embeds-default', '.cache-dir'],
                    hide: ['.media-config', '.embed-references', '.conditioning-config', '.cache-dir-vae']
                },
                image_embeds: {
                    show: ['.cache-dir'],
                    hide: ['.media-config', '.embed-references', '.conditioning-config', '.text-embed-filter', '.cache-dir-vae']
                },
                conditioning: {
                    show: ['.conditioning-config', '.cache-dir-vae'],
                    hide: ['.media-config', '.embed-references', '.text-embed-filter', '.text-embeds-default', '.cache-dir']
                }
            },
            type: {
                local: {
                    show: ['.storage-local'],
                    hide: ['.storage-aws', '.storage-csv', '.storage-huggingface']
                },
                aws: {
                    show: ['.storage-aws'],
                    hide: ['.storage-local', '.storage-csv', '.storage-huggingface']
                },
                csv: {
                    show: ['.storage-csv'],
                    hide: ['.storage-local', '.storage-aws', '.storage-huggingface']
                },
                huggingface: {
                    show: ['.storage-huggingface'],
                    hide: ['.storage-local', '.storage-aws', '.storage-csv']
                }
            },
            caption_strategy: {
                parquet: {
                    show: ['.parquet-config'],
                    hide: []
                },
                textfile: {
                    show: [],
                    hide: ['.parquet-config']
                },
                instanceprompt: {
                    show: [],
                    hide: ['.parquet-config']
                },
                filename: {
                    show: [],
                    hide: ['.parquet-config']
                },
                csv: {
                    show: [],
                    hide: ['.parquet-config']
                },
                huggingface: {
                    show: [],
                    hide: ['.parquet-config']
                }
            }
        };
    }

    setupConditioningGenerators() {
        return {
            superresolution: {
                params: {
                    blur_radius: { type: 'number', default: 2.5, step: 0.1 },
                    blur_type: { type: 'select', options: ['gaussian', 'motion', 'box'], default: 'gaussian' },
                    add_noise: { type: 'checkbox', default: true },
                    noise_level: { type: 'number', default: 0.03, min: 0, max: 1, step: 0.01 },
                    jpeg_quality: { type: 'number', default: 85, min: 10, max: 100 },
                    downscale_factor: { type: 'number', default: 2, min: 1, max: 8 }
                }
            },
            jpeg_artifacts: {
                params: {
                    quality_mode: { type: 'select', options: ['fixed', 'range'], default: 'range' },
                    quality_range: { type: 'text', default: '[10, 30]' },
                    compression_rounds: { type: 'number', default: 1, min: 1, max: 5 },
                    enhance_blocks: { type: 'checkbox', default: false }
                }
            },
            depth_midas: {
                params: {
                    model_type: { type: 'select', options: ['DPT', 'DPT_Large', 'DPT_Hybrid'], default: 'DPT' }
                }
            },
            random_masks: {
                params: {
                    mask_types: { type: 'text', default: '["rectangle", "circle", "brush", "irregular"]' },
                    min_coverage: { type: 'number', default: 0.1, min: 0, max: 1, step: 0.05 },
                    max_coverage: { type: 'number', default: 0.5, min: 0, max: 1, step: 0.05 },
                    output_mode: { type: 'select', options: ['mask', 'composite'], default: 'mask' }
                }
            },
            canny: {
                params: {
                    low_threshold: { type: 'number', default: 100, min: 0, max: 255 },
                    high_threshold: { type: 'number', default: 200, min: 0, max: 255 }
                }
            }
        };
    }

    bindEvents() {
        // Mode switcher
        document.getElementById('builderModeBtn')?.addEventListener('click', () => this.setMode('builder'));
        document.getElementById('jsonModeBtn')?.addEventListener('click', () => this.setMode('json'));
        document.getElementById('formatJsonBtn')?.addEventListener('click', () => this.formatJSON());
        document.getElementById('importJsonBtn')?.addEventListener('click', () => this.importFromJSON());

        // Add dataset buttons
        document.querySelectorAll('.add-dataset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const type = e.target.closest('.add-dataset-btn').dataset.type || 'image';
                this.addDataset(type);
            });
        });

        // Preset buttons
        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const preset = e.target.closest('.preset-btn').dataset.preset;
                this.loadPreset(preset);
            });
        });
    }

    setMode(mode) {
        const builderMode = document.getElementById('builderMode');
        const jsonMode = document.getElementById('jsonMode');
        const builderBtn = document.getElementById('builderModeBtn');
        const jsonBtn = document.getElementById('jsonModeBtn');

        if (mode === 'builder') {
            builderMode.style.display = 'block';
            jsonMode.style.display = 'none';
            builderBtn.classList.add('active');
            jsonBtn.classList.remove('active');
            this.syncToJSON();
        } else {
            builderMode.style.display = 'none';
            jsonMode.style.display = 'block';
            builderBtn.classList.remove('active');
            jsonBtn.classList.add('active');
        }
    }

    loadFromJSON() {
        try {
            const json = this.jsonEditor.value;
            if (json) {
                this.datasets = JSON.parse(json);
                this.normalizeDatasets();
            }
        } catch (e) {
            console.error('Failed to parse JSON:', e);
        }
    }

    normalizeDatasets() {
        this.datasets.forEach(dataset => {
            // Convert string numbers to actual numbers
            const numericFields = ['repeats', 'resolution', 'minimum_image_size', 'maximum_image_size',
                                 'target_downsample_size', 'probability', 'minimum_aspect_ratio',
                                 'maximum_aspect_ratio', 'write_batch_size'];

            numericFields.forEach(field => {
                if (typeof dataset[field] === 'string') {
                    dataset[field] = parseFloat(dataset[field]);
                }
            });

            // Convert boolean strings
            const booleanFields = ['crop', 'disabled', 'default', 'prepend_instance_prompt',
                                 'only_instance_prompt', 'vae_cache_clear_each_epoch',
                                 'hash_filenames', 'preserve_data_backend_cache',
                                 'is_regularisation_data'];

            booleanFields.forEach(field => {
                if (typeof dataset[field] === 'string') {
                    dataset[field] = dataset[field] === 'true';
                }
            });

            // Parse crop_aspect_buckets if it's a string
            if (typeof dataset.crop_aspect_buckets === 'string') {
                try {
                    dataset.crop_aspect_buckets = JSON.parse(dataset.crop_aspect_buckets);
                } catch (e) {
                    dataset.crop_aspect_buckets = dataset.crop_aspect_buckets.split(',').map(v => parseFloat(v.trim()));
                }
            }

            // Normalise conditioning blocks
            if (dataset.conditioning) {
                let conditioningEntries = dataset.conditioning;
                if (!Array.isArray(conditioningEntries)) {
                    conditioningEntries = [conditioningEntries];
                }

                dataset.conditioning = conditioningEntries
                    .map(entry => {
                        if (!entry || typeof entry !== 'object') {
                            return null;
                        }

                        const normalized = { ...entry };
                        const entryType = normalized.type || normalized.conditioning_type || 'superresolution';

                        if (!dataset.conditioning_type && typeof normalized.conditioning_type === 'string') {
                            dataset.conditioning_type = normalized.conditioning_type;
                        }

                        let params = {};
                        if (normalized.params && typeof normalized.params === 'object') {
                            params = { ...normalized.params };
                        }

                        // Promote remaining keys into params so they are retained/editable
                        Object.entries(normalized).forEach(([key, value]) => {
                            if (['type', 'params', 'conditioning_type'].includes(key)) {
                                return;
                            }
                            if (params[key] === undefined) {
                                params[key] = value;
                            }
                        });

                        // Ensure captions lists are preserved as multi-line strings for editing
                        return {
                            type: entryType,
                            params
                        };
                    })
                    .filter(Boolean);
            }

            if (Array.isArray(dataset.conditioning) && dataset.conditioning.length > 0) {
                if (Array.isArray(dataset.conditioning_data) && dataset.conditioning_data.length > 0) {
                    dataset.conditioning_data = [];
                }
            }

            // Disable Hugging Face streaming option until backend support improves
            if (dataset.streaming !== false) {
                dataset.streaming = false;
            }
        });
    }

    syncToJSON() {
        const json = JSON.stringify(this.datasets, null, 4);
        this.jsonEditor.value = json;
    }

    formatJSON() {
        try {
            const json = JSON.parse(this.jsonEditor.value);
            this.jsonEditor.value = JSON.stringify(json, null, 4);
            this.showToast('JSON formatted successfully!', 'success');
        } catch (e) {
            this.showToast('Invalid JSON format', 'error');
        }
    }

    importFromJSON() {
        this.loadFromJSON();
        this.render();
        this.setMode('builder');
        this.showToast('Configuration imported successfully!', 'success');
    }

    showToast(message, type = 'info') {
        // Use existing toast system if available, otherwise console log
        if (window.showToast) {
            window.showToast(message, type);
        } else if (window.trainerUI?.showToast) {
            window.trainerUI.showToast(message, type);
        } else {
            // Fallback when toast is not available (silent)
        }
    }

    addDataset(type = 'image') {
        const newDataset = this.getDefaultDataset(type);
        this.datasets.push(newDataset);
        this.render();
        this.syncToJSON();

        // Scroll to new dataset
        setTimeout(() => {
            const newItem = this.container.lastElementChild;
            newItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    }

    getDefaultDataset(type) {
        const defaults = {
            image: {
                id: `dataset-${Date.now()}`,
                type: 'local',
                dataset_type: 'image',
                instance_data_dir: '/path/to/images',
                resolution: 1024,
                resolution_type: 'pixel_area',
                minimum_image_size: 512,
                caption_strategy: 'textfile',
                metadata_backend: 'discovery',
                crop: true,
                crop_style: 'random',
                crop_aspect: 'square',
                cache_dir_vae: 'cache/vae',
                repeats: 0,
                disabled: false,
                hash_filenames: true
            },
            text_embeds: {
                id: `text-embeds-${Date.now()}`,
                type: 'local',
                dataset_type: 'text_embeds',
                default: false,
                cache_dir: 'cache/text',
                write_batch_size: 128,
                disabled: false
            },
            image_embeds: {
                id: `image-embeds-${Date.now()}`,
                type: 'local',
                dataset_type: 'image_embeds',
                cache_dir: 'cache/image_embeds',
                disabled: false
            },
            conditioning: {
                id: `conditioning-${Date.now()}`,
                type: 'local',
                dataset_type: 'conditioning',
                conditioning_type: 'controlnet',
                instance_data_dir: '/path/to/conditioning',
                cache_dir_vae: 'cache/vae/conditioning',
                disabled: false,
                hash_filenames: true
            },
            video: {
                id: `video-${Date.now()}`,
                type: 'local',
                dataset_type: 'video',
                instance_data_dir: '/path/to/videos',
                resolution: 480,
                resolution_type: 'pixel_area',
                caption_strategy: 'textfile',
                metadata_backend: 'discovery',
                cache_dir_vae: 'cache/vae/video',
                crop: false,
                video: {
                    num_frames: 125,
                    min_frames: 125,
                    is_i2v: true
                },
                repeats: 0,
                disabled: false,
                hash_filenames: true
            }
        };

        return defaults[type] || defaults.image;
    }

    removeDataset(index) {
        const dataset = this.datasets[index];

        // Check if this dataset is referenced by other datasets
        if (dataset.dataset_type === 'text_embeds' || dataset.dataset_type === 'image_embeds') {
            const refField = dataset.dataset_type === 'text_embeds' ? 'text_embeds' : 'image_embeds';
            const referencingDatasets = [];

            this.datasets.forEach((d, idx) => {
                if (idx !== index && d[refField] === dataset.id) {
                    referencingDatasets.push(d.id);
                }
            });

            if (referencingDatasets.length > 0) {
                alert(
                    `Cannot delete "${dataset.id}" because it is referenced by the following dataset(s):\n\n` +
                    referencingDatasets.map(id => `• ${id}`).join('\n') +
                    `\n\nPlease remove these references first or delete the referencing datasets.`
                );
                return;
            }
        }

        if (confirm('Are you sure you want to remove this dataset?')) {
            this.datasets.splice(index, 1);
            this.render();
            this.syncToJSON();
        }
    }

    duplicateDataset(index) {
        const dataset = JSON.parse(JSON.stringify(this.datasets[index]));
        dataset.id = `${dataset.id}-copy-${Date.now()}`;
        this.datasets.splice(index + 1, 0, dataset);
        this.render();
        this.syncToJSON();
    }

    updateDataset(index, field, value) {
        // Handle nested fields
        if (field.includes('.')) {
            const parts = field.split('.');
            let obj = this.datasets[index];
            for (let i = 0; i < parts.length - 1; i++) {
                if (!obj[parts[i]]) obj[parts[i]] = {};
                obj = obj[parts[i]];
            }
            obj[parts[parts.length - 1]] = value;
        } else {
            this.datasets[index][field] = value;
        }

        if (field === 'conditioning_data' && value && value.length) {
            if (Array.isArray(this.datasets[index].conditioning) && this.datasets[index].conditioning.length) {
                this.datasets[index].conditioning = [];
                this.showToast('Removed conditioning generators because linked datasets were selected.', 'info');
            }
        }

        // Update dependencies
        this.updateFieldDependencies(index, field, value);

        // Update type badge if needed
        if (field === 'dataset_type') {
            this.updateTypeBadge(index);
        }

        this.syncToJSON();
    }

    updateFieldDependencies(index, field, value) {
        const item = this.container.children[index];
        if (!item) return;

        const deps = this.fieldDependencies[field];
        if (!deps || !deps[value]) return;

        // Show/hide dependent fields
        deps[value].show.forEach(selector => {
            item.querySelectorAll(selector).forEach(el => el.style.display = 'block');
        });

        deps[value].hide.forEach(selector => {
            item.querySelectorAll(selector).forEach(el => el.style.display = 'none');
        });

        // Special handling for crop
        if (field === 'crop') {
            item.setAttribute('data-crop', value);
        }
    }

    updateTypeBadge(index) {
        const item = this.container.children[index];
        const badge = item.querySelector('.dataset-type-badge');
        const dataset = this.datasets[index];

        badge.className = `dataset-type-badge type-${dataset.dataset_type}`;
        badge.textContent = dataset.dataset_type.replace('_', ' ');
    }

    render() {
        if (!this.container) return;

        this.container.innerHTML = '';
        this.datasets.forEach((dataset, index) => {
            const element = this.createDatasetElement(dataset, index);
            this.container.appendChild(element);
        });
    }

    createDatasetElement(dataset, index) {
        const template = document.getElementById('dataset-item-template');
        if (!template) {
            console.error('Dataset item template not found');
            return document.createElement('div');
        }

        const clone = template.content.cloneNode(true);
        const item = clone.querySelector('.dataset-item');

        // Set index
        item.setAttribute('data-index', index);

        // Populate basic fields
        this.populateFields(item, dataset, index);

        // Bind events
        this.bindDatasetEvents(item, index);

        // Apply dependencies
        this.applyAllDependencies(item, dataset);

        return clone;
    }

    populateFields(item, dataset, index) {
        // ID and type badge
        const idInput = item.querySelector('.dataset-id-input');
        idInput.value = dataset.id || '';

        const badge = item.querySelector('.dataset-type-badge');
        badge.className = `dataset-type-badge type-${dataset.dataset_type || 'image'}`;
        badge.textContent = (dataset.dataset_type || 'image').replace('_', ' ');

        // Icon
        const icon = item.querySelector('.dataset-icon');
        const iconMap = {
            image: 'fa-image',
            video: 'fa-video',
            text_embeds: 'fa-font',
            image_embeds: 'fa-shapes',
            conditioning: 'fa-adjust'
        };
        icon.className = `dataset-icon fas ${iconMap[dataset.dataset_type] || 'fa-database'}`;

        // Populate all fields
        item.querySelectorAll('[data-field]').forEach(input => {
            const field = input.getAttribute('data-field');
            const value = this.getNestedValue(dataset, field);

            if (input.type === 'checkbox') {
                input.checked = value === true;
            } else if (input.tagName === 'SELECT') {
                input.value = value || '';
            } else if (field === 'crop_aspect_buckets' && Array.isArray(value)) {
                input.value = value.join(', ');
            } else {
                input.value = value !== undefined ? value : '';
            }
        });

        // Populate conditioning if present
        if (dataset.conditioning && Array.isArray(dataset.conditioning)) {
            this.populateConditioning(item, dataset.conditioning, index);
        }
    }

    getNestedValue(obj, path) {
        return path.split('.').reduce((acc, part) => acc && acc[part], obj);
    }

    bindDatasetEvents(item, index) {
        // ID input
        const idInput = item.querySelector('.dataset-id-input');
        idInput.addEventListener('change', (e) => {
            this.updateDataset(index, 'id', e.target.value);
        });

        // Collapse/expand
        const collapseBtn = item.querySelector('.btn-collapse');
        collapseBtn.addEventListener('click', () => {
            item.classList.toggle('collapsed');
            const icon = collapseBtn.querySelector('i');
            icon.classList.toggle('fa-chevron-down');
            icon.classList.toggle('fa-chevron-up');
        });

        // Duplicate
        const duplicateBtn = item.querySelector('.btn-duplicate');
        duplicateBtn.addEventListener('click', () => this.duplicateDataset(index));

        // Remove
        const removeBtn = item.querySelector('.btn-remove');
        removeBtn.addEventListener('click', () => this.removeDataset(index));

        // All other fields
        item.querySelectorAll('[data-field]').forEach(input => {
            const field = input.getAttribute('data-field');

            input.addEventListener('change', (e) => {
                let value = e.target.value;

                // Convert types
                if (input.type === 'checkbox') {
                    value = e.target.checked;
                } else if (input.type === 'number') {
                    value = parseFloat(value) || 0;
                } else if (field === 'crop_aspect_buckets') {
                    value = value.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
                }

                this.updateDataset(index, field, value);
            });
        });

        // Add conditioning button
        const addCondBtn = item.querySelector('.add-conditioning-btn');
        if (addCondBtn) {
            addCondBtn.addEventListener('click', () => this.addConditioning(index));
        }
    }

    applyAllDependencies(item, dataset) {
        // Apply dataset_type dependencies
        if (dataset.dataset_type) {
            this.applyDependency(item, 'dataset_type', dataset.dataset_type);
        }

        // Apply storage type dependencies
        if (dataset.type) {
            this.applyDependency(item, 'type', dataset.type);
        }

        // Apply caption strategy dependencies
        if (dataset.caption_strategy) {
            this.applyDependency(item, 'caption_strategy', dataset.caption_strategy);
        }

        // Apply crop dependencies
        item.setAttribute('data-crop', dataset.crop || false);
    }

    applyDependency(item, field, value) {
        const deps = this.fieldDependencies[field];
        if (!deps || !deps[value]) return;

        deps[value].show.forEach(selector => {
            item.querySelectorAll(selector).forEach(el => el.style.display = 'block');
        });

        deps[value].hide.forEach(selector => {
            item.querySelectorAll(selector).forEach(el => el.style.display = 'none');
        });
    }

    // Conditioning Management
    addConditioning(datasetIndex) {
        if (!this.datasets[datasetIndex].conditioning) {
            this.datasets[datasetIndex].conditioning = [];
        }

        if (this.datasets[datasetIndex].conditioning_data && this.datasets[datasetIndex].conditioning_data.length) {
            this.datasets[datasetIndex].conditioning_data = [];
            this.showToast('Cleared linked conditioning datasets; generators take precedence.', 'info');
        }

        const newCond = {
            type: 'superresolution',
            params: this.getDefaultConditioningParams('superresolution')
        };

        this.datasets[datasetIndex].conditioning.push(newCond);
        this.render();
        this.syncToJSON();
    }

    getDefaultConditioningParams(type) {
        const generator = this.conditioningGenerators[type];
        if (!generator) return {};

        const params = {};
        Object.entries(generator.params).forEach(([key, config]) => {
            params[key] = config.default;
        });

        return params;
    }

    populateConditioning(item, conditioningList, datasetIndex) {
        const container = item.querySelector('.conditioning-list');
        if (!container) return;

        container.innerHTML = '';

        conditioningList.forEach((cond, condIndex) => {
            const condItem = this.createConditioningItem(cond, datasetIndex, condIndex);
            container.appendChild(condItem);
        });
    }

    createConditioningItem(cond, datasetIndex, condIndex) {
        const template = document.getElementById('conditioning-item-template');
        if (!template) return document.createElement('div');

        const clone = template.content.cloneNode(true);
        const item = clone.querySelector('.conditioning-item');

        item.setAttribute('data-cond-index', condIndex);

        // Set type
        const typeSelect = item.querySelector('.conditioning-type');
        typeSelect.value = cond.type;

        // Create params
        const paramsContainer = item.querySelector('.conditioning-params');
        this.createConditioningParams(paramsContainer, cond.type, cond.params || {}, datasetIndex, condIndex);

        // Bind events
        typeSelect.addEventListener('change', (e) => {
            this.datasets[datasetIndex].conditioning[condIndex].type = e.target.value;
            this.datasets[datasetIndex].conditioning[condIndex].params = this.getDefaultConditioningParams(e.target.value);
            this.render();
            this.syncToJSON();
        });

        item.querySelector('.btn-remove-conditioning').addEventListener('click', () => {
            this.datasets[datasetIndex].conditioning.splice(condIndex, 1);
            this.render();
            this.syncToJSON();
        });

        return clone;
    }

    createConditioningParams(container, type, params, datasetIndex, condIndex) {
        const generator = this.conditioningGenerators[type];
        if (!generator) return;

        const definedParams = generator.params || {};

        Object.entries(definedParams).forEach(([key, config]) => {
            const group = document.createElement('div');
            group.className = 'form-group';

            const label = document.createElement('label');
            label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            group.appendChild(label);

            let input;

            if (config.type === 'select') {
                input = document.createElement('select');
                input.className = 'form-control';
                config.options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    input.appendChild(option);
                });
            } else if (config.type === 'checkbox') {
                input = document.createElement('input');
                input.type = 'checkbox';
                input.checked = params[key] || config.default;
            } else {
                input = document.createElement('input');
                input.type = config.type || 'text';
                input.className = 'form-control';
                if (config.min !== undefined) input.min = config.min;
                if (config.max !== undefined) input.max = config.max;
                if (config.step !== undefined) input.step = config.step;
            }

            input.value = params[key] !== undefined ? params[key] : config.default;

            input.addEventListener('change', (e) => {
                let value = e.target.value;
                if (input.type === 'checkbox') value = e.target.checked;
                else if (input.type === 'number') value = parseFloat(value);

                this.datasets[datasetIndex].conditioning[condIndex].params[key] = value;
                this.syncToJSON();
            });

            group.appendChild(input);
            container.appendChild(group);
        });

        // Surface any additional parameters that weren't in the generator schema
        Object.entries(params || {}).forEach(([key, value]) => {
            if (definedParams[key] !== undefined) {
                return;
            }

            const group = document.createElement('div');
            group.className = 'form-group';

            const label = document.createElement('label');
            label.textContent = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            group.appendChild(label);

            const input = document.createElement('textarea');
            input.className = 'form-control';
            const isArrayValue = Array.isArray(value);
            input.rows = isArrayValue ? Math.min(6, value.length + 1) : 3;
            input.value = isArrayValue ? value.join('\n') : value ?? '';

            input.addEventListener('change', (e) => {
                const text = e.target.value;
                const lines = text.split('\n').map(line => line.trim()).filter(Boolean);
                let nextValue;
                if (isArrayValue) {
                    nextValue = lines;
                } else if (!Number.isNaN(Number(text)) && text.trim() !== '') {
                    nextValue = Number(text);
                } else {
                    nextValue = text;
                }

                this.datasets[datasetIndex].conditioning[condIndex].params[key] = nextValue;
                this.syncToJSON();
            });

            group.appendChild(input);
            container.appendChild(group);
        });
    }

    // Presets
    loadPreset(presetName) {
        const presets = this.getPresets();
        if (presets[presetName]) {
            this.datasets = JSON.parse(JSON.stringify(presets[presetName]));
            this.render();
            this.syncToJSON();
            this.showToast(`Loaded ${presetName} preset successfully!`, 'success');
        }
    }

    getPresets() {
        return {
            'simple-local': [
                {
                    id: "training-images",
                    type: "local",
                    dataset_type: "image",
                    instance_data_dir: "/path/to/images",
                    resolution: 1024,
                    resolution_type: "pixel_area",
                    caption_strategy: "textfile",
                    metadata_backend: "discovery",
                    cache_dir_vae: "cache/vae",
                    crop: true,
                    crop_style: "random",
                    crop_aspect: "square",
                    repeats: 0,
                    hash_filenames: true
                },
                {
                    id: "text-embeds",
                    type: "local",
                    dataset_type: "text_embeds",
                    default: true,
                    cache_dir: "cache/text",
                    write_batch_size: 128
                }
            ],
            'aws-with-local-cache': [
                {
                    id: "aws-images",
                    type: "aws",
                    dataset_type: "image",
                    aws_bucket_name: "my-training-bucket",
                    aws_data_prefix: "training-data/",
                    aws_endpoint_url: "",
                    aws_region_name: "us-east-1",
                    cache_dir_vae: "/local/cache/vae",
                    resolution: 1024,
                    resolution_type: "pixel_area",
                    caption_strategy: "textfile",
                    image_embeds: "local-image-embeds",
                    text_embeds: "local-text-embeds",
                    hash_filenames: true
                },
                {
                    id: "local-image-embeds",
                    type: "local",
                    dataset_type: "image_embeds"
                },
                {
                    id: "local-text-embeds",
                    type: "local",
                    dataset_type: "text_embeds",
                    default: true,
                    cache_dir: "/local/cache/text"
                }
            ],
            'video-training': [
                {
                    id: "video-dataset",
                    type: "local",
                    dataset_type: "video",
                    instance_data_dir: "/path/to/videos",
                    resolution: 480,
                    resolution_type: "pixel_area",
                    caption_strategy: "textfile",
                    cache_dir_vae: "cache/vae/video",
                    crop: false,
                    video: {
                        num_frames: 125,
                        min_frames: 125,
                        is_i2v: true
                    },
                    hash_filenames: true
                },
                {
                    id: "text-embeds",
                    type: "local",
                    dataset_type: "text_embeds",
                    default: true,
                    cache_dir: "cache/text/video"
                }
            ],
            'controlnet-training': [
                {
                    id: "training-images",
                    type: "local",
                    dataset_type: "image",
                    instance_data_dir: "/path/to/images",
                    resolution: 1024,
                    resolution_type: "pixel_area",
                    caption_strategy: "textfile",
                    cache_dir_vae: "cache/vae",
                    conditioning_data: "canny-conditioning",
                    conditioning: [
                        {
                            type: "canny",
                            params: {
                                low_threshold: 100,
                                high_threshold: 200
                            }
                        }
                    ],
                    hash_filenames: true
                },
                {
                    id: "canny-conditioning",
                    type: "local",
                    dataset_type: "conditioning",
                    conditioning_type: "controlnet"
                },
                {
                    id: "text-embeds",
                    type: "local",
                    dataset_type: "text_embeds",
                    default: true,
                    cache_dir: "cache/text"
                }
            ]
        };
    }
}

    // Export to window
    window.DataloaderBuilder = DataloaderBuilder;

    // Don't auto-initialize - let dataloader-htmx-init.js handle it
    // when the datasets tab is actually loaded
})(); // End IIFE
