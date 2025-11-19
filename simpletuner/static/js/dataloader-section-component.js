// Alpine component logic for the datasets builder (extracted from trainer_dataloader_section.html)
(() => {
if (window.dataloaderSectionComponent) {
    return;
}
const CONDITIONING_GENERATOR_TYPES = [
    { value: 'canny', label: 'Canny Edges', description: 'Detect edge maps using the Canny operator.' },
    { value: 'edges', label: 'Edges (OpenCV)', description: 'Generic edge detector (alias for Canny).' },
    { value: 'hed', label: 'HED Edge Detector', description: 'Holistically-Nested Edge Detection network.' },
    { value: 'lineart', label: 'Line Art', description: 'Line art extraction tuned for anime/illustrations.' },
    { value: 'lineart_coarse', label: 'Line Art (Coarse)', description: 'Coarse line art extraction variant.' },
    { value: 'scribble', label: 'Scribble', description: 'Generates scribble-style outlines.' },
    { value: 'depth', label: 'Depth (Automatic)', description: 'Depth map using default depth estimator (alias for MiDaS).' },
    { value: 'depth_midas', label: 'Depth (MiDaS)', description: 'Depth estimation using MiDaS.' },
    { value: 'depth_leres', label: 'Depth (LeReS)', description: 'Depth estimation tuned for outdoor scenes.' },
    { value: 'normal_map', label: 'Normal Map', description: 'Surface normals derived from depth.' },
    { value: 'random_masks', label: 'Random Masks', description: 'Randomly generated masks for inpainting.' },
    { value: 'binary_mask', label: 'Binary Mask', description: 'Binary mask generation for inpainting workflows.' },
    { value: 'inpainting', label: 'Inpainting Mask', description: 'Alias for random mask based generators.' },
    { value: 'jpeg_artifacts', label: 'JPEG Artifacts', description: 'Generates degraded JPEG control inputs.' },
    { value: 'superresolution', label: 'Super Resolution', description: 'Reconstructs high-res conditioning frames.' },
    { value: 'content_shuffle', label: 'Content Shuffle', description: 'Shuffles image content for texture variations.' },
    { value: 'tile', label: 'Tile / Texture', description: 'Tile-friendly control inputs for seamless textures.' },
    { value: 'rembg', label: 'Foreground Mask (Rembg)', description: 'Foreground/background segmentation using Rembg.' },
    { value: '__custom__', label: 'Other (enter manually)', description: 'Specify a custom generator id.' },
];

const SKIP_DISCOVERY_OPTIONS = [
    {
        value: 'text',
        label: 'Skip text embeds',
        hint: 'Skips discovering new captions or generating updated text embeddings.'
    },
    {
        value: 'vae',
        label: 'Skip VAE cache',
        hint: 'Prevents VAE cache refreshes for new or updated images.'
    },
    {
        value: 'aspect',
        label: 'Skip aspect buckets',
        hint: 'Avoids recalculating aspect buckets; only safe when image dimensions are unchanged.'
    },
    {
        value: 'metadata',
        label: 'Skip metadata discovery',
        hint: 'Disables general file discovery. Use only when the dataset contents are frozen.'
    }
];

window.CONDITIONING_GENERATOR_TYPES = window.CONDITIONING_GENERATOR_TYPES || CONDITIONING_GENERATOR_TYPES;

const DATASET_COLLAPSE_SECTION_MAP = {
    card: '_collapsed',
    basic: '_showBasicSection',
    sizing: '_showSizing',
    cropping: '_showCropping',
    storage: '_showStorage',
    captions: '_showCaptions',
    video: '_showVideoSettings',
    conditioning: '_showConditioning',
    advanced: '_showAdvanced'
};

const COLLAPSED_SECTIONS_ENDPOINT = '/api/webui/ui-state/collapsed-sections/datasets';

function dataloaderSectionComponent() {
    return {
    storeReady: false,
    _storeReadyHandled: false,
    dangerMode: false,
    _trainerCache: null,
    _captionFilterCache: null,
    _captionFilterCacheTime: 0,
    _collapsedState: {},
    _collapseStateLoaded: false,
    _collapseStateLoading: false,
    _collapseSaveTimeout: null,

    init() {
        window.dataloaderSectionComponentInstance = this;
        if (window.__pendingGeneratorSync) {
            this.syncGeneratorSelectionsForAll();
            delete window.__pendingGeneratorSync;
        }
        this.dangerMode = this.computeDangerModeEnabled();

        if (!this._dangerModeHandler) {
            this._dangerModeHandler = (event) => {
                if (event && typeof event.detail?.enabled !== 'undefined') {
                    this.dangerMode = !!event.detail.enabled;
                } else {
                    this.dangerMode = this.computeDangerModeEnabled();
                }
            };
            window.addEventListener('trainer-danger-mode-changed', this._dangerModeHandler);
        }

        if (Alpine.store('trainer')) {
            this.onStoreReady();
        }
    },

    onStoreReady() {
        if (this._storeReadyHandled) {
            return;
        }
        this._storeReadyHandled = true;
        this.storeReady = true;
        if (Array.isArray(this.datasets)) {
            this.datasets.forEach((dataset) => this.ensureDatasetRuntimeState(dataset));
        }
        this.ensureCaptionFiltersLoaded();
        this.syncCaptionFilterSelections();
        this.syncGeneratorSelectionsForAll();
        this.dangerMode = this.computeDangerModeEnabled();
        if (!this._collapseStateLoaded && !this._collapseStateLoading) {
            this.loadCollapsedState();
        } else if (this._collapseStateLoaded) {
            this.applyCollapsedStateAll();
        }
    },

    get trainer() {
        if (!this._trainerCache) {
            this._trainerCache = Alpine.store('trainer') || {};
        }
        return this._trainerCache;
    },
    get dataLoaderMode() {
        return this.trainer.dataLoaderMode || 'builder';
    },
    get datasets() {
        return this.trainer.datasets || [];
    },
    get datasetsLoading() {
        return this.trainer.datasetsLoading || false;
    },
    get dataLoaderJson() {
        return this.trainer.dataLoaderJson || '';
    },
    set dataLoaderJson(value) {
        const trainer = Alpine.store('trainer');
        if (trainer) {
            trainer.dataLoaderJson = value;
            this.markAsUnsaved();
        }
    },
    get hasUnsavedChanges() {
        return this.trainer.hasUnsavedChanges || false;
    },
    get modelContext() {
        const trainer = Alpine.store('trainer');
        if (!trainer) {
            return {};
        }
        return trainer.modelContext || (typeof trainer.defaultModelContext === 'function' ? trainer.defaultModelContext() : {});
    },
    get conditioningFeaturesActive() {
        const context = this.modelContext || {};
        return Boolean(context.controlnetEnabled || context.requiresConditioningDataset);
    },
    get conditioningSupported() {
        return this.conditioningFeaturesActive;
    },
    get conditioningRequired() {
        return Boolean(this.modelContext.requiresConditioningDataset);
    },
    get isVideoModel() {
        const context = this.modelContext || {};
        return this.normalizeBoolean(context.isVideoModel) || this.normalizeBoolean(context.supportsVideo);
    },
    get isAudioModel() {
        const trainer = Alpine.store('trainer');
        if (!trainer) return false;
        // Try modelContext first
        if (trainer.modelContext && trainer.modelContext.isAudioModel) {
            return true;
        }
        // Fallback to config values similar to the wizard logic
        const config = trainer.configValues || {};
        const family = (
            config['model_family'] ||
            config['--model_family'] ||
            ''
        ).toString().toLowerCase();
        return family === 'ace_step';
    },
    get requiresStrictI2VDatasets() {
        const context = this.modelContext || {};
        const active = this.normalizeBoolean(context.strictI2VActive);
        if (active) {
            console.debug('[DatasetBuilder] strict I2V active for current model flavour', context);
        }
        return active;
    },
    get imageDatasetCount() {
        if (!Array.isArray(this.datasets)) {
            return 0;
        }
        return this.datasets.filter((dataset) => {
            if (!dataset || typeof dataset !== 'object') {
                return false;
            }
            const type = typeof dataset.dataset_type === 'string' ? dataset.dataset_type.toLowerCase() : '';
            if (type !== 'image') {
                return false;
            }
            return !dataset.disabled;
        }).length;
    },
    get videoDatasetCount() {
        if (!Array.isArray(this.datasets)) {
            return 0;
        }
        return this.datasets.filter((dataset) => {
            if (!dataset || typeof dataset !== 'object') {
                return false;
            }
            const type = typeof dataset.dataset_type === 'string' ? dataset.dataset_type.toLowerCase() : '';
            if (type !== 'video') {
                return false;
            }
            return !dataset.disabled;
        }).length;
    },
    get conditioningDatasetType() {
        return this.modelContext.conditioningDatasetType || 'conditioning';
    },
    get availableGeneratorTypes() {
        return CONDITIONING_GENERATOR_TYPES;
    },
    get captionFilterOptions() {
        const now = Date.now();
        if (this._captionFilterCache && (now - this._captionFilterCacheTime) < 1000) {
            return this._captionFilterCache;
        }

        const trainer = Alpine.store('trainer');
        if (!trainer || !Array.isArray(trainer.captionFilters)) {
            this._captionFilterCache = [];
            this._captionFilterCacheTime = now;
            return [];
        }

        this._captionFilterCache = trainer.captionFilters.map((filter) => {
            if (!filter) {
                return { value: '', label: '', description: '' };
            }
            const value = filter.path || filter.name || '';
            return {
                value,
                name: filter.name || value,
                label: filter.label || filter.name || value,
                description: filter.description || ''
            };
        }).filter((option) => option.value);
        this._captionFilterCacheTime = now;
        return this._captionFilterCache;
    },
    ensureCaptionFiltersLoaded() {
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.loadCaptionFilters === 'function') {
            trainer.loadCaptionFilters();
        }
    },
    ensureDatasetRuntimeState(dataset) {
        if (!dataset || typeof dataset !== 'object') {
            return;
        }
        if (dataset._connectionStatus === undefined) {
            dataset._connectionStatus = null;
        }
        if (dataset._connectionMessage === undefined) {
            dataset._connectionMessage = '';
        }
        if (dataset._connectionDetails === undefined) {
            dataset._connectionDetails = null;
        }
        if (dataset._connectionTesting === undefined) {
            dataset._connectionTesting = false;
        }
        if (dataset._awsEndpointVisible === undefined) {
            dataset._awsEndpointVisible = false;
        }
        if (dataset._awsAccessVisible === undefined) {
            dataset._awsAccessVisible = false;
        }
        if (dataset._awsSecretVisible === undefined) {
            dataset._awsSecretVisible = false;
        }
        if (!Array.isArray(dataset._skipDiscovery)) {
            dataset._skipDiscovery = this.parseSkipDiscovery(dataset.skip_file_discovery);
        }
        // Initialize text_embeds and image_embeds to empty string if undefined or null
        // This ensures x-model bindings work correctly in select dropdowns
        if (dataset.text_embeds === undefined || dataset.text_embeds === null) {
            dataset.text_embeds = '';
        }
        if (dataset.image_embeds === undefined || dataset.image_embeds === null) {
            dataset.image_embeds = '';
        }
        if (!Array.isArray(dataset.conditioning)) {
            if (dataset.conditioning && typeof dataset.conditioning === 'object') {
                dataset.conditioning = [dataset.conditioning];
            } else {
                dataset.conditioning = [];
            }
        }
        if (dataset._showAdvanced === undefined) {
            dataset._showAdvanced = false;
        }
        if (dataset._showConditioning === undefined) {
            dataset._showConditioning = true;
        }
        if (dataset._collapsed === undefined) {
            dataset._collapsed = false;
        }
        if (dataset._showBasicSection === undefined) {
            dataset._showBasicSection = true;
        }
        if (dataset._showSizing === undefined) {
            dataset._showSizing = true;
        }
        if (dataset._showCropping === undefined) {
            dataset._showCropping = true;
        }
        if (dataset._showStorage === undefined) {
            dataset._showStorage = true;
        }
        if (dataset._showCaptions === undefined) {
            dataset._showCaptions = true;
        }
        if (dataset._showVideoSettings === undefined) {
            dataset._showVideoSettings = true;
        }
        if (this._collapseStateLoaded) {
            this.applyCollapsedStateToDataset(dataset);
        }
    },
    syncCaptionFilterSelections() {
        if (!Array.isArray(this.datasets)) {
            return;
        }
        const trainer = Alpine.store('trainer');
        const filters = trainer && Array.isArray(trainer.captionFilters) ? trainer.captionFilters : [];
        const canonicalMap = new Map();
        filters.forEach((filter) => {
            if (!filter) {
                return;
            }
            const canonical = filter.path || filter.name || '';
            if (!canonical) {
                return;
            }
            canonicalMap.set(canonical, canonical);
            if (filter.name && filter.name !== canonical) {
                canonicalMap.set(filter.name, canonical);
            }
        });
        this.datasets.forEach((dataset) => {
            if (!dataset) {
                return;
            }
            this.ensureDatasetRuntimeState(dataset);
            if (dataset.dataset_type === 'text_embeds') {
                const current = typeof dataset.caption_filter_list === 'string' ? dataset.caption_filter_list.trim() : '';
                if (!current) {
                    dataset._selectedCaptionFilter = '';
                    return;
                }
                const resolved = trainer && typeof trainer.resolveCaptionFilterValue === 'function'
                    ? trainer.resolveCaptionFilterValue(current)
                    : current;
                const canonical = canonicalMap.get(resolved) || canonicalMap.get(current);
                if (canonical) {
                    dataset.caption_filter_list = canonical;
                    dataset._selectedCaptionFilter = canonical;
                } else {
                    dataset._selectedCaptionFilter = '__custom__';
                }
            } else {
                dataset._selectedCaptionFilter = '';
            }
        });
    },
    syncGeneratorSelectionsForAll() {
        if (!Array.isArray(this.datasets)) {
            return;
        }
        this.datasets.forEach((dataset) => {
            this.ensureDatasetRuntimeState(dataset);
            this.syncGeneratorSelections(dataset);
        });
    },
    supportsConnectionTest(dataset) {
        if (!dataset) {
            return false;
        }
        const type = dataset.type || '';
        const datasetType = dataset.dataset_type || '';
        if (['aws', 'csv', 'huggingface'].includes(type)) {
            return true;
        }
        if (datasetType === 'image_embeds') {
            return true;
        }
        return false;
    },
    touchConnectionField(dataset) {
        if (!dataset) {
            return;
        }
        this.ensureDatasetRuntimeState(dataset);
        dataset._connectionStatus = null;
        dataset._connectionMessage = '';
        dataset._connectionDetails = null;
        this.markAsUnsaved();
    },
    async testDatasetConnection(dataset) {
        const trainer = Alpine.store('trainer');
        if (!dataset || !trainer || typeof trainer.prepareDatasetsForSave !== 'function') {
            window.showToast('Dataset builder not ready for connection test.', 'error');
            return;
        }

        this.ensureDatasetRuntimeState(dataset);
        dataset._connectionTesting = true;
        dataset._connectionStatus = null;
        dataset._connectionMessage = '';
        dataset._connectionDetails = null;

        try {
            const sanitized = trainer.prepareDatasetsForSave([dataset]);
            if (!Array.isArray(sanitized) || !sanitized[0]) {
                throw new Error('Unable to prepare dataset payload.');
            }
            const payload = {
                dataset: sanitized[0],
                configs_dir: trainer.defaults?.configs_dir || null,
            };
            const response = await fetch('/api/datasets/test-connection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const error = await response.json().catch(() => ({}));
                const message = error?.detail?.message || error?.detail || response.statusText || 'Connection test failed.';
                throw new Error(message);
            }
            const result = await response.json();
            dataset._connectionStatus = 'success';
            dataset._connectionMessage = result.message || 'Connection successful.';
            dataset._connectionDetails = result.details || null;
            window.showToast(dataset._connectionMessage, 'success');
        } catch (error) {
            console.error('Dataset connection test failed:', error);
            dataset._connectionStatus = 'error';
            dataset._connectionMessage = error?.message || 'Connection test failed.';
            dataset._connectionDetails = null;
            window.showToast(dataset._connectionMessage, 'error');
        } finally {
            dataset._connectionTesting = false;
        }
    },
    syncGeneratorSelections(dataset) {
        if (!dataset) {
            return;
        }
        this.ensureDatasetRuntimeState(dataset);
        if (!Array.isArray(dataset.conditioning)) {
            dataset.conditioning = [];
        }
        dataset.conditioning.forEach((generator, index) => {
            if (!generator) {
                dataset.conditioning[index] = {};
                generator = dataset.conditioning[index];
            }
            const type = typeof generator.type === 'string' ? generator.type.trim() : '';
            if (this.isKnownGeneratorType(type)) {
                generator._selectedType = type;
                generator._customType = '';
            } else if (type) {
                generator._selectedType = '__custom__';
                generator._customType = type;
            } else {
                generator._selectedType = '';
                generator._customType = '';
            }
        });
    },
    datasetOptionsForType(dataset, type) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !Array.isArray(trainer.datasets)) {
            return [];
        }
        return trainer.datasets
            .filter((entry) => entry && entry.id && entry.dataset_type === type && entry.id !== dataset?.id)
            .map((entry) => ({
                id: entry.id,
                label: `${entry.id}${entry.dataset_type ? ` (${entry.dataset_type})` : ''}`
            }));
    },
    renderDatasetOptions(dataset, type, emptyLabel) {
        const options = this.datasetOptionsForType(dataset, type);
        let html = `<option value="">${emptyLabel}</option>`;
        options.forEach(option => {
            html += `<option value="${this.escapeHtml(option.id)}">${this.escapeHtml(option.label)}</option>`;
        });
        return html;
    },
    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    },
    getGeneratorOption(value) {
        return CONDITIONING_GENERATOR_TYPES.find((option) => option.value === value) || null;
    },
    getGeneratorDescription(value) {
        const option = this.getGeneratorOption(value);
        return option ? option.description || '' : '';
    },
    isKnownGeneratorType(value) {
        if (!value) {
            return false;
        }
        return CONDITIONING_GENERATOR_TYPES.some((option) => option.value === value && option.value !== '__custom__');
    },
    onGeneratorTypeSelect(generator) {
        if (!generator) {
            return;
        }
        const trainer = Alpine.store('trainer');
        if (generator._selectedType === '__custom__') {
            if (!generator._customType) {
                generator.type = '';
            } else {
                generator.type = generator._customType.trim();
            }
        } else {
            generator.type = generator._selectedType || '';
            generator._customType = '';
        }
        if (trainer && typeof trainer.markDatasetsDirty === 'function') {
            trainer.markDatasetsDirty();
        }
    },
    onGeneratorTypeManual(generator) {
        if (!generator) {
            return;
        }
        const trainer = Alpine.store('trainer');
        const value = generator._customType ? generator._customType.trim() : '';
        generator._customType = value;
        generator.type = value;
        generator._selectedType = '__custom__';
        if (trainer && typeof trainer.markDatasetsDirty === 'function') {
            trainer.markDatasetsDirty();
        }
    },
    conditioningOptions(dataset) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !dataset) {
            return [];
        }
        const options = trainer.getConditioningDatasetIds ? trainer.getConditioningDatasetIds(dataset.id) : [];
        const current = Array.isArray(dataset.conditioning_data) ? dataset.conditioning_data : [];
        if (!this.conditioningSupported && dataset.dataset_type !== 'conditioning') {
            return [...new Set(current)];
        }
        return [...new Set([...options, ...current])];
    },
    datasetSafeId(dataset) {
        if (!dataset || !dataset.id) {
            return 'dataset';
        }
        return dataset.id.toString().replace(/[^a-zA-Z0-9_-]/g, '-');
    },
    normalizeBoolean(value) {
        if (value === null || value === undefined) {
            return false;
        }
        if (typeof value === 'boolean') {
            return value;
        }
        if (typeof value === 'number') {
            return value !== 0;
        }
        if (typeof value === 'string') {
            const normalized = value.trim().toLowerCase();
            if (!normalized) {
                return false;
            }
            if (['true', '1', 'yes', 'on'].includes(normalized)) {
                return true;
            }
            if (['false', '0', 'no', 'off'].includes(normalized)) {
                return false;
            }
        }
        return Boolean(value);
    },
    shouldShowConditioningSection(dataset) {
        if (!dataset) {
            return false;
        }
        const type = dataset.dataset_type;
        const hasLinks = Array.isArray(dataset.conditioning_data) && dataset.conditioning_data.length > 0;
        const hasGenerators = Array.isArray(dataset.conditioning) && dataset.conditioning.length > 0;
        const featuresActive = this.conditioningSupported;

        if (!featuresActive && type !== 'conditioning' && !hasLinks && !hasGenerators) {
            return false;
        }

        if (type === 'conditioning') {
            return true;
        }

        if (type === 'image' || type === 'video') {
            return featuresActive || hasLinks || hasGenerators;
        }

        return hasLinks || hasGenerators;
    },
    collapseSectionKey(dataset, section) {
        if (!dataset || !dataset.id) {
            return null;
        }
        return `${dataset.id}::${section}`;
    },
    sectionAvailable(dataset, section) {
        if (!dataset) {
            return false;
        }
        switch (section) {
            case 'sizing':
            case 'cropping':
                return dataset.dataset_type !== 'text_embeds' && dataset.dataset_type !== 'image_embeds';
            case 'captions':
                return !['text_embeds', 'image_embeds'].includes(dataset.dataset_type);
            case 'video':
                return dataset.dataset_type === 'video';
            case 'conditioning':
                return this.shouldShowConditioningSection(dataset);
            default:
                return true;
        }
    },
    sectionIsCollapsed(dataset, section) {
        if (!dataset) {
            return false;
        }
        if (section === 'card') {
            return !!dataset._collapsed;
        }
        const prop = DATASET_COLLAPSE_SECTION_MAP[section];
        if (!prop) {
            return false;
        }
        const value = dataset[prop];
        if (prop.startsWith('_show')) {
            if (typeof value === 'boolean') {
                return !value;
            }
            return false;
        }
        return !value;
    },
    sectionIsExpanded(dataset, section) {
        return !this.sectionIsCollapsed(dataset, section);
    },
    setSectionCollapsed(dataset, section, collapsed, options = {}) {
        if (!dataset) {
            return;
        }
        const prop = DATASET_COLLAPSE_SECTION_MAP[section];
        if (!prop) {
            return;
        }
        const shouldCollapse = !!collapsed;
        if (section === 'card') {
            dataset._collapsed = shouldCollapse;
        } else if (prop.startsWith('_show')) {
            dataset[prop] = !shouldCollapse;
        } else {
            dataset[prop] = !shouldCollapse;
        }

        if (!this._collapsedState) {
            this._collapsedState = {};
        }

        const key = this.collapseSectionKey(dataset, section);
        if (key) {
            if (this.sectionAvailable(dataset, section)) {
                this._collapsedState[key] = shouldCollapse;
            } else {
                delete this._collapsedState[key];
            }
        }

        if (!options.silent && this._collapseStateLoaded) {
            this.queueCollapsedStateSave();
        }
    },
    toggleSectionCollapsed(dataset, section) {
        const collapsed = this.sectionIsCollapsed(dataset, section);
        this.setSectionCollapsed(dataset, section, !collapsed);
    },
    collectCollapsedState() {
        const state = {};
        if (!Array.isArray(this.datasets)) {
            return state;
        }
        this.datasets.forEach((dataset) => {
            if (!dataset || !dataset.id) {
                return;
            }
            Object.keys(DATASET_COLLAPSE_SECTION_MAP).forEach((section) => {
                if (!this.sectionAvailable(dataset, section)) {
                    return;
                }
                const key = this.collapseSectionKey(dataset, section);
                if (key) {
                    state[key] = this.sectionIsCollapsed(dataset, section);
                }
            });
        });
        return state;
    },
    queueCollapsedStateSave() {
        if (!this._collapseStateLoaded) {
            return;
        }
        if (this._collapseSaveTimeout) {
            clearTimeout(this._collapseSaveTimeout);
        }
        this._collapseSaveTimeout = setTimeout(() => {
            this.saveCollapsedState();
        }, 200);
    },
    async saveCollapsedState() {
        if (!this._collapseStateLoaded) {
            return;
        }
        const sections = this.collectCollapsedState();
        try {
            await fetch(COLLAPSED_SECTIONS_ENDPOINT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sections })
            });
            this._collapsedState = { ...sections };
        } catch (error) {
            console.warn('[DatasetBuilder] Failed to save collapsed state', error);
        }
        if (this._collapseSaveTimeout) {
            clearTimeout(this._collapseSaveTimeout);
            this._collapseSaveTimeout = null;
        }
    },
    async loadCollapsedState() {
        if (this._collapseStateLoading) {
            return;
        }
        this._collapseStateLoading = true;
        try {
            const response = await fetch(COLLAPSED_SECTIONS_ENDPOINT);
            if (response.ok) {
                const data = await response.json();
                if (data && typeof data === 'object') {
                    this._collapsedState = data;
                } else {
                    this._collapsedState = {};
                }
            } else {
                this._collapsedState = {};
            }
        } catch (error) {
            console.warn('[DatasetBuilder] Failed to load collapsed state', error);
            this._collapsedState = {};
        }
        this._collapseStateLoading = false;
        this._collapseStateLoaded = true;
        this.applyCollapsedStateAll();
    },
    applyCollapsedStateAll() {
        if (!Array.isArray(this.datasets)) {
            return;
        }
        this.datasets.forEach((dataset) => this.applyCollapsedStateToDataset(dataset));
    },
    applyCollapsedStateToDataset(dataset) {
        if (!dataset || !dataset.id || !this._collapsedState) {
            return;
        }
        Object.keys(DATASET_COLLAPSE_SECTION_MAP).forEach((section) => {
            if (!this.sectionAvailable(dataset, section)) {
                return;
            }
            const key = this.collapseSectionKey(dataset, section);
            if (!key || !(key in this._collapsedState)) {
                return;
            }
            this.setSectionCollapsed(dataset, section, !!this._collapsedState[key], { silent: true });
        });
    },
    onConditioningDataChange(dataset, event) {
        const trainer = Alpine.store('trainer');
        if (event && event.target && event.target.options) {
            const selected = Array.from(event.target.options)
                .filter((option) => option.selected)
                .map((option) => option.value)
                .filter((value) => typeof value === 'string' && value.trim() !== '')
                .map((value) => value.trim());
            dataset.conditioning_data = selected;
        }
        if (trainer && typeof trainer.markDatasetsDirty === 'function') {
            trainer.markDatasetsDirty();
        }
    },
    addConditioningGenerator(dataset) {
        const trainer = Alpine.store('trainer');
        if (!trainer || typeof trainer.addConditioningGenerator !== 'function') {
            window.showToast('Conditioning tools not ready. Please wait and try again.', 'error');
            return;
        }
        trainer.addConditioningGenerator(dataset);
        this.syncGeneratorSelections(dataset);
    },
    removeConditioningGenerator(dataset, generator) {
        const trainer = Alpine.store('trainer');
        if (!trainer || typeof trainer.removeConditioningGenerator !== 'function') {
            window.showToast('Conditioning tools not ready. Please wait and try again.', 'error');
            return;
        }
        const key = generator?._uiKey || generator?.id;
        trainer.removeConditioningGenerator(dataset, key);
    },
    onGeneratorTypeChange(generator) {
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.markDatasetsDirty === 'function') {
            trainer.markDatasetsDirty();
        }
    },
    onGeneratorParamsBlur(generator) {
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.updateConditioningParams === 'function') {
            trainer.updateConditioningParams(generator, generator._paramsText || '{}');
        }
    },
    onGeneratorCaptionsModeChange(generator) {
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.setConditioningCaptionsMode === 'function') {
            trainer.setConditioningCaptionsMode(generator, generator._captionsMode || 'inherit');
        }
    },
    onGeneratorCaptionsInput(generator) {
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.updateConditioningCaptionsValue === 'function') {
            trainer.updateConditioningCaptionsValue(generator, generator._captionsValue || '');
        }
    },

    // Direct method calls with error handling
    async saveDatasets() {
        const trainer = Alpine.store('trainer');
        if (!trainer) {
            window.showToast('Dataset builder not ready. Please wait and try again.', 'error');
            return;
        }

        try {
            // Direct save without going through the trainer's dialog system
            const payloadDatasets = trainer.prepareDatasetsForSave();
            if (trainer.datasetValidationErrors && Object.keys(trainer.datasetValidationErrors).length > 0) {
                window.showToast('Resolve dataset validation errors before saving.', 'error');
                return;
            }
            const response = await fetch('/api/datasets/plan', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    datasets: payloadDatasets,
                    createBackup: false
                })
            });

            if (response.ok) {
                trainer.hasUnsavedChanges = false;
                trainer.datasetValidationErrors = {};
                const result = await response.json();
                let message = 'Dataset configuration saved';
                if (result.backupPath) {
                    message += ` (backup: ${result.backupPath})`;
                }
                window.showToast(message, 'success');
                trainer.refreshDatasetsJson();
            } else {
                const error = await response.json();
                if (error.detail && error.detail.validations) {
                    // Map validation errors to field-specific errors
                    const fieldErrors = {};
                    const errorMessages = [];
                    error.detail.validations.forEach(validation => {
                        if (validation.field && validation.field.includes('.')) {
                            fieldErrors[validation.field] = validation.message;
                            // Extract dataset ID and field name for display
                            const parts = validation.field.split('.');
                            const datasetId = parts[0];
                            const fieldName = parts.slice(1).join('.');
                            errorMessages.push(`${datasetId}: ${validation.message}`);
                        }
                    });
                    trainer.datasetValidationErrors = fieldErrors;

                    // Show detailed error messages
                    if (errorMessages.length > 0) {
                        const errorList = errorMessages.map(msg => `• ${msg}`).join('<br>');
                        const errorHtml = `<div style="text-align: left;"><strong>Validation errors:</strong><br>${errorList}</div>`;
                        window.showToast(errorHtml, 'error', 8000);
                    } else {
                        window.showToast('Failed to save: validation errors', 'error');
                    }
                } else {
                    window.showToast(`Failed to save: ${error.detail || 'Unknown error'}`, 'error');
                }
            }
        } catch (error) {
            console.error('Error saving datasets:', error);
            window.showToast('Failed to save dataset configuration', 'error');
        }
    },

    onAddDatasetClick(type) {
        if (type === 'video' && !this.isVideoModel) {
            const confirmed = window.confirm('This model is not marked as video-capable. Add a video dataset anyway?');
            if (!confirmed) {
                return;
            }
        }
        this.addDataset(type);
    },

    addDataset(type) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !trainer.addDataset) {
            window.showToast('Dataset builder not ready. Please wait and try again.', 'error');
            return;
        }
        trainer.addDataset(type);
    },

    removeDataset(datasetId) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !trainer.removeDataset) {
            window.showToast('Cannot remove dataset. Please wait and try again.', 'error');
            return;
        }
        trainer.removeDataset(datasetId);
    },

    duplicateDataset(datasetId) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !trainer.duplicateDataset) {
            window.showToast('Cannot duplicate dataset. Please wait and try again.', 'error');
            return;
        }
        trainer.duplicateDataset(datasetId);
    },

    switchDataLoaderMode(mode) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !trainer.switchDataLoaderMode) {
            window.showToast('Cannot switch mode. Please wait and try again.', 'error');
            return;
        }
        trainer.switchDataLoaderMode(mode);
    },

    loadPreset(presetName) {
        const trainer = Alpine.store('trainer');
        if (!trainer || !trainer.loadPreset) {
            window.showToast('Cannot load preset. Please wait and try again.', 'error');
            return;
        }
        trainer.loadPreset(presetName);
    },

    hasFieldError(datasetId, fieldName) {
        const trainer = Alpine.store('trainer');
        return trainer && trainer.hasFieldError ? trainer.hasFieldError(datasetId, fieldName) : false;
    },

    getFieldError(datasetId, fieldName) {
        const trainer = Alpine.store('trainer');
        return trainer && trainer.getFieldError ? trainer.getFieldError(datasetId, fieldName) : '';
    },

    getDatasetTypeIcon(type) {
        const trainer = Alpine.store('trainer');
        return trainer && trainer.getDatasetTypeIcon ? trainer.getDatasetTypeIcon(type) : '';
    },

    async saveJsonDatasets() {
        const trainer = Alpine.store('trainer');
        if (!trainer) {
            window.showToast('Dataset builder not ready. Please wait and try again.', 'error');
            return;
        }

        try {
            // Parse the JSON
            const parsedDatasets = JSON.parse(this.dataLoaderJson);

            // Update trainer datasets
            trainer.datasets = parsedDatasets;

            // Save using the direct method
            await this.saveDatasets();
        } catch (e) {
            window.showToast('Invalid JSON format. Please fix the JSON and try again.', 'error');
        }
    },
    onCaptionFilterSelect(dataset) {
        if (!dataset) {
            return;
        }
        const trainer = Alpine.store('trainer');
        if (dataset._selectedCaptionFilter === '__custom__') {
            // Leave existing path/value as-is for manual entry
        } else if (!dataset._selectedCaptionFilter) {
            dataset.caption_filter_list = '';
        } else {
            dataset.caption_filter_list = dataset._selectedCaptionFilter;
        }
        if (trainer && typeof trainer.markDatasetsDirty === 'function') {
            trainer.markDatasetsDirty();
        }
    },
    onStorageBackendChange(dataset) {
        if (!dataset) {
            return;
        }
        const previousStrategy = dataset.caption_strategy;
        const previousBackend = dataset.metadata_backend;

        if (dataset.type === 'csv') {
            if (!previousStrategy || previousStrategy === 'textfile') {
                dataset.caption_strategy = 'csv';
            }
        } else if (dataset.type === 'huggingface') {
            if (!previousStrategy || previousStrategy === 'textfile') {
                dataset.caption_strategy = 'huggingface';
            }
            if (!previousBackend || previousBackend === '' || previousBackend === 'discovery') {
                dataset.metadata_backend = 'huggingface';
            }
        }

        this.markAsUnsaved();
    },
    openDatasetPathBrowser(dataset, fieldId) {
        if (!dataset || !fieldId) {
            return;
        }
        const wizard = window.datasetWizardComponentInstance;
        if (!wizard || typeof wizard.openFileBrowser !== 'function') {
            if (window.showToast) {
                window.showToast('File browser is not ready yet. Try again in a moment.', 'warning');
            } else {
                console.warn('Dataset file browser unavailable');
            }
            return;
        }
        const maybePromise = wizard.openFileBrowser(fieldId, {
            dataset,
            context: 'manual',
            initialPath: dataset[fieldId] || null,
            onConfirm: () => {
                this.markAsUnsaved();
            }
        });
        if (maybePromise && typeof maybePromise.catch === 'function') {
            maybePromise.catch((error) => {
                console.error('Failed to open dataset file browser', error);
                if (window.showToast) {
                    window.showToast('Unable to open file browser. Check console for details.', 'error');
                }
            });
        }
    },
    computeDangerModeEnabled() {
        const toggle = document.getElementById('i_know_what_i_am_doing');
        if (toggle) {
            return toggle.checked;
        }
        const trainer = Alpine.store('trainer');
        if (trainer && typeof trainer.isDangerModeEnabled === 'function') {
            try {
                return !!trainer.isDangerModeEnabled();
            } catch (error) {
                console.debug('Danger mode probe failed', error);
            }
        }
        return false;
    },
    skipDiscoveryOptions() {
        return SKIP_DISCOVERY_OPTIONS;
    },
    hasSkipDiscovery(dataset, option) {
        if (!dataset || !option) {
            return false;
        }
        const tokens = Array.isArray(dataset._skipDiscovery)
            ? dataset._skipDiscovery
            : this.parseSkipDiscovery(dataset.skip_file_discovery);
        return tokens.includes(option);
    },
    setSkipDiscovery(dataset, option, enabled) {
        if (!dataset || !option) {
            return;
        }
        const normalized = option.trim();
        let tokens = Array.isArray(dataset._skipDiscovery)
            ? [...dataset._skipDiscovery]
            : this.parseSkipDiscovery(dataset.skip_file_discovery);
        tokens = tokens.filter((token) => token !== normalized);
        if (enabled) {
            tokens.push(normalized);
        }
        dataset._skipDiscovery = tokens;
        this.updateSkipDiscoveryString(dataset);
        this.markAsUnsaved();
    },
    updateSkipDiscoveryString(dataset) {
        if (!dataset) {
            return;
        }
        const tokens = Array.isArray(dataset._skipDiscovery)
            ? dataset._skipDiscovery
            : this.parseSkipDiscovery(dataset.skip_file_discovery);
        const unique = Array.from(new Set(tokens.map((token) => token.trim()).filter((token) => token.length > 0)));
        dataset._skipDiscovery = unique;
        dataset.skip_file_discovery = unique.join(' ');
    },
    parseSkipDiscovery(value) {
        if (Array.isArray(value)) {
            return value.map((token) => (typeof token === 'string' ? token.trim() : String(token || '')).trim())
                .filter((token) => token.length > 0);
        }
        if (typeof value !== 'string') {
            return [];
        }
        return value.split(/[,\s]+/)
            .map((token) => token.trim())
            .filter((token) => token.length > 0);
    },
    dangerModeEnabled() {
        return !!this.dangerMode;
    },

    markAsUnsaved() {
        if (!this._markAsUnsavedDebounce) {
            this._markAsUnsavedDebounce = requestAnimationFrame(() => {
                const trainer = Alpine.store('trainer');
                if (trainer) {
                    if (typeof trainer.markDatasetsDirty === 'function') {
                        trainer.markDatasetsDirty({ refresh: this.dataLoaderMode !== 'json' });
                    } else {
                        trainer.hasUnsavedChanges = true;
                    }
                }
                this._markAsUnsavedDebounce = null;
            });
        }
    }
    }
}
window.dataloaderSectionComponent = dataloaderSectionComponent;
})();
