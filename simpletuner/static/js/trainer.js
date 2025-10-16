// Main trainer functionality with proper CORS handling
(function() {
    // Skip if already defined
    if (window.TrainerUI) {
        console.warn('TrainerUI already defined, skipping');
        return;
    }

    class TrainerUI {
    constructor() {
        this.form = null; // Will be set after DOM is loaded
        this.apiBaseUrl = null; // Will be set after config detection
        this.callbackUrl = null; // Will be set after config detection
        this.modelFlavours = {}; // Cache model flavour responses by family
        this.modelDetails = {}; // Cache model metadata keyed by family
        this._modelDetailsRequestToken = null;
        this.fsdpModal = null;
        this.fsdpDetectionData = null;
        this.init();
    }

    async init() {
        // Wait for server configuration to be ready
        await window.ServerConfig.waitForReady();

        // Set URLs based on detected configuration
        this.apiBaseUrl = window.ApiClient ? window.ApiClient.apiBaseUrl : window.ServerConfig.apiBaseUrl;
        this.callbackUrl = window.ApiClient ? window.ApiClient.callbackBaseUrl : window.ServerConfig.callbackUrl;
        // TrainerUI configured with API and callback URLs

        // Cache root form reference for payload serialization
        this.form = document.getElementById('trainer-form');

        // Listen for future HTMX swaps so we can re-bind dynamic behaviours
        this.registerHTMXHandlers();

        // Initialize dependency manager
        await this.initializeDependencyManager();

        this.setupTabNavigation();
        this.setupFieldDependencies();
        this.setupButtonHandlers();
        this.setupJSONEditors();
        this.setupSidebarToggle();
        this.updateWebhookConfig();
    }

    async initializeDependencyManager() {
        try {
            // Initialize the dependency manager
            await window.dependencyManager.initialize();

            // Only register fields if we have a container
            const container = document.querySelector('.tab-fragment') || document.body;
            if (container) {
                window.dependencyManager.initializeFieldsInContainer(container);
            }

            // Listen for field changes
            window.addEventListener('fieldChanged', (e) => {
                console.log('Field changed:', e.detail.field, '=', e.detail.value);
            });
        } catch (error) {
            console.warn('Dependency manager initialization failed:', error);
            // Continue without dependency manager
        }
    }

    registerHTMXHandlers() {
        if (!window.htmx || this._htmxHandlersRegistered) {
            return;
        }

        document.body.addEventListener('htmx:afterSwap', (event) => {
            const target = event.target;
            if (!target) {
                return;
            }

            // Refresh cached form reference if it was swapped
            if (target.id === 'trainer-form' || (typeof target.closest === 'function' && target.closest('#trainer-form'))) {
                this.form = document.getElementById('trainer-form');
            }

            // Rebind behaviours when a trainer tab fragment updates
            let fragment = null;
            if (typeof target.matches === 'function' && target.matches('.tab-fragment')) {
                fragment = target;
            } else if (typeof target.querySelector === 'function') {
                fragment = target.querySelector('.tab-fragment');
            }

            if (fragment) {
                if (window.dependencyManager && typeof window.dependencyManager.initializeFieldsInContainer === 'function') {
                    window.dependencyManager.initializeFieldsInContainer(fragment);
                }

                this.setupFieldDependencies();
                this.setupJSONEditors();
                this.setupButtonHandlers();
            }
        });

        this._htmxHandlersRegistered = true;
    }

    setupSidebarToggle() {
        const toggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');

        if (toggle && sidebar) {
            toggle.addEventListener('click', () => {
                sidebar.classList.toggle('show');
            });

            // Close sidebar when clicking outside on mobile
            document.addEventListener('click', (e) => {
                if (window.innerWidth < 992 &&
                    !sidebar.contains(e.target) &&
                    !toggle.contains(e.target) &&
                    sidebar.classList.contains('show')) {
                    sidebar.classList.remove('show');
                }
            });
        }
    }

    updateWebhookConfig() {
        // Update the webhook configuration with the detected callback URL
        const webhookTextarea = document.getElementById('webhook_config');
        if (webhookTextarea && window.ServerConfig.mode === 'unified') {
            try {
                const config = JSON.parse(webhookTextarea.value);
                config.callback_url = `${this.callbackUrl}/callback`;
                webhookTextarea.value = JSON.stringify(config, null, 4);
                // Updated webhook config for unified mode
            } catch (e) {
                console.error('Failed to update webhook config:', e);
            }
        }
    }

    setupTabNavigation() {
        const tabButtons = document.querySelectorAll('.tab-btn');

        if (!tabButtons.length) {
            return;
        }

        tabButtons.forEach((button) => {
            if (button.dataset.navigationBound) {
                return;
            }

            button.addEventListener('click', () => {
                tabButtons.forEach((otherButton) => {
                    if (otherButton !== button) {
                        otherButton.classList.remove('active');
                    }
                });

                button.classList.add('active');
            });

            button.dataset.navigationBound = 'true';
        });
    }

    setupFieldDependencies() {
        this.initializeModelFlavours();

        const modelType = document.getElementById('model_type');
        const modelFamily = document.getElementById('model_family');
        const loraType = document.getElementById('lora_type');

        const updateDependencies = async () => {
            const modelTypeValue = modelType ? modelType.value : undefined;
            const modelFamilyValue = modelFamily ? modelFamily.value : undefined;
            const loraTypeValue = loraType ? loraType.value : undefined;

            this.updateLoraVisibility(modelTypeValue);
            this.updateFluxLoraVisibility(modelFamilyValue, modelTypeValue);
            this.updateLycorisVisibility(loraTypeValue, modelTypeValue);
            this.updateQuantizationAvailability(modelTypeValue);
            await this.updateModelFlavours(modelFamilyValue);
            await this.updateTextEncoderCapabilities(modelFamilyValue);
        };

        if (modelType && !modelType.dataset.dependenciesBound) {
            modelType.addEventListener('change', updateDependencies);
            modelType.dataset.dependenciesBound = 'true';
        }

        if (modelFamily && !modelFamily.dataset.dependenciesBound) {
            modelFamily.addEventListener('change', updateDependencies);
            modelFamily.dataset.dependenciesBound = 'true';
        }

        if (loraType && !loraType.dataset.dependenciesBound) {
            loraType.addEventListener('change', updateDependencies);
            loraType.dataset.dependenciesBound = 'true';
        }

        updateDependencies();
        this.observeDangerModeToggle();
        this.enforceDangerModeGates();
    }

    initializeModelFlavours() {
        if (!this.modelFlavours) {
            this.modelFlavours = {};
        }
    }

    async fetchModelDetails(modelFamily) {
        if (!modelFamily) {
            return null;
        }

        if (this.modelDetails[modelFamily]) {
            return this.modelDetails[modelFamily];
        }

        try {
            const response = await fetch(`${this.callbackUrl}/api/models/${modelFamily}`);
            if (!response.ok) {
                throw new Error(`Failed to fetch details for ${modelFamily}`);
            }
            const data = await response.json();
            this.modelDetails[modelFamily] = data;
            return data;
        } catch (error) {
            console.error(`Error fetching details for ${modelFamily}:`, error);
            return null;
        }
    }

    setFieldVisibility(fieldName, visible) {
        const manager = window.dependencyManager;
        if (manager && typeof manager.setFieldVisibility === 'function') {
            const state = manager.fieldStates?.get?.(fieldName);
            if (state) {
                manager.setFieldVisibility(fieldName, visible);
                return;
            }
        }

        const element = document.getElementById(fieldName) || document.querySelector(`[name="${fieldName}"]`);
        if (!element) {
            return;
        }

        const container = element.closest('.mb-3') || element.closest('.field-wrapper') || element.parentElement;
        if (container) {
            container.style.display = visible ? '' : 'none';
        }

        element.disabled = !visible;
    }

    configHasField(fieldName) {
        const trainerStore = window.Alpine?.store?.('trainer');
        if (!trainerStore) {
            return false;
        }

        const rawConfig = trainerStore.activeEnvironmentConfig || {};
        if (!rawConfig || typeof rawConfig !== 'object') {
            return false;
        }

        const variants = new Set();
        const baseName = fieldName.startsWith('--') ? fieldName.slice(2) : fieldName;
        variants.add(fieldName);
        variants.add(`--${baseName}`);
        variants.add(baseName);
        variants.add(baseName.toLowerCase());
        variants.add(baseName.replace(/_/g, '').toLowerCase());

        for (const key of variants) {
            if (Object.prototype.hasOwnProperty.call(rawConfig, key)) {
                const value = rawConfig[key];
                if (value === null || value === undefined) {
                    continue;
                }
                if (typeof value === 'string') {
                    if (value.trim() === '') {
                        continue;
                    }
                    return true;
                }
                return true;
            }
        }

        return false;
    }

    hasPersistedValue(fieldName) {
        if (this.configHasField(fieldName)) {
            return true;
        }

        const manager = window.dependencyManager;
        if (!manager || !manager.fieldStates) {
            return false;
        }

        const state = manager.fieldStates.get(fieldName);
        if (!state) {
            return false;
        }

        const value = state.value;
        if (typeof value === 'string') {
            return value.trim() !== '';
        }
        return Boolean(value);
    }

    async updateTextEncoderCapabilities(modelFamily) {
        const requestToken = Symbol('model-details');
        this._modelDetailsRequestToken = requestToken;

        const details = await this.fetchModelDetails(modelFamily);

        if (this._modelDetailsRequestToken !== requestToken) {
            return; // A newer request superseded this one
        }

        if (modelFamily && !details) {
            return; // Preserve existing visibility when metadata lookup fails
        }

        let supportsTextEncoder = false;
        let encoderCount = 0;

        if (details && details.attributes) {
            supportsTextEncoder = Boolean(details.attributes.supports_text_encoder_training);
            const configuration = details.attributes.text_encoder_configuration;
            if (configuration && typeof configuration === 'object') {
                encoderCount = Object.keys(configuration).length;
            }
        }

        if (supportsTextEncoder && encoderCount === 0) {
            encoderCount = 1;
        }

        const precisionFields = [
            'text_encoder_1_precision',
            'text_encoder_2_precision',
            'text_encoder_3_precision',
            'text_encoder_4_precision',
        ];
        const trainingFields = ['train_text_encoder', 'text_encoder_lr'];

        const shouldShowTraining = supportsTextEncoder;
        trainingFields.forEach((fieldName) => {
            this.setFieldVisibility(fieldName, shouldShowTraining || this.hasPersistedValue(fieldName));
        });

        precisionFields.forEach((fieldName, index) => {
            const withinSupportedRange = supportsTextEncoder && index < encoderCount;
            const hasValue = this.hasPersistedValue(fieldName);
            this.setFieldVisibility(fieldName, withinSupportedRange || hasValue);
        });
    }

    async fetchModelFlavours(modelFamily) {
        if (!modelFamily) {
            return [];
        }

        // Check cache first
        if (this.modelFlavours[modelFamily]) {
            return this.modelFlavours[modelFamily];
        }

        try {
            const response = await fetch(`${this.callbackUrl}/api/models/${modelFamily}/flavours`);
            if (!response.ok) {
                throw new Error(`Failed to fetch flavours for ${modelFamily}`);
            }
            const data = await response.json();
            // Cache the flavours - just use the raw values from backend
            this.modelFlavours[modelFamily] = data.flavours.map(flavour => ({
                value: flavour,
                label: flavour,
                path: '' // Path will be determined by backend based on flavour
            }));
            return this.modelFlavours[modelFamily];
        } catch (error) {
            console.error(`Error fetching flavours for ${modelFamily}:`, error);
            return [];
        }
    }

    async updateModelFlavours(modelFamily) {
        const flavourSelect = document.getElementById('model_flavour');

        if (!flavourSelect) {
            return;
        }

        const wrapper = flavourSelect.closest('.mb-3') || flavourSelect.parentElement;
        const initialValue = flavourSelect.dataset.currentValue
            || flavourSelect.dataset.initialValue
            || flavourSelect.value
            || '';

        if (!modelFamily) {
            flavourSelect.innerHTML = '<option value="">Select a model family first</option>';
            flavourSelect.disabled = true;
            if (wrapper) {
                wrapper.classList.remove('d-none');
            }
            return;
        }

        flavourSelect.disabled = true;
        flavourSelect.innerHTML = '<option value="">Loading flavours...</option>';

        const flavours = await this.fetchModelFlavours(modelFamily);

        // Reset options regardless of result to avoid stale entries
        flavourSelect.innerHTML = '<option value="">Default</option>';

        if (!Array.isArray(flavours) || flavours.length === 0) {
            flavourSelect.innerHTML = '<option value="">No flavours available</option>';
            flavourSelect.disabled = true;
            flavourSelect.dataset.currentValue = '';
            return;
        }

        const normalizedFlavours = flavours.map((entry) => (
            typeof entry === 'string'
                ? { value: entry, label: entry, path: '' }
                : entry
        ));

        normalizedFlavours.forEach((flavour) => {
            const option = document.createElement('option');
            option.value = flavour.value;
            option.textContent = flavour.label || flavour.value;
            if (flavour.path) {
                option.dataset.path = flavour.path;
            }
            flavourSelect.appendChild(option);
        });

        if (!flavourSelect.dataset.listenerAdded) {
            flavourSelect.addEventListener('change', (event) => {
                flavourSelect.dataset.currentValue = event.target.value || '';

                const selectedOption = event.target.selectedOptions[0];
                const modelPath = document.getElementById('model_path');

                if (selectedOption && selectedOption.dataset.path && modelPath) {
                    modelPath.value = selectedOption.dataset.path;
                    modelPath.dataset.autofilled = 'true';
                } else if (!event.target.value && modelPath && modelPath.dataset.autofilled === 'true') {
                    modelPath.value = '';
                    delete modelPath.dataset.autofilled;
                }
            });
            flavourSelect.dataset.listenerAdded = 'true';
        }

        let targetValue = '';

        if (initialValue && normalizedFlavours.some((flavour) => flavour.value === initialValue)) {
            targetValue = initialValue;
        } else if (normalizedFlavours.length === 1) {
            targetValue = normalizedFlavours[0].value;
        } else {
            const details = await this.fetchModelDetails(modelFamily);
            const defaultFlavour = details?.default_flavour
                || details?.attributes?.default_model_flavour
                || '';

            if (defaultFlavour && normalizedFlavours.some((flavour) => flavour.value === defaultFlavour)) {
                targetValue = defaultFlavour;
            }
        }

        if (targetValue) {
            flavourSelect.value = targetValue;
            flavourSelect.dataset.currentValue = targetValue;

            const selectedOption = flavourSelect.selectedOptions[0];
            const modelPath = document.getElementById('model_path');

            if (selectedOption && selectedOption.dataset.path && modelPath) {
                modelPath.value = selectedOption.dataset.path;
                modelPath.dataset.autofilled = 'true';
            }
        } else {
            flavourSelect.value = '';
            flavourSelect.dataset.currentValue = '';

            const modelPath = document.getElementById('model_path');
            if (modelPath && modelPath.dataset.autofilled === 'true') {
                modelPath.value = '';
                delete modelPath.dataset.autofilled;
            }
        }

        flavourSelect.disabled = false;
    }

    observeDangerModeToggle() {
        const dangerToggle = document.getElementById('i_know_what_i_am_doing');
        if (dangerToggle && !dangerToggle.dataset.gateListener) {
            dangerToggle.addEventListener('change', () => {
                this.enforceDangerModeGates();
            });
            dangerToggle.dataset.gateListener = 'true';
        }
    }

    isDangerModeEnabled() {
        const dangerToggle = document.getElementById('i_know_what_i_am_doing');
        if (dangerToggle) {
            return dangerToggle.checked;
        }

        const trainerStore = (window.Alpine && typeof window.Alpine.store === 'function')
            ? window.Alpine.store('trainer')
            : undefined;
        const descriptor = trainerStore?.formValueStore?.['--i_know_what_i_am_doing']
            || trainerStore?.formValueStore?.['i_know_what_i_am_doing'];

        if (descriptor) {
            if (descriptor.kind === 'checkbox') {
                const values = Array.isArray(descriptor.value) ? descriptor.value : [];
                return values.some((value) => {
                    if (typeof value === 'string') {
                        return ['true', 'on', '1'].includes(value.toLowerCase());
                    }
                    return Boolean(value);
                });
            }
            if (descriptor.kind === 'single') {
                if (typeof descriptor.value === 'string') {
                    return ['true', 'on', '1'].includes(descriptor.value.toLowerCase());
                }
                return Boolean(descriptor.value);
            }
        }

        const fragment = document.querySelector('.tab-fragment');
        if (fragment && typeof fragment.dataset.dangerMode !== 'undefined') {
            const value = fragment.dataset.dangerMode;
            return value === 'true' || value === '1';
        }

        return false;
    }

    enforceDangerModeGates() {
        const predictionSelect = document.getElementById('prediction_type');
        if (!predictionSelect) {
            return;
        }

        const enabled = this.isDangerModeEnabled();
        predictionSelect.disabled = !enabled;

        const wrapper = predictionSelect.closest('.mb-3') || predictionSelect.parentElement;
        if (wrapper) {
            wrapper.classList.toggle('danger-mode-locked', !enabled);

            let hint = wrapper.querySelector('.danger-mode-hint');
            if (!hint) {
                hint = document.createElement('div');
                hint.className = 'form-text text-warning danger-mode-hint';
                hint.textContent = 'Enable "I Know What I\'m Doing" in Advanced settings to customise prediction type.';
                wrapper.appendChild(hint);
            }
            hint.style.display = enabled ? 'none' : '';
        }

        const fragment = document.querySelector('.tab-fragment');
        if (fragment) {
            fragment.dataset.dangerMode = enabled ? 'true' : 'false';
        }

        try {
            window.dispatchEvent(new CustomEvent('trainer-danger-mode-changed', {
                detail: { enabled }
            }));
        } catch (error) {
            console.debug('trainer-danger-mode-changed dispatch failed', error);
        }

        const loraAlphaInput = document.getElementById('lora_alpha');
        if (loraAlphaInput) {
            loraAlphaInput.disabled = !enabled;
            const loraWrapper = loraAlphaInput.closest('.mb-3') || loraAlphaInput.parentElement;
            if (loraWrapper) {
                loraWrapper.classList.toggle('danger-mode-locked', !enabled);

                let loraHint = loraWrapper.querySelector('.danger-mode-hint');
                if (!loraHint) {
                    loraHint = document.createElement('div');
                    loraHint.className = 'form-text text-warning danger-mode-hint';
                    loraHint.textContent = 'Enable "I Know What I\'m Doing" in Advanced settings to override LoRA alpha.';
                    loraWrapper.appendChild(loraHint);
                }
                loraHint.style.display = enabled ? 'none' : '';
            }
        }
    }

    updateLoraVisibility(modelType) {
        const loraConfig = document.getElementById('section-lora_config');
        if (loraConfig) {
            // Show LoRA config section only if model_type is 'lora'
            loraConfig.classList.toggle('field-disabled', modelType !== 'lora');
        }
    }

    updateQuantizationAvailability(modelType) {
        const disableQuant = modelType !== 'lora';
        const quantFieldIds = ['base_model_precision', 'text_encoder_1_precision', 'quantize_via'];

        quantFieldIds.forEach((fieldId) => {
            const input = document.getElementById(fieldId);
            if (!input) {
                return;
            }

            input.disabled = disableQuant;

            const wrapper = input.closest('.mb-3') || input.parentElement;
            if (wrapper) {
                wrapper.classList.toggle('field-disabled', disableQuant);
            }
        });

        const quantSection = document.getElementById('section-quantization');
        if (quantSection) {
            quantSection.classList.toggle('field-disabled', disableQuant);
        }
    }

    updateFluxLoraVisibility(modelFamily, modelType) {
        const fluxLoraGroup = document.getElementById('flux-lora-group');
        if (fluxLoraGroup) {
            const isVisible = modelFamily === 'flux' && modelType === 'lora';
            fluxLoraGroup.classList.toggle('field-disabled', !isVisible);
        }
    }

    updateLycorisVisibility(loraType, modelType) {
        const lycorisGroup = document.getElementById('lycoris-config-group');
        if (lycorisGroup) {
            const isVisible = loraType === 'lycoris' && modelType === 'lora';
            lycorisGroup.classList.toggle('field-disabled', !isVisible);
        }
    }

    setupButtonHandlers() {
        const validateBtn = document.getElementById('validateBtn');
        const runBtn = document.getElementById('runBtn');
        const cancelBtn = document.getElementById('cancelBtn');

        const validateUsesHTMX = validateBtn?.hasAttribute('hx-post');
        if (validateBtn && !validateUsesHTMX && !validateBtn.dataset.listenerAdded) {
            validateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleValidate();
            });
            validateBtn.dataset.listenerAdded = 'true';
        } else if (!validateBtn) {
            console.error('Validate button not found');
        }

        if (!runBtn) {
            console.error('Run button not found');
        }

        const cancelUsesHTMX = cancelBtn?.hasAttribute('hx-post');
        if (cancelBtn && !cancelUsesHTMX && !cancelBtn.dataset.listenerAdded) {
            cancelBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleCancel();
            });
            cancelBtn.dataset.listenerAdded = 'true';
        } else if (!cancelBtn) {
            console.error('Cancel button not found');
        }

        this.setupFsdpDetection();
    }

    setupFsdpDetection() {
        const detectBtn = document.getElementById('fsdp-detect-blocks');
        const refreshBtn = document.getElementById('fsdp-refresh-blocks');
        const applyBtn = document.getElementById('fsdp-apply-selection');

        if (detectBtn && !detectBtn.dataset.listenerAdded) {
            detectBtn.addEventListener('click', () => this.handleFsdpDetection(false));
            detectBtn.dataset.listenerAdded = 'true';
        }

        if (refreshBtn && !refreshBtn.dataset.listenerAdded) {
            refreshBtn.addEventListener('click', () => this.handleFsdpDetection(true));
            refreshBtn.dataset.listenerAdded = 'true';
        }

        if (applyBtn && !applyBtn.dataset.listenerAdded) {
            applyBtn.addEventListener('click', () => this.applyFsdpSelection());
            applyBtn.dataset.listenerAdded = 'true';
        }
    }

    async handleFsdpDetection(forceRefresh = false) {
        const statusEl = document.getElementById('fsdp-detection-status');
        const detectBtn = document.getElementById('fsdp-detect-blocks');
        const refreshBtn = document.getElementById('fsdp-refresh-blocks');
        const modal = this.getFsdpModal();
        const spinner = document.getElementById('fsdp-detect-spinner');
        const content = document.getElementById('fsdp-detect-content');
        const errorBox = document.getElementById('fsdp-detect-error');

        const modelFamilyEl = document.getElementById('model_family');
        const modelFamily = modelFamilyEl ? modelFamilyEl.value.trim() : '';

        if (!modelFamily) {
            this.showToast('Select a model family before detecting FSDP blocks.', 'warning');
            return;
        }

        const pretrainedEl = document.getElementById('pretrained_model_name_or_path');
        const transformerEl = document.getElementById('pretrained_transformer_model_name_or_path');
        const flavourEl = document.getElementById('model_flavour');

        const payload = {
            pretrained_model: (pretrainedEl && pretrainedEl.value.trim()) || (transformerEl && transformerEl.value.trim()) || '',
            model_flavour: flavourEl ? flavourEl.value.trim() : '',
            force_refresh: forceRefresh,
        };

        if (statusEl) {
            const action = forceRefresh ? 'Refreshing' : 'Detecting';
            statusEl.textContent = `${action} blocks for ${modelFamily}...`;
        }

        if (detectBtn) {
            detectBtn.disabled = true;
        }
        if (refreshBtn) {
            refreshBtn.disabled = true;
        }

        if (spinner) {
            spinner.classList.remove('d-none');
        }
        if (content) {
            content.classList.add('d-none');
        }
        if (errorBox) {
            errorBox.classList.add('d-none');
            errorBox.textContent = '';
        }

        if (modal) {
            modal.show();
        }

        try {
            const endpoint = `${this.apiBaseUrl}/api/models/${encodeURIComponent(modelFamily)}/fsdp-blocks`;
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                let detail = 'Failed to detect FSDP blocks.';
                try {
                    const data = await response.json();
                    detail = data?.detail || detail;
                } catch (parseError) {
                    // ignore JSON parse errors
                }
                throw new Error(detail);
            }

            const data = await response.json();
            this.fsdpDetectionData = data;
            this.populateFsdpDetectionModal(data);

            if (statusEl) {
                const source = data.cached ? 'cached result' : 'fresh detection';
                statusEl.textContent = `Loaded ${data.transformer_classes?.length || 0} classes (${source}).`;
            }

            if (refreshBtn) {
                refreshBtn.classList.remove('d-none');
            }
        } catch (error) {
            console.error('FSDP detection failed:', error);
            if (errorBox) {
                errorBox.textContent = error.message || 'Failed to detect FSDP blocks.';
                errorBox.classList.remove('d-none');
            }
            if (statusEl) {
                statusEl.textContent = 'Detection failed. See details for more information.';
            }
        } finally {
            if (spinner) {
                spinner.classList.add('d-none');
            }
            if (content) {
                content.classList.remove('d-none');
            }
            if (detectBtn) {
                detectBtn.disabled = false;
            }
            if (refreshBtn) {
                refreshBtn.disabled = false;
            }
        }
    }

    populateFsdpDetectionModal(data) {
        const rowsEl = document.getElementById('fsdp-detect-rows');
        const metadataEl = document.getElementById('fsdp-detect-metadata');
        const noSplitContainer = document.getElementById('fsdp-no-split-container');
        const noSplitTemplate = document.getElementById('fsdp-no-split-template');
        const noSplitList = document.getElementById('fsdp-no-split-list');
        const field = document.getElementById('fsdp_transformer_layer_cls_to_wrap');
        const existing = field ? field.value.split(',').map((s) => s.trim()).filter(Boolean) : [];

        if (rowsEl) {
            rowsEl.innerHTML = '';
        }

        const classes = Array.isArray(data.transformer_classes) ? data.transformer_classes : [];

        if (metadataEl) {
            const total = data.total_parameter_count || 0;
            const formatter = new Intl.NumberFormat();
            const parts = [
                `Model: ${data.model_family || 'unknown'}`,
                data.pretrained_model ? `Source: ${data.pretrained_model}` : null,
                `Total parameters considered: ${formatter.format(total)}`,
            ].filter(Boolean);
            metadataEl.textContent = parts.join(' • ');
        }

        if (rowsEl) {
            const formatter = new Intl.NumberFormat();
            classes.forEach((entry, index) => {
                const className = entry.class_name;
                const checkboxId = `fsdp-detect-class-${index}`;
                const isSelected = existing.includes(className);

                const row = document.createElement('tr');

                const selectCell = document.createElement('td');
                selectCell.className = 'align-middle';
                selectCell.innerHTML = `
                    <input type="checkbox" class="form-check-input" id="${checkboxId}"
                        value="${className}" ${isSelected ? 'checked' : ''}>
                `;

                const classCell = document.createElement('td');
                classCell.className = 'align-middle fw-semibold';
                classCell.textContent = className;

                const occurrencesCell = document.createElement('td');
                occurrencesCell.className = 'align-middle text-center';
                occurrencesCell.textContent = entry.occurrences ?? '-';

                const paramsCell = document.createElement('td');
                paramsCell.className = 'align-middle text-end';
                paramsCell.textContent = formatter.format(entry.total_params ?? 0);

                const exampleCell = document.createElement('td');
                exampleCell.className = 'align-middle';
                const samples = Array.isArray(entry.sample_paths) ? entry.sample_paths.slice(0, 3) : [];
                exampleCell.textContent = samples.join(', ') || '—';

                row.appendChild(selectCell);
                row.appendChild(classCell);
                row.appendChild(occurrencesCell);
                row.appendChild(paramsCell);
                row.appendChild(exampleCell);

                rowsEl.appendChild(row);
            });
        }

        if (noSplitContainer && noSplitList) {
            const modules = Array.isArray(data.no_split_modules) ? data.no_split_modules : [];
            if (modules.length) {
                noSplitList.innerHTML = '';
                modules.forEach((moduleName) => {
                    const badge = document.createElement('span');
                    badge.className = 'badge bg-secondary me-1 mb-1';
                    badge.textContent = moduleName;
                    noSplitList.appendChild(badge);
                });
                noSplitContainer.classList.remove('d-none');
            } else {
                noSplitList.innerHTML = '';
                noSplitContainer.classList.add('d-none');
            }
        }
    }

    applyFsdpSelection() {
        const checkboxes = Array.from(document.querySelectorAll('#fsdp-detect-rows input[type="checkbox"]'));
        const selected = checkboxes
            .filter((input) => input.checked)
            .map((input) => input.value)
            .filter((value) => typeof value === 'string' && value.trim().length > 0);

        const field = document.getElementById('fsdp_transformer_layer_cls_to_wrap');
        if (field) {
            field.value = selected.join(',');
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
        }

        if (window.Alpine?.store) {
            const trainerStore = window.Alpine.store('trainer');
            if (trainerStore && typeof trainerStore.markFormDirty === 'function') {
                trainerStore.markFormDirty();
            }
        }

        const modal = this.getFsdpModal();
        if (modal) {
            modal.hide();
        }

        const statusEl = document.getElementById('fsdp-detection-status');
        if (statusEl) {
            if (selected.length) {
                statusEl.textContent = `Selected ${selected.length} block class${selected.length === 1 ? '' : 'es'}.`;
            } else {
                statusEl.textContent = 'Cleared FSDP block selection.';
            }
        }
    }

    getFsdpModal() {
        if (!window.bootstrap) {
            return null;
        }
        if (!this.fsdpModal) {
            const modalEl = document.getElementById('fsdpBlockModal');
            if (modalEl) {
                this.fsdpModal = new window.bootstrap.Modal(modalEl, {
                    backdrop: 'static',
                    keyboard: true,
                });
            }
        }
        return this.fsdpModal;
    }

    async handleValidate() {
        // Starting validation
        const payload = this.getPayload();
        if (!payload) {
            console.error('Failed to get payload');
            return;
        }
        // Validation payload prepared

        const button = document.getElementById('validateBtn');
        this.setButtonLoading(button, true, 'Validating...');

        try {
            const response = await this.apiCall('/api/training/validate', payload);
            // Validation completed successfully
            this.handleResponse(response, 'Configuration validated successfully!');
        } catch (error) {
            console.error('Validation error:', error);
            this.showError(error.message);
        } finally {
            this.setButtonLoading(button, false, '<i class="fas fa-check-circle"></i> Validate Config');
        }
    }

    async handleRun() {
        const payload = this.getPayload();
        if (!payload) return;

        const button = document.getElementById('runBtn');
        this.setButtonLoading(button, true, 'Starting...');

        try {
            const response = await this.apiCall('/api/training/start', payload);
            this.handleResponse(response, 'Training started successfully!');
            // Reset event list when starting new training
            if (window.eventHandler) {
                window.eventHandler.resetEventList();
                window.eventHandler.lastEventIndex = 0; // Reset index for new training
            }
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setButtonLoading(button, false, '<i class="fas fa-play"></i> Start Training');
        }
    }

    async handleCancel() {
        const jobInput = document.getElementById('job_id')
            || this.form?.querySelector('[name="job_id"]');
        let jobId = jobInput?.value?.trim();

        if (!jobId) {
            jobId = await this.fetchCurrentJobId();
        }

        if (!jobId) {
            console.warn('Cancel requested but no active job_id could be determined');
            this.showToast('No active training job to cancel.', 'warning');
            return;
        }

        const button = document.getElementById('cancelBtn');
        if (button) {
            this.setButtonLoading(button, true, 'Cancelling...');
        }

        this.showToast('Cancelling training... This may take a moment.', 'info');

        try {
            const response = await this.apiCall('/api/training/cancel', { job_id: jobId });
            this.handleResponse(response, 'Training cancelled successfully!');
        } catch (error) {
            this.showError(error.message);
        } finally {
            if (button) {
                this.setButtonLoading(button, false, '<i class="fas fa-stop"></i> Cancel Training');
            }
        }
    }

    getPayload() {
        if (!this.form) {
            this.form = document.getElementById('trainer-form');
        }

        if (!this.form) {
            this.showError('Trainer form is not available');
            return null;
        }

        // Sync builder data if in builder mode
        if (document.getElementById('builderModeBtn')?.classList.contains('active')) {
            window.dataloaderBuilder?.syncToJSON();
        }

        const formData = new FormData(this.form);

        const trainerStore = window.Alpine && typeof window.Alpine.store === 'function'
            ? window.Alpine.store('trainer')
            : null;

        if (trainerStore) {
            try {
                if (typeof trainerStore.normalizeCheckboxFormData === 'function') {
                    trainerStore.normalizeCheckboxFormData.call(trainerStore, formData);
                }
                if (typeof trainerStore.ensureCompleteFormData === 'function') {
                    trainerStore.ensureCompleteFormData.call(trainerStore, formData);
                }
                if (typeof trainerStore.appendConfigValuesToFormData === 'function') {
                    trainerStore.appendConfigValuesToFormData.call(
                        trainerStore,
                        formData,
                        trainerStore.activeEnvironmentConfig || {},
                    );
                }
                if (typeof trainerStore.normalizeCheckboxFormData === 'function') {
                    trainerStore.normalizeCheckboxFormData.call(trainerStore, formData);
                }
            } catch (error) {
                console.warn('Unable to merge full trainer form data from store:', error);
            }
        }

        const payload = {
            trainer_config: {},
            dataloader_config: [],
            webhook_config: [
                {
                    webhook_type: "raw",
                    callback_url: `${this.callbackUrl}/callback`,
                    log_level: "info"
                }
            ],
            job_id: formData.get('job_id')
        };

        // Process form data
        for (const [key, value] of formData.entries()) {
            if (key.startsWith('--')) {
                const element = document.querySelector(`[name="${key}"]`);
                if (element?.type === 'checkbox') {
                    if (element.checked) {
                        payload.trainer_config[key] = 'true';
                    }
                } else if (value) {
                    payload.trainer_config[key] = value;
                }
            } else if (key === 'dataloader_config') {
                try {
                    payload.dataloader_config = JSON.parse(value);
                } catch (error) {
                    this.showError('Invalid JSON for Dataloader Config');
                    return null;
                }
            } else if (key === '--discord_webhooks') {
                try {
                    const discordWebhooks = JSON.parse(value);
                    // Add user-configured Discord webhooks to the list
                    if (Array.isArray(discordWebhooks) && discordWebhooks.length > 0) {
                        payload.webhook_config.push(...discordWebhooks);
                    }
                } catch (error) {
                    console.warn('Invalid JSON for Discord Webhooks, skipping');
                }
            }
        }

        return payload;
    }

    async fetchCurrentJobId() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/training/status`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json'
                },
                credentials: 'include'
            });

            if (!response.ok) {
                return null;
            }

            const data = await response.json();
            return data?.job_id || data?.jobId || null;
        } catch (error) {
            console.warn('Failed to retrieve current job id for cancellation', error);
            return null;
        }
    }

    async apiCall(endpoint, data) {
        const response = await fetch(`${this.apiBaseUrl}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            credentials: 'include', // Include credentials for CORS
            body: JSON.stringify(data),
        });

        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else {
            throw new Error('Invalid response format from server');
        }
    }

    handleResponse(data, successMessage) {
        if (data.detail) {
            this.showToast(data.detail, 'error');
        } else if (data.result) {
            this.showToast(successMessage, 'success');
        } else {
            this.showToast('Unexpected response format', 'error');
        }
    }

    setButtonLoading(button, loading, html) {
        if (!button) return;
        button.disabled = loading;
        button.innerHTML = loading ? '<span class="spinner-border spinner-border-sm me-2"></span>' + html : html;
    }

    showStatus(message, type = 'info') {
        const statusContainer = document.getElementById('statusContainer');
        if (!statusContainer) return;

        const statusId = 'status-' + Date.now();

        const alertClass = type === 'error' ? 'alert-danger' :
                         type === 'success' ? 'alert-success' : 'alert-info';

        const statusHTML = `
            <div id="${statusId}" class="alert ${alertClass} alert-dismissible fade show" role="alert">
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `;

        statusContainer.insertAdjacentHTML('beforeend', statusHTML);

        // Auto-dismiss after 5 seconds
        setTimeout(() => {
            const alertEl = document.getElementById(statusId);
            if (alertEl && window.bootstrap) {
                const alert = new bootstrap.Alert(alertEl);
                alert.close();
            }
        }, 5000);
    }

    showToast(message, type = 'success') {
        const toastContainer = document.querySelector('.toast-container');
        if (!toastContainer) {
            console.warn('Toast container not found, showing alert instead');
            alert(message);
            return;
        }

        const toastId = 'toast-' + Date.now();

        const toastHTML = `
            <div id="${toastId}" class="toast align-items-center text-white bg-${type === 'success' ? 'success' : type === 'error' ? 'danger' : 'info'}" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'} me-2"></i>
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        toastContainer.insertAdjacentHTML('beforeend', toastHTML);
        const toastEl = document.getElementById(toastId);

        if (toastEl && window.bootstrap && window.bootstrap.Toast) {
            const toast = new bootstrap.Toast(toastEl, {
                autohide: true,
                delay: 5000
            });
            toast.show();

            toastEl.addEventListener('hidden.bs.toast', () => {
                toastEl.remove();
            });
        } else {
            // Fallback if Bootstrap isn't available
            console.warn('Bootstrap Toast not available, using fallback');
            if (toastEl) {
                toastEl.style.display = 'block';
                toastEl.style.opacity = '1';
                setTimeout(() => {
                    toastEl.style.opacity = '0';
                    setTimeout(() => toastEl.remove(), 300);
                }, 5000);
            }
        }
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    setupJSONEditors() {
        document.querySelectorAll('.json-editor').forEach(editor => {
            editor.addEventListener('blur', function() {
                try {
                    const json = JSON.parse(this.value);
                    this.value = JSON.stringify(json, null, 4);
                } catch (e) {
                    // Invalid JSON, don't format
                }
            });

            // Add tab key support for JSON editors
            editor.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const start = this.selectionStart;
                    const end = this.selectionEnd;
                    this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                    this.selectionStart = this.selectionEnd = start + 4;
                }
            });
        });
    }
}

    // Export to window
    window.TrainerUI = TrainerUI;

    // Initialize trainer UI when DOM is ready
    document.addEventListener('DOMContentLoaded', () => {
        if (!window.trainerUI) {
            window.trainerUI = new TrainerUI();
            // TrainerUI initialized
        } else {
            // TrainerUI already exists
        }
    });
})(); // End IIFE
