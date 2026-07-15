/**
 * Checkpoints Module
 * Handles training checkpoint browsing, validation, and cleanup management
 */

if (!window.checkpointsManager) {
    window.checkpointsManager = function() {
        return {
            // Hero CTA hint management
            ...window.HintMixin.createMultiHint({
                useApi: false,
                storageKey: 'st_checkpoints_hints',
                hintKeys: ['hero']
            }),
            showHeroCTA() { return this.hints.hero; },
            dismissHeroCTA() { this.dismissHint('hero'); },
            restoreHeroCTA() { this.showHint('hero'); },

            // State
            checkpoints: [],
            selectedCheckpoint: null,
            environment: null,
            sortBy: 'step-desc',
            retentionLimit: 10,
            originalRetentionLimit: 10,
            retentionDirty: false,
            cleanupPreview: null,
            validationResult: null,
            inferenceSelection: [],
            inference: {
                activeTab: 'setup',
                promptSources: {
                    configured_prompt: null,
                    builtin_count: 0,
                    configured_user_library: null,
                    user_libraries: [],
                    unsupported_multigpu_modes: [],
                    inference_defaults: {
                        num_inference_steps: null,
                        guidance_scale: null,
                        validation_resolution: null
                    }
                },
                promptSourcesLoaded: false,
                form: {
                    use_configured_prompt: true,
                    use_builtin_library: false,
                    user_library_filename: '',
                    custom_prompts: '',
                    filename_style: 'descriptive',
                    keep_loaded: false,
                    streaming_preview: false,
                    idle_timeout_minutes: 15,
                    seed: '',
                    num_inference_steps: '',
                    guidance_scale: '',
                    validation_resolution: ''
                },
                session: null,
                interactivePrompt: '',
                history: [],
                historySelection: [],
                historyPage: 1,
                historyTotal: 0,
                statusTimer: null
            },
            markdownService: window.markdownService || null,
            loading: {
                checkpoints: false,
                validation: false,
                delete: false,
                preview: false,
                cleanup: false,
                saveRetention: false,
                cloudOutputs: false,
                inferenceSources: false,
                inferenceStart: false,
                inferenceHistory: false,
                inferenceDelete: false,
                inferenceAction: false
            },
            cloudOutputs: [],
            visibilitySettings: {
                validity: true,
                images: true,
                size: true,
                tags: true,
                path: true
            },

            // HuggingFace upload state
            hfAuthenticated: false,
            hfUsername: null,
            uploadConfig: {
                push_to_hub: false,
                hub_model_id: null
            },
            uploadTasks: {},
            uploadModalCheckpoints: [],

            // SSE listener cleanup
            sseListenerCleanup: null,

            // Lightbox state
            lightbox: {
                isOpen: false,
                currentIndex: 0,
                images: [],
                zoom: 1,
                isDragging: false,
                dragStart: { x: 0, y: 0 },
                imagePosition: { x: 0, y: 0 }
            },

            // Lifecycle
            async init() {
                // Wait for auth before making any API calls
                const canProceed = await window.waitForAuthReady();
                if (!canProceed) {
                    return;
                }

                // Load hint state first (non-blocking)
                this.loadHints();

                try {
                    await this.ensureEnvironment();
                    this.setupSSEListeners();
                } catch (error) {
                    console.error('Unable to initialize checkpoints manager:', error);
                    if (window.showToast) {
                        window.showToast('Failed to determine active environment', 'error');
                    }
                    return;
                }

                this.markdownService = window.markdownService || this.markdownService;

                await this.loadVisibilitySettings();
                await this.loadInferenceSettings();
                await this.loadCheckpoints();
                await this.loadRetentionConfig();
                await this.checkHuggingFaceAuth();
                await this.loadUploadConfig();
                await this.loadCloudOutputs();
            },

            async ensureEnvironment() {
                if (this.environment) {
                    return this.environment;
                }

                const response = await ApiClient.fetch('/api/configs/active');
                if (!response.ok) {
                    throw new Error('Failed to get active config');
                }

                const configData = await response.json();
                this.environment = configData.name;
                return this.environment;
            },

            normalizeCheckpoint(raw, index) {
                const name = raw.name || raw.id || `checkpoint-${index}`;
                const id = raw.id || name;

                let createdAt = raw.created_at || null;
                if (!createdAt && raw.timestamp) {
                    const epochMs = typeof raw.timestamp === 'number'
                        ? raw.timestamp * 1000
                        : Date.parse(raw.timestamp);
                    if (!Number.isNaN(epochMs)) {
                        createdAt = new Date(epochMs).toISOString();
                    }
                }

                if (!createdAt && raw.modified_at) {
                    const modified = typeof raw.modified_at === 'number'
                        ? raw.modified_at * 1000
                        : Date.parse(raw.modified_at);
                    if (!Number.isNaN(modified)) {
                        createdAt = new Date(modified).toISOString();
                    }
                }

                const sizeBytes = raw.size_bytes ?? raw.size ?? 0;
                const validation = raw.validation || {};
                let validationStatus = raw.validation_status ?? 'pending';
                if (validation.valid === true) {
                    validationStatus = 'valid';
                } else if (validation.valid === false) {
                    validationStatus = 'invalid';
                }
                const validationMessage = validation.error || validation.message || null;
                const assets = Array.isArray(raw.assets) ? raw.assets : [];
                const readme = raw.readme || null;

                const checkpoint = {
                    ...raw,
                    id,
                    name,
                    created_at: createdAt,
                    size_bytes: sizeBytes,
                    size: sizeBytes,
                    files: raw.files || [],
                    validation_status: validationStatus,
                    validation_message: validationMessage,
                    validation,
                    assets,
                    readme,
                    // Upload tracking properties
                    uploading: false,
                    uploadProgress: 0,
                    uploadMessage: ''
                };

                this._applyReadmeMetadata(checkpoint);
                return checkpoint;
            },

            _timestampForSort(checkpoint) {
                const source = checkpoint.created_at || checkpoint.timestamp || 0;
                if (typeof source === 'number') {
                    return source;
                }
                const parsed = Date.parse(source);
                return Number.isNaN(parsed) ? 0 : parsed;
            },

            _applyReadmeMetadata(checkpoint) {
                if (!checkpoint) {
                    return;
                }

                const service = this.markdownService || window.markdownService;
                if (service && typeof service.render === 'function') {
                    try {
                        const result = service.render(checkpoint.readme, checkpoint.assets);
                        checkpoint.readme_html = result.html || '';
                        checkpoint.readme_tags = Array.isArray(result.tags) ? result.tags : [];
                        checkpoint.readme_gallery = Array.isArray(result.gallery) ? result.gallery : [];
                        checkpoint.readme_metadata = result.metadata || {};
                        checkpoint.readme_front_matter = result.frontMatterRaw || null;
                        return;
                    } catch (error) {
                        console.warn('Failed to render checkpoint README via MarkdownService:', error);
                    }
                }

                const legacySource = typeof checkpoint.readme === 'string'
                    ? checkpoint.readme
                    : checkpoint.readme?.body || '';
                checkpoint.readme_html = this._renderReadmeLegacy(legacySource);
                checkpoint.readme_tags = Array.isArray(checkpoint.readme_tags) ? checkpoint.readme_tags : [];
                checkpoint.readme_gallery = Array.isArray(checkpoint.readme_gallery) ? checkpoint.readme_gallery : [];
                checkpoint.readme_metadata = checkpoint.readme_metadata || {};
            },

            // Data Loading
            async loadCheckpoints() {
                this.loading.checkpoints = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints?environment=${encodeURIComponent(environment)}&sort_by=${this.sortBy}`);
                    if (!response.ok) {
                        throw new Error('Failed to load checkpoints');
                    }

                    const data = await response.json();
                    const checkpoints = Array.isArray(data.checkpoints) ? data.checkpoints : [];
                    this.checkpoints = checkpoints.map((cp, idx) => this.normalizeCheckpoint(cp, idx));
                    const availableNames = new Set(this.checkpoints.map(cp => cp.name));
                    this.inferenceSelection = this.inferenceSelection.filter(name => availableNames.has(name));
                    this.sortCheckpoints();

                    // Calculate common aspect ratio from existing validation images
                    this.$nextTick(() => {
                        this.updatePlaceholderAspectRatios();
                    });
                } catch (error) {
                    console.error('Failed to load checkpoints:', error);
                    if (window.showToast) {
                        window.showToast('Failed to load checkpoints', 'error');
                    }
                    this.checkpoints = [];
                } finally {
                    this.loading.checkpoints = false;
                }
            },

            async loadRetentionConfig() {
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/retention?environment=${encodeURIComponent(environment)}`);
                    if (!response.ok) {
                        throw new Error('Failed to load retention config');
                    }

                    const data = await response.json();
                    this.retentionLimit = data.retention_limit ?? 10;
                    this.originalRetentionLimit = this.retentionLimit;
                    this.retentionDirty = false;
                } catch (error) {
                    console.error('Failed to load retention config:', error);
                    if (window.showToast) {
                        window.showToast('Failed to load retention settings', 'error');
                    }
                    this.retentionLimit = 10;
                    this.originalRetentionLimit = 10;
                }
            },

            // Sorting
            sortCheckpoints() {
                const [field, order] = this.sortBy.split('-');

                this.checkpoints.sort((a, b) => {
                    let valA;
                    let valB;

                    switch (field) {
                        case 'step':
                            valA = a.step || 0;
                            valB = b.step || 0;
                            break;
                        case 'date':
                            valA = this._timestampForSort(a);
                            valB = this._timestampForSort(b);
                            break;
                        case 'size':
                            valA = a.size_bytes ?? a.size ?? 0;
                            valB = b.size_bytes ?? b.size ?? 0;
                            break;
                        default:
                            return 0;
                    }

                    return order === 'asc' ? valA - valB : valB - valA;
                });
            },

            // Checkpoint Selection
            async selectCheckpoint(id) {
                const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                if (!checkpoint) {
                    return;
                }

                this.selectedCheckpoint = checkpoint;
                this.validationResult = null;

                // Collapse the checkpoint notes section when switching checkpoints
                this.$nextTick(() => {
                    const collapseEl = document.getElementById('checkpointNotesCollapse');
                    if (collapseEl && window.bootstrap) {
                        const bsCollapse = window.bootstrap.Collapse.getInstance(collapseEl);
                        if (bsCollapse) {
                            bsCollapse.hide();
                        }
                    }
                });
            },

            // Validation
            async validateCheckpoint(id) {
                if (this.loading.validation) {
                    return;
                }

                this.loading.validation = true;
                this.validationResult = null;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/${encodeURIComponent(id)}/validate`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ environment })
                    });

                    if (!response.ok) {
                        throw new Error('Validation request failed');
                    }

                    const result = await response.json();
                    this.validationResult = result;

                    const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                    if (checkpoint) {
                        checkpoint.validation_status = result.valid ? 'valid' : 'invalid';
                        checkpoint.validation_message = result.error || result.message || null;
                        checkpoint.validation = {
                            valid: result.valid,
                            error: result.error || null,
                        };
                    }

                    if (this.selectedCheckpoint && (this.selectedCheckpoint.id === id || this.selectedCheckpoint.name === id)) {
                        this.selectedCheckpoint.validation_status = result.valid ? 'valid' : 'invalid';
                        this.selectedCheckpoint.validation_message = result.error || result.message || null;
                        this.selectedCheckpoint.validation = {
                            valid: result.valid,
                            error: result.error || null,
                        };
                    }
                } catch (error) {
                    console.error('Failed to validate checkpoint:', error);
                    this.validationResult = {
                        valid: false,
                        message: error.message || 'Validation failed'
                    };
                    if (window.showToast) {
                        window.showToast('Failed to validate checkpoint', 'error');
                    }
                } finally {
                    this.loading.validation = false;
                }
            },

            // Deletion
            async deleteCheckpoint(id) {
                if (this.loading.delete) {
                    return;
                }

                const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                if (!checkpoint) {
                    return;
                }

                if (!confirm('Are you sure you want to delete this checkpoint? This action cannot be undone.')) {
                    return;
                }

                this.loading.delete = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/${encodeURIComponent(checkpoint.name)}?environment=${encodeURIComponent(environment)}`, {
                        method: 'DELETE'
                    });

                    if (!response.ok) {
                        throw new Error('Failed to delete checkpoint');
                    }

                    this.checkpoints = this.checkpoints.filter(cp => cp.id !== checkpoint.id);
                    this.inferenceSelection = this.inferenceSelection.filter(name => name !== checkpoint.name);

                    if (this.selectedCheckpoint && (this.selectedCheckpoint.id === checkpoint.id || this.selectedCheckpoint.name === checkpoint.name)) {
                        this.selectedCheckpoint = null;
                        this.validationResult = null;
                    }

                    if (window.showToast) {
                        window.showToast('Checkpoint deleted successfully', 'success');
                    }
                } catch (error) {
                    console.error('Failed to delete checkpoint:', error);
                    if (window.showToast) {
                        window.showToast('Failed to delete checkpoint', 'error');
                    }
                } finally {
                    this.loading.delete = false;
                }
            },

            _renderReadmeLegacy(content) {
                if (!content) {
                    return '';
                }

                const escape = (str) => String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');

                let html = escape(content);

                html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {
                    const safeAlt = escape(alt);
                    const safeUrl = url.replace(/"/g, '&quot;');
                    return `<img src="${safeUrl}" alt="${safeAlt}" class="readme-image" />`;
                });

                html = html.replace(/\n{2,}/g, '</p><p>');
                html = html.replace(/\n/g, '<br />');

                return `<p>${html}</p>`;
            },

            // Checkpoint Inference
            async loadInferenceSettings() {
                try {
                    const response = await ApiClient.fetch('/api/webui/ui-state/checkpoint-inference');
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to load checkpoint inference settings');
                    }
                    this.inference.form.filename_style = data.filename_style;
                    this.inference.form.keep_loaded = data.keep_loaded;
                    this.inference.form.streaming_preview = data.streaming_preview;
                } catch (error) {
                    console.error('Failed to load checkpoint inference settings:', error);
                }
            },

            async saveInferenceSettings() {
                try {
                    const response = await ApiClient.fetch('/api/webui/ui-state/checkpoint-inference', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            filename_style: this.inference.form.filename_style,
                            keep_loaded: this.inference.form.keep_loaded,
                            streaming_preview: this.inference.form.streaming_preview
                        })
                    });
                    const data = await response.json();
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to save checkpoint inference settings');
                    }
                } catch (error) {
                    console.error('Failed to save checkpoint inference settings:', error);
                    if (window.showToast) window.showToast(error.message, 'error');
                }
            },

            isInferenceSelected(checkpointName) {
                return this.inferenceSelection.includes(checkpointName);
            },

            toggleInferenceCheckpoint(checkpointName) {
                if (this.isInferenceSelected(checkpointName)) {
                    this.inferenceSelection = this.inferenceSelection.filter(name => name !== checkpointName);
                } else {
                    this.inferenceSelection = [...this.inferenceSelection, checkpointName];
                }
                if (this.inferenceSelection.length !== 1) {
                    this.inference.form.keep_loaded = false;
                }
            },

            selectedInferenceCheckpoints() {
                return this.inferenceSelection
                    .map(name => this.checkpoints.find(checkpoint => checkpoint.name === name))
                    .filter(Boolean);
            },

            async openInferenceWorkspace(tab = 'setup') {
                this.inference.activeTab = tab;
                const modalElement = document.getElementById('checkpointInferenceModal');
                if (modalElement && window.bootstrap) {
                    window.bootstrap.Modal.getOrCreateInstance(modalElement).show();
                }
                const requests = [this.loadInferenceHistory(1)];
                if (tab === 'setup') requests.push(this.loadInferencePromptSources());
                await Promise.all(requests);
            },

            async loadInferencePromptSources() {
                this.loading.inferenceSources = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch(
                        `/api/checkpoints/inference/prompt-sources?environment=${encodeURIComponent(environment)}`
                    );
                    if (!response.ok) {
                        throw new Error('Failed to load inference prompt sources');
                    }
                    const promptSources = await response.json();
                    const previousDefaults = this.inference.promptSources.inference_defaults || {};
                    const nextDefaults = promptSources.inference_defaults || {};
                    const stepsMatchPrevious = String(this.inference.form.num_inference_steps) === String(
                        previousDefaults.num_inference_steps ?? ''
                    );
                    const guidanceMatchesPrevious = String(this.inference.form.guidance_scale) === String(
                        previousDefaults.guidance_scale ?? ''
                    );
                    const resolutionMatchesPrevious = String(this.inference.form.validation_resolution) === String(
                        previousDefaults.validation_resolution ?? ''
                    );
                    if (!this.inference.promptSourcesLoaded
                        || this.inference.form.num_inference_steps === ''
                        || stepsMatchPrevious) {
                        this.inference.form.num_inference_steps = nextDefaults.num_inference_steps ?? '';
                    }
                    if (!this.inference.promptSourcesLoaded
                        || this.inference.form.guidance_scale === ''
                        || guidanceMatchesPrevious) {
                        this.inference.form.guidance_scale = nextDefaults.guidance_scale ?? '';
                    }
                    if (!this.inference.promptSourcesLoaded
                        || this.inference.form.validation_resolution === ''
                        || resolutionMatchesPrevious) {
                        this.inference.form.validation_resolution = nextDefaults.validation_resolution ?? '';
                    }
                    this.inference.promptSources = promptSources;
                    if (!this.inference.promptSourcesLoaded) {
                        this.inference.form.user_library_filename = this.inference.promptSources.configured_user_library || '';
                        this.inference.form.use_configured_prompt = Boolean(this.inference.promptSources.configured_prompt);
                        this.inference.promptSourcesLoaded = true;
                    }
                } catch (error) {
                    console.error('Failed to load inference prompt sources:', error);
                    if (window.showToast) window.showToast(error.message, 'error');
                } finally {
                    this.loading.inferenceSources = false;
                }
            },

            inferenceCustomPrompts() {
                return this.inference.form.custom_prompts
                    .split(/\r?\n/)
                    .map(prompt => prompt.trim())
                    .filter(Boolean);
            },

            selectedInferenceLibrary() {
                return this.inference.promptSources.user_libraries.find(
                    library => library.filename === this.inference.form.user_library_filename
                ) || null;
            },

            inferencePromptCount() {
                let count = this.inferenceCustomPrompts().length;
                if (this.inference.form.use_configured_prompt && this.inference.promptSources.configured_prompt) count += 1;
                if (this.inference.form.use_builtin_library) count += this.inference.promptSources.builtin_count || 0;
                const library = this.selectedInferenceLibrary();
                if (library) count += library.prompt_count || 0;
                return count;
            },

            inferenceRunCount() {
                return this.inferencePromptCount() * this.inferenceSelection.length * this.inferenceResolutionCount();
            },

            inferenceMultigpuWarning() {
                const modes = this.inference.promptSources.unsupported_multigpu_modes || [];
                if (!modes.length) return '';
                const labels = modes.map(mode => mode.replaceAll('-', ' '));
                const configuredModes = labels.length === 1
                    ? labels[0]
                    : `${labels.slice(0, -1).join(', ')} and ${labels.at(-1)}`;
                return `Checkpoint inference currently runs on one process and will not use this environment's configured ${configuredModes} validation ${labels.length === 1 ? 'mode' : 'modes'}.`;
            },

            inferenceResolutionCount() {
                const resolutions = String(this.inference.form.validation_resolution || '')
                    .split(',')
                    .map(resolution => resolution.trim())
                    .filter(Boolean);
                return Math.max(1, resolutions.length);
            },

            inferenceSettings() {
                const settings = {};
                if (this.inference.form.seed !== '') settings.seed = Number(this.inference.form.seed);
                if (this.inference.form.num_inference_steps !== '') {
                    settings.num_inference_steps = Number(this.inference.form.num_inference_steps);
                }
                if (this.inference.form.guidance_scale !== '') {
                    settings.guidance_scale = Number(this.inference.form.guidance_scale);
                }
                if (this.inference.form.validation_resolution.trim() !== '') {
                    settings.validation_resolution = this.inference.form.validation_resolution.trim();
                }
                return settings;
            },

            async _inferenceJson(response) {
                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Inference request failed');
                }
                return data;
            },

            async startInference() {
                if (this.loading.inferenceStart || !this.inferenceSelection.length || !this.inferencePromptCount()) return;
                this.loading.inferenceStart = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch('/api/checkpoints/inference/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            environment,
                            checkpoint_names: this.inferenceSelection,
                            use_configured_prompt: this.inference.form.use_configured_prompt,
                            use_builtin_library: this.inference.form.use_builtin_library,
                            user_library_filename: this.inference.form.user_library_filename || null,
                            custom_prompts: this.inferenceCustomPrompts(),
                            filename_style: this.inference.form.filename_style,
                            keep_loaded: this.inference.form.keep_loaded,
                            streaming_preview: this.inference.form.streaming_preview,
                            idle_timeout_minutes: Number(this.inference.form.idle_timeout_minutes),
                            settings: this.inferenceSettings()
                        })
                    });
                    this.inference.session = await this._inferenceJson(response);
                    this.scheduleInferenceStatusPoll();
                } catch (error) {
                    console.error('Failed to start checkpoint inference:', error);
                    if (window.showToast) window.showToast(error.message, 'error');
                } finally {
                    this.loading.inferenceStart = false;
                }
            },

            scheduleInferenceStatusPoll() {
                if (this.inference.statusTimer) window.clearTimeout(this.inference.statusTimer);
                this.inference.statusTimer = window.setTimeout(() => this.pollInferenceStatus(), 1000);
            },

            async pollInferenceStatus() {
                if (!this.inference.session?.session_id) return;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch(
                        `/api/checkpoints/inference/${encodeURIComponent(this.inference.session.session_id)}/status?environment=${encodeURIComponent(environment)}`
                    );
                    this.inference.session = await this._inferenceJson(response);
                    const active = ['pending', 'loading', 'running', 'unloading'].includes(this.inference.session.status);
                    if (active) {
                        this.scheduleInferenceStatusPoll();
                    } else {
                        await this.loadInferenceHistory(1);
                    }
                } catch (error) {
                    console.error('Failed to poll inference status:', error);
                    if (window.showToast) window.showToast(error.message, 'error');
                }
            },

            inferenceProgressPercent() {
                const session = this.inference.session;
                if (!session?.total_prompts) return 0;
                return Math.min(100, Math.round((session.completed_prompts || 0) / session.total_prompts * 100));
            },

            inferenceDisplayOutput() {
                const session = this.inference.session;
                if (!session) return null;
                const latest = session.latest_output || null;
                const preview = session.preview || null;
                if (!session.streaming_preview || !preview) return latest;
                if (!latest) return preview;
                const previewTime = Date.parse(preview.updated_at || '') || 0;
                const latestTime = Date.parse(latest.created_at || '') || 0;
                return previewTime > latestTime ? preview : latest;
            },

            inferenceIsActive() {
                return ['pending', 'queued', 'loading', 'running', 'unloading', 'cancelling']
                    .includes(this.inference.session?.status);
            },

            inferenceHistoryPageCount() {
                return Math.max(1, Math.ceil(this.inference.historyTotal / 24));
            },

            isInferenceHistorySelected(mediaPath) {
                return this.inference.historySelection.includes(mediaPath);
            },

            toggleInferenceHistoryItem(mediaPath) {
                if (this.isInferenceHistorySelected(mediaPath)) {
                    this.inference.historySelection = this.inference.historySelection.filter(path => path !== mediaPath);
                } else {
                    this.inference.historySelection = [...this.inference.historySelection, mediaPath];
                }
            },

            inferenceHistoryPageSelected() {
                return this.inference.history.length > 0
                    && this.inference.history.every(item => this.isInferenceHistorySelected(item.media_path));
            },

            toggleInferenceHistoryPage() {
                if (this.inferenceHistoryPageSelected()) {
                    this.inference.historySelection = [];
                    return;
                }
                this.inference.historySelection = this.inference.history.map(item => item.media_path);
            },

            async deleteInferenceHistorySelection() {
                const mediaPaths = this.inference.historySelection.slice();
                if (!mediaPaths.length || this.loading.inferenceDelete) return;
                const label = mediaPaths.length === 1 ? 'generation' : 'generations';
                if (!confirm(`Delete ${mediaPaths.length} selected ${label}? This cannot be undone.`)) return;

                this.loading.inferenceDelete = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch('/api/checkpoints/inference/history', {
                        method: 'DELETE',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ environment, media_paths: mediaPaths })
                    });
                    const result = await this._inferenceJson(response);
                    const remaining = Math.max(0, this.inference.historyTotal - result.deleted_count);
                    const page = Math.min(this.inference.historyPage, Math.max(1, Math.ceil(remaining / 24)));
                    this.inference.historySelection = [];
                    await this.loadInferenceHistory(page);
                    if (window.showToast) {
                        window.showToast(`Deleted ${result.deleted_count} ${label}.`, 'success');
                    }
                } catch (error) {
                    console.error('Failed to delete inference history:', error);
                    if (window.showToast) window.showToast(error.message, 'error');
                } finally {
                    this.loading.inferenceDelete = false;
                }
            },

            async queueInteractivePrompt() {
                const prompt = this.inference.interactivePrompt.trim();
                if (!prompt || this.loading.inferenceAction || this.inference.session?.status !== 'loaded') return;
                this.loading.inferenceAction = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch(
                        `/api/checkpoints/inference/${encodeURIComponent(this.inference.session.session_id)}/generate`,
                        {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                environment,
                                custom_prompts: [prompt],
                                filename_style: this.inference.form.filename_style,
                                settings: this.inferenceSettings()
                            })
                        }
                    );
                    await this._inferenceJson(response);
                    this.inference.interactivePrompt = '';
                    this.inference.session.status = 'running';
                    this.scheduleInferenceStatusPoll();
                } catch (error) {
                    if (window.showToast) window.showToast(error.message, 'error');
                } finally {
                    this.loading.inferenceAction = false;
                }
            },

            async stopInference(cancel = false) {
                if (!this.inference.session?.session_id || this.loading.inferenceAction) return;
                this.loading.inferenceAction = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const action = cancel ? 'cancel' : 'unload';
                    const response = await ApiClient.fetch(
                        `/api/checkpoints/inference/${encodeURIComponent(this.inference.session.session_id)}/${action}`,
                        {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ environment })
                        }
                    );
                    const result = await this._inferenceJson(response);
                    this.inference.session.status = result.status;
                    this.scheduleInferenceStatusPoll();
                } catch (error) {
                    if (window.showToast) window.showToast(error.message, 'error');
                } finally {
                    this.loading.inferenceAction = false;
                }
            },

            async loadInferenceHistory(page = 1) {
                this.loading.inferenceHistory = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch(
                        `/api/checkpoints/inference/history?environment=${encodeURIComponent(environment)}&page=${page}&page_size=24`
                    );
                    const data = await this._inferenceJson(response);
                    this.inference.history = data.items || [];
                    this.inference.historySelection = [];
                    this.inference.historyPage = data.page;
                    this.inference.historyTotal = data.total;
                } catch (error) {
                    console.error('Failed to load inference history:', error);
                } finally {
                    this.loading.inferenceHistory = false;
                }
            },

            // Retention Management
            markRetentionDirty() {
                this.retentionDirty = this.retentionLimit !== this.originalRetentionLimit;
                this.cleanupPreview = null;
            },

            async saveRetentionConfig() {
                if (this.loading.saveRetention || !this.retentionDirty) {
                    return;
                }

                this.loading.saveRetention = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch('/api/checkpoints/retention', {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
                            retention_limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to save retention config');
                    }

                    this.originalRetentionLimit = this.retentionLimit;
                    this.retentionDirty = false;

                    if (window.showToast) {
                        window.showToast('Retention settings saved', 'success');
                    }
                } catch (error) {
                    console.error('Failed to save retention config:', error);
                    if (window.showToast) {
                        window.showToast('Failed to save retention settings', 'error');
                    }
                } finally {
                    this.loading.saveRetention = false;
                }
            },

            // Cleanup Operations
            async previewCleanup() {
                if (this.loading.preview) {
                    return;
                }

                this.loading.preview = true;
                this.cleanupPreview = null;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch('/api/checkpoints/cleanup/preview', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
                            limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to preview cleanup');
                    }

                    const data = await response.json();
                    const checkpointsToRemove = Array.isArray(data.checkpoints_to_remove)
                        ? data.checkpoints_to_remove.map((cp, idx) => this.normalizeCheckpoint(cp, idx))
                        : [];

                    this.cleanupPreview = {
                        ...data,
                        checkpoints_to_remove: checkpointsToRemove,
                        count_to_remove: data.count_to_remove ?? checkpointsToRemove.length
                    };

                    if (window.showToast) {
                        window.showToast(
                            `Preview: ${this.cleanupPreview.count_to_remove} checkpoint(s) will be removed`,
                            'info'
                        );
                    }
                } catch (error) {
                    console.error('Failed to preview cleanup:', error);
                    if (window.showToast) {
                        window.showToast('Failed to preview cleanup', 'error');
                    }
                } finally {
                    this.loading.preview = false;
                }
            },

            async executeCleanup() {
                if (this.loading.cleanup) {
                    return;
                }

                if (!this.cleanupPreview || this.cleanupPreview.count_to_remove === 0) {
                    if (window.showToast) {
                        window.showToast('No checkpoints to remove', 'info');
                    }
                    return;
                }

                if (!confirm(`This will permanently delete ${this.cleanupPreview.count_to_remove} checkpoint(s). This action cannot be undone. Continue?`)) {
                    return;
                }

                this.loading.cleanup = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await ApiClient.fetch('/api/checkpoints/cleanup/execute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
                            limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to execute cleanup');
                    }

                    const result = await response.json();
                    const removedCheckpoints = Array.isArray(result.removed_checkpoints)
                        ? result.removed_checkpoints.map((cp, idx) => this.normalizeCheckpoint(cp, idx))
                        : [];
                    const removedIds = removedCheckpoints.map(cp => cp.id);

                    if (window.showToast) {
                        const removedCount = result.count_removed ?? removedIds.length;
                        window.showToast(
                            `Cleanup complete: ${removedCount} checkpoint(s) removed`,
                            'success'
                        );
                    }

                    await this.loadCheckpoints();

                    this.cleanupPreview = null;
                    if (this.selectedCheckpoint && removedIds.includes(this.selectedCheckpoint.id)) {
                        this.selectedCheckpoint = null;
                        this.validationResult = null;
                    }
                } catch (error) {
                    console.error('Failed to execute cleanup:', error);
                    if (window.showToast) {
                        window.showToast('Failed to execute cleanup', 'error');
                    }
                } finally {
                    this.loading.cleanup = false;
                }
            },

            // Update placeholder aspect ratios based on existing images
            updatePlaceholderAspectRatios() {
                // Find the first checkpoint with validation images to get aspect ratio
                let aspectRatio = null;
                const imgElements = document.querySelectorAll('.checkpoint-preview-image img');

                for (let img of imgElements) {
                    if (img.complete && img.naturalWidth > 0) {
                        aspectRatio = img.naturalWidth / img.naturalHeight;
                        break;
                    }
                }

                // If we found an aspect ratio, apply it to all placeholders
                if (aspectRatio) {
                    const placeholders = document.querySelectorAll('.no-validation-placeholder');
                    placeholders.forEach(placeholder => {
                        placeholder.style.aspectRatio = aspectRatio;
                    });
                }

                // Also set up load listeners for images that haven't loaded yet
                imgElements.forEach(img => {
                    if (!img.complete) {
                        img.addEventListener('load', () => {
                            if (!aspectRatio && img.naturalWidth > 0) {
                                aspectRatio = img.naturalWidth / img.naturalHeight;
                                const placeholders = document.querySelectorAll('.no-validation-placeholder');
                                placeholders.forEach(placeholder => {
                                    placeholder.style.aspectRatio = aspectRatio;
                                });
                            }
                        }, { once: true });
                    }
                });
            },

            // Utility Functions - delegate to UIHelpers
            formatDate(dateString) {
                return window.UIHelpers?.formatRelativeTime(dateString, { fallback: 'Unknown' }) || 'Unknown';
            },

            formatDateTime(dateString) {
                return window.UIHelpers?.formatDateTime(dateString, { fallback: 'Unknown' }) || 'Unknown';
            },

            formatSize(bytes) {
                if (!bytes || bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                const index = Math.floor(Math.log(bytes) / Math.log(k));
                return `${parseFloat((bytes / Math.pow(k, index)).toFixed(2))} ${sizes[index]}`;
            },

            // Visibility Settings Management
            async loadVisibilitySettings() {
                // Load visibility settings from localStorage
                const stored = localStorage.getItem('checkpoint_visibility_settings');
                if (stored) {
                    try {
                        const settings = JSON.parse(stored);
                        Object.assign(this.visibilitySettings, settings);
                    } catch (error) {
                        console.error('Failed to parse stored visibility settings:', error);
                    }
                }
            },

            async saveVisibilitySettings() {
                // Save visibility settings to localStorage
                try {
                    localStorage.setItem('checkpoint_visibility_settings', JSON.stringify(this.visibilitySettings));

                    // Trigger aspect ratio recalculation after visibility changes
                    this.$nextTick(() => {
                        this.updatePlaceholderAspectRatios();
                    });
                } catch (error) {
                    console.error('Error saving visibility settings:', error);
                }
            },

            // HuggingFace Integration
            async checkHuggingFaceAuth() {
                try {
                    const response = await ApiClient.fetch('/api/publishing/token/validate');
                    if (response.ok) {
                        const data = await response.json();
                        this.hfAuthenticated = data.valid === true;
                        this.hfUsername = data.username || null;
                    } else {
                        this.hfAuthenticated = false;
                        this.hfUsername = null;
                    }
                } catch (error) {
                    console.error('Failed to check HuggingFace auth:', error);
                    this.hfAuthenticated = false;
                    this.hfUsername = null;
                }
            },

            async loadUploadConfig() {
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/configs/${environment}`);
                    if (response.ok) {
                        const data = await response.json();
                        const config = data.config || data;
                        this.uploadConfig.push_to_hub = config['--push_to_hub'] || config.push_to_hub || false;
                        this.uploadConfig.hub_model_id = config['--hub_model_id'] || config.hub_model_id || null;
                    }
                } catch (error) {
                    console.error('Failed to load upload config:', error);
                }
            },

            canUpload() {
                return this.hfAuthenticated && this.uploadConfig.push_to_hub && this.uploadConfig.hub_model_id;
            },

            getUploadTooltip() {
                if (!this.hfAuthenticated) {
                    return 'Login to HuggingFace on the Publishing tab to upload';
                }
                if (!this.uploadConfig.push_to_hub) {
                    return 'Enable push_to_hub on the Publishing tab to upload';
                }
                if (!this.uploadConfig.hub_model_id) {
                    return 'Configure a target repository on the Publishing tab to upload';
                }
                return 'Upload checkpoint to HuggingFace Hub';
            },

            async uploadCheckpoint(checkpointId) {
                const checkpoint = this.checkpoints.find(c => c.id === checkpointId);
                if (!checkpoint || !this.canUpload()) {
                    return;
                }

                try {
                    // Get the callback URL for webhook notifications
                    const callbackUrl = `${window.location.origin}/callback`;

                    const response = await fetch(`/api/checkpoints/${checkpoint.name}/upload`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            environment: this.environment,
                            repo_id: this.uploadConfig.hub_model_id,
                            callback_url: callbackUrl
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Upload failed');
                    }

                    const result = await response.json();

                    // Store task for tracking with checkpoint info
                    this.uploadTasks[result.task_id] = {
                        ...result,
                        checkpoint: checkpoint.name
                    };

                    if (window.showToast) {
                        window.showToast(`Upload started for ${checkpoint.name}`, 'success');
                    }

                    // Start monitoring the upload
                    this.monitorUpload(result.task_id);
                } catch (error) {
                    console.error('Upload failed:', error);
                    if (window.showToast) {
                        window.showToast(`Upload failed: ${error.message}`, 'error');
                    }
                }
            },

            async monitorUpload(taskId) {
                const checkStatus = async () => {
                    try {
                        const response = await fetch(`/api/checkpoints/upload/${taskId}/status`);
                        if (!response.ok) {
                            throw new Error('Failed to get upload status');
                        }

                        const status = await response.json();
                        this.uploadTasks[taskId] = status;

                        if (status.status === 'completed') {
                            if (window.showToast) {
                                window.showToast(`Upload completed for ${status.checkpoint}`, 'success');
                            }
                        } else if (status.status === 'failed') {
                            if (window.showToast) {
                                window.showToast(`Upload failed: ${status.error}`, 'error');
                            }
                        } else if (status.status === 'running') {
                            // Continue monitoring
                            setTimeout(() => checkStatus(), 2000);
                        }
                    } catch (error) {
                        console.error('Failed to monitor upload:', error);
                    }
                };

                checkStatus();
            },

            showUploadAllModal() {
                if (!this.canUpload()) {
                    return;
                }

                // Collect all checkpoint names
                this.uploadModalCheckpoints = this.checkpoints.map(c => c.name);

                // Show modal using Alpine.js event
                this.$dispatch('show-upload-all-modal');
            },

            async uploadAllCheckpoints(uploadMode) {
                if (!this.uploadModalCheckpoints.length || !this.canUpload()) {
                    return;
                }

                try {
                    // Get the callback URL for webhook notifications
                    const callbackUrl = `${window.location.origin}/callback`;

                    const response = await ApiClient.fetch('/api/checkpoints/upload', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            environment: this.environment,
                            checkpoint_names: this.uploadModalCheckpoints,
                            repo_id: this.uploadConfig.hub_model_id,
                            upload_mode: uploadMode,
                            callback_url: callbackUrl
                        })
                    });

                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.detail || 'Upload failed');
                    }

                    const result = await response.json();

                    // Store all tasks
                    result.tasks.forEach(task => {
                        this.uploadTasks[task.task_id] = task;
                        this.monitorUpload(task.task_id);
                    });

                    if (window.showToast) {
                        window.showToast(`Started uploading ${result.count} checkpoints`, 'success');
                    }

                    // Close modal
                    this.$dispatch('close-upload-all-modal');
                } catch (error) {
                    console.error('Upload all failed:', error);
                    if (window.showToast) {
                        window.showToast(`Upload failed: ${error.message}`, 'error');
                    }
                }
            },

            // SSE Support
            setupSSEListeners() {
                console.log('[Checkpoints] Setting up SSE listeners');
                console.log('[Checkpoints] SSEManager available?', !!window.SSEManager);

                // Remove any existing listener
                if (this.sseListenerCleanup) {
                    this.sseListenerCleanup();
                }

                // Add listener for checkpoint upload messages
                const checkpointListener = (data) => {
                    console.log('[Checkpoints] Checkpoint SSE event received:', data);
                    if (data.message_type === 'checkpoint_upload') {
                        this.handleUploadProgress(data);
                    } else {
                        console.log('[Checkpoints] Non-upload checkpoint message:', data.message_type);
                    }
                };

                // Add a catch-all debug listener
                const debugListener = (data) => {
                    if (data.message_type === 'checkpoint_upload') {
                        console.log('[Checkpoints DEBUG] Found checkpoint_upload in event!', data);
                    }
                };

                // Register with SSE Manager if available
                if (window.SSEManager && window.SSEManager.addEventListener) {
                    console.log('[Checkpoints] Registering SSE listeners');

                    // Register checkpoint listener
                    window.SSEManager.addEventListener('callback:checkpoint', checkpointListener);

                    // Also listen to other event types in case it's coming through differently
                    const eventTypes = ['callback:progress', 'callback:validation', 'callback:alert', 'callback:status', 'callback:job', 'callback:debug', 'callback'];
                    eventTypes.forEach(type => {
                        window.SSEManager.addEventListener(type, debugListener);
                    });

                    // Store cleanup function
                    this.sseListenerCleanup = () => {
                        if (window.SSEManager && window.SSEManager.removeEventListener) {
                            window.SSEManager.removeEventListener('callback:checkpoint', checkpointListener);
                            eventTypes.forEach(type => {
                                window.SSEManager.removeEventListener(type, debugListener);
                            });
                        }
                    };
                } else {
                    console.error('[Checkpoints] SSEManager not available!');
                }
            },

            handleUploadProgress(data) {
                console.log('[Checkpoints] Received upload progress:', data);

                // Extract data from extras field (where webhook data is stored)
                const extras = data.extras || {};
                const taskId = extras.task_id;
                const task = this.uploadTasks[taskId];

                if (!task) {
                    // Unknown task, might be from another session
                    console.warn('[Checkpoints] Unknown task ID:', taskId, 'Available tasks:', Object.keys(this.uploadTasks));
                    return;
                }

                // Update task info
                task.status = extras.status;
                task.progress = extras.progress || 0;
                task.message = extras.message || data.body || '';
                task.error = extras.error;

                // Find checkpoint card and update UI
                const checkpointName = task.checkpoint || extras.checkpoint;
                const checkpoint = this.checkpoints.find(c => c.name === checkpointName);

                const event = extras.event;

                if (checkpoint) {
                    if (event === 'started') {
                        checkpoint.uploading = true;
                        checkpoint.uploadProgress = 0;
                        checkpoint.uploadMessage = 'Starting upload...';
                    } else if (event === 'progress') {
                        checkpoint.uploading = true;
                        checkpoint.uploadProgress = task.progress;
                        checkpoint.uploadMessage = task.message;
                    } else if (event === 'completed') {
                        checkpoint.uploading = true;
                        checkpoint.uploadProgress = 100;
                        checkpoint.uploadMessage = 'Upload completed ✓';

                        if (window.showToast) {
                            window.showToast(`Successfully uploaded ${checkpointName}`, 'success');
                        }

                        setTimeout(() => {
                            checkpoint.uploading = false;
                            delete this.uploadTasks[taskId];
                            checkpoint.uploadProgress = 0;
                            checkpoint.uploadMessage = '';
                        }, 5000);
                    } else if (event === 'failed') {
                        checkpoint.uploading = false;
                        checkpoint.uploadProgress = 0;
                        checkpoint.uploadMessage = '';

                        if (window.showToast) {
                            window.showToast(`Upload failed: ${task.error || 'Unknown error'}`, 'error');
                        }

                        delete this.uploadTasks[taskId];
                    }
                }

                // For batch uploads, update overall progress
                if (task.mode === 'batch' && task.checkpoints) {
                    // Calculate overall progress for batch
                    const batchProgress = task.progress || 0;
                    const batchMessage = `Batch upload: ${task.message || 'Processing...'}`;

                    // Update all checkpoints in the batch
                    task.checkpoints.forEach(cpName => {
                        const cp = this.checkpoints.find(c => c.name === cpName);
                        if (cp) {
                            cp.uploading = extras.status === 'running';
                            cp.uploadProgress = batchProgress;
                            cp.uploadMessage = batchMessage;
                        }
                    });

                    // Clean up on completion
                    if (event === 'completed' || event === 'failed') {
                        setTimeout(() => {
                            task.checkpoints.forEach(cpName => {
                                const cp = this.checkpoints.find(c => c.name === cpName);
                                if (cp) {
                                    cp.uploading = false;
                                    cp.uploadProgress = 0;
                                    cp.uploadMessage = '';
                                }
                            });
                            delete this.uploadTasks[taskId];
                        }, 3000);
                    }
                }
            },

            // Lightbox Methods
            openLightbox(event, imageSrc, caption = '', allImages = [], currentIndex = 0) {
                event.stopPropagation();

                this.openLightboxDirect(imageSrc, caption, allImages, currentIndex);

                this.lightbox.isOpen = true;
                this.lightbox.currentIndex = currentIndex;
                this.lightbox.images = allImages.length > 0 ? allImages : [{ src: imageSrc, caption: caption }];
                this.lightbox.zoom = 1;
                this.lightbox.imagePosition = { x: 0, y: 0 };
                this.lightbox.isDragging = false;

                document.body.style.overflow = 'hidden';
                this._boundKeydownHandler = this.handleLightboxKeydown.bind(this);
                document.addEventListener('keydown', this._boundKeydownHandler);
            },

            openLightboxDirect(imageSrc, caption, allImages, currentIndex) {
                const testLightbox = document.createElement('div');
                testLightbox.id = 'test-lightbox';
                testLightbox.style.cssText = `
                    position: fixed !important;
                    top: 0 !important;
                    left: 0 !important;
                    width: 100vw !important;
                    height: 100vh !important;
                    background: rgba(0, 0, 0, 0.9) !important;
                    z-index: 999999 !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    opacity: 1 !important;
                    visibility: visible !important;
                `;

                const isVideo = imageSrc && imageSrc.startsWith('data:video/');
                let mediaElement;

                if (isVideo) {
                    mediaElement = document.createElement('video');
                    mediaElement.src = imageSrc;
                    mediaElement.controls = true;
                    mediaElement.autoplay = true;
                    mediaElement.loop = true;
                } else {
                    mediaElement = document.createElement('img');
                    mediaElement.src = imageSrc;
                }

                mediaElement.style.cssText = `
                    max-width: 90vw !important;
                    max-height: 90vh !important;
                    object-fit: contain !important;
                `;

                const closeBtn = document.createElement('button');
                closeBtn.textContent = '✕';
                closeBtn.style.cssText = `
                    position: absolute !important;
                    top: 20px !important;
                    right: 20px !important;
                    width: 40px !important;
                    height: 40px !important;
                    background: rgba(255, 255, 255, 0.2) !important;
                    border: none !important;
                    border-radius: 50% !important;
                    color: white !important;
                    font-size: 20px !important;
                    cursor: pointer !important;
                    z-index: 10 !important;
                `;

                closeBtn.onclick = () => {
                    document.body.removeChild(testLightbox);
                    document.body.style.overflow = '';
                };

                testLightbox.onclick = (e) => {
                    if (e.target === testLightbox) {
                        document.body.removeChild(testLightbox);
                        document.body.style.overflow = '';
                    }
                };

                mediaElement.onclick = (e) => {
                    e.stopPropagation();
                };

                testLightbox.appendChild(mediaElement);
                testLightbox.appendChild(closeBtn);
                document.body.appendChild(testLightbox);
                document.body.style.overflow = 'hidden';
            },

            closeLightbox() {
                this.lightbox.isOpen = false;
                this.lightbox.zoom = 1;
                this.lightbox.imagePosition = { x: 0, y: 0 };
                this.lightbox.isDragging = false;

                document.body.style.overflow = '';

                if (this._boundKeydownHandler) {
                    document.removeEventListener('keydown', this._boundKeydownHandler);
                    this._boundKeydownHandler = null;
                }
            },

            getLightboxImageStyle() {
                const transform = `translate(${this.lightbox.imagePosition.x}px, ${this.lightbox.imagePosition.y}px) scale(${this.lightbox.zoom})`;
                return {
                    transform: transform,
                    cursor: this.lightbox.isDragging ? 'grabbing' : 'grab'
                };
            },

            navigateLightbox(direction) {
                const newIndex = this.lightbox.currentIndex + direction;
                if (newIndex >= 0 && newIndex < this.lightbox.images.length) {
                    this.lightbox.currentIndex = newIndex;
                    this.lightbox.zoom = 1;
                    this.lightbox.imagePosition = { x: 0, y: 0 };
                }
            },

            zoomLightbox(action) {
                switch (action) {
                    case 'in':
                        this.lightbox.zoom = Math.min(3, this.lightbox.zoom + 0.25);
                        break;
                    case 'out':
                        this.lightbox.zoom = Math.max(0.5, this.lightbox.zoom - 0.25);
                        break;
                    case 'reset':
                        this.lightbox.zoom = 1;
                        this.lightbox.imagePosition = { x: 0, y: 0 };
                        break;
                }
            },

            handleLightboxWheel(event) {
                event.preventDefault();
                const delta = event.deltaY > 0 ? -0.1 : 0.1;
                this.lightbox.zoom = Math.max(0.5, Math.min(3, this.lightbox.zoom + delta));
            },

            startLightboxDrag(event) {
                if (this.lightbox.zoom <= 1) return;

                this.lightbox.isDragging = true;
                this.lightbox.dragStart = {
                    x: event.clientX - this.lightbox.imagePosition.x,
                    y: event.clientY - this.lightbox.imagePosition.y
                };

                const handleMouseMove = (e) => {
                    if (!this.lightbox.isDragging) return;

                    this.lightbox.imagePosition = {
                        x: e.clientX - this.lightbox.dragStart.x,
                        y: e.clientY - this.lightbox.dragStart.y
                    };
                };

                const handleMouseUp = () => {
                    this.lightbox.isDragging = false;
                    document.removeEventListener('mousemove', handleMouseMove);
                    document.removeEventListener('mouseup', handleMouseUp);
                };

                document.addEventListener('mousemove', handleMouseMove);
                document.addEventListener('mouseup', handleMouseUp);
            },

            handleLightboxKeydown(event) {
                if (!this.lightbox.isOpen) return;

                switch (event.key) {
                    case 'Escape':
                        this.closeLightbox();
                        break;
                    case 'ArrowLeft':
                        this.navigateLightbox(-1);
                        break;
                    case 'ArrowRight':
                        this.navigateLightbox(1);
                        break;
                    case '+':
                    case '=':
                        this.zoomLightbox('in');
                        break;
                    case '-':
                    case '_':
                        this.zoomLightbox('out');
                        break;
                    case '0':
                        this.zoomLightbox('reset');
                        break;
                }
            },

            onLightboxImageLoad() {
                console.log('Lightbox image loaded');
            },

            getCheckpointImages(checkpoint) {
                const images = [];

                if (checkpoint.assets && checkpoint.assets.length > 0) {
                    checkpoint.assets.forEach((asset, index) => {
                        images.push({
                            src: asset.data,
                            caption: asset.name || `Step ${checkpoint.step} - Validation ${index + 1}`
                        });
                    });
                }

                return images;
            },

            handleImageClick(event, checkpoint, imageIndex = null) {
                const images = this.getCheckpointImages(checkpoint);
                const currentIndex = imageIndex !== null ? imageIndex : images.length - 1;
                const currentImage = images[currentIndex];

                if (currentImage) {
                    this.openLightbox(event, currentImage.src, currentImage.caption, images, currentIndex);
                }
            },

            // Cloud Outputs Methods
            async loadCloudOutputs() {
                this.loading.cloudOutputs = true;
                try {
                    const response = await fetch('/api/cloud/storage');
                    if (response.ok) {
                        const data = await response.json();
                        this.cloudOutputs = data.Buckets || [];
                    } else {
                        this.cloudOutputs = [];
                    }
                } catch (error) {
                    console.error('Failed to load cloud outputs:', error);
                    this.cloudOutputs = [];
                } finally {
                    this.loading.cloudOutputs = false;
                }
            },

            browseCloudOutput(bucketName) {
                // Open a modal or navigate to browse the bucket contents
                // For now, show an alert with bucket info
                window.open(`/api/cloud/storage/${bucketName}`, '_blank');
            }
        };
    };
}
