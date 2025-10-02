/**
 * Publishing Module
 * Handles HuggingFace Hub publishing configuration
 */

if (!window.publishingManager) {
    window.publishingManager = function() {
        return {
            // State
            config: {
                push_to_hub: false,
                push_checkpoints_to_hub: false,
                namespace: '',
                model_name: '',
                private: false,
                safe_for_work: false
            },
            original: {},
            authStatus: null,
            namespaces: [],
            repoCheck: {
                status: '', // '', 'checking', 'exists', 'available', 'error'
                message: ''
            },
            license: {
                name: '',
                text: '',
                url: '',
                source: ''
            },
            loading: {
                config: false,
                auth: true,
                namespaces: false,
                save: false,
                repoCheck: false
            },
            saveStatus: '', // '', 'success', 'error'
            errorMessage: '',
            isDirty: false,
            repositoryCheckTimeout: null,
            showLoginModal: false,
            loginToken: '',
            savingToken: false,
            loggingOut: false,
            _debug: true,

            // Lifecycle
            async init() {
                console.log('[Publishing] Initializing publishing manager...');

                // Check auth status first
                await this.checkAuthStatus();

                await this.loadConfig();
                await this.loadNamespaces();

                // If model_name exists but namespace is empty, use first namespace (username) as default
                if (this.config.model_name && !this.config.namespace && this.namespaces.length > 0) {
                    console.log('[Publishing] Using default namespace:', this.namespaces[0]);
                    this.config.namespace = this.namespaces[0];
                    this.original.namespace = this.namespaces[0];
                    // Check repository now that we have both namespace and model name
                    await this.checkRepository();
                }

                this.updateLicense();
                console.log('[Publishing] Initialization complete. Config:', this.config);

                // Watch for config changes and automatically mark dirty
                // Using deep watch to detect changes to nested properties
                if (this.$watch) {
                    console.log('[Publishing] Setting up config watcher...');
                    this.$watch('config', (value, oldValue) => {
                        console.log('[Publishing] Config changed:', { old: oldValue, new: value });
                        this.markDirty();
                    }, { deep: true });
                } else {
                    console.warn('[Publishing] $watch not available, using manual dirty tracking');
                }

                // Listen for global form save events to reset dirty state
                window.addEventListener('config-saved', () => {
                    console.log('[Publishing] Config saved event received, resetting dirty state');
                    this.original = this.clone(this.config);
                    this.isDirty = false;
                    this.saveStatus = '';
                    this.errorMessage = '';
                });

                // Listen for token update events to refresh auth status
                window.addEventListener('token-updated', () => {
                    console.log('[Publishing] Token updated event received, rechecking auth');
                    this.checkAuthStatus();
                });
            },

            // Data Loading
            async loadConfig() {
                this.loading.config = true;
                console.log('[Publishing] Loading configuration...');
                try {
                    const response = await fetch('/api/configs/active');
                    if (!response.ok) {
                        throw new Error('Failed to load configuration');
                    }
                    const data = await response.json();
                    console.log('[Publishing] Received config data:', data);

                    // Extract publishing-related config from data.config
                    const config = data.config || {};
                    const push_to_hub = Boolean(config.push_to_hub || config['--push_to_hub']);
                    const push_checkpoints_to_hub = Boolean(config.push_checkpoints_to_hub || config['--push_checkpoints_to_hub']);
                    const hub_model_id = config.hub_model_id || config['--hub_model_id'] || '';
                    const private_hub = Boolean(config.private_hub || config['--private_hub']);
                    const safe_for_work = Boolean(config.model_card_safe_for_work || config['--model_card_safe_for_work']);

                    // Parse hub_model_id - can be "namespace/model" or just "model"
                    let namespace = '';
                    let model_name = '';
                    if (hub_model_id.includes('/')) {
                        [namespace, model_name] = hub_model_id.split('/');
                    } else if (hub_model_id) {
                        // Just model name without namespace
                        model_name = hub_model_id;
                    }

                    this.config = {
                        push_to_hub: push_to_hub,
                        push_checkpoints_to_hub: push_checkpoints_to_hub,
                        namespace: namespace,
                        model_name: model_name,
                        private: private_hub,
                        safe_for_work: safe_for_work
                    };

                    console.log('[Publishing] Parsed config:', this.config);

                    // Store original config for dirty checking
                    this.original = this.clone(this.config);

                    // If we have a namespace and model name, check repository status
                    if (this.config.namespace && this.config.model_name) {
                        await this.checkRepository();
                    }
                } catch (error) {
                    console.error('[Publishing] Failed to load config:', error);
                    if (window.showToast) {
                        window.showToast('Failed to load configuration', 'error');
                    }
                } finally {
                    this.loading.config = false;
                }
            },

            async loadNamespaces() {
                this.loading.namespaces = true;
                try {
                    const response = await fetch('/api/publishing/namespaces');
                    if (!response.ok) {
                        throw new Error('Failed to load namespaces');
                    }
                    const data = await response.json();
                    this.namespaces = data.namespaces || [];
                } catch (error) {
                    console.error('Failed to load namespaces:', error);
                    this.namespaces = [];
                    // Don't show error toast if not authenticated - this is expected
                    if (error.message && !error.message.includes('401')) {
                        if (window.showToast) {
                            window.showToast('Failed to load namespaces. Please check your HuggingFace token.', 'warning');
                        }
                    }
                } finally {
                    this.loading.namespaces = false;
                }
            },

            async checkAuthStatus() {
                this.loading.auth = true;
                try {
                    const response = await fetch('/api/publishing/token/validate');
                    if (response.ok) {
                        this.authStatus = await response.json();
                    } else {
                        this.authStatus = { valid: false, message: 'Failed to check authentication status' };
                    }
                } catch (error) {
                    console.error('[Publishing] Auth check failed:', error);
                    this.authStatus = { valid: false, message: 'Failed to check authentication status' };
                } finally {
                    this.loading.auth = false;
                }
            },

            // Repository Validation
            async checkRepository() {
                if (!this.config.namespace || !this.config.model_name) {
                    this.repoCheck = { status: '', message: '' };
                    return;
                }

                this.repoCheck.status = 'checking';
                this.loading.repoCheck = true;

                try {
                    const repo_id = `${this.config.namespace}/${this.config.model_name}`;
                    const response = await fetch('/api/publishing/repository/check', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            repo_id: repo_id
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Repository check failed');
                    }

                    const data = await response.json();
                    this.repoCheck = {
                        status: data.exists ? 'exists' : 'available',
                        message: data.message || ''
                    };
                } catch (error) {
                    console.error('Failed to check repository:', error);
                    this.repoCheck = {
                        status: 'error',
                        message: error.message || 'Failed to check repository status'
                    };
                } finally {
                    this.loading.repoCheck = false;
                }
            },

            debounceRepositoryCheck() {
                // Clear existing timeout
                if (this.repositoryCheckTimeout) {
                    clearTimeout(this.repositoryCheckTimeout);
                }

                // Set new timeout for 500ms
                this.repositoryCheckTimeout = setTimeout(() => {
                    this.checkRepository();
                }, 500);
            },

            // License Management
            updateLicense() {
                // License info is typically derived from model family
                // This would be populated based on active config's model_family
                // For now, we'll set a placeholder that can be enhanced
                const licenseMap = {
                    'sdxl': {
                        name: 'OpenRAIL++-M License',
                        text: 'The model is licensed under the CreativeML OpenRAIL++-M License.',
                        url: 'https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md',
                        source: 'Stable Diffusion XL'
                    },
                    'flux': {
                        name: 'FLUX.1 [dev] Non-Commercial License',
                        text: 'This model is licensed under the FLUX.1 [dev] Non-Commercial License.',
                        url: 'https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md',
                        source: 'FLUX.1'
                    },
                    'pixart': {
                        name: 'OpenRAIL-M License',
                        text: 'The model is licensed under the CreativeML OpenRAIL-M License.',
                        url: 'https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/blob/main/LICENSE.md',
                        source: 'PixArt'
                    },
                    'sd3': {
                        name: 'Stable Diffusion 3 Community License',
                        text: 'This model is licensed under the Stable Diffusion 3 Medium Community License.',
                        url: 'https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE',
                        source: 'Stable Diffusion 3'
                    }
                };

                // Try to get model family from active config
                // This is a simplified approach - in production you'd fetch this from the config
                const modelFamily = this.getModelFamily();
                if (modelFamily && licenseMap[modelFamily]) {
                    this.license = licenseMap[modelFamily];
                } else {
                    // Default/unknown license
                    this.license = {
                        name: 'License Information',
                        text: 'License will be inherited from the base model. Please verify the license terms before publishing.',
                        url: '',
                        source: ''
                    };
                }
            },

            getModelFamily() {
                // This would ideally come from the active config
                // For now, return empty string to trigger default license
                // In production, you'd integrate with the trainer store or fetch from API
                const trainerStore = Alpine.store('trainer');
                if (trainerStore && trainerStore.config && trainerStore.config.model_family) {
                    return trainerStore.config.model_family;
                }
                return '';
            },

            // State Management
            markDirty() {
                const wasDirty = this.isDirty;
                this.isDirty = JSON.stringify(this.config) !== JSON.stringify(this.original);
                if (!wasDirty && this.isDirty) {
                    console.log('[Publishing] Config marked dirty');
                    // Mark the global form as dirty so the save button activates
                    if (Alpine.store('trainer') && Alpine.store('trainer').markFormDirty) {
                        Alpine.store('trainer').markFormDirty();
                    }
                }
                this.saveStatus = '';
                this.errorMessage = '';
            },

            get isValid() {
                if (!this.config.push_to_hub) {
                    return true; // Valid if publishing is disabled
                }
                // Valid if publishing is enabled and we have namespace + model_name
                const valid = Boolean(this.config.namespace && this.config.model_name);
                console.log('[Publishing] Validation check:', { valid, config: this.config });
                return valid;
            },

            clone(obj) {
                return JSON.parse(JSON.stringify(obj));
            },

            // Actions
            // Note: Publishing config is now saved as part of the main trainer form
            // via hidden fields. This method is kept for backwards compatibility
            // but triggers the global save instead.
            async saveConfig() {
                console.log('[Publishing] Save via global form...');

                // Trigger the global save dialog
                if (Alpine.store('trainer') && Alpine.store('trainer').saveConfig) {
                    await Alpine.store('trainer').saveConfig();
                }
            },

            discardChanges() {
                this.config = this.clone(this.original);
                this.isDirty = false;
                this.saveStatus = '';
                this.errorMessage = '';

                // Reset repository check
                if (this.config.namespace && this.config.model_name) {
                    this.checkRepository();
                } else {
                    this.repoCheck = { status: '', message: '' };
                }
            },

            // Authentication Management
            async loginWithToken(token) {
                console.log('[Publishing] Saving token...');
                this.savingToken = true;
                try {
                    const response = await fetch('/api/publishing/token/save', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ token: token })
                    });

                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to save token');
                    }

                    const data = await response.json();
                    console.log('[Publishing] Token saved successfully:', data);

                    if (window.showToast) {
                        window.showToast(`Logged in as ${data.username}`, 'success');
                    }

                    // Update auth status directly
                    this.authStatus = data;

                    // Close modal and clear token
                    this.showLoginModal = false;
                    this.loginToken = '';

                    // Reload namespaces now that we're authenticated
                    await this.loadNamespaces();

                    return { success: true, data };
                } catch (error) {
                    console.error('[Publishing] Failed to save token:', error);
                    if (window.showToast) {
                        window.showToast(error.message || 'Failed to save token', 'error');
                    }
                    return { success: false, error: error.message };
                } finally {
                    this.savingToken = false;
                }
            },

            async logout() {
                console.log('[Publishing] Logging out...');
                this.loggingOut = true;
                try {
                    const response = await fetch('/api/publishing/token/logout', {
                        method: 'POST'
                    });

                    if (!response.ok) {
                        const error = await response.json().catch(() => ({}));
                        throw new Error(error.detail || 'Failed to logout');
                    }

                    console.log('[Publishing] Logged out successfully');

                    if (window.showToast) {
                        window.showToast('Logged out successfully', 'success');
                    }

                    // Update auth status
                    this.authStatus = { valid: false, message: 'Not connected' };

                    // Clear namespaces
                    this.namespaces = [];

                    return { success: true };
                } catch (error) {
                    console.error('[Publishing] Failed to logout:', error);
                    if (window.showToast) {
                        window.showToast(error.message || 'Failed to logout', 'error');
                    }
                    return { success: false, error: error.message };
                } finally {
                    this.loggingOut = false;
                }
            }
        };
    };
}