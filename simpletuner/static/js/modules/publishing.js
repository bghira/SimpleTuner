/**
 * Publishing Module - Minimal Authentication Only
 * Handles HuggingFace Hub authentication functionality
 * Form fields are now managed by the dependency manager
 */

// Publishing authentication functionality
document.addEventListener('alpine:init', () => {
    // Authentication functions that can be used by inline Alpine components
    window.publishingAuth = {
        // Initialize state for Alpine components
        init() {
            return {
                loading: { auth: true },
                authStatus: null,
                showLoginModal: false,
                loginToken: '',
                savingToken: false,
                loggingOut: false,
                checkAuthStatus: this.checkAuthStatus,
                loginWithToken: this.loginWithToken,
                logout: this.logout
            };
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

                // Trigger a refresh of any components that need auth status
                window.dispatchEvent(new CustomEvent('token-updated', { detail: data }));

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

                // Trigger a refresh of any components that need auth status
                window.dispatchEvent(new CustomEvent('token-updated', { detail: null }));

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

    window.hfRepoSelector = {
        init(fieldId, initialValue) {
            const initial = typeof initialValue === 'string' ? initialValue : '';
            return {
                fieldId,
                initialValue: initial,
                namespaces: [],
                selectedNamespace: '',
                repoName: '',
                fullRepo: initial,
                loading: false,
                checking: false,
                error: '',
                checkResult: null,

                initialize() {
                    this.parseInitial();
                    this.syncHiddenField();
                    this.updateFullRepo();
                    this.loadNamespaces();
                },

                parseInitial() {
                    if (!this.initialValue) {
                        this.fullRepo = '';
                        this.repoName = '';
                        return;
                    }

                    const parts = this.initialValue.split('/');
                    if (parts.length > 1) {
                        this.selectedNamespace = parts.shift();
                        this.repoName = parts.join('/');
                    } else {
                        this.repoName = this.initialValue;
                    }
                    this.fullRepo = this.initialValue;
                },

                async loadNamespaces(force = false) {
                    if (this.loading) {
                        return;
                    }

                    if (!force && this.namespaces.length > 0) {
                        return;
                    }

                    this.loading = true;
                    this.error = '';

                    try {
                        const response = await fetch('/api/publishing/namespaces');
                        if (!response.ok) {
                            const detail = await response.json().catch(() => ({}));
                            throw new Error(detail.detail || 'Failed to load namespaces');
                        }

                        const data = await response.json();
                        let list = Array.isArray(data.namespaces) ? data.namespaces.filter(Boolean) : [];

                        if (this.selectedNamespace && !list.includes(this.selectedNamespace)) {
                            list = [this.selectedNamespace, ...list];
                        }

                        this.namespaces = Array.from(new Set(list));

                        if (!this.selectedNamespace && this.namespaces.length) {
                            this.selectedNamespace = this.namespaces[0];
                        }
                    } catch (error) {
                        console.error('[Publishing] namespace load failed:', error);
                        this.error = error.message || 'Unable to load namespaces';
                    } finally {
                        this.loading = false;
                        this.updateFullRepo();
                    }
                },

                onRepoInput(event) {
                    if (event && typeof event.target.value === 'string') {
                        this.repoName = event.target.value;
                    }
                    this.updateFullRepo();
                },

                updateFullRepo() {
                    let repo = (this.repoName || '').trim();

                    if (repo.includes('/')) {
                        const parts = repo.split('/').filter(Boolean);
                        if (parts.length >= 2) {
                            const typedNamespace = parts.shift();
                            const typedRepo = parts.join('/');

                            if (typedNamespace) {
                                this.selectedNamespace = typedNamespace;
                                if (!this.namespaces.includes(typedNamespace)) {
                                    this.namespaces = [typedNamespace, ...this.namespaces];
                                }
                            }

                            repo = typedRepo;
                            this.repoName = typedRepo;
                        }
                    }

                    const namespace = (this.selectedNamespace || '').trim();

                    if (namespace && repo) {
                        this.fullRepo = `${namespace}/${repo}`;
                    } else {
                        this.fullRepo = '';
                    }

                    this.clearStatus();
                    this.syncHiddenField();
                },

                syncHiddenField() {
                    if (!this.$refs || !this.$refs.hiddenField) {
                        return;
                    }

                    const hidden = this.$refs.hiddenField;
                    const previous = hidden.value;
                    hidden.value = this.fullRepo;

                    if (previous !== this.fullRepo) {
                        hidden.dispatchEvent(new Event('input', { bubbles: true }));
                        hidden.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                },

                clearStatus() {
                    this.checkResult = null;
                },

                async checkAvailability() {
                    if (!this.fullRepo || !this.fullRepo.includes('/')) {
                        this.checkResult = { type: 'warning', message: 'Enter namespace/model before checking.' };
                        return;
                    }

                    this.checking = true;
                    this.checkResult = null;

                    try {
                        const response = await fetch('/api/publishing/repository/check', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ repo_id: this.fullRepo })
                        });

                        if (!response.ok) {
                            const error = await response.json().catch(() => ({}));
                            throw new Error(error.detail || 'Failed to check repository');
                        }

                        const result = await response.json();

                        if (result.available) {
                            this.checkResult = {
                                type: 'success',
                                message: result.message || 'Repository name is available.'
                            };
                        } else if (result.exists) {
                            this.checkResult = {
                                type: 'warning',
                                message: result.message || 'Repository already exists.'
                            };
                        } else {
                            this.checkResult = {
                                type: 'info',
                                message: result.message || 'Check completed.'
                            };
                        }
                    } catch (error) {
                        console.error('[Publishing] repository check failed:', error);
                        this.checkResult = {
                            type: 'error',
                            message: error.message || 'Failed to check repository'
                        };
                    } finally {
                        this.checking = false;
                    }
                },

                get checkStatusMessage() {
                    return this.checkResult ? this.checkResult.message : '';
                },

                get statusClass() {
                    if (!this.checkResult) {
                        return 'status-info';
                    }

                    switch (this.checkResult.type) {
                        case 'success':
                            return 'status-success';
                        case 'warning':
                            return 'status-warning';
                        case 'error':
                            return 'status-error';
                        default:
                            return 'status-info';
                    }
                }
            };
        }
    };
});

// Initialize conditional fields for publishing tab
document.addEventListener('DOMContentLoaded', function() {
    initializeConditionalFields();
});

function initializeConditionalFields() {
    const conditionalFields = document.querySelectorAll('.conditional-field');
    conditionalFields.forEach(field => {
        const condition = field.dataset.conditionalOn;
        if (condition) {
            updateFieldVisibility(field, condition);

            // Listen for changes on the controlling field
            const controlField = document.querySelector(`[name="${condition}"]`);
            if (controlField) {
                controlField.addEventListener('change', () => {
                    updateFieldVisibility(field, condition);
                });
            }
        }
    });
}

function updateFieldVisibility(field, condition) {
    const controlField = document.querySelector(`[name="${condition}"]`);
    if (controlField) {
        const isActive = controlField.type === 'checkbox'
            ? controlField.checked
            : Boolean(controlField.value);

        field.classList.toggle('active', isActive);
        field.querySelectorAll('input, select, textarea').forEach(input => {
            input.disabled = !isActive;
        });
    }
}

// Reinitialize after HTMX swaps
document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail.target.id && evt.detail.target.id.includes('publishing-tab-content')) {
        initializeConditionalFields();
    }
});
