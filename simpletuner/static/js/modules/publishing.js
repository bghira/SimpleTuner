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
