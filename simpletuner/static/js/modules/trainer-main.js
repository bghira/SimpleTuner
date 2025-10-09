/**
 * Main Trainer Module
 * Coordinates all trainer modules and provides the main interface
 */

import { TrainerValidation } from './trainer-validation.js';
import { TrainerActions } from './trainer-actions.js';
import { TrainerForm } from './trainer-form.js';
import { TrainerUI } from './trainer-ui.js';

class TrainerMain {
    constructor() {
        this.apiBaseUrl = null;
        this.callbackUrl = null;
        this.initialized = false;

        // Module instances
        this.validation = null;
        this.actions = null;
        this.form = null;
        this.ui = null;
    }

    async init() {
        if (this.initialized) {
            console.warn('TrainerMain already initialized');
            return;
        }

        // Wait for server configuration
        await window.ServerConfig.waitForReady();

        // Set URLs
        this.apiBaseUrl = window.ServerConfig.apiBaseUrl;
        this.callbackUrl = window.ServerConfig.callbackUrl;

        // Initialize modules
        this.validation = new TrainerValidation(this.apiBaseUrl, this.callbackUrl);
        this.actions = new TrainerActions(this.apiBaseUrl, this.callbackUrl);
        this.form = new TrainerForm();
        this.ui = new TrainerUI();

        // Setup components
        this.setupButtonHandlers();
        this.form.initializeDependencies();
        this.form.setupJSONEditors();
        this.ui.setupSidebarToggle();
        this.updateWebhookConfig();

        // Setup HTMX event handlers
        this.setupHTMXHandlers();

        // Initialize dependency manager if available
        if (window.dependencyManager) {
            await window.dependencyManager.initialize();
            window.dependencyManager.initializeFieldsInContainer(this.form.form);
        }

        window.TrainerActionsInstance = this.actions;

        await this.refreshStatus();

        this.initialized = true;
        console.log('TrainerMain initialized successfully');
    }

    /**
     * Setup button event handlers
     */
    setupButtonHandlers() {
        // For HTMX-enabled buttons, we don't need click handlers
        // But we can add HTMX event listeners for feedback

        document.body.addEventListener('htmx:beforeRequest', (evt) => {
            const button = evt.detail.elt;
            if (button.id === 'validateBtn' || button.id === 'runBtn' || button.id === 'cancelBtn') {
                this.ui.addLoadingSpinner(button);
            }
        });

        document.body.addEventListener('htmx:afterRequest', (evt) => {
            const button = evt.detail.elt;
            if (button.id === 'validateBtn' || button.id === 'runBtn' || button.id === 'cancelBtn') {
                this.ui.removeLoadingSpinner(button);

                if (evt.detail.successful) {
                    this.handleSuccessResponse(button.id);
                } else {
                    this.ui.showToast('Operation failed', 'error');
                }
            }
        });

        // Fallback for non-HTMX buttons
        const validateBtn = document.getElementById('validateBtn');
        const runBtn = document.getElementById('runBtn');
        const cancelBtn = document.getElementById('cancelBtn');

        if (validateBtn && !validateBtn.hasAttribute('hx-post')) {
            validateBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleValidate();
            });
        }

        if (runBtn && !runBtn.hasAttribute('hx-post')) {
            runBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleRun();
            });
        }

        if (cancelBtn && !cancelBtn.hasAttribute('hx-post')) {
            cancelBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleCancel();
            });
        }
    }

    /**
     * Setup HTMX event handlers
     */
    setupHTMXHandlers() {
        // Handle HTMX errors
        document.body.addEventListener('htmx:responseError', (evt) => {
            console.error('HTMX error:', evt.detail);
            this.ui.showToast('Network error occurred', 'error');
        });

        // Handle form swaps
        document.body.addEventListener('htmx:afterSwap', (evt) => {
            // Reinitialize components if needed
            if (evt.detail.target.id === 'tab-content') {
                this.form.initializeDependencies();
                this.form.setupJSONEditors();
            }
        });
    }

    /**
     * Handle validation (fallback for non-HTMX)
     */
    async handleValidate() {
        try {
            this.ui.showLoadingOverlay('Validating configuration...');
            const html = await this.validation.validateConfig(this.form.form);
            this.ui.updateResultContainer('validation-results', html);
            this.ui.showToast('Validation completed', 'success');
        } catch (error) {
            console.error('Validation error:', error);
            this.ui.showToast(error.message, 'error');
        } finally {
            this.ui.hideLoadingOverlay();
        }
    }

    /**
     * Handle run training (fallback for non-HTMX)
     */
    async handleRun() {
        try {
            const payload = this.validation.getPayload(this.form.form);
            this.ui.showLoadingOverlay('Starting training...');
            const html = await this.actions.startTraining(payload);
            this.ui.updateResultContainer('training-status', html);
            this.actions.updateButtonStates(true);

            // Initialize event handler if available
            if (window.eventHandler) {
                window.eventHandler.resetEventList();
                window.eventHandler.lastEventIndex = 0;
            }
        } catch (error) {
            console.error('Training start error:', error);
            this.ui.showToast(error.message, 'error');
        } finally {
            this.ui.hideLoadingOverlay();
        }
    }

    /**
     * Handle cancel training (fallback for non-HTMX)
     */
    async handleCancel() {
        const jobId = document.getElementById('job_id')?.value;
        if (!jobId) {
            this.ui.showToast('No active training job', 'warning');
            return;
        }

        try {
            this.ui.showLoadingOverlay('Cancelling training...');
            const html = await this.actions.cancelTraining(jobId);
            this.ui.updateResultContainer('training-status', html);
            this.actions.updateButtonStates(false);
        } catch (error) {
            console.error('Cancel error:', error);
            this.ui.showToast(error.message, 'error');
        } finally {
            this.ui.hideLoadingOverlay();
        }
    }

    /**
     * Handle successful response
     */
    handleSuccessResponse(buttonId) {
        const messages = {
            'validateBtn': 'Configuration validated successfully!',
            'runBtn': 'Training started successfully!',
            'cancelBtn': 'Training cancelled successfully!'
        };

        const message = messages[buttonId];
        if (message) {
            if (window.__trainerActionToastFallback) {
                delete window.__trainerActionToastFallback;
                return;
            }
            this.ui.showToast(message, 'success');
        }

        // Update button states for run/cancel
        if (buttonId === 'runBtn') {
            this.actions.updateButtonStates(true);
        } else if (buttonId === 'cancelBtn') {
            this.actions.updateButtonStates(false);
        }
    }

    /**
     * Update webhook configuration
     */
    updateWebhookConfig() {
        const webhookTextarea = document.getElementById('webhook_config');
        if (webhookTextarea && window.ServerConfig.mode === 'unified') {
            try {
                const config = JSON.parse(webhookTextarea.value);
                config.callback_url = `${this.callbackUrl}/callback`;
                webhookTextarea.value = JSON.stringify(config, null, 4);
            } catch (e) {
                console.error('Failed to update webhook config:', e);
            }
        }
    }

    /**
     * Public API methods
     */
    showToast(message, type) {
        this.ui.showToast(message, type);
    }

    showError(message) {
        this.ui.showToast(message, 'error');
    }

    getFormData() {
        return this.form.getFormData();
    }

    setFormData(data) {
        this.form.setFormData(data);
    }

    async refreshStatus() {
        try {
            const statusPayload = await this.actions.getTrainingStatus();
            const status = (statusPayload.status || '').toLowerCase();
            const jobId = statusPayload.job_id || null;
            const isTraining = ['running', 'starting', 'initializing'].includes(status) || (jobId && !['completed', 'idle', 'cancelled', 'error', 'failed'].includes(status));

            this.actions.updateButtonStates(isTraining);

            const store = window.Alpine && typeof window.Alpine.store === 'function' ? window.Alpine.store('trainer') : null;

            const normalizeProgress = (progress) => {
                if (!progress) {
                    return null;
                }
                if (typeof progress === 'object' && !Array.isArray(progress) && Object.keys(progress).length === 0) {
                    return null;
                }
                if (progress.reset) {
                    return { reset: true };
                }
                const percentValue = Number(progress.percent || progress.percentage || 0);
                const clampedPercent = Number.isFinite(percentValue) ? Math.max(0, Math.min(100, percentValue)) : 0;
                return {
                    percent: clampedPercent,
                    step: progress.step || progress.current_step || 0,
                    total_steps: progress.total_steps || progress.total || 0,
                    epoch: progress.epoch || 0,
                    loss: progress.loss ?? 'N/A',
                    learning_rate: progress.learning_rate ?? progress.lr ?? 'N/A',
                };
            };

            const normalizedProgress = normalizeProgress(statusPayload.progress);

            if (store) {
                store.isTraining = isTraining;
                if (normalizedProgress) {
                    if (normalizedProgress.reset) {
                        store.trainingProgress = {};
                    } else {
                        store.trainingProgress = normalizedProgress;
                    }
                }
            }

            const detail = {
                status: status || (isTraining ? 'running' : 'idle'),
                job_id: jobId,
            };
            if (normalizedProgress) {
                if (normalizedProgress.reset) {
                    detail.progress = { reset: true };
                } else {
                    detail.progress = normalizedProgress;
                }
            }

            window.dispatchEvent(new CustomEvent('training-status', { detail }));
            if (normalizedProgress) {
                if (normalizedProgress.reset) {
                    window.dispatchEvent(new CustomEvent('training-progress', { detail: { reset: true } }));
                } else {
                    window.dispatchEvent(new CustomEvent('training-progress', { detail: normalizedProgress }));
                }
            }

            if (document && document.body) {
                document.body.dataset.trainingActive = isTraining ? 'true' : 'false';
            }

            if (isTraining && window.initSSE) {
                window.initSSE();
            }

            const statusContainer = document.getElementById('training-status');
            if (statusContainer && isTraining && !statusContainer.innerHTML.trim()) {
                statusContainer.innerHTML = `
                    <div class="alert alert-info">
                        <h6 class="mb-1"><i class="fas fa-cog fa-spin"></i> Training In Progress</h6>
                        <p class="mb-0"><small>Job ID: ${jobId || 'pending'}</small></p>
                    </div>
                `;
            } else if (statusContainer && !isTraining) {
                statusContainer.innerHTML = '';
            }
        } catch (error) {
            console.warn('Unable to refresh training status', error);
        }
    }
}

// Export and initialize
window.TrainerMain = TrainerMain;

// Auto-initialize on DOM ready
document.addEventListener('DOMContentLoaded', async () => {
    if (!window.trainerMain) {
        window.trainerMain = new TrainerMain();
        await window.trainerMain.init();

        // Expose utility functions for backward compatibility
        window.showToast = (message, type) => window.trainerMain.showToast(message, type);
        window.showError = (message) => window.trainerMain.showError(message);
    }
});
