/**
 * Trainer UI Module
 * Handles UI feedback, toasts, and loading states
 */

export class TrainerUI {
    constructor() {
        this.toastContainer = document.querySelector('.toast-container');
        this.initializeToastContainer();
    }

    /**
     * Initialize toast container if not exists
     */
    initializeToastContainer() {
        if (!this.toastContainer) {
            this.toastContainer = document.createElement('div');
            this.toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            document.body.appendChild(this.toastContainer);
        }
    }

    /**
     * Show toast notification
     */
    showToast(message, type = 'success') {
        const toastId = 'toast-' + Date.now();

        // Play sound for this notification type
        if (window.SoundManager) {
            window.SoundManager.play(type);
        }

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

        this.toastContainer.insertAdjacentHTML('beforeend', toastHTML);
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
            // Fallback without Bootstrap
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

    /**
     * Show loading overlay
     */
    showLoadingOverlay(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay') || this.createLoadingOverlay();
        const overlayText = overlay.querySelector('.loading-text');

        if (overlayText) {
            overlayText.textContent = message;
        }

        overlay.classList.remove('d-none');
    }

    /**
     * Hide loading overlay
     */
    hideLoadingOverlay() {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.add('d-none');
        }
    }

    /**
     * Create loading overlay element
     */
    createLoadingOverlay() {
        const overlay = document.createElement('div');
        overlay.id = 'loading-overlay';
        overlay.className = 'position-fixed top-0 start-0 w-100 h-100 d-none';
        overlay.style.cssText = 'background: rgba(0,0,0,0.5); z-index: 9999;';

        overlay.innerHTML = `
            <div class="d-flex justify-content-center align-items-center h-100">
                <div class="text-center text-white">
                    <div class="spinner-border mb-3" role="status" style="width: 3rem; height: 3rem;">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <div class="loading-text">Loading...</div>
                </div>
            </div>
        `;

        document.body.appendChild(overlay);
        return overlay;
    }

    /**
     * Show status message
     */
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

    /**
     * Update result container with HTML
     */
    updateResultContainer(containerId, html) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = html;
            // Trigger HTMX processing on new content
            if (window.htmx) {
                htmx.process(container);
            }
        }
    }

    /**
     * Add loading spinner to element
     */
    addLoadingSpinner(element) {
        if (!element) return;

        element.dataset.originalContent = element.innerHTML;
        element.disabled = true;
        element.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status">
                <span class="visually-hidden">Loading...</span>
            </span>
            Loading...
        `;
    }

    /**
     * Remove loading spinner from element
     */
    removeLoadingSpinner(element) {
        if (!element || !element.dataset.originalContent) return;

        element.innerHTML = element.dataset.originalContent;
        element.disabled = false;
        delete element.dataset.originalContent;
    }

    /**
     * Setup sidebar toggle
     */
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
}
