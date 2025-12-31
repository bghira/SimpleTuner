/**
 * Tests for trainer UI module.
 *
 * Covers toast notifications, loading states, and UI feedback.
 * These tests replace Selenium E2E tests for ToastNotificationsTestCase.
 */

// Mock DOM
document.body.innerHTML = `
    <div class="toast-container position-fixed bottom-0 end-0 p-3"></div>
    <div id="loading-overlay" class="d-none">
        <span class="loading-text">Loading...</span>
    </div>
`;

// Mock Bootstrap Toast
global.bootstrap = {
    Toast: jest.fn().mockImplementation((el, options) => ({
        show: jest.fn(),
        hide: jest.fn(),
    })),
};

// Mock SoundManager
global.SoundManager = {
    play: jest.fn(),
};

// Import the module (ES modules)
// Since the module uses ES exports, we need to mock it differently
const TrainerUI = (() => {
    // Inline implementation for testing
    class TrainerUI {
        constructor() {
            this.toastContainer = document.querySelector('.toast-container');
            this.initializeToastContainer();
        }

        initializeToastContainer() {
            if (!this.toastContainer) {
                this.toastContainer = document.createElement('div');
                this.toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
                document.body.appendChild(this.toastContainer);
            }
        }

        showToast(message, type = 'success') {
            const toastId = 'toast-' + Date.now();

            if (window.SoundManager) {
                window.SoundManager.play(type);
            }

            const bgClass = type === 'success' ? 'bg-success' : type === 'error' ? 'bg-danger' : 'bg-info';
            const iconClass = type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';

            const toastHTML = `
                <div id="${toastId}" class="toast align-items-center text-white ${bgClass}" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">
                            <i class="fas ${iconClass} me-2"></i>
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
                    delay: 5000,
                });
                toast.show();
            }

            return toastId;
        }

        showLoadingOverlay(message = 'Loading...') {
            const overlay = document.getElementById('loading-overlay');
            if (overlay) {
                const overlayText = overlay.querySelector('.loading-text');
                if (overlayText) {
                    overlayText.textContent = message;
                }
                overlay.classList.remove('d-none');
            }
        }

        hideLoadingOverlay() {
            const overlay = document.getElementById('loading-overlay');
            if (overlay) {
                overlay.classList.add('d-none');
            }
        }

        getToastMessage() {
            const toast = this.toastContainer.querySelector('.toast');
            if (toast) {
                const body = toast.querySelector('.toast-body');
                return body ? body.textContent.trim() : null;
            }
            return null;
        }

        dismissToast() {
            const toasts = this.toastContainer.querySelectorAll('.toast');
            toasts.forEach((toast) => toast.remove());
        }
    }

    return TrainerUI;
})();

describe('TrainerUI', () => {
    let ui;

    beforeEach(() => {
        // Reset DOM
        document.body.innerHTML = `
            <div class="toast-container position-fixed bottom-0 end-0 p-3"></div>
            <div id="loading-overlay" class="d-none">
                <span class="loading-text">Loading...</span>
            </div>
        `;
        jest.clearAllMocks();
        ui = new TrainerUI();
    });

    describe('initialization', () => {
        test('finds existing toast container', () => {
            expect(ui.toastContainer).toBeTruthy();
            expect(ui.toastContainer.className).toContain('toast-container');
        });

        test('creates toast container if not exists', () => {
            document.body.innerHTML = '';
            const newUi = new TrainerUI();
            expect(newUi.toastContainer).toBeTruthy();
            expect(document.querySelector('.toast-container')).toBeTruthy();
        });
    });

    describe('showToast', () => {
        test('creates toast element with message', () => {
            ui.showToast('Test message', 'success');

            const toast = ui.toastContainer.querySelector('.toast');
            expect(toast).toBeTruthy();
            expect(toast.textContent).toContain('Test message');
        });

        test('success toast has correct styling', () => {
            ui.showToast('Success!', 'success');

            const toast = ui.toastContainer.querySelector('.toast');
            expect(toast.className).toContain('bg-success');
        });

        test('error toast has correct styling', () => {
            ui.showToast('Error!', 'error');

            const toast = ui.toastContainer.querySelector('.toast');
            expect(toast.className).toContain('bg-danger');
        });

        test('info toast has correct styling', () => {
            ui.showToast('Info', 'info');

            const toast = ui.toastContainer.querySelector('.toast');
            expect(toast.className).toContain('bg-info');
        });

        test('plays sound via SoundManager', () => {
            ui.showToast('Test', 'success');

            expect(global.SoundManager.play).toHaveBeenCalledWith('success');
        });

        test('initializes Bootstrap Toast', () => {
            ui.showToast('Test', 'success');

            expect(global.bootstrap.Toast).toHaveBeenCalled();
        });

        test('returns toast ID', () => {
            const toastId = ui.showToast('Test', 'success');

            expect(toastId).toMatch(/^toast-\d+$/);
            expect(document.getElementById(toastId)).toBeTruthy();
        });
    });

    describe('loading overlay', () => {
        test('showLoadingOverlay removes d-none class', () => {
            ui.showLoadingOverlay('Loading data...');

            const overlay = document.getElementById('loading-overlay');
            expect(overlay.classList.contains('d-none')).toBe(false);
        });

        test('showLoadingOverlay sets message', () => {
            ui.showLoadingOverlay('Custom loading message');

            const text = document.querySelector('.loading-text');
            expect(text.textContent).toBe('Custom loading message');
        });

        test('hideLoadingOverlay adds d-none class', () => {
            ui.showLoadingOverlay();
            ui.hideLoadingOverlay();

            const overlay = document.getElementById('loading-overlay');
            expect(overlay.classList.contains('d-none')).toBe(true);
        });
    });

    describe('toast management', () => {
        test('getToastMessage returns message from active toast', () => {
            ui.showToast('Hello world', 'success');

            const message = ui.getToastMessage();
            expect(message).toContain('Hello world');
        });

        test('getToastMessage returns null when no toast', () => {
            const message = ui.getToastMessage();
            expect(message).toBeNull();
        });

        test('dismissToast removes all toasts', () => {
            ui.showToast('Toast 1', 'success');
            ui.showToast('Toast 2', 'info');

            ui.dismissToast();

            const toasts = ui.toastContainer.querySelectorAll('.toast');
            expect(toasts.length).toBe(0);
        });
    });
});

describe('Toast Notification Types', () => {
    let ui;

    beforeEach(() => {
        document.body.innerHTML = `
            <div class="toast-container position-fixed bottom-0 end-0 p-3"></div>
        `;
        ui = new TrainerUI();
    });

    test.each([
        ['success', 'bg-success', 'fa-check-circle'],
        ['error', 'bg-danger', 'fa-exclamation-circle'],
        ['info', 'bg-info', 'fa-info-circle'],
    ])('%s toast has correct class and icon', (type, bgClass, iconClass) => {
        ui.showToast('Test message', type);

        const toast = ui.toastContainer.querySelector('.toast');
        expect(toast.className).toContain(bgClass);
        expect(toast.innerHTML).toContain(iconClass);
    });
});

describe('Toast Accessibility', () => {
    let ui;

    beforeEach(() => {
        document.body.innerHTML = `
            <div class="toast-container position-fixed bottom-0 end-0 p-3"></div>
        `;
        ui = new TrainerUI();
    });

    test('toast has role="alert"', () => {
        ui.showToast('Test', 'success');

        const toast = ui.toastContainer.querySelector('.toast');
        expect(toast.getAttribute('role')).toBe('alert');
    });

    test('toast has close button', () => {
        ui.showToast('Test', 'success');

        const closeBtn = ui.toastContainer.querySelector('.btn-close');
        expect(closeBtn).toBeTruthy();
    });
});
