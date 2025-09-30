(function() {
    const STORAGE_KEY = 'simpletuner-active-tab';

    function getContentElementFromButton(button) {
        const targetId = button.getAttribute('data-bs-target');
        if (!targetId) {
            return null;
        }
        return document.querySelector(targetId.replace('-tab', '-tab-content'));
    }

    function ensureBootstrapTabs() {
        if (typeof bootstrap === 'undefined') {
            return;
        }

        const tabElements = document.querySelectorAll('.nav-tabs button[data-bs-toggle="tab"]');
        tabElements.forEach(button => {
            // Initialize tab instance for keyboard navigation support
            new bootstrap.Tab(button);

            button.addEventListener('shown.bs.tab', () => {
                const contentElement = getContentElementFromButton(button);
                if (contentElement) {
                    htmx.trigger(contentElement, 'tab-refresh');
                }
                localStorage.setItem(STORAGE_KEY, button.id);
            });
        });
    }

    function registerValidationBadgeUpdates() {
        document.body.addEventListener('validation-update', evt => {
            const badge = document.getElementById('validation-badge');
            if (!badge || !evt.detail) {
                return;
            }

            const errorCount = evt.detail.errorCount || 0;
            const warningCount = evt.detail.warningCount || 0;

            if (errorCount > 0) {
                badge.textContent = errorCount;
                badge.classList.remove('d-none', 'bg-success', 'bg-warning');
                badge.classList.add('bg-danger');
            } else if (warningCount > 0) {
                badge.textContent = warningCount;
                badge.classList.remove('d-none', 'bg-success', 'bg-danger');
                badge.classList.add('bg-warning');
            } else {
                badge.classList.add('d-none');
            }
        });
    }

    function registerConfigChangeReload() {
        document.body.addEventListener('config-changed', () => {
            const activeButton = document.querySelector('.nav-tabs .nav-link.active');
            if (!activeButton) {
                return;
            }
            const contentElement = getContentElementFromButton(activeButton);
            if (contentElement) {
                htmx.trigger(contentElement, 'tab-refresh');
            }
        });
    }

    function registerKeyboardShortcuts() {
        document.addEventListener('keydown', event => {
            if (!event.ctrlKey && !event.metaKey) {
                return;
            }

            const buttons = Array.from(document.querySelectorAll('.nav-tabs button[data-bs-toggle="tab"]'));
            if (buttons.length === 0) {
                return;
            }

            const activeButton = document.querySelector('.nav-tabs button[data-bs-toggle="tab"].active');
            const currentIndex = buttons.indexOf(activeButton);

            if (event.key >= '1' && event.key <= '9') {
                const index = parseInt(event.key, 10) - 1;
                if (index < buttons.length) {
                    event.preventDefault();
                    buttons[index].click();
                }
            } else if (event.key === 'ArrowLeft' && currentIndex > 0) {
                event.preventDefault();
                buttons[currentIndex - 1].click();
            } else if (event.key === 'ArrowRight' && currentIndex < buttons.length - 1) {
                event.preventDefault();
                buttons[currentIndex + 1].click();
            }
        });
    }

    function registerHtmxSwapEvents() {
        document.body.addEventListener('htmx:afterSwap', evt => {
            if (!evt.detail.target || !evt.detail.target.id) {
                return;
            }

            if (evt.detail.target.id.endsWith('-tab-content')) {
                document.dispatchEvent(new CustomEvent('tab-loaded', {
                    detail: {
                        tabId: evt.detail.target.id,
                        tab: evt.detail.target.closest('.tab-pane')
                    }
                }));
            }
        });
    }

    document.addEventListener('DOMContentLoaded', () => {
        ensureBootstrapTabs();
        registerValidationBadgeUpdates();
        registerConfigChangeReload();
        registerKeyboardShortcuts();
        registerHtmxSwapEvents();
    });
})();
