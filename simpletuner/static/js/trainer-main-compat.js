/**
 * Trainer Main Module - Compatibility Version
 * For browsers without ES6 module support
 */

(function(window) {
    'use strict';

    /**
     * Main coordinator for the trainer application
     * Compatible version for older browsers
     */
    function TrainerMain() {
        this.initialized = false;
        this.modules = {};
    }

    TrainerMain.prototype.initialize = function() {
        if (this.initialized) {
            return;
        }

        console.log('Initializing Trainer (compatibility mode)');

        // Load required modules in compatibility mode
        var self = this;
        var modulesToLoad = [
            '/static/js/trainer-validation-compat.js',
            '/static/js/trainer-actions-compat.js',
            '/static/js/trainer-form-compat.js',
            '/static/js/trainer-ui-compat.js'
        ];

        // Load modules sequentially
        this.loadModulesSequentially(modulesToLoad, function() {
            self.setupEventListeners();
            self.setupHTMXIntegration();
            self.initialized = true;
            console.log('Trainer initialized successfully (compatibility mode)');
        });
    };

    TrainerMain.prototype.loadModulesSequentially = function(modules, callback) {
        var self = this;
        var index = 0;

        function loadNext() {
            if (index >= modules.length) {
                if (callback) callback();
                return;
            }

            var script = document.createElement('script');
            script.src = modules[index];
            script.onload = function() {
                index++;
                loadNext();
            };
            script.onerror = function() {
                console.error('Failed to load module:', modules[index]);
                index++;
                loadNext();
            };
            document.head.appendChild(script);
        }

        loadNext();
    };

    TrainerMain.prototype.setupEventListeners = function() {
        var self = this;

        // Validation button click handler
        document.addEventListener('click', function(e) {
            if (e.target && e.target.id === 'validate-config') {
                e.preventDefault();
                if (window.TrainerValidation) {
                    window.TrainerValidation.validateConfiguration();
                }
            }
        });

        // Start training button handler
        document.addEventListener('click', function(e) {
            if (e.target && e.target.id === 'start-training') {
                e.preventDefault();
                if (window.TrainerActions) {
                    window.TrainerActions.startTraining();
                }
            }
        });

        // Form change handlers
        document.addEventListener('change', function(e) {
            if (e.target && e.target.matches('input, select, textarea')) {
                if (window.TrainerForm) {
                    window.TrainerForm.handleFieldChange(e.target);
                }
            }
        });
    };

    TrainerMain.prototype.setupHTMXIntegration = function() {
        if (!window.htmx) {
            console.warn('HTMX not found, some features may not work');
            return;
        }

        // Setup HTMX event handlers
        document.body.addEventListener('htmx:afterSwap', function(evt) {
            // Reinitialize any components after HTMX swap
            if (window.TrainerUI) {
                window.TrainerUI.reinitialize();
            }
        });

        document.body.addEventListener('htmx:sendError', function(evt) {
            console.error('HTMX send error:', evt.detail);
            if (window.TrainerUI) {
                window.TrainerUI.showError('Network error occurred');
            }
        });
    };

    // Register the module
    window.TrainerModule_trainer_main = {
        TrainerMain: TrainerMain
    };

    // Also make it available directly
    window.TrainerMain = TrainerMain;

})(window);
