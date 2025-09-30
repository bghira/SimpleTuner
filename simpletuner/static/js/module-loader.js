/**
 * Module Loader Wrapper - Provides ES6 module fallback for older browsers
 */

(function(window) {
    'use strict';

    // Check if native ES6 modules are supported
    var supportsModules = 'noModule' in HTMLScriptElement.prototype;

    // Module registry for fallback loading
    var moduleRegistry = {};
    var modulePromises = {};

    /**
     * Load a module with fallback support
     * @param {string} modulePath - Path to the module
     * @returns {Promise} Promise that resolves to the module exports
     */
    window.loadModule = function(modulePath) {
        // If already loaded, return from registry
        if (moduleRegistry[modulePath]) {
            return Promise.resolve(moduleRegistry[modulePath]);
        }

        // If already loading, return existing promise
        if (modulePromises[modulePath]) {
            return modulePromises[modulePath];
        }

        // If ES6 modules are supported, use dynamic import
        if (supportsModules) {
            modulePromises[modulePath] = import(modulePath)
                .then(function(module) {
                    moduleRegistry[modulePath] = module;
                    return module;
                })
                .catch(function(error) {
                    console.error('Failed to load module:', modulePath, error);
                    throw error;
                });
            return modulePromises[modulePath];
        }

        // Fallback: Load as regular script and expect global registration
        modulePromises[modulePath] = new Promise(function(resolve, reject) {
            var script = document.createElement('script');

            // Convert module path to non-module version
            var fallbackPath = modulePath.replace('.js', '-compat.js');

            script.onload = function() {
                // Module should have registered itself globally
                var moduleName = getModuleNameFromPath(modulePath);
                if (window[moduleName]) {
                    moduleRegistry[modulePath] = window[moduleName];
                    resolve(window[moduleName]);
                } else {
                    reject(new Error('Module did not register itself: ' + moduleName));
                }
            };

            script.onerror = function() {
                reject(new Error('Failed to load script: ' + fallbackPath));
            };

            script.src = fallbackPath;
            document.head.appendChild(script);
        });

        return modulePromises[modulePath];
    };

    /**
     * Extract module name from path
     * @param {string} path - Module path
     * @returns {string} Module name
     */
    function getModuleNameFromPath(path) {
        var filename = path.split('/').pop();
        var name = filename.replace('.js', '').replace(/-/g, '_');
        return 'TrainerModule_' + name;
    }

    /**
     * Register a module for fallback loading
     * Used by compatibility versions of modules
     * @param {string} name - Module name
     * @param {object} exports - Module exports
     */
    window.registerModule = function(name, exports) {
        window[name] = exports;
    };

    /**
     * Define a module with dependencies (AMD-like pattern)
     * @param {string} name - Module name
     * @param {array} dependencies - Array of dependency paths
     * @param {function} factory - Module factory function
     */
    window.defineModule = function(name, dependencies, factory) {
        if (typeof dependencies === 'function') {
            factory = dependencies;
            dependencies = [];
        }

        // Load all dependencies
        var depPromises = dependencies.map(function(dep) {
            if (dep === 'exports') {
                return {};
            }
            return window.loadModule(dep);
        });

        Promise.all(depPromises).then(function(deps) {
            var module = { exports: {} };
            var moduleExports = factory.apply(null, deps.concat([module.exports]));

            // Use return value or module.exports
            var finalExports = moduleExports || module.exports;

            // Register the module
            window.registerModule(name, finalExports);
        }).catch(function(error) {
            console.error('Failed to define module:', name, error);
        });
    };

    // Helper to check if we should use module loading
    window.shouldUseModules = function() {
        return supportsModules;
    };

    // Polyfill for dynamic import if needed
    if (!supportsModules && typeof window.import === 'undefined') {
        window.import = function(path) {
            return window.loadModule(path);
        };
    }

})(window);