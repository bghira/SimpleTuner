/**
 * Debouncing utility for form inputs and other events
 * Reduces unnecessary server requests and improves performance
 */

(function(window) {
    'use strict';

    /**
     * Creates a debounced version of a function
     * @param {Function} func - The function to debounce
     * @param {number} wait - Milliseconds to wait
     * @param {boolean} immediate - Trigger on leading edge instead of trailing
     * @returns {Function} The debounced function
     */
    function debounce(func, wait, immediate) {
        var timeout;
        var pendingResolvers = [];

        return function debounced() {
            var context = this;
            var args = arguments;

            var later = function() {
                timeout = null;
                var result;
                if (!immediate) {
                    result = func.apply(context, args);
                }
                // Resolve all pending promises with the result
                var resolvers = pendingResolvers;
                pendingResolvers = [];
                resolvers.forEach(function(resolve) { resolve(result); });
            };

            var callNow = immediate && !timeout;
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);

            var result;
            if (callNow) {
                result = func.apply(context, args);
                // Resolve immediately for immediate mode
                return Promise.resolve(result);
            }

            // Return a promise that resolves when the debounced call executes
            return new Promise(function(resolve) {
                pendingResolvers.push(resolve);
            });
        };
    }

    /**
     * Creates a throttled version of a function
     * @param {Function} func - The function to throttle
     * @param {number} limit - Minimum milliseconds between calls
     * @returns {Function} The throttled function
     */
    function throttle(func, limit) {
        var inThrottle;

        return function throttled() {
            var args = arguments;
            var context = this;

            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(function() {
                    inThrottle = false;
                }, limit);
            }
        };
    }

    /**
     * Debounce manager for form inputs
     */
    var FormDebouncer = {
        // Store debounced functions by element ID
        debouncedFunctions: {},

        /**
         * Initialize debouncing for form inputs
         * @param {Object} config - Configuration options
         */
        init: function(config) {
            config = config || {};
            var defaultWait = config.wait || 500;
            var selector = config.selector || 'input[data-debounce], textarea[data-debounce], select[data-debounce]';

            // Setup debouncing for existing elements
            document.querySelectorAll(selector).forEach(function(element) {
                FormDebouncer.setupDebounce(element, defaultWait);
            });

            // Watch for new elements
            if (window.MutationObserver) {
                var observer = new MutationObserver(function(mutations) {
                    mutations.forEach(function(mutation) {
                        mutation.addedNodes.forEach(function(node) {
                            if (node.nodeType === 1) { // Element node
                                if (node.matches && node.matches(selector)) {
                                    FormDebouncer.setupDebounce(node, defaultWait);
                                }
                                // Check descendants
                                if (node.querySelectorAll) {
                                    node.querySelectorAll(selector).forEach(function(el) {
                                        FormDebouncer.setupDebounce(el, defaultWait);
                                    });
                                }
                            }
                        });
                    });
                });

                observer.observe(document.body, {
                    childList: true,
                    subtree: true
                });
            }
        },

        /**
         * Setup debounce for a specific element
         * @param {Element} element - The input element
         * @param {number} defaultWait - Default wait time
         */
        setupDebounce: function(element, defaultWait) {
            var wait = parseInt(element.getAttribute('data-debounce')) || defaultWait;
            var eventType = element.getAttribute('data-debounce-event') || 'input';
            var immediate = element.hasAttribute('data-debounce-immediate');

            // Create unique ID if not present
            if (!element.id) {
                element.id = 'debounced-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            }

            // Remove existing listener if any
            if (FormDebouncer.debouncedFunctions[element.id]) {
                element.removeEventListener(eventType, FormDebouncer.debouncedFunctions[element.id]);
            }

            // Create debounced function
            var originalHandler = function(event) {
                // Add loading state
                element.classList.add('debouncing');

                // Suppress dirty tracking during debounced HTMX validation
                // These are validation requests, not new user input
                var trainerStore = window.Alpine && window.Alpine.store ? window.Alpine.store('trainer') : null;
                var wasSuppressed = trainerStore && trainerStore._suppressDirtyTracking;
                if (trainerStore && !wasSuppressed) {
                    trainerStore._suppressDirtyTracking = true;
                }

                // If element has HTMX attributes, trigger HTMX
                if (window.htmx && element.hasAttribute('hx-trigger')) {
                    htmx.trigger(element, eventType);
                }

                // Custom event for other handlers - use non-bubbling to avoid form handlers
                var customEvent = new CustomEvent('debounced-' + eventType, {
                    detail: { originalEvent: event },
                    bubbles: false,  // Don't bubble to form handlers
                    cancelable: true
                });
                element.dispatchEvent(customEvent);

                // Restore dirty tracking after a microtask
                if (trainerStore && !wasSuppressed) {
                    Promise.resolve().then(function() {
                        trainerStore._suppressDirtyTracking = false;
                    });
                }

                // Remove loading state
                setTimeout(function() {
                    element.classList.remove('debouncing');
                }, 100);
            };

            var debouncedHandler = debounce(originalHandler, wait, immediate);
            FormDebouncer.debouncedFunctions[element.id] = debouncedHandler;

            // Add event listener
            element.addEventListener(eventType, debouncedHandler);
        },

        /**
         * Manually trigger debounced validation
         * @param {string|Element} elementOrId - Element or ID
         */
        trigger: function(elementOrId) {
            var element = typeof elementOrId === 'string'
                ? document.getElementById(elementOrId)
                : elementOrId;

            if (element && FormDebouncer.debouncedFunctions[element.id]) {
                FormDebouncer.debouncedFunctions[element.id]();
            }
        },

        /**
         * Clear debounce for an element
         * @param {string|Element} elementOrId - Element or ID
         */
        clear: function(elementOrId) {
            var element = typeof elementOrId === 'string'
                ? document.getElementById(elementOrId)
                : elementOrId;

            if (element && element.id && FormDebouncer.debouncedFunctions[element.id]) {
                delete FormDebouncer.debouncedFunctions[element.id];
            }
        }
    };

    // Request batching utility
    var RequestBatcher = {
        batches: {},
        batchTimeout: 50, // ms to wait before sending batch

        /**
         * Add a request to a batch
         * @param {string} batchKey - Key to group requests
         * @param {Object} data - Request data
         * @param {Function} callback - Callback for this specific request
         */
        add: function(batchKey, data, callback) {
            if (!RequestBatcher.batches[batchKey]) {
                RequestBatcher.batches[batchKey] = {
                    requests: [],
                    timer: null
                };
            }

            var batch = RequestBatcher.batches[batchKey];
            batch.requests.push({
                data: data,
                callback: callback
            });

            // Clear existing timer
            if (batch.timer) {
                clearTimeout(batch.timer);
            }

            // Set new timer
            batch.timer = setTimeout(function() {
                RequestBatcher.flush(batchKey);
            }, RequestBatcher.batchTimeout);
        },

        /**
         * Send all requests in a batch
         * @param {string} batchKey - Batch to flush
         */
        flush: function(batchKey) {
            var batch = RequestBatcher.batches[batchKey];
            if (!batch || batch.requests.length === 0) return;

            // Extract data and callbacks
            var allData = batch.requests.map(function(r) { return r.data; });
            var callbacks = batch.requests.map(function(r) { return r.callback; });

            // Clear batch
            delete RequestBatcher.batches[batchKey];

            // Send batched request (customize based on your API)
            if (window.htmx) {
                // For HTMX, trigger a custom event with batched data
                document.body.dispatchEvent(new CustomEvent('batch-request', {
                    detail: {
                        batchKey: batchKey,
                        data: allData,
                        callbacks: callbacks
                    }
                }));
            }
        }
    };

    // Export utilities
    window.debounce = debounce;
    window.throttle = throttle;
    window.FormDebouncer = FormDebouncer;
    window.RequestBatcher = RequestBatcher;

    // Auto-initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            FormDebouncer.init();
        });
    } else {
        FormDebouncer.init();
    }

})(window);
