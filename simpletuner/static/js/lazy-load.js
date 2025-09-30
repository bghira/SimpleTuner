/**
 * Lazy Loading Utility for Heavy Components
 * Uses Intersection Observer API with fallback
 */

(function(window) {
    'use strict';

    var LazyLoader = {
        // Configuration
        config: {
            rootMargin: '50px 0px', // Start loading 50px before visible
            threshold: 0.01, // 1% visibility triggers load
            loadingClass: 'lazy-loading',
            loadedClass: 'lazy-loaded',
            errorClass: 'lazy-error',
            retryAttempts: 3,
            retryDelay: 1000
        },

        // Track observers and elements
        observers: new WeakMap(),
        elements: new Set(),
        retryCount: new WeakMap(),

        /**
         * Initialize lazy loading
         * @param {Object} options - Configuration options
         */
        init: function(options) {
            // Merge options
            Object.assign(LazyLoader.config, options || {});

            // Check for IntersectionObserver support
            if (!('IntersectionObserver' in window)) {
                // Fallback: load all lazy elements immediately
                console.warn('IntersectionObserver not supported, loading all lazy content');
                LazyLoader.loadAllFallback();
                return;
            }

            // Setup lazy loading for existing elements
            LazyLoader.observeElements();

            // Watch for new lazy elements
            LazyLoader.watchForNewElements();
        },

        /**
         * Observe elements for lazy loading
         * @param {NodeList|Array|Element} elements - Elements to observe
         */
        observeElements: function(elements) {
            if (!elements) {
                elements = document.querySelectorAll('[data-lazy], .lazy-load');
            }

            // Ensure we have an array
            if (elements instanceof Element) {
                elements = [elements];
            } else if (elements instanceof NodeList) {
                elements = Array.from(elements);
            }

            elements.forEach(function(element) {
                if (!LazyLoader.elements.has(element)) {
                    LazyLoader.observeElement(element);
                }
            });
        },

        /**
         * Observe a single element
         * @param {Element} element - Element to observe
         */
        observeElement: function(element) {
            // Create observer for this element
            var observer = new IntersectionObserver(function(entries) {
                entries.forEach(function(entry) {
                    if (entry.isIntersecting || entry.intersectionRatio > 0) {
                        LazyLoader.loadElement(entry.target);
                    }
                });
            }, LazyLoader.config);

            // Start observing
            observer.observe(element);
            LazyLoader.observers.set(element, observer);
            LazyLoader.elements.add(element);

            // Check if already visible
            if (LazyLoader.isElementVisible(element)) {
                LazyLoader.loadElement(element);
            }
        },

        /**
         * Load a lazy element
         * @param {Element} element - Element to load
         */
        loadElement: function(element) {
            // Stop observing
            var observer = LazyLoader.observers.get(element);
            if (observer) {
                observer.unobserve(element);
                LazyLoader.observers.delete(element);
            }

            // Add loading class
            element.classList.add(LazyLoader.config.loadingClass);

            // Determine load type
            var loadType = element.getAttribute('data-lazy-type') || 'auto';

            switch (loadType) {
                case 'image':
                    LazyLoader.loadImage(element);
                    break;
                case 'iframe':
                    LazyLoader.loadIframe(element);
                    break;
                case 'htmx':
                    LazyLoader.loadHTMX(element);
                    break;
                case 'component':
                    LazyLoader.loadComponent(element);
                    break;
                case 'auto':
                default:
                    LazyLoader.autoLoad(element);
                    break;
            }
        },

        /**
         * Auto-detect and load element
         * @param {Element} element - Element to load
         */
        autoLoad: function(element) {
            if (element.tagName === 'IMG') {
                LazyLoader.loadImage(element);
            } else if (element.tagName === 'IFRAME') {
                LazyLoader.loadIframe(element);
            } else if (element.hasAttribute('hx-get') || element.hasAttribute('data-hx-get')) {
                LazyLoader.loadHTMX(element);
            } else {
                LazyLoader.loadComponent(element);
            }
        },

        /**
         * Load lazy image
         * @param {Element} img - Image element
         */
        loadImage: function(img) {
            var src = img.getAttribute('data-lazy-src') || img.getAttribute('data-src');
            if (!src) {
                LazyLoader.handleError(img, 'No image source specified');
                return;
            }

            var tempImg = new Image();

            tempImg.onload = function() {
                img.src = src;
                img.classList.remove(LazyLoader.config.loadingClass);
                img.classList.add(LazyLoader.config.loadedClass);
                LazyLoader.elements.delete(img);
            };

            tempImg.onerror = function() {
                LazyLoader.handleError(img, 'Failed to load image');
            };

            tempImg.src = src;
        },

        /**
         * Load lazy iframe
         * @param {Element} iframe - Iframe element
         */
        loadIframe: function(iframe) {
            var src = iframe.getAttribute('data-lazy-src') || iframe.getAttribute('data-src');
            if (!src) {
                LazyLoader.handleError(iframe, 'No iframe source specified');
                return;
            }

            iframe.onload = function() {
                iframe.classList.remove(LazyLoader.config.loadingClass);
                iframe.classList.add(LazyLoader.config.loadedClass);
                LazyLoader.elements.delete(iframe);
            };

            iframe.onerror = function() {
                LazyLoader.handleError(iframe, 'Failed to load iframe');
            };

            iframe.src = src;
        },

        /**
         * Load HTMX component
         * @param {Element} element - Element with HTMX attributes
         */
        loadHTMX: function(element) {
            if (!window.htmx) {
                LazyLoader.handleError(element, 'HTMX not available');
                return;
            }

            // Transfer data attributes to hx attributes
            var hxGet = element.getAttribute('data-hx-get');
            if (hxGet) {
                element.setAttribute('hx-get', hxGet);
                element.removeAttribute('data-hx-get');
            }

            // Process with HTMX
            htmx.process(element);

            // Trigger the request
            htmx.trigger(element, 'load');

            // Listen for swap completion
            element.addEventListener('htmx:afterSwap', function() {
                element.classList.remove(LazyLoader.config.loadingClass);
                element.classList.add(LazyLoader.config.loadedClass);
                LazyLoader.elements.delete(element);
            }, { once: true });

            // Handle errors
            element.addEventListener('htmx:responseError', function() {
                LazyLoader.handleError(element, 'HTMX request failed');
            }, { once: true });
        },

        /**
         * Load generic component
         * @param {Element} element - Component element
         */
        loadComponent: function(element) {
            var componentUrl = element.getAttribute('data-lazy-component');
            if (!componentUrl) {
                // Just mark as loaded if no URL specified
                element.classList.remove(LazyLoader.config.loadingClass);
                element.classList.add(LazyLoader.config.loadedClass);
                LazyLoader.elements.delete(element);
                return;
            }

            // Fetch component HTML
            fetch(componentUrl)
                .then(function(response) {
                    if (!response.ok) throw new Error('Network response was not ok');
                    return response.text();
                })
                .then(function(html) {
                    element.innerHTML = html;
                    element.classList.remove(LazyLoader.config.loadingClass);
                    element.classList.add(LazyLoader.config.loadedClass);
                    LazyLoader.elements.delete(element);

                    // Trigger custom event
                    element.dispatchEvent(new CustomEvent('lazy-loaded', {
                        bubbles: true,
                        detail: { componentUrl: componentUrl }
                    }));
                })
                .catch(function() {
                    LazyLoader.handleError(element, 'Failed to load component');
                });
        },

        /**
         * Handle load errors with retry
         * @param {Element} element - Element that failed to load
         * @param {string} error - Error message
         */
        handleError: function(element, error) {
            var retryCount = LazyLoader.retryCount.get(element) || 0;

            if (retryCount < LazyLoader.config.retryAttempts) {
                // Retry after delay
                LazyLoader.retryCount.set(element, retryCount + 1);
                setTimeout(function() {
                    LazyLoader.loadElement(element);
                }, LazyLoader.config.retryDelay * (retryCount + 1));
            } else {
                // Max retries reached
                element.classList.remove(LazyLoader.config.loadingClass);
                element.classList.add(LazyLoader.config.errorClass);
                LazyLoader.elements.delete(element);

                console.error('Lazy load failed:', error, element);

                // Dispatch error event
                element.dispatchEvent(new CustomEvent('lazy-error', {
                    bubbles: true,
                    detail: { error: error }
                }));
            }
        },

        /**
         * Check if element is visible (for fallback)
         * @param {Element} element - Element to check
         * @returns {boolean} Whether element is visible
         */
        isElementVisible: function(element) {
            var rect = element.getBoundingClientRect();
            return (
                rect.top < window.innerHeight &&
                rect.bottom > 0 &&
                rect.left < window.innerWidth &&
                rect.right > 0
            );
        },

        /**
         * Fallback: Load all lazy elements
         */
        loadAllFallback: function() {
            var elements = document.querySelectorAll('[data-lazy], .lazy-load');
            elements.forEach(function(element) {
                LazyLoader.loadElement(element);
            });
        },

        /**
         * Watch for new lazy elements
         */
        watchForNewElements: function() {
            if (!window.MutationObserver) return;

            var observer = new MutationObserver(function(mutations) {
                mutations.forEach(function(mutation) {
                    mutation.addedNodes.forEach(function(node) {
                        if (node.nodeType === 1) { // Element node
                            // Check the node itself
                            if ((node.hasAttribute && node.hasAttribute('data-lazy')) ||
                                (node.classList && node.classList.contains('lazy-load'))) {
                                LazyLoader.observeElement(node);
                            }

                            // Check descendants
                            if (node.querySelectorAll) {
                                var lazyElements = node.querySelectorAll('[data-lazy], .lazy-load');
                                LazyLoader.observeElements(lazyElements);
                            }
                        }
                    });
                });
            });

            observer.observe(document.body, {
                childList: true,
                subtree: true
            });
        },

        /**
         * Manually trigger loading of visible elements
         */
        loadVisible: function() {
            LazyLoader.elements.forEach(function(element) {
                if (LazyLoader.isElementVisible(element)) {
                    LazyLoader.loadElement(element);
                }
            });
        },

        /**
         * Destroy lazy loading for an element
         * @param {Element} element - Element to destroy
         */
        destroy: function(element) {
            var observer = LazyLoader.observers.get(element);
            if (observer) {
                observer.disconnect();
                LazyLoader.observers.delete(element);
            }
            LazyLoader.elements.delete(element);
            LazyLoader.retryCount.delete(element);
        }
    };

    // Export
    window.LazyLoader = LazyLoader;

    // Auto-initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            LazyLoader.init();
        });
    } else {
        LazyLoader.init();
    }

})(window);