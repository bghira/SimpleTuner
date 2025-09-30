/**
 * SSE Manager - Singleton for managing Server-Sent Events connections
 * Handles connection lifecycle, retries, and message routing
 */

(function(window) {
    'use strict';

    var SSEManager = (function() {
        // Private variables
        var instance = null;
        var eventSource = null;
        var reconnectTimeout = null;
        var retryCount = 0;
        var maxRetries = 10;
        var baseRetryDelay = 1000; // 1 second
        var maxRetryDelay = 30000; // 30 seconds
        var heartbeatInterval = null;
        var lastHeartbeat = null;
        var connectionUrl = '/api/events';
        var listeners = {};
        var connectionState = 'disconnected'; // disconnected, connecting, connected

        /**
         * Calculate retry delay with exponential backoff
         */
        function calculateRetryDelay() {
            var delay = Math.min(baseRetryDelay * Math.pow(2, retryCount), maxRetryDelay);
            return delay + (Math.random() * 1000); // Add jitter
        }

        /**
         * Update connection status in UI
         */
        function updateConnectionStatus(status, message) {
            connectionState = status;

            // Update UI if function exists
            if (typeof window.updateConnectionStatus === 'function') {
                window.updateConnectionStatus(status, message);
            }

            // Notify listeners
            notifyListeners('connection-status', {
                status: status,
                message: message
            });
        }

        /**
         * Notify all listeners for an event type
         */
        function notifyListeners(eventType, data) {
            if (listeners[eventType]) {
                listeners[eventType].forEach(function(callback) {
                    try {
                        callback(data);
                    } catch (error) {
                        console.error('Error in SSE listener:', error);
                    }
                });
            }
        }

        /**
         * Setup heartbeat monitoring
         */
        function setupHeartbeat() {
            clearInterval(heartbeatInterval);
            lastHeartbeat = Date.now();

            heartbeatInterval = setInterval(function() {
                var timeSinceLastHeartbeat = Date.now() - lastHeartbeat;

                // If no heartbeat for 60 seconds, consider connection dead
                if (timeSinceLastHeartbeat > 60000 && connectionState === 'connected') {
                    console.warn('SSE heartbeat timeout, reconnecting...');
                    reconnect();
                }
            }, 10000); // Check every 10 seconds
        }

        /**
         * Clean up resources
         */
        function cleanup() {
            clearTimeout(reconnectTimeout);
            clearInterval(heartbeatInterval);

            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        }

        /**
         * Reconnect with exponential backoff
         */
        function reconnect() {
            cleanup();

            if (retryCount >= maxRetries) {
                updateConnectionStatus('disconnected', 'Maximum retries reached');
                console.error('SSE: Maximum reconnection attempts reached');
                return;
            }

            var delay = calculateRetryDelay();
            updateConnectionStatus('reconnecting', 'Reconnecting in ' + Math.round(delay / 1000) + 's...');

            reconnectTimeout = setTimeout(function() {
                retryCount++;
                connect();
            }, delay);
        }

        /**
         * Establish SSE connection
         */
        function connect() {
            if (eventSource && eventSource.readyState !== EventSource.CLOSED) {
                return; // Already connected or connecting
            }

            try {
                updateConnectionStatus('connecting', 'Connecting...');
                eventSource = new EventSource(connectionUrl);

                eventSource.onopen = function() {
                    console.log('SSE connection established');
                    retryCount = 0; // Reset retry count on successful connection
                    updateConnectionStatus('connected', 'Connected');
                    setupHeartbeat();
                };

                eventSource.onmessage = function(event) {
                    lastHeartbeat = Date.now();

                    try {
                        var data = JSON.parse(event.data);
                        handleMessage(data);
                    } catch (error) {
                        console.error('Error parsing SSE message:', error);
                    }
                };

                eventSource.onerror = function(error) {
                    console.error('SSE connection error:', error);
                    updateConnectionStatus('disconnected', 'Connection lost');

                    // EventSource will auto-reconnect, but we want custom retry logic
                    if (eventSource.readyState === EventSource.CLOSED) {
                        reconnect();
                    }
                };

                // Handle specific event types
                eventSource.addEventListener('heartbeat', function(event) {
                    lastHeartbeat = Date.now();
                    notifyListeners('heartbeat', { timestamp: lastHeartbeat });
                });

                eventSource.addEventListener('training_progress', function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('training_progress', data);
                    } catch (error) {
                        console.error('Error parsing training progress:', error);
                    }
                });

                eventSource.addEventListener('validation_complete', function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('validation_complete', data);
                    } catch (error) {
                        console.error('Error parsing validation complete:', error);
                    }
                });

            } catch (error) {
                console.error('Failed to create EventSource:', error);
                updateConnectionStatus('disconnected', 'Failed to connect');
                reconnect();
            }
        }

        /**
         * Handle incoming messages
         */
        function handleMessage(data) {
            // Route message based on type
            switch (data.type) {
                case 'training_progress':
                    // Update training progress UI
                    if (window.htmx) {
                        htmx.trigger('#training-progress', 'update-progress', data);
                    }
                    notifyListeners('training_progress', data);
                    break;

                case 'validation_complete':
                    notifyListeners('validation_complete', data);
                    break;

                case 'error':
                    if (window.showToast) {
                        window.showToast(data.message || 'An error occurred', 'error');
                    }
                    notifyListeners('error', data);
                    break;

                case 'notification':
                    if (window.showToast) {
                        window.showToast(data.message, data.level || 'info');
                    }
                    notifyListeners('notification', data);
                    break;

                case 'heartbeat':
                    // Already handled by event listener
                    break;

                default:
                    // Generic message handling
                    notifyListeners(data.type || 'message', data);
            }
        }

        /**
         * Public API
         */
        return {
            /**
             * Initialize SSE connection
             */
            init: function(config) {
                if (instance) {
                    return instance;
                }

                config = config || {};

                // Apply configuration
                if (config.url) connectionUrl = config.url;
                if (config.maxRetries) maxRetries = config.maxRetries;
                if (config.baseRetryDelay) baseRetryDelay = config.baseRetryDelay;
                if (config.maxRetryDelay) maxRetryDelay = config.maxRetryDelay;

                // Check browser support
                if (typeof EventSource === 'undefined') {
                    console.error('SSE not supported in this browser');
                    updateConnectionStatus('disconnected', 'SSE not supported');
                    return null;
                }

                // Set up page unload handler
                window.addEventListener('beforeunload', function() {
                    cleanup();
                });

                // Start connection
                connect();

                instance = this;
                return instance;
            },

            /**
             * Manually connect
             */
            connect: connect,

            /**
             * Manually disconnect
             */
            disconnect: function() {
                cleanup();
                updateConnectionStatus('disconnected', 'Manually disconnected');
            },

            /**
             * Add event listener
             */
            addEventListener: function(eventType, callback) {
                if (!listeners[eventType]) {
                    listeners[eventType] = [];
                }
                listeners[eventType].push(callback);
            },

            /**
             * Remove event listener
             */
            removeEventListener: function(eventType, callback) {
                if (listeners[eventType]) {
                    listeners[eventType] = listeners[eventType].filter(function(cb) {
                        return cb !== callback;
                    });
                }
            },

            /**
             * Get connection state
             */
            getState: function() {
                return {
                    connectionState: connectionState,
                    retryCount: retryCount,
                    readyState: eventSource ? eventSource.readyState : EventSource.CLOSED
                };
            },

            /**
             * Reset retry count
             */
            resetRetries: function() {
                retryCount = 0;
            }
        };
    })();

    // Export to window
    window.SSEManager = SSEManager;

})(window);
