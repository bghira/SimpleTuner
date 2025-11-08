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
        var CALLBACK_EVENT_TYPES = ['progress', 'validation', 'job', 'status', 'alert', 'checkpoint', 'metric', 'debug'];
        var connectionState = 'disconnected'; // disconnected, connecting, connected

        // Store listener references for cleanup
        var heartbeatListener = null;
        var trainingProgressListener = null;
        var validationCompleteListener = null;
        var lifecycleStageListener = null;
        var trainingStatusListener = null;
        var connectionListener = null;
        var callbackEventListeners = {};

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
        function notifyListeners(eventType, data, rawEvent) {
            if (listeners[eventType]) {
                listeners[eventType].forEach(function(callback) {
                    try {
                        callback(data, rawEvent);
                    } catch (error) {
                        console.error('Error in SSE listener:', error);
                    }
                });
            }
        }

        function toNumber(value) {
            if (value === null || value === undefined || value === '') {
                return null;
            }
            var num = Number(value);
            return Number.isFinite(num) ? num : null;
        }

        function severityToLevel(severity) {
            switch (String(severity || '').toLowerCase()) {
                case 'success':
                    return 'success';
                case 'warning':
                    return 'warning';
                case 'error':
                case 'critical':
                    return 'danger';
                case 'debug':
                    return 'secondary';
                default:
                    return 'info';
            }
        }

        function transformProgressPayload(payload) {
            if (!payload || typeof payload !== 'object') {
                return null;
            }

            var progress = (payload.progress && typeof payload.progress === 'object') ? payload.progress : {};
            var stage = (payload.stage && typeof payload.stage === 'object') ? payload.stage : null;
            var data = (payload.data && typeof payload.data === 'object') ? payload.data : {};
            var extras = {};

            if (progress.metrics && typeof progress.metrics === 'object') {
                extras = Object.assign({}, progress.metrics);
            }

            if (progress.extra && typeof progress.extra === 'object') {
                extras = Object.assign({}, extras, progress.extra);
            }

            if (payload.metrics && typeof payload.metrics === 'object') {
                extras = Object.assign({}, extras, payload.metrics);
            }

            if (payload.extras && typeof payload.extras === 'object') {
                extras = Object.assign({}, extras, payload.extras);
            }

            if (data && typeof data === 'object') {
                extras = Object.assign({}, extras, data);
            }

            if (extras.state && typeof extras.state === 'object') {
                extras = Object.assign({}, extras, extras.state);
                delete extras.state;
            }

            if (stage) {
                var stageProgress = (stage.progress && typeof stage.progress === 'object') ? stage.progress : {};
                var stageLabel = stage.label || stageProgress.label || stage.key || stage.progress_type || 'Stage';
                var stageCurrent = toNumber(stageProgress.current || stage.current);
                var stageTotal = toNumber(stageProgress.total || stage.total);
                var stagePercent = toNumber(stageProgress.percent || stage.percent);
                if (stagePercent === null && stageCurrent !== null && stageTotal) {
                    stagePercent = (stageCurrent / stageTotal) * 100;
                }
                if (!Number.isFinite(stagePercent)) {
                    stagePercent = 0;
                }

                return {
                    type: 'lifecycle.stage',
                    job_id: payload.job_id || extras.job_id || null,
                    stage: {
                        key: stage.key || stage.progress_type || stageLabel,
                        label: stageLabel,
                        status: stage.status || 'running',
                        percent: Math.round(stagePercent || 0),
                        current: stageCurrent || 0,
                        total: stageTotal || 0,
                    },
                    status: stage.status || 'running',
                    raw: payload
                };
            }

            var jobId = payload.job_id || extras.job_id || null;

            var currentStep = toNumber(progress.current);
            if (currentStep === null) {
                currentStep = toNumber(extras.current_step || extras.global_step || extras.step);
            }
            if (currentStep === null && payload.current_step !== undefined) {
                currentStep = toNumber(payload.current_step);
            }

            var totalSteps = toNumber(progress.total);
            if (totalSteps === null) {
                totalSteps = toNumber(extras.total_steps || extras.total_num_steps || extras.max_steps);
            }
            if (totalSteps === null && payload.total_num_steps !== undefined) {
                totalSteps = toNumber(payload.total_num_steps);
            }

            var percent = toNumber(progress.percent);
            if (percent === null) {
                percent = toNumber(extras.percent);
            }
            if (percent === null && currentStep !== null && totalSteps) {
                percent = (currentStep / totalSteps) * 100;
            }
            if (!Number.isFinite(percent)) {
                percent = 0;
            }
            var clampedPercent = Math.max(0, Math.min(100, percent));
            var roundedPercent = Math.round(clampedPercent * 100) / 100;

            var epoch = toNumber(extras.epoch || extras.current_epoch);
            if (epoch === null && payload.current_epoch !== undefined) {
                epoch = toNumber(payload.current_epoch);
            }

            var totalEpochs = toNumber(extras.total_epochs || extras.total_num_epochs || extras.final_epoch);
            if (totalEpochs === null && payload.total_num_epochs !== undefined) {
                totalEpochs = toNumber(payload.total_num_epochs);
            }

            var loss = extras.loss !== undefined ? toNumber(extras.loss) : null;
            if (loss === null && extras.train_loss !== undefined) {
                loss = toNumber(extras.train_loss);
            }

            var learningRate = extras.learning_rate !== undefined ? toNumber(extras.learning_rate) : null;
            if (learningRate === null && extras.lr !== undefined) {
                learningRate = toNumber(extras.lr);
            }

            var label = progress.label || payload.title || payload.message || payload.headline || payload.readable_type || '';

            return {
                type: 'training.progress',
                job_id: jobId,
                percentage: roundedPercent,
                current_step: currentStep || 0,
                total_steps: totalSteps || 0,
                epoch: epoch || 0,
                total_epochs: totalEpochs || 0,
                loss: loss !== null ? loss : undefined,
                lr: learningRate !== null ? learningRate : undefined,
                label: label,
                raw: payload
            };
        }

        function handleCallbackEvent(category, payload) {
            var data = payload;
            var rawEvent = null;

            if (payload && typeof payload === 'object' && Object.prototype.hasOwnProperty.call(payload, 'event')) {
                rawEvent = payload.event;
                data = Object.prototype.hasOwnProperty.call(payload, 'payload') ? payload.payload : payload.data;
            }

            notifyListeners('callback:' + category, data, rawEvent);
            notifyListeners('callback', { category: category, payload: data, event: rawEvent });

            switch (category) {
                case 'progress': {
                    var progressMessages = transformProgressPayload(payload);
                    if (Array.isArray(progressMessages)) {
                        progressMessages.forEach(function(msg) {
                            if (msg) {
                                handleMessage(msg);
                            }
                        });
                    } else if (progressMessages) {
                        handleMessage(progressMessages);
                    }
                    break;
                }
                case 'validation': {
                    // Handle nested payload structure - type might be in payload.payload.type
                    var actualPayload = payload.payload || payload;
                    var eventType = actualPayload && actualPayload.type ? String(actualPayload.type).toLowerCase() : '';
                    var isReplay = actualPayload.is_replay || payload.is_replay;

                    // Don't show toasts for intermediary validation images or replays
                    if (eventType === 'validation.image' || isReplay) {
                        break;
                    }

                    var validationMessage = actualPayload.headline || actualPayload.body || actualPayload.message || payload.headline || payload.body || payload.message || 'Validation complete';
                    var validationData = {
                        type: 'validation_complete',
                        message: validationMessage,
                        images: payload.images || [],  // Preserve images from payload
                        payload: payload
                    };
                    handleMessage(validationData);
                    // Also emit to HTMX if event dock exists
                    if (window.htmx) {
                        var eventDock = document.querySelector('#eventList');
                        if (eventDock) {
                            htmx.trigger(eventDock, 'new-validation-event', validationData);
                        }
                    }
                    break;
                }
                case 'alert': {
                    var alertMessage = payload.headline || payload.body || 'Alert received';
                    var level = severityToLevel(payload.severity);
                    var alertData = {
                        type: level === 'danger' ? 'error' : 'notification',
                        message: alertMessage,
                        level: level,
                        payload: payload
                    };
                    handleMessage(alertData);
                    break;
                }
                case 'status':
                case 'job':
                case 'checkpoint': {
                    var statusMessage = payload.headline || payload.body;
                    if (statusMessage) {
                        handleMessage({
                            type: 'notification',
                            message: statusMessage,
                            level: severityToLevel(payload.severity),
                            payload: payload
                        });
                    }
                    break;
                }
                case 'notification': {
                    var notifyMessage = payload.headline || payload.body || payload.message;
                    if (notifyMessage) {
                        handleMessage({
                            type: 'notification',
                            message: notifyMessage,
                            level: severityToLevel(payload.severity),
                            payload: payload
                        });
                    }
                    break;
                }
                case 'error': {
                    var errorMessage = payload.message || payload.body || payload.headline || 'Training error occurred';
                    handleMessage({
                        type: 'error',
                        message: errorMessage,
                        level: 'danger',
                        payload: payload
                    });
                    break;
                }
                case 'debug':
                default: {
                    var genericMessage = payload.headline || payload.body || payload.message;
                    if (genericMessage) {
                        handleMessage({
                            type: 'notification',
                            message: genericMessage,
                            level: severityToLevel(payload.severity),
                            payload: payload
                        });
                    }
                    break;
                }
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
                // Remove all event listeners before closing
                eventSource.removeEventListener('heartbeat', heartbeatListener);
                eventSource.removeEventListener('training.progress', trainingProgressListener);
                eventSource.removeEventListener('validation_complete', validationCompleteListener);
                eventSource.removeEventListener('lifecycle.stage', lifecycleStageListener);
                eventSource.removeEventListener('training.status', trainingStatusListener);
                eventSource.removeEventListener('connection', connectionListener);

                // Remove callback event listeners
                CALLBACK_EVENT_TYPES.forEach(function(category) {
                    var listener = callbackEventListeners[category];
                    if (listener) {
                        eventSource.removeEventListener('callback:' + category, listener);
                    }
                });

                eventSource.close();
                eventSource = null;
            }

            // Clear stored listener references
            heartbeatListener = null;
            trainingProgressListener = null;
            validationCompleteListener = null;
            lifecycleStageListener = null;
            trainingStatusListener = null;
            connectionListener = null;
            callbackEventListeners = {};
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

                eventSource.onerror = function(errorEvent) {
                    // Some backends emit custom "error" events with a JSON payload.
                    if (errorEvent && errorEvent.data) {
                        try {
                            var parsed = JSON.parse(errorEvent.data);
                            handleCallbackEvent('error', parsed);
                        } catch (parseErr) {
                            console.error('Failed to parse SSE error payload:', parseErr);
                            handleCallbackEvent('error', { message: String(errorEvent.data || 'Unknown error') });
                        }
                        return;
                    }

                    console.error('SSE connection error:', errorEvent);
                    updateConnectionStatus('disconnected', 'Connection lost');

                    // EventSource will auto-reconnect, but we want custom retry logic
                    if (eventSource.readyState === EventSource.CLOSED) {
                        reconnect();
                    }
                };

                // Handle specific event types - store listener references for cleanup
                heartbeatListener = function(event) {
                    lastHeartbeat = Date.now();
                    notifyListeners('heartbeat', { timestamp: lastHeartbeat });
                };
                eventSource.addEventListener('heartbeat', heartbeatListener);

                trainingProgressListener = function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('training.progress', data);
                    } catch (error) {
                        console.error('Error parsing training progress:', error);
                    }
                };
                eventSource.addEventListener('training.progress', trainingProgressListener);

                validationCompleteListener = function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('validation_complete', data);
                    } catch (error) {
                        console.error('Error parsing validation complete:', error);
                    }
                };
                eventSource.addEventListener('validation_complete', validationCompleteListener);

                lifecycleStageListener = function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('lifecycle.stage', data);
                        window.dispatchEvent(new CustomEvent('startup-progress', { detail: data }));
                    } catch (error) {
                        console.error('Error parsing lifecycle stage:', error);
                    }
                };
                eventSource.addEventListener('lifecycle.stage', lifecycleStageListener);

                trainingStatusListener = function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        notifyListeners('training.status', data);
                    } catch (error) {
                        console.error('Error parsing training status:', error);
                    }
                };
                eventSource.addEventListener('training.status', trainingStatusListener);

                connectionListener = function(event) {
                    lastHeartbeat = Date.now();
                    try {
                        var data = JSON.parse(event.data);
                        handleMessage(data);
                    } catch (error) {
                        console.error('Error parsing connection event:', error);
                    }
                };
                eventSource.addEventListener('connection', connectionListener);

                CALLBACK_EVENT_TYPES.forEach(function(category) {
                    var listener = function(event) {
                        lastHeartbeat = Date.now();
                        var parsedData = null;
                        try {
                            parsedData = JSON.parse(event.data);
                        } catch (error) {
                            console.error('Error parsing callback event:', error);
                            parsedData = event.data;
                        }
                        try {
                            handleCallbackEvent(category, { payload: parsedData, event: event });
                        } catch (error) {
                            console.error('Error handling callback event:', error);
                        }
                    };
                    callbackEventListeners[category] = listener;
                    eventSource.addEventListener('callback:' + category, listener);
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
                case 'training.progress':
                    // Update training progress UI
                    if (window.htmx && !data.reset) {
                        const progressEl = document.querySelector('#training-progress');
                        if (progressEl) {
                            htmx.trigger(progressEl, 'update-progress', data);
                        }
                    }
                    notifyListeners('training.progress', data);
                    break;

                case 'lifecycle.stage':
                    notifyListeners('lifecycle.stage', data);
                    window.dispatchEvent(new CustomEvent('startup-progress', { detail: data }));
                    break;

                case 'validation_complete':
                    notifyListeners('validation_complete', data);
                    break;

                case 'error':
                    notifyListeners('error', data);
                    break;

                case 'notification':
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
                    instance.disconnect();
                    instance.clearAllListeners();
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
             * Remove all listeners for a specific event type
             */
            removeAllListeners: function(eventType) {
                if (listeners[eventType]) {
                    delete listeners[eventType];
                }
            },

            /**
             * Clear all registered listeners
             */
            clearAllListeners: function() {
                listeners = {};
            },

            /**
             * Destroy the SSE manager instance completely
             */
            destroy: function() {
                cleanup();
                this.clearAllListeners();
                instance = null;
                updateConnectionStatus('disconnected', 'Manager destroyed');
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
            },

            /**
             * Expose progress normalizer for external consumers
             */
            normalizeProgressPayload: function(payload) {
                var result = transformProgressPayload(payload);
                if (!result || result.type === 'lifecycle.stage') {
                    return null;
                }
                return result;
            },

            /**
             * Expose severity mapping helper
             */
            mapSeverityToLevel: severityToLevel
        };
    })();

    // Export to window
    window.SSEManager = SSEManager;

})(window);
