// Server Configuration Detection Module
// Automatically detects SimpleTuner server mode and configures appropriate endpoints
(function() {
    // Define the ServerConfig object
    window.ServerConfig = {
        mode: 'unknown',
        apiBaseUrl: 'http://localhost:8001',
        callbackUrl: 'http://localhost:8002', // Default to separate mode
        isReady: false,

        // Initialize and detect server configuration
        init: async function() {
            try {
                // Query the health endpoint on the trainer port
                const response = await fetch('http://localhost:8001/health', {
                    method: 'GET',
                    mode: 'cors',
                    credentials: 'omit',
                    cache: 'no-cache'
                });

                if (response.ok) {
                    const health = await response.json();
                    this.mode = health.mode || 'trainer';

                    // Configure URLs based on detected mode
                    if (this.mode === 'unified') {
                        // In unified mode, all endpoints are on port 8001
                        this.callbackUrl = 'http://localhost:8001';
                        // Server running in unified mode - all endpoints on port 8001
                    } else if (this.mode === 'trainer') {
                        // In trainer-only mode, callback endpoints might be on 8002
                        // Try to detect if callback server is running
                        try {
                            const callbackResponse = await fetch('http://localhost:8002/health', {
                                method: 'GET',
                                mode: 'cors',
                                credentials: 'omit',
                                cache: 'no-cache',
                                signal: AbortSignal.timeout(1000) // 1 second timeout
                            });
                            if (callbackResponse.ok) {
                                // Separate callback server detected on port 8002
                            }
                        } catch (err) {
                            console.warn('No callback server detected on port 8002, some features may not work');
                        }
                    }

                    this.isReady = true;
                    // Server configuration detected

                    // Dispatch event to notify other scripts
                    window.dispatchEvent(new CustomEvent('serverConfigReady', { detail: this }));

                } else {
                    throw new Error(`Health check failed with status ${response.status}`);
                }

            } catch (error) {
                console.error('Failed to detect server configuration:', error);
                console.warn('Using default configuration (separate mode)');

                // Fall back to default configuration
                this.mode = 'separate';
                this.isReady = true;

                // Dispatch event even on failure
                window.dispatchEvent(new CustomEvent('serverConfigReady', { detail: this }));
            }
        },

        // Helper to get the appropriate URL for an endpoint
        getEndpointUrl: function(endpoint) {
            // Determine which base URL to use based on endpoint type
            const isCallbackEndpoint = endpoint.includes('/callback') ||
                                     endpoint.includes('/broadcast') ||
                                     endpoint.includes('/events') ||
                                     endpoint.includes('/models');

            const baseUrl = (isCallbackEndpoint && this.mode !== 'unified')
                          ? this.callbackUrl
                          : this.apiBaseUrl;

            return `${baseUrl}${endpoint}`;
        },

        // Wait for configuration to be ready
        waitForReady: function() {
            if (this.isReady) {
                return Promise.resolve();
            }

            return new Promise((resolve) => {
                window.addEventListener('serverConfigReady', () => resolve(), { once: true });
            });
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => window.ServerConfig.init());
    } else {
        // DOM already loaded
        window.ServerConfig.init();
    }
})();
