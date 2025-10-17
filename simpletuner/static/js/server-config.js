// Server Configuration Detection Module
// Automatically detects SimpleTuner server mode and configures appropriate endpoints
(function() {
    // Define the ServerConfig object
    window.ServerConfig = {
        mode: 'unknown',
        apiBaseUrl: `${window.location.origin}`,
        callbackUrl: `${window.location.origin}`, // Default to same origin
        isReady: false,

        // Initialize and detect server configuration
        init: async function() {
            try {
                // Query the health endpoint on the current host
                const response = await fetch(`${this.apiBaseUrl}/health`, {
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
                        // In unified mode, all endpoints are on the same host
                        this.callbackUrl = this.apiBaseUrl;
                        // Server running in unified mode - all endpoints on same host
                    } else if (this.mode === 'trainer') {
                        // In trainer-only mode, callback endpoints might be on a different port
                        // Try to detect if callback server is running on a different port
                        try {
                            // Try common callback ports by modifying the current origin
                            const currentUrl = new URL(this.apiBaseUrl);
                            const callbackPorts = [8002, 8001]; // Try different ports

                            for (const port of callbackPorts) {
                                if (port.toString() !== currentUrl.port) {
                                    const callbackUrl = `${currentUrl.protocol}//${currentUrl.hostname}:${port}`;
                                    const callbackResponse = await fetch(`${callbackUrl}/health`, {
                                        method: 'GET',
                                        mode: 'cors',
                                        credentials: 'omit',
                                        cache: 'no-cache',
                                        signal: AbortSignal.timeout(1000) // 1 second timeout
                                    });
                                    if (callbackResponse.ok) {
                                        this.callbackUrl = callbackUrl;
                                        break;
                                    }
                                }
                            }
                        } catch (err) {
                            console.warn('No callback server detected, some features may not work');
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
