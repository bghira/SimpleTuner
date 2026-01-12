/**
 * Connection & Auth State Factory
 *
 * State for user auth, polling intervals, and SSE connection.
 */

window.cloudConnectionStateFactory = function() {
    return {
        currentUser: null,
        hasAdminAccess: false,
        pollingStatus: {
            active: false,
            preference: null,
            loading: false,
        },
        pollingInterval: null,
        inlineProgressInterval: null,
        sseConnection: {
            status: 'unknown',  // 'connected', 'disconnected', 'reconnecting', 'unknown'
            message: '',
            lastUpdated: null,
        },
    };
};
