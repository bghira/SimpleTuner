/**
 * Queue State Factory
 *
 * State for job queue management and settings.
 */

window.cloudQueueStateFactory = function() {
    return {
        showQueuePanel: false,
        queueStats: {
            queue_depth: 0,
            running: 0,
            max_concurrent: 5,
            user_max_concurrent: 2,
            avg_wait_seconds: null,
            by_status: {},
            by_user: {},
        },
        queueSettings: {
            max_concurrent: 5,
            user_max_concurrent: 2,
            team_max_concurrent: 10,
            enable_fair_share: false,
        },
        queuePendingJobs: [],
        queueBlockedJobs: [],
        loadingQueueStats: false,
        savingQueueSettings: false,
        processingQueue: false,
        cleaningQueue: false,
        cleanupDays: 30,
        lastCleanupResult: null,
        pendingApprovals: {
            count: 0,
            loading: false,
            lastLoaded: null,
        },
    };
};
