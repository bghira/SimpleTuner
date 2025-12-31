/**
 * System Status State Factory
 *
 * State for system health, status page, and prometheus metrics.
 */

window.cloudSystemStatusStateFactory = function() {
    return {
        systemStatus: {
            loading: false,
            operational: null,
            ongoing_incidents: [],
            in_progress_maintenances: [],
            scheduled_maintenances: [],
            status_page_url: null,
            error: null,
        },
        healthCheck: {
            loading: false,
            lastChecked: null,
            status: null,
            uptime_seconds: null,
            components: [],
            timestamp: null,
            error: null,
            autoRefreshInterval: null,
        },
        prometheusMetrics: {
            loading: false,
            visible: false,
            raw: '',
            parsed: {
                uptime: null,
                totalJobs: null,
                totalCost: null,
                activeJobs: null,
                jobsByStatus: {},
            },
            error: null,
        },
    };
};
