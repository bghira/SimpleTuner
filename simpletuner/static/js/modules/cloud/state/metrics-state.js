/**
 * Metrics & Billing State Factory
 *
 * State for metrics, billing info, and cost limits.
 */

window.cloudMetricsStateFactory = function() {
    return {
        metrics: {
            credit_balance: null,
            estimated_jobs_remaining: null,
            total_cost: 0,
            job_count: 0,
            avg_job_duration_seconds: null,
            jobs_by_status: {},
            cost_by_day: [],
            period_days: 30,
        },
        metricsPeriod: 30,
        metricsLoading: false,
        billingState: {
            loading: false,
            fetched: false,
            error: null,
        },
        costLimit: {
            loading: false,
            saving: false,
            status: null,
        },
    };
};
