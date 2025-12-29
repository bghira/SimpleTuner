/**
 * Jobs State Factory
 *
 * State for job list, filtering, and selection.
 */

window.cloudJobsStateFactory = function() {
    return {
        jobs: [],
        jobsLoading: false,
        jobsInitialized: false,
        syncing: false,
        selectedJob: null,
        statusFilter: null,
        jobsSortOrder: 'desc',
        jobSearchQuery: '',
        jobSearchDebounceTimer: null,
    };
};
