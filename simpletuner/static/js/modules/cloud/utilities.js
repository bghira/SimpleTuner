/**
 * Cloud Dashboard - Utility Methods Module
 *
 * Format helpers, computed properties, and general utilities.
 */

window.cloudUtilityMethods = {
    // NOTE: All getters moved to index.js final return object to avoid spread evaluation issues
    // Wizard step helpers kept as methods that call getters (getters defined in index.js)

    wizardIsStep(step, target) {
        // For final step: target can be 'last' to auto-calculate
        if (target === 'last') {
            return step === this.wizardTotalSteps;
        }
        return step === target;
    },

    wizardIsDataStep(step) {
        return step === 2 && this.wizardRequiresUpload;
    },

    wizardIsFinalStep(step) {
        return step === this.wizardTotalSteps;
    },

    // Job card helpers
    jobIsQueued(job) {
        return ['pending', 'queued'].includes(job.status);
    },

    jobIsRunning(job) {
        return job.status === 'running';
    },

    jobIsFailed(job) {
        return job.status === 'failed';
    },

    jobIsTerminal(job) {
        return ['completed', 'failed', 'cancelled'].includes(job.status);
    },

    jobHasSnapshot(job) {
        return job.metadata?.snapshot?.abbrev;
    },

    jobDisplayName(job) {
        return job.metadata?.tracker_run_name || job.config_name || job.job_id.substring(0, 8);
    },

    formatDuration(seconds) {
        return window.UIHelpers?.formatDuration(seconds) || '--';
    },

    formatWaitTime(seconds) {
        return window.UIHelpers?.formatWaitTime(seconds) || 'N/A';
    },

    getUploadStageTitle(stage) {
        const titles = {
            'scanning': 'Scanning Files',
            'packaging': 'Packaging Data',
            'uploading': 'Uploading',
            'complete': 'Complete',
            'error': 'Error',
        };
        return titles[stage] || 'Processing';
    },

    formatCost(cost) {
        if (cost === null || cost === undefined) return '--';
        return `$${cost.toFixed(2)}`;
    },

    getJobWaitTime(job) {
        if (!job.queued_at) return 'N/A';
        const queuedAt = new Date(job.queued_at).getTime();
        const now = Date.now();
        const seconds = (now - queuedAt) / 1000;
        return this.formatWaitTime(seconds);
    },

    formatDateCompact(isoString, compact = false) {
        if (!isoString) return '--';
        // Use compact relative time on mobile or when compact flag is set
        if (compact || window.innerWidth < 768) {
            return window.UIHelpers?.formatRelativeTime(isoString, { compact: true }) || '--';
        }
        return window.UIHelpers?.formatDateTime(isoString) || '--';
    },

    formatBytes(bytes) {
        if (!bytes) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        let i = 0;
        while (bytes >= 1024 && i < units.length - 1) {
            bytes /= 1024;
            i++;
        }
        return `${bytes.toFixed(1)} ${units[i]}`;
    },

    statusColor(status) {
        const colors = {
            'pending': 'secondary',
            'uploading': 'info',
            'queued': 'info',
            'running': 'primary',
            'completed': 'success',
            'failed': 'danger',
            'cancelled': 'warning',
        };
        return colors[status] || 'secondary';
    },

    statusIcon(status) {
        const icons = {
            'pending': 'fa-clock',
            'uploading': 'fa-upload',
            'queued': 'fa-hourglass-half',
            'running': 'fa-play',
            'completed': 'fa-check',
            'failed': 'fa-times',
            'cancelled': 'fa-ban',
        };
        return icons[status] || 'fa-circle';
    },

    jobTypeIcon(jobType) {
        return jobType === 'cloud' ? 'fa-cloud' : 'fa-desktop';
    },

    async fetchPollingStatus() {
        this.pollingStatus.loading = true;
        try {
            const response = await fetch('/api/cloud/polling/status');
            if (response.ok) {
                const data = await response.json();
                this.pollingStatus.active = data.is_active;
                this.pollingStatus.preference = data.preference;
            }
        } catch (error) {
            console.error('Failed to fetch polling status:', error);
        } finally {
            this.pollingStatus.loading = false;
        }
    },

    async togglePolling() {
        const current = this.pollingStatus.preference;
        const newValue = current === false ? true : false;

        this.pollingStatus.loading = true;
        try {
            const response = await fetch('/api/cloud/polling/setting', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enabled: newValue }),
            });

            if (response.ok) {
                const data = await response.json();
                this.pollingStatus.active = data.is_active;
                this.pollingStatus.preference = data.enabled;

                if (window.showToast) {
                    const statusMsg = data.is_active ? 'active' : 'inactive';
                    window.showToast(`Background polling is now ${statusMsg}`, 'success');
                }
            } else {
                if (window.showToast) window.showToast('Failed to update polling setting', 'error');
            }
        } catch (error) {
            console.error('Failed to toggle polling:', error);
            if (window.showToast) window.showToast('Failed to toggle polling', 'error');
        } finally {
            this.pollingStatus.loading = false;
        }
    },
};

// Computed property getters - these need to be defined as getters on the component
window.cloudComputedProperties = {
    get activeJobs() {
        return this.jobs.filter(j => ['pending', 'uploading', 'queued', 'running'].includes(j.status));
    },

    get completedJobs() {
        return this.jobs.filter(j => ['completed', 'failed', 'cancelled'].includes(j.status));
    },

    get filteredJobs() {
        let filtered = [...this.jobs];

        // Apply status filter
        if (this.statusFilter) {
            filtered = filtered.filter(j => j.status === this.statusFilter);
        }

        // Apply search filter
        if (this.jobSearchQuery && this.jobSearchQuery.trim()) {
            const query = this.jobSearchQuery.toLowerCase().trim();
            filtered = filtered.filter(j => {
                const jobId = (j.job_id || '').toLowerCase();
                const configName = (j.config_name || '').toLowerCase();
                const trackerName = (j.metadata?.tracker_run_name || '').toLowerCase();
                const status = (j.status || '').toLowerCase();
                const provider = (j.provider || '').toLowerCase();

                return jobId.includes(query) ||
                       configName.includes(query) ||
                       trackerName.includes(query) ||
                       status.includes(query) ||
                       provider.includes(query);
            });
        }

        // Apply sort order
        filtered.sort((a, b) => {
            const dateA = new Date(a.created_at || 0);
            const dateB = new Date(b.created_at || 0);
            return this.jobsSortOrder === 'desc' ? dateB - dateA : dateA - dateB;
        });

        return filtered;
    },

    get failedCount() {
        return this.jobs.filter(j => j.status === 'failed').length;
    },
};
