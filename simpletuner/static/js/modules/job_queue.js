/**
 * Job Queue Manager - Unified view of local and cloud training jobs
 */

if (!window.jobQueueManager) {
window.jobQueueManager = function jobQueueManager() {
    return {
        // Hero CTA state (using HintMixin)
        ...(window.HintMixin?.createSingleHint({
            useApi: true,
            hintKey: 'job_queue_hero',
        }) || { heroDismissed: false, loadHeroCTAState() {}, dismissHeroCTA() {}, restoreHeroCTA() {} }),
        loaded: false,

        // State
        jobs: [],
        selectedJob: null,
        loading: false,

        // Filters
        typeFilter: 'all',  // 'all', 'local', 'cloud'
        providerFilter: '',  // '', 'replicate', etc.
        statusFilter: '',   // '', 'running', 'pending', 'completed', 'failed'

        // Admin filters
        isAdmin: false,
        adminUserFilter: '',  // User ID to filter by (admin only)

        // Logs modal
        logs: '',
        logsJobId: '',
        logsLoading: false,
        logsModal: null,

        // Polling
        pollingInterval: null,

        async init() {
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }

            // Load hero CTA dismissed state
            await this.loadHeroCTAState();
            await this.checkAdminStatus();
            await this.loadJobs();
            this.loaded = true;
            this.startPolling();
        },

        async checkAdminStatus() {
            try {
                const response = await fetch('/api/users/me');
                if (response.ok) {
                    const data = await response.json();
                    this.isAdmin = data.is_admin || false;
                }
            } catch (error) {
                console.error('Failed to check admin status:', error);
            }
        },

        startPolling() {
            this.stopPolling();
            // Poll every 10 seconds for job updates
            this.pollingInterval = setInterval(() => {
                this.loadJobs(true);  // silent refresh
            }, 10000);
        },

        stopPolling() {
            if (this.pollingInterval) {
                clearInterval(this.pollingInterval);
                this.pollingInterval = null;
            }
        },

        async loadJobs(silent = false) {
            if (!silent) {
                this.loading = true;
            }

            try {
                const params = new URLSearchParams();
                params.set('limit', '100');

                if (this.typeFilter !== 'all') {
                    params.set('job_type', this.typeFilter);
                }
                if (this.providerFilter) {
                    params.set('provider', this.providerFilter);
                }
                if (this.statusFilter) {
                    params.set('status', this.statusFilter);
                }

                // Use admin endpoint when filtering by user
                let url;
                if (this.isAdmin && this.adminUserFilter && this.adminUserFilter.trim()) {
                    url = `/api/queue/user/${this.adminUserFilter.trim()}?${params}`;
                } else {
                    url = `/api/cloud/jobs?${params}`;
                }

                const response = await fetch(url);
                if (response.ok) {
                    const data = await response.json();
                    this.jobs = data.jobs || [];

                    // Update selected job if still in list
                    if (this.selectedJob) {
                        const updated = this.jobs.find(j => j.job_id === this.selectedJob.job_id);
                        if (updated) {
                            this.selectedJob = updated;
                        }
                    }
                }
            } catch (error) {
                console.error('Failed to load jobs:', error);
            } finally {
                this.loading = false;
            }
        },

        selectJob(job) {
            this.selectedJob = job;
        },

        async cancelJob(jobId) {
            if (!confirm('Are you sure you want to cancel this job?')) {
                return;
            }

            try {
                const response = await fetch(`/api/cloud/jobs/${jobId}/cancel`, {
                    method: 'POST'
                });
                if (response.ok) {
                    await this.loadJobs();
                    if (window.showToast) {
                        window.showToast('Job cancelled', 'success');
                    }
                } else {
                    const data = await response.json();
                    if (window.showToast) {
                        window.showToast(data.detail || 'Failed to cancel job', 'error');
                    }
                }
            } catch (error) {
                console.error('Failed to cancel job:', error);
                if (window.showToast) {
                    window.showToast('Failed to cancel job', 'error');
                }
            }
        },

        async deleteJob(jobId) {
            if (!confirm('Are you sure you want to delete this job from history?')) {
                return;
            }

            try {
                const response = await fetch(`/api/cloud/jobs/${jobId}`, {
                    method: 'DELETE'
                });
                if (response.ok) {
                    if (this.selectedJob && this.selectedJob.job_id === jobId) {
                        this.selectedJob = null;
                    }
                    await this.loadJobs();
                    if (window.showToast) {
                        window.showToast('Job deleted', 'success');
                    }
                } else {
                    const data = await response.json();
                    if (window.showToast) {
                        window.showToast(data.detail || 'Failed to delete job', 'error');
                    }
                }
            } catch (error) {
                console.error('Failed to delete job:', error);
                if (window.showToast) {
                    window.showToast('Failed to delete job', 'error');
                }
            }
        },

        async viewLogs(jobId) {
            this.logsJobId = jobId;
            this.logs = '';
            this.logsLoading = true;

            // Show modal
            if (!this.logsModal) {
                const modalEl = this.$refs.logsModal;
                if (modalEl) {
                    this.logsModal = new bootstrap.Modal(modalEl);
                }
            }
            if (this.logsModal) {
                this.logsModal.show();
            }

            try {
                const response = await fetch(`/api/cloud/jobs/${jobId}/logs`);
                if (response.ok) {
                    const data = await response.json();
                    this.logs = data.logs || 'No logs available';
                } else {
                    this.logs = 'Failed to load logs';
                }
            } catch (error) {
                console.error('Failed to load logs:', error);
                this.logs = 'Failed to load logs: ' + error.message;
            } finally {
                this.logsLoading = false;
            }
        },

        // Status icon mapping
        statusIcon(status) {
            const icons = {
                'pending': 'fa-clock',
                'queued': 'fa-list',
                'uploading': 'fa-upload',
                'running': 'fa-play',
                'completed': 'fa-check',
                'failed': 'fa-times',
                'cancelled': 'fa-ban'
            };
            return icons[status] || 'fa-question';
        },

        // Format relative date - delegates to UIHelpers
        formatDate(dateStr) {
            return window.UIHelpers?.formatRelativeTime(dateStr, { fallback: dateStr }) || dateStr;
        },

        // Format full date/time - delegates to UIHelpers
        formatDateTime(dateStr) {
            return window.UIHelpers?.formatDateTime(dateStr, { fallback: dateStr }) || dateStr;
        },

        // Format duration in seconds to human readable
        formatDuration(seconds) {
            if (!seconds || seconds < 0) return '';

            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);

            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        },

        // Cleanup on destroy
        destroy() {
            this.stopPolling();
        }
    };
};
}
