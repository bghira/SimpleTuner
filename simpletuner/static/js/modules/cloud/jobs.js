/**
 * Cloud Dashboard - Jobs Management Module
 *
 * Handles job loading, syncing, filtering, cancellation, and deletion.
 */

window.cloudJobMethods = {
    async loadJobs(syncActive = false) {
        if (!syncActive) {
            this.jobsLoading = true;
        }

        try {
            const params = new URLSearchParams();
            params.set('limit', '100');
            params.set('provider', this.activeProvider);

            const response = await fetch(`/api/cloud/jobs?${params}`);
            if (response.ok) {
                const data = await response.json();
                const newJobs = data.jobs || [];

                if (this.jobs.length > 0) {
                    // Update in place to preserve UI state
                    this.updateJobsInPlace(newJobs);
                } else {
                    this.jobs = newJobs;
                }

                this.jobsInitialized = true;
            }
        } catch (error) {
            console.error('Failed to load jobs:', error);
        } finally {
            this.jobsLoading = false;
        }
    },

    updateJobsInPlace(newJobs) {
        const existingIds = new Set(this.jobs.map(j => j.job_id));
        const newIds = new Set(newJobs.map(j => j.job_id));

        // Update existing jobs
        for (const newJob of newJobs) {
            const idx = this.jobs.findIndex(j => j.job_id === newJob.job_id);
            if (idx !== -1) {
                // Preserve inline progress state
                const existing = this.jobs[idx];
                newJob.inline_stage = existing.inline_stage || newJob.inline_stage;
                newJob.inline_log = existing.inline_log || newJob.inline_log;
                newJob.inline_progress = existing.inline_progress || newJob.inline_progress;
                this.jobs[idx] = newJob;
            } else {
                this.jobs.unshift(newJob);
            }
        }

        // Remove deleted jobs
        this.jobs = this.jobs.filter(j => newIds.has(j.job_id));

        // Update selected job if it still exists
        if (this.selectedJob) {
            const updated = this.jobs.find(j => j.job_id === this.selectedJob.job_id);
            if (updated) {
                this.selectedJob = updated;
            } else {
                this.selectedJob = null;
            }
        }
    },

    async syncJobs() {
        if (this.syncing) return;
        this.syncing = true;
        try {
            const response = await fetch('/api/cloud/jobs/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider: this.activeProvider }),
            });

            if (response.ok) {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(`Synced ${data.updated || 0} job(s)`, 'success');
                }
                await this.loadJobs();
            } else {
                if (window.showToast) {
                    window.showToast('Failed to sync jobs', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to sync jobs:', error);
            if (window.showToast) {
                window.showToast('Failed to sync jobs', 'error');
            }
        } finally {
            this.syncing = false;
        }
    },

    async cancelJob(jobId) {
        if (!confirm('Are you sure you want to cancel this job?')) return;

        try {
            const response = await fetch(`/api/cloud/jobs/${jobId}/cancel`, { method: 'POST' });
            if (response.ok) {
                this.loadJobs();
                if (window.showToast) {
                    window.showToast('Job cancelled', 'success');
                }
            } else {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to cancel job');
            }
        } catch (error) {
            if (window.showToast) {
                window.showToast(error.message, 'error');
            }
        }
    },

    async deleteJob(jobId) {
        if (!confirm('Are you sure you want to remove this job from history?')) return;
        try {
            const response = await fetch(`/api/cloud/jobs/${jobId}`, { method: 'DELETE' });
            if (response.ok) {
                this.loadJobs();
                this.loadMetrics();
                if (this.selectedJob && this.selectedJob.job_id === jobId) {
                    this.selectedJob = null;
                }
                if (window.showToast) {
                    window.showToast('Job removed from history', 'success');
                }
            } else {
                const data = await response.json();
                throw new Error(data.detail || 'Failed to delete job');
            }
        } catch (error) {
            if (window.showToast) {
                window.showToast(error.message, 'error');
            }
        }
    },

    selectJob(job) {
        this.selectedJob = job;
    },
};
