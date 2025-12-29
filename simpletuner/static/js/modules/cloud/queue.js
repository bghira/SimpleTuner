/**
 * Cloud Dashboard - Queue Management Module
 *
 * Handles queue stats, concurrency settings, and queue admin operations.
 */

window.cloudQueueMethods = {
    // NOTE: All getters (visiblePendingJobs, visibleBlockedJobs, userQueuePosition, userEstimatedWait)
    // moved to index.js final return object to avoid spread evaluation issues

    toggleQueuePanel() {
        this.showQueuePanel = !this.showQueuePanel;
        if (this.showQueuePanel) {
            this.refreshQueueStats();
            this.loadQueuePendingJobs();
            this.loadQueueBlockedJobs();
        }
    },

    async refreshQueueStats() {
        if (this.loadingQueueStats) return;
        this.loadingQueueStats = true;
        try {
            const response = await fetch('/api/queue/stats');
            if (response.ok) {
                const data = await response.json();
                Object.assign(this.queueStats, data);
                this.queueSettings.max_concurrent = data.max_concurrent || 5;
                this.queueSettings.user_max_concurrent = data.user_max_concurrent || 2;
                this.queueSettings.team_max_concurrent = data.team_max_concurrent || 10;
                this.queueSettings.enable_fair_share = data.enable_fair_share || false;
            }
        } catch (error) {
            console.error('Failed to load queue stats:', error);
        } finally {
            this.loadingQueueStats = false;
        }
    },

    async loadQueuePendingJobs() {
        try {
            const response = await fetch('/api/queue?status=pending&limit=20');
            if (response.ok) {
                const data = await response.json();
                this.queuePendingJobs = data.entries || [];

                // Auto-show queue panel if user has pending jobs
                if (!this.hasAdminAccess && this.queuePendingJobs.length > 0) {
                    const userId = this.currentUser?.id;
                    const hasOwnPending = this.queuePendingJobs.some(job => job.user_id === userId);
                    if (hasOwnPending) {
                        this.showQueuePanel = true;
                    }
                }
            }
        } catch (error) {
            console.error('Failed to load pending jobs:', error);
        }
    },

    async loadQueueBlockedJobs() {
        try {
            const response = await fetch('/api/queue?status=blocked&limit=20');
            if (response.ok) {
                const data = await response.json();
                this.queueBlockedJobs = data.entries || [];
            }
        } catch (error) {
            console.error('Failed to load blocked jobs:', error);
        }
    },

    async updateConcurrencyLimits() {
        this.savingQueueSettings = true;
        try {
            const response = await fetch('/api/queue/concurrency', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    max_concurrent: this.queueSettings.max_concurrent,
                    user_max_concurrent: this.queueSettings.user_max_concurrent,
                    team_max_concurrent: this.queueSettings.team_max_concurrent,
                    enable_fair_share: this.queueSettings.enable_fair_share,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                this.queueSettings.max_concurrent = data.max_concurrent;
                this.queueSettings.user_max_concurrent = data.user_max_concurrent;
                this.queueSettings.team_max_concurrent = data.team_max_concurrent;
                this.queueSettings.enable_fair_share = data.enable_fair_share;
                if (window.showToast) {
                    window.showToast('Queue settings updated', 'success');
                }
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to update settings', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to update concurrency:', error);
            if (window.showToast) {
                window.showToast('Failed to update settings', 'error');
            }
        } finally {
            this.savingQueueSettings = false;
        }
    },

    async processQueue() {
        if (this.processingQueue) return;
        this.processingQueue = true;
        try {
            const response = await fetch('/api/queue/process', { method: 'POST' });
            if (response.ok) {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(`Dispatched ${data.dispatched} job(s)`, 'success');
                }
                await this.refreshQueueStats();
                await this.loadJobs();
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to process queue', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to process queue:', error);
            if (window.showToast) {
                window.showToast('Failed to process queue', 'error');
            }
        } finally {
            this.processingQueue = false;
        }
    },

    async cleanupQueue() {
        if (this.cleaningQueue) return;
        if (!confirm(`Remove completed entries older than ${this.cleanupDays} days?`)) return;

        this.cleaningQueue = true;
        try {
            const response = await fetch(`/api/queue/cleanup?days=${this.cleanupDays}`, {
                method: 'POST',
            });
            if (response.ok) {
                const data = await response.json();
                this.lastCleanupResult = data;
                if (window.showToast) {
                    window.showToast(`Cleaned up ${data.deleted} entries`, 'success');
                }
            } else {
                if (window.showToast) {
                    window.showToast('Failed to cleanup queue', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to cleanup queue:', error);
            if (window.showToast) {
                window.showToast('Failed to cleanup queue', 'error');
            }
        } finally {
            this.cleaningQueue = false;
        }
    },

    async cancelQueuedJob(jobId) {
        if (!confirm('Cancel this queued job?')) return;
        try {
            const response = await fetch(`/api/queue/${jobId}/cancel`, { method: 'POST' });
            if (response.ok) {
                if (window.showToast) {
                    window.showToast('Job cancelled', 'success');
                }
                await this.refreshQueueStats();
                await this.loadQueuePendingJobs();
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

    async approveQueuedJob(jobId) {
        try {
            const response = await fetch(`/api/queue/${jobId}/approve`, { method: 'POST' });
            if (response.ok) {
                if (window.showToast) {
                    window.showToast('Job approved', 'success');
                }
                await this.refreshQueueStats();
                await this.loadQueueBlockedJobs();
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to approve job', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to approve job:', error);
            if (window.showToast) {
                window.showToast('Failed to approve job', 'error');
            }
        }
    },

    async rejectQueuedJob(jobId, reason) {
        try {
            const response = await fetch(`/api/queue/${jobId}/reject?reason=${encodeURIComponent(reason)}`, {
                method: 'POST',
            });
            if (response.ok) {
                if (window.showToast) {
                    window.showToast('Job rejected', 'success');
                }
                await this.refreshQueueStats();
                await this.loadQueueBlockedJobs();
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to reject job', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to reject job:', error);
            if (window.showToast) {
                window.showToast('Failed to reject job', 'error');
            }
        }
    },

    openApprovalModal(action, job) {
        this.approvalModal.open = true;
        this.approvalModal.action = action;
        this.approvalModal.job = job;
        this.approvalModal.reason = '';
        this.approvalModal.notes = '';
        this.approvalModal.processing = false;
    },

    closeApprovalModal() {
        this.approvalModal.open = false;
        this.approvalModal.action = null;
        this.approvalModal.job = null;
        this.approvalModal.reason = '';
        this.approvalModal.notes = '';
        this.approvalModal.processing = false;
    },

    async submitApprovalAction() {
        if (!this.approvalModal.job) return;

        const jobId = this.approvalModal.job.job_id;
        const action = this.approvalModal.action;

        if (action === 'reject' && !this.approvalModal.reason.trim()) {
            if (window.showToast) {
                window.showToast('Rejection reason is required', 'error');
            }
            return;
        }

        this.approvalModal.processing = true;

        try {
            if (action === 'approve') {
                await this.approveQueuedJob(jobId);
            } else {
                await this.rejectQueuedJob(jobId, this.approvalModal.reason);
            }
            this.closeApprovalModal();
        } catch (error) {
            console.error('Failed to process approval action:', error);
        } finally {
            this.approvalModal.processing = false;
        }
    },
};
