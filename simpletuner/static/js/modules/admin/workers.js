/**
 * Admin Panel - Workers Management Module
 *
 * Handles GPU worker CRUD, status monitoring, and token management.
 */

window.adminWorkerMethods = {
    async loadWorkers() {
        this.workersLoading = true;
        try {
            const response = await fetch('/api/admin/workers');
            if (response.ok) {
                const data = await response.json();
                this.workers = data.workers || [];
                this.updateWorkerStats();
            } else {
                console.error('Failed to load workers:', response.statusText);
                if (window.showToast) window.showToast('Failed to load workers', 'error');
            }
        } catch (error) {
            console.error('Failed to load workers:', error);
            if (window.showToast) window.showToast('Failed to load workers', 'error');
        } finally {
            this.workersLoading = false;
        }
    },

    updateWorkerStats() {
        this.workerStats = {
            total: this.workers.length,
            idle: this.workers.filter(w => w.status === 'idle').length,
            busy: this.workers.filter(w => w.status === 'busy').length,
            offline: this.workers.filter(w => w.status === 'offline').length,
        };
    },

    getWorkerStatusBadgeClass(status) {
        const classes = {
            'idle': 'bg-success',
            'busy': 'bg-info',
            'offline': 'bg-secondary',
            'connecting': 'bg-warning',
            'draining': 'bg-warning',
        };
        return classes[status] || 'bg-secondary';
    },

    formatWorkerTimeAgo(timestamp) {
        if (!timestamp) return 'Never';
        const date = new Date(timestamp);
        const now = new Date();
        const seconds = Math.floor((now - date) / 1000);

        if (seconds < 60) return 'Just now';
        if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
        if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
        return Math.floor(seconds / 86400) + 'd ago';
    },

    parseWorkerLabels(str) {
        if (!str) return {};
        const labels = {};
        str.split(',').forEach(pair => {
            const [key, value] = pair.split('=').map(s => s.trim());
            if (key && value) labels[key] = value;
        });
        return labels;
    },

    showAddWorkerModal() {
        this.workerForm = {
            name: '',
            worker_type: 'persistent',
            labels_str: '',
        };
        this.workerFormOpen = true;
    },

    async createWorker() {
        if (!this.workerForm.name) return;

        this.saving = true;
        this.error = null;
        try {
            const response = await fetch('/api/admin/workers', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: this.workerForm.name,
                    worker_type: this.workerForm.worker_type,
                    labels: this.parseWorkerLabels(this.workerForm.labels_str),
                }),
            });

            if (response.ok) {
                const data = await response.json();

                // Show token modal
                this.workerToken = data.token;
                this.workerConnectionCommand = data.connection_command || `# Configure your worker with token: ${data.token}`;
                this.workerFormOpen = false;
                this.workerTokenModalOpen = true;

                if (window.showToast) {
                    window.showToast('Worker created successfully', 'success');
                }

                // Refresh list
                await this.loadWorkers();
            } else {
                const data = await response.json();
                this.error = window.UIHelpers?.extractErrorMessage(data, 'Failed to create worker') || 'Failed to create worker';
                if (window.showToast) window.showToast(this.error, 'error');
            }
        } catch (error) {
            console.error('Failed to create worker:', error);
            this.error = 'Network error';
            if (window.showToast) window.showToast('Failed to create worker', 'error');
        } finally {
            this.saving = false;
        }
    },

    async drainWorker(worker) {
        if (!confirm(`Drain worker ${worker.name}? It will finish the current job and stop accepting new ones.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/admin/workers/${worker.worker_id}/drain`, {
                method: 'POST',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Worker draining initiated', 'success');
                await this.loadWorkers();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to drain worker', 'error');
            }
        } catch (error) {
            console.error('Failed to drain worker:', error);
            if (window.showToast) window.showToast('Failed to drain worker', 'error');
        }
    },

    async rotateWorkerToken(worker) {
        if (!confirm(`Rotate token for ${worker.name}? The old token will be immediately invalidated.`)) {
            return;
        }

        try {
            const response = await fetch(`/api/admin/workers/${worker.worker_id}/token`, {
                method: 'POST',
            });
            if (response.ok) {
                const data = await response.json();
                this.workerToken = data.token;
                this.workerConnectionCommand = data.connection_command || `# Configure your worker with token: ${data.token}`;
                this.workerTokenModalOpen = true;
                if (window.showToast) window.showToast('Token rotated successfully', 'success');
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to rotate token', 'error');
            }
        } catch (error) {
            console.error('Failed to rotate token:', error);
            if (window.showToast) window.showToast('Failed to rotate token', 'error');
        }
    },

    confirmDeleteWorker(worker) {
        this.deletingWorker = worker;
        this.deleteWorkerOpen = true;
    },

    async deleteWorker() {
        if (!this.deletingWorker) return;

        this.saving = true;
        try {
            const response = await fetch(`/api/admin/workers/${this.deletingWorker.worker_id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Worker removed', 'success');
                this.deleteWorkerOpen = false;
                this.deletingWorker = null;
                await this.loadWorkers();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to remove worker', 'error');
            }
        } catch (error) {
            console.error('Failed to remove worker:', error);
            if (window.showToast) window.showToast('Failed to remove worker', 'error');
        } finally {
            this.saving = false;
        }
    },

    copyWorkerToken() {
        if (this.workerToken) {
            navigator.clipboard.writeText(this.workerToken).then(() => {
                if (window.showToast) window.showToast('Token copied to clipboard', 'success');
            }).catch(err => {
                console.error('Failed to copy token:', err);
            });
        }
    },

    copyWorkerCommand() {
        if (this.workerConnectionCommand) {
            navigator.clipboard.writeText(this.workerConnectionCommand).then(() => {
                if (window.showToast) window.showToast('Command copied to clipboard', 'success');
            }).catch(err => {
                console.error('Failed to copy command:', err);
            });
        }
    },

    startWorkerAutoRefresh() {
        // Auto-refresh workers every 30 seconds when on the workers tab
        if (this.workerRefreshInterval) {
            clearInterval(this.workerRefreshInterval);
        }
        this.workerRefreshInterval = setInterval(() => {
            if (this.activeTab === 'workers') {
                this.loadWorkers();
            }
        }, 30000);
    },

    stopWorkerAutoRefresh() {
        if (this.workerRefreshInterval) {
            clearInterval(this.workerRefreshInterval);
            this.workerRefreshInterval = null;
        }
    },
};
