/**
 * Checkpoints Module
 * Handles training checkpoint browsing, validation, and cleanup management
 */

if (!window.checkpointsManager) {
    window.checkpointsManager = function() {
        return {
            // State
            checkpoints: [],
            selectedCheckpoint: null,
            sortBy: 'step-desc',
            retentionLimit: 10,
            originalRetentionLimit: 10,
            retentionDirty: false,
            cleanupPreview: null,
            validationResult: null,
            loading: {
                checkpoints: false,
                validation: false,
                delete: false,
                preview: false,
                cleanup: false,
                saveRetention: false
            },

            // Lifecycle
            async init() {
                await this.loadCheckpoints();
                await this.loadRetentionConfig();
            },

            // Data Loading
            async loadCheckpoints() {
                this.loading.checkpoints = true;
                try {
                    // Get active config name
                    const configResponse = await fetch('/api/configs/active');
                    if (!configResponse.ok) {
                        throw new Error('Failed to get active config');
                    }
                    const configData = await configResponse.json();
                    const environment = configData.name;

                    // Load checkpoints for this environment
                    const response = await fetch(`/api/checkpoints?environment=${encodeURIComponent(environment)}&sort_by=${this.sortBy}`);
                    if (!response.ok) {
                        throw new Error('Failed to load checkpoints');
                    }
                    const data = await response.json();
                    this.checkpoints = data.checkpoints || [];
                } catch (error) {
                    console.error('Failed to load checkpoints:', error);
                    if (window.showToast) {
                        window.showToast('Failed to load checkpoints', 'error');
                    }
                    this.checkpoints = [];
                } finally {
                    this.loading.checkpoints = false;
                }
            },

            async loadRetentionConfig() {
                try {
                    // Get active config name
                    const configResponse = await fetch('/api/configs/active');
                    if (!configResponse.ok) {
                        throw new Error('Failed to get active config');
                    }
                    const configData = await configResponse.json();
                    const environment = configData.name;

                    // Load retention config for this environment
                    const response = await fetch(`/api/checkpoints/retention?environment=${encodeURIComponent(environment)}`);
                    if (!response.ok) {
                        throw new Error('Failed to load retention config');
                    }
                    const data = await response.json();
                    this.retentionLimit = data.retention_limit || 10;
                    this.originalRetentionLimit = this.retentionLimit;
                    this.retentionDirty = false;
                } catch (error) {
                    console.error('Failed to load retention config:', error);
                    // Use default value on error
                    this.retentionLimit = 10;
                    this.originalRetentionLimit = 10;
                }
            },

            // Sorting
            sortCheckpoints() {
                const [field, order] = this.sortBy.split('-');

                this.checkpoints.sort((a, b) => {
                    let valA, valB;

                    switch (field) {
                        case 'step':
                            valA = a.step || 0;
                            valB = b.step || 0;
                            break;
                        case 'date':
                            valA = new Date(a.created_at || 0).getTime();
                            valB = new Date(b.created_at || 0).getTime();
                            break;
                        case 'size':
                            valA = a.size || 0;
                            valB = b.size || 0;
                            break;
                        default:
                            return 0;
                    }

                    if (order === 'asc') {
                        return valA - valB;
                    } else {
                        return valB - valA;
                    }
                });
            },

            // Checkpoint Selection
            async selectCheckpoint(id) {
                const checkpoint = this.checkpoints.find(cp => cp.id === id);
                if (!checkpoint) {
                    return;
                }

                this.selectedCheckpoint = checkpoint;
                this.validationResult = null;

                // Load detailed info if needed
                try {
                    const response = await fetch(`/api/checkpoints/${id}/details`);
                    if (response.ok) {
                        const data = await response.json();
                        // Update selected checkpoint with detailed info
                        this.selectedCheckpoint = { ...this.selectedCheckpoint, ...data };
                    }
                } catch (error) {
                    console.error('Failed to load checkpoint details:', error);
                    // Continue with basic info
                }
            },

            // Validation
            async validateCheckpoint(id) {
                if (this.loading.validation) {
                    return;
                }

                this.loading.validation = true;
                this.validationResult = null;

                try {
                    const response = await fetch(`/api/checkpoints/${id}/validate`, {
                        method: 'POST'
                    });

                    if (!response.ok) {
                        throw new Error('Validation request failed');
                    }

                    const result = await response.json();
                    this.validationResult = result;

                    // Update checkpoint status
                    const checkpoint = this.checkpoints.find(cp => cp.id === id);
                    if (checkpoint) {
                        checkpoint.validation_status = result.valid ? 'valid' : 'invalid';
                    }

                    if (this.selectedCheckpoint && this.selectedCheckpoint.id === id) {
                        this.selectedCheckpoint.validation_status = result.valid ? 'valid' : 'invalid';
                    }

                    if (window.showToast) {
                        window.showToast(
                            result.valid ? 'Checkpoint is valid' : 'Checkpoint validation failed',
                            result.valid ? 'success' : 'error'
                        );
                    }
                } catch (error) {
                    console.error('Failed to validate checkpoint:', error);
                    this.validationResult = {
                        valid: false,
                        message: error.message || 'Validation failed'
                    };
                    if (window.showToast) {
                        window.showToast('Failed to validate checkpoint', 'error');
                    }
                } finally {
                    this.loading.validation = false;
                }
            },

            // Deletion
            async deleteCheckpoint(id) {
                if (this.loading.delete) {
                    return;
                }

                if (!confirm('Are you sure you want to delete this checkpoint? This action cannot be undone.')) {
                    return;
                }

                this.loading.delete = true;

                try {
                    const response = await fetch(`/api/checkpoints/${id}`, {
                        method: 'DELETE'
                    });

                    if (!response.ok) {
                        throw new Error('Failed to delete checkpoint');
                    }

                    // Remove from list
                    this.checkpoints = this.checkpoints.filter(cp => cp.id !== id);

                    // Clear selection if deleted checkpoint was selected
                    if (this.selectedCheckpoint && this.selectedCheckpoint.id === id) {
                        this.selectedCheckpoint = null;
                        this.validationResult = null;
                    }

                    if (window.showToast) {
                        window.showToast('Checkpoint deleted successfully', 'success');
                    }
                } catch (error) {
                    console.error('Failed to delete checkpoint:', error);
                    if (window.showToast) {
                        window.showToast('Failed to delete checkpoint', 'error');
                    }
                } finally {
                    this.loading.delete = false;
                }
            },

            // Retention Management
            markRetentionDirty() {
                this.retentionDirty = this.retentionLimit !== this.originalRetentionLimit;
                // Clear cleanup preview when retention limit changes
                this.cleanupPreview = null;
            },

            async saveRetentionConfig() {
                if (this.loading.saveRetention || !this.retentionDirty) {
                    return;
                }

                this.loading.saveRetention = true;

                try {
                    // Get active config name
                    const configResponse = await fetch('/api/configs/active');
                    if (!configResponse.ok) {
                        throw new Error('Failed to get active config');
                    }
                    const configData = await configResponse.json();
                    const environment = configData.name;

                    // Save retention config
                    const response = await fetch('/api/checkpoints/retention', {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment: environment,
                            retention_limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to save retention config');
                    }

                    this.originalRetentionLimit = this.retentionLimit;
                    this.retentionDirty = false;

                    if (window.showToast) {
                        window.showToast('Retention settings saved', 'success');
                    }
                } catch (error) {
                    console.error('Failed to save retention config:', error);
                    if (window.showToast) {
                        window.showToast('Failed to save retention settings', 'error');
                    }
                } finally {
                    this.loading.saveRetention = false;
                }
            },

            // Cleanup Operations
            async previewCleanup() {
                if (this.loading.preview) {
                    return;
                }

                this.loading.preview = true;
                this.cleanupPreview = null;

                try {
                    const response = await fetch('/api/checkpoints/cleanup/preview', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            retention_limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to preview cleanup');
                    }

                    const data = await response.json();
                    this.cleanupPreview = data;

                    if (window.showToast) {
                        window.showToast(
                            `Preview: ${data.to_remove} checkpoint(s) will be removed`,
                            'info'
                        );
                    }
                } catch (error) {
                    console.error('Failed to preview cleanup:', error);
                    if (window.showToast) {
                        window.showToast('Failed to preview cleanup', 'error');
                    }
                } finally {
                    this.loading.preview = false;
                }
            },

            async executeCleanup() {
                if (this.loading.cleanup) {
                    return;
                }

                if (!this.cleanupPreview || this.cleanupPreview.to_remove === 0) {
                    if (window.showToast) {
                        window.showToast('No checkpoints to remove', 'info');
                    }
                    return;
                }

                const message = `This will permanently delete ${this.cleanupPreview.to_remove} checkpoint(s) and reclaim ${this.formatSize(this.cleanupPreview.space_to_reclaim)} of disk space. This action cannot be undone. Continue?`;

                if (!confirm(message)) {
                    return;
                }

                this.loading.cleanup = true;

                try {
                    const response = await fetch('/api/checkpoints/cleanup/execute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            retention_limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to execute cleanup');
                    }

                    const result = await response.json();

                    if (window.showToast) {
                        window.showToast(
                            `Cleanup complete: ${result.removed_count} checkpoint(s) removed`,
                            'success'
                        );
                    }

                    // Refresh checkpoints list
                    await this.loadCheckpoints();

                    // Clear preview and selection
                    this.cleanupPreview = null;
                    if (this.selectedCheckpoint && result.removed_ids && result.removed_ids.includes(this.selectedCheckpoint.id)) {
                        this.selectedCheckpoint = null;
                        this.validationResult = null;
                    }
                } catch (error) {
                    console.error('Failed to execute cleanup:', error);
                    if (window.showToast) {
                        window.showToast('Failed to execute cleanup', 'error');
                    }
                } finally {
                    this.loading.cleanup = false;
                }
            },

            // Utility Functions
            formatDate(dateString) {
                if (!dateString) return 'Unknown';
                try {
                    const date = new Date(dateString);
                    const now = new Date();
                    const diffMs = now - date;
                    const diffMins = Math.floor(diffMs / 60000);
                    const diffHours = Math.floor(diffMs / 3600000);
                    const diffDays = Math.floor(diffMs / 86400000);

                    if (diffMins < 1) return 'Just now';
                    if (diffMins < 60) return `${diffMins} min ago`;
                    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
                    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;

                    return date.toLocaleDateString();
                } catch (e) {
                    return 'Unknown';
                }
            },

            formatDateTime(dateString) {
                if (!dateString) return 'Unknown';
                try {
                    const date = new Date(dateString);
                    return date.toLocaleString();
                } catch (e) {
                    return 'Unknown';
                }
            },

            formatSize(bytes) {
                if (!bytes || bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
            }
        };
    };
}
