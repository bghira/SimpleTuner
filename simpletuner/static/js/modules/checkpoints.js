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
            environment: null,
            sortBy: 'step-desc',
            retentionLimit: 10,
            originalRetentionLimit: 10,
            retentionDirty: false,
            cleanupPreview: null,
            validationResult: null,
            markdownService: window.markdownService || null,
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
                try {
                    await this.ensureEnvironment();
                } catch (error) {
                    console.error('Unable to initialize checkpoints manager:', error);
                    if (window.showToast) {
                        window.showToast('Failed to determine active environment', 'error');
                    }
                    return;
                }

                this.markdownService = window.markdownService || this.markdownService;

                await this.loadCheckpoints();
                await this.loadRetentionConfig();
            },

            async ensureEnvironment() {
                if (this.environment) {
                    return this.environment;
                }

                const response = await fetch('/api/configs/active');
                if (!response.ok) {
                    throw new Error('Failed to get active config');
                }

                const configData = await response.json();
                this.environment = configData.name;
                return this.environment;
            },

            normalizeCheckpoint(raw, index) {
                const name = raw.name || raw.id || `checkpoint-${index}`;
                const id = raw.id || name;

                let createdAt = raw.created_at || null;
                if (!createdAt && raw.timestamp) {
                    const epochMs = typeof raw.timestamp === 'number'
                        ? raw.timestamp * 1000
                        : Date.parse(raw.timestamp);
                    if (!Number.isNaN(epochMs)) {
                        createdAt = new Date(epochMs).toISOString();
                    }
                }

                if (!createdAt && raw.modified_at) {
                    const modified = typeof raw.modified_at === 'number'
                        ? raw.modified_at * 1000
                        : Date.parse(raw.modified_at);
                    if (!Number.isNaN(modified)) {
                        createdAt = new Date(modified).toISOString();
                    }
                }

                const sizeBytes = raw.size_bytes ?? raw.size ?? 0;
                const validation = raw.validation || {};
                let validationStatus = raw.validation_status ?? 'pending';
                if (validation.valid === true) {
                    validationStatus = 'valid';
                } else if (validation.valid === false) {
                    validationStatus = 'invalid';
                }
                const validationMessage = validation.error || validation.message || null;
                const assets = Array.isArray(raw.assets) ? raw.assets : [];
                const readme = raw.readme || null;

                const checkpoint = {
                    ...raw,
                    id,
                    name,
                    created_at: createdAt,
                    size_bytes: sizeBytes,
                    size: sizeBytes,
                    files: raw.files || [],
                    validation_status: validationStatus,
                    validation_message: validationMessage,
                    validation,
                    assets,
                    readme,
                };

                this._applyReadmeMetadata(checkpoint);
                return checkpoint;
            },

            _timestampForSort(checkpoint) {
                const source = checkpoint.created_at || checkpoint.timestamp || 0;
                if (typeof source === 'number') {
                    return source;
                }
                const parsed = Date.parse(source);
                return Number.isNaN(parsed) ? 0 : parsed;
            },

            _applyReadmeMetadata(checkpoint) {
                if (!checkpoint) {
                    return;
                }

                const service = this.markdownService || window.markdownService;
                if (service && typeof service.render === 'function') {
                    try {
                        const result = service.render(checkpoint.readme, checkpoint.assets);
                        checkpoint.readme_html = result.html || '';
                        checkpoint.readme_tags = Array.isArray(result.tags) ? result.tags : [];
                        checkpoint.readme_gallery = Array.isArray(result.gallery) ? result.gallery : [];
                        checkpoint.readme_metadata = result.metadata || {};
                        checkpoint.readme_front_matter = result.frontMatterRaw || null;
                        return;
                    } catch (error) {
                        console.warn('Failed to render checkpoint README via MarkdownService:', error);
                    }
                }

                const legacySource = typeof checkpoint.readme === 'string'
                    ? checkpoint.readme
                    : checkpoint.readme?.body || '';
                checkpoint.readme_html = this._renderReadmeLegacy(legacySource);
                checkpoint.readme_tags = Array.isArray(checkpoint.readme_tags) ? checkpoint.readme_tags : [];
                checkpoint.readme_gallery = Array.isArray(checkpoint.readme_gallery) ? checkpoint.readme_gallery : [];
                checkpoint.readme_metadata = checkpoint.readme_metadata || {};
            },

            // Data Loading
            async loadCheckpoints() {
                this.loading.checkpoints = true;
                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints?environment=${encodeURIComponent(environment)}&sort_by=${this.sortBy}`);
                    if (!response.ok) {
                        throw new Error('Failed to load checkpoints');
                    }

                    const data = await response.json();
                    const checkpoints = Array.isArray(data.checkpoints) ? data.checkpoints : [];
                    this.checkpoints = checkpoints.map((cp, idx) => this.normalizeCheckpoint(cp, idx));
                    this.sortCheckpoints();
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
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/retention?environment=${encodeURIComponent(environment)}`);
                    if (!response.ok) {
                        throw new Error('Failed to load retention config');
                    }

                    const data = await response.json();
                    this.retentionLimit = data.retention_limit ?? 10;
                    this.originalRetentionLimit = this.retentionLimit;
                    this.retentionDirty = false;
                } catch (error) {
                    console.error('Failed to load retention config:', error);
                    if (window.showToast) {
                        window.showToast('Failed to load retention settings', 'error');
                    }
                    this.retentionLimit = 10;
                    this.originalRetentionLimit = 10;
                }
            },

            // Sorting
            sortCheckpoints() {
                const [field, order] = this.sortBy.split('-');

                this.checkpoints.sort((a, b) => {
                    let valA;
                    let valB;

                    switch (field) {
                        case 'step':
                            valA = a.step || 0;
                            valB = b.step || 0;
                            break;
                        case 'date':
                            valA = this._timestampForSort(a);
                            valB = this._timestampForSort(b);
                            break;
                        case 'size':
                            valA = a.size_bytes ?? a.size ?? 0;
                            valB = b.size_bytes ?? b.size ?? 0;
                            break;
                        default:
                            return 0;
                    }

                    return order === 'asc' ? valA - valB : valB - valA;
                });
            },

            // Checkpoint Selection
            async selectCheckpoint(id) {
                const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                if (!checkpoint) {
                    return;
                }

                this.selectedCheckpoint = checkpoint;
                this.validationResult = null;
            },

            // Validation
            async validateCheckpoint(id) {
                if (this.loading.validation) {
                    return;
                }

                this.loading.validation = true;
                this.validationResult = null;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/${encodeURIComponent(id)}/validate`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ environment })
                    });

                    if (!response.ok) {
                        throw new Error('Validation request failed');
                    }

                    const result = await response.json();
                    this.validationResult = result;

                    const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                    if (checkpoint) {
                        checkpoint.validation_status = result.valid ? 'valid' : 'invalid';
                        checkpoint.validation_message = result.error || result.message || null;
                        checkpoint.validation = {
                            valid: result.valid,
                            error: result.error || null,
                        };
                    }

                    if (this.selectedCheckpoint && (this.selectedCheckpoint.id === id || this.selectedCheckpoint.name === id)) {
                        this.selectedCheckpoint.validation_status = result.valid ? 'valid' : 'invalid';
                        this.selectedCheckpoint.validation_message = result.error || result.message || null;
                        this.selectedCheckpoint.validation = {
                            valid: result.valid,
                            error: result.error || null,
                        };
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

                const checkpoint = this.checkpoints.find(cp => cp.id === id || cp.name === id);
                if (!checkpoint) {
                    return;
                }

                if (!confirm('Are you sure you want to delete this checkpoint? This action cannot be undone.')) {
                    return;
                }

                this.loading.delete = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch(`/api/checkpoints/${encodeURIComponent(checkpoint.name)}?environment=${encodeURIComponent(environment)}`, {
                        method: 'DELETE'
                    });

                    if (!response.ok) {
                        throw new Error('Failed to delete checkpoint');
                    }

                    this.checkpoints = this.checkpoints.filter(cp => cp.id !== checkpoint.id);

                    if (this.selectedCheckpoint && (this.selectedCheckpoint.id === checkpoint.id || this.selectedCheckpoint.name === checkpoint.name)) {
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

            _renderReadmeLegacy(content) {
                if (!content) {
                    return '';
                }

                const escape = (str) => String(str)
                    .replace(/&/g, '&amp;')
                    .replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;');

                let html = escape(content);

                html = html.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {
                    const safeAlt = escape(alt);
                    const safeUrl = url.replace(/"/g, '&quot;');
                    return `<img src="${safeUrl}" alt="${safeAlt}" class="readme-image" />`;
                });

                html = html.replace(/\n{2,}/g, '</p><p>');
                html = html.replace(/\n/g, '<br />');

                return `<p>${html}</p>`;
            },

            // Retention Management
            markRetentionDirty() {
                this.retentionDirty = this.retentionLimit !== this.originalRetentionLimit;
                this.cleanupPreview = null;
            },

            async saveRetentionConfig() {
                if (this.loading.saveRetention || !this.retentionDirty) {
                    return;
                }

                this.loading.saveRetention = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch('/api/checkpoints/retention', {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
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
                    const environment = await this.ensureEnvironment();
                    const response = await fetch('/api/checkpoints/cleanup/preview', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
                            limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to preview cleanup');
                    }

                    const data = await response.json();
                    const checkpointsToRemove = Array.isArray(data.checkpoints_to_remove)
                        ? data.checkpoints_to_remove.map((cp, idx) => this.normalizeCheckpoint(cp, idx))
                        : [];

                    this.cleanupPreview = {
                        ...data,
                        checkpoints_to_remove: checkpointsToRemove,
                        count_to_remove: data.count_to_remove ?? checkpointsToRemove.length
                    };

                    if (window.showToast) {
                        window.showToast(
                            `Preview: ${this.cleanupPreview.count_to_remove} checkpoint(s) will be removed`,
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

                if (!this.cleanupPreview || this.cleanupPreview.count_to_remove === 0) {
                    if (window.showToast) {
                        window.showToast('No checkpoints to remove', 'info');
                    }
                    return;
                }

                if (!confirm(`This will permanently delete ${this.cleanupPreview.count_to_remove} checkpoint(s). This action cannot be undone. Continue?`)) {
                    return;
                }

                this.loading.cleanup = true;

                try {
                    const environment = await this.ensureEnvironment();
                    const response = await fetch('/api/checkpoints/cleanup/execute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            environment,
                            limit: this.retentionLimit
                        })
                    });

                    if (!response.ok) {
                        throw new Error('Failed to execute cleanup');
                    }

                    const result = await response.json();
                    const removedCheckpoints = Array.isArray(result.removed_checkpoints)
                        ? result.removed_checkpoints.map((cp, idx) => this.normalizeCheckpoint(cp, idx))
                        : [];
                    const removedIds = removedCheckpoints.map(cp => cp.id);

                    if (window.showToast) {
                        const removedCount = result.count_removed ?? removedIds.length;
                        window.showToast(
                            `Cleanup complete: ${removedCount} checkpoint(s) removed`,
                            'success'
                        );
                    }

                    await this.loadCheckpoints();

                    this.cleanupPreview = null;
                    if (this.selectedCheckpoint && removedIds.includes(this.selectedCheckpoint.id)) {
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
                } catch (error) {
                    return 'Unknown';
                }
            },

            formatDateTime(dateString) {
                if (!dateString) return 'Unknown';
                try {
                    const date = new Date(dateString);
                    return date.toLocaleString();
                } catch (error) {
                    return 'Unknown';
                }
            },

            formatSize(bytes) {
                if (!bytes || bytes === 0) return '0 B';
                const k = 1024;
                const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
                const index = Math.floor(Math.log(bytes) / Math.log(k));
                return `${parseFloat((bytes / Math.pow(k, index)).toFixed(2))} ${sizes[index]}`;
            }
        };
    };
}
