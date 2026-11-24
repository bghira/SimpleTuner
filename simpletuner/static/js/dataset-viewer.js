/**
 * Dataset Viewer Alpine.js component.
 *
 * Provides browsing of cached dataset contents including bucket summaries,
 * thumbnail grids with pagination, per-file metadata inspection, and
 * standalone metadata scanning with SSE progress updates.
 */
window.datasetViewerComponent = function () {
    return {
        // Viewer state
        datasets: [],
        loading: false,
        _summaryRequest: 0,
        expandedDataset: null,
        sourceShards: null,
        shardSearch: '',
        shardLoading: false,
        shardError: '',
        _shardRequest: 0,
        selectedBucket: null,
        thumbnails: [],
        loadingThumbnails: false,
        thumbnailOffset: 0,
        thumbnailLimit: 24,
        _thumbnailRequest: 0,
        selectedFile: null,
        selectedFileDatasetId: null,
        loadingFileDetail: false,
        _fileRequest: 0,
        previewOriginal: null,
        previewIntermediary: null,
        previewCropped: null,
        loadingPreview: false,
        cropOverlay: null,

        // Crop drag state
        cropDragging: false,
        cropDragStart: null,
        cropModified: false,
        savingCrop: false,
        _intermediaryImgEl: null,

        // Scan state
        scanning: false,
        scanProgress: { dataset_id: '', current: 0, total: 0 },
        _scanHandler: null,
        _queueHandler: null,
        _scanJobId: null,
        _scanQueueId: null,
        _scanQueueIndex: -1,
        _scanPlanScope: null,
        _scanRequest: 0,
        _scanStatusVersion: 0,
        _scanStarting: false,

        // Stage 3-5 state
        captionStatus: {},
        filterReport: {},
        datasetGraph: null,

        // Conditioning pair viewer state
        conditioningPairs: null,
        loadingPairs: false,
        conditioningPairSelection: null,
        _pairsRequest: 0,
        pairsOffset: 0,
        pairsLimit: 6,

        // Force scan confirmation dialog
        showingForceScanDialog: false,
        forceScanDatasetId: null,
        forceScanClearVae: false,
        forceScanClearConditioning: false,

        // Conditioning match state
        conditioningMatch: null,
        loadingConditioningMatch: false,

        // Bbox editor state
        bboxCanvas: null,
        bboxModified: false,
        savingBbox: false,

        // Cache job state
        caching: false,
        cacheError: '',
        cacheProgress: { dataset_id: '', cache_type: '', stage: '', current: 0, total: 0 },
        cacheCapabilities: { text_embeds: false, vae: false, conditioning_types: [] },
        _cacheHandler: null,
        _cacheJobId: null,
        _cacheRequest: 0,
        _cacheStatusVersion: 0,
        _cacheStarting: false,
        _cacheReconnectHandler: null,

        init() {
            this._restoreNavState();
            this.loadAllSummaries();
            this._checkActiveScan();
            this._checkActiveCacheJob();
            this.loadCacheCapabilities();
            this._setupSSEListener();
            this._setupConfigListener();
        },

        destroy() {
            this._clearSourceShards();
            this._cacheRequest++;
            this._scanRequest++;
            this._summaryRequest++;
            this._teardownSSEListener();
            this._teardownConfigListener();
        },

        // --- Navigation state persistence ---

        _saveNavState() {
            const state = {
                expandedDataset: this.expandedDataset,
                selectedBucketKey: this.selectedBucket?.key || null,
            };
            try {
                sessionStorage.setItem('dataset-viewer-nav', JSON.stringify(state));
            } catch (_) { /* quota or private mode */ }
        },

        _restoreNavState() {
            try {
                const raw = sessionStorage.getItem('dataset-viewer-nav');
                if (!raw) return;
                const state = JSON.parse(raw);
                if (state.expandedDataset) {
                    this.expandedDataset = state.expandedDataset;
                }
                // Bucket can only be resolved after summaries load, so stash the key
                this._pendingBucketKey = state.selectedBucketKey || null;
            } catch (_) { /* parse error or private mode */ }
        },

        // --- SSE handling ---

        _parseSSEDetail(event) {
            try {
                return typeof event.detail === 'string'
                    ? JSON.parse(event.detail)
                    : event.detail;
            } catch (e) { return null; }
        },

        _setupSSEListener() {
            this._scanHandler = (event) => {
                const data = this._parseSSEDetail(event);
                if (data) this._handleScanEvent(data);
            };
            this._queueHandler = (event) => {
                const data = this._parseSSEDetail(event);
                if (data) this._handleQueueEvent(data);
            };
            this._cacheHandler = (event) => {
                const data = this._parseSSEDetail(event);
                if (data) this._handleCacheEvent(data);
            };
            this._cacheReconnectHandler = () => {
                this._checkActiveCacheJob();
                this._checkActiveScan();
            };
            window.addEventListener('sse:connected', this._cacheReconnectHandler);
            window.addEventListener('sse:dataset_scan', this._scanHandler);
            window.addEventListener('sse:dataset_scan_queue', this._queueHandler);
            window.addEventListener('sse:dataset_cache', this._cacheHandler);
        },

        _teardownSSEListener() {
            if (this._cacheReconnectHandler) {
                window.removeEventListener('sse:connected', this._cacheReconnectHandler);
                this._cacheReconnectHandler = null;
            }
            if (this._scanHandler) {
                window.removeEventListener('sse:dataset_scan', this._scanHandler);
                this._scanHandler = null;
            }
            if (this._queueHandler) {
                window.removeEventListener('sse:dataset_scan_queue', this._queueHandler);
                this._queueHandler = null;
            }
            if (this._cacheHandler) {
                window.removeEventListener('sse:dataset_cache', this._cacheHandler);
                this._cacheHandler = null;
            }
        },

        _setupConfigListener() {
            this._configHandler = () => {
                this.expandedDataset = null;
                this._clearSourceShards();
                this.selectedBucket = null;
                this.thumbnails = [];
                this.closeFileDetail();
                this.conditioningPairs = null;
                this.captionStatus = {};
                this.filterReport = {};
                this.datasetGraph = null;
                this._cacheRequest++;
                this._cacheJobId = null;
                this._cacheStarting = false;
                this.caching = false;
                this.cacheError = '';
                this.cacheCapabilities = { text_embeds: false, vae: false, conditioning_types: [] };
                this._scanRequest++;
                this._scanStarting = false;
                this._scanPlanScope = null;
                this._clearScanProgress();
                this.loadAllSummaries();
                this._checkActiveScan();
                this._checkActiveCacheJob();
                this.loadCacheCapabilities();
            };
            window.addEventListener('environment-changed', this._configHandler);
            document.body.addEventListener('config-changed', this._configHandler);
            this._viewerTabHandler = () => {
                this.loadAllSummaries();
                this._checkActiveScan();
                this._checkActiveCacheJob();
            };
            window.addEventListener('viewer-tab-activated', this._viewerTabHandler);
        },

        _teardownConfigListener() {
            if (this._configHandler) {
                window.removeEventListener('environment-changed', this._configHandler);
                document.body.removeEventListener('config-changed', this._configHandler);
                this._configHandler = null;
            }
            if (this._viewerTabHandler) {
                window.removeEventListener('viewer-tab-activated', this._viewerTabHandler);
                this._viewerTabHandler = null;
            }
        },

        _handleScanEvent(data) {
            if (data.plan_scope !== this._scanPlanScope) return;
            if (this._scanQueueId ? data.queue_id !== this._scanQueueId : data.job_id !== this._scanJobId) return;
            if (this._scanQueueId) {
                if (data.queue_index < this._scanQueueIndex) return;
                this._scanQueueIndex = data.queue_index;
            }
            this._scanStatusVersion++;
            if (['pending', 'running', 'cancelling'].includes(data.status)) {
                this.scanning = true;
                this.scanProgress = {
                    status: data.status,
                    dataset_id: data.dataset_id || this.scanProgress.dataset_id,
                    current: data.current || 0,
                    total: data.total || 0,
                };
                return;
            }
            if (!this._scanQueueId) this._clearScanProgress();
            if (data.status === 'completed') this._refreshDatasetSummary(data.dataset_id);
            if (data.status === 'failed' && data.error) window.showToast?.('Scan failed: ' + data.error, 'error');
        },

        _handleQueueEvent(data) {
            if (data.queue_id !== this._scanQueueId || data.plan_scope !== this._scanPlanScope) return;
            this._scanStatusVersion++;
            if (data.active === false) {
                this._clearScanProgress();
                this.loadAllSummaries();
            }
        },

        _clearScanProgress() {
            this._scanJobId = null;
            this._scanQueueId = null;
            this._scanQueueIndex = -1;
            this.scanning = false;
            this.scanProgress = { dataset_id: '', current: 0, total: 0 };
        },

        _handleCacheEvent(data) {
            if (!data.job_id || data.job_id !== this._cacheJobId) return;
            this._cacheStatusVersion++;
            if (['pending', 'loading_model', 'running', 'cancelling'].includes(data.status)) {
                this.caching = true;
                this.cacheProgress = {
                    dataset_id: data.dataset_id,
                    cache_type: data.cache_type,
                    status: data.status,
                    stage: data.stage || '',
                    current: data.current ?? 0,
                    total: data.total ?? 0,
                };
                return;
            }
            this._cacheJobId = null;
            this.caching = false;
            this.cacheError = data.status === 'failed' ? data.error || 'Cache job failed' : '';
            if (data.status === 'completed') {
                const label = (data.cache_type || '').replace(/_/g, ' ');
                window.showToast?.('Cache job completed: ' + label, 'success');
            } else if (this.cacheError) {
                window.showToast?.('Cache job failed: ' + this.cacheError, 'error');
            }
        },

        async _refreshDatasetSummary(datasetId) {
            const request = this._summaryRequest;
            try {
                const resp = await fetch('/api/datasets/viewer/summary?dataset_id=' + encodeURIComponent(datasetId));
                if (!resp.ok) return;
                const summary = await resp.json();
                if (request !== this._summaryRequest) return;
                const idx = this.datasets.findIndex(d => d.dataset_id === datasetId);
                if (idx >= 0) {
                    this.datasets[idx] = summary;
                }
            } catch (e) {
                // silently fail, user can refresh manually
            }
        },

        // --- Scan operations ---

        async _checkActiveScan() {
            if (this._scanStarting) return;
            const request = this._scanRequest;
            const version = this._scanStatusVersion;
            const jobId = this._scanJobId;
            const queueId = this._scanQueueId;
            const endpoint = queueId ? '/queue-status?queue_id=' + encodeURIComponent(queueId)
                : jobId ? '/status?job_id=' + encodeURIComponent(jobId) : '/active';
            try {
                const resp = await fetch('/api/datasets/scan' + endpoint);
                if (request !== this._scanRequest || version !== this._scanStatusVersion) return;
                if ((queueId || jobId) && resp.status === 404) {
                    this._clearScanProgress();
                    return;
                }
                if (!resp.ok) return;
                const data = await resp.json();
                if (request !== this._scanRequest || version !== this._scanStatusVersion) return;
                if (queueId) {
                    this._handleQueueEvent(data);
                    if (!data.active) return;
                    this.scanning = true;
                    const job = Object.values(data.datasets)[data.current_index];
                    if (job) this._handleScanEvent(job);
                } else if (!jobId) {
                    this._scanPlanScope = data.plan_scope;
                    if (!data.active) return;
                    this._scanJobId = data.job_id;
                    this._scanQueueId = data.queue_id;
                    this._handleScanEvent(data);
                } else {
                    this._handleScanEvent(data);
                }
            } catch (e) { /* ignore */ }
        },

        async scanDataset(datasetId, { forceRescan = false, clearVaeCache = false, clearConditioningCache = false } = {}) {
            await this._startScan('', datasetId, {
                dataset_id: datasetId,
                force_rescan: forceRescan,
                clear_vae_cache: clearVaeCache,
                clear_conditioning_cache: clearConditioningCache,
            });
        },

        showForceScanDialog(datasetId) {
            this.forceScanDatasetId = datasetId;
            this.forceScanClearVae = false;
            this.forceScanClearConditioning = false;
            this.showingForceScanDialog = true;
        },

        closeForceScanDialog() {
            this.showingForceScanDialog = false;
            this.forceScanDatasetId = null;
            this.forceScanClearVae = false;
            this.forceScanClearConditioning = false;
        },

        confirmForceScan() {
            const datasetId = this.forceScanDatasetId;
            const clearVaeCache = this.forceScanClearVae;
            const clearConditioningCache = this.forceScanClearConditioning;
            this.closeForceScanDialog();
            this.scanDataset(datasetId, { forceRescan: true, clearVaeCache, clearConditioningCache });
        },

        async scanAll() {
            await this._startScan('/all', 'all', {});
        },

        async _startScan(endpoint, datasetId, payload) {
            const request = ++this._scanRequest;
            this._scanStarting = true;
            this.scanning = true;
            this.scanProgress = { dataset_id: datasetId, status: 'pending', current: 0, total: 0 };
            try {
                const resp = await fetch('/api/datasets/scan' + endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...payload, plan_scope: this._scanPlanScope }),
                });
                const data = await resp.json();
                if (request !== this._scanRequest) return;
                if (!resp.ok) throw new Error(data.detail || 'Scan request failed');
                this._scanPlanScope = data.plan_scope;
                this._scanJobId = data.job_id || null;
                this._scanQueueId = data.queue_id || null;
            } catch (err) {
                if (request !== this._scanRequest) return;
                this._clearScanProgress();
                window.showToast?.(err.message, 'error');
            } finally {
                if (request === this._scanRequest) {
                    this._scanStarting = false;
                    await this._checkActiveScan();
                }
            }
        },

        async cancelScan() {
            const request = this._scanRequest;
            const params = this._scanQueueId ? { queue_id: this._scanQueueId } : { job_id: this._scanJobId };
            if (!this._scanQueueId && !this._scanJobId) return;
            try {
                const resp = await fetch('/api/datasets/scan/cancel?' + new URLSearchParams(params), { method: 'POST' });
                if (!resp.ok) throw new Error('Cancel request failed');
                if (request === this._scanRequest) await this._checkActiveScan();
            } catch (err) {
                if (request === this._scanRequest) window.showToast?.(err.message, 'error');
            }
        },

        // --- Cache operations ---

        async loadCacheCapabilities() {
            const request = this._cacheRequest;
            try {
                const resp = await fetch('/api/datasets/cache/capabilities');
                if (!resp.ok) return;
                const capabilities = await resp.json();
                if (request === this._cacheRequest) this.cacheCapabilities = capabilities;
            } catch (e) { /* ignore */ }
        },

        async _checkActiveCacheJob() {
            if (this._cacheStarting) return;
            const request = this._cacheRequest;
            const version = this._cacheStatusVersion;
            const jobId = this._cacheJobId;
            const endpoint = jobId ? '/status?job_id=' + encodeURIComponent(jobId) : '/active';
            try {
                const resp = await fetch('/api/datasets/cache' + endpoint);
                if (request !== this._cacheRequest || version !== this._cacheStatusVersion) return;
                if (jobId && resp.status === 404) {
                    this._cacheJobId = null;
                    this.caching = false;
                    return;
                }
                if (!resp.ok) return;
                const data = await resp.json();
                if (request !== this._cacheRequest || version !== this._cacheStatusVersion) return;
                if (jobId || data.active) {
                    this._cacheJobId = data.job_id;
                    this._handleCacheEvent(data);
                }
            } catch (e) { /* ignore */ }
        },

        async startCacheJob(datasetId, cacheType) {
            const request = ++this._cacheRequest;
            this._cacheStarting = true;
            this.caching = true;
            this.cacheError = '';
            this.cacheProgress = {
                dataset_id: datasetId, cache_type: cacheType, stage: 'Starting...', current: 0, total: 0,
            };
            try {
                const resp = await fetch('/api/datasets/cache/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: datasetId,
                        cache_type: cacheType,
                        plan_scope: this.cacheCapabilities.plan_scope,
                    }),
                });
                const data = await resp.json();
                if (request !== this._cacheRequest) return;
                if (!resp.ok) throw new Error(data.detail || 'Cache request failed');
                this._cacheJobId = data.job_id;
            } catch (err) {
                if (request !== this._cacheRequest) return;
                this.cacheError = err.message || 'Cache request failed';
                this.caching = false;
                window.showToast?.(this.cacheError, 'error');
            } finally {
                if (request === this._cacheRequest) {
                    this._cacheStarting = false;
                    await this._checkActiveCacheJob();
                }
            }
        },

        async cancelCacheJob() {
            const jobId = this._cacheJobId;
            if (!jobId) return;
            try {
                const resp = await fetch('/api/datasets/cache/cancel?job_id=' + encodeURIComponent(jobId), { method: 'POST' });
                const data = await resp.json();
                if (jobId !== this._cacheJobId) return;
                if (!resp.ok) throw new Error(data.detail || 'Cancel request failed');
                if (data.cancelled) await this._checkActiveCacheJob();
            } catch (err) {
                if (jobId === this._cacheJobId) window.showToast?.(err.message, 'error');
            }
        },

        // --- Viewer operations ---

        async loadAllSummaries() {
            const request = ++this._summaryRequest;
            this.loading = true;
            try {
                const resp = await fetch('/api/datasets/viewer/summaries');
                if (request !== this._summaryRequest) return;
                if (!resp.ok) {
                    console.error('Failed to load dataset summaries:', resp.status);
                    this.datasets = [];
                    return;
                }
                const summaries = await resp.json();
                if (request !== this._summaryRequest) return;
                this.datasets = summaries;
                const expanded = summaries.find(dataset => dataset.dataset_id === this.expandedDataset);
                if (expanded?.supports_shard_listing && !this.sourceShards && !this.shardLoading) {
                    this.loadSourceShards(expanded.dataset_id);
                }

                // Load supplementary data for cached datasets in parallel
                const fetchPromises = [];
                for (const ds of this.datasets) {
                    if (ds.has_cache) {
                        fetchPromises.push(this._loadCaptionStatus(ds.dataset_id));
                        fetchPromises.push(this._loadFilterReport(ds.dataset_id));
                    }
                }
                fetchPromises.push(this._loadGraph());
                await Promise.allSettled(fetchPromises);
                if (request !== this._summaryRequest) return;

                // Restore pending bucket selection from sessionStorage
                if (this._pendingBucketKey && this.expandedDataset) {
                    const ds = this.datasets.find(d => d.dataset_id === this.expandedDataset);
                    if (ds?.buckets) {
                        const bucket = ds.buckets.find(b => b.key === this._pendingBucketKey);
                        if (bucket) {
                            this.selectedBucket = bucket;
                            this.thumbnailOffset = 0;
                            this.loadBucketThumbnails(this.expandedDataset);
                        }
                    }
                    this._pendingBucketKey = null;
                }
            } catch (err) {
                if (request !== this._summaryRequest) return;
                console.error('Error loading dataset summaries:', err);
                this.datasets = [];
            } finally {
                if (request === this._summaryRequest) this.loading = false;
            }
        },

        async _loadCaptionStatus(datasetId) {
            const request = this._summaryRequest;
            try {
                const resp = await fetch('/api/datasets/viewer/caption-status?dataset_id=' + encodeURIComponent(datasetId));
                if (resp.ok) {
                    const status = await resp.json();
                    if (request === this._summaryRequest) this.captionStatus[datasetId] = status;
                }
            } catch (e) { /* silently fail */ }
        },

        async _loadFilterReport(datasetId) {
            const request = this._summaryRequest;
            try {
                const resp = await fetch('/api/datasets/viewer/filtered?dataset_id=' + encodeURIComponent(datasetId) + '&limit=0');
                if (resp.ok) {
                    const report = await resp.json();
                    if (request === this._summaryRequest) this.filterReport[datasetId] = report;
                }
            } catch (e) { /* silently fail */ }
        },

        async _loadGraph() {
            const request = this._summaryRequest;
            try {
                const resp = await fetch('/api/datasets/viewer/graph');
                if (resp.ok) {
                    const graph = await resp.json();
                    if (request === this._summaryRequest) this.datasetGraph = graph;
                }
            } catch (e) { /* silently fail */ }
        },

        async selectConditioningPair(sourceId, conditioningId) {
            this.conditioningPairSelection = {
                sourceId,
                conditioningId,
                sourceRevision: this.datasets.find(d => d.dataset_id === sourceId)?.view_revision,
                conditioningRevision: this.datasets.find(d => d.dataset_id === conditioningId)?.view_revision,
            };
            this.pairsOffset = 0;
            await this.loadConditioningPairs();
        },

        async loadConditioningPairs() {
            const selection = this.conditioningPairSelection;
            if (!selection) return;
            const request = ++this._pairsRequest;
            this.loadingPairs = true;
            this.conditioningPairs = null;
            try {
                const params = new URLSearchParams({
                    source_id: selection.sourceId,
                    conditioning_id: selection.conditioningId,
                    limit: String(this.pairsLimit),
                    offset: String(this.pairsOffset),
                });
                if (selection.sourceRevision) params.set('source_view_revision', selection.sourceRevision);
                if (selection.conditioningRevision) params.set('conditioning_view_revision', selection.conditioningRevision);
                const resp = await fetch('/api/datasets/viewer/conditioning-pairs?' + params);
                if (resp.status === 409) {
                    if (request !== this._pairsRequest) return;
                    this._clearConditioningPairs();
                    const clearedRequest = this._pairsRequest;
                    await Promise.all([
                        this._refreshDatasetSummary(selection.sourceId),
                        this._refreshDatasetSummary(selection.conditioningId),
                    ]);
                    if (clearedRequest === this._pairsRequest) {
                        window.showToast?.('The dataset changed. Select the conditioning link again.', 'info');
                    }
                } else if (resp.ok) {
                    const page = await resp.json();
                    if (request === this._pairsRequest) {
                        this.conditioningPairs = page;
                        selection.sourceRevision = page.source_view_revision ?? selection.sourceRevision;
                        selection.conditioningRevision = page.conditioning_view_revision ?? selection.conditioningRevision;
                    }
                }
            } catch (e) {
                if (request === this._pairsRequest) console.error('Error loading conditioning pairs:', e);
            } finally {
                if (request === this._pairsRequest) this.loadingPairs = false;
            }
        },

        async pairsNextPage() {
            if (!this.conditioningPairs) return;
            if (this.pairsOffset + this.pairsLimit >= (this.conditioningPairs.source_total ?? this.conditioningPairs.total_pairs)) return;
            this.pairsOffset += this.pairsLimit;
            await this.loadConditioningPairs();
        },

        async pairsPrevPage() {
            if (this.pairsOffset <= 0) return;
            this.pairsOffset = Math.max(0, this.pairsOffset - this.pairsLimit);
            await this.loadConditioningPairs();
        },

        _clearConditioningPairs() {
            this._pairsRequest++;
            this.loadingPairs = false;
            this.conditioningPairs = null;
            this.conditioningPairSelection = null;
            this.pairsOffset = 0;
        },

        toggleDataset(datasetId) {
            this.expandedDataset = this.expandedDataset === datasetId ? null : datasetId;
            this._clearSourceShards();
            this._clearBucketSelection();
            this.closeFileDetail();
            this._clearConditioningPairs();
            this._saveNavState();
            if (this.expandedDataset && this.datasets.find(ds => ds.dataset_id === datasetId)?.supports_shard_listing) {
                this.loadSourceShards(datasetId);
            }
        },

        _clearSourceShards() {
            this._shardRequest++;
            this.sourceShards = null;
            this.shardSearch = '';
            this.shardLoading = false;
            this.shardError = '';
        },

        async loadSourceShards(datasetId, offset = 0, refresh = false) {
            if (this.expandedDataset !== datasetId) return;
            const request = ++this._shardRequest;
            const query = this.shardSearch.trim();
            this.shardLoading = true;
            this.shardError = '';
            const current = () => request === this._shardRequest && query === this.shardSearch.trim();
            try {
                const params = new URLSearchParams({dataset_id: datasetId, query, offset, refresh});
                const response = await fetch('/api/datasets/viewer/shards?' + params);
                const result = await response.json();
                if (!current()) return;
                if (!response.ok) throw new Error(result.detail || 'Unable to list source shards.');
                this.sourceShards = result;
            } catch (error) {
                if (!current()) return;
                this.sourceShards = null;
                this.shardError = error.message;
            } finally {
                if (request === this._shardRequest) this.shardLoading = false;
            }
        },

        _clearBucketSelection() {
            this._thumbnailRequest++;
            this.loadingThumbnails = false;
            this.selectedBucket = null;
            this.thumbnails = [];
            this.thumbnailOffset = 0;
        },

        async selectBucket(datasetId, bucket) {
            if (this.selectedBucket?.key === bucket.key && this.expandedDataset === datasetId) {
                this._clearBucketSelection();
                this._saveNavState();
                return;
            }
            this.thumbnails = [];
            this.selectedBucket = bucket;
            this.thumbnailOffset = 0;
            this._saveNavState();
            await this.loadBucketThumbnails(datasetId);
        },

        async loadBucketThumbnails(datasetId) {
            if (!this.selectedBucket) return;
            const request = ++this._thumbnailRequest;
            const revision = this.datasets.find(ds => ds.dataset_id === datasetId)?.view_revision;
            const isCurrent = () => request === this._thumbnailRequest;
            const isCurrentRevision = () =>
                revision === this.datasets.find(ds => ds.dataset_id === datasetId)?.view_revision;
            this.loadingThumbnails = true;
            try {
                const params = new URLSearchParams({
                    dataset_id: datasetId,
                    bucket_key: this.selectedBucket.key,
                    limit: String(this.thumbnailLimit),
                    offset: String(this.thumbnailOffset),
                });
                if (revision) params.set('view_revision', revision);
                const resp = await fetch('/api/datasets/viewer/thumbnails?' + params);
                if (!isCurrent()) return;
                if (resp.status === 409 || !isCurrentRevision()) {
                    await this._refreshStaleView(datasetId);
                    return;
                }
                if (!resp.ok) {
                    console.error('Failed to load thumbnails:', resp.status);
                    this.thumbnails = [];
                    return;
                }
                const thumbnails = await resp.json();
                if (isCurrent() && isCurrentRevision()) this.thumbnails = thumbnails;
            } catch (err) {
                if (isCurrent()) {
                    console.error('Error loading thumbnails:', err);
                    this.thumbnails = [];
                }
            } finally {
                if (isCurrent()) this.loadingThumbnails = false;
            }
        },

        async _refreshStaleView(datasetId) {
            if (this.expandedDataset === datasetId) {
                this._clearBucketSelection();
                this._saveNavState();
            }
            await this._refreshDatasetSummary(datasetId);
            if (this.expandedDataset === datasetId && !this.selectedBucket) {
                window.showToast?.('The dataset changed. Select a bucket to browse its current contents.', 'info');
            }
        },

        async nextPage(datasetId) {
            if (!this.selectedBucket) return;
            if (this.thumbnailOffset + this.thumbnailLimit >= this.selectedBucket.file_count) return;
            this.thumbnailOffset += this.thumbnailLimit;
            await this.loadBucketThumbnails(datasetId);
        },

        async prevPage(datasetId) {
            if (this.thumbnailOffset <= 0) return;
            this.thumbnailOffset = Math.max(0, this.thumbnailOffset - this.thumbnailLimit);
            await this.loadBucketThumbnails(datasetId);
        },

        async showFileDetail(datasetId, filePath, viewRevision) {
            const request = ++this._fileRequest;
            this._destroyBboxCanvas();
            this.loadingFileDetail = true;
            this.loadingPreview = true;
            this.selectedFile = { path: filePath };
            this.selectedFileDatasetId = datasetId;
            this.savingCrop = false;
            this.rebuildingMetadata = false;
            this.deletingVaeCache = false;
            this.previewOriginal = null;
            this.previewIntermediary = null;
            this.previewCropped = null;
            this.cropOverlay = null;
            this.conditioningMatch = null;
            this.loadingConditioningMatch = false;

            const params = new URLSearchParams({
                dataset_id: datasetId,
                file_path: filePath,
            });
            const ds = this.datasets.find(d => d.dataset_id === datasetId);
            const expectedRevision = viewRevision ?? ds?.view_revision;
            if (expectedRevision) params.set('view_revision', expectedRevision);
            let staleView = false;
            const readDetail = response => {
                staleView ||= response.status === 409;
                return response.ok ? response.json() : null;
            };

            // Load metadata and preview in parallel
            const fetches = [
                fetch('/api/datasets/viewer/file-metadata?' + params).then(readDetail),
                fetch('/api/datasets/viewer/preview?' + params).then(readDetail),
            ];

            // Also fetch conditioning match if this dataset has conditioning
            const hasConditioning = (ds?.conditioning_data?.length > 0) ||
                                    (ds?.conditioning_generators?.length > 0);
            if (hasConditioning) {
                this.loadingConditioningMatch = true;
                fetches.push(
                    fetch('/api/datasets/viewer/conditioning-match?' + params).then(readDetail)
                );
            }

            const results = await Promise.allSettled(fetches);
            if (request !== this._fileRequest) return;
            const currentRevision = this.datasets.find(d => d.dataset_id === datasetId)?.view_revision;
            if (staleView || currentRevision !== ds?.view_revision) {
                this.closeFileDetail();
                await this._refreshStaleView(datasetId);
                return;
            }

            if (results[0].status === 'fulfilled' && results[0].value) {
                this.selectedFile = results[0].value;
            } else {
                this.selectedFile = { path: filePath, found: false };
            }
            this.loadingFileDetail = false;

            if (results[1].status === 'fulfilled' && results[1].value) {
                const pv = results[1].value;
                this.previewOriginal = pv.original || null;
                this.previewIntermediary = pv.intermediary || null;
                this.previewCropped = pv.cropped || null;
            }
            this.loadingPreview = false;

            if (hasConditioning && results[2]?.status === 'fulfilled') {
                this.conditioningMatch = results[2].value;
            }
            this.loadingConditioningMatch = false;

            // Initialize bbox canvas if entities exist
            if (this.selectedFile?.bbox_entities?.length > 0) {
                this.$nextTick(() => {
                    if (request === this._fileRequest) this._initBboxCanvas();
                });
            }
        },

        /**
         * Called when the intermediary preview <img> loads. Computes the
         * crop overlay in display coordinates. Since the preview IS the
         * intermediary-resized image, crop_coordinates map directly via
         * the display/natural scale factor.
         */
        onIntermediaryLoad(imgEl) {
            this.cropOverlay = null;
            this._intermediaryImgEl = imgEl;
            this.cropModified = false;
            const f = this.selectedFile;
            if (!f || !f.crop_coordinates || !f.target_size || !f.intermediary_size) return;
            this._computeCropOverlay();
        },

        /**
         * Compute crop overlay as percentages of the intermediary dimensions.
         * crop_coordinates and target_size are in intermediary coordinate space,
         * so we use intermediary_size as the reference (not the preview image's
         * naturalWidth/Height, which may differ due to thumbnail downscaling).
         * Percentage CSS positioning maps correctly because the preview image
         * shares the same aspect ratio as the intermediary.
         */
        _computeCropOverlay() {
            const f = this.selectedFile;
            if (!f || !f.crop_coordinates || !f.target_size || !f.intermediary_size) return;

            const intW = f.intermediary_size[0];
            const intH = f.intermediary_size[1];
            if (!intW || !intH) return;

            // crop_coordinates are (top, left) matching cropping.py convention
            const top = Math.max(0, Math.min(f.crop_coordinates[0], intH));
            const left = Math.max(0, Math.min(f.crop_coordinates[1], intW));
            const tw = Math.min(f.target_size[0], intW - left);
            const th = Math.min(f.target_size[1], intH - top);

            this.cropOverlay = {
                leftPct: left / intW * 100,
                topPct: top / intH * 100,
                widthPct: tw / intW * 100,
                heightPct: th / intH * 100,
                natLeft: left,
                natTop: top,
                natW: tw,
                natH: th,
            };
        },

        // --- Crop dragging ---

        onCropMouseDown(e) {
            if (!this.selectedFile?.metadata_editable || !this.cropOverlay || !this._intermediaryImgEl) return;
            e.preventDefault();
            this.cropDragging = true;
            this.cropDragStart = {
                mouseX: e.clientX,
                mouseY: e.clientY,
                natLeft: this.cropOverlay.natLeft,
                natTop: this.cropOverlay.natTop,
            };

            const onMove = (ev) => this._onCropDrag(ev);
            const onUp = () => {
                this.cropDragging = false;
                this.cropDragStart = null;
                window.removeEventListener('mousemove', onMove);
                window.removeEventListener('mouseup', onUp);
            };
            window.addEventListener('mousemove', onMove);
            window.addEventListener('mouseup', onUp);
        },

        _onCropDrag(e) {
            if (!this.cropDragging || !this.cropDragStart || !this.cropOverlay) return;
            const f = this.selectedFile;
            const img = this._intermediaryImgEl;
            if (!f || !f.intermediary_size || !img || !img.clientWidth || !img.clientHeight) return;

            const intW = f.intermediary_size[0];
            const intH = f.intermediary_size[1];

            // Convert pixel mouse delta to intermediary-coordinate delta
            const pxToIntX = intW / img.clientWidth;
            const pxToIntY = intH / img.clientHeight;
            const dx = (e.clientX - this.cropDragStart.mouseX) * pxToIntX;
            const dy = (e.clientY - this.cropDragStart.mouseY) * pxToIntY;

            const maxLeft = intW - this.cropOverlay.natW;
            const maxTop = intH - this.cropOverlay.natH;
            const newLeft = Math.max(0, Math.min(this.cropDragStart.natLeft + dx, maxLeft));
            const newTop = Math.max(0, Math.min(this.cropDragStart.natTop + dy, maxTop));

            this.cropOverlay = {
                ...this.cropOverlay,
                natLeft: newLeft,
                natTop: newTop,
                leftPct: newLeft / intW * 100,
                topPct: newTop / intH * 100,
            };

            // Write back as (top, left) matching cropping.py convention
            this.selectedFile.crop_coordinates = [Math.round(newTop), Math.round(newLeft)];
            this.cropModified = true;
        },

        closeFileDetail() {
            this._fileRequest++;
            this._destroyBboxCanvas();
            this.loadingFileDetail = false;
            this.loadingPreview = false;
            this.selectedFile = null;
            this.selectedFileDatasetId = null;
            this.previewOriginal = null;
            this.previewIntermediary = null;
            this.previewCropped = null;
            this.cropOverlay = null;
            this.cropModified = false;
            this.conditioningMatch = null;
            this.loadingConditioningMatch = false;
        },

        _initBboxCanvas() {
            this._destroyBboxCanvas();
            const container = this.$refs.bboxCanvasContainer;
            if (!container || typeof BboxCanvas === 'undefined') return;

            const f = this.selectedFile;
            const w = f.target_size?.[0] || f.intermediary_size?.[0] || 1024;
            const h = f.target_size?.[1] || f.intermediary_size?.[1] || 1024;

            this.bboxCanvas = new BboxCanvas(container, {
                width: w,
                height: h,
                readOnly: !f.metadata_editable,
                onChange: () => {
                    this.bboxModified = true;
                },
            });

            if (f.bbox_entities) {
                this.bboxCanvas.setBoxes(f.bbox_entities);
            }

            // Use the intermediary preview as background if available
            if (this.previewIntermediary) {
                this.bboxCanvas.setBackgroundImage(this.previewIntermediary);
            }
            this.bboxModified = false;
        },

        _destroyBboxCanvas() {
            if (this.bboxCanvas) {
                this.bboxCanvas.destroy();
                this.bboxCanvas = null;
            }
            this.bboxModified = false;
            this.savingBbox = false;
        },

        async _writeSelectedFile(action, method, fields = {}) {
            const request = this._fileRequest;
            const datasetId = this.selectedFileDatasetId;
            const file = this.selectedFile;
            if (!file?.metadata_editable) return null;
            const response = await fetch('/api/datasets/viewer/' + action, {
                method,
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_id: datasetId,
                    file_path: file.path,
                    view_revision: file.view_revision,
                    metadata_revision: file.metadata_revision,
                    ...fields,
                }),
            });
            if (request !== this._fileRequest) return null;
            if (response.status === 409) {
                this.closeFileDetail();
                await this._refreshStaleView(datasetId);
                return null;
            }
            const data = await response.json().catch(() => null);
            if (request !== this._fileRequest) return null;
            if (!response.ok) throw new Error(data?.detail || 'File action failed');
            if (data?.metadata_revision) file.metadata_revision = data.metadata_revision;
            return data;
        },

        async saveBboxEntities() {
            if (!this.selectedFile || !this.bboxCanvas) return;
            const request = this._fileRequest;
            this.savingBbox = true;
            try {
                const boxes = this.bboxCanvas.getBoxes();
                const entities = boxes.length > 0 ? boxes : null;
                if (!await this._writeSelectedFile('bbox-entities', 'PATCH', { bbox_entities: entities })) return;
                this.selectedFile.bbox_entities = entities;
                this.bboxModified = false;
                window.showToast?.('Bounding boxes saved', 'success');
            } catch (err) {
                if (request === this._fileRequest) window.showToast?.(err.message, 'error');
            } finally {
                if (request === this._fileRequest) this.savingBbox = false;
            }
        },

        async saveCropCoordinates() {
            const coordinates = this.selectedFile?.crop_coordinates;
            if (!coordinates) return;
            const request = this._fileRequest;
            this.savingCrop = true;
            try {
                const data = await this._writeSelectedFile('crop-coordinates', 'PATCH', { crop_coordinates: coordinates });
                if (!data) return;
                if (data.cropped) this.previewCropped = data.cropped;
                this.cropModified = false;
                window.showToast?.('Crop coordinates saved', 'success');
            } catch (err) {
                if (request === this._fileRequest) window.showToast?.(err.message, 'error');
            } finally {
                if (request === this._fileRequest) this.savingCrop = false;
            }
        },

        // --- Single-file actions ---

        rebuildingMetadata: false,
        deletingVaeCache: false,

        async rebuildFileMetadata() {
            const file = this.selectedFile;
            if (!file) return;
            const datasetId = this.selectedFileDatasetId;
            const request = this._fileRequest;
            this.rebuildingMetadata = true;
            try {
                const data = await this._writeSelectedFile('rebuild-metadata', 'POST');
                if (!data) return;
                window.showToast?.('Metadata rebuilt successfully', 'success');
                await this.showFileDetail(datasetId, file.path, data.view_revision ?? file.view_revision);
            } catch (err) {
                if (request === this._fileRequest) window.showToast?.(err.message, 'error');
            } finally {
                if (request === this._fileRequest) this.rebuildingMetadata = false;
            }
        },

        async deleteVaeCacheFile() {
            if (!this.selectedFile) return;
            const request = this._fileRequest;
            this.deletingVaeCache = true;
            try {
                const data = await this._writeSelectedFile('delete-vae-cache', 'POST');
                if (!data) return;
                if (data.deleted) {
                    window.showToast?.('VAE cache file deleted', 'success');
                } else {
                    window.showToast?.('VAE cache: ' + (data.error || 'File not found'), 'warning');
                }
            } catch (err) {
                if (request === this._fileRequest) window.showToast?.(err.message, 'error');
            } finally {
                if (request === this._fileRequest) this.deletingVaeCache = false;
            }
        },
    };
};
