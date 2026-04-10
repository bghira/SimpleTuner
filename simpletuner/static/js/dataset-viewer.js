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
        expandedDataset: null,
        selectedBucket: null,
        thumbnails: [],
        loadingThumbnails: false,
        thumbnailOffset: 0,
        thumbnailLimit: 24,
        selectedFile: null,
        loadingFileDetail: false,
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

        // Stage 3-5 state
        captionStatus: {},
        filterReport: {},
        datasetGraph: null,

        // Conditioning pair viewer state
        conditioningPairs: null,
        loadingPairs: false,
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
            window.addEventListener('sse:dataset_scan', this._scanHandler);
            window.addEventListener('sse:dataset_scan_queue', this._queueHandler);
            window.addEventListener('sse:dataset_cache', this._cacheHandler);
        },

        _teardownSSEListener() {
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
                this.selectedBucket = null;
                this.thumbnails = [];
                this.selectedFile = null;
                this.conditioningPairs = null;
                this.captionStatus = {};
                this.filterReport = {};
                this.datasetGraph = null;
                this._destroyBboxCanvas();
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
            if (data.status === 'running') {
                this.scanning = true;
                this.scanProgress = {
                    dataset_id: data.dataset_id || this.scanProgress.dataset_id,
                    current: data.current || 0,
                    total: data.total || 0,
                };
            } else if (data.status === 'completed') {
                this.scanning = false;
                this.scanProgress = { dataset_id: '', current: 0, total: 0 };
                this._refreshDatasetSummary(data.dataset_id);
            } else if (data.status === 'failed' || data.status === 'cancelled') {
                this.scanning = false;
                this.scanProgress = { dataset_id: '', current: 0, total: 0 };
                if (data.error) {
                    console.error('Scan failed for', data.dataset_id, ':', data.error);
                    if (window.showToast) {
                        window.showToast('Scan failed: ' + data.error, 'error');
                    }
                }
            }
        },

        _handleQueueEvent(data) {
            if (data.completed >= data.total) {
                this.scanning = false;
                this.scanProgress = { dataset_id: '', current: 0, total: 0 };
                this.loadAllSummaries();
            }
        },

        _handleCacheEvent(data) {
            if (data.status === 'loading_model' || data.status === 'running') {
                this.caching = true;
                this.cacheError = '';
                this.cacheProgress = {
                    dataset_id: data.dataset_id || this.cacheProgress.dataset_id,
                    cache_type: data.cache_type || this.cacheProgress.cache_type,
                    stage: data.stage || '',
                    current: data.current || 0,
                    total: data.total || 0,
                };
            } else if (data.status === 'completed') {
                this.caching = false;
                this.cacheError = '';
                this.cacheProgress = { dataset_id: '', cache_type: '', stage: '', current: 0, total: 0 };
                if (window.showToast) {
                    const label = (data.cache_type || '').replace(/_/g, ' ');
                    window.showToast('Cache job completed: ' + label, 'success');
                }
            } else if (data.status === 'failed' || data.status === 'cancelled') {
                this.caching = false;
                this.cacheProgress = { dataset_id: '', cache_type: '', stage: '', current: 0, total: 0 };
                if (data.error) {
                    this.cacheError = data.error;
                    if (window.showToast) {
                        window.showToast('Cache job failed: ' + data.error, 'error');
                    }
                }
            }
        },

        async _refreshDatasetSummary(datasetId) {
            try {
                const resp = await fetch('/api/datasets/viewer/summary?dataset_id=' + encodeURIComponent(datasetId));
                if (!resp.ok) return;
                const summary = await resp.json();
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
            try {
                const resp = await fetch('/api/datasets/scan/active');
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.active) {
                    this.scanning = true;
                    this.scanProgress = {
                        dataset_id: data.dataset_id || '',
                        current: data.current || 0,
                        total: data.total || 0,
                    };
                }
            } catch (e) { /* ignore */ }
        },

        async scanDataset(datasetId, { forceRescan = false, clearVaeCache = false, clearConditioningCache = false } = {}) {
            this.scanning = true;
            this.scanProgress = { dataset_id: datasetId, current: 0, total: 0 };
            try {
                const resp = await fetch('/api/datasets/scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: datasetId,
                        force_rescan: forceRescan,
                        clear_vae_cache: clearVaeCache,
                        clear_conditioning_cache: clearConditioningCache,
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Scan request failed');
                }
                // Progress updates arrive via SSE
            } catch (err) {
                console.error('Error starting scan:', err);
                this.scanning = false;
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            }
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
            this.scanning = true;
            this.scanProgress = { dataset_id: 'all', current: 0, total: 0 };
            try {
                const resp = await fetch('/api/datasets/scan/all', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({}),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Scan all request failed');
                }
            } catch (err) {
                console.error('Error starting scan all:', err);
                this.scanning = false;
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            }
        },

        async cancelScan() {
            try {
                await fetch('/api/datasets/scan/cancel', { method: 'POST' });
                this.scanning = false;
                this.scanProgress = { dataset_id: '', current: 0, total: 0 };
            } catch (err) {
                console.error('Error cancelling scan:', err);
            }
        },

        // --- Cache operations ---

        async loadCacheCapabilities() {
            try {
                const resp = await fetch('/api/datasets/cache/capabilities');
                if (!resp.ok) return;
                this.cacheCapabilities = await resp.json();
            } catch (e) { /* ignore */ }
        },

        async _checkActiveCacheJob() {
            try {
                const resp = await fetch('/api/datasets/cache/active');
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.active) {
                    this.caching = true;
                    this.cacheProgress = {
                        dataset_id: data.dataset_id || '',
                        cache_type: data.cache_type || '',
                        stage: data.stage || '',
                        current: data.current || 0,
                        total: data.total || 0,
                    };
                }
            } catch (e) { /* ignore */ }
        },

        async startCacheJob(datasetId, cacheType) {
            this.caching = true;
            this.cacheError = '';
            this.cacheProgress = { dataset_id: datasetId, cache_type: cacheType, stage: 'Starting...', current: 0, total: 0 };
            try {
                const resp = await fetch('/api/datasets/cache/start', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ dataset_id: datasetId, cache_type: cacheType }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Cache request failed');
                }
            } catch (err) {
                console.error('Error starting cache job:', err);
                this.caching = false;
                this.cacheProgress = { dataset_id: '', cache_type: '', stage: '', current: 0, total: 0 };
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            }
        },

        async cancelCacheJob() {
            try {
                await fetch('/api/datasets/cache/cancel', { method: 'POST' });
                this.caching = false;
                this.cacheProgress = { dataset_id: '', cache_type: '', stage: '', current: 0, total: 0 };
            } catch (err) {
                console.error('Error cancelling cache job:', err);
            }
        },

        // --- Viewer operations ---

        async loadAllSummaries() {
            this.loading = true;
            try {
                const resp = await fetch('/api/datasets/viewer/summaries');
                if (!resp.ok) {
                    console.error('Failed to load dataset summaries:', resp.status);
                    this.datasets = [];
                    return;
                }
                this.datasets = await resp.json();

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
                console.error('Error loading dataset summaries:', err);
                this.datasets = [];
            } finally {
                this.loading = false;
            }
        },

        async _loadCaptionStatus(datasetId) {
            try {
                const resp = await fetch('/api/datasets/viewer/caption-status?dataset_id=' + encodeURIComponent(datasetId));
                if (resp.ok) {
                    this.captionStatus[datasetId] = await resp.json();
                }
            } catch (e) { /* silently fail */ }
        },

        async _loadFilterReport(datasetId) {
            try {
                const resp = await fetch('/api/datasets/viewer/filtered?dataset_id=' + encodeURIComponent(datasetId) + '&limit=0');
                if (resp.ok) {
                    this.filterReport[datasetId] = await resp.json();
                }
            } catch (e) { /* silently fail */ }
        },

        async _loadGraph() {
            try {
                const resp = await fetch('/api/datasets/viewer/graph');
                if (resp.ok) {
                    this.datasetGraph = await resp.json();
                }
            } catch (e) { /* silently fail */ }
        },

        async loadConditioningPairs(sourceId, conditioningId) {
            this.loadingPairs = true;
            this.conditioningPairs = null;
            try {
                const params = new URLSearchParams({
                    source_id: sourceId,
                    conditioning_id: conditioningId,
                    limit: String(this.pairsLimit),
                    offset: String(this.pairsOffset),
                });
                const resp = await fetch('/api/datasets/viewer/conditioning-pairs?' + params);
                if (resp.ok) {
                    this.conditioningPairs = await resp.json();
                }
            } catch (e) {
                console.error('Error loading conditioning pairs:', e);
            } finally {
                this.loadingPairs = false;
            }
        },

        async pairsNextPage(sourceId, conditioningId) {
            if (!this.conditioningPairs) return;
            if (this.pairsOffset + this.pairsLimit >= this.conditioningPairs.total_pairs) return;
            this.pairsOffset += this.pairsLimit;
            await this.loadConditioningPairs(sourceId, conditioningId);
        },

        async pairsPrevPage(sourceId, conditioningId) {
            if (this.pairsOffset <= 0) return;
            this.pairsOffset = Math.max(0, this.pairsOffset - this.pairsLimit);
            await this.loadConditioningPairs(sourceId, conditioningId);
        },

        toggleDataset(datasetId) {
            this.expandedDataset = this.expandedDataset === datasetId ? null : datasetId;
            this.selectedBucket = null;
            this.thumbnails = [];
            this.conditioningPairs = null;
            this.pairsOffset = 0;
            this._saveNavState();
        },

        async selectBucket(datasetId, bucket) {
            if (this.selectedBucket?.key === bucket.key && this.expandedDataset === datasetId) {
                this.selectedBucket = null;
                this.thumbnails = [];
                this._saveNavState();
                return;
            }
            this.selectedBucket = bucket;
            this.thumbnailOffset = 0;
            this._saveNavState();
            await this.loadBucketThumbnails(datasetId);
        },

        async loadBucketThumbnails(datasetId) {
            if (!this.selectedBucket) return;
            this.loadingThumbnails = true;
            try {
                const params = new URLSearchParams({
                    dataset_id: datasetId,
                    bucket_key: this.selectedBucket.key,
                    limit: String(this.thumbnailLimit),
                    offset: String(this.thumbnailOffset),
                });
                const resp = await fetch('/api/datasets/viewer/thumbnails?' + params);
                if (!resp.ok) {
                    console.error('Failed to load thumbnails:', resp.status);
                    this.thumbnails = [];
                    return;
                }
                this.thumbnails = await resp.json();
            } catch (err) {
                console.error('Error loading thumbnails:', err);
                this.thumbnails = [];
            } finally {
                this.loadingThumbnails = false;
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

        async showFileDetail(datasetId, filePath) {
            this._destroyBboxCanvas();
            this.loadingFileDetail = true;
            this.loadingPreview = true;
            this.selectedFile = { path: filePath };
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

            // Load metadata and preview in parallel
            const fetches = [
                fetch('/api/datasets/viewer/file-metadata?' + params).then(r => r.ok ? r.json() : null),
                fetch('/api/datasets/viewer/preview?' + params).then(r => r.ok ? r.json() : null),
            ];

            // Also fetch conditioning match if this dataset has conditioning
            const ds = this.datasets.find(d => d.dataset_id === datasetId);
            const hasConditioning = (ds?.conditioning_data?.length > 0) ||
                                    (ds?.conditioning_generators?.length > 0);
            if (hasConditioning) {
                this.loadingConditioningMatch = true;
                fetches.push(
                    fetch('/api/datasets/viewer/conditioning-match?' + params)
                        .then(r => r.ok ? r.json() : null)
                );
            }

            const results = await Promise.allSettled(fetches);

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
                this.$nextTick(() => this._initBboxCanvas());
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
            if (!this.cropOverlay || !this._intermediaryImgEl) return;
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
            this._destroyBboxCanvas();
            this.selectedFile = null;
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

        async saveBboxEntities() {
            const f = this.selectedFile;
            if (!f || !this.bboxCanvas || !this.expandedDataset) return;
            this.savingBbox = true;
            try {
                const entities = this.bboxCanvas.getBoxes();
                const resp = await fetch('/api/datasets/viewer/bbox-entities', {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: this.expandedDataset,
                        file_path: f.path,
                        bbox_entities: entities.length > 0 ? entities : null,
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Save failed');
                }
                this.selectedFile.bbox_entities = entities.length > 0 ? entities : null;
                this.bboxModified = false;
                if (window.showToast) {
                    window.showToast('Bounding boxes saved', 'success');
                }
            } catch (err) {
                console.error('Error saving bbox entities:', err);
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            } finally {
                this.savingBbox = false;
            }
        },

        async saveCropCoordinates() {
            const f = this.selectedFile;
            if (!f || !f.crop_coordinates || !this.expandedDataset) return;
            this.savingCrop = true;
            try {
                const resp = await fetch('/api/datasets/viewer/crop-coordinates', {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: this.expandedDataset,
                        file_path: f.path,
                        crop_coordinates: f.crop_coordinates,
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Save failed');
                }
                const data = await resp.json();
                if (data.cropped) {
                    this.previewCropped = data.cropped;
                }
                this.cropModified = false;
                if (window.showToast) {
                    window.showToast('Crop coordinates saved', 'success');
                }
            } catch (err) {
                console.error('Error saving crop coordinates:', err);
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            } finally {
                this.savingCrop = false;
            }
        },

        // --- Single-file actions ---

        rebuildingMetadata: false,
        deletingVaeCache: false,

        async rebuildFileMetadata() {
            const f = this.selectedFile;
            if (!f || !this.expandedDataset) return;
            this.rebuildingMetadata = true;
            try {
                const resp = await fetch('/api/datasets/viewer/rebuild-metadata', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: this.expandedDataset,
                        file_path: f.path,
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Rebuild failed');
                }
                const data = await resp.json();
                if (data.metadata) {
                    // Update the selected file with the new metadata
                    const m = data.metadata;
                    if (m.target_size) f.target_size = m.target_size;
                    if (m.intermediary_size) f.intermediary_size = m.intermediary_size;
                    if (m.crop_coordinates) f.crop_coordinates = m.crop_coordinates;
                    if (m.aspect_ratio != null) f.aspect_ratio = m.aspect_ratio;
                    // Recompute overlay and re-fetch previews
                    this._computeCropOverlay();
                    this.cropModified = false;
                    // Refresh the preview images with updated metadata
                    const params = new URLSearchParams({
                        dataset_id: this.expandedDataset,
                        file_path: f.path,
                    });
                    const pvResp = await fetch('/api/datasets/viewer/preview?' + params);
                    if (pvResp.ok) {
                        const pv = await pvResp.json();
                        this.previewOriginal = pv.original || null;
                        this.previewIntermediary = pv.intermediary || null;
                        this.previewCropped = pv.cropped || null;
                    }
                }
                if (window.showToast) {
                    window.showToast('Metadata rebuilt successfully', 'success');
                }
            } catch (err) {
                console.error('Error rebuilding metadata:', err);
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            } finally {
                this.rebuildingMetadata = false;
            }
        },

        async deleteVaeCacheFile() {
            const f = this.selectedFile;
            if (!f || !this.expandedDataset) return;
            this.deletingVaeCache = true;
            try {
                const resp = await fetch('/api/datasets/viewer/delete-vae-cache', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        dataset_id: this.expandedDataset,
                        file_path: f.path,
                    }),
                });
                if (!resp.ok) {
                    const err = await resp.json().catch(() => ({ detail: 'Unknown error' }));
                    throw new Error(err.detail || 'Delete failed');
                }
                const data = await resp.json();
                if (data.deleted) {
                    if (window.showToast) {
                        window.showToast('VAE cache file deleted', 'success');
                    }
                } else {
                    const reason = data.error || 'File not found';
                    if (window.showToast) {
                        window.showToast('VAE cache: ' + reason, 'warning');
                    }
                }
            } catch (err) {
                console.error('Error deleting VAE cache file:', err);
                if (window.showToast) {
                    window.showToast(err.message, 'error');
                }
            } finally {
                this.deletingVaeCache = false;
            }
        },
    };
};
