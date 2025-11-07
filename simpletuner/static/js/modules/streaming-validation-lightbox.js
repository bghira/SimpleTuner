/**
 * Streaming Validation Lightbox Module
 * Displays intermediary validation samples as they are streamed during training
 * with instant content replacement and auto-timeout functionality.
 */

class StreamingValidationLightbox {
    constructor() {
        this.lightboxElement = null;
        this.isVisible = false;
        this.isMinimized = false;
        this.lastSampleTime = null;
        this.autoCloseTimeout = null;
        this.autoCloseDelay = 30000; // 30 seconds
        this.isExpanded = false;
        this.currentImageData = null;
        this.currentVideoData = null;
        this.hasNewUpdate = false;

        this.init();
    }

    init() {
        console.log('[StreamingValidationLightbox] Initializing...');
        this.createLightbox();
        this.setupEventListeners();
        this.registerSSEListener();
        console.log('[StreamingValidationLightbox] Initialization complete');
    }

    createLightbox() {
        // Check if lightbox already exists
        if (document.getElementById('streaming-validation-lightbox')) {
            this.lightboxElement = document.getElementById('streaming-validation-lightbox');
            return;
        }

        // Create lightbox HTML structure
        const lightboxHTML = `
            <div id="streaming-validation-lightbox" class="streaming-validation-lightbox" style="display: none;">
                <div class="streaming-validation-backdrop"></div>
                <div class="streaming-validation-minimized-tab" title="Expand validation preview">
                    <div class="streaming-validation-minimized-badge"></div>
                    <i class="fas fa-chevron-right"></i>
                </div>
                <div class="streaming-validation-content">
                    <div class="streaming-validation-header">
                        <h4 class="streaming-validation-title">Validation Preview</h4>
                        <div class="streaming-validation-controls">
                            <button class="streaming-validation-expand"
                                    aria-label="Toggle expand"
                                    title="Expand/Collapse">
                                <i class="fas fa-expand"></i>
                            </button>
                            <button class="streaming-validation-copy"
                                    aria-label="Copy to clipboard"
                                    title="Copy image to clipboard">
                                <i class="fas fa-copy"></i>
                            </button>
                            <button class="streaming-validation-close"
                                    aria-label="Close"
                                    title="Close">&times;</button>
                        </div>
                    </div>
                    <div class="streaming-validation-body">
                        <div class="streaming-validation-progress-container">
                            <div class="streaming-validation-progress-bar" role="progressbar"
                                 aria-valuenow="0" aria-valuemin="0" aria-valuemax="100">
                                <div class="streaming-validation-progress-fill"></div>
                            </div>
                            <div class="streaming-validation-progress-label">Step 0 / 0</div>
                        </div>
                        <div class="streaming-validation-media-container">
                            <img class="streaming-validation-image" src="" alt="" style="display: none;">
                            <video class="streaming-validation-video" style="display: none;"
                                   controls muted playsinline loop></video>
                            <div class="streaming-validation-loading">
                                <i class="fas fa-spinner fa-spin"></i>
                                <span>Waiting for validation samples...</span>
                            </div>
                        </div>
                        <div class="streaming-validation-info">
                            <div class="streaming-validation-prompt"></div>
                            <div class="streaming-validation-meta"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', lightboxHTML);
        this.lightboxElement = document.getElementById('streaming-validation-lightbox');
    }

    setupEventListeners() {
        const closeBtn = this.lightboxElement.querySelector('.streaming-validation-close');
        const expandBtn = this.lightboxElement.querySelector('.streaming-validation-expand');
        const copyBtn = this.lightboxElement.querySelector('.streaming-validation-copy');
        const backdrop = this.lightboxElement.querySelector('.streaming-validation-backdrop');
        const minimizedTab = this.lightboxElement.querySelector('.streaming-validation-minimized-tab');

        // Close button now minimizes instead of closing
        closeBtn.addEventListener('click', () => this.minimize());

        // Only close on backdrop click when expanded
        backdrop.addEventListener('click', () => {
            if (this.isExpanded) {
                this.minimize();
            }
        });

        // Minimized tab click restores to PiP mode
        minimizedTab.addEventListener('click', () => this.restore());

        // Expand handler
        expandBtn.addEventListener('click', () => this.toggleExpand());

        // Copy handler
        copyBtn.addEventListener('click', () => this.copyToClipboard());

        // Keyboard shortcuts
        document.addEventListener('keydown', (event) => {
            if (!this.isVisible) return;

            // Don't respond to shortcuts when minimized
            if (this.isMinimized) {
                if (event.key === 'Escape') {
                    this.close(); // Escape fully closes even when minimized
                }
                return;
            }

            switch (event.key) {
                case 'Escape':
                    this.minimize();
                    break;
                case 'c':
                case 'C':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        this.copyToClipboard();
                    }
                    break;
                case 'f':
                case 'F':
                    if (!event.ctrlKey && !event.metaKey) {
                        event.preventDefault();
                        this.toggleExpand();
                    }
                    break;
            }
        });
    }

    registerSSEListener() {
        // Wait for SSE manager to be available
        const attemptRegistration = () => {
            if (window.sseManager && typeof window.sseManager.addEventListener === 'function') {
                window.sseManager.addEventListener('callback:validation', (payload) => {
                    this.handleValidationEvent(payload);
                });
                console.log('[StreamingValidationLightbox] Registered SSE validation listener');
            } else {
                setTimeout(attemptRegistration, 500);
            }
        };

        attemptRegistration();
    }

    handleValidationEvent(payload) {
        console.log('[StreamingValidationLightbox] Received validation event:', payload);

        // Only handle validation.image events (intermediary samples)
        if (payload.type !== 'validation.image') {
            console.log('[StreamingValidationLightbox] Skipping - not validation.image type:', payload.type);
            return;
        }

        if (payload.is_replay) {
            console.log('[StreamingValidationLightbox] Skipping - is_replay flag set');
            return;
        }

        console.log('[StreamingValidationLightbox] Processing validation image');

        const data = payload.data || {};
        const step = parseInt(data.step) || 0;
        const totalSteps = this.extractTotalSteps(data.step_label) || 0;

        // Show lightbox if not already visible, restore if minimized
        if (!this.isVisible) {
            console.log('[StreamingValidationLightbox] Showing lightbox');
            this.show();
        } else if (this.isMinimized) {
            // If minimized, show update badge but don't auto-restore
            console.log('[StreamingValidationLightbox] Showing update badge (minimized)');
            this.showUpdateBadge();
        }

        // Update content instantly (no flicker)
        this.updateContent(payload, step, totalSteps);

        // Reset auto-close timeout
        this.resetAutoCloseTimeout();

        // Update last sample time
        this.lastSampleTime = Date.now();
    }

    extractTotalSteps(stepLabel) {
        if (!stepLabel) return 0;
        const match = stepLabel.match(/(\d+)\/(\d+)/);
        return match ? parseInt(match[2]) : 0;
    }

    updateContent(payload, step, totalSteps) {
        const images = payload.images || [];
        const videos = payload.videos || [];
        const data = payload.data || {};

        const imgElement = this.lightboxElement.querySelector('.streaming-validation-image');
        const videoElement = this.lightboxElement.querySelector('.streaming-validation-video');
        const loadingElement = this.lightboxElement.querySelector('.streaming-validation-loading');
        const promptElement = this.lightboxElement.querySelector('.streaming-validation-prompt');
        const metaElement = this.lightboxElement.querySelector('.streaming-validation-meta');

        // Update prompt
        promptElement.textContent = data.prompt || payload.body || '';

        // Update meta information
        const resolution = data.resolution || [];
        const resolutionText = resolution.length >= 2 ? `${resolution[0]}x${resolution[1]}` : '';
        const validationType = data.validation_type || '';
        metaElement.innerHTML = `
            ${validationType ? `<span class="badge bg-secondary">${this.escapeHtml(validationType)}</span>` : ''}
            ${resolutionText ? `<span class="badge bg-info">${resolutionText}</span>` : ''}
        `;

        // Update progress bar
        this.updateProgressBar(step, totalSteps);

        // Handle media display
        if (videos.length > 0) {
            // Display video
            const videoSrc = this.normalizeVideoSrc(videos[0]);
            if (videoSrc) {
                this.currentVideoData = videoSrc;
                this.currentImageData = null;

                // Instant replacement - no fade
                videoElement.src = videoSrc;
                videoElement.style.display = 'block';
                imgElement.style.display = 'none';
                loadingElement.style.display = 'none';
            }
        } else if (images.length > 0) {
            // Display image
            const imageSrc = this.normalizeImageSrc(images[0]);
            if (imageSrc) {
                this.currentImageData = imageSrc;
                this.currentVideoData = null;

                // Instant replacement - no fade
                imgElement.src = imageSrc;
                imgElement.alt = data.prompt || 'Validation preview';
                imgElement.style.display = 'block';
                videoElement.style.display = 'none';
                loadingElement.style.display = 'none';
            }
        }
    }

    updateProgressBar(step, totalSteps) {
        const progressBar = this.lightboxElement.querySelector('.streaming-validation-progress-bar');
        const progressFill = this.lightboxElement.querySelector('.streaming-validation-progress-fill');
        const progressLabel = this.lightboxElement.querySelector('.streaming-validation-progress-label');

        const percentage = totalSteps > 0 ? (step / totalSteps) * 100 : 0;

        progressBar.setAttribute('aria-valuenow', percentage);
        progressBar.setAttribute('aria-valuemax', '100');
        progressFill.style.width = `${percentage}%`;
        progressLabel.textContent = `Step ${step} / ${totalSteps}`;
    }

    normalizeImageSrc(image) {
        if (!image) return null;

        if (typeof image === 'string') {
            const value = image.trim();
            if (!value) return null;

            // Already a data URI or URL
            if (value.startsWith('data:') || value.startsWith('http://') ||
                value.startsWith('https://') || value.startsWith('//')) {
                return value;
            }

            // Assume base64
            if (/^[A-Za-z0-9+/]+=*$/.test(value)) {
                return 'data:image/png;base64,' + value;
            }

            return value;
        }

        if (typeof image === 'object') {
            const data = image.src || image.url || image.data || image.base64 || image.image || image.image_base64;
            if (!data || typeof data !== 'string') return null;

            const trimmed = data.trim();
            if (trimmed.startsWith('data:') || trimmed.startsWith('http://') ||
                trimmed.startsWith('https://') || trimmed.startsWith('//')) {
                return trimmed;
            }

            // Wrap as base64
            const mime = image.mime_type || image.mime || 'image/png';
            return 'data:' + mime + ';base64,' + trimmed;
        }

        return null;
    }

    normalizeVideoSrc(video) {
        if (!video) return null;

        if (typeof video === 'string') {
            const value = video.trim();
            if (!value) return null;
            if (value.startsWith('data:') || value.startsWith('http://') ||
                value.startsWith('https://') || value.startsWith('//')) {
                return value;
            }
            if (/^[A-Za-z0-9+/]+=*$/.test(value)) {
                return 'data:video/mp4;base64,' + value;
            }
            return value;
        }

        if (typeof video === 'object') {
            const data = video.src || video.url || video.data || video.base64 || video.video || video.video_base64;
            if (!data || typeof data !== 'string') return null;
            const trimmed = data.trim();
            if (trimmed.startsWith('data:') || trimmed.startsWith('http://') ||
                trimmed.startsWith('https://') || trimmed.startsWith('//')) {
                return trimmed;
            }
            const mime = video.mime_type || video.mime || 'video/mp4';
            return 'data:' + mime + ';base64,' + trimmed;
        }

        return null;
    }

    show() {
        this.lightboxElement.style.display = 'flex';
        this.isVisible = true;
        this.isMinimized = false;
        this.lightboxElement.classList.remove('minimized');
        this.hideUpdateBadge();

        // Only prevent body scrolling when expanded
        if (this.isExpanded) {
            document.body.style.overflow = 'hidden';
        }
    }

    minimize() {
        // Minimize to tab instead of fully closing
        this.isMinimized = true;
        this.lightboxElement.classList.add('minimized');
        document.body.style.overflow = '';

        // Collapse if expanded
        if (this.isExpanded) {
            this.isExpanded = false;
            this.lightboxElement.classList.remove('expanded');

            const expandBtn = this.lightboxElement.querySelector('.streaming-validation-expand');
            const icon = expandBtn.querySelector('i');
            icon.classList.remove('fa-compress');
            icon.classList.add('fa-expand');
            expandBtn.setAttribute('title', 'Expand');
        }

        // Don't clear auto-close timeout - it should still close after 30s
    }

    restore() {
        // Restore from minimized state to PiP mode
        this.isMinimized = false;
        this.lightboxElement.classList.remove('minimized');
        this.hideUpdateBadge();
    }

    close() {
        // Fully close/hide the lightbox (called by timeout)
        this.lightboxElement.style.display = 'none';
        this.isVisible = false;
        this.isMinimized = false;
        this.lightboxElement.classList.remove('minimized');
        document.body.style.overflow = '';
        this.clearAutoCloseTimeout();
        this.hideUpdateBadge();

        // Reset expanded state
        if (this.isExpanded) {
            this.isExpanded = false;
            this.lightboxElement.classList.remove('expanded');

            const expandBtn = this.lightboxElement.querySelector('.streaming-validation-expand');
            const icon = expandBtn.querySelector('i');
            icon.classList.remove('fa-compress');
            icon.classList.add('fa-expand');
            expandBtn.setAttribute('title', 'Expand');
        }
    }

    showUpdateBadge() {
        const minimizedTab = this.lightboxElement.querySelector('.streaming-validation-minimized-tab');
        if (minimizedTab) {
            minimizedTab.classList.add('has-update');
        }
    }

    hideUpdateBadge() {
        const minimizedTab = this.lightboxElement.querySelector('.streaming-validation-minimized-tab');
        if (minimizedTab) {
            minimizedTab.classList.remove('has-update');
        }
    }

    toggleExpand() {
        const expandBtn = this.lightboxElement.querySelector('.streaming-validation-expand');
        const icon = expandBtn.querySelector('i');

        this.isExpanded = !this.isExpanded;

        if (this.isExpanded) {
            this.lightboxElement.classList.add('expanded');
            icon.classList.remove('fa-expand');
            icon.classList.add('fa-compress');
            expandBtn.setAttribute('title', 'Collapse');
            document.body.style.overflow = 'hidden';
        } else {
            this.lightboxElement.classList.remove('expanded');
            icon.classList.remove('fa-compress');
            icon.classList.add('fa-expand');
            expandBtn.setAttribute('title', 'Expand');
            document.body.style.overflow = '';
        }
    }

    async copyToClipboard() {
        const copyBtn = this.lightboxElement.querySelector('.streaming-validation-copy');
        const icon = copyBtn.querySelector('i');
        const originalClass = icon.className;

        try {
            // Determine what to copy (image or video)
            if (this.currentImageData) {
                // For images, try to copy the actual image to clipboard
                if (this.currentImageData.startsWith('data:image/')) {
                    // Convert data URL to blob
                    const blob = await this.dataURLToBlob(this.currentImageData);
                    await navigator.clipboard.write([
                        new ClipboardItem({
                            [blob.type]: blob
                        })
                    ]);
                } else {
                    // Fallback to copying URL as text
                    await navigator.clipboard.writeText(this.currentImageData);
                }
            } else if (this.currentVideoData) {
                // For videos, copy the URL/data as text
                await navigator.clipboard.writeText(this.currentVideoData);
            } else {
                throw new Error('No media to copy');
            }

            // Show success feedback
            icon.className = 'fas fa-check';
            setTimeout(() => {
                icon.className = originalClass;
            }, 2000);
        } catch (error) {
            console.error('[StreamingValidationLightbox] Failed to copy to clipboard:', error);

            // Show error feedback
            icon.className = 'fas fa-times';
            setTimeout(() => {
                icon.className = originalClass;
            }, 2000);
        }
    }

    async dataURLToBlob(dataURL) {
        const response = await fetch(dataURL);
        return await response.blob();
    }

    resetAutoCloseTimeout() {
        this.clearAutoCloseTimeout();

        this.autoCloseTimeout = setTimeout(() => {
            if (this.isVisible) {
                console.log('[StreamingValidationLightbox] Auto-closing after 30 seconds of inactivity');
                this.close();
            }
        }, this.autoCloseDelay);
    }

    clearAutoCloseTimeout() {
        if (this.autoCloseTimeout) {
            clearTimeout(this.autoCloseTimeout);
            this.autoCloseTimeout = null;
        }
    }

    escapeHtml(text) {
        if (typeof text !== 'string') return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('[StreamingValidationLightbox] DOM ready, creating instance');
    window.streamingValidationLightbox = new StreamingValidationLightbox();
    window.StreamingValidationLightbox = StreamingValidationLightbox;
    console.log('[StreamingValidationLightbox] Instance created and attached to window');
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = StreamingValidationLightbox;
}
