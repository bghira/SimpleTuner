/**
 * Event Lightbox Module
 * Displays training event images and image galleries.
 */

class EventLightbox {
    constructor() {
        this.lightboxElement = null;
        this.currentIndex = 0;
        this.images = [];
        this.trigger = null;
        this.previousOverflow = '';
        this.imageRequest = 0;
        this.imageSelector = '[data-lightbox="event-images"], [data-lightbox-src]';

        this.init();
    }

    init() {
        // Create lightbox element if it doesn't exist
        this.createLightbox();

        // Set up event delegation for event images
        this.setupEventDelegation();

        // Set up keyboard navigation
        this.setupKeyboardNavigation();
    }

    createLightbox() {
        // Check if lightbox already exists
        if (document.getElementById('event-lightbox')) {
            this.lightboxElement = document.getElementById('event-lightbox');
            return;
        }

        // Create lightbox HTML structure
        const lightboxHTML = `
            <div id="event-lightbox" class="event-lightbox" role="dialog" aria-modal="true"
                 aria-labelledby="event-lightbox-title" style="display: none;">
                <div class="event-lightbox-backdrop"></div>
                <div class="event-lightbox-content">
                    <div class="event-lightbox-header">
                        <h4 class="event-lightbox-title" id="event-lightbox-title">Image</h4>
                        <button class="event-lightbox-close" aria-label="Close">&times;</button>
                    </div>
                    <div class="event-lightbox-image-container">
                        <img class="event-lightbox-image" src="" alt="">
                        <div class="event-lightbox-loading">Loading...</div>
                    </div>
                    <div class="event-lightbox-controls">
                        <button class="event-lightbox-prev" aria-label="Previous image">‹</button>
                        <span class="event-lightbox-counter">1 / 1</span>
                        <button class="event-lightbox-next" aria-label="Next image">›</button>
                    </div>
                </div>
            </div>
        `;

        // Append to body
        document.body.insertAdjacentHTML('beforeend', lightboxHTML);
        this.lightboxElement = document.getElementById('event-lightbox');

        // Set up event listeners for lightbox controls
        this.setupLightboxControls();
    }

    setupLightboxControls() {
        const closeBtn = this.lightboxElement.querySelector('.event-lightbox-close');
        const backdrop = this.lightboxElement.querySelector('.event-lightbox-backdrop');
        const prevBtn = this.lightboxElement.querySelector('.event-lightbox-prev');
        const nextBtn = this.lightboxElement.querySelector('.event-lightbox-next');

        // Close handlers
        closeBtn.addEventListener('click', () => this.close());
        backdrop.addEventListener('click', () => this.close());

        // Navigation handlers
        prevBtn.addEventListener('click', () => this.navigate(-1));
        nextBtn.addEventListener('click', () => this.navigate(1));
    }

    setupEventDelegation() {
        document.addEventListener('click', (event) => {
            const target = event.target.closest(this.imageSelector);
            if (target) {
                event.preventDefault();
                this.open(target);
            }
        });
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (event) => {
            if (!this.lightboxElement || this.lightboxElement.style.display === 'none') {
                return;
            }

            switch (event.key) {
                case 'Escape':
                    this.close();
                    break;
                case 'ArrowLeft':
                    this.navigate(-1);
                    break;
                case 'ArrowRight':
                    this.navigate(1);
                    break;
            }
        });
    }

    open(clickedImage) {
        const gallery = clickedImage.closest('[data-lightbox-gallery], .event-images');
        this.images = gallery ? Array.from(gallery.querySelectorAll(this.imageSelector)) : [clickedImage];
        this.currentIndex = this.images.indexOf(clickedImage);
        this.trigger = clickedImage;
        this.previousOverflow = document.body.style.overflow;

        this.lightboxElement.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        this.showImage(this.currentIndex);
        this.lightboxElement.querySelector('.event-lightbox-close').focus();
    }

    close() {
        this.imageRequest++;
        this.lightboxElement.style.display = 'none';
        document.body.style.overflow = this.previousOverflow;
        this.images = [];
        this.trigger?.focus();
        this.trigger = null;
    }

    navigate(direction) {
        if (this.images.length <= 1) return;

        const newIndex = this.currentIndex + direction;

        // Handle wrap-around
        let targetIndex;
        if (newIndex < 0) {
            targetIndex = this.images.length - 1;
        } else if (newIndex >= this.images.length) {
            targetIndex = 0;
        } else {
            targetIndex = newIndex;
        }

        this.currentIndex = targetIndex;
        this.showImage(targetIndex);
    }

    showImage(index) {
        const image = this.images[index];
        const imgElement = this.lightboxElement.querySelector('.event-lightbox-image');
        const loadingElement = this.lightboxElement.querySelector('.event-lightbox-loading');
        const counterElement = this.lightboxElement.querySelector('.event-lightbox-counter');
        const titleElement = this.lightboxElement.querySelector('.event-lightbox-title');

        const request = ++this.imageRequest;
        loadingElement.style.display = 'block';
        loadingElement.style.color = '';
        loadingElement.textContent = 'Loading...';
        imgElement.style.opacity = '0';

        const altText = image.dataset.lightboxLabel || image.getAttribute('alt') || 'Image';
        titleElement.textContent = altText;

        // Update counter
        counterElement.textContent = `${index + 1} / ${this.images.length}`;

        // Update navigation buttons state
        this.updateNavigationButtons();

        const preview = new Image();
        preview.className = imgElement.className;
        preview.alt = altText;
        preview.onload = () => {
            if (request !== this.imageRequest) return;
            imgElement.replaceWith(preview);
            preview.style.opacity = '1';
            loadingElement.style.display = 'none';
        };
        preview.onerror = () => {
            if (request !== this.imageRequest) return;
            loadingElement.textContent = 'Failed to load image';
            loadingElement.style.color = '#dc3545';
        };
        preview.src = image.dataset.lightboxSrc || image.src;
    }

    updateNavigationButtons() {
        const prevBtn = this.lightboxElement.querySelector('.event-lightbox-prev');
        const nextBtn = this.lightboxElement.querySelector('.event-lightbox-next');

        // Disable buttons if only one image
        if (this.images.length <= 1) {
            prevBtn.disabled = true;
            nextBtn.disabled = true;
            return;
        }

        // Enable buttons
        prevBtn.disabled = false;
        nextBtn.disabled = false;
    }

    /**
     * Initialize lightbox for images in a container (called by event-renderer)
     * This is a no-op since we use event delegation, but kept for compatibility
     */
    initializeImages(container) {
        // Event delegation handles all clicks, so no per-element setup needed
        // This method exists for compatibility with event-renderer.js
    }
}

// Initialize the event lightbox when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.eventLightbox = new EventLightbox();
    // Also expose the class for checking instanceof
    window.EventLightbox = EventLightbox;
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EventLightbox;
}
