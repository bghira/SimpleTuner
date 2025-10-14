/**
 * Event Lightbox Module
 * Handles lightbox functionality for event images
 */

class EventLightbox {
    constructor() {
        this.lightboxElement = null;
        this.currentGroup = null;
        this.currentIndex = 0;
        this.images = [];

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
            <div id="event-lightbox" class="event-lightbox" style="display: none;">
                <div class="event-lightbox-backdrop"></div>
                <div class="event-lightbox-content">
                    <div class="event-lightbox-header">
                        <h4 class="event-lightbox-title">Event Image</h4>
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
        // Use event delegation to handle clicks on event images
        document.addEventListener('click', (event) => {
            const target = event.target;

            // Check if clicked element is an event image with lightbox data
            if (target.matches('[data-lightbox="event-images"]')) {
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
        const group = clickedImage.getAttribute('data-lightbox-group');
        const clickedIndex = parseInt(clickedImage.getAttribute('data-lightbox-index')) || 0;

        // Collect all images in the same group
        this.images = Array.from(document.querySelectorAll(`[data-lightbox-group="${group}"]`));
        this.currentGroup = group;
        this.currentIndex = clickedIndex;

        // Show lightbox
        this.lightboxElement.style.display = 'flex';
        document.body.style.overflow = 'hidden'; // Prevent background scrolling

        // Load and display the current image
        this.showImage(this.currentIndex);

        // Focus management for accessibility
        this.lightboxElement.focus();
    }

    close() {
        this.lightboxElement.style.display = 'none';
        document.body.style.overflow = ''; // Restore scrolling
        this.currentGroup = null;
        this.images = [];
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

        // Show loading state
        loadingElement.style.display = 'block';
        imgElement.style.opacity = '0';

        // Update title from alt text or group name
        const altText = image.getAttribute('alt') || 'Event Image';
        titleElement.textContent = altText;

        // Update counter
        counterElement.textContent = `${index + 1} / ${this.images.length}`;

        // Update navigation buttons state
        this.updateNavigationButtons();

        // Load the image
        const tempImg = new Image();
        tempImg.onload = () => {
            imgElement.src = image.src;
            imgElement.alt = altText;
            imgElement.style.opacity = '1';
            loadingElement.style.display = 'none';
        };

        tempImg.onerror = () => {
            loadingElement.textContent = 'Failed to load image';
            loadingElement.style.color = '#dc3545';
        };

        tempImg.src = image.src;
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
}

// Initialize the event lightbox when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.eventLightbox = new EventLightbox();
});

// Export for potential use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EventLightbox;
}
