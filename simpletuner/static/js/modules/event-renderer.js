/**
 * Event Renderer - Renders SSE callback events into the event dock
 * Handles validation images, debug images, and all other event types
 */

(function(window) {
    'use strict';

    var EventRenderer = (function() {
        var eventList = null;
        var maxEvents = 500;
        var processedEventIds = new Set(); // Track processed event IDs to prevent duplicates
        var listenersRegistered = false; // Track if event listeners are already registered
        var severityIcons = {
            'success': 'fas fa-check-circle text-success',
            'info': 'fas fa-info-circle text-info',
            'warning': 'fas fa-exclamation-triangle text-warning',
            'error': 'fas fa-exclamation-circle text-danger',
            'danger': 'fas fa-exclamation-circle text-danger',
            'critical': 'fas fa-times-circle text-danger',
            'debug': 'fas fa-bug text-secondary',
            'secondary': 'fas fa-info-circle text-secondary'
        };

        var eventTypeIcons = {
            'validation': 'fas fa-check-double text-success',
            'validation_complete': 'fas fa-check-double text-success',
            'checkpoint': 'fas fa-save text-primary',
            'progress': 'fas fa-chart-line text-info',
            'debug': 'fas fa-bug text-muted',
            'error': 'fas fa-exclamation-triangle text-danger',
            'notification': 'fas fa-bell text-muted'
        };

        function init() {
            eventList = document.getElementById('eventList');
            if (!eventList) {
                console.warn('[EventRenderer] eventList element not found');
                return;
            }

            // Clear placeholder loading message but keep any existing events
            const firstChild = eventList.firstElementChild;
            if (
                eventList.childElementCount <= 1 &&
                firstChild &&
                firstChild.classList.contains('text-muted')
            ) {
                eventList.innerHTML = '';
            }

            // Register SSE listeners
            if (window.sseManager) {
                registerListeners();
            } else {
                // Wait for SSE manager to be ready
                setTimeout(init, 500);
            }
        }

        function registerListeners() {
            // Prevent double-registration of listeners
            if (listenersRegistered) {
                console.log('[EventRenderer] Listeners already registered, skipping');
                return;
            }

            var callbackTypes = ['validation', 'checkpoint', 'progress', 'alert', 'status', 'job', 'debug', 'metric'];

            callbackTypes.forEach(function(type) {
                window.sseManager.addEventListener('callback:' + type, function(payload) {
                    renderCallbackEvent(type, payload);
                });
            });

            // Also listen for direct event types
            window.sseManager.addEventListener('notification', function(payload) {
                renderEvent('notification', payload);
            });

            window.sseManager.addEventListener('error', function(payload) {
                renderEvent('error', payload);
            });

            listenersRegistered = true;
            console.log('[EventRenderer] Registered SSE event listeners');
        }

        function renderCallbackEvent(category, payload) {
            // Skip intermediary validation images - these are only for the streaming lightbox
            // Final validation images (type: 'validation') should still appear in the event list
            if (payload.type === 'validation.image') {
                console.log('[EventRenderer] Skipping intermediary validation image for event dock');
                return;
            }

            // Check for duplicate events by ID
            var eventId = payload.id;
            if (eventId && processedEventIds.has(eventId)) {
                // Skip duplicate event
                return;
            }

            // Mark event as processed
            if (eventId) {
                processedEventIds.add(eventId);

                // Limit size of processed IDs set to prevent memory issues
                if (processedEventIds.size > maxEvents * 2) {
                    // Remove oldest half of IDs (convert to array, remove first half, convert back to Set)
                    var idsArray = Array.from(processedEventIds);
                    processedEventIds = new Set(idsArray.slice(idsArray.length / 2));
                }
            }

            var eventType = category;
            var message = payload.headline || payload.body || payload.message || '';
            var severity = payload.severity || 'info';
            var images = payload.images || [];
            var videos = payload.videos || [];
            var audios = payload.audios || [];

            renderEvent(eventType, {
                message: message,
                severity: severity,
                images: images,
                videos: videos,
                audios: audios,
                timestamp: payload.timestamp || payload.timestamp_display || new Date().toISOString(),
                raw: payload
            });
        }

        function renderEvent(type, data) {
            if (!eventList) return;

            var eventItem = createEventElement(type, data);
            if (!eventItem) return;

            // Remove oldest event if at limit
            while (eventList.children.length >= maxEvents) {
                eventList.removeChild(eventList.lastChild);
            }

            // Prepend new event (newest at top)
            eventList.insertBefore(eventItem, eventList.firstChild);

            // Scroll to top to show newest event
            eventList.scrollTop = 0;

            // Initialize lightbox for any images in this event
            if (window.eventLightbox && data.images && data.images.length > 0) {
                window.eventLightbox.initializeImages(eventItem);
            }
        }

        function createEventElement(type, data) {
            var message = escapeHtml(data.message || '');
            if (!message) return null;

            var severity = data.severity || 'info';
            var timestamp = formatTimestamp(data.timestamp);
            var icon = getEventIcon(type, severity);
            var severityClass = getSeverityClass(severity);

            var eventItem = document.createElement('div');
            eventItem.className = 'event-item border-bottom py-2';

            var contentHTML =
                '<div class="d-flex align-items-start">' +
                    '<i class="' + icon + ' me-2 mt-1"></i>' +
                    '<div class="flex-grow-1">' +
                        '<div class="event-headline text-' + severityClass + '">' + message + '</div>';

            // Add images if present
            if (data.images && data.images.length > 0) {
                contentHTML += renderImages(data.images, message, type);
            }
            if (data.videos && data.videos.length > 0) {
                contentHTML += renderVideos(data.videos, message, type);
            }
            if (data.audios && data.audios.length > 0) {
                contentHTML += renderAudios(data.audios, message, type);
            }

            contentHTML +=
                        '<small class="text-muted">' + timestamp + '</small>' +
                    '</div>' +
                '</div>';

            eventItem.innerHTML = contentHTML;
            if (typeof severity === 'string') {
                eventItem.dataset.severity = severity.toLowerCase();
            }
            if (type) {
                eventItem.dataset.messageType = String(type);
            }
            return eventItem;
        }

        function renderImages(images, alt, eventType) {
            var imagesHTML = '<div class="event-images d-flex flex-wrap gap-2 mt-2">';
            var altText = escapeHtml(alt || 'Event image');
            var group = escapeHtml(eventType || 'event');

            images.forEach(function(image, idx) {
                var src = normalizeImageSrc(image);
                if (!src) return;

                imagesHTML +=
                    '<img src="' + escapeHtml(src) + '" ' +
                    'alt="' + altText + '" ' +
                    'class="event-image img-fluid rounded border cursor-pointer" ' +
                    'loading="lazy" ' +
                    'data-lightbox="event-images" ' +
                    'data-lightbox-group="event-' + group + '" ' +
                    'data-lightbox-index="' + idx + '" />';
            });

            imagesHTML += '</div>';
            return imagesHTML;
        }

        function renderAudios(audios, alt, eventType) {
            var audiosHTML = '<div class="event-audios d-flex flex-column gap-2 mt-2">';
            var group = escapeHtml(eventType || 'event');
            var altText = escapeHtml(alt || 'Event audio');

            audios.forEach(function(audio, idx) {
                var src = normalizeAudioSrc(audio);
                if (!src) return;

                audiosHTML +=
                    '<audio src="' + escapeHtml(src) + '" ' +
                    'aria-label="' + altText + '" ' +
                    'class="event-audio w-100" ' +
                    'data-audio-group="event-' + group + '" ' +
                    'data-audio-index="' + idx + '" ' +
                    'controls preload="metadata"></audio>';
            });

            audiosHTML += '</div>';
            return audiosHTML;
        }

        function renderVideos(videos, alt, eventType) {
            var videosHTML = '<div class="event-videos d-flex flex-wrap gap-2 mt-2">';
            var altText = escapeHtml(alt || 'Event video');
            var group = escapeHtml(eventType || 'event');

            videos.forEach(function(video, idx) {
                var src = normalizeVideoSrc(video);
                if (!src) return;

                videosHTML +=
                    '<video src="' + escapeHtml(src) + '" ' +
                    'aria-label="' + altText + '" ' +
                    'class="event-video img-fluid rounded border" ' +
                    'controls muted playsinline loop></video>';
            });

            videosHTML += '</div>';
            return videosHTML;
        }

        function normalizeImageSrc(image) {
            if (!image) return null;

            if (typeof image === 'string') {
                var value = image.trim();
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
                var data = image.src || image.url || image.data || image.base64 || image.image || image.image_base64;
                if (!data || typeof data !== 'string') return null;

                data = data.trim();
                if (data.startsWith('data:') || data.startsWith('http://') ||
                    data.startsWith('https://') || data.startsWith('//')) {
                    return data;
                }

                // Wrap as base64
                var mime = image.mime_type || image.mime || 'image/png';
                return 'data:' + mime + ';base64,' + data;
            }

            return null;
        }

        function normalizeAudioSrc(audio) {
            if (!audio) return null;

            if (typeof audio === 'string') {
                var value = audio.trim();
                if (!value) return null;
                if (value.startsWith('data:') || value.startsWith('http://') ||
                    value.startsWith('https://') || value.startsWith('//')) {
                    return value;
                }
                if (/^[A-Za-z0-9+/]+=*$/.test(value)) {
                    return 'data:audio/wav;base64,' + value;
                }
                return value;
            }

            if (typeof audio === 'object') {
                var data = audio.src || audio.url || audio.data || audio.base64 || audio.audio || audio.audio_base64;
                if (!data || typeof data !== 'string') return null;
                var trimmed = data.trim();
                if (trimmed.startsWith('data:') || trimmed.startsWith('http://') ||
                    trimmed.startsWith('https://') || trimmed.startsWith('//')) {
                    return trimmed;
                }
                var mime = audio.mime_type || audio.mime || 'audio/wav';
                return 'data:' + mime + ';base64,' + trimmed;
            }

            return null;
        }

        function normalizeVideoSrc(video) {
            if (!video) return null;

            if (typeof video === 'string') {
                var value = video.trim();
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
                var data = video.src || video.url || video.data || video.base64 || video.video || video.video_base64;
                if (!data || typeof data !== 'string') return null;
                data = data.trim();
                if (data.startsWith('data:') || data.startsWith('http://') ||
                    data.startsWith('https://') || data.startsWith('//')) {
                    return data;
                }
                var mime = video.mime_type || video.mime || 'video/mp4';
                return 'data:' + mime + ';base64,' + data;
            }

            return null;
        }

        function getEventIcon(type, severity) {
            // Try event type specific icon first
            var icon = eventTypeIcons[type];
            if (icon) return icon;

            // Fall back to severity icon
            return severityIcons[severity] || severityIcons['info'];
        }

        function getSeverityClass(severity) {
            var mapping = {
                'success': 'success',
                'info': 'info',
                'warning': 'warning',
                'error': 'danger',
                'critical': 'danger',
                'danger': 'danger',
                'debug': 'secondary',
                'secondary': 'secondary'
            };
            return mapping[severity] || 'info';
        }

        function formatTimestamp(timestamp) {
            if (!timestamp) return '';

            try {
                var date = new Date(timestamp);
                if (isNaN(date.getTime())) return '';

                var hours = String(date.getHours()).padStart(2, '0');
                var minutes = String(date.getMinutes()).padStart(2, '0');
                var seconds = String(date.getSeconds()).padStart(2, '0');

                return hours + ':' + minutes + ':' + seconds;
            } catch (e) {
                return '';
            }
        }

        function escapeHtml(text) {
            if (typeof text !== 'string') return '';
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Public API
        return {
            init: init,
            renderEvent: renderEvent
        };
    })();

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', EventRenderer.init);
    } else {
        EventRenderer.init();
    }

    // Expose globally
    window.EventRenderer = EventRenderer;

})(window);
