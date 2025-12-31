/**
 * Cloud Dashboard - Job Notifications Module
 *
 * Handles browser notifications for job completion events.
 */

window.cloudJobNotificationMethods = {
    initJobNotifications() {
        // Request notification permission on first interaction
        if ('Notification' in window && Notification.permission === 'default') {
            // Will request on first job submission
            this._notificationPermissionRequested = false;
        }

        // Listen for cloud job update events via SSE
        if (window.SSEManager) {
            window.SSEManager.addEventListener('cloud_job_update', (data) => {
                this.handleCloudJobUpdate(data);
            });

            // Also listen via the generic message handler
            window.SSEManager.addEventListener('message', (data) => {
                if (data && data.type === 'cloud_job_update') {
                    this.handleCloudJobUpdate(data);
                }
            });
        }

        // Listen for custom event from EventSource if SSEManager doesn't catch it
        document.addEventListener('cloud-job-update', (event) => {
            this.handleCloudJobUpdate(event.detail);
        });
    },

    async requestNotificationPermission() {
        if (!('Notification' in window)) {
            console.log('Browser does not support notifications');
            return false;
        }

        if (Notification.permission === 'granted') {
            return true;
        }

        if (Notification.permission === 'denied') {
            return false;
        }

        try {
            const permission = await Notification.requestPermission();
            return permission === 'granted';
        } catch (error) {
            console.error('Failed to request notification permission:', error);
            return false;
        }
    },

    handleCloudJobUpdate(data) {
        if (!data) return;

        const jobId = data.job_id || data.jobId;
        const newStatus = data.new_status || data.status || (data.updates && data.updates.status);
        const jobName = data.config_name || jobId?.substring(0, 12) || 'Job';

        // Only show notifications for terminal states
        if (newStatus === 'completed') {
            this.showJobCompletedNotification(jobId, jobName, data);
        } else if (newStatus === 'failed') {
            const errorMessage = data.error_message || (data.updates && data.updates.error_message) || 'Unknown error';
            this.showJobFailedNotification(jobId, jobName, errorMessage, data);
        } else if (newStatus === 'cancelled') {
            this.showJobCancelledNotification(jobId, jobName);
        }

        // Refresh job list if on cloud tab
        if (typeof this.loadJobs === 'function') {
            this.loadJobs();
        }
    },

    showJobCompletedNotification(jobId, jobName, data) {
        // Show toast
        if (window.showToast) {
            const cost = data.cost_usd || (data.updates && data.updates.cost_usd);
            let message = `Job "${jobName}" completed successfully!`;
            if (cost) {
                message += ` (Cost: $${cost.toFixed(2)})`;
            }
            window.showToast(message, 'success');
        }

        // Show browser notification
        this.showBrowserNotification(
            'Job Completed',
            `${jobName} finished successfully`,
            'success',
            jobId
        );

        // Play success sound if enabled
        this.playNotificationSound('success');
    },

    showJobFailedNotification(jobId, jobName, errorMessage, data) {
        // Show toast
        if (window.showToast) {
            window.showToast(`Job "${jobName}" failed: ${errorMessage}`, 'error');
        }

        // Show browser notification
        this.showBrowserNotification(
            'Job Failed',
            `${jobName}: ${errorMessage}`,
            'error',
            jobId
        );

        // Play error sound if enabled
        this.playNotificationSound('error');
    },

    showJobCancelledNotification(jobId, jobName) {
        // Show toast
        if (window.showToast) {
            window.showToast(`Job "${jobName}" was cancelled`, 'info');
        }
    },

    async showBrowserNotification(title, body, type, jobId) {
        // Check if browser notifications are enabled in settings
        if (!this.notificationsEnabled) {
            return;
        }

        // Request permission if needed
        if ('Notification' in window) {
            if (Notification.permission === 'default') {
                const granted = await this.requestNotificationPermission();
                if (!granted) return;
            } else if (Notification.permission !== 'granted') {
                return;
            }

            try {
                const notification = new Notification(title, {
                    body: body,
                    icon: '/static/img/favicon.png',
                    badge: '/static/img/favicon.png',
                    tag: jobId || 'cloud-job',
                    requireInteraction: type === 'error',
                });

                notification.onclick = () => {
                    window.focus();
                    notification.close();
                    // If we have a job ID, could navigate to job details
                };

                // Auto-close success notifications after 5 seconds
                if (type === 'success') {
                    setTimeout(() => notification.close(), 5000);
                }
            } catch (error) {
                console.error('Failed to show notification:', error);
            }
        }
    },

    playNotificationSound(type) {
        // Check if sound is enabled in settings
        if (!this.notificationSoundEnabled) {
            return;
        }

        // Simple notification sounds using Web Audio API
        try {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const oscillator = audioContext.createOscillator();
            const gainNode = audioContext.createGain();

            oscillator.connect(gainNode);
            gainNode.connect(audioContext.destination);

            if (type === 'success') {
                // Pleasant two-tone chime for success
                oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
                oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.1); // E5
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.3);
            } else if (type === 'error') {
                // Lower tone for error
                oscillator.frequency.setValueAtTime(220, audioContext.currentTime); // A3
                gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
                gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
                oscillator.start(audioContext.currentTime);
                oscillator.stop(audioContext.currentTime + 0.5);
            }
        } catch (error) {
            // Audio not supported or blocked, ignore silently
        }
    },

    // NOTE: notificationsEnabled and notificationSoundEnabled getters/setters
    // moved to index.js final return object to avoid spread evaluation issues

    toggleNotifications(enabled) {
        this.notificationsEnabled = enabled;
        if (enabled) {
            this.requestNotificationPermission();
        }
    },

    toggleNotificationSound(enabled) {
        this.notificationSoundEnabled = enabled;
    },
};
