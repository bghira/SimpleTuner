/**
 * Admin Panel - Notifications Module
 *
 * Handles notification channel CRUD and testing.
 */

window.adminNotificationMethods = {
    loadNotificationsIfNeeded() {
        if (!this.notifications.loaded) {
            this.loadNotificationChannels();
            this.loadNotificationPresets();
            this.loadNotificationPreferences();
            this.loadEventTypes();
            this.loadNotificationsSkipState();
        }
    },

    loadNotificationsSkipState() {
        const stored = localStorage.getItem('notifications_setup_skipped');
        this.notifications.skipped = stored === 'true';
    },

    skipNotificationsSetup() {
        this.notifications.skipped = true;
        localStorage.setItem('notifications_setup_skipped', 'true');
    },

    undoSkipNotificationsSetup() {
        this.notifications.skipped = false;
        localStorage.removeItem('notifications_setup_skipped');
    },

    async loadNotificationChannels() {
        this.notifications.loading = true;
        try {
            const response = await fetch('/api/cloud/notifications/channels');
            if (response.ok) {
                const data = await response.json();
                this.notifications.channels = data.channels || [];
                this.notifications.loaded = true;
            }
        } catch (error) {
            console.error('Failed to load notification channels:', error);
        } finally {
            this.notifications.loading = false;
        }
    },

    async loadNotificationPresets() {
        try {
            const response = await fetch('/api/cloud/notifications/presets');
            if (response.ok) {
                const data = await response.json();
                this.notifications.presets = data.presets || [];
            }
        } catch (error) {
            console.error('Failed to load notification presets:', error);
        }
    },

    async loadNotificationHistory() {
        try {
            const response = await fetch('/api/cloud/notifications/history?limit=50');
            if (response.ok) {
                const data = await response.json();
                this.notifications.history = data.entries || [];
            }
        } catch (error) {
            console.error('Failed to load notification history:', error);
        }
    },

    async loadNotificationStatus() {
        try {
            const response = await fetch('/api/cloud/notifications/status');
            if (response.ok) {
                this.notifications.status = await response.json();
            }
        } catch (error) {
            console.error('Failed to load notification status:', error);
        }
    },

    showChannelForm(type = 'email') {
        this.editingChannel = null;
        this.channelForm = this.getDefaultChannelForm(type);
        this.channelFormOpen = true;
    },

    showChannelFormWithPreset(presetName) {
        const preset = this.notifications.presets.find(p => p.name === presetName);
        if (!preset) return;

        this.editingChannel = null;
        this.channelForm = {
            channel_type: 'email',
            name: preset.display_name || preset.name,
            is_enabled: true,
            smtp_host: preset.smtp_host || '',
            smtp_port: preset.smtp_port || 587,
            smtp_username: '',
            smtp_password: '',
            smtp_from_address: '',
            smtp_use_tls: preset.smtp_use_tls !== false,
            imap_enabled: preset.imap_host ? true : false,
            imap_host: preset.imap_host || '',
            imap_port: preset.imap_port || 993,
        };
        this.channelFormOpen = true;
    },

    showEditChannelForm(channel) {
        this.editingChannel = channel;
        this.channelForm = {
            channel_type: channel.channel_type,
            name: channel.name,
            is_enabled: channel.is_enabled,
            smtp_host: channel.smtp_host || '',
            smtp_port: channel.smtp_port || 587,
            smtp_username: channel.smtp_username || '',
            smtp_password: '',
            smtp_from_address: channel.smtp_from_address || '',
            smtp_use_tls: channel.smtp_use_tls !== false,
            imap_enabled: channel.imap_enabled || false,
            imap_host: channel.imap_host || '',
            imap_port: channel.imap_port || 993,
            webhook_url: channel.webhook_url || '',
        };
        this.channelFormOpen = true;
    },

    getDefaultChannelForm(type) {
        if (type === 'email') {
            return {
                channel_type: 'email',
                name: '',
                is_enabled: true,
                smtp_host: '',
                smtp_port: 587,
                smtp_username: '',
                smtp_password: '',
                smtp_from_address: '',
                smtp_use_tls: true,
                imap_enabled: false,
                imap_host: '',
                imap_port: 993,
            };
        }
        return {
            channel_type: type,
            name: '',
            is_enabled: true,
            webhook_url: '',
        };
    },

    async saveChannel() {
        this.saving = true;
        try {
            let response;
            if (this.editingChannel) {
                response = await fetch(`/api/cloud/notifications/channels/${this.editingChannel.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.channelForm),
                });
            } else {
                response = await fetch('/api/cloud/notifications/channels', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.channelForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingChannel ? 'Channel updated' : 'Channel created', 'success');
                }
                this.channelFormOpen = false;
                await this.loadNotificationChannels();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save channel', 'error');
            }
        } catch (error) {
            console.error('Failed to save channel:', error);
            if (window.showToast) window.showToast('Failed to save channel', 'error');
        } finally {
            this.saving = false;
        }
    },

    async testChannel(channel) {
        this.notifications.testing = channel.id;
        this.notifications.testResult = null;
        try {
            const response = await fetch(`/api/cloud/notifications/channels/${channel.id}/test`, {
                method: 'POST',
            });
            if (response.ok) {
                this.notifications.testResult = await response.json();
            } else {
                const data = await response.json();
                this.notifications.testResult = {
                    success: false,
                    error: data.detail || 'Test failed',
                };
            }
        } catch (error) {
            console.error('Failed to test channel:', error);
            this.notifications.testResult = {
                success: false,
                error: 'Network error',
            };
        } finally {
            this.notifications.testing = null;
        }
    },

    async toggleChannelEnabled(channel) {
        try {
            const response = await fetch(`/api/cloud/notifications/channels/${channel.id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ is_enabled: !channel.is_enabled }),
            });
            if (response.ok) {
                channel.is_enabled = !channel.is_enabled;
                if (window.showToast) {
                    window.showToast(channel.is_enabled ? 'Channel enabled' : 'Channel disabled', 'success');
                }
            }
        } catch (error) {
            console.error('Failed to toggle channel:', error);
        }
    },

    async deleteChannel(channel) {
        if (!confirm(`Delete notification channel "${channel.name}"?`)) return;

        try {
            const response = await fetch(`/api/cloud/notifications/channels/${channel.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Channel deleted', 'success');
                await this.loadNotificationChannels();
            } else {
                if (window.showToast) window.showToast('Failed to delete channel', 'error');
            }
        } catch (error) {
            console.error('Failed to delete channel:', error);
            if (window.showToast) window.showToast('Failed to delete channel', 'error');
        }
    },

    getChannelTypeIcon(type) {
        // Get full icon class and extract just the icon name
        const full = window.UIHelpers?.getChannelIcon(type) || 'fas fa-bell';
        const match = full.match(/fa-[\w-]+$/);
        return match ? match[0] : 'fa-bell';
    },

    getChannelTypeBadgeClass(type) {
        return window.UIHelpers?.getChannelTypeBadgeClass(type) || 'bg-secondary';
    },

    getChannelConfigSummary(channel) {
        if (channel.channel_type === 'email') {
            return `${channel.smtp_host}:${channel.smtp_port}`;
        }
        if (channel.webhook_url) {
            return channel.webhook_url.substring(0, 40) + '...';
        }
        return '-';
    },

    shouldShowNotificationsHeroCTA() {
        return this.notifications.channels.length === 0 && !this.notifications.skipped;
    },

    // --- Event Type and Preference Methods ---

    async loadEventTypes() {
        try {
            const response = await fetch('/api/cloud/notifications/events');
            if (response.ok) {
                const data = await response.json();
                this.eventTypes = data.event_types || [];
            }
        } catch (error) {
            console.error('Failed to load event types:', error);
        }
    },

    async loadNotificationPreferences() {
        try {
            const response = await fetch('/api/cloud/notifications/preferences');
            if (response.ok) {
                const data = await response.json();
                this.notifications.preferences = data.preferences || [];
            }
        } catch (error) {
            console.error('Failed to load notification preferences:', error);
        }
    },

    showPreferenceForm() {
        this.editingPreference = null;
        this.preferenceForm = {
            event_type: this.eventTypes[0]?.id || '',
            channel_id: this.notifications.channels[0]?.id || null,
            is_enabled: true,
            recipients: '',
            min_severity: 'info',
        };
        this.preferenceFormOpen = true;
    },

    showEditPreferenceForm(pref) {
        this.editingPreference = pref;
        this.preferenceForm = {
            event_type: pref.event_type,
            channel_id: pref.channel_id,
            is_enabled: pref.is_enabled,
            recipients: (pref.recipients || []).join(', '),
            min_severity: pref.min_severity || 'info',
        };
        this.preferenceFormOpen = true;
    },

    async savePreference() {
        this.saving = true;
        try {
            const payload = {
                event_type: this.preferenceForm.event_type,
                channel_id: this.preferenceForm.channel_id,
                is_enabled: this.preferenceForm.is_enabled,
                recipients: this.preferenceForm.recipients
                    ? this.preferenceForm.recipients.split(',').map(r => r.trim()).filter(r => r)
                    : [],
                min_severity: this.preferenceForm.min_severity,
            };

            let response;
            if (this.editingPreference) {
                response = await fetch(`/api/cloud/notifications/preferences/${this.editingPreference.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            } else {
                response = await fetch('/api/cloud/notifications/preferences', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingPreference ? 'Preference updated' : 'Preference created', 'success');
                }
                this.preferenceFormOpen = false;
                await this.loadNotificationPreferences();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save preference', 'error');
            }
        } catch (error) {
            console.error('Failed to save preference:', error);
            if (window.showToast) window.showToast('Failed to save preference', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deletePreference(pref) {
        if (!confirm(`Delete routing for "${this.getEventTypeName(pref.event_type)}"?`)) return;

        try {
            const response = await fetch(`/api/cloud/notifications/preferences/${pref.id}`, {
                method: 'DELETE',
            });
            if (response.ok || response.status === 204) {
                if (window.showToast) window.showToast('Preference deleted', 'success');
                await this.loadNotificationPreferences();
            } else {
                if (window.showToast) window.showToast('Failed to delete preference', 'error');
            }
        } catch (error) {
            console.error('Failed to delete preference:', error);
            if (window.showToast) window.showToast('Failed to delete preference', 'error');
        }
    },

    getEventTypeName(eventType) {
        const found = this.eventTypes.find(e => e.id === eventType);
        return found?.name || eventType.replace(/[._]/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    },

    getEventTypeCategory(eventType) {
        const found = this.eventTypes.find(e => e.id === eventType);
        return found?.category || 'Other';
    },

    getChannelName(channelId) {
        const channel = this.notifications.channels.find(c => c.id === channelId);
        return channel?.name || `Channel #${channelId}`;
    },

    getSeverityBadgeClass(severity) {
        return window.UIHelpers?.getSeverityBadgeClass(severity) || 'bg-secondary';
    },

    getGroupedEventTypes() {
        const groups = {};
        for (const et of this.eventTypes) {
            const cat = et.category || 'Other';
            if (!groups[cat]) groups[cat] = [];
            groups[cat].push(et);
        }
        return groups;
    },
};
