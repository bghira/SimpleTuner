/**
 * Notifications Alpine.js Component
 *
 * Manages notification channels, preferences, and delivery history.
 */

window.notificationsComponent = function() {
    return {
        // State
        loading: false,
        saving: false,
        deleting: false,
        // Hero CTA state (using HintMixin with localStorage)
        ...(window.HintMixin?.createSingleHint({
            useApi: false,
            storageKey: 'notifications_hero_dismissed',
        }) || { heroDismissed: false, loadHeroCTAState() {}, dismissHeroCTA() {}, restoreHeroCTA() {} }),
        channels: [],
        preferences: [],
        history: [],
        presets: [],
        eventTypes: [],
        deliveryStats: {
            sent_today: 0,
            failed_today: 0,
        },
        testingChannelId: null,
        historyLoading: false,
        expandedCategories: [],

        // Modal state
        editingChannel: null,
        deletingChannel: null,
        channelModal: null,
        deleteModal: null,

        // Form state
        channelForm: {
            channel_type: 'email',
            name: '',
            is_enabled: true,
            smtp_host: '',
            smtp_port: 587,
            smtp_username: '',
            smtp_password: '',
            smtp_use_tls: true,
            smtp_from_address: '',
            smtp_from_name: 'SimpleTuner Cloud',
            webhook_url: '',
            webhook_secret: '',
            // IMAP settings for email response handling
            imap_enabled: false,
            imap_host: '',
            imap_port: 993,
            imap_username: '',
            imap_password: '',
            imap_use_ssl: true,
        },

        async init() {
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }

            this.loadChannels();
            this.loadPreferences();
            this.loadEventTypes();
            this.loadPresets();
            this.loadHistory();
            this.loadHeroCTAState();

            // Initialize Bootstrap modals
            this.$nextTick(() => {
                if (typeof bootstrap !== 'undefined') {
                    if (this.$refs.channelModal) {
                        this.channelModal = new bootstrap.Modal(this.$refs.channelModal);
                    }
                    if (this.$refs.deleteModal) {
                        this.deleteModal = new bootstrap.Modal(this.$refs.deleteModal);
                    }
                }
            });
        },

        // Hero CTA methods provided by HintMixin spread above

        async loadChannels() {
            this.loading = true;
            try {
                const response = await fetch('/api/cloud/notifications/channels');
                if (response.ok) {
                    const data = await response.json();
                    this.channels = data.channels || [];
                }
            } catch (error) {
                console.error('Failed to load channels:', error);
            } finally {
                this.loading = false;
            }
        },

        async loadPreferences() {
            try {
                const response = await fetch('/api/cloud/notifications/preferences');
                if (response.ok) {
                    const data = await response.json();
                    this.preferences = data.preferences || [];
                }
            } catch (error) {
                console.error('Failed to load preferences:', error);
            }
        },

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

        async loadPresets() {
            try {
                const response = await fetch('/api/cloud/notifications/presets');
                if (response.ok) {
                    const data = await response.json();
                    this.presets = data.presets || [];
                }
            } catch (error) {
                console.error('Failed to load presets:', error);
            }
        },

        async loadHistory() {
            this.historyLoading = true;
            try {
                const response = await fetch('/api/cloud/notifications/history?limit=20');
                if (response.ok) {
                    const data = await response.json();
                    this.history = data.entries || [];

                    // Calculate today's stats
                    const today = new Date().toDateString();
                    this.deliveryStats.sent_today = this.history.filter(
                        e => new Date(e.sent_at).toDateString() === today
                    ).length;
                }
            } catch (error) {
                console.error('Failed to load history:', error);
            } finally {
                this.historyLoading = false;
            }
        },

        get groupedEventTypes() {
            const groups = {};
            for (const event of this.eventTypes) {
                const category = event.category || 'Other';
                if (!groups[category]) {
                    groups[category] = [];
                }
                groups[category].push(event);
            }
            return groups;
        },

        toggleCategory(category) {
            const idx = this.expandedCategories.indexOf(category);
            if (idx === -1) {
                this.expandedCategories.push(category);
            } else {
                this.expandedCategories.splice(idx, 1);
            }
        },

        getPreferenceChannel(eventTypeId) {
            const pref = this.preferences.find(p => p.event_type === eventTypeId);
            return pref ? pref.channel_id.toString() : '';
        },

        async updatePreference(eventTypeId, channelId) {
            try {
                if (!channelId) {
                    // Find and delete existing preference
                    const pref = this.preferences.find(p => p.event_type === eventTypeId);
                    if (pref) {
                        await fetch(`/api/cloud/notifications/preferences/${pref.id}`, {
                            method: 'DELETE',
                        });
                    }
                } else {
                    // Create or update preference
                    const response = await fetch('/api/cloud/notifications/preferences', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            event_type: eventTypeId,
                            channel_id: parseInt(channelId),
                            is_enabled: true,
                            recipients: [],
                            min_severity: 'info',
                        }),
                    });
                    if (!response.ok) {
                        throw new Error('Failed to save preference');
                    }
                }
                await this.loadPreferences();
            } catch (error) {
                console.error('Failed to update preference:', error);
                if (window.showToast) {
                    window.showToast('Failed to update preference', 'error');
                }
            }
        },

        showChannelModal(defaultType) {
            this.editingChannel = null;
            this.resetChannelForm();
            if (defaultType) {
                this.channelForm.channel_type = defaultType;
            }
            if (this.channelModal) {
                this.channelModal.show();
            }
        },

        editChannel(channel) {
            this.editingChannel = channel;
            this.channelForm = {
                channel_type: channel.channel_type,
                name: channel.name,
                is_enabled: channel.is_enabled,
                smtp_host: channel.smtp_host || '',
                smtp_port: channel.smtp_port || 587,
                smtp_username: channel.smtp_username || '',
                smtp_password: '', // Don't pre-fill password
                smtp_use_tls: channel.smtp_use_tls !== false,
                smtp_from_address: channel.smtp_from_address || '',
                smtp_from_name: channel.smtp_from_name || 'SimpleTuner Cloud',
                webhook_url: channel.webhook_url || '',
                webhook_secret: '', // Don't pre-fill secret
                imap_enabled: channel.imap_enabled || false,
                imap_host: channel.imap_host || '',
                imap_port: channel.imap_port || 993,
                imap_username: channel.imap_username || '',
                imap_password: '', // Don't pre-fill password
                imap_use_ssl: channel.imap_use_ssl !== false,
            };
            if (this.channelModal) {
                this.channelModal.show();
            }
        },

        resetChannelForm() {
            this.channelForm = {
                channel_type: 'email',
                name: '',
                is_enabled: true,
                smtp_host: '',
                smtp_port: 587,
                smtp_username: '',
                smtp_password: '',
                smtp_use_tls: true,
                smtp_from_address: '',
                smtp_from_name: 'SimpleTuner Cloud',
                webhook_url: '',
                webhook_secret: '',
                imap_enabled: false,
                imap_host: '',
                imap_port: 993,
                imap_username: '',
                imap_password: '',
                imap_use_ssl: true,
            };
        },

        async saveChannel() {
            this.saving = true;
            try {
                const payload = {
                    channel_type: this.channelForm.channel_type,
                    name: this.channelForm.name,
                    is_enabled: this.channelForm.is_enabled,
                };

                // Add type-specific fields
                if (this.channelForm.channel_type === 'email') {
                    payload.smtp_host = this.channelForm.smtp_host;
                    payload.smtp_port = this.channelForm.smtp_port;
                    payload.smtp_username = this.channelForm.smtp_username;
                    payload.smtp_use_tls = this.channelForm.smtp_use_tls;
                    payload.smtp_from_address = this.channelForm.smtp_from_address;
                    payload.smtp_from_name = this.channelForm.smtp_from_name;
                    if (this.channelForm.smtp_password) {
                        payload.smtp_password = this.channelForm.smtp_password;
                    }
                    // IMAP settings for email response handling
                    payload.imap_enabled = this.channelForm.imap_enabled;
                    if (this.channelForm.imap_enabled) {
                        payload.imap_host = this.channelForm.imap_host;
                        payload.imap_port = this.channelForm.imap_port;
                        payload.imap_username = this.channelForm.imap_username;
                        payload.imap_use_ssl = this.channelForm.imap_use_ssl;
                        if (this.channelForm.imap_password) {
                            payload.imap_password = this.channelForm.imap_password;
                        }
                    }
                } else if (this.channelForm.channel_type === 'webhook' || this.channelForm.channel_type === 'slack') {
                    payload.webhook_url = this.channelForm.webhook_url;
                    if (this.channelForm.webhook_secret) {
                        payload.webhook_secret = this.channelForm.webhook_secret;
                    }
                }

                let response;
                if (this.editingChannel) {
                    response = await fetch(`/api/cloud/notifications/channels/${this.editingChannel.id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                } else {
                    response = await fetch('/api/cloud/notifications/channels', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload),
                    });
                }

                if (response.ok) {
                    if (this.channelModal) {
                        this.channelModal.hide();
                    }
                    await this.loadChannels();
                    if (window.showToast) {
                        window.showToast(
                            this.editingChannel ? 'Channel updated' : 'Channel created',
                            'success'
                        );
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to save channel');
                }
            } catch (error) {
                console.error('Failed to save channel:', error);
                if (window.showToast) {
                    window.showToast(error.message || 'Failed to save channel', 'error');
                }
            } finally {
                this.saving = false;
            }
        },

        confirmDeleteChannel(channel) {
            this.deletingChannel = channel;
            if (this.deleteModal) {
                this.deleteModal.show();
            }
        },

        async deleteChannel() {
            if (!this.deletingChannel) return;

            this.deleting = true;
            try {
                const response = await fetch(`/api/cloud/notifications/channels/${this.deletingChannel.id}`, {
                    method: 'DELETE',
                });

                if (response.ok || response.status === 204) {
                    if (this.deleteModal) {
                        this.deleteModal.hide();
                    }
                    await this.loadChannels();
                    await this.loadPreferences();
                    if (window.showToast) {
                        window.showToast('Channel deleted', 'success');
                    }
                } else {
                    throw new Error('Failed to delete channel');
                }
            } catch (error) {
                console.error('Failed to delete channel:', error);
                if (window.showToast) {
                    window.showToast('Failed to delete channel', 'error');
                }
            } finally {
                this.deleting = false;
                this.deletingChannel = null;
            }
        },

        async toggleChannel(channel) {
            try {
                const response = await fetch(`/api/cloud/notifications/channels/${channel.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ is_enabled: !channel.is_enabled }),
                });

                if (response.ok) {
                    await this.loadChannels();
                    if (window.showToast) {
                        window.showToast(
                            channel.is_enabled ? 'Channel disabled' : 'Channel enabled',
                            'success'
                        );
                    }
                }
            } catch (error) {
                console.error('Failed to toggle channel:', error);
            }
        },

        async testChannel(channelId) {
            this.testingChannelId = channelId;
            try {
                const response = await fetch(`/api/cloud/notifications/channels/${channelId}/test`, {
                    method: 'POST',
                });

                const data = await response.json();
                if (data.success) {
                    if (window.showToast) {
                        window.showToast(`Test successful (${data.latency_ms?.toFixed(0) || '?'}ms)`, 'success');
                    }
                } else {
                    if (window.showToast) {
                        window.showToast(`Test failed: ${data.error || 'Unknown error'}`, 'error');
                    }
                }
            } catch (error) {
                console.error('Failed to test channel:', error);
                if (window.showToast) {
                    window.showToast('Failed to test channel', 'error');
                }
            } finally {
                this.testingChannelId = null;
            }
        },

        async applyPreset(presetId) {
            try {
                const response = await fetch(`/api/cloud/notifications/presets/${presetId}`);
                if (response.ok) {
                    const preset = await response.json();
                    this.showChannelModal('email');
                    this.channelForm.smtp_host = preset.smtp_host;
                    this.channelForm.smtp_port = preset.smtp_port;
                    this.channelForm.smtp_use_tls = preset.smtp_use_tls;
                    this.channelForm.name = preset.name;
                    if (window.showToast) {
                        window.showToast(`Applied ${preset.name} preset`, 'info');
                    }
                }
            } catch (error) {
                console.error('Failed to apply preset:', error);
            }
        },

        async skipNotifications() {
            try {
                await fetch('/api/cloud/notifications/skip', { method: 'POST' });
                this.dismissHeroCTA();
            } catch (error) {
                console.error('Failed to skip notifications:', error);
            }
        },

        // Helper methods
        getChannelIcon(type) {
            return window.UIHelpers?.getChannelIcon(type) || 'fas fa-bell';
        },

        getChannelDetail(channel) {
            switch (channel.channel_type) {
                case 'email':
                    return channel.smtp_host || '';
                case 'webhook':
                case 'slack':
                    if (!channel.webhook_url) return '';
                    try {
                        return new URL(channel.webhook_url).hostname;
                    } catch {
                        return channel.webhook_url;
                    }
                default:
                    return '';
            }
        },

        getStatusIcon(status) {
            return window.UIHelpers?.getStatusIcon(status) || 'fa-question';
        },

        formatEventType(type) {
            return window.UIHelpers?.formatEventType(type) || type;
        },

        formatTime(timestamp) {
            return window.UIHelpers?.formatRelativeTime(timestamp) || '';
        },
    };
};
