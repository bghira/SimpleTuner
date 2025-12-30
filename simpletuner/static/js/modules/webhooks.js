/**
 * Webhooks Module - Manage Discord and Custom Webhook Destinations
 * Handles configuration of multiple webhook endpoints for training notifications
 */

document.addEventListener('alpine:init', () => {
    // Webhook manager component
    window.webhookManager = {
        init() {
            return {
                webhooks: [],
                libraryWebhooks: [],
                loadingLibrary: false,
                modalOpen: false,
                editingIndex: null,
                isDirty: false,
                editWebhook: {
                    webhook_type: 'discord',
                    webhook_url: '',
                    callback_url: '',
                    message_prefix: '',
                    log_level: 'info'
                },

                // Initialize from server-rendered data attribute and load library
                async initialize() {
                    // Read initial webhooks from data attribute (set by server)
                    const container = this.$el;
                    const initialData = container.getAttribute('data-initial-webhooks');
                    if (initialData) {
                        try {
                            const parsed = JSON.parse(initialData);
                            if (Array.isArray(parsed)) {
                                this.webhooks = parsed;
                            } else if (parsed && typeof parsed === 'object') {
                                this.webhooks = [parsed];
                            }
                            // Update hidden field with loaded data
                            this.updateHiddenField();
                        } catch (e) {
                            console.warn('Could not parse initial webhook config:', e);
                        }
                    }
                    await this.refreshLibrary();
                },

                async refreshLibrary() {
                    this.loadingLibrary = true;
                    try {
                        const response = await fetch('/api/configs?config_type=webhook');
                        if (response.ok) {
                            const data = await response.json();
                            this.libraryWebhooks = data.configs || [];
                        }
                    } catch (error) {
                        console.error('Failed to load webhook library:', error);
                    } finally {
                        this.loadingLibrary = false;
                    }
                },

                getEmptyWebhook() {
                    return {
                        webhook_type: 'discord',
                        webhook_url: '',
                        callback_url: '',
                        message_prefix: '',
                        log_level: 'info'
                    };
                },

                openModal(index = null) {
                    if (index !== null && index !== undefined) {
                        this.editingIndex = index;
                        this.editWebhook = { ...this.getEmptyWebhook(), ...this.webhooks[index] };
                    } else {
                        this.editingIndex = null;
                        this.editWebhook = this.getEmptyWebhook();
                    }
                    this.modalOpen = true;
                    this.$nextTick(() => {
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                    });
                },

                closeModal() {
                    this.modalOpen = false;
                    this.editingIndex = null;
                    this.editWebhook = this.getEmptyWebhook();
                },

                saveWebhook() {
                    if (!this.validateWebhook(this.editWebhook)) {
                        return;
                    }

                    const cleanWebhook = {
                        webhook_type: this.editWebhook.webhook_type,
                        log_level: this.editWebhook.log_level
                    };

                    if (this.editWebhook.webhook_type === 'discord') {
                        cleanWebhook.webhook_url = this.editWebhook.webhook_url;
                        if (this.editWebhook.message_prefix) {
                            cleanWebhook.message_prefix = this.editWebhook.message_prefix;
                        }
                    } else {
                        cleanWebhook.callback_url = this.editWebhook.callback_url;
                    }

                    if (this.editingIndex !== null) {
                        this.webhooks[this.editingIndex] = cleanWebhook;
                    } else {
                        this.webhooks.push(cleanWebhook);
                    }

                    this.updateHiddenField();
                    this.isDirty = true;
                    this.closeModal();

                    if (window.showToast) {
                        window.showToast('Inline webhook added', 'success');
                    }
                },

                removeWebhook(index) {
                    this.webhooks.splice(index, 1);
                    this.updateHiddenField();
                    this.isDirty = true;
                    if (window.showToast) {
                        window.showToast('Webhook removed', 'success');
                    }
                },

                isWebhookActive(name) {
                    return this.webhooks.some(w => w.name === name);
                },

                async toggleWebhook(config) {
                    const existingIndex = this.webhooks.findIndex(w => w.name === config.name);

                    if (existingIndex >= 0) {
                        // Remove
                        this.webhooks.splice(existingIndex, 1);
                        if (window.showToast) {
                            window.showToast(`Removed "${config.name}"`, 'success');
                        }
                    } else {
                        // Add - fetch full config from API
                        try {
                            const response = await fetch(`/api/configs/webhooks/${encodeURIComponent(config.name)}`);
                            if (response.ok) {
                                const data = await response.json();
                                const webhookConfig = {
                                    name: config.name,
                                    ...data.config
                                };
                                this.webhooks.push(webhookConfig);
                                if (window.showToast) {
                                    window.showToast(`Added "${config.name}"`, 'success');
                                }
                            } else {
                                throw new Error('Failed to load webhook config');
                            }
                        } catch (error) {
                            console.error('Failed to add webhook:', error);
                            if (window.showToast) {
                                window.showToast('Failed to add webhook', 'error');
                            }
                            return;
                        }
                    }

                    this.updateHiddenField();
                    this.isDirty = true;
                },

                async testLibraryWebhook(name) {
                    if (window.showToast) {
                        window.showToast(`Testing "${name}"...`, 'info');
                    }

                    try {
                        const response = await fetch(`/api/configs/webhooks/${encodeURIComponent(name)}/test`, {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' }
                        });
                        const result = await response.json();

                        if (result.success) {
                            if (window.showToast) {
                                window.showToast(result.message || 'Test successful', 'success');
                            }
                        } else {
                            if (window.showToast) {
                                window.showToast(result.error || 'Test failed', 'error');
                            }
                        }
                    } catch (error) {
                        console.error('Webhook test failed:', error);
                        if (window.showToast) {
                            window.showToast('Test failed: ' + error.message, 'error');
                        }
                    }
                },

                validateWebhook(webhook) {
                    const url = webhook.webhook_type === 'discord' ? webhook.webhook_url : webhook.callback_url;

                    if (!url || url.trim() === '') {
                        if (window.showToast) {
                            window.showToast('URL is required', 'error');
                        }
                        return false;
                    }

                    try {
                        new URL(url);
                    } catch (e) {
                        if (window.showToast) {
                            window.showToast('Invalid URL format', 'error');
                        }
                        return false;
                    }

                    if (webhook.webhook_type === 'discord' && !url.includes('discord.com/api/webhooks/')) {
                        if (window.showToast) {
                            window.showToast('Discord URL should contain discord.com/api/webhooks/', 'warning');
                        }
                    }

                    return true;
                },

                updateHiddenField() {
                    const hiddenField = document.querySelector('[name="--webhook_config"]');
                    if (hiddenField) {
                        hiddenField.value = JSON.stringify(this.webhooks);
                        hiddenField.dispatchEvent(new Event('input', { bubbles: true }));
                        hiddenField.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }
            };
        }
    };
});

// Re-initialize after HTMX swaps
document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail.target.id && evt.detail.target.id.includes('publishing-tab-content')) {
        const webhookSection = document.querySelector('[x-data*="webhookManager"]');
        if (webhookSection && webhookSection.__x) {
            webhookSection.__x.$data.initialize();
        }
    }
});
