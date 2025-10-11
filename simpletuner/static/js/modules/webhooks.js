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
                showAddForm: false,
                editingIndex: null,
                newWebhook: {
                    webhook_type: 'discord',
                    webhook_url: '',
                    message_prefix: '',
                    log_level: 'info'
                },

                // Initialize from hidden field if present
                initialize() {
                    const hiddenField = document.querySelector('[name="--discord_webhooks"]');
                    if (hiddenField && hiddenField.value) {
                        try {
                            const parsed = JSON.parse(hiddenField.value);
                            if (Array.isArray(parsed)) {
                                this.webhooks = parsed;
                            }
                        } catch (e) {
                            console.warn('Could not parse existing discord webhook config:', e);
                        }
                    }
                },

                getEmptyWebhook() {
                    return {
                        webhook_type: 'discord',
                        webhook_url: '',
                        message_prefix: '',
                        log_level: 'info'
                    };
                },

                addWebhook() {
                    if (this.validateWebhook(this.newWebhook)) {
                        // Create a clean webhook object with only the fields needed for this type
                        const cleanWebhook = {
                            webhook_type: this.newWebhook.webhook_type,
                            log_level: this.newWebhook.log_level
                        };

                        if (this.newWebhook.webhook_type === 'discord') {
                            cleanWebhook.webhook_url = this.newWebhook.webhook_url;
                            if (this.newWebhook.message_prefix) {
                                cleanWebhook.message_prefix = this.newWebhook.message_prefix;
                            }
                        } else {
                            // raw type
                            cleanWebhook.callback_url = this.newWebhook.callback_url;
                        }

                        if (this.editingIndex !== null) {
                            // Update existing webhook
                            this.webhooks[this.editingIndex] = cleanWebhook;
                            this.editingIndex = null;
                        } else {
                            // Add new webhook
                            this.webhooks.push(cleanWebhook);
                        }
                        this.newWebhook = this.getEmptyWebhook();
                        this.showAddForm = false;
                        this.updateHiddenField();
                    }
                },

                editWebhook(index) {
                    this.editingIndex = index;
                    this.newWebhook = { ...this.webhooks[index] };
                    this.showAddForm = true;
                },

                deleteWebhook(index) {
                    if (confirm('Remove this webhook?')) {
                        this.webhooks.splice(index, 1);
                        this.updateHiddenField();
                    }
                },

                cancelEdit() {
                    this.showAddForm = false;
                    this.editingIndex = null;
                    this.newWebhook = this.getEmptyWebhook();
                },

                validateWebhook(webhook) {
                    // Get the appropriate URL field based on webhook type
                    const url = webhook.webhook_type === 'discord' ? webhook.webhook_url : webhook.callback_url;

                    if (!url || url.trim() === '') {
                        if (window.showToast) {
                            window.showToast('URL is required', 'error');
                        }
                        return false;
                    }

                    // Basic URL validation
                    try {
                        new URL(url);
                    } catch (e) {
                        if (window.showToast) {
                            window.showToast('Invalid URL format', 'error');
                        }
                        return false;
                    }

                    // Discord-specific validation
                    if (webhook.webhook_type === 'discord') {
                        if (!url.includes('discord.com/api/webhooks/')) {
                            if (window.showToast) {
                                window.showToast('Discord webhook URL must contain discord.com/api/webhooks/', 'error');
                            }
                            return false;
                        }
                    }

                    return true;
                },

                updateHiddenField() {
                    const hiddenField = document.querySelector('[name="--discord_webhooks"]');
                    if (hiddenField) {
                        // Store user-configured webhooks (localhost callback is added by trainer.js)
                        hiddenField.value = JSON.stringify(this.webhooks);
                        hiddenField.dispatchEvent(new Event('input', { bubbles: true }));
                        hiddenField.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                },

                getLogLevelBadgeClass(level) {
                    const classes = {
                        'debug': 'bg-purple-100 text-purple-800',
                        'info': 'bg-blue-100 text-blue-800',
                        'warning': 'bg-yellow-100 text-yellow-800',
                        'error': 'bg-red-100 text-red-800',
                        'critical': 'bg-red-600 text-white'
                    };
                    return classes[level] || classes['info'];
                },

                getWebhookTypeBadge(type) {
                    return type === 'discord' ? 'Discord' : type === 'raw' ? 'Custom' : type;
                }
            };
        }
    };
});

// Re-initialize after HTMX swaps
document.body.addEventListener('htmx:afterSwap', function(evt) {
    if (evt.detail.target.id && evt.detail.target.id.includes('publishing-tab-content')) {
        // Trigger Alpine re-initialization if needed
        const webhookSection = document.querySelector('[x-data*="webhookManager"]');
        if (webhookSection && webhookSection.__x) {
            webhookSection.__x.$data.initialize();
        }
    }
});
