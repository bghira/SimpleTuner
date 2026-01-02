/**
 * Cloud Dashboard - Main Component Index
 *
 * Combines all modules into the main cloudDashboardComponent.
 * This file must be loaded after all other cloud modules.
 *
 * Module loading order:
 * State modules (loaded first):
 *   state/setup-state.js
 *   state/provider-state.js
 *   state/jobs-state.js
 *   state/metrics-state.js
 *   state/publishing-state.js
 *   state/system-status-state.js
 *   state/queue-state.js
 *   state/upload-state.js
 *   state/ui-state.js
 *   state/modals-state.js
 *   state/connection-state.js
 *   state/index.js
 *
 * Method modules:
 *   providers.js
 *   jobs.js
 *   job-submission.js
 *   job-logs.js
 *   metrics.js
 *   system-status.js
 *   queue.js
 *   admin-modal.js
 *   onboarding.js
 *   utilities.js
 *   job-notifications.js
 *   s3_config.js
 *
 * Main component:
 *   index.js (this file)
 */

if (!window.cloudDashboardComponent) {
    window.cloudDashboardComponent = function(initialData) {
        try {
        // Use the composed state factory
        const state = window.cloudStateFactory(initialData);

        // Core methods that stay in the main component
        const coreMethods = {
            /**
             * Load essential data required for initial UI render.
             * These must complete before the dashboard is usable.
             */
            _loadCoreData() {
                this.loadProviders();
                this.validateApiKey();
                this.loadJobs();
                this.loadMetrics();
            },

            /**
             * Load secondary data that enhances the UI but isn't critical.
             * Can run in parallel after core data loads.
             */
            _loadOptionalData() {
                this.loadSystemStatus();
                this.loadPublishingStatus();
                this.loadDatasetStatus();
                this.loadHints();
                this.loadOnboardingState();
                this.loadAvailableConfigs();
                this.loadPendingApprovals();
                this.fetchPollingStatus();
                this.refreshQueueStats();
            },

            /**
             * Start background polling and notification handlers.
             */
            _startBackgroundTasks() {
                if (typeof this.initJobNotifications === 'function') {
                    this.initJobNotifications();
                }
                this.startPolling();
                this.startInlineProgressPolling();
                this.startHealthCheckAutoRefresh();
                this.updateSetupStatus();
            },

            /**
             * Main initialization sequence.
             * Groups calls into: core data, optional data, background tasks.
             */
            _runCoreInitialization() {
                this._loadCoreData();
                this._loadOptionalData();
                this._startBackgroundTasks();
            },

            async init() {
                // Check first-run setup status before loading everything else
                await this.checkSetupStatus();

                // Only proceed with normal initialization if setup is complete AND user is logged in
                if (!this.setupState.needsSetup && !this.setupState.needsLogin) {
                    this._runCoreInitialization();
                    this.initSSEConnectionMonitor();

                    this.$watch('activeProvider', () => {
                        this.jobsInitialized = false;
                        this.loadJobs();
                        this.loadProviderConfig();
                    });
                }
            },

            initSSEConnectionMonitor() {
                // Get initial state from SSEManager if available
                if (window.SSEManager) {
                    try {
                        const state = window.SSEManager.getState();
                        this.sseConnection.status = state.connectionState || 'unknown';
                        this.sseConnection.lastUpdated = new Date();
                    } catch (e) {
                        this.sseConnection.status = 'unknown';
                    }
                }

                // Listen for connection status events
                this._sseEventHandler = (event) => {
                    if (event.detail) {
                        this.sseConnection.status = event.detail.status || 'unknown';
                        this.sseConnection.message = event.detail.message || '';
                        this.sseConnection.lastUpdated = new Date();
                    }
                };
                document.addEventListener('trainer-connection-status', this._sseEventHandler);

                // Also hook into the global updateConnectionStatus if we can
                this._originalUpdateConnectionStatus = window.updateConnectionStatus;
                const self = this;
                window.updateConnectionStatus = function(status, message) {
                    self.sseConnection.status = status || 'unknown';
                    self.sseConnection.message = message || '';
                    self.sseConnection.lastUpdated = new Date();
                    if (self._originalUpdateConnectionStatus) {
                        self._originalUpdateConnectionStatus(status, message);
                    }
                };
            },

            // Note: sseStatusColor, sseStatusIcon getters moved to final return object to avoid spread evaluation issue

            async checkSetupStatus() {
                this.setupState.loading = true;
                const maxRetries = 2;

                for (let attempt = 0; attempt < maxRetries; attempt++) {
                    try {
                        const response = await fetch('/api/auth/setup/status');
                        if (response.ok) {
                            const data = await response.json();
                            this.setupState.needsSetup = data.needs_setup;
                            this.setupState.hasAdmin = data.has_admin;

                            // If setup is complete (users exist), check if we're authenticated
                            if (!data.needs_setup && data.user_count > 0) {
                                await this._checkAuthentication();
                            }

                            this.setupState.loading = false;
                            return;
                        } else if (response.status === 404) {
                            // Endpoint doesn't exist - auth module not loaded, no setup needed
                            this.setupState.needsSetup = false;
                            this.setupState.needsLogin = false;
                            this.setupState.loading = false;
                            return;
                        }
                        // Other HTTP errors (5xx, etc.) - will retry
                    } catch (error) {
                        // Network error - will retry
                        if (attempt < maxRetries - 1) {
                            await new Promise(r => setTimeout(r, 1000)); // Wait 1s before retry
                            continue;
                        }
                    }
                }

                // All retries exhausted - proceed but log warning
                console.warn('Could not determine setup status after retries, proceeding without setup');
                this.setupState.needsSetup = false;
                this.setupState.loading = false;
            },

            async _checkAuthentication() {
                try {
                    const response = await fetch('/api/auth/check');
                    if (response.ok) {
                        const data = await response.json();
                        if (data.authenticated) {
                            this.setupState.needsLogin = false;
                            this.setupState.currentUser = {
                                id: data.user_id,
                                username: data.username,
                                is_admin: data.is_admin,
                            };
                            // Also update the global cloudAuth store
                            if (Alpine.store('cloudAuth')) {
                                Alpine.store('cloudAuth').currentUser = this.setupState.currentUser;
                            }
                        } else {
                            this.setupState.needsLogin = true;
                            this.setupState.currentUser = null;
                        }
                    } else {
                        // Auth endpoint not available or error
                        this.setupState.needsLogin = false;
                    }
                } catch (error) {
                    console.warn('Failed to check authentication:', error);
                    this.setupState.needsLogin = false;
                }
            },

            async submitLogin() {
                if (!this.setupState.loginForm.username || !this.setupState.loginForm.password) {
                    this.setupState.error = 'Username and password are required';
                    return;
                }

                this.setupState.submitting = true;
                this.setupState.error = null;

                try {
                    const response = await fetch('/api/auth/login', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            username: this.setupState.loginForm.username,
                            password: this.setupState.loginForm.password,
                            remember_me: this.setupState.loginForm.rememberMe,
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.success) {
                            this.setupState.needsLogin = false;
                            this.setupState.currentUser = data.user;

                            // Update global cloudAuth store
                            if (Alpine.store('cloudAuth')) {
                                Alpine.store('cloudAuth').currentUser = data.user;
                                Alpine.store('cloudAuth').loaded = true;
                            }

                            if (window.showToast) {
                                window.showToast(`Welcome back, ${data.user.username || data.user.display_name}!`, 'success');
                            }

                            // Run full initialization now that we're logged in
                            this._runCoreInitialization();
                            this.initSSEConnectionMonitor();

                            this.$watch('activeProvider', () => {
                                this.jobsInitialized = false;
                                this.loadJobs();
                                this.loadProviderConfig();
                            });
                        }
                    } else {
                        const data = await response.json().catch(() => ({}));
                        this.setupState.error = window.UIHelpers?.extractErrorMessage(data, 'Invalid username or password') || data.detail || 'Invalid username or password';
                    }
                } catch (error) {
                    console.error('Login failed:', error);
                    this.setupState.error = 'Network error. Please try again.';
                } finally {
                    this.setupState.submitting = false;
                }
            },

            // Note: setupFormValid getter moved to final return object to avoid spread evaluation issue

            async submitFirstRunSetup() {
                if (!this.setupFormValid) return;

                this.setupState.submitting = true;
                this.setupState.error = null;

                try {
                    const response = await fetch('/api/auth/setup/first-admin', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            email: this.setupState.form.email,
                            username: this.setupState.form.username,
                            password: this.setupState.form.password,
                            display_name: this.setupState.form.displayName || null,
                        }),
                    });

                    if (response.ok) {
                        const data = await response.json();
                        if (data.success) {
                            // Setup complete, mark as no longer needing setup
                            this.setupState.needsSetup = false;
                            if (window.showToast) {
                                window.showToast('Admin account created successfully!', 'success');
                            }
                            // Re-run full initialization
                            this._runCoreInitialization();
                            this.initSSEConnectionMonitor();

                            this.$watch('activeProvider', () => {
                                this.jobsInitialized = false;
                                this.loadJobs();
                                this.loadProviderConfig();
                            });
                        }
                    } else {
                        const data = await response.json();
                        this.setupState.error = data.detail || 'Failed to create admin account';
                    }
                } catch (error) {
                    this.setupState.error = 'Network error: ' + error.message;
                } finally {
                    this.setupState.submitting = false;
                }
            },

            destroy() {
                this.stopPolling();
                this.stopInlineProgressPolling();
                this.stopHealthCheckAutoRefresh();

                // Clean up SSE connection monitor
                if (this._sseEventHandler) {
                    document.removeEventListener('trainer-connection-status', this._sseEventHandler);
                    this._sseEventHandler = null;
                }
                if (this._originalUpdateConnectionStatus) {
                    window.updateConnectionStatus = this._originalUpdateConnectionStatus;
                    this._originalUpdateConnectionStatus = null;
                }
            },

            getActiveProvider() {
                return this.providers.find(p => p.id === this.activeProvider) || null;
            },

            isActiveProviderConfigured() {
                const provider = this.getActiveProvider();
                return provider?.configured || false;
            },

            startPolling() {
                this.stopPolling();
                this.pollingInterval = setInterval(() => {
                    this.loadJobs(true);
                    this.loadMetrics();
                    this.loadPendingApprovals();
                }, 30000);
            },

            stopPolling() {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                }
            },

            startInlineProgressPolling() {
                this.stopInlineProgressPolling();
                this.inlineProgressInterval = setInterval(() => {
                    this.fetchInlineProgress();
                }, 5000);
            },

            stopInlineProgressPolling() {
                if (this.inlineProgressInterval) {
                    clearInterval(this.inlineProgressInterval);
                    this.inlineProgressInterval = null;
                }
            },

            async fetchInlineProgress() {
                const runningJobs = this.jobs.filter(j => j.status === 'running');
                if (runningJobs.length === 0) return;

                for (const job of runningJobs) {
                    try {
                        const response = await fetch(`/api/cloud/jobs/${job.job_id}/inline-progress`);
                        if (response.ok) {
                            const data = await response.json();
                            const idx = this.jobs.findIndex(j => j.job_id === job.job_id);
                            if (idx !== -1) {
                                this.jobs[idx].inline_stage = data.stage;
                                this.jobs[idx].inline_log = data.last_log;
                                this.jobs[idx].inline_progress = data.progress;
                            }
                        }
                    } catch (error) {
                        // Silently ignore
                    }
                }
            },

            updateSetupStatus() {
                const trainerStore = window.Alpine?.store('trainer');
                this.setupStatus.datasetsConfigured = trainerStore?.datasets?.length > 0;
                this.setupStatus.activeConfigExists = this.availableConfigs?.length > 0;
                this.setupStatus.outputConfigured =
                    this.publishingStatus.push_to_hub ||
                    this.publishingStatus.s3_configured ||
                    (this.webhookUrl && this.webhookUrl.trim().length > 0);
            },

            // Note: hasDatasets getter moved to final return object

            async loadDatasetStatus() {
                this.datasetStatus.loading = true;
                try {
                    const response = await fetch('/api/datasets/plan');
                    if (response.ok) {
                        const data = await response.json();
                        const datasets = data.datasets || [];
                        this.datasetStatus.configured = datasets.length > 0;
                        this.datasetStatus.count = datasets.length;
                    }
                } catch (error) {
                    console.error('Failed to load dataset status:', error);
                } finally {
                    this.datasetStatus.loading = false;
                }
            },

            // Note: hasActiveConfig, hasOutputDestination, allSetupComplete, onboardingComplete getters moved to final return object

            // --- Advanced Configuration Methods ---
            async loadAdvancedConfig() {
                this.advancedConfig.loading = true;
                try {
                    const response = await fetch(`/api/cloud/providers/${this.activeProvider}/advanced`);
                    if (response.ok) {
                        const data = await response.json();
                        this.advancedConfig.ssl_verify = data.ssl_verify !== false;
                        this.advancedConfig.ssl_ca_bundle = data.ssl_ca_bundle || '';
                        this.advancedConfig.proxy_url = data.proxy_url || '';
                        this.advancedConfig.http_timeout = data.http_timeout || 30;
                        this.advancedConfig.webhook_ip_allowlist_enabled = data.webhook_ip_allowlist_enabled || false;
                        this.advancedConfig.webhook_allowed_ips = data.webhook_allowed_ips || [];
                    }
                } catch (error) {
                    console.error('Failed to load advanced config:', error);
                } finally {
                    this.advancedConfig.loading = false;
                }
            },

            async saveAdvancedSetting(key, value) {
                this.advancedConfig.saving = true;
                try {
                    const response = await fetch(`/api/cloud/providers/${this.activeProvider}/advanced`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ [key]: value }),
                    });
                    if (response.ok) {
                        if (window.showToast) {
                            window.showToast('Setting saved', 'success');
                        }
                    } else {
                        if (window.showToast) {
                            window.showToast('Failed to save setting', 'error');
                        }
                    }
                } catch (error) {
                    console.error('Failed to save advanced setting:', error);
                    if (window.showToast) {
                        window.showToast('Failed to save setting', 'error');
                    }
                } finally {
                    this.advancedConfig.saving = false;
                }
            },

            async toggleSslVerification() {
                if (this.advancedConfig.ssl_verify) {
                    // Turning off - require acknowledgment
                    if (!this.advancedConfig.sslWarningAcknowledged) {
                        if (!confirm('Disabling SSL verification is insecure and should only be used for development. Continue?')) {
                            return;
                        }
                        this.advancedConfig.sslWarningAcknowledged = true;
                    }
                }
                const newValue = !this.advancedConfig.ssl_verify;
                this.advancedConfig.ssl_verify = newValue;
                await this.saveAdvancedSetting('ssl_verify', newValue);
            },

            async toggleIpAllowlist() {
                const newValue = !this.advancedConfig.webhook_ip_allowlist_enabled;
                this.advancedConfig.webhook_ip_allowlist_enabled = newValue;
                await this.saveAdvancedSetting('webhook_ip_allowlist_enabled', newValue);
            },

            async addIpToAllowlist() {
                const ip = this.advancedConfig.newIpEntry.trim();
                if (!ip) return;

                // Basic validation
                const ipv4Regex = /^(\d{1,3}\.){3}\d{1,3}(\/\d{1,2})?$/;
                const ipv6Regex = /^([0-9a-fA-F:]+)(\/\d{1,3})?$/;
                if (!ipv4Regex.test(ip) && !ipv6Regex.test(ip)) {
                    this.advancedConfig.ipValidationError = 'Invalid IP or CIDR format';
                    return;
                }

                if (this.advancedConfig.webhook_allowed_ips.includes(ip)) {
                    this.advancedConfig.ipValidationError = 'IP already in list';
                    return;
                }

                this.advancedConfig.ipValidationError = null;
                this.advancedConfig.webhook_allowed_ips.push(ip);
                this.advancedConfig.newIpEntry = '';
                await this.saveAdvancedSetting('webhook_allowed_ips', this.advancedConfig.webhook_allowed_ips);
            },

            async removeIpFromAllowlist(ip) {
                this.advancedConfig.webhook_allowed_ips = this.advancedConfig.webhook_allowed_ips.filter(i => i !== ip);
                await this.saveAdvancedSetting('webhook_allowed_ips', this.advancedConfig.webhook_allowed_ips);
            },

            debounceJobSearch() {
                if (this.jobSearchDebounceTimer) {
                    clearTimeout(this.jobSearchDebounceTimer);
                }
                this.jobSearchDebounceTimer = setTimeout(() => {
                    this.filterJobsDebounced();
                }, 200);
            },

            filterJobsDebounced() {
                // Triggers reactivity
            },

            handleSheetTouchStart(e) {
                const handle = e.target.closest('.sheet-handle') || e.target.classList.contains('sheet-handle');
                if (handle) {
                    this.sheetDragging = true;
                    this.sheetStartY = e.touches[0].clientY;
                    this.sheetCurrentY = 0;
                }
            },

            handleSheetTouchMove(e) {
                if (!this.sheetDragging) return;
                const deltaY = e.touches[0].clientY - this.sheetStartY;
                if (deltaY > 0) {
                    this.sheetCurrentY = deltaY;
                    this.sheetTransformStyle = `transform: translateY(${deltaY}px)`;
                }
            },

            handleSheetTouchEnd() {
                if (!this.sheetDragging) return;
                this.sheetDragging = false;
                if (this.sheetCurrentY > 100) {
                    this.selectedJob = null;
                }
                this.sheetTransformStyle = '';
                this.sheetCurrentY = 0;
            },
        };

        // Merge all modules together
        return {
            ...state,
            ...coreMethods,
            ...(window.cloudProviderMethods || {}),
            ...(window.cloudJobMethods || {}),
            ...(window.cloudSubmissionMethods || {}),
            ...(window.cloudLogMethods || {}),
            ...(window.cloudMetricsMethods || {}),
            ...(window.cloudSystemMethods || {}),
            ...(window.cloudQueueMethods || {}),
            ...(window.cloudAdminModalMethods || {}),
            ...(window.cloudOnboardingMethods || {}),
            ...(window.cloudUtilityMethods || {}),
            ...(window.cloudJobNotificationMethods || {}),
            ...(window.cloudS3ConfigMethods || {}),
            // Computed properties from utilities module
            get activeJobs() {
                return this.jobs.filter(j => ['pending', 'uploading', 'queued', 'running'].includes(j.status));
            },
            get completedJobs() {
                return this.jobs.filter(j => ['completed', 'failed', 'cancelled'].includes(j.status));
            },
            get filteredJobs() {
                let filtered = [...this.jobs];
                if (this.statusFilter) {
                    filtered = filtered.filter(j => j.status === this.statusFilter);
                }
                if (this.jobSearchQuery && this.jobSearchQuery.trim()) {
                    const query = this.jobSearchQuery.toLowerCase().trim();
                    filtered = filtered.filter(j => {
                        const jobId = (j.job_id || '').toLowerCase();
                        const configName = (j.config_name || '').toLowerCase();
                        const trackerName = (j.metadata?.tracker_run_name || '').toLowerCase();
                        const status = (j.status || '').toLowerCase();
                        const provider = (j.provider || '').toLowerCase();
                        return jobId.includes(query) || configName.includes(query) ||
                               trackerName.includes(query) || status.includes(query) ||
                               provider.includes(query);
                    });
                }
                filtered.sort((a, b) => {
                    const dateA = new Date(a.created_at || 0);
                    const dateB = new Date(b.created_at || 0);
                    return this.jobsSortOrder === 'desc' ? dateB - dateA : dateA - dateB;
                });
                return filtered;
            },
            get failedCount() {
                return this.jobs.filter(j => j.status === 'failed').length;
            },
            get runningCount() {
                return this.jobs.filter(j => j.status === 'running').length;
            },
            // Getters that were in coreMethods - moved here to avoid spread evaluation
            get sseStatusColor() {
                if (!this.sseConnection) return 'secondary';
                const statusColors = {
                    'connected': 'success',
                    'disconnected': 'danger',
                    'reconnecting': 'warning',
                    'connecting': 'info',
                };
                return statusColors[this.sseConnection.status] || 'secondary';
            },
            get sseStatusIcon() {
                if (!this.sseConnection) return 'fa-plug';
                const statusIcons = {
                    'connected': 'fa-plug',
                    'disconnected': 'fa-plug-circle-xmark',
                    'reconnecting': 'fa-plug-circle-exclamation',
                    'connecting': 'fa-plug-circle-bolt',
                };
                return statusIcons[this.sseConnection.status] || 'fa-plug';
            },
            get setupFormValid() {
                if (!this.setupState?.form) return false;
                const form = this.setupState.form;
                return (
                    form.email &&
                    form.email.includes('@') &&
                    form.username &&
                    form.username.length >= 3 &&
                    form.password &&
                    form.password.length >= 8 &&
                    form.password === form.confirmPassword
                );
            },
            get hasDatasets() {
                if (!this.datasetStatus) return false;
                if (this.datasetStatus.configured) return true;
                const trainerStore = window.Alpine?.store('trainer');
                return trainerStore?.datasets?.length > 0;
            },
            get hasActiveConfig() {
                return this.availableConfigs?.length > 0;
            },
            get hasOutputDestination() {
                if (!this.publishingStatus) return false;
                return this.publishingStatus.push_to_hub ||
                       this.publishingStatus.s3_configured ||
                       (this.webhookUrl && this.webhookUrl.trim().length > 0);
            },
            get allSetupComplete() {
                return this.hasDatasets && this.hasActiveConfig && this.hasOutputDestination;
            },
            get onboardingComplete() {
                if (!this.onboarding) return false;
                return this.onboarding.data_understood &&
                       this.onboarding.results_understood &&
                       this.onboarding.cost_understood;
            },
            // From utilities.js - wizard getters
            get wizardRequiresUpload() {
                return this.preSubmitModal?.dataUploadPreview?.requires_upload || false;
            },
            get wizardTotalSteps() {
                return this.wizardRequiresUpload ? 3 : 2;
            },
            get wizardDataConsentBlocked() {
                return this.wizardRequiresUpload &&
                       this.preSubmitModal?.dataConsent === 'ask' &&
                       !this.preSubmitModal?.dataConsentConfirmed;
            },
            get wizardUploadDenied() {
                return this.wizardRequiresUpload && this.preSubmitModal?.dataConsent === 'deny';
            },
            get wizardCanSubmit() {
                if (this.preSubmitModal?.loading || this.submitting) return false;
                if (this.wizardDataConsentBlocked) return false;
                if (this.wizardUploadDenied) return false;
                const wc = this.preSubmitModal?.webhookCheck;
                if (wc && this.webhookUrl && wc.tested && !wc.success && !wc.skipped) return false;
                // Block if any quota with action="block" would be exceeded
                const qi = this.preSubmitModal?.quotaImpact;
                if (qi?.impacts?.some(i => i.would_exceed && i.action === 'block')) return false;
                return true;
            },
            get wizardCostDisplay() {
                const estimate = this.preSubmitModal?.costEstimate;
                if (!estimate?.has_estimate) return 'N/A';
                return '~$' + estimate.estimated_cost_usd.toFixed(2);
            },
            get wizardHardwareCostDisplay() {
                const costPerHour = this.preSubmitModal?.costEstimate?.hardware_cost_per_hour;
                return costPerHour ? ('$' + costPerHour.toFixed(2) + '/hr') : 'N/A';
            },
            // From queue.js - queue getters
            get visiblePendingJobs() {
                if (this.hasAdminAccess) {
                    return this.queuePendingJobs || [];
                }
                const userId = this.currentUser?.id;
                return (this.queuePendingJobs || []).map(job => ({
                    ...job,
                    is_own: job.user_id === userId,
                })).filter(job => job.is_own);
            },
            get visibleBlockedJobs() {
                if (this.hasAdminAccess) {
                    return this.queueBlockedJobs || [];
                }
                const userId = this.currentUser?.id;
                return (this.queueBlockedJobs || []).map(job => ({
                    ...job,
                    is_own: job.user_id === userId,
                })).filter(job => job.is_own);
            },
            get userQueuePosition() {
                if (this.hasAdminAccess) return null;
                const userId = this.currentUser?.id;
                const ownJob = (this.queuePendingJobs || []).find(job => job.user_id === userId);
                return ownJob?.position || null;
            },
            get userEstimatedWait() {
                if (!this.userQueuePosition || !this.queueStats?.avg_wait_seconds) return null;
                return this.userQueuePosition * this.queueStats.avg_wait_seconds;
            },
            // From job-notifications.js - notification settings
            get notificationsEnabled() {
                const stored = localStorage.getItem('cloud_notifications_enabled');
                return stored !== 'false';
            },
            set notificationsEnabled(value) {
                localStorage.setItem('cloud_notifications_enabled', value ? 'true' : 'false');
            },
            get notificationSoundEnabled() {
                const stored = localStorage.getItem('cloud_notification_sound_enabled');
                return stored === 'true';
            },
            set notificationSoundEnabled(value) {
                localStorage.setItem('cloud_notification_sound_enabled', value ? 'true' : 'false');
            },
            // Localhost detection for webhook configuration
            get isLocalhost() {
                return ['localhost', '127.0.0.1', '[::1]'].some(
                    h => window.location.hostname === h || window.location.hostname.endsWith('.localhost')
                );
            },
        };
        } catch (err) {
            console.error('cloudDashboardComponent construction failed:', err);
            // Return minimal fallback to prevent Alpine errors
            return {
                setupState: { loading: false, needsSetup: false, error: 'Component failed to load: ' + err.message, form: {} },
                sseConnection: { status: 'unknown', message: '', lastUpdated: null },
                jobs: [],
                providers: [],
                init() { console.error('Cloud dashboard failed to initialize'); },
            };
        }
    };
}
