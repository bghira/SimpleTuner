/**
 * Cloud Dashboard - Job Submission Module
 *
 * Handles pre-submit modal, payload preparation, and job submission.
 */

window.cloudSubmissionMethods = {
    /**
     * Reset pre-submit modal to initial state
     */
    resetPreSubmitModal() {
        this.preSubmitModal.open = true;
        this.preSubmitModal.wizardStep = 1;
        this.preSubmitModal.loading = true;
        this.preSubmitModal.snapshotMessage = '';
        this.preSubmitModal.dataUploadPreview = null;
        this.preSubmitModal.dataConsentConfirmed = false;
        this.preSubmitModal.webhookCheck = {
            tested: false,
            testing: false,
            success: null,
            error: null,
            skipped: false,
        };
    },

    /**
     * Apply pre-submit check response to modal state
     */
    applyPreSubmitData(data) {
        this.preSubmitModal.gitAvailable = data.git_available;
        this.preSubmitModal.repoPresent = data.repo_present;
        this.preSubmitModal.isDirty = data.is_dirty;
        this.preSubmitModal.dirtyCount = data.dirty_count;
        this.preSubmitModal.dirtyPaths = data.dirty_paths || [];
        this.preSubmitModal.currentCommit = data.current_commit;
        this.preSubmitModal.currentAbbrev = data.current_abbrev;
        this.preSubmitModal.currentBranch = data.current_branch;
        this.preSubmitModal.trackerRunName = data.tracker_run_name || '';
        this.preSubmitModal.configName = data.config_name || '';
        this.preSubmitModal.snapshotName = '';
    },

    async openPreSubmitModal() {
        if (!this.isActiveProviderConfigured()) {
            const provider = this.getActiveProvider();
            if (window.showToast) {
                window.showToast(`Please configure ${provider?.name || this.activeProvider} first`, 'error');
            }
            return;
        }

        this.resetPreSubmitModal();

        try {
            const [preSubmitResp, consentResp] = await Promise.all([
                fetch('/api/cloud/pre-submit-check'),
                fetch('/api/cloud/data-consent/setting'),
            ]);

            if (preSubmitResp.ok) {
                this.applyPreSubmitData(await preSubmitResp.json());
            }

            if (consentResp.ok) {
                const consentData = await consentResp.json();
                this.preSubmitModal.dataConsent = consentData.consent;
                if (consentData.consent === 'allow') {
                    this.preSubmitModal.dataConsentConfirmed = true;
                }
            }

            await Promise.all([
                this.loadDataUploadPreview(),
                this.loadCostEstimate(),
                this.loadConfigPreview(),
                this.checkWebhookReachability(),
            ]);

        } catch (error) {
            console.error('Failed to load pre-submit check:', error);
        } finally {
            this.preSubmitModal.loading = false;
        }
    },

    async loadDataUploadPreview() {
        try {
            const dataloader = window.Alpine?.store('trainer')?.datasets || [];
            if (dataloader.length === 0) {
                this.preSubmitModal.dataUploadPreview = {
                    requires_upload: false,
                    consent_mode: this.preSubmitModal.dataConsent,
                    datasets: [],
                    total_files: 0,
                    total_size_mb: 0,
                    message: 'No datasets configured.',
                };
                return;
            }

            const response = await fetch('/api/cloud/data-consent/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataloader),
            });

            if (response.ok) {
                this.preSubmitModal.dataUploadPreview = await response.json();
                if (!this.preSubmitModal.dataUploadPreview.requires_upload) {
                    this.preSubmitModal.dataConsentConfirmed = true;
                }
            }
        } catch (error) {
            console.error('Failed to load data upload preview:', error);
        }
    },

    async loadCostEstimate() {
        try {
            const dataloader = window.Alpine?.store('trainer')?.datasets || [];
            const response = await fetch('/api/cloud/cost-estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataloader),
            });
            if (response.ok) {
                this.preSubmitModal.costEstimate = await response.json();
                // Also load quota impact if we have a cost estimate
                if (this.preSubmitModal.costEstimate?.estimated_cost) {
                    await this.loadQuotaImpact(this.preSubmitModal.costEstimate.estimated_cost);
                }
            }
        } catch (error) {
            console.error('Failed to load cost estimate:', error);
        }
    },

    async loadQuotaImpact(estimatedCost) {
        try {
            const response = await fetch(`/api/quotas/cost-estimate?estimated_cost=${estimatedCost}`, {
                method: 'POST',
            });
            if (response.ok) {
                this.preSubmitModal.quotaImpact = await response.json();
            }
        } catch (error) {
            // Non-critical - just log and continue
            console.debug('Failed to load quota impact:', error);
        }
    },

    async loadConfigPreview() {
        try {
            const dataloader = window.Alpine?.store('trainer')?.datasets || [];
            this.preSubmitModal.dataloaderPreview = dataloader;

            if (this.selectedConfigName) {
                const response = await fetch(`/api/configs/${encodeURIComponent(this.selectedConfigName)}`);
                if (response.ok) {
                    const data = await response.json();
                    this.preSubmitModal.configPreview = data.config || {};
                }
            } else {
                const response = await fetch('/api/configs/active');
                if (response.ok) {
                    const data = await response.json();
                    this.preSubmitModal.configPreview = data.config || {};
                }
            }
        } catch (error) {
            console.error('Failed to load config preview:', error);
            this.preSubmitModal.configPreview = { error: 'Failed to load config' };
        }
    },

    async checkWebhookReachability() {
        // Skip if no webhook configured
        if (!this.webhookUrl) {
            this.preSubmitModal.webhookCheck.tested = true;
            this.preSubmitModal.webhookCheck.skipped = true;
            return;
        }

        this.preSubmitModal.webhookCheck.testing = true;
        this.preSubmitModal.webhookCheck.error = null;

        try {
            const fullUrl = this.webhookUrl.replace(/\/$/, '') + '/api/cloud/webhook/replicate';
            const response = await fetch('/api/cloud/webhook/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    webhook_url: fullUrl,
                    provider: this.activeProvider || 'replicate',
                    method: 'direct',  // Quick internal test
                }),
            });

            if (response.ok) {
                const result = await response.json();
                this.preSubmitModal.webhookCheck.success = result.success;
                this.preSubmitModal.webhookCheck.error = result.error || null;
            } else {
                const data = await response.json();
                this.preSubmitModal.webhookCheck.success = false;
                this.preSubmitModal.webhookCheck.error = data.detail || 'Webhook test failed';
            }
        } catch (error) {
            this.preSubmitModal.webhookCheck.success = false;
            this.preSubmitModal.webhookCheck.error = 'Network error: ' + error.message;
        } finally {
            this.preSubmitModal.webhookCheck.tested = true;
            this.preSubmitModal.webhookCheck.testing = false;
        }
    },

    skipWebhookCheck() {
        this.preSubmitModal.webhookCheck.skipped = true;
    },

    async retryWebhookCheck() {
        this.preSubmitModal.webhookCheck.tested = false;
        this.preSubmitModal.webhookCheck.skipped = false;
        await this.checkWebhookReachability();
    },

    closePreSubmitModal() {
        this.preSubmitModal.open = false;
        this.preSubmitModal.wizardStep = 1;
    },

    saveQuickSubmitMode(enabled) {
        localStorage.setItem('cloud_quick_submit_mode', enabled ? 'true' : 'false');
    },

    startUploadProgress(uploadId) {
        this.uploadProgress.active = true;
        this.uploadProgress.stage = 'scanning';
        this.uploadProgress.current = 0;
        this.uploadProgress.total = 0;
        this.uploadProgress.percent = 0;
        this.uploadProgress.message = 'Starting...';
        this.uploadProgress.error = null;

        if (this.uploadProgress.eventSource) {
            this.uploadProgress.eventSource.close();
        }

        const eventSource = new EventSource(`/api/webhooks/upload/progress/${uploadId}`);
        this.uploadProgress.eventSource = eventSource;

        eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.uploadProgress.stage = data.stage;
                this.uploadProgress.current = data.current || 0;
                this.uploadProgress.total = data.total || 0;
                this.uploadProgress.percent = data.percent || 0;
                this.uploadProgress.message = data.message || `${data.stage}...`;

                if (data.error) {
                    this.uploadProgress.error = data.error;
                    this.uploadProgress.active = false;
                    eventSource.close();
                } else if (data.done) {
                    this.uploadProgress.active = false;
                    eventSource.close();
                }
            } catch (e) {
                console.error('Error parsing upload progress:', e);
            }
        };

        eventSource.onerror = () => {
            this.uploadProgress.active = false;
            eventSource.close();
        };
    },

    stopUploadProgress() {
        if (this.uploadProgress.eventSource) {
            this.uploadProgress.eventSource.close();
            this.uploadProgress.eventSource = null;
        }
        this.uploadProgress.active = false;
    },

    dismissUploadError() {
        this.uploadProgress.error = null;
        this.uploadProgress.active = false;
        this.uploadProgress.stage = '';
        this.uploadProgress.current = 0;
        this.uploadProgress.total = 0;
        this.uploadProgress.percent = 0;
        this.uploadProgress.message = '';
    },

    async retryUpload() {
        // Clear the error and re-open the pre-submit modal
        this.uploadProgress.error = null;
        this.uploadProgress.active = false;

        // Re-open the modal so user can retry submission
        await this.openPreSubmitModal();
    },

    generateUploadId() {
        return crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substr(2, 9);
    },

    buildBasePayload(uploadId) {
        return {
            webhook_url: this.webhookUrl || null,
            snapshot_name: this.preSubmitModal.snapshotName || null,
            snapshot_message: this.preSubmitModal.snapshotMessage || null,
            tracker_run_name: this.preSubmitModal.trackerRunName || null,
            upload_id: uploadId,
        };
    },

    async fetchCurrentConfig() {
        let config = {};
        let dataloaderConfig = [];
        let configName = null;

        const configResp = await fetch('/api/configs/active');
        if (configResp.ok) {
            const configData = await configResp.json();
            config = configData.config || {};
            configName = configData.name || null;
        }

        const dataloaderResp = await fetch('/api/datasets/plan');
        if (dataloaderResp.ok) {
            const dataloaderData = await dataloaderResp.json();
            dataloaderConfig = dataloaderData.datasets || [];
        }

        return { config, dataloaderConfig, configName };
    },

    async prepareJobPayload(uploadId) {
        const payload = this.buildBasePayload(uploadId);

        if (this.selectedConfigName) {
            payload.config_name_to_load = this.selectedConfigName;
            return { success: true, payload };
        }

        try {
            const { config, dataloaderConfig, configName } = await this.fetchCurrentConfig();

            if (Object.keys(config).length === 0) {
                return {
                    success: false,
                    error: 'No active configuration found. Please save your config first.',
                };
            }

            payload.config = config;
            payload.dataloader_config = dataloaderConfig;
            payload.config_name = configName;
            return { success: true, payload };
        } catch (error) {
            return {
                success: false,
                error: 'Failed to fetch current configuration',
            };
        }
    },

    async submitJobToProvider(payload) {
        const response = await fetch('/api/cloud/jobs/submit?provider=replicate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        return await response.json();
    },

    handleSubmissionSuccess(data) {
        if (window.showToast) {
            let msg;
            let toastType;

            if (data.idempotent_hit) {
                // Duplicate request detected via idempotency key
                msg = `Duplicate detected: Job ${data.job_id} was already submitted`;
                toastType = 'info';
            } else {
                msg = `Job submitted: ${data.job_id}`;
                if (data.data_uploaded) {
                    msg += ' (data uploaded)';
                }
                toastType = 'success';
            }

            window.showToast(msg, toastType);

            if (data.cost_limit_warning) {
                window.showToast(data.cost_limit_warning, 'warning');
            }
        }

        this.loadJobs();

        if (this.providerConfig.cost_limit_enabled) {
            this.loadCostLimitStatus();
        }
    },

    handleSubmissionError(error) {
        this.submitError = typeof error === 'string' ? error : (error.message || 'Failed to submit job');
        if (window.showToast) {
            window.showToast(this.submitError, 'error');
        }
    },

    async submitCloudJob() {
        if (this.submitting) return;

        this.submitting = true;
        this.submitError = null;
        this.closePreSubmitModal();

        const uploadId = this.generateUploadId();

        // Start upload progress if needed
        const needsUpload = this.preSubmitModal.dataUploadPreview?.requires_upload;
        if (needsUpload) {
            this.startUploadProgress(uploadId);
        }

        // Prepare payload
        const prepared = await this.prepareJobPayload(uploadId);
        if (!prepared.success) {
            this.handleSubmissionError(prepared.error);
            this.submitting = false;
            this.stopUploadProgress();
            return;
        }

        // Submit to provider
        try {
            const data = await this.submitJobToProvider(prepared.payload);

            if (data.success) {
                this.handleSubmissionSuccess(data);
            } else {
                this.handleSubmissionError(data.error || 'Failed to submit job');
            }
        } catch (error) {
            this.handleSubmissionError(error);
        } finally {
            this.submitting = false;
            this.stopUploadProgress();
        }
    },
};
