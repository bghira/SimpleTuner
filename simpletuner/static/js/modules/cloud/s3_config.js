/**
 * Cloud Dashboard - S3 Configuration Module
 *
 * Handles S3/cloud storage configuration for model delivery.
 */

window.cloudS3ConfigMethods = {
    // S3 Config state - should be merged into main component state
    s3Config: {
        preset: 'aws',
        endpoint_url: '',
        region: '',
        bucket: '',
        access_key_id: '',
        secret_access_key: '',
        path_prefix: '',
        use_path_style: false,
        public_read: false,
        configured: false,
        showAdvanced: false,
        testing: false,
        saving: false,
        testResult: null,
    },

    initS3Config() {
        this.loadS3Config();
    },

    applyS3Preset(preset) {
        this.s3Config.preset = preset;
        this.s3Config.testResult = null;

        switch (preset) {
            case 'aws':
                this.s3Config.endpoint_url = '';
                this.s3Config.use_path_style = false;
                break;
            case 'backblaze':
                this.s3Config.endpoint_url = 'https://s3.us-west-000.backblazeb2.com';
                this.s3Config.use_path_style = true;
                break;
            case 'minio':
                this.s3Config.endpoint_url = 'http://localhost:9000';
                this.s3Config.use_path_style = true;
                break;
            case 'custom':
                this.s3Config.endpoint_url = '';
                break;
        }
    },

    async loadS3Config() {
        try {
            const response = await fetch('/api/cloud/providers/replicate/config');
            if (response.ok) {
                const data = await response.json();
                const s3 = data.config?.s3_delivery || {};

                if (s3.bucket) {
                    this.s3Config.configured = true;
                    this.s3Config.bucket = s3.bucket || '';
                    this.s3Config.region = s3.region || '';
                    this.s3Config.endpoint_url = s3.endpoint_url || '';
                    this.s3Config.path_prefix = s3.path_prefix || '';
                    this.s3Config.use_path_style = s3.use_path_style || false;
                    this.s3Config.public_read = s3.public_read || false;

                    // Determine preset
                    if (!s3.endpoint_url) {
                        this.s3Config.preset = 'aws';
                    } else if (s3.endpoint_url.includes('backblazeb2.com')) {
                        this.s3Config.preset = 'backblaze';
                    } else if (s3.endpoint_url.includes('localhost') || s3.endpoint_url.includes('minio')) {
                        this.s3Config.preset = 'minio';
                    } else {
                        this.s3Config.preset = 'custom';
                    }
                }

                // Update publishing status for onboarding
                if (this.publishingStatus) {
                    this.publishingStatus.s3_configured = this.s3Config.configured;
                }
            }
        } catch (error) {
            console.error('Failed to load S3 config:', error);
        }
    },

    async testS3Connection() {
        this.s3Config.testing = true;
        this.s3Config.testResult = null;

        try {
            const response = await fetch('/api/cloud/providers/replicate/test-s3', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    bucket: this.s3Config.bucket,
                    region: this.s3Config.region,
                    endpoint_url: this.s3Config.endpoint_url,
                    access_key_id: this.s3Config.access_key_id,
                    secret_access_key: this.s3Config.secret_access_key,
                    use_path_style: this.s3Config.use_path_style,
                }),
            });

            const data = await response.json();
            if (response.ok && data.success) {
                this.s3Config.testResult = {
                    success: true,
                    message: `Connection successful! Bucket "${this.s3Config.bucket}" is accessible.`,
                };
            } else {
                this.s3Config.testResult = {
                    success: false,
                    message: data.detail || data.error || 'Connection failed. Check your credentials.',
                };
            }
        } catch (error) {
            console.error('Failed to test S3 connection:', error);
            this.s3Config.testResult = {
                success: false,
                message: 'Connection test failed. Please check your settings.',
            };
        } finally {
            this.s3Config.testing = false;
        }
    },

    async saveS3Config() {
        this.s3Config.saving = true;

        try {
            const payload = {
                s3_delivery: {
                    bucket: this.s3Config.bucket,
                    region: this.s3Config.region,
                    endpoint_url: this.s3Config.endpoint_url || null,
                    path_prefix: this.s3Config.path_prefix || '',
                    use_path_style: this.s3Config.use_path_style,
                    public_read: this.s3Config.public_read,
                },
            };

            // Only include credentials if they were changed
            if (this.s3Config.access_key_id) {
                payload.s3_delivery.access_key_id = this.s3Config.access_key_id;
            }
            if (this.s3Config.secret_access_key) {
                payload.s3_delivery.secret_access_key = this.s3Config.secret_access_key;
            }

            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            if (response.ok) {
                this.s3Config.configured = true;
                this.s3Config.access_key_id = '';
                this.s3Config.secret_access_key = '';

                // Update publishing status for onboarding
                if (this.publishingStatus) {
                    this.publishingStatus.s3_configured = true;
                }

                // Close modal
                if (this.$refs.s3ConfigModal) {
                    const modal = bootstrap.Modal.getInstance(this.$refs.s3ConfigModal);
                    if (modal) modal.hide();
                }

                if (window.showToast) {
                    window.showToast('S3 configuration saved', 'success');
                }
            } else {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(data.detail || 'Failed to save configuration', 'error');
                }
            }
        } catch (error) {
            console.error('Failed to save S3 config:', error);
            if (window.showToast) {
                window.showToast('Failed to save configuration', 'error');
            }
        } finally {
            this.s3Config.saving = false;
        }
    },

    showS3ConfigModal() {
        this.s3Config.testResult = null;
        if (this.$refs.s3ConfigModal) {
            const modal = new bootstrap.Modal(this.$refs.s3ConfigModal);
            modal.show();
        }
    },
};
