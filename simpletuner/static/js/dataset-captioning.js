/**
 * Dataset Captioning Alpine.js component.
 *
 * Shows CaptionFlow availability and install guidance for the Dataset page.
 */
window.datasetCaptioningComponent = function () {
    return {
        loading: false,
        submitting: false,
        error: '',
        submitMessage: '',
        installCommand: "pip install 'simpletuner[captioning]'",
        datasets: [],
        selectedDatasetId: '',
        captionJobs: [],
        activeJobId: '',
        jobLogs: '',
        jobStatusError: '',
        statusPollTimer: null,
        autoScrollLogs: true,
        viewMode: 'builder',
        rawConfig: `orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10
  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
      repetition_penalty: 1.05
      skip_special_tokens: true
      stop:
        - "<|end|>"
        - "<|endoftext|>"
        - "<|im_end|>"
    stages:
      - name: "base_caption"
        model: "Qwen/Qwen2.5-VL-3B-Instruct"
        prompts:
          - "describe this image in detail"
        output_field: "caption"
      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to ONLY the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35`,
        form: {
            model: 'Qwen/Qwen2.5-VL-3B-Instruct',
            prompt: "You're a caption bot designed to output image descriptions. Describe what you see. Output only the caption.",
            output_field: 'captions',
            worker_count: 1,
            batch_size: 8,
            chunk_size: 256,
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.95,
            gpu_memory_utilization: 0.92,
            export_textfiles: true,
            any_gpu: true,
            no_wait: false,
        },
        capabilities: {
            installed: false,
            ready: false,
            install_command: "pip install 'simpletuner[captioning]'",
        },

        init() {
            this.loadCapabilities();
            this.startStatusPolling();
        },

        destroy() {
            if (this.statusPollTimer) {
                clearInterval(this.statusPollTimer);
                this.statusPollTimer = null;
            }
        },

        async loadCapabilities() {
            this.loading = true;
            this.error = '';
            try {
                const response = await fetch('/api/datasets/captioning/capabilities');
                if (!response.ok) {
                    const detail = await response.json().catch(() => ({}));
                    throw new Error(detail.detail || 'Failed to check captioning dependencies');
                }
                const data = await response.json();
                this.capabilities = {
                    ...this.capabilities,
                    ...data,
                };
                if (this.capabilities.ready) {
                    await this.loadDatasets();
                    await this.loadCaptionJobs();
                }
            } catch (err) {
                this.error = err.message || 'Failed to check captioning dependencies';
            } finally {
                this.loading = false;
            }
        },

        startStatusPolling() {
            if (this.statusPollTimer) {
                clearInterval(this.statusPollTimer);
            }
            this.statusPollTimer = setInterval(() => {
                if (this.capabilities.ready) {
                    this.loadCaptionJobs();
                }
            }, 3000);
        },

        async loadCaptionJobs() {
            this.jobStatusError = '';
            try {
                const response = await fetch('/api/cloud/jobs?provider=captionflow&limit=10');
                if (!response.ok) {
                    const detail = await response.json().catch(() => ({}));
                    throw new Error(detail.detail || 'Failed to load captioning jobs');
                }
                const data = await response.json();
                const jobs = Array.isArray(data.jobs) ? data.jobs : [];
                this.captionJobs = jobs;
                if (!this.activeJobId && jobs.length > 0) {
                    const active = jobs.find((job) => ['running', 'queued', 'pending'].includes(String(job.status || '').toLowerCase()));
                    this.activeJobId = (active || jobs[0]).job_id;
                }
                if (this.activeJobId && !jobs.some((job) => job.job_id === this.activeJobId)) {
                    this.activeJobId = jobs[0]?.job_id || '';
                }
                if (this.activeJobId) {
                    await this.loadJobLogs(this.activeJobId);
                }
            } catch (err) {
                this.jobStatusError = err.message || 'Failed to load captioning jobs';
            }
        },

        async loadJobLogs(jobId) {
            if (!jobId) {
                this.jobLogs = '';
                return;
            }
            const shouldScroll = this.shouldAutoScrollLogs();
            try {
                const response = await fetch(`/api/cloud/jobs/${jobId}/logs`);
                if (!response.ok) {
                    const detail = await response.json().catch(() => ({}));
                    throw new Error(detail.detail || 'Failed to load captioning logs');
                }
                const data = await response.json();
                this.jobLogs = String(data.logs || '');
            } catch (err) {
                this.jobLogs = err.message || 'Failed to load captioning logs';
            }
            this.scrollLogsAfterUpdate(shouldScroll);
        },

        selectCaptionJob(jobId) {
            this.activeJobId = jobId;
            this.autoScrollLogs = true;
            this.loadJobLogs(jobId);
        },

        shouldAutoScrollLogs() {
            const viewer = this.$refs.captioningLogViewer;
            if (!viewer) return true;
            const distanceFromBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight;
            return this.autoScrollLogs && distanceFromBottom < 48;
        },

        handleLogScroll() {
            const viewer = this.$refs.captioningLogViewer;
            if (!viewer) return;
            const distanceFromBottom = viewer.scrollHeight - viewer.scrollTop - viewer.clientHeight;
            this.autoScrollLogs = distanceFromBottom < 48;
        },

        scrollLogsAfterUpdate(shouldScroll) {
            if (!shouldScroll) return;
            this.$nextTick(() => {
                const viewer = this.$refs.captioningLogViewer;
                if (viewer) {
                    viewer.scrollTop = viewer.scrollHeight;
                }
            });
        },

        activeCaptionJob() {
            return this.captionJobs.find((job) => job.job_id === this.activeJobId) || null;
        },

        captionJobStatusClass(status) {
            const normalized = String(status || '').toLowerCase();
            if (['running'].includes(normalized)) return 'bg-info text-dark';
            if (['queued', 'pending'].includes(normalized)) return 'bg-warning text-dark';
            if (['completed', 'success'].includes(normalized)) return 'bg-success';
            if (['failed', 'error', 'cancelled'].includes(normalized)) return 'bg-danger';
            return 'bg-secondary';
        },

        captionJobLabel(job) {
            const id = String(job?.job_id || '').slice(0, 8);
            const name = job?.config_name || 'Captioning';
            return id ? `${name} (${id})` : name;
        },

        formatJobTime(value) {
            if (!value) return '';
            const timestamp = Date.parse(value);
            if (Number.isNaN(timestamp)) return String(value);
            return new Date(timestamp).toLocaleString();
        },

        async loadDatasets() {
            const response = await fetch('/api/datasets/viewer/summaries');
            if (!response.ok) {
                const detail = await response.json().catch(() => ({}));
                throw new Error(detail.detail || 'Failed to load datasets');
            }
            const datasets = await response.json();
            this.datasets = datasets.filter((dataset) => {
                const type = String(dataset.config?.dataset_type || 'image').toLowerCase();
                return !['text_embeds', 'image_embeds', 'audio'].includes(type);
            });
            if (!this.selectedDatasetId && this.datasets.length > 0) {
                this.selectedDatasetId = this.datasets[0].dataset_id;
            }
            this.syncExportModeForDataset();
        },

        selectedDataset() {
            return this.datasets.find((dataset) => dataset.dataset_id === this.selectedDatasetId) || null;
        },

        selectedDatasetBackendType() {
            const dataset = this.selectedDataset();
            return String(dataset?.config?.type || 'local').toLowerCase();
        },

        selectedDatasetSupportsTextfileExport() {
            return this.selectedDatasetBackendType() === 'local';
        },

        syncExportModeForDataset() {
            if (!this.selectedDatasetSupportsTextfileExport()) {
                this.form.export_textfiles = false;
            }
        },

        async startCaptioningJob() {
            this.error = '';
            this.submitMessage = '';

            if (!this.selectedDatasetId) {
                this.error = 'Select a dataset to caption';
                return;
            }
            if (this.viewMode === 'raw' && !String(this.rawConfig || '').trim()) {
                this.error = 'Raw CaptionFlow config is required';
                return;
            }

            this.submitting = true;
            try {
                const payload = {
                    dataset_id: this.selectedDatasetId,
                    raw_config: this.viewMode === 'raw' ? this.rawConfig : null,
                    export_textfiles: this.selectedDatasetSupportsTextfileExport()
                        ? Boolean(this.form.export_textfiles)
                        : false,
                    worker_count: Number(this.form.worker_count),
                    output_field: this.form.output_field,
                    any_gpu: Boolean(this.form.any_gpu),
                    no_wait: Boolean(this.form.no_wait),
                };
                if (this.viewMode === 'builder') {
                    Object.assign(payload, {
                        model: this.form.model,
                        prompt: this.form.prompt,
                        batch_size: Number(this.form.batch_size),
                        chunk_size: Number(this.form.chunk_size),
                        max_tokens: Number(this.form.max_tokens),
                        temperature: Number(this.form.temperature),
                        top_p: Number(this.form.top_p),
                        gpu_memory_utilization: Number(this.form.gpu_memory_utilization),
                    });
                }
                const response = await fetch('/api/datasets/captioning/jobs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || 'Failed to start captioning job');
                }

                if (data.status === 'queued') {
                    this.submitMessage = `Captioning job ${data.job_id} queued at position ${data.queue_position}.`;
                } else if (data.status === 'running') {
                    const gpuText = Array.isArray(data.allocated_gpus) && data.allocated_gpus.length
                        ? ` on GPU ${data.allocated_gpus.join(', ')}`
                        : '';
                    this.submitMessage = `Captioning job ${data.job_id} started${gpuText}.`;
                } else {
                    this.submitMessage = data.reason || `Captioning job status: ${data.status}`;
                }
                if (data.job_id) {
                    this.activeJobId = data.job_id;
                    await this.loadCaptionJobs();
                }

                if (window.showToast) {
                    window.showToast(this.submitMessage, data.status === 'rejected' ? 'warning' : 'success');
                }
            } catch (err) {
                this.error = err.message || 'Failed to start captioning job';
                if (window.showToast) {
                    window.showToast(this.error, 'error');
                }
            } finally {
                this.submitting = false;
            }
        },

        async copyInstallCommand() {
            const command = this.capabilities.install_command || this.installCommand;
            try {
                if (!navigator.clipboard || typeof navigator.clipboard.writeText !== 'function') {
                    throw new Error('Clipboard is unavailable');
                }
                await navigator.clipboard.writeText(command);
                if (window.showToast) {
                    window.showToast('Install command copied', 'success');
                }
            } catch (err) {
                if (window.showToast) {
                    window.showToast(err.message || 'Failed to copy install command', 'error');
                }
            }
        },
    };
};
