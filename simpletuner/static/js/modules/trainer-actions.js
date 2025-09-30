/**
 * Trainer Actions Module
 * Handles training start, stop, and cancel operations
 */

export class TrainerActions {
    constructor(apiBaseUrl, callbackUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.callbackUrl = callbackUrl;
    }

    /**
     * Start training
     */
    async startTraining(payload) {
        const response = await fetch(`${this.apiBaseUrl}/training/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/html'
            },
            credentials: 'include',
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Failed to start training: ${response.status}`);
        }

        const html = await response.text();

        // Check if we need to initialize SSE
        if (response.ok && window.initSSE) {
            window.initSSE();
        }

        return html;
    }

    /**
     * Cancel training
     */
    async cancelTraining(jobId) {
        const formData = new FormData();
        formData.append('job_id', jobId);

        const response = await fetch(`${this.apiBaseUrl}/training/cancel`, {
            method: 'POST',
            credentials: 'include',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Failed to cancel training: ${response.status}`);
        }

        const html = await response.text();

        // Close SSE connection if exists
        if (response.ok && window.closeSSE) {
            window.closeSSE();
        }

        return html;
    }

    /**
     * Get training status
     */
    async getTrainingStatus() {
        const response = await fetch(`${this.apiBaseUrl}/training/status`, {
            method: 'GET',
            credentials: 'include'
        });

        if (!response.ok) {
            throw new Error(`Failed to get status: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Update button states based on training status
     */
    updateButtonStates(isTraining) {
        const validateBtn = document.getElementById('validateBtn');
        const runBtn = document.getElementById('runBtn');
        const cancelBtn = document.getElementById('cancelBtn');

        if (isTraining) {
            if (validateBtn) validateBtn.disabled = true;
            if (runBtn) runBtn.disabled = true;
            if (cancelBtn) cancelBtn.disabled = false;
        } else {
            if (validateBtn) validateBtn.disabled = false;
            if (runBtn) runBtn.disabled = false;
            if (cancelBtn) cancelBtn.disabled = true;
        }
    }

    /**
     * Set button loading state
     */
    setButtonLoading(button, loading, html) {
        if (!button) return;

        button.disabled = loading;

        if (loading) {
            button.dataset.originalHtml = button.innerHTML;
            button.innerHTML = `<span class="spinner-border spinner-border-sm me-2"></span>${html}`;
        } else {
            button.innerHTML = button.dataset.originalHtml || html;
        }
    }
}