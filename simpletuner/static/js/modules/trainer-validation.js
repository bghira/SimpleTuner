/**
 * Trainer Validation Module
 * Handles form validation and validation API calls
 */

export class TrainerValidation {
    constructor(apiBaseUrl, callbackUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.callbackUrl = callbackUrl;
    }

    /**
     * Get form payload and validate
     */
    getPayload(form) {
        const formData = new FormData(form);
        const payload = {
            trainer_config: {},
            dataloader_config: [],
            webhook_config: {
                webhook_type: "raw",
                callback_url: `${this.callbackUrl}/callback`
            },
            job_id: formData.get('job_id')
        };

        // Process form data
        for (const [key, value] of formData.entries()) {
            if (key.startsWith('--')) {
                const element = document.querySelector(`[name="${key}"]`);
                if (element?.type === 'checkbox') {
                    if (element.checked) {
                        payload.trainer_config[key] = 'true';
                    }
                } else if (value) {
                    payload.trainer_config[key] = value;
                }
            } else if (key === 'dataloader_config') {
                try {
                    payload.dataloader_config = JSON.parse(value);
                } catch (error) {
                    throw new Error('Invalid JSON for Dataloader Config');
                }
            } else if (key === 'webhook_config') {
                try {
                    const webhooksConfig = JSON.parse(value);
                    payload.webhook_config = { ...payload.webhook_config, ...webhooksConfig };
                } catch (error) {
                    console.warn('Invalid JSON for Webhooks Config, using defaults');
                }
            }
        }

        return payload;
    }

    /**
     * Validate configuration
     */
    async validateConfig(form) {
        const payload = this.getPayload(form);

        const response = await fetch(`${this.apiBaseUrl}/training/validate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'text/html'
            },
            credentials: 'include',
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`Validation failed: ${response.status}`);
        }

        return await response.text();
    }

    /**
     * Check for mutual exclusivity of epochs and steps
     */
    validateEpochsSteps(numEpochs, maxSteps) {
        const epochs = parseInt(numEpochs) || 0;
        const steps = parseInt(maxSteps) || 0;

        if (epochs === 0 && steps === 0) {
            return {
                valid: false,
                error: "Either num_train_epochs or max_train_steps must be greater than 0."
            };
        }

        return { valid: true };
    }

    /**
     * Validate required fields
     */
    validateRequiredFields(form) {
        const requiredFields = form.querySelectorAll('[required]');
        const errors = [];

        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                const label = field.closest('.mb-3')?.querySelector('label')?.textContent || field.name;
                errors.push(`${label} is required`);
            }
        });

        return errors;
    }
}
