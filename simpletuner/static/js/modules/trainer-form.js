/**
 * Trainer Form Module
 * Handles form interactions and field dependencies
 */

export class TrainerForm {
    constructor() {
        this.form = document.getElementById('configForm') || document.getElementById('trainer-form');
        this.modelFlavours = {};
    }

    /**
     * Initialize form dependencies
     */
    initializeDependencies() {
        const modelType = document.getElementById('model_type');
        const modelFamily = document.getElementById('model_family');
        const loraType = document.getElementById('lora_type');

        const updateDependencies = async () => {
            this.updateLoraVisibility(modelType?.value);
            this.updateFluxLoraVisibility(modelFamily?.value, modelType?.value);
            this.updateLycorisVisibility(loraType?.value, modelType?.value);
            await this.updateModelFlavours(modelFamily?.value);
        };

        modelType?.addEventListener('change', updateDependencies);
        modelFamily?.addEventListener('change', updateDependencies);
        loraType?.addEventListener('change', updateDependencies);

        // Initial update
        updateDependencies();
    }

    /**
     * Update LoRA visibility based on model type
     */
    updateLoraVisibility(modelType) {
        const loraConfig = document.getElementById('lora-config');
        if (loraConfig) {
            loraConfig.classList.toggle('field-disabled', modelType !== 'lora');
        }
    }

    /**
     * Update Flux LoRA visibility
     */
    updateFluxLoraVisibility(modelFamily, modelType) {
        const fluxLoraGroup = document.getElementById('flux-lora-group');
        if (fluxLoraGroup) {
            const isVisible = modelFamily === 'flux' && modelType === 'lora';
            fluxLoraGroup.classList.toggle('field-disabled', !isVisible);
        }
    }

    /**
     * Update Lycoris visibility
     */
    updateLycorisVisibility(loraType, modelType) {
        const lycorisGroup = document.getElementById('lycoris-config-group');
        if (lycorisGroup) {
            const isVisible = loraType === 'lycoris' && modelType === 'lora';
            lycorisGroup.classList.toggle('field-disabled', !isVisible);
        }
    }

    /**
     * Update model flavours dropdown
     */
    async updateModelFlavours(modelFamily) {
        const flavourSelect = document.getElementById('model_flavour');
        const flavourGroup = document.getElementById('model-flavour-group');

        if (!flavourSelect || !flavourGroup) return;

        if (!modelFamily) {
            flavourGroup.style.display = 'none';
            return;
        }

        // Show loading state
        flavourSelect.innerHTML = '<option value="">Loading flavours...</option>';
        flavourGroup.style.display = 'block';

        try {
            // Fetch flavours
            const response = await fetch(`/api/models/${modelFamily}/flavours`);
            const data = await response.json();

            // Update dropdown
            flavourSelect.innerHTML = '<option value="">Default</option>';

            if (data.flavours && data.flavours.length > 0) {
                data.flavours.forEach(flavour => {
                    const option = document.createElement('option');
                    option.value = flavour;
                    option.textContent = flavour;
                    flavourSelect.appendChild(option);
                });

                // Auto-select if only one option
                if (data.flavours.length === 1) {
                    flavourSelect.value = data.flavours[0];
                    this.updateModelPath(data.flavours[0]);
                }
            }
        } catch (error) {
            console.error('Failed to fetch model flavours:', error);
            flavourSelect.innerHTML = '<option value="">Error loading flavours</option>';
        }
    }

    /**
     * Update model path based on flavour selection
     */
    updateModelPath(flavour) {
        const modelPath = document.getElementById('model_path');
        if (modelPath && flavour) {
            modelPath.value = flavour;
            modelPath.dispatchEvent(new Event('change', { bubbles: true }));
        }
    }

    /**
     * Setup JSON editors with formatting
     */
    setupJSONEditors() {
        document.querySelectorAll('.json-editor').forEach(editor => {
            editor.addEventListener('blur', function() {
                try {
                    const json = JSON.parse(this.value);
                    this.value = JSON.stringify(json, null, 4);
                } catch (e) {
                    // Invalid JSON, don't format
                }
            });

            // Add tab key support
            editor.addEventListener('keydown', function(e) {
                if (e.key === 'Tab') {
                    e.preventDefault();
                    const start = this.selectionStart;
                    const end = this.selectionEnd;
                    this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                    this.selectionStart = this.selectionEnd = start + 4;
                }
            });
        });
    }

    /**
     * Get form data as object
     */
    getFormData() {
        const formData = new FormData(this.form);
        const data = {};

        for (const [key, value] of formData.entries()) {
            data[key] = value;
        }

        return data;
    }

    /**
     * Set form data from object
     */
    setFormData(data) {
        Object.entries(data).forEach(([key, value]) => {
            const field = this.form.querySelector(`[name="${key}"]`);
            if (field) {
                if (field.type === 'checkbox') {
                    field.checked = value === 'true' || value === true;
                } else {
                    field.value = value;
                }
            }
        });
    }

    /**
     * Reset form to defaults
     */
    reset() {
        this.form.reset();
        // Trigger change events to update dependencies
        this.form.querySelectorAll('select').forEach(select => {
            select.dispatchEvent(new Event('change', { bubbles: true }));
        });
    }
}
