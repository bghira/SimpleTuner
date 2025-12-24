(function () {
    'use strict';

    /**
     * Memory Presets Modal Component
     *
     * Allows users to select and apply memory optimization presets
     * from a modal dialog. Presets are fetched from the backend API
     * based on the currently selected model family.
     */

    function memoryPresetsComponent() {
        return {
            isOpen: false,
            loading: false,
            error: null,

            // Presets data from API
            presets: [],
            maxSwappableBlocks: null,
            unsupportedBackends: [],
            systemRamGb: null,

            // Selection state
            selectedTab: 'basic',
            selectedPresets: {},  // Map of backend -> level
            customBlockSwapCount: 0,

            // Track which fields should be modified
            _targetFields: null,

            init() {
                window.__memoryPresetsInstance = this;
                this._eventHandler = this._handleOpenEvent.bind(this);
                window.addEventListener('open-memory-presets', this._eventHandler);
            },

            destroy() {
                window.removeEventListener('open-memory-presets', this._eventHandler);
                if (window.__memoryPresetsInstance === this) {
                    delete window.__memoryPresetsInstance;
                }
            },

            _handleOpenEvent(event) {
                const detail = event.detail || {};
                this.open(detail.targetFields);
            },

            async open(targetFields) {
                this._targetFields = targetFields || null;
                this.isOpen = true;
                this.error = null;
                this.selectedPresets = {};
                this.customBlockSwapCount = 0;
                this.selectedTab = 'basic';

                await this.loadPresets();
            },

            close() {
                this.isOpen = false;
                this._targetFields = null;
            },

            getModelFamily() {
                // Try to get from trainer store first (keys are prefixed with --)
                const trainerStore = window.Alpine?.store?.('trainer');
                if (trainerStore?.activeEnvironmentConfig) {
                    const config = trainerStore.activeEnvironmentConfig;
                    // Check both dashed and non-dashed keys
                    const modelFamily = config['--model_family'] || config['model_family'];
                    if (modelFamily) {
                        return modelFamily;
                    }
                }

                // Fallback to DOM element
                const modelFamilyEl = document.getElementById('model_family');
                return modelFamilyEl ? modelFamilyEl.value.trim() : null;
            },

            getModelType() {
                // Try to get from trainer store first
                const trainerStore = window.Alpine?.store?.('trainer');
                if (trainerStore?.activeEnvironmentConfig) {
                    const config = trainerStore.activeEnvironmentConfig;
                    const modelType = config['--model_type'] || config['model_type'];
                    if (modelType) {
                        return modelType;
                    }
                }

                // Fallback to DOM element
                const modelTypeEl = document.getElementById('model_type');
                return modelTypeEl ? modelTypeEl.value.trim() : 'lora';
            },

            async loadPresets() {
                const modelFamily = this.getModelFamily();
                if (!modelFamily) {
                    this.error = 'Please select a model family first.';
                    this.presets = [];
                    return;
                }

                this.loading = true;
                this.error = null;

                try {
                    const [presetsRes, memoryRes] = await Promise.all([
                        fetch(`/api/models/${encodeURIComponent(modelFamily)}/acceleration-presets`),
                        fetch('/api/hardware/memory')
                    ]);

                    if (presetsRes.ok) {
                        const data = await presetsRes.json();
                        this.presets = data.presets || [];
                        this.maxSwappableBlocks = data.max_swappable_blocks;
                        this.unsupportedBackends = data.unsupported_backends || [];
                    } else {
                        console.warn('[MEMORY PRESETS] Failed to load presets:', presetsRes.status);
                        this.presets = [];
                    }

                    if (memoryRes.ok) {
                        const memData = await memoryRes.json();
                        this.systemRamGb = memData.total_gb || null;
                    }
                } catch (err) {
                    console.error('[MEMORY PRESETS] Error loading presets:', err);
                    this.error = 'Failed to load presets. Please try again.';
                } finally {
                    this.loading = false;
                }
            },

            get lowSystemRam() {
                return this.systemRamGb !== null && this.systemRamGb < 64;
            },

            get hasRamIntensiveSelection() {
                for (const [backend, level] of Object.entries(this.selectedPresets)) {
                    const preset = this.presets.find(p => p.backend === backend && p.level === level);
                    if (preset && preset.requires_min_system_ram_gb >= 64) {
                        return true;
                    }
                }
                return this.customBlockSwapCount > 0;
            },

            getPresetsForTab(tab) {
                const modelType = this.getModelType();
                return this.presets.filter(p => {
                    // Filter by tab
                    if (p.tab !== tab) return false;
                    // Filter by model_type if required
                    if (p.requires_model_type && !p.requires_model_type.includes(modelType)) {
                        return false;
                    }
                    return true;
                });
            },

            getPresetsGroupedByBackend(tab) {
                const presets = this.getPresetsForTab(tab);
                const groups = {};
                const order = [];

                for (const preset of presets) {
                    // Use display_group for UI grouping if present, otherwise fall back to backend
                    const groupKey = preset.display_group || preset.backend;
                    if (!groups[groupKey]) {
                        groups[groupKey] = {
                            backend: groupKey,
                            label: this.getBackendLabel(groupKey),
                            presets: []
                        };
                        order.push(groupKey);
                    }
                    groups[groupKey].presets.push(preset);
                }

                // Sort presets within each group by level (basic < balanced < aggressive)
                const levelOrder = { basic: 0, conservative: 1, balanced: 2, aggressive: 3, extreme: 4, zero1: 0, zero2: 1, zero3: 2 };
                for (const group of Object.values(groups)) {
                    group.presets.sort((a, b) => (levelOrder[a.level] ?? 99) - (levelOrder[b.level] ?? 99));
                }

                return order.map(groupKey => groups[groupKey]);
            },

            getBackendLabel(backend) {
                const labels = {
                    'RAMTORCH': 'RamTorch Streaming',
                    'MUSUBI_BLOCK_SWAP': 'Block Swap',
                    'GROUP_OFFLOAD': 'Group Offload',
                    'deepspeed': 'DeepSpeed ZeRO',
                    'DEEPSPEED_ZERO_1': 'DeepSpeed ZeRO',
                    'DEEPSPEED_ZERO_2': 'DeepSpeed ZeRO',
                    'DEEPSPEED_ZERO_3': 'DeepSpeed ZeRO',
                    'SDNQ': 'SDNQ',
                    'TORCHAO': 'TorchAO',
                    'QUANTO': 'Quanto',
                    'BITSANDBYTES': 'BitsAndBytes',
                };
                return labels[backend] || backend;
            },

            isPresetSelected(preset) {
                return this.selectedPresets[preset.backend] === preset.level;
            },

            togglePreset(preset) {
                const exclusiveBackends = ['RAMTORCH', 'GROUP_OFFLOAD', 'MUSUBI_BLOCK_SWAP'];

                if (this.isPresetSelected(preset)) {
                    // Deselect
                    delete this.selectedPresets[preset.backend];
                } else {
                    // If selecting an exclusive backend, deselect the others
                    if (exclusiveBackends.includes(preset.backend)) {
                        for (const backend of exclusiveBackends) {
                            if (backend !== preset.backend) {
                                delete this.selectedPresets[backend];
                            }
                        }
                        // Clear custom block swap when selecting RamTorch or Group Offload
                        if (preset.backend === 'RAMTORCH' || preset.backend === 'GROUP_OFFLOAD') {
                            this.customBlockSwapCount = 0;
                        }
                    }
                    // If preset has a group, deselect other presets in the same group
                    if (preset.group) {
                        for (const otherPreset of this.presets) {
                            if (otherPreset.group === preset.group && otherPreset.backend !== preset.backend) {
                                delete this.selectedPresets[otherPreset.backend];
                            }
                        }
                    }
                    // Select this preset
                    this.selectedPresets[preset.backend] = preset.level;
                }
                // Force reactivity
                this.selectedPresets = { ...this.selectedPresets };
            },

            getSelectedConfig() {
                const mergedConfig = {};

                // Merge all selected presets' configs
                for (const [backend, level] of Object.entries(this.selectedPresets)) {
                    const preset = this.presets.find(p => p.backend === backend && p.level === level);
                    if (preset?.config) {
                        Object.assign(mergedConfig, preset.config);
                    }
                }

                // Apply custom block swap if set
                if (this.customBlockSwapCount > 0) {
                    mergedConfig.musubi_blocks_to_swap = this.customBlockSwapCount;
                    // Clear RamTorch if using custom block swap
                    if (mergedConfig.ramtorch) {
                        delete mergedConfig.ramtorch;
                        delete mergedConfig.ramtorch_target_modules;
                    }
                }

                return mergedConfig;
            },

            get hasSelection() {
                return Object.keys(this.selectedPresets).length > 0 || this.customBlockSwapCount > 0;
            },

            apply() {
                if (!this.hasSelection) {
                    window.showToast?.('No presets selected.', 'warning');
                    return;
                }

                const config = this.getSelectedConfig();
                console.log('[MEMORY PRESETS] Applying config:', config);

                // Memory optimization fields that should be reset before applying new presets
                const memoryOptFields = [
                    'ramtorch',
                    'ramtorch_target_modules',
                    'musubi_blocks_to_swap',
                    'enable_group_offload',
                    'group_offload_type',
                    'deepspeed_config',
                ];

                // Default/reset values for each field
                const resetValues = {
                    'ramtorch': false,
                    'ramtorch_target_modules': '',
                    'musubi_blocks_to_swap': 0,
                    'enable_group_offload': false,
                    'group_offload_type': '',
                    'deepspeed_config': '',
                };

                // Apply to form fields via trainer store
                const trainerStore = window.Alpine?.store?.('trainer');

                // First, reset all memory optimization fields
                for (const field of memoryOptFields) {
                    const canonicalKey = `--${field}`;
                    const resetValue = resetValues[field];

                    if (trainerStore) {
                        if (typeof trainerStore.updateConfigValue === 'function') {
                            trainerStore.updateConfigValue(canonicalKey, resetValue);
                        } else if (trainerStore.activeEnvironmentConfig) {
                            trainerStore.activeEnvironmentConfig[canonicalKey] = resetValue;
                        }
                        if (trainerStore.configValues) {
                            trainerStore.configValues[canonicalKey] = resetValue;
                        }
                    }

                    // Reset DOM element
                    const el = document.getElementById(field) || document.querySelector(`[name="${field}"]`);
                    if (el) {
                        if (el.type === 'checkbox') {
                            el.checked = Boolean(resetValue);
                        } else {
                            el.value = resetValue;
                        }
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                }

                console.log('[MEMORY PRESETS] Reset existing memory optimization settings');
                if (trainerStore) {
                    for (const [key, value] of Object.entries(config)) {
                        // Canonicalize key to match store format (add -- prefix)
                        const canonicalKey = key.startsWith('--') ? key : `--${key}`;

                        // Use the trainer store's config update method
                        if (typeof trainerStore.updateConfigValue === 'function') {
                            trainerStore.updateConfigValue(canonicalKey, value);
                        } else {
                            // Fallback: update activeEnvironmentConfig directly
                            if (trainerStore.activeEnvironmentConfig) {
                                trainerStore.activeEnvironmentConfig[canonicalKey] = value;
                            }
                        }

                        // Also update configValues if present
                        if (trainerStore.configValues) {
                            trainerStore.configValues[canonicalKey] = value;
                        }
                    }

                    // Sync to DOM
                    if (typeof trainerStore.applyStoredValues === 'function') {
                        trainerStore.applyStoredValues();
                    }

                    // Mark form as dirty
                    if (typeof trainerStore.markFormDirty === 'function') {
                        trainerStore.markFormDirty();
                    }
                }

                // Also update DOM elements directly (fields use non-dashed names)
                for (const [key, value] of Object.entries(config)) {
                    // Remove -- prefix for DOM element lookup
                    const fieldName = key.startsWith('--') ? key.slice(2) : key;
                    const el = document.getElementById(fieldName) || document.querySelector(`[name="${fieldName}"]`);
                    if (el) {
                        if (el.type === 'checkbox') {
                            el.checked = Boolean(value);
                        } else {
                            el.value = value;
                        }
                        // Trigger change event
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                    }
                }

                // Show success message
                const presetCount = Object.keys(this.selectedPresets).length;
                const message = this.customBlockSwapCount > 0
                    ? `Applied ${presetCount} preset(s) and custom block swap (${this.customBlockSwapCount} blocks)`
                    : `Applied ${presetCount} memory optimization preset(s)`;
                window.showToast?.(message, 'success');

                this.close();
            },

            onBlockSwapSliderChange() {
                // Clear RamTorch and Group Offload if custom block swap is set
                if (this.customBlockSwapCount > 0) {
                    delete this.selectedPresets['RAMTORCH'];
                    delete this.selectedPresets['GROUP_OFFLOAD'];
                    this.selectedPresets = { ...this.selectedPresets };
                }
            }
        };
    }

    // Register the Alpine component
    if (window.Alpine) {
        window.Alpine.data('memoryPresetsComponent', memoryPresetsComponent);
    } else {
        // Wait for Alpine to be available
        document.addEventListener('alpine:init', () => {
            window.Alpine.data('memoryPresetsComponent', memoryPresetsComponent);
        });
    }

    // Global function to open the modal
    window.openMemoryPresetsModal = function(targetFields) {
        const instance = window.__memoryPresetsInstance;
        if (instance) {
            instance.open(targetFields);
        } else {
            // Dispatch event for when Alpine hasn't initialized the component yet
            window.dispatchEvent(new CustomEvent('open-memory-presets', {
                detail: { targetFields }
            }));
        }
    };

    /**
     * Inject a "Load Presets" button into the memory_optimization section header.
     * This runs after tab content is loaded via HTMX.
     */
    function injectPresetsButton() {
        const sectionId = 'section-memory_optimization';
        const section = document.getElementById(sectionId);
        if (!section) {
            return;
        }

        // Check if button already exists
        if (section.querySelector('.memory-presets-btn')) {
            return;
        }

        // Find the section title element
        const sectionTitle = section.querySelector('.section-title');
        if (!sectionTitle) {
            return;
        }

        // Create the button
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-sm btn-outline-primary memory-presets-btn ms-2';
        btn.innerHTML = '<i class="fas fa-magic me-1"></i>Load Presets';
        btn.title = 'Load memory optimization presets for your model';
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            window.openMemoryPresetsModal();
        });

        // Insert after the section title text (before any badges)
        const badge = sectionTitle.querySelector('.badge');
        if (badge) {
            sectionTitle.insertBefore(btn, badge);
        } else {
            sectionTitle.appendChild(btn);
        }

        console.log('[MEMORY PRESETS] Injected presets button into memory_optimization section');
    }

    // Inject button when model tab loads
    function setupButtonInjection() {
        // Initial injection attempt
        injectPresetsButton();

        // Re-inject after HTMX swaps (tab switches)
        document.body.addEventListener('htmx:afterSwap', (evt) => {
            // Check if the swap target might contain our section
            if (evt.detail.target.id === 'tab-content' ||
                evt.detail.target.id === 'model-tab-content' ||
                evt.detail.target.querySelector?.('#section-memory_optimization')) {
                // Small delay to ensure DOM is ready
                setTimeout(injectPresetsButton, 50);
            }
        });

        // Also listen for HTMX settle events
        document.body.addEventListener('htmx:afterSettle', (evt) => {
            if (document.getElementById('section-memory_optimization')) {
                setTimeout(injectPresetsButton, 50);
            }
        });
    }

    // Initialize button injection
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupButtonInjection);
    } else {
        setupButtonInjection();
    }
})();
