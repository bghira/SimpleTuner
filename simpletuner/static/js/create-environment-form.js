/**
 * Shared Alpine component for creating training environments.
 * Provides the same behaviour for the environments tab modal and onboarding flow.
 */
(function (global) {
    const noop = () => {};
    const sanitizeConfigName =
        global.sanitizeConfigName ||
        function (name) {
            if (typeof name !== "string") {
                return name;
            }
            const trimmed = name.trim();
            return trimmed.toLowerCase().endsWith(".json") ? trimmed.slice(0, -5).trim() : trimmed;
        };

    console.info("[EnvForm] script loaded", { version: "2025-10-16" });

    function createEnvironmentFormComponent(options = {}) {
        return {
            options,
            loadingInitialData: false,
            submitting: false,
            error: "",
            newEnvironmentPathTouched: false,
            initialLoadDone: false,
            newEnvironment: {
                name: "",
                model_family: "",
                model_flavour: null,
                model_type: "lora",
                lora_type: "standard",
                dataloader_path: "",
                description: "",
                example: "",
            },
            modelFamilies: [],
            modelFlavours: [],
            modelFlavourCache: {},
            modelTypes: [
                { value: "lora", label: "LoRA" },
                { value: "full", label: "Full Model" },
            ],
            loraTypes: [
                { value: "standard", label: "Standard", disabled: false },
                { value: "lycoris", label: "LyCORIS", disabled: false },
            ],
            examples: [],
            examplesMap: {},
            dataloaderConfigs: [],
            createNewDataloader: true,
            selectedDataloaderPath: "",
            exampleLocked: false,

            async init() {
                console.info("[EnvForm] init called", { autoLoad: this.options.autoLoad });
                if (this.options.autoLoad !== false) {
                    await this.prepareForDisplay(this.options.preset || {});
                } else {
                    queueMicrotask(async () => {
                        if (!this.initialLoadDone) {
                            console.info("[EnvForm] autoLoad disabled; running fallback prepareForDisplay()", {
                                hasPreset: Boolean(this.options.preset && Object.keys(this.options.preset).length),
                            });
                            await this.prepareForDisplay(this.options.preset || {});
                        }
                    });
                }
            },

            resetForm() {
                this.newEnvironment = {
                    name: "",
                    model_family: "",
                    model_flavour: null,
                    model_type: "lora",
                    lora_type: "standard",
                    dataloader_path: "",
                    description: "",
                    example: "",
                };
                this.newEnvironmentPathTouched = false;
                this.modelFlavours = [];
                this.createNewDataloader = true;
                this.selectedDataloaderPath = "";
                this.exampleLocked = false;
                this.error = "";
            },

            async prepareForDisplay(preset = {}) {
                this.initialLoadDone = true;
                this.loadingInitialData = true;
                console.info("[EnvForm] loadingInitialData -> true");
                this.resetForm();
                if (preset && typeof preset === "object" && Object.keys(preset).length > 0) {
                    this.applyPreset(preset);
                }
                try {
                    if (global.ServerConfig && typeof global.ServerConfig.waitForReady === "function") {
                        await global.ServerConfig.waitForReady();
                    }
                    console.info("[EnvForm] Preparing environment form", {
                        presetApplied: Boolean(preset && Object.keys(preset).length),
                        apiBaseUrl: global.ApiClient ? global.ApiClient.apiBaseUrl : undefined,
                        callbackUrl: global.ApiClient ? global.ApiClient.callbackBaseUrl : undefined,
                    });
                    await Promise.all([this.ensureModelFamilies(), this.ensureExamples(), this.ensureDataloaderConfigs()]);
                    if (this.newEnvironment.model_family) {
                        await this.loadModelFlavours(this.newEnvironment.model_family);
                    } else if (this.modelFamilies.length > 0) {
                        const defaultFamily = this.modelFamilies[0];
                        this.newEnvironment.model_family = defaultFamily;
                        await this.loadModelFlavours(defaultFamily);
                    }

                    if (this.dataloaderConfigs.length > 0) {
                        this.createNewDataloader = false;
                        this.selectedDataloaderPath = this.dataloaderConfigs[0].path || "";
                        this.newEnvironment.dataloader_path = this.selectedDataloaderPath;
                    } else {
                        this.createNewDataloader = true;
                        this.newEnvironment.dataloader_path = this.computeDefaultDataloaderPath();
                    }
                    this.exampleLocked = false;
                    this.error = "";
                    this.$dispatch("environment-form-ready");
                } finally {
                    console.info("[EnvForm] prepareForDisplay completed, clearing loading flag");
                    this.loadingInitialData = false;
                    console.info("[EnvForm] loadingInitialData -> false");
                }
            },

            applyPreset(preset) {
                if (!preset || typeof preset !== "object") {
                    return;
                }
                if (preset.name) {
                    this.newEnvironment.name = sanitizeConfigName(preset.name);
                }
                if (preset.model_family) {
                    this.newEnvironment.model_family = preset.model_family;
                }
                if (preset.model_flavour) {
                    this.newEnvironment.model_flavour = preset.model_flavour;
                }
                if (preset.model_type) {
                    this.newEnvironment.model_type = preset.model_type;
                }
                if (preset.lora_type) {
                    this.newEnvironment.lora_type = preset.lora_type;
                }
                if (preset.description) {
                    this.newEnvironment.description = preset.description;
                }
            },

            updateNewEnvironmentName(value) {
                const sanitized = sanitizeConfigName(value || "");
                this.newEnvironment.name = sanitized;
                if (this.createNewDataloader && !this.newEnvironmentPathTouched) {
                    this.newEnvironment.dataloader_path = sanitized ? `${sanitized}/multidatabackend.json` : "";
                }
            },

            markEnvironmentPathTouched() {
                this.newEnvironmentPathTouched = true;
            },

            selectExistingDataloader(path) {
                this.selectedDataloaderPath = path;
                if (!this.createNewDataloader) {
                    this.newEnvironment.dataloader_path = path;
                }
            },

            toggleCreateNewDataloader(checked) {
                this.createNewDataloader = checked;
                if (checked) {
                    if (!this.newEnvironmentPathTouched) {
                        this.newEnvironment.dataloader_path = this.computeDefaultDataloaderPath();
                    }
                } else if (this.dataloaderConfigs.length > 0) {
                    if (!this.selectedDataloaderPath) {
                        this.selectedDataloaderPath = this.dataloaderConfigs[0].path || "";
                    }
                    this.newEnvironment.dataloader_path = this.selectedDataloaderPath;
                } else {
                    this.createNewDataloader = true;
                    if (!this.newEnvironmentPathTouched) {
                        this.newEnvironment.dataloader_path = this.computeDefaultDataloaderPath();
                    }
                }
            },

            handleModelTypeChange(value) {
                this.newEnvironment.model_type = value;
                if (value !== "lora") {
                    this.newEnvironment.lora_type = "standard";
                }
            },

            async handleExampleSelection(value) {
                this.newEnvironment.example = value;
                if (!value) {
                    this.exampleLocked = false;
                    this.newEnvironment.description = "";
                    if (this.dataloaderConfigs.length > 0) {
                        this.createNewDataloader = false;
                        this.selectedDataloaderPath = this.dataloaderConfigs[0]?.path || "";
                        if (this.selectedDataloaderPath) {
                            this.newEnvironment.dataloader_path = this.selectedDataloaderPath;
                        }
                    } else {
                        this.createNewDataloader = true;
                        this.newEnvironmentPathTouched = false;
                        this.newEnvironment.dataloader_path = this.computeDefaultDataloaderPath();
                    }
                    return;
                }

                const example = this.examplesMap[value];
                if (!example || !example.defaults) {
                    this.exampleLocked = false;
                    return;
                }

                const defaults = example.defaults;
                if (defaults.model_family) {
                    this.newEnvironment.model_family = defaults.model_family;
                    await this.loadModelFlavours(defaults.model_family);
                }
                this.newEnvironment.model_flavour = defaults.model_flavour ?? null;

                const modelType = defaults.model_type || "lora";
                this.newEnvironment.model_type = modelType;
                this.handleModelTypeChange(modelType);
                if (modelType === "lora") {
                    this.newEnvironment.lora_type = defaults.lora_type || "standard";
                }

                this.newEnvironment.description = example.description || "";
                this.exampleLocked = true;
                this.createNewDataloader = true;
                this.newEnvironmentPathTouched = false;
                this.newEnvironment.dataloader_path = this.computeDefaultDataloaderPath();
                this.selectedDataloaderPath = "";
            },

            computeDefaultDataloaderPath() {
                if (!this.newEnvironment.name) {
                    return "";
                }
                return `${this.newEnvironment.name}/multidatabackend.json`;
            },

            resolveEndpoint(path, options = {}) {
                const helper = global.ApiClient;
                if (helper && typeof helper.resolve === "function") {
                    return helper.resolve(path, options);
                }
                const normalizedPath = path.startsWith("/") ? path : `/${path}`;
                const baseUrl = (global.ServerConfig && global.ServerConfig.apiBaseUrl) || global.location.origin;
                return `${baseUrl}${normalizedPath}`;
            },

            apiFetch(path, fetchOptions = {}, resolveOptions = {}) {
                const helper = global.ApiClient;
                if (helper && typeof helper.fetch === "function") {
                    return helper.fetch(path, fetchOptions, { forceApi: true, ...resolveOptions });
                }
                const url = this.resolveEndpoint(path, { forceApi: true, ...resolveOptions });
                return global.fetch(url, fetchOptions);
            },

            async ensureModelFamilies() {
                if (this.modelFamilies.length > 0) {
                    return;
                }
                console.info("[EnvForm] ensureModelFamilies fetching");
                try {
                    const response = await this.apiFetch("/api/models");
                    console.info("[EnvForm] /api/models response", response);
                    if (response.ok) {
                        const data = await response.json();
                        console.info("[EnvForm] model families payload", data);
                        this.modelFamilies = Array.isArray(data.families) ? data.families : [];
                        if (this.modelFamilies.length > 0) {
                            this.loadingInitialData = false;
                            console.info("[EnvForm] Families loaded, loadingInitialData -> false");
                        }
                    }
                } catch (error) {
                    console.error("Failed to load model families:", error);
                    this.loadingInitialData = false;
                }
            },

            async loadModelFlavours(family) {
                if (!family) {
                    this.modelFlavours = [];
                    return;
                }
                if (this.modelFlavourCache[family]) {
                    this.modelFlavours = this.modelFlavourCache[family];
                    return;
                }
                try {
                    const response = await this.apiFetch(`/api/models/${family}/flavours`);
                    console.info("[EnvForm] /api/models/", family, "response", response);
                    if (response.ok) {
                        const data = await response.json();
                        const flavours = Array.isArray(data.flavours) ? data.flavours : [];
                        this.modelFlavourCache[family] = flavours;
                        this.modelFlavours = flavours;
                    } else {
                        this.modelFlavours = [];
                    }
                } catch (error) {
                    console.error("Failed to load model flavours:", error);
                    this.modelFlavours = [];
                }
            },

            async ensureExamples() {
                if (this.examples.length > 0) {
                    return;
                }
                try {
                    const response = await this.apiFetch("/api/configs/examples");
                    if (response.ok) {
                        const data = await response.json();
                        this.examples = Array.isArray(data.examples) ? data.examples : [];
                        this.examplesMap = Object.fromEntries(
                            this.examples.map((example) => [example.name, example])
                        );
                    }
                } catch (error) {
                    console.error("Failed to load examples:", error);
                }
            },

            async ensureDataloaderConfigs() {
                try {
                    const store = global.Alpine ? global.Alpine.store("dataloaderConfigs") : null;
                    if (store) {
                        await store.load();
                        this.dataloaderConfigs = store.configs || [];
                    }
                } catch (error) {
                    console.error("Failed to load dataloader configs:", error);
                }
            },

            async generateProjectName() {
                try {
                    const response = await this.apiFetch("/api/configs/project-name");
                    if (!response.ok) {
                        throw new Error("Failed to generate name");
                    }
                    const data = await response.json();
                    if (data?.name) {
                        this.updateNewEnvironmentName(data.name);
                    }
                } catch (error) {
                    console.error("Failed to generate project name:", error);
                    global.showToast?.("Could not generate project name", "error");
                }
            },

            buildPayload() {
                const sanitizedName = sanitizeConfigName(this.newEnvironment.name || "");
                if (!sanitizedName || !this.newEnvironment.model_family) {
                    global.showToast?.("Name and model family are required", "warning");
                    this.error = "Name and model family are required.";
                    return null;
                }
                this.newEnvironment.name = sanitizedName;
                const payload = {
                    name: sanitizedName,
                    model_family: this.newEnvironment.model_family,
                    model_flavour: this.newEnvironment.model_flavour || null,
                    model_type: this.newEnvironment.model_type || "lora",
                    lora_type:
                        this.newEnvironment.model_type === "lora"
                            ? this.newEnvironment.lora_type || "standard"
                            : null,
                    description: this.newEnvironment.description || null,
                    example: this.newEnvironment.example || null,
                    dataloader_path: this.createNewDataloader
                        ? this.newEnvironment.dataloader_path || null
                        : this.selectedDataloaderPath || null,
                    create_dataloader: this.createNewDataloader,
                };
                if (!payload.dataloader_path && this.createNewDataloader) {
                    const fallback = this.computeDefaultDataloaderPath();
                    payload.dataloader_path = fallback || null;
                }
                if (!payload.dataloader_path && !this.createNewDataloader && this.selectedDataloaderPath) {
                    payload.dataloader_path = this.selectedDataloaderPath;
                }
                if (!payload.dataloader_path && !this.createNewDataloader && this.dataloaderConfigs.length > 0) {
                    payload.dataloader_path = this.dataloaderConfigs[0].path;
                }
                return payload;
            },

            async submit() {
                if (this.submitting) {
                    return;
                }
                const payload = this.buildPayload();
                if (!payload) {
                    return;
                }

                this.submitting = true;
                this.error = "";
                this.$dispatch("environment-submit-start", { payload });
                try {
                    const response = await this.apiFetch("/api/configs/environments", {
                        method: "POST",
                        headers: {
                            "Content-Type": "application/json",
                        },
                        body: JSON.stringify(payload),
                    });

                    if (!response.ok) {
                        const errorPayload = await response.json().catch(() => ({}));
                        const message = errorPayload.detail || "Failed to create environment";
                        this.error = message;
                        global.showToast?.(message, "error");
                        this.$dispatch("environment-create-error", {
                            message,
                            status: response.status,
                            detail: errorPayload,
                        });
                        return;
                    }

                    const result = await response.json().catch(() => ({}));
                    if (this.options.toastOnSuccess !== false) {
                        global.showToast?.(`Environment "${payload.name}" created`, "success");
                    }
                    this.$dispatch("environment-created", {
                        payload,
                        result,
                        environment: result && result.environment ? result.environment : null,
                    });
                    if (this.options.resetOnSuccess !== false) {
                        this.resetForm();
                    }
                } catch (error) {
                    console.error("Failed to create environment:", error);
                    const message = "Failed to create environment";
                    this.error = message;
                    global.showToast?.(message, "error");
                    this.$dispatch("environment-create-error", { message, error });
                } finally {
                    this.submitting = false;
                    this.$dispatch("environment-submit-end");
                }
            },
        };
    }

    global.createEnvironmentFormComponent = createEnvironmentFormComponent;
})(window);
