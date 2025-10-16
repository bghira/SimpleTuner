(function () {
    const DEFAULTS = {
        mode: "inline",
        stage: "stage2",
        gradientAccumulationSteps: 1,
        gradientClipping: 1,
        precision: "bf16", // Changed from fp16 to bf16 for better stability
        offloadParam: "none",
        offloadOptimizer: "none",
        offloadPath: "",
        zero3Init: true,
        zero3Save16: false,
        zero3Gather16: true,
        zeroQuantizedWeights: false,
        zeroQuantizedGradients: false,
        zeroHpzPartitionSize: 8,
        configFilePath: "",
        includeOptimizer: true,
        optimizerType: "auto",
        optimizerCustomType: "",
        optimizerLr: 0.0001,
        optimizerBeta1: 0.9,
        optimizerBeta2: 0.999,
        optimizerEps: 1e-6,
        optimizerWeightDecay: 0.01,
        optimizerTorchAdam: false,
        optimizerAdamWMode: true,
        includeScheduler: true,
        schedulerType: "WarmupLR",
        schedulerCustomType: "",
        schedulerWarmupMinLr: 0,
        schedulerWarmupMaxLr: 0.0001,
        schedulerWarmupNumSteps: 0,
    };

    function cloneDefaults() {
        return JSON.parse(JSON.stringify(DEFAULTS));
    }

    function normaliseStage(stage) {
        if (typeof stage === "number") {
            return stage >= 3 ? "stage3" : stage <= 1 ? "stage1" : "stage2";
        }
        if (typeof stage === "string") {
            const trimmed = stage.trim().toLowerCase();
            if (trimmed.includes("3")) return "stage3";
            if (trimmed.includes("1")) return "stage1";
        }
        return "stage2";
    }

    function toNumber(value, fallback) {
        const numeric = Number(value);
        if (Number.isFinite(numeric)) {
            return numeric;
        }
        return fallback;
    }

    function parseOptionalNumber(value) {
        if (value === null || value === undefined || value === "") {
            return undefined;
        }
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : undefined;
    }

    function buildInlineConfig(state) {
        const stageValue = normaliseStage(state.stage);
        const zeroStage = stageValue === "stage3" ? 3 : stageValue === "stage1" ? 1 : 2;

        const zeroOpt = {
            stage: zeroStage,
        };

        if (zeroStage >= 2) {
            zeroOpt.overlap_comm = true;
            zeroOpt.contiguous_gradients = true;
            zeroOpt.reduce_bucket_size = "auto";
        }

        if (zeroStage === 3) {
            zeroOpt.stage3_prefetch_bucket_size = "auto";
            zeroOpt.stage3_param_persistence_threshold = "auto";
            zeroOpt.stage3_gather_16bit_weights_on_model_save = !!state.zero3Gather16;
        }

        if (state.offloadParam && state.offloadParam !== "none") {
            zeroOpt.offload_param = {
                device: state.offloadParam,
                pin_memory: state.offloadParam !== "nvme" ? true : true,
            };
            if (state.offloadParam === "nvme") {
                zeroOpt.offload_param.nvme_path = state.offloadPath?.trim() || "none";
            }
        }

        if (state.offloadOptimizer && state.offloadOptimizer !== "none") {
            zeroOpt.offload_optimizer = {
                device: state.offloadOptimizer,
            };
            if (state.offloadOptimizer === "nvme") {
                zeroOpt.offload_optimizer.nvme_path = state.offloadPath?.trim() || "none";
            }
        }

        if (state.zeroQuantizedWeights) {
            zeroOpt.zero_quantized_weights = true;
        }

        if (state.zeroQuantizedGradients) {
            zeroOpt.zero_quantized_gradients = true;
        }

        if (state.zeroQuantizedWeights && state.zeroHpzPartitionSize) {
            const partitionSize = toNumber(state.zeroHpzPartitionSize, null);
            if (partitionSize) {
                zeroOpt.zero_hpz_partition_size = partitionSize;
            }
        }

        const config = {
            zero_optimization: zeroOpt,
            gradient_accumulation_steps: toNumber(state.gradientAccumulationSteps, 1),
            steps_per_print: 2000,
            train_batch_size: "auto",
            train_micro_batch_size_per_gpu: "auto",
            wall_clock_breakdown: false,
        };

        const clipValue = state.gradientClipping;
        if (clipValue !== "" && clipValue !== null && clipValue !== undefined) {
            const parsed = Number(clipValue);
            if (Number.isFinite(parsed)) {
                config.gradient_clipping = parsed;
            }
        }

        if (state.precision === "fp16") {
            config.fp16 = {
                enabled: true,
                loss_scale: 0,
                loss_scale_window: 1000,
                initial_scale_power: 16,
                hysteresis: 2,
                min_loss_scale: 1,
            };
        } else if (state.precision === "bf16") {
            config.bf16 = {
                enabled: true,
            };
        }

        if (zeroStage === 3 && state.zero3Init) {
            config.zero3_init_flag = true;
        } else {
            delete config.zero3_init_flag;
        }

        if (zeroStage === 3 && state.zero3Save16) {
            config.zero3_save_16bit_model = true;
        } else {
            delete config.zero3_save_16bit_model;
        }

        if (state.includeOptimizer) {
            let optimizerType = state.optimizerType || "auto";
            if (optimizerType === "auto") {
                const offloadDevice = zeroOpt.offload_optimizer?.device || "";
                optimizerType = typeof offloadDevice === "string" && offloadDevice.toLowerCase() === "cpu"
                    ? "CPUAdam"
                    : "FusedAdam";
            } else if (optimizerType === "custom") {
                optimizerType = (state.optimizerCustomType || "").trim() || "FusedAdam";
            }

            const optimizerParams = {};
            const lrValue = parseOptionalNumber(state.optimizerLr);
            if (lrValue !== undefined) {
                optimizerParams.lr = lrValue;
            }
            const beta1 = parseOptionalNumber(state.optimizerBeta1);
            const beta2 = parseOptionalNumber(state.optimizerBeta2);
            if (beta1 !== undefined || beta2 !== undefined) {
                optimizerParams.betas = [beta1 ?? 0.9, beta2 ?? 0.999];
            }
            const epsValue = parseOptionalNumber(state.optimizerEps);
            if (epsValue !== undefined) {
                optimizerParams.eps = epsValue;
            }
            const weightDecayValue = parseOptionalNumber(state.optimizerWeightDecay);
            if (weightDecayValue !== undefined) {
                optimizerParams.weight_decay = weightDecayValue;
            }
            if (state.optimizerTorchAdam) {
                optimizerParams.torch_adam = true;
            }
            if (state.optimizerAdamWMode !== undefined && state.optimizerAdamWMode !== true) {
                optimizerParams.adam_w_mode = Boolean(state.optimizerAdamWMode);
            }

            config.optimizer = {
                type: optimizerType,
                params: optimizerParams,
            };
        }

        if (state.includeScheduler) {
            let schedulerType = state.schedulerType || "WarmupLR";
            if (schedulerType === "custom") {
                schedulerType = (state.schedulerCustomType || "").trim() || "WarmupLR";
            }
            const schedulerParams = {};
            const warmupMin = parseOptionalNumber(state.schedulerWarmupMinLr);
            if (warmupMin !== undefined) {
                schedulerParams.warmup_min_lr = warmupMin;
            }
            const warmupMax = parseOptionalNumber(state.schedulerWarmupMaxLr);
            if (warmupMax !== undefined) {
                schedulerParams.warmup_max_lr = warmupMax;
            }
            const warmupSteps = parseOptionalNumber(state.schedulerWarmupNumSteps);
            if (warmupSteps !== undefined) {
                schedulerParams.warmup_num_steps = warmupSteps;
            }

            config.scheduler = {
                type: schedulerType,
                params: schedulerParams,
            };
        }

        return config;
    }

    function extractStageFromConfig(config) {
        if (!config || typeof config !== "object") {
            return "stage2";
        }
        const fromZero = config.zero_optimization?.stage;
        if (fromZero !== undefined) {
            return normaliseStage(fromZero);
        }
        if (config.zero_stage !== undefined) {
            return normaliseStage(config.zero_stage);
        }
        return "stage2";
    }

    const pendingRequests = [];

    function ensureFieldId(field) {
        if (!field) {
            return null;
        }
        if (!field.id) {
            field.id = `deepspeed-field-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
        }
        return field.id;
    }

    function resolveField(targetCandidate) {
        if (!targetCandidate) {
            return null;
        }

        if (targetCandidate instanceof Element) {
            const id = ensureFieldId(targetCandidate);
            return { id, element: targetCandidate };
        }

        if (typeof targetCandidate === "string") {
            const element = document.getElementById(targetCandidate);
            if (element) {
                const id = ensureFieldId(element);
                return { id, element };
            }
            return { id: targetCandidate, element: null };
        }

        return null;
    }

    function requestOpen(targetCandidate) {
        const resolved = resolveField(targetCandidate);
        if (!resolved || !resolved.id) {
            console.warn("[DeepSpeed Builder] Could not resolve target field for DeepSpeed builder.");
            return;
        }

        const targetFieldId = resolved.id;

        if (window.__deepspeedBuilderInstance) {
            window.__deepspeedBuilderInstance.open(targetFieldId);
            return;
        }
        pendingRequests.push(targetFieldId);
        window.dispatchEvent(
            new CustomEvent("open-deepspeed-builder", {
                detail: { targetFieldId },
            }),
        );
    }

    window.__deepspeedBuilderQueue = pendingRequests;
    window.openDeepSpeedBuilder = requestOpen;

    window.openDeepSpeedBuilderFromButton = function (button) {
        if (!button) {
            return;
        }
        const explicitTarget = button.getAttribute("data-target-field");
        let field = explicitTarget ? document.getElementById(explicitTarget) : null;

        if (!field) {
            const container = button.closest(".mb-3, .form-group, .config-card, .col-md-6");
            if (container) {
                field =
                    container.querySelector("textarea[id]") ||
                    container.querySelector("textarea") ||
                    container.querySelector("input[id]") ||
                    container.querySelector("input");
            }
        }

        if (!field) {
            console.warn("[DeepSpeed Builder] Unable to locate DeepSpeed config field near button.");
            return;
        }

        requestOpen(field);
    };

    function registerAlpineComponent() {
        if (!window.Alpine || typeof window.Alpine.data !== "function") {
            return false;
        }

        window.Alpine.data("deepspeedBuilderComponent", () => ({
            isOpen: false,
            targetFieldId: null,
            builtOutput: "",
            warnings: [],
            isCustom: false,
            customContent: "",
            stage: DEFAULTS.stage,
            gradientAccumulationSteps: DEFAULTS.gradientAccumulationSteps,
            gradientClipping: DEFAULTS.gradientClipping,
            precision: DEFAULTS.precision,
            offloadParam: DEFAULTS.offloadParam,
            offloadOptimizer: DEFAULTS.offloadOptimizer,
            offloadPath: DEFAULTS.offloadPath,
            zero3Init: DEFAULTS.zero3Init,
            zero3Save16: DEFAULTS.zero3Save16,
            zero3Gather16: DEFAULTS.zero3Gather16,
            zeroQuantizedWeights: DEFAULTS.zeroQuantizedWeights,
            zeroQuantizedGradients: DEFAULTS.zeroQuantizedGradients,
            zeroHpzPartitionSize: DEFAULTS.zeroHpzPartitionSize,
            configFilePath: DEFAULTS.configFilePath,
            mode: DEFAULTS.mode,
            includeOptimizer: DEFAULTS.includeOptimizer,
            optimizerType: DEFAULTS.optimizerType,
            optimizerCustomType: DEFAULTS.optimizerCustomType,
            optimizerLr: DEFAULTS.optimizerLr,
            optimizerBeta1: DEFAULTS.optimizerBeta1,
            optimizerBeta2: DEFAULTS.optimizerBeta2,
            optimizerEps: DEFAULTS.optimizerEps,
            optimizerWeightDecay: DEFAULTS.optimizerWeightDecay,
            optimizerTorchAdam: DEFAULTS.optimizerTorchAdam,
            optimizerAdamWMode: DEFAULTS.optimizerAdamWMode,
            includeScheduler: DEFAULTS.includeScheduler,
            schedulerType: DEFAULTS.schedulerType,
            schedulerCustomType: DEFAULTS.schedulerCustomType,
            schedulerWarmupMinLr: DEFAULTS.schedulerWarmupMinLr,
            schedulerWarmupMaxLr: DEFAULTS.schedulerWarmupMaxLr,
            schedulerWarmupNumSteps: DEFAULTS.schedulerWarmupNumSteps,

            init() {
                window.__deepspeedBuilderInstance = this;
                this._builderEventHandler = (event) => {
                    if (event?.detail?.targetFieldId) {
                        this.open(event.detail.targetFieldId);
                    }
                };
                window.addEventListener("open-deepspeed-builder", this._builderEventHandler);
                if (Array.isArray(window.__deepspeedBuilderQueue) && window.__deepspeedBuilderQueue.length > 0) {
                    const queued = window.__deepspeedBuilderQueue.splice(0);
                    queued.forEach((id) => this.open(id));
                }
            },

            destroy() {
                if (this._builderEventHandler) {
                    window.removeEventListener("open-deepspeed-builder", this._builderEventHandler);
                }
                if (window.__deepspeedBuilderInstance === this) {
                    window.__deepspeedBuilderInstance = null;
                }
            },

            resetState() {
                const defaults = cloneDefaults();
                this.mode = defaults.mode;
                this.stage = defaults.stage;
                this.gradientAccumulationSteps = defaults.gradientAccumulationSteps;
                this.gradientClipping = defaults.gradientClipping;
                this.precision = defaults.precision;
                this.offloadParam = defaults.offloadParam;
                this.offloadOptimizer = defaults.offloadOptimizer;
                this.offloadPath = defaults.offloadPath;
                this.zero3Init = defaults.zero3Init;
                this.zero3Save16 = defaults.zero3Save16;
                this.zero3Gather16 = defaults.zero3Gather16;
                this.zeroQuantizedWeights = defaults.zeroQuantizedWeights;
                this.zeroQuantizedGradients = defaults.zeroQuantizedGradients;
                this.zeroHpzPartitionSize = defaults.zeroHpzPartitionSize;
                this.configFilePath = defaults.configFilePath;
                this.customContent = "";
                this.isCustom = false;
                this.warnings = [];
                this.builtOutput = "";
                this.includeOptimizer = defaults.includeOptimizer;
                this.optimizerType = defaults.optimizerType;
                this.optimizerCustomType = defaults.optimizerCustomType;
                this.optimizerLr = defaults.optimizerLr;
                this.optimizerBeta1 = defaults.optimizerBeta1;
                this.optimizerBeta2 = defaults.optimizerBeta2;
                this.optimizerEps = defaults.optimizerEps;
                this.optimizerWeightDecay = defaults.optimizerWeightDecay;
                this.optimizerTorchAdam = defaults.optimizerTorchAdam;
                this.optimizerAdamWMode = defaults.optimizerAdamWMode;
                this.includeScheduler = defaults.includeScheduler;
                this.schedulerType = defaults.schedulerType;
                this.schedulerCustomType = defaults.schedulerCustomType;
                this.schedulerWarmupMinLr = defaults.schedulerWarmupMinLr;
                this.schedulerWarmupMaxLr = defaults.schedulerWarmupMaxLr;
                this.schedulerWarmupNumSteps = defaults.schedulerWarmupNumSteps;
            },

            open(targetFieldId) {
                this.targetFieldId = targetFieldId;
                const field = document.getElementById(targetFieldId);
                if (!field) {
                    console.warn("[DeepSpeed Builder] Could not locate field", targetFieldId);
                    return;
                }
                this.resetState();
                this.importFromField(field.value);
                this.isOpen = true;
                document.body.classList.add("builder-modal-open");
            },

            close() {
                this.isOpen = false;
                this.targetFieldId = null;
                document.body.classList.remove("builder-modal-open");
            },

            importFromField(raw) {
                const value = (raw || "").trim();
                if (!value) {
                    this.recompute();
                    return;
                }
                if (value.startsWith("{") || value.startsWith("[")) {
                    try {
                        const parsed = JSON.parse(value);
                        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                            if (parsed.deepspeed_config_file) {
                                this.mode = "configFile";
                                this.configFilePath = parsed.deepspeed_config_file || "";
                                if (typeof parsed.zero3_init_flag === "boolean") {
                                    this.zero3Init = parsed.zero3_init_flag;
                                }
                                if (typeof parsed.zero3_save_16bit_model === "boolean") {
                                    this.zero3Save16 = parsed.zero3_save_16bit_model;
                                }
                                if (typeof parsed.zero3_gather_16bit_weights_on_model_save === "boolean") {
                                    this.zero3Gather16 = parsed.zero3_gather_16bit_weights_on_model_save;
                                }
                            } else {
                                this.mode = "inline";
                                this.populateInlineFromConfig(parsed);
                            }
                            this.isCustom = false;
                        } else {
                            this.mode = "inline";
                            this.isCustom = true;
                            this.customContent = value;
                        }
                    } catch (error) {
                        console.warn("[DeepSpeed Builder] Unable to parse JSON:", error);
                        this.mode = "inline";
                        this.isCustom = true;
                        this.customContent = value;
                    }
                } else {
                    this.mode = "configFile";
                    this.configFilePath = value;
                }
                this.recompute();
            },

            populateInlineFromConfig(config) {
                this.stage = extractStageFromConfig(config);

                if (typeof config.gradient_accumulation_steps === "number") {
                    this.gradientAccumulationSteps = config.gradient_accumulation_steps;
                } else if (typeof config.zero_optimization?.gradient_accumulation_steps === "number") {
                    this.gradientAccumulationSteps = config.zero_optimization.gradient_accumulation_steps;
                }

                if (typeof config.gradient_clipping === "number") {
                    this.gradientClipping = config.gradient_clipping;
                } else {
                    this.gradientClipping = "";
                }

                if (config.fp16?.enabled) {
                    this.precision = "fp16";
                } else if (config.bf16?.enabled) {
                    this.precision = "bf16";
                } else {
                    this.precision = "none";
                }

                const zeroOpt = config.zero_optimization || {};
                if (zeroOpt.offload_param?.device) {
                    this.offloadParam = zeroOpt.offload_param.device.toLowerCase();
                    if (zeroOpt.offload_param.nvme_path) {
                        this.offloadPath = zeroOpt.offload_param.nvme_path;
                    }
                }
                if (zeroOpt.offload_optimizer?.device) {
                    this.offloadOptimizer = zeroOpt.offload_optimizer.device.toLowerCase();
                    if (!this.offloadPath && zeroOpt.offload_optimizer.nvme_path) {
                        this.offloadPath = zeroOpt.offload_optimizer.nvme_path;
                    }
                }

                if (typeof config.zero3_init_flag === "boolean") {
                    this.zero3Init = config.zero3_init_flag;
                } else if (typeof zeroOpt.zero3_init_flag === "boolean") {
                    this.zero3Init = zeroOpt.zero3_init_flag;
                } else {
                    this.zero3Init = this.stage === "stage3";
                }

                if (typeof config.zero3_save_16bit_model === "boolean") {
                    this.zero3Save16 = config.zero3_save_16bit_model;
                } else if (typeof zeroOpt.zero3_save_16bit_model === "boolean") {
                    this.zero3Save16 = zeroOpt.zero3_save_16bit_model;
                } else {
                    this.zero3Save16 = false;
                }

                if (typeof zeroOpt.stage3_gather_16bit_weights_on_model_save === "boolean") {
                    this.zero3Gather16 = zeroOpt.stage3_gather_16bit_weights_on_model_save;
                } else {
                    this.zero3Gather16 = true;
                }

                if (typeof zeroOpt.zero_quantized_weights === "boolean") {
                    this.zeroQuantizedWeights = zeroOpt.zero_quantized_weights;
                }
                if (typeof zeroOpt.zero_quantized_gradients === "boolean") {
                    this.zeroQuantizedGradients = zeroOpt.zero_quantized_gradients;
                }
                if (typeof zeroOpt.zero_hpz_partition_size === "number") {
                    this.zeroHpzPartitionSize = zeroOpt.zero_hpz_partition_size;
                }

                const optimizerConfig = config.optimizer;
                if (optimizerConfig && typeof optimizerConfig === "object") {
                    this.includeOptimizer = true;
                    const optTypeRaw = optimizerConfig.type || optimizerConfig.name || "FusedAdam";
                    const normalisedOptType = String(optTypeRaw).toLowerCase();
                    const knownTypes = {
                        auto: "auto",
                        adam: "Adam",
                        adamw: "AdamW",
                        fusedadam: "FusedAdam",
                        cpuadam: "CPUAdam",
                        onebitadam: "OneBitAdam",
                        onebitlamb: "OneBitLamb",
                        zerooneadam: "ZeroOneAdam",
                    };
                    if (normalisedOptType in knownTypes) {
                        this.optimizerType = knownTypes[normalisedOptType];
                        this.optimizerCustomType = "";
                    } else {
                        this.optimizerType = "custom";
                        this.optimizerCustomType = String(optTypeRaw || "").trim();
                    }
                    const params = optimizerConfig.params || {};
                    if (params.lr !== undefined) {
                        this.optimizerLr = params.lr;
                    }
                    if (Array.isArray(params.betas) && params.betas.length >= 2) {
                        this.optimizerBeta1 = params.betas[0];
                        this.optimizerBeta2 = params.betas[1];
                    } else {
                        this.optimizerBeta1 = DEFAULTS.optimizerBeta1;
                        this.optimizerBeta2 = DEFAULTS.optimizerBeta2;
                    }
                    if (params.eps !== undefined) {
                        this.optimizerEps = params.eps;
                    } else {
                        this.optimizerEps = DEFAULTS.optimizerEps;
                    }
                    if (params.weight_decay !== undefined) {
                        this.optimizerWeightDecay = params.weight_decay;
                    } else {
                        this.optimizerWeightDecay = DEFAULTS.optimizerWeightDecay;
                    }
                    if (params.torch_adam !== undefined) {
                        this.optimizerTorchAdam = Boolean(params.torch_adam);
                    } else {
                        this.optimizerTorchAdam = DEFAULTS.optimizerTorchAdam;
                    }
                    if (params.adam_w_mode !== undefined) {
                        this.optimizerAdamWMode = Boolean(params.adam_w_mode);
                    } else {
                        this.optimizerAdamWMode = DEFAULTS.optimizerAdamWMode;
                    }
                } else {
                    this.includeOptimizer = DEFAULTS.includeOptimizer;
                    this.optimizerType = DEFAULTS.optimizerType;
                    this.optimizerCustomType = DEFAULTS.optimizerCustomType;
                    this.optimizerLr = DEFAULTS.optimizerLr;
                    this.optimizerBeta1 = DEFAULTS.optimizerBeta1;
                    this.optimizerBeta2 = DEFAULTS.optimizerBeta2;
                    this.optimizerEps = DEFAULTS.optimizerEps;
                    this.optimizerWeightDecay = DEFAULTS.optimizerWeightDecay;
                    this.optimizerTorchAdam = DEFAULTS.optimizerTorchAdam;
                    this.optimizerAdamWMode = DEFAULTS.optimizerAdamWMode;
                }

                const schedulerConfig = config.scheduler;
                if (schedulerConfig && typeof schedulerConfig === "object") {
                    this.includeScheduler = true;
                    const schedTypeRaw = schedulerConfig.type || "WarmupLR";
                    const normalisedSched = String(schedTypeRaw).toLowerCase();
                    const knownSchedulers = {
                        warmuplr: "WarmupLR",
                        onecycle: "OneCycle",
                        polynomial: "Polynomial",
                        constant: "Constant",
                    };
                    if (normalisedSched in knownSchedulers) {
                        this.schedulerType = knownSchedulers[normalisedSched];
                        this.schedulerCustomType = "";
                    } else {
                        this.schedulerType = "custom";
                        this.schedulerCustomType = String(schedTypeRaw || "").trim();
                    }
                    const params = schedulerConfig.params || {};
                    if (params.warmup_min_lr !== undefined) {
                        this.schedulerWarmupMinLr = params.warmup_min_lr;
                    } else {
                        this.schedulerWarmupMinLr = DEFAULTS.schedulerWarmupMinLr;
                    }
                    if (params.warmup_max_lr !== undefined) {
                        this.schedulerWarmupMaxLr = params.warmup_max_lr;
                    } else {
                        this.schedulerWarmupMaxLr = DEFAULTS.schedulerWarmupMaxLr;
                    }
                    if (params.warmup_num_steps !== undefined) {
                        this.schedulerWarmupNumSteps = params.warmup_num_steps;
                    } else {
                        this.schedulerWarmupNumSteps = DEFAULTS.schedulerWarmupNumSteps;
                    }
                } else {
                    this.includeScheduler = DEFAULTS.includeScheduler;
                    this.schedulerType = DEFAULTS.schedulerType;
                    this.schedulerCustomType = DEFAULTS.schedulerCustomType;
                    this.schedulerWarmupMinLr = DEFAULTS.schedulerWarmupMinLr;
                    this.schedulerWarmupMaxLr = DEFAULTS.schedulerWarmupMaxLr;
                    this.schedulerWarmupNumSteps = DEFAULTS.schedulerWarmupNumSteps;
                }
            },

            recompute() {
                if (this.mode === "configFile") {
                    const result = {};
                    const path = (this.configFilePath || "").trim();
                    if (path) {
                        result.deepspeed_config_file = path;
                    }
                    if (this.zero3Init) {
                        result.zero3_init_flag = true;
                    }
                    if (this.zero3Save16) {
                        result.zero3_save_16bit_model = true;
                    }
                    if (typeof this.zero3Gather16 === "boolean") {
                        result.zero3_gather_16bit_weights_on_model_save = this.zero3Gather16;
                    }
                    const serialized = Object.keys(result).length ? JSON.stringify(result, null, 2) : "";
                    if (this.builtOutput !== serialized) {
                        this.builtOutput = serialized;
                    }
                    return;
                }

                if (this.isCustom) {
                    this.builtOutput = this.customContent;
                    return;
                }

                const config = buildInlineConfig(this);
                const serialized = JSON.stringify(config, null, 2);
                if (this.builtOutput !== serialized) {
                    this.builtOutput = serialized;
                }
            },

            setMode(mode) {
                if (this.mode === mode) {
                    return;
                }
                this.mode = mode;
                if (mode === "configFile" && !this.configFilePath) {
                    this.configFilePath = "";
                }
                if (mode === "inline" && this.isCustom) {
                    // keep custom content
                }
                this.recompute();
            },

            handleCustomInput(event) {
                this.customContent = event?.target?.value || "";
                this.isCustom = true;
                this.recompute();
            },

            resetToBuilderDefaults() {
                this.resetState();
                this.recompute();
            },

            ensureNumber(field, fallback) {
                const value = Number(this[field]);
                if (!Number.isFinite(value)) {
                    this[field] = fallback;
                }
            },

            canApply() {
                if (this.mode === "configFile") {
                    return Boolean((this.configFilePath || "").trim());
                }
                if (this.isCustom) {
                    return Boolean((this.customContent || "").trim());
                }
                return Boolean((this.builtOutput || "").trim());
            },

            apply() {
                if (!this.canApply()) {
                    window.showToast?.("Please complete the DeepSpeed configuration first.", "warning");
                    return;
                }
                const field = document.getElementById(this.targetFieldId);
                if (!field) {
                    window.showToast?.("Could not locate DeepSpeed configuration field.", "error");
                    this.close();
                    return;
                }
                const value =
                    this.mode === "configFile"
                        ? this.builtOutput
                        : this.isCustom
                          ? this.customContent
                          : this.builtOutput;
                field.value = value;
                field.dispatchEvent(new Event("input", { bubbles: true }));
                field.dispatchEvent(new Event("change", { bubbles: true }));
                const trainerStore = window.Alpine?.store?.("trainer");
                if (trainerStore && typeof trainerStore.markFormDirty === "function") {
                    trainerStore.markFormDirty();
                }
                window.showToast?.("DeepSpeed configuration updated.", "success");
                this.close();
            },

            copyPreview() {
                const payload = this.builtOutput || this.customContent || "";
                if (!payload) {
                    return;
                }
                if (navigator.clipboard?.writeText) {
                    navigator.clipboard
                        .writeText(payload)
                        .then(() => window.showToast?.("DeepSpeed config copied to clipboard.", "success"))
                        .catch(() => window.showToast?.("Unable to access clipboard.", "warning"));
                }
            },
        }));
        return true;
    }

    document.addEventListener("alpine:init", () => {
        registerAlpineComponent();
    });

    if (document.readyState === "complete" || document.readyState === "interactive") {
        setTimeout(() => {
            if (!registerAlpineComponent()) {
                document.addEventListener(
                    "DOMContentLoaded",
                    () => {
                        registerAlpineComponent();
                    },
                    { once: true },
                );
            }
        }, 0);
    }
})();
