(function () {
    const state = {
        metadata: null,
        inflight: null,
    };

    const normalizeOptionValue = (value) => {
        if (value === null || value === undefined) {
            return undefined;
        }
        if (typeof value === "string") {
            const trimmed = value.trim();
            if (!trimmed) {
                return undefined;
            }
            const lowered = trimmed.toLowerCase();
            if (lowered === "true") {
                return true;
            }
            if (lowered === "false") {
                return false;
            }
            if (!Number.isNaN(Number(trimmed))) {
                const numeric = Number(trimmed);
                return Number.isInteger(numeric) ? parseInt(trimmed, 10) : numeric;
            }
            return value;
        }
        return value;
    };

    const ensureOptionArray = (options) => {
        if (!Array.isArray(options) || options.length === 0) {
            return [{ key: "", value: "" }];
        }
        return options.map((entry) => ({
            key: entry && entry.key !== undefined ? entry.key : "",
            value: entry && entry.value !== undefined ? entry.value : "",
        }));
    };

    const toOverrideArray = (mapping) => {
        const rows = [];
        if (mapping && typeof mapping === "object") {
            Object.entries(mapping).forEach(([key, payload]) => {
                if (key == null) {
                    return;
                }
                const row = {
                    key,
                    algo: (payload && payload.algo) || "",
                    options: [],
                };
                if (payload && typeof payload === "object") {
                    Object.entries(payload).forEach(([optionKey, optionValue]) => {
                        if (optionKey === "algo") {
                            return;
                        }
                        row.options.push({
                            key: optionKey,
                            value: optionValue,
                        });
                    });
                }
                row.options = ensureOptionArray(row.options);
                rows.push(row);
            });
        }
        if (!rows.length) {
            rows.push({
                key: "",
                algo: "",
                options: [{ key: "", value: "" }],
            });
        }
        return rows;
    };

    const serializeOverrides = (rows) => {
        const result = {};
        if (!Array.isArray(rows)) {
            return result;
        }
        rows.forEach((row) => {
            const overrideKey = (row && row.key ? row.key : "").trim();
            if (!overrideKey) {
                return;
            }
            const payload = {};
            const algo = (row && row.algo ? row.algo : "").trim();
            if (algo) {
                payload.algo = algo;
            }
            const options = Array.isArray(row && row.options ? row.options : []) ? row.options : [];
            options.forEach((option) => {
                const optionKey = (option && option.key ? option.key : "").trim();
                if (!optionKey) {
                    return;
                }
                const normalized = normalizeOptionValue(option ? option.value : undefined);
                if (normalized !== undefined) {
                    payload[optionKey] = normalized;
                }
            });
            if (Object.keys(payload).length > 0) {
                result[overrideKey] = payload;
            }
        });
        return result;
    };

    const fetchMetadata = async (force = false) => {
        if (state.metadata && !force) {
            return state.metadata;
        }
        if (state.inflight) {
            return state.inflight;
        }
        const url = force ? "/api/lycoris/metadata?force_refresh=true" : "/api/lycoris/metadata";
        state.inflight = fetch(url)
            .then(async (response) => {
                if (!response.ok) {
                    throw new Error("Failed to load LyCORIS metadata");
                }
                const data = await response.json();
                state.metadata = data;
                state.inflight = null;
                return data;
            })
            .catch((error) => {
                state.inflight = null;
                throw error;
            });
        return state.inflight;
    };

    window.lycorisConfigurator = {
        fetchMetadata,
        getMetadata: () => state.metadata,
        toOverrideArray,
        serializeOverrides,
        normalizeOptionValue,
        createEmptyOverride: () => ({ key: "", algo: "", options: [{ key: "", value: "" }] }),
        createEmptyOption: () => ({ key: "", value: "" }),
    };
})();
