(function () {
    if (window.ensureDeepSpeedBuilderAssets) {
        return;
    }

    const CSS_HREF = "/static/css/deepspeed-builder.css";
    const JS_SRC = "/static/js/deepspeed-builder.js";

    function ensureCssLoaded() {
        if (document.querySelector('link[data-deepspeed-builder-css]')) {
            return;
        }
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = CSS_HREF;
        link.dataset.deepspeedBuilderCss = "true";
        document.head.appendChild(link);
    }

    function loadScript() {
        return new Promise((resolve, reject) => {
            if (document.querySelector('script[data-deepspeed-builder-script]')) {
                resolve();
                return;
            }

            const script = document.createElement("script");
            script.src = JS_SRC;
            script.defer = true;
            script.dataset.deepspeedBuilderScript = "true";
            script.onload = resolve;
            script.onerror = () => reject(new Error("Failed to load DeepSpeed builder script"));

            document.head.appendChild(script);
        });
    }

    function ensureAssets() {
        ensureCssLoaded();

        if (typeof window.openDeepSpeedBuilder === "function" || typeof window.openDeepSpeedBuilderFromButton === "function") {
            return Promise.resolve();
        }

        if (!window.__deepspeedBuilderScriptPromise) {
            window.__deepspeedBuilderScriptPromise = loadScript()
                .finally(() => {
                    window.__deepspeedBuilderScriptPromise = null;
                });
        }

        return window.__deepspeedBuilderScriptPromise;
    }

    function handleError(error) {
        console.warn("[DeepSpeed Builder] Asset load failed:", error);
        if (typeof window.showToast === "function") {
            window.showToast("Failed to load DeepSpeed builder assets.", "error");
        }
    }

    window.ensureDeepSpeedBuilderAssets = function ensureDeepSpeedBuilderAssets() {
        return ensureAssets().catch(handleError);
    };

    window.launchDeepSpeedBuilderButton = function launchDeepSpeedBuilderButton(button) {
        if (!button) {
            return;
        }

        ensureAssets()
            .then(() => {
                if (typeof window.openDeepSpeedBuilderFromButton === "function") {
                    window.openDeepSpeedBuilderFromButton(button);
                }
            })
            .catch(handleError);
    };

    window.launchDeepSpeedBuilder = function launchDeepSpeedBuilder(target) {
        if (!target) {
            return;
        }

        ensureAssets()
            .then(() => {
                const resolvedTarget = target instanceof Element ? target : document.getElementById(target) || target;
                if (typeof window.openDeepSpeedBuilder === "function") {
                    window.openDeepSpeedBuilder(resolvedTarget);
                } else if (typeof window.openDeepSpeedBuilderFromButton === "function" && resolvedTarget instanceof Element) {
                    window.openDeepSpeedBuilderFromButton(resolvedTarget);
                }
            })
            .catch(handleError);
    };

    if (document.readyState === "complete" || document.readyState === "interactive") {
        ensureCssLoaded();
    } else {
        document.addEventListener("DOMContentLoaded", ensureCssLoaded, { once: true });
    }
})();
