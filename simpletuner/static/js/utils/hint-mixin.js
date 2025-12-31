/**
 * HintMixin - Unified hint/dismissal system
 *
 * Consolidates hint management patterns across components.
 * Supports both API-backed (server-persisted) and localStorage modes.
 *
 * Usage:
 *
 * // API-backed single hint (for hero CTAs):
 * const hintManager = window.HintMixin.createSingleHint({
 *     useApi: true,
 *     hintKey: 'orgs_hero',
 * });
 * // Returns: { showHeroCTA: true, loadHeroCTAState(), dismissHeroCTA(), restoreHeroCTA() }
 *
 * // localStorage single hint:
 * const hintManager = window.HintMixin.createSingleHint({
 *     useApi: false,
 *     storageKey: 'notifications_hero_dismissed',
 * });
 *
 * // API-backed multiple hints (for admin panels):
 * const hintManager = window.HintMixin.createMultiHint({
 *     useApi: true,
 *     prefix: 'admin_',
 *     hintKeys: ['overview', 'users', 'levels'],
 * });
 * // Returns: { hints: {...}, loadHints(), dismissHint(key), showHint(key), restoreAllHints() }
 *
 * // localStorage multiple hints:
 * const hintManager = window.HintMixin.createMultiHint({
 *     useApi: false,
 *     storageKey: 'cloud_hints',
 *     hintKeys: ['dataloader', 'git'],
 * });
 */

window.HintMixin = {
    /**
     * Create a single hint manager (showHeroCTA pattern)
     */
    createSingleHint(options) {
        const { useApi = true, hintKey = '', storageKey = '' } = options;

        return {
            showHeroCTA: true,

            async loadHeroCTAState() {
                if (useApi) {
                    try {
                        const response = await fetch('/api/cloud/hints');
                        if (response.ok) {
                            const data = await response.json();
                            const dismissed = data.dismissed_hints || [];
                            this.showHeroCTA = !dismissed.includes(hintKey);
                        }
                    } catch (error) {
                        console.warn('Failed to load hint state:', error);
                    }
                } else {
                    this.showHeroCTA = localStorage.getItem(storageKey) !== 'true';
                }
            },

            async dismissHeroCTA() {
                this.showHeroCTA = false;
                if (useApi) {
                    try {
                        await fetch(`/api/cloud/hints/dismiss/${hintKey}`, { method: 'POST' });
                    } catch (error) {
                        console.warn('Failed to dismiss hint:', error);
                    }
                } else {
                    localStorage.setItem(storageKey, 'true');
                }
            },

            async restoreHeroCTA() {
                this.showHeroCTA = true;
                if (useApi) {
                    try {
                        await fetch(`/api/cloud/hints/show/${hintKey}`, { method: 'POST' });
                    } catch (error) {
                        console.warn('Failed to restore hint:', error);
                    }
                } else {
                    localStorage.removeItem(storageKey);
                }
            },
        };
    },

    /**
     * Create a multi-hint manager (hints object pattern)
     */
    createMultiHint(options) {
        const { useApi = true, prefix = '', storageKey = '', hintKeys = [] } = options;

        // Initialize hints object with all keys set to true (visible)
        const hints = {};
        hintKeys.forEach(key => {
            hints[key] = true;
        });

        return {
            hints,
            hintsLoading: false,

            async loadHints() {
                this.hintsLoading = true;
                try {
                    if (useApi) {
                        const response = await fetch('/api/cloud/hints');
                        if (response.ok) {
                            const data = await response.json();
                            const dismissed = data.dismissed_hints || [];
                            hintKeys.forEach(key => {
                                this.hints[key] = !dismissed.includes(prefix + key);
                            });
                        }
                    } else {
                        const stored = localStorage.getItem(storageKey);
                        if (stored) {
                            try {
                                const parsed = JSON.parse(stored);
                                hintKeys.forEach(key => {
                                    if (parsed[key + '_dismissed'] !== undefined) {
                                        this.hints[key] = !parsed[key + '_dismissed'];
                                    }
                                });
                            } catch (e) {
                                console.warn('Failed to parse hints from localStorage');
                            }
                        }
                    }
                } catch (error) {
                    console.warn('Failed to load hints:', error);
                } finally {
                    this.hintsLoading = false;
                }
            },

            async dismissHint(key) {
                this.hints[key] = false;
                if (useApi) {
                    try {
                        await fetch(`/api/cloud/hints/dismiss/${prefix}${key}`, { method: 'POST' });
                    } catch (error) {
                        console.warn('Failed to dismiss hint:', error);
                    }
                } else {
                    this._saveHintsToStorage();
                }
            },

            async showHint(key) {
                this.hints[key] = true;
                if (useApi) {
                    try {
                        await fetch(`/api/cloud/hints/show/${prefix}${key}`, { method: 'POST' });
                    } catch (error) {
                        console.warn('Failed to show hint:', error);
                    }
                } else {
                    this._saveHintsToStorage();
                }
            },

            restoreAllHints() {
                hintKeys.forEach(key => this.showHint(key));
            },

            // Note: This must be a method, not a getter, because getters are
            // evaluated once when spread with {...} and become static values.
            anyHintsDismissed() {
                return hintKeys.some(key => !this.hints[key]);
            },

            _saveHintsToStorage() {
                if (useApi) return;
                const toStore = {};
                hintKeys.forEach(key => {
                    toStore[key + '_dismissed'] = !this.hints[key];
                });
                localStorage.setItem(storageKey, JSON.stringify(toStore));
            },
        };
    },
};
