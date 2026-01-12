/**
 * Cloud Dashboard - Onboarding Module
 *
 * Handles progressive disclosure onboarding flow, hints, and setup checklist.
 */

window.cloudOnboardingMethods = {
    loadHints() {
        const stored = localStorage.getItem('cloud_hints');
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                Object.assign(this.hints, parsed);
            } catch (e) {
                console.warn('Failed to parse hints from localStorage');
            }
        }
    },

    dismissHint(hintKey) {
        this.hints[hintKey + '_dismissed'] = true;
        this.saveHints();
    },

    saveHints() {
        localStorage.setItem('cloud_hints', JSON.stringify(this.hints));
    },

    restoreHints() {
        this.hints = {
            dataloader_dismissed: false,
            git_dismissed: false,
        };
        this.saveHints();
    },

    showHint(hintKey) {
        // Un-dismiss a specific hint
        this.hints[hintKey + '_dismissed'] = false;
        this.saveHints();
    },

    loadOnboardingState() {
        const stored = localStorage.getItem('cloud_onboarding');
        if (stored) {
            try {
                const parsed = JSON.parse(stored);
                Object.assign(this.onboarding, parsed);
            } catch (e) {
                console.warn('Failed to parse onboarding state from localStorage');
            }
        }
    },

    saveOnboardingState() {
        localStorage.setItem('cloud_onboarding', JSON.stringify(this.onboarding));
    },

    markOnboardingStep(step) {
        this.onboarding[step] = true;
        this.saveOnboardingState();
    },

    completeOnboardingStep(step) {
        const stepMapping = {
            'data': 'data_understood',
            'results': 'results_understood',
            'cost': 'cost_understood',
        };
        const stateKey = stepMapping[step];
        if (stateKey) {
            this.onboarding[stateKey] = true;
            this.saveOnboardingState();
        }
    },

    async completeOnboarding() {
        this.onboarding.cost_understood = true;
        this.saveOnboardingState();

        if (this.quickCostLimitEnabled) {
            await this.saveQuickCostLimit();
        }

        if (window.showToast) {
            window.showToast('Onboarding complete! You can now submit cloud training jobs.', 'success');
        }
    },

    resetOnboarding() {
        this.onboarding = {
            data_understood: false,
            results_understood: false,
            cost_understood: false,
        };
        this.saveOnboardingState();
    },

    skipOnboarding() {
        // Skip all onboarding steps at once (for users who want to proceed quickly)
        this.onboarding.data_understood = true;
        this.onboarding.results_understood = true;
        this.onboarding.cost_understood = true;
        this.saveOnboardingState();
        if (window.showToast) {
            window.showToast('Onboarding skipped. You can restart it from Settings.', 'info');
        }
    },

    async saveQuickCostLimit() {
        if (!this.quickCostLimitEnabled) return;

        try {
            const response = await fetch('/api/cloud/providers/replicate/config', {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    cost_limit_enabled: true,
                    cost_limit_amount: this.quickCostLimitAmount,
                    cost_limit_period: this.quickCostLimitPeriod,
                }),
            });

            if (response.ok) {
                await this.loadCostLimitStatus();
                if (window.showToast) {
                    window.showToast('Cost limit configured', 'success');
                }
            }
        } catch (error) {
            console.error('Failed to save cost limit:', error);
        }
    },
};
