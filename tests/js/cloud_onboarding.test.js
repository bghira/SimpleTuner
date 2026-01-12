/**
 * Tests for cloud onboarding module.
 *
 * Tests the onboarding flow, hint management, and progressive disclosure.
 */

// Mock localStorage
const localStorageMock = (() => {
    let store = {};
    return {
        getItem: jest.fn((key) => store[key] || null),
        setItem: jest.fn((key, value) => {
            store[key] = value;
        }),
        removeItem: jest.fn((key) => {
            delete store[key];
        }),
        clear: jest.fn(() => {
            store = {};
        }),
    };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Load the onboarding module
require('../../simpletuner/static/js/modules/cloud/onboarding.js');

describe('cloudOnboardingMethods', () => {
    let context;

    beforeEach(() => {
        // Reset localStorage mock
        localStorageMock.clear();
        jest.clearAllMocks();

        // Create a fresh context with default state
        context = {
            hints: {
                dataloader_dismissed: false,
                git_dismissed: false,
            },
            onboarding: {
                data_understood: false,
                results_understood: false,
                cost_understood: false,
            },
            quickCostLimitEnabled: false,
            quickCostLimitAmount: 50,
            quickCostLimitPeriod: 'monthly',
            advancedConfig: {
                saving: false,
                ssl_verify: true,
                sslWarningAcknowledged: false,
                webhook_allowed_ips: [],
                webhook_ip_allowlist_enabled: false,
                newIpEntry: '',
                ipValidationError: null,
            },
        };

        // Bind methods to context
        Object.keys(window.cloudOnboardingMethods).forEach((key) => {
            if (typeof window.cloudOnboardingMethods[key] === 'function') {
                context[key] = window.cloudOnboardingMethods[key].bind(context);
            }
        });
    });

    describe('hint management', () => {
        test('loadHints loads from localStorage', () => {
            const storedHints = {
                dataloader_dismissed: true,
                git_dismissed: false,
            };
            localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(storedHints));

            context.loadHints();

            expect(localStorageMock.getItem).toHaveBeenCalledWith('cloud_hints');
            expect(context.hints.dataloader_dismissed).toBe(true);
        });

        test('loadHints handles invalid JSON gracefully', () => {
            localStorageMock.getItem.mockReturnValueOnce('invalid json');

            // Should not throw
            expect(() => context.loadHints()).not.toThrow();
            // Hints should remain at defaults
            expect(context.hints.dataloader_dismissed).toBe(false);
        });

        test('loadHints handles null localStorage', () => {
            localStorageMock.getItem.mockReturnValueOnce(null);

            expect(() => context.loadHints()).not.toThrow();
            expect(context.hints.dataloader_dismissed).toBe(false);
        });

        test('dismissHint marks hint as dismissed and saves', () => {
            context.dismissHint('dataloader');

            expect(context.hints.dataloader_dismissed).toBe(true);
            expect(localStorageMock.setItem).toHaveBeenCalledWith(
                'cloud_hints',
                expect.stringContaining('dataloader_dismissed')
            );
        });

        test('restoreHints resets all hints', () => {
            context.hints.dataloader_dismissed = true;
            context.hints.git_dismissed = true;

            context.restoreHints();

            expect(context.hints.dataloader_dismissed).toBe(false);
            expect(context.hints.git_dismissed).toBe(false);
            expect(localStorageMock.setItem).toHaveBeenCalled();
        });

        test('showHint un-dismisses a specific hint', () => {
            context.hints.dataloader_dismissed = true;

            context.showHint('dataloader');

            expect(context.hints.dataloader_dismissed).toBe(false);
        });
    });

    describe('onboarding state', () => {
        test('loadOnboardingState loads from localStorage', () => {
            const storedState = {
                data_understood: true,
                results_understood: false,
                cost_understood: false,
            };
            localStorageMock.getItem.mockReturnValueOnce(JSON.stringify(storedState));

            context.loadOnboardingState();

            expect(localStorageMock.getItem).toHaveBeenCalledWith('cloud_onboarding');
            expect(context.onboarding.data_understood).toBe(true);
        });

        test('loadOnboardingState handles invalid JSON', () => {
            localStorageMock.getItem.mockReturnValueOnce('not json');

            expect(() => context.loadOnboardingState()).not.toThrow();
            expect(context.onboarding.data_understood).toBe(false);
        });

        test('saveOnboardingState persists to localStorage', () => {
            context.onboarding.data_understood = true;

            context.saveOnboardingState();

            expect(localStorageMock.setItem).toHaveBeenCalledWith(
                'cloud_onboarding',
                expect.stringContaining('"data_understood":true')
            );
        });

        test('markOnboardingStep sets step and saves', () => {
            context.markOnboardingStep('data_understood');

            expect(context.onboarding.data_understood).toBe(true);
            expect(localStorageMock.setItem).toHaveBeenCalled();
        });
    });

    describe('completeOnboardingStep', () => {
        test('maps "data" to data_understood', () => {
            context.completeOnboardingStep('data');

            expect(context.onboarding.data_understood).toBe(true);
            expect(localStorageMock.setItem).toHaveBeenCalled();
        });

        test('maps "results" to results_understood', () => {
            context.completeOnboardingStep('results');

            expect(context.onboarding.results_understood).toBe(true);
        });

        test('maps "cost" to cost_understood', () => {
            context.completeOnboardingStep('cost');

            expect(context.onboarding.cost_understood).toBe(true);
        });

        test('handles unknown step gracefully', () => {
            context.completeOnboardingStep('unknown');

            // Should not throw, should not change state
            expect(context.onboarding.data_understood).toBe(false);
        });
    });

    describe('skipOnboarding', () => {
        test('marks all steps as complete', () => {
            context.skipOnboarding();

            expect(context.onboarding.data_understood).toBe(true);
            expect(context.onboarding.results_understood).toBe(true);
            expect(context.onboarding.cost_understood).toBe(true);
        });

        test('saves state to localStorage', () => {
            context.skipOnboarding();

            expect(localStorageMock.setItem).toHaveBeenCalled();
        });

        test('shows toast if available', () => {
            window.showToast = jest.fn();

            context.skipOnboarding();

            expect(window.showToast).toHaveBeenCalledWith(
                expect.stringContaining('skipped'),
                'info'
            );

            delete window.showToast;
        });
    });

    describe('resetOnboarding', () => {
        test('resets all steps to incomplete', () => {
            context.onboarding.data_understood = true;
            context.onboarding.results_understood = true;
            context.onboarding.cost_understood = true;

            context.resetOnboarding();

            expect(context.onboarding.data_understood).toBe(false);
            expect(context.onboarding.results_understood).toBe(false);
            expect(context.onboarding.cost_understood).toBe(false);
        });
    });

});

describe('Onboarding Step Order', () => {
    test('steps must be completed in order', () => {
        const onboarding = {
            data_understood: false,
            results_understood: false,
            cost_understood: false,
        };

        function canCompleteStep(step, state) {
            if (step === 'data') return true;
            if (step === 'results') return state.data_understood;
            if (step === 'cost') return state.data_understood && state.results_understood;
            return false;
        }

        // Initially only data step can be completed
        expect(canCompleteStep('data', onboarding)).toBe(true);
        expect(canCompleteStep('results', onboarding)).toBe(false);
        expect(canCompleteStep('cost', onboarding)).toBe(false);

        // After data step
        onboarding.data_understood = true;
        expect(canCompleteStep('results', onboarding)).toBe(true);
        expect(canCompleteStep('cost', onboarding)).toBe(false);

        // After results step
        onboarding.results_understood = true;
        expect(canCompleteStep('cost', onboarding)).toBe(true);
    });

    test('all steps complete indicates onboarding finished', () => {
        function isOnboardingComplete(state) {
            return state.data_understood &&
                   state.results_understood &&
                   state.cost_understood;
        }

        expect(isOnboardingComplete({
            data_understood: true,
            results_understood: true,
            cost_understood: false,
        })).toBe(false);

        expect(isOnboardingComplete({
            data_understood: true,
            results_understood: true,
            cost_understood: true,
        })).toBe(true);
    });
});
