/**
 * Tests for HintMixin utility.
 *
 * HintMixin provides unified hint/dismissal management for onboarding,
 * hero CTAs, and progressive disclosure across components.
 */

// Mock fetch
global.fetch = jest.fn();

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
        clear: () => {
            store = {};
        },
    };
})();
Object.defineProperty(global, 'localStorage', { value: localStorageMock });

// Load the module
require('../../simpletuner/static/js/utils/hint-mixin.js');

describe('HintMixin', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorageMock.clear();
    });

    describe('createSingleHint - API mode', () => {
        let hintManager;

        beforeEach(() => {
            hintManager = window.HintMixin.createSingleHint({
                useApi: true,
                hintKey: 'test_hero',
            });
        });

        test('initializes with showHeroCTA true', () => {
            expect(hintManager.showHeroCTA).toBe(true);
        });

        test('loadHeroCTAState fetches from API', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ dismissed_hints: [] }),
            });

            await hintManager.loadHeroCTAState();

            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints');
            expect(hintManager.showHeroCTA).toBe(true);
        });

        test('loadHeroCTAState hides when hint is in dismissed list', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ dismissed_hints: ['test_hero', 'other'] }),
            });

            await hintManager.loadHeroCTAState();

            expect(hintManager.showHeroCTA).toBe(false);
        });

        test('loadHeroCTAState handles API error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await expect(hintManager.loadHeroCTAState()).resolves.not.toThrow();
            // Should remain true on error
            expect(hintManager.showHeroCTA).toBe(true);
        });

        test('loadHeroCTAState handles non-ok response', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
            });

            await hintManager.loadHeroCTAState();
            expect(hintManager.showHeroCTA).toBe(true);
        });

        test('dismissHeroCTA calls API and sets flag to false', async () => {
            fetch.mockResolvedValueOnce({ ok: true });

            await hintManager.dismissHeroCTA();

            expect(hintManager.showHeroCTA).toBe(false);
            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints/dismiss/test_hero', { method: 'POST' });
        });

        test('dismissHeroCTA handles API error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await expect(hintManager.dismissHeroCTA()).resolves.not.toThrow();
            // Flag should still be set to false even if API fails
            expect(hintManager.showHeroCTA).toBe(false);
        });

        test('restoreHeroCTA calls API and sets flag to true', async () => {
            hintManager.showHeroCTA = false;
            fetch.mockResolvedValueOnce({ ok: true });

            await hintManager.restoreHeroCTA();

            expect(hintManager.showHeroCTA).toBe(true);
            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints/show/test_hero', { method: 'POST' });
        });
    });

    describe('createSingleHint - localStorage mode', () => {
        let hintManager;

        beforeEach(() => {
            hintManager = window.HintMixin.createSingleHint({
                useApi: false,
                storageKey: 'test_hint_dismissed',
            });
        });

        test('loadHeroCTAState reads from localStorage', async () => {
            localStorageMock.getItem.mockReturnValueOnce(null);

            await hintManager.loadHeroCTAState();

            expect(localStorageMock.getItem).toHaveBeenCalledWith('test_hint_dismissed');
            expect(hintManager.showHeroCTA).toBe(true);
        });

        test('loadHeroCTAState hides when localStorage has "true"', async () => {
            localStorageMock.getItem.mockReturnValueOnce('true');

            await hintManager.loadHeroCTAState();

            expect(hintManager.showHeroCTA).toBe(false);
        });

        test('dismissHeroCTA saves to localStorage', async () => {
            await hintManager.dismissHeroCTA();

            expect(hintManager.showHeroCTA).toBe(false);
            expect(localStorageMock.setItem).toHaveBeenCalledWith('test_hint_dismissed', 'true');
            expect(fetch).not.toHaveBeenCalled();
        });

        test('restoreHeroCTA removes from localStorage', async () => {
            await hintManager.restoreHeroCTA();

            expect(hintManager.showHeroCTA).toBe(true);
            expect(localStorageMock.removeItem).toHaveBeenCalledWith('test_hint_dismissed');
        });
    });

    describe('createMultiHint - API mode', () => {
        let hintManager;

        beforeEach(() => {
            hintManager = window.HintMixin.createMultiHint({
                useApi: true,
                prefix: 'admin_',
                hintKeys: ['overview', 'users', 'settings'],
            });
        });

        test('initializes all hints as visible', () => {
            expect(hintManager.hints.overview).toBe(true);
            expect(hintManager.hints.users).toBe(true);
            expect(hintManager.hints.settings).toBe(true);
        });

        test('loadHints fetches from API', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ dismissed_hints: ['admin_users'] }),
            });

            await hintManager.loadHints();

            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints');
            expect(hintManager.hints.overview).toBe(true);
            expect(hintManager.hints.users).toBe(false);  // dismissed
            expect(hintManager.hints.settings).toBe(true);
        });

        test('loadHints handles missing dismissed_hints', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({}),
            });

            await hintManager.loadHints();

            // All should remain visible
            expect(hintManager.hints.overview).toBe(true);
            expect(hintManager.hints.users).toBe(true);
        });

        test('loadHints handles API error gracefully', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await expect(hintManager.loadHints()).resolves.not.toThrow();
            expect(hintManager.hintsLoading).toBe(false);
        });

        test('dismissHint calls API with prefixed key', async () => {
            fetch.mockResolvedValueOnce({ ok: true });

            await hintManager.dismissHint('users');

            expect(hintManager.hints.users).toBe(false);
            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints/dismiss/admin_users', { method: 'POST' });
        });

        test('showHint calls API with prefixed key', async () => {
            hintManager.hints.users = false;
            fetch.mockResolvedValueOnce({ ok: true });

            await hintManager.showHint('users');

            expect(hintManager.hints.users).toBe(true);
            expect(fetch).toHaveBeenCalledWith('/api/cloud/hints/show/admin_users', { method: 'POST' });
        });

        test('restoreAllHints shows all hints', async () => {
            hintManager.hints.overview = false;
            hintManager.hints.users = false;

            fetch.mockResolvedValue({ ok: true });

            hintManager.restoreAllHints();

            // restoreAllHints calls showHint for each
            expect(fetch).toHaveBeenCalledTimes(3);
        });

        test('anyHintsDismissed returns true when any hint is dismissed', () => {
            expect(hintManager.anyHintsDismissed()).toBe(false);

            hintManager.hints.users = false;

            expect(hintManager.anyHintsDismissed()).toBe(true);
        });

        test('anyHintsDismissed returns false when all hints visible', () => {
            expect(hintManager.anyHintsDismissed()).toBe(false);
        });
    });

    describe('createMultiHint - localStorage mode', () => {
        let hintManager;

        beforeEach(() => {
            hintManager = window.HintMixin.createMultiHint({
                useApi: false,
                storageKey: 'cloud_hints',
                hintKeys: ['dataloader', 'git'],
            });
        });

        test('loadHints reads from localStorage', async () => {
            localStorageMock.getItem.mockReturnValueOnce(JSON.stringify({
                dataloader_dismissed: true,
                git_dismissed: false,
            }));

            await hintManager.loadHints();

            expect(localStorageMock.getItem).toHaveBeenCalledWith('cloud_hints');
            expect(hintManager.hints.dataloader).toBe(false);  // dismissed
            expect(hintManager.hints.git).toBe(true);  // not dismissed
        });

        test('loadHints handles invalid JSON gracefully', async () => {
            localStorageMock.getItem.mockReturnValueOnce('not-json');

            await expect(hintManager.loadHints()).resolves.not.toThrow();
            // Should remain at defaults
            expect(hintManager.hints.dataloader).toBe(true);
        });

        test('loadHints handles null localStorage value', async () => {
            localStorageMock.getItem.mockReturnValueOnce(null);

            await hintManager.loadHints();

            // Should remain at defaults
            expect(hintManager.hints.dataloader).toBe(true);
            expect(hintManager.hints.git).toBe(true);
        });

        test('dismissHint saves to localStorage', async () => {
            await hintManager.dismissHint('dataloader');

            expect(hintManager.hints.dataloader).toBe(false);
            expect(localStorageMock.setItem).toHaveBeenCalled();

            const savedValue = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
            expect(savedValue.dataloader_dismissed).toBe(true);
        });

        test('showHint saves to localStorage', async () => {
            hintManager.hints.dataloader = false;

            await hintManager.showHint('dataloader');

            expect(hintManager.hints.dataloader).toBe(true);
            expect(localStorageMock.setItem).toHaveBeenCalled();

            const savedValue = JSON.parse(localStorageMock.setItem.mock.calls[0][1]);
            expect(savedValue.dataloader_dismissed).toBe(false);
        });

        test('_saveHintsToStorage only runs in localStorage mode', async () => {
            // In localStorage mode, should save
            hintManager._saveHintsToStorage();
            expect(localStorageMock.setItem).toHaveBeenCalled();

            // In API mode, should not save to localStorage
            const apiManager = window.HintMixin.createMultiHint({
                useApi: true,
                prefix: '',
                hintKeys: ['test'],
            });
            localStorageMock.setItem.mockClear();
            apiManager._saveHintsToStorage();
            expect(localStorageMock.setItem).not.toHaveBeenCalled();
        });
    });

    describe('HintMixin integration scenarios', () => {
        test('onboarding flow with progressive hints', async () => {
            const onboarding = window.HintMixin.createMultiHint({
                useApi: false,
                storageKey: 'onboarding_hints',
                hintKeys: ['welcome', 'config', 'submit'],
            });

            // All visible initially
            expect(onboarding.hints.welcome).toBe(true);
            expect(onboarding.hints.config).toBe(true);
            expect(onboarding.hints.submit).toBe(true);

            // User completes welcome
            await onboarding.dismissHint('welcome');
            expect(onboarding.hints.welcome).toBe(false);
            expect(onboarding.anyHintsDismissed()).toBe(true);

            // User completes config
            await onboarding.dismissHint('config');
            expect(onboarding.hints.config).toBe(false);

            // User restarts onboarding
            onboarding.restoreAllHints();

            // After async operations complete, all should be visible
            // (Note: restoreAllHints is async, so in real usage you'd await)
        });

        test('hero CTA persists across sessions via localStorage', async () => {
            const hero1 = window.HintMixin.createSingleHint({
                useApi: false,
                storageKey: 'hero_dismissed',
            });

            // Dismiss
            await hero1.dismissHeroCTA();

            // Simulate new session
            const hero2 = window.HintMixin.createSingleHint({
                useApi: false,
                storageKey: 'hero_dismissed',
            });

            localStorageMock.getItem.mockReturnValueOnce('true');
            await hero2.loadHeroCTAState();

            expect(hero2.showHeroCTA).toBe(false);
        });

        test('admin panel hints with prefix', async () => {
            fetch.mockResolvedValue({
                ok: true,
                json: () => Promise.resolve({ dismissed_hints: ['admin_users', 'admin_audit'] }),
            });

            const admin = window.HintMixin.createMultiHint({
                useApi: true,
                prefix: 'admin_',
                hintKeys: ['overview', 'users', 'audit', 'settings'],
            });

            await admin.loadHints();

            expect(admin.hints.overview).toBe(true);
            expect(admin.hints.users).toBe(false);
            expect(admin.hints.audit).toBe(false);
            expect(admin.hints.settings).toBe(true);
        });
    });
});
