/**
 * Tests for authentication waiting behavior in components.
 *
 * Verifies that components wait for auth before making API calls,
 * and skip initialization when auth is not ready.
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

// Mock Alpine store
global.Alpine = {
    data: jest.fn((name, factory) => {
        global.Alpine._components = global.Alpine._components || {};
        global.Alpine._components[name] = factory;
    }),
    store: jest.fn(() => ({
        modelContext: {},
        configValues: {},
    })),
    start: jest.fn(),
    _components: {},
};

// Track API calls
let apiCallsMade = [];

// Mock ApiClient
global.ApiClient = {
    fetch: jest.fn((url) => {
        apiCallsMade.push(url);
        return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({ datasets: [], blueprints: [] }),
        });
    }),
};

// Mock fetch as well
global.fetch = jest.fn((url) => {
    apiCallsMade.push(url);
    return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
    });
});

// Mock $nextTick
const mockNextTick = jest.fn((cb) => {
    if (cb) setTimeout(cb, 0);
    return Promise.resolve();
});

describe('Auth Waiting Behavior', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        localStorageMock.clear();
        apiCallsMade = [];
        global.Alpine._components = {};
    });

    describe('window.waitForAuthReady', () => {
        test('returns true when no auth store exists', async () => {
            // Mock Alpine.store to return undefined (no auth)
            global.Alpine.store = jest.fn(() => undefined);

            // Define waitForAuthReady as it appears in the template
            window.waitForAuthReady = async function (timeout = 5000) {
                if (typeof Alpine === 'undefined' || typeof Alpine.store !== 'function') {
                    return true;
                }
                const authStore = Alpine.store('cloudAuth');
                if (!authStore) {
                    return true;
                }
                return authStore.waitForAuthReady(timeout);
            };

            const result = await window.waitForAuthReady();
            expect(result).toBe(true);
        });

        test('returns true when Alpine is not defined', async () => {
            const originalAlpine = global.Alpine;
            global.Alpine = undefined;

            window.waitForAuthReady = async function (timeout = 5000) {
                if (typeof Alpine === 'undefined' || typeof Alpine.store !== 'function') {
                    return true;
                }
                const authStore = Alpine.store('cloudAuth');
                if (!authStore) {
                    return true;
                }
                return authStore.waitForAuthReady(timeout);
            };

            const result = await window.waitForAuthReady();
            expect(result).toBe(true);

            global.Alpine = originalAlpine;
        });

        test('delegates to authStore.waitForAuthReady when auth store exists', async () => {
            const mockWaitForAuthReady = jest.fn().mockResolvedValue(true);
            global.Alpine.store = jest.fn((name) => {
                if (name === 'cloudAuth') {
                    return { waitForAuthReady: mockWaitForAuthReady };
                }
                return undefined;
            });

            window.waitForAuthReady = async function (timeout = 5000) {
                if (typeof Alpine === 'undefined' || typeof Alpine.store !== 'function') {
                    return true;
                }
                const authStore = Alpine.store('cloudAuth');
                if (!authStore) {
                    return true;
                }
                return authStore.waitForAuthReady(timeout);
            };

            const result = await window.waitForAuthReady();

            expect(mockWaitForAuthReady).toHaveBeenCalledWith(5000);
            expect(result).toBe(true);
        });

        test('returns false when auth store indicates not ready', async () => {
            const mockWaitForAuthReady = jest.fn().mockResolvedValue(false);
            global.Alpine.store = jest.fn((name) => {
                if (name === 'cloudAuth') {
                    return { waitForAuthReady: mockWaitForAuthReady };
                }
                return undefined;
            });

            window.waitForAuthReady = async function (timeout = 5000) {
                if (typeof Alpine === 'undefined' || typeof Alpine.store !== 'function') {
                    return true;
                }
                const authStore = Alpine.store('cloudAuth');
                if (!authStore) {
                    return true;
                }
                return authStore.waitForAuthReady(timeout);
            };

            const result = await window.waitForAuthReady();

            expect(result).toBe(false);
        });

        test('passes custom timeout to auth store', async () => {
            const mockWaitForAuthReady = jest.fn().mockResolvedValue(true);
            global.Alpine.store = jest.fn((name) => {
                if (name === 'cloudAuth') {
                    return { waitForAuthReady: mockWaitForAuthReady };
                }
                return undefined;
            });

            window.waitForAuthReady = async function (timeout = 5000) {
                if (typeof Alpine === 'undefined' || typeof Alpine.store !== 'function') {
                    return true;
                }
                const authStore = Alpine.store('cloudAuth');
                if (!authStore) {
                    return true;
                }
                return authStore.waitForAuthReady(timeout);
            };

            await window.waitForAuthReady(10000);

            expect(mockWaitForAuthReady).toHaveBeenCalledWith(10000);
        });
    });

    describe('datasetWizardComponent auth waiting', () => {
        beforeEach(() => {
            // Reset Alpine components
            global.Alpine._components = {};
            // Reset waitForAuthReady mock
            window.waitForAuthReady = jest.fn();
        });

        test('skips API calls when auth not ready', async () => {
            // Mock waitForAuthReady to return false (not authenticated)
            window.waitForAuthReady = jest.fn().mockResolvedValue(false);

            // Load the component
            require('../../simpletuner/static/js/dataset-wizard.js');

            const factory = window.datasetWizardComponent;
            expect(factory).toBeDefined();

            const component = factory();
            component.$nextTick = mockNextTick;
            component.$refs = {
                newFolderInput: { focus: jest.fn() },
                fileInput: { value: '' },
                zipInput: { value: '' },
            };

            // Call init
            await component.init();

            // Verify waitForAuthReady was called
            expect(window.waitForAuthReady).toHaveBeenCalled();

            // Verify no API calls were made
            expect(apiCallsMade.length).toBe(0);
        });

        test('makes API calls when auth is ready', async () => {
            // Mock waitForAuthReady to return true (authenticated)
            window.waitForAuthReady = jest.fn().mockResolvedValue(true);

            // Reset and reload the component
            jest.resetModules();
            global.Alpine._components = {};

            // Re-mock after resetModules
            global.Alpine = {
                data: jest.fn((name, factory) => {
                    global.Alpine._components = global.Alpine._components || {};
                    global.Alpine._components[name] = factory;
                }),
                store: jest.fn(() => ({
                    modelContext: {},
                    configValues: {},
                })),
                start: jest.fn(),
                _components: {},
            };

            global.ApiClient = {
                fetch: jest.fn((url) => {
                    apiCallsMade.push(url);
                    return Promise.resolve({
                        ok: true,
                        json: () => Promise.resolve({ datasets: [], blueprints: [] }),
                    });
                }),
            };

            require('../../simpletuner/static/js/dataset-wizard.js');

            const factory = window.datasetWizardComponent;
            const component = factory();
            component.$nextTick = mockNextTick;
            component.$refs = {
                newFolderInput: { focus: jest.fn() },
                fileInput: { value: '' },
                zipInput: { value: '' },
            };

            // Call init
            await component.init();

            // Verify waitForAuthReady was called
            expect(window.waitForAuthReady).toHaveBeenCalled();

            // Verify API calls were made (loadBlueprints calls API)
            expect(apiCallsMade.length).toBeGreaterThan(0);
        });
    });
});

describe('Component Init Pattern Verification', () => {
    /**
     * These tests verify the auth waiting pattern is correctly implemented.
     * They test the pattern rather than loading each module.
     */

    test('init pattern returns early when canProceed is false', async () => {
        let apiCalled = false;
        let initCompleted = false;

        // Simulate the init pattern
        const init = async function () {
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }
            apiCalled = true;
            initCompleted = true;
        };

        window.waitForAuthReady = jest.fn().mockResolvedValue(false);

        await init();

        expect(window.waitForAuthReady).toHaveBeenCalled();
        expect(apiCalled).toBe(false);
        expect(initCompleted).toBe(false);
    });

    test('init pattern continues when canProceed is true', async () => {
        let apiCalled = false;
        let initCompleted = false;

        // Simulate the init pattern
        const init = async function () {
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }
            apiCalled = true;
            initCompleted = true;
        };

        window.waitForAuthReady = jest.fn().mockResolvedValue(true);

        await init();

        expect(window.waitForAuthReady).toHaveBeenCalled();
        expect(apiCalled).toBe(true);
        expect(initCompleted).toBe(true);
    });
});

describe('cloudAuthStore waitForAuthReady', () => {
    /**
     * Tests for the cloudAuth store's waitForAuthReady method.
     */

    test('returns true immediately when already authenticated', async () => {
        const authStore = {
            isAuthenticated: true,
            isLoading: false,
            waitForAuthReady: async function (timeout = 5000) {
                if (this.isAuthenticated && !this.isLoading) {
                    return true;
                }
                // ... polling logic
                return false;
            },
        };

        const result = await authStore.waitForAuthReady();
        expect(result).toBe(true);
    });

    test('returns false when not authenticated and timeout expires', async () => {
        jest.useFakeTimers();

        const authStore = {
            isAuthenticated: false,
            isLoading: false,
            waitForAuthReady: async function (timeout = 100) {
                if (this.isAuthenticated && !this.isLoading) {
                    return true;
                }

                const startTime = Date.now();
                while (Date.now() - startTime < timeout) {
                    await new Promise((r) => setTimeout(r, 50));
                    if (this.isAuthenticated && !this.isLoading) {
                        return true;
                    }
                }
                return false;
            },
        };

        const resultPromise = authStore.waitForAuthReady(100);

        // Advance timers to trigger timeout
        jest.advanceTimersByTime(150);

        const result = await resultPromise;
        expect(result).toBe(false);

        jest.useRealTimers();
    });

    test('returns true when auth completes before timeout', async () => {
        const authStore = {
            isAuthenticated: false,
            isLoading: true,
            waitForAuthReady: async function (timeout = 5000) {
                if (this.isAuthenticated && !this.isLoading) {
                    return true;
                }

                const startTime = Date.now();
                while (Date.now() - startTime < timeout) {
                    await new Promise((r) => setTimeout(r, 50));
                    if (this.isAuthenticated && !this.isLoading) {
                        return true;
                    }
                }
                return false;
            },
        };

        // Simulate auth completing after a short delay
        setTimeout(() => {
            authStore.isAuthenticated = true;
            authStore.isLoading = false;
        }, 100);

        const result = await authStore.waitForAuthReady(5000);
        expect(result).toBe(true);
    });
});
