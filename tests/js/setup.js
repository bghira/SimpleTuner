/**
 * Jest setup for Alpine.js component tests.
 *
 * Sets up the testing environment with mocks for browser APIs
 * and Alpine.js initialization.
 */

// Mock fetch API
global.fetch = jest.fn(() =>
    Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
        text: () => Promise.resolve(''),
    })
);

// Mock EventSource for SSE
global.EventSource = class MockEventSource {
    constructor(url) {
        this.url = url;
        this.readyState = 0;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
    }

    close() {
        this.readyState = 2;
    }
};

// Mock console to suppress noise in tests
global.console = {
    ...console,
    log: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
};

// Mock crypto.randomUUID
if (!global.crypto) {
    global.crypto = {};
}
global.crypto.randomUUID = () => 'test-uuid-' + Math.random().toString(36).substr(2, 9);

// Mock Alpine.js
global.Alpine = {
    data: jest.fn((name, factory) => {
        global.Alpine._components = global.Alpine._components || {};
        global.Alpine._components[name] = factory;
    }),
    start: jest.fn(),
    _components: {},
};

// Utility to get a component instance
global.getAlpineComponent = (name, initialData = {}) => {
    const factory = global.Alpine._components[name];
    if (!factory) {
        throw new Error(`Alpine component '${name}' not found`);
    }
    return factory(initialData);
};

// Reset mocks before each test
beforeEach(() => {
    fetch.mockClear();
    global.Alpine._components = {};
});
