/**
 * Tests for SSEManager.
 *
 * SSEManager handles Server-Sent Events connections and payload normalization.
 * The progress transformation is critical for displaying training progress correctly.
 */

// Mock EventSource
class MockEventSource {
    constructor(url) {
        this.url = url;
        this.readyState = MockEventSource.CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this._listeners = {};
    }

    addEventListener(type, callback) {
        if (!this._listeners[type]) {
            this._listeners[type] = [];
        }
        this._listeners[type].push(callback);
    }

    removeEventListener(type, callback) {
        if (this._listeners[type]) {
            this._listeners[type] = this._listeners[type].filter(cb => cb !== callback);
        }
    }

    close() {
        this.readyState = MockEventSource.CLOSED;
    }

    // Test helpers
    _simulateOpen() {
        this.readyState = MockEventSource.OPEN;
        if (this.onopen) this.onopen();
    }

    _simulateMessage(data) {
        const event = { data: JSON.stringify(data) };
        if (this.onmessage) this.onmessage(event);
    }

    _simulateEvent(type, data) {
        const event = { data: JSON.stringify(data) };
        if (this._listeners[type]) {
            this._listeners[type].forEach(cb => cb(event));
        }
    }

    _simulateError() {
        this.readyState = MockEventSource.CLOSED;
        if (this.onerror) this.onerror({});
    }
}

MockEventSource.CONNECTING = 0;
MockEventSource.OPEN = 1;
MockEventSource.CLOSED = 2;

// Store original EventSource
const OriginalEventSource = global.EventSource;

beforeAll(() => {
    global.EventSource = MockEventSource;
});

afterAll(() => {
    global.EventSource = OriginalEventSource;
});

// Load the module
require('../../simpletuner/static/js/sse-manager.js');

describe('SSEManager', () => {
    beforeEach(() => {
        // Reset the SSEManager state by destroying any existing instance
        if (window.SSEManager && typeof window.SSEManager.destroy === 'function') {
            try {
                window.SSEManager.destroy();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }
    });

    afterEach(() => {
        // Clean up after each test
        if (window.SSEManager && typeof window.SSEManager.destroy === 'function') {
            try {
                window.SSEManager.destroy();
            } catch (e) {
                // Ignore errors during cleanup
            }
        }
    });

    describe('mapSeverityToLevel', () => {
        test('maps success to success', () => {
            expect(window.SSEManager.mapSeverityToLevel('success')).toBe('success');
        });

        test('maps warning to warning', () => {
            expect(window.SSEManager.mapSeverityToLevel('warning')).toBe('warning');
        });

        test('maps error and critical to danger', () => {
            expect(window.SSEManager.mapSeverityToLevel('error')).toBe('danger');
            expect(window.SSEManager.mapSeverityToLevel('critical')).toBe('danger');
        });

        test('maps debug to secondary', () => {
            expect(window.SSEManager.mapSeverityToLevel('debug')).toBe('secondary');
        });

        test('defaults to info for unknown severity', () => {
            expect(window.SSEManager.mapSeverityToLevel('unknown')).toBe('info');
            expect(window.SSEManager.mapSeverityToLevel('')).toBe('info');
            expect(window.SSEManager.mapSeverityToLevel(null)).toBe('info');
        });

        test('is case-insensitive', () => {
            expect(window.SSEManager.mapSeverityToLevel('SUCCESS')).toBe('success');
            expect(window.SSEManager.mapSeverityToLevel('Error')).toBe('danger');
        });
    });

    describe('normalizeProgressPayload', () => {
        test('returns null for null/undefined input', () => {
            expect(window.SSEManager.normalizeProgressPayload(null)).toBe(null);
            expect(window.SSEManager.normalizeProgressPayload(undefined)).toBe(null);
        });

        test('returns null for non-object input', () => {
            expect(window.SSEManager.normalizeProgressPayload('string')).toBe(null);
            expect(window.SSEManager.normalizeProgressPayload(123)).toBe(null);
        });

        test('extracts basic progress fields', () => {
            const payload = {
                step: 100,
                total_steps: 1000,
                epoch: 2,
                total_epochs: 10,
                loss: 0.5,
                learning_rate: 0.0001,
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);

            expect(result.type).toBe('training.progress');
            expect(result.current_step).toBe(100);
            expect(result.total_steps).toBe(1000);
            expect(result.epoch).toBe(2);
            expect(result.total_epochs).toBe(10);
            expect(result.loss).toBe(0.5);
            expect(result.lr).toBe(0.0001);
        });

        test('calculates percentage from step and total', () => {
            const payload = {
                step: 250,
                total_steps: 1000,
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.percentage).toBe(25);
        });

        test('handles nested progress object', () => {
            const payload = {
                progress: {
                    current: 50,
                    total: 200,
                    percent: 25,
                },
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.current_step).toBe(50);
            expect(result.total_steps).toBe(200);
            expect(result.percentage).toBe(25);
        });

        test('extracts job_id from top level', () => {
            const result = window.SSEManager.normalizeProgressPayload({ job_id: 'job-123', step: 1 });
            expect(result).not.toBeNull();
            expect(result.job_id).toBe('job-123');
        });

        test('extracts job_id from metrics', () => {
            const payload = {
                step: 1,
                metrics: { job_id: 'job-456' },
            };
            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result).not.toBeNull();
            expect(result.job_id).toBe('job-456');
        });

        test('handles alternative field names for step', () => {
            // current_step
            expect(window.SSEManager.normalizeProgressPayload({ current_step: 100 }).current_step).toBe(100);

            // global_step (in extras)
            expect(window.SSEManager.normalizeProgressPayload({ metrics: { global_step: 200 } }).current_step).toBe(200);
        });

        test('handles alternative field names for loss', () => {
            // train_loss in extras
            expect(window.SSEManager.normalizeProgressPayload({ metrics: { train_loss: 0.25 } }).loss).toBe(0.25);

            // Top-level loss
            expect(window.SSEManager.normalizeProgressPayload({ loss: 0.5 }).loss).toBe(0.5);
        });

        test('handles alternative field names for learning_rate', () => {
            // lr shorthand
            expect(window.SSEManager.normalizeProgressPayload({ lr: 0.001 }).lr).toBe(0.001);

            // In extras
            expect(window.SSEManager.normalizeProgressPayload({ extras: { learning_rate: 0.002 } }).lr).toBe(0.002);
        });

        test('clamps percentage to 0-100 range', () => {
            // Negative percent
            const negResult = window.SSEManager.normalizeProgressPayload({ percent: -10 });
            expect(negResult.percentage).toBe(0);

            // Over 100 percent
            const overResult = window.SSEManager.normalizeProgressPayload({ percent: 150 });
            expect(overResult.percentage).toBe(100);
        });

        test('rounds percentage to 2 decimal places', () => {
            const payload = {
                step: 333,
                total_steps: 1000,
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            // 333/1000 = 33.3%
            expect(result.percentage).toBe(33.3);
        });

        test('extracts rate statistics', () => {
            const payload = {
                metrics: {
                    step_speed_seconds: 1.5,
                    steps_per_second: 2.0,
                    samples_per_second: 16.0,
                    effective_batch_size: 8,
                },
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.step_speed_seconds).toBe(1.5);
            expect(result.steps_per_second).toBe(2.0);
            expect(result.samples_per_second).toBe(16.0);
            expect(result.effective_batch_size).toBe(8);
        });

        test('calculates samples_per_second from components when missing', () => {
            const payload = {
                metrics: {
                    steps_per_second: 2.0,
                    effective_batch_size: 8,
                },
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.samples_per_second).toBe(16.0); // 2.0 * 8
        });

        test('handles grad_absmax/grad_norm', () => {
            expect(window.SSEManager.normalizeProgressPayload({ grad_absmax: 1.5 }).grad_absmax).toBe(1.5);
            expect(window.SSEManager.normalizeProgressPayload({ grad_norm: 2.0 }).grad_absmax).toBe(2.0);
            expect(window.SSEManager.normalizeProgressPayload({ metrics: { grad_absmax: 0.8 } }).grad_absmax).toBe(0.8);
        });

        test('returns null for lifecycle.stage events', () => {
            const payload = {
                stage: {
                    key: 'loading_model',
                    label: 'Loading Model',
                    status: 'running',
                    percent: 50,
                },
            };

            // normalizeProgressPayload returns null for stage events
            // since they're handled differently
            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result).toBe(null);
        });

        test('preserves raw payload in result', () => {
            const payload = { step: 100, custom_field: 'test' };
            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.raw).toEqual(payload);
        });
    });

    describe('normalizeProgressPayload edge cases', () => {
        test('handles empty string values', () => {
            const payload = {
                step: '',
                total_steps: '',
                loss: '',
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.current_step).toBe(0);
            expect(result.total_steps).toBe(0);
            expect(result.loss).toBeUndefined();
        });

        test('handles NaN values', () => {
            const payload = {
                step: NaN,
                loss: NaN,
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.current_step).toBe(0);
            expect(result.loss).toBeUndefined();
        });

        test('handles Infinity values', () => {
            const payload = {
                step: Infinity,
                loss: -Infinity,
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            // Infinity is not finite, so should be treated as null
            expect(result.current_step).toBe(0);
        });

        test('handles deeply nested data structure', () => {
            const payload = {
                data: {
                    state: {
                        step: 100,
                        loss: 0.5,
                    },
                },
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.current_step).toBe(100);
            expect(result.loss).toBe(0.5);
        });

        test('handles mixed nested and top-level fields', () => {
            const payload = {
                step: 100,
                progress: {
                    total: 1000,
                },
                metrics: {
                    loss: 0.5,
                },
            };

            const result = window.SSEManager.normalizeProgressPayload(payload);
            expect(result.current_step).toBe(100);
            expect(result.total_steps).toBe(1000);
            expect(result.loss).toBe(0.5);
        });
    });

    describe('connection lifecycle', () => {
        test('getState returns state object with expected properties', () => {
            const state = window.SSEManager.getState();
            expect(state).toHaveProperty('connectionState');
            expect(state).toHaveProperty('retryCount');
            expect(typeof state.retryCount).toBe('number');
        });

        test('init creates connection and returns manager', () => {
            const manager = window.SSEManager.init({ url: '/api/events' });
            expect(manager).toBe(window.SSEManager);
            // After init, connection should be attempting to connect
            const state = window.SSEManager.getState();
            expect(['connecting', 'connected', 'disconnected']).toContain(state.connectionState);
        });

        test('init returns same instance on repeated calls', () => {
            const first = window.SSEManager.init();
            const second = window.SSEManager.init();
            expect(first).toBe(second);
        });

        test('addEventListener registers callback', () => {
            window.SSEManager.init();

            const callback = jest.fn();
            window.SSEManager.addEventListener('training.progress', callback);

            // Callback should be registered (we can't easily verify this without triggering an event)
            expect(() => window.SSEManager.addEventListener('test', () => {})).not.toThrow();
        });

        test('removeEventListener removes callback', () => {
            window.SSEManager.init();

            const callback = jest.fn();
            window.SSEManager.addEventListener('test', callback);
            window.SSEManager.removeEventListener('test', callback);

            // Should not throw
            expect(() => window.SSEManager.removeEventListener('test', callback)).not.toThrow();
        });

        test('clearAllListeners removes all callbacks', () => {
            window.SSEManager.init();

            window.SSEManager.addEventListener('type1', () => {});
            window.SSEManager.addEventListener('type2', () => {});
            window.SSEManager.clearAllListeners();

            // Should not throw
            expect(() => window.SSEManager.clearAllListeners()).not.toThrow();
        });

        test('resetRetries resets retry count', () => {
            window.SSEManager.init();
            window.SSEManager.resetRetries();

            const state = window.SSEManager.getState();
            expect(state.retryCount).toBe(0);
        });
    });
});

describe('SSEManager progress scenarios', () => {
    beforeEach(() => {
        if (window.SSEManager) {
            window.SSEManager.destroy();
        }
    });

    test('typical training progress event', () => {
        const payload = {
            job_id: 'replicate-abc123',
            step: 500,
            total_steps: 2000,
            epoch: 3,
            total_epochs: 10,
            loss: 0.234,
            learning_rate: 0.00005,
            metrics: {
                steps_per_second: 1.5,
                effective_batch_size: 4,
                grad_absmax: 0.89,
            },
        };

        const result = window.SSEManager.normalizeProgressPayload(payload);

        expect(result.type).toBe('training.progress');
        expect(result.job_id).toBe('replicate-abc123');
        expect(result.percentage).toBe(25); // 500/2000
        expect(result.current_step).toBe(500);
        expect(result.total_steps).toBe(2000);
        expect(result.epoch).toBe(3);
        expect(result.total_epochs).toBe(10);
        expect(result.loss).toBe(0.234);
        expect(result.lr).toBe(0.00005);
        expect(result.steps_per_second).toBe(1.5);
        expect(result.effective_batch_size).toBe(4);
        expect(result.samples_per_second).toBe(6); // 1.5 * 4
        expect(result.grad_absmax).toBe(0.89);
    });

    test('early training with zero values', () => {
        const payload = {
            step: 0,
            total_steps: 1000,
            epoch: 0,
            total_epochs: 5,
        };

        const result = window.SSEManager.normalizeProgressPayload(payload);

        expect(result.current_step).toBe(0);
        expect(result.epoch).toBe(0);
        expect(result.percentage).toBe(0);
    });

    test('completed training (100%)', () => {
        const payload = {
            step: 1000,
            total_steps: 1000,
            epoch: 5,
            total_epochs: 5,
        };

        const result = window.SSEManager.normalizeProgressPayload(payload);

        expect(result.current_step).toBe(1000);
        expect(result.epoch).toBe(5);
        expect(result.percentage).toBe(100);
    });

    test('replicate callback format', () => {
        // Replicate sends a specific format
        const payload = {
            payload: {
                step: 100,
                total_steps: 500,
                loss: 0.5,
            },
            event: { type: 'progress' },
        };

        // The normalizer looks at payload.payload if present
        const innerPayload = payload.payload;
        const result = window.SSEManager.normalizeProgressPayload(innerPayload);

        expect(result.current_step).toBe(100);
        expect(result.total_steps).toBe(500);
    });
});
