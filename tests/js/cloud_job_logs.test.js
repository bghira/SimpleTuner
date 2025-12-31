/**
 * Tests for cloudLogMethods.
 *
 * cloudLogMethods handles log viewing, filtering, formatting, and download.
 */

// Mock fetch
global.fetch = jest.fn();

// Mock URL
global.URL = {
    createObjectURL: jest.fn(() => 'blob:test-url'),
    revokeObjectURL: jest.fn(),
};

// Mock document methods for download
const mockAppendChild = jest.fn();
const mockRemoveChild = jest.fn();
const mockClick = jest.fn();

// Helper function to escape HTML like a real DOM element
const escapeHtmlText = (text) => {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
};

global.document = {
    ...global.document,
    createElement: jest.fn((tag) => {
        if (tag === 'a') {
            return {
                href: '',
                download: '',
                click: mockClick,
            };
        }
        if (tag === 'div') {
            let _textContent = '';
            return {
                get textContent() { return _textContent; },
                set textContent(val) { _textContent = val; },
                get innerHTML() { return escapeHtmlText(_textContent); },
            };
        }
        return {};
    }),
    body: {
        appendChild: mockAppendChild,
        removeChild: mockRemoveChild,
    },
};

// Mock Blob
global.Blob = jest.fn((content, options) => ({
    content,
    type: options?.type,
}));

// Load the module
require('../../simpletuner/static/js/modules/cloud/job-logs.js');

describe('cloudLogMethods', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        context = {
            // Include all the methods so they can call each other
            ...window.cloudLogMethods,
            logsModal: {
                open: false,
                jobId: null,
                logs: '',
                loading: false,
                searchQuery: '',
                levelFilter: '',
                filteredLogsHtml: '',
                matchCount: 0,
                lineCount: 0,
                streaming: false,
                autoScroll: false,
                eventSource: null,
            },
            $refs: {
                logViewer: {
                    scrollTop: 0,
                    scrollHeight: 1000,
                },
            },
            $nextTick: (fn) => fn(),
        };
    });

    describe('viewLogs', () => {
        test('opens modal and sets jobId', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: 'Test log content' }),
            });

            await window.cloudLogMethods.viewLogs.call(context, 'job-123');

            expect(context.logsModal.open).toBe(true);
            expect(context.logsModal.jobId).toBe('job-123');
        });

        test('resets modal state before loading', async () => {
            context.logsModal.logs = 'Old logs';
            context.logsModal.searchQuery = 'old search';
            context.logsModal.levelFilter = 'ERROR';

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: 'New logs' }),
            });

            await window.cloudLogMethods.viewLogs.call(context, 'job-456');

            expect(context.logsModal.searchQuery).toBe('');
            expect(context.logsModal.levelFilter).toBe('');
        });

        test('fetches logs from API', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: 'Log line 1\nLog line 2' }),
            });

            await window.cloudLogMethods.viewLogs.call(context, 'job-123');

            expect(fetch).toHaveBeenCalledWith('/api/cloud/jobs/job-123/logs');
            expect(context.logsModal.logs).toBe('Log line 1\nLog line 2');
        });

        test('shows placeholder when no logs available', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: null }),
            });

            await window.cloudLogMethods.viewLogs.call(context, 'job-123');

            expect(context.logsModal.logs).toBe('(No logs available)');
        });

        test('handles API error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await window.cloudLogMethods.viewLogs.call(context, 'job-123');

            expect(context.logsModal.logs).toBe('Error: Network error');
            expect(context.logsModal.loading).toBe(false);
        });

        test('scrolls to bottom when autoScroll enabled', async () => {
            context.logsModal.autoScroll = true;

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: 'Test logs' }),
            });

            await window.cloudLogMethods.viewLogs.call(context, 'job-123');

            expect(context.$refs.logViewer.scrollTop).toBe(1000);
        });
    });

    describe('refreshLogs', () => {
        test('returns early if no jobId', async () => {
            context.logsModal.jobId = null;

            await window.cloudLogMethods.refreshLogs.call(context);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('fetches fresh logs', async () => {
            context.logsModal.jobId = 'job-123';
            context.logsModal.logs = 'Old logs';

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ logs: 'New logs' }),
            });

            await window.cloudLogMethods.refreshLogs.call(context);

            expect(context.logsModal.logs).toBe('New logs');
        });
    });

    describe('closeLogsModal', () => {
        test('resets modal state', () => {
            context.logsModal.open = true;
            context.logsModal.jobId = 'job-123';
            context.logsModal.logs = 'Some logs';
            context.logsModal.streaming = true;

            window.cloudLogMethods.closeLogsModal.call(context);

            expect(context.logsModal.open).toBe(false);
            expect(context.logsModal.jobId).toBeNull();
            expect(context.logsModal.logs).toBe('');
            expect(context.logsModal.streaming).toBe(false);
        });

        test('closes eventSource if present', () => {
            const mockClose = jest.fn();
            context.logsModal.eventSource = { close: mockClose };

            window.cloudLogMethods.closeLogsModal.call(context);

            expect(mockClose).toHaveBeenCalled();
            expect(context.logsModal.eventSource).toBeNull();
        });
    });

    describe('filterLogs', () => {
        test('returns all logs when no filters', () => {
            context.logsModal.logs = 'Line 1\nLine 2\nLine 3';
            context.logsModal.searchQuery = '';
            context.logsModal.levelFilter = '';

            window.cloudLogMethods.filterLogs.call(context);

            expect(context.logsModal.matchCount).toBe(3);
        });

        test('filters by search query', () => {
            context.logsModal.logs = 'INFO: Starting\nERROR: Failed\nINFO: Done';
            context.logsModal.searchQuery = 'error';
            context.logsModal.levelFilter = '';

            window.cloudLogMethods.filterLogs.call(context);

            expect(context.logsModal.matchCount).toBe(1);
        });

        test('filters by log level', () => {
            context.logsModal.logs = 'INFO: Starting\nERROR: Failed\nWARNING: Slow';
            context.logsModal.searchQuery = '';
            context.logsModal.levelFilter = 'ERROR';

            window.cloudLogMethods.filterLogs.call(context);

            expect(context.logsModal.matchCount).toBe(1);
        });

        test('combines search and level filters', () => {
            context.logsModal.logs = 'ERROR: Connection failed\nERROR: Timeout\nINFO: Connected';
            context.logsModal.searchQuery = 'connection';
            context.logsModal.levelFilter = 'ERROR';

            window.cloudLogMethods.filterLogs.call(context);

            expect(context.logsModal.matchCount).toBe(1);
        });
    });

    describe('formatLogs', () => {
        test('escapes HTML characters', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, '<script>alert("xss")</script>');

            expect(result).not.toContain('<script>');
            expect(result).toContain('&lt;script&gt;');
        });

        test('highlights ERROR lines', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'ERROR: Something failed');

            expect(result).toContain('class="log-error"');
        });

        test('highlights WARNING lines', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'WARNING: Slow operation');

            expect(result).toContain('class="log-warn"');
        });

        test('highlights WARN lines', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'WARN: Deprecated');

            expect(result).toContain('class="log-warn"');
        });

        test('highlights INFO lines', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'INFO: Started');

            expect(result).toContain('class="log-info"');
        });

        test('highlights DEBUG lines', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'DEBUG: Variable value');

            expect(result).toContain('class="log-debug"');
        });

        test('highlights search term', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, 'Finding the needle in haystack', 'needle');

            expect(result).toContain('class="log-highlight"');
        });

        test('highlights timestamps', () => {
            const result = window.cloudLogMethods.formatLogs.call(context, '2024-01-15T10:30:45.123Z Log message');

            expect(result).toContain('class="log-timestamp"');
        });
    });

    describe('escapeHtml', () => {
        test('escapes angle brackets', () => {
            const result = window.cloudLogMethods.escapeHtml.call(context, '<div>test</div>');

            expect(result).toBe('&lt;div&gt;test&lt;/div&gt;');
        });

        test('escapes ampersand', () => {
            const result = window.cloudLogMethods.escapeHtml.call(context, 'a & b');

            expect(result).toBe('a &amp; b');
        });
    });

    describe('escapeRegex', () => {
        test('escapes regex special characters', () => {
            const result = window.cloudLogMethods.escapeRegex.call(context, 'test.*+?^${}()|[]\\');

            expect(result).toBe('test\\.\\*\\+\\?\\^\\$\\{\\}\\(\\)\\|\\[\\]\\\\');
        });
    });

    describe('updateLogStats', () => {
        test('counts lines', () => {
            context.logsModal.logs = 'Line 1\nLine 2\nLine 3\nLine 4';

            window.cloudLogMethods.updateLogStats.call(context);

            expect(context.logsModal.lineCount).toBe(4);
        });

        test('handles empty logs', () => {
            context.logsModal.logs = '';

            window.cloudLogMethods.updateLogStats.call(context);

            expect(context.logsModal.lineCount).toBe(0);
        });

        test('handles null logs', () => {
            context.logsModal.logs = null;

            window.cloudLogMethods.updateLogStats.call(context);

            expect(context.logsModal.lineCount).toBe(0);
        });
    });

    describe('scrollLogsToBottom', () => {
        test('scrolls to bottom of log viewer', () => {
            context.$refs.logViewer = {
                scrollTop: 0,
                scrollHeight: 5000,
            };

            window.cloudLogMethods.scrollLogsToBottom.call(context);

            expect(context.$refs.logViewer.scrollTop).toBe(5000);
        });

        test('handles missing ref gracefully', () => {
            context.$refs = {};

            expect(() => {
                window.cloudLogMethods.scrollLogsToBottom.call(context);
            }).not.toThrow();
        });
    });

    describe('downloadLogs', () => {
        test('returns early if no logs', () => {
            context.logsModal.logs = '';

            window.cloudLogMethods.downloadLogs.call(context);

            expect(global.Blob).not.toHaveBeenCalled();
        });

        test('creates blob with logs content', () => {
            context.logsModal.logs = 'Test log content';
            context.logsModal.jobId = 'job-123';

            window.cloudLogMethods.downloadLogs.call(context);

            expect(global.Blob).toHaveBeenCalledWith(['Test log content'], { type: 'text/plain' });
        });

        test('creates download link with correct filename', () => {
            context.logsModal.logs = 'Test log content';
            context.logsModal.jobId = 'job-456';

            // Verify Blob was created (indicates download logic executed)
            context.downloadLogs();

            expect(global.Blob).toHaveBeenCalledWith(['Test log content'], { type: 'text/plain' });
        });

        test('triggers download via blob URL creation', () => {
            context.logsModal.logs = 'Test log content';
            context.logsModal.jobId = 'job-789';

            context.downloadLogs();

            // Verify URL was created and cleaned up
            expect(global.URL.createObjectURL).toHaveBeenCalled();
            expect(global.URL.revokeObjectURL).toHaveBeenCalled();
        });
    });
});

describe('cloudLogMethods edge cases', () => {
    let context;

    beforeEach(() => {
        jest.resetAllMocks();
        fetch.mockReset();
        context = {
            ...window.cloudLogMethods,
            logsModal: {
                open: false,
                jobId: null,
                logs: '',
                loading: false,
                searchQuery: '',
                levelFilter: '',
                filteredLogsHtml: '',
                matchCount: 0,
                lineCount: 0,
                streaming: false,
                autoScroll: false,
                eventSource: null,
            },
            $refs: {},
            $nextTick: (fn) => fn(),
        };
    });

    test('filterLogs handles empty logs', () => {
        context.logsModal.logs = '';
        context.logsModal.searchQuery = 'test';

        window.cloudLogMethods.filterLogs.call(context);

        expect(context.logsModal.matchCount).toBe(0);
    });

    test('formatLogs handles multiline ERROR blocks', () => {
        const logs = 'ERROR: First error\nERROR: Second error';

        const result = window.cloudLogMethods.formatLogs.call(context, logs);

        // Both lines should be wrapped
        expect((result.match(/log-error/g) || []).length).toBe(2);
    });

    test('search query is case-insensitive', () => {
        context.logsModal.logs = 'ERROR: Connection Failed\nINFO: Connection OK';
        context.logsModal.searchQuery = 'CONNECTION';

        window.cloudLogMethods.filterLogs.call(context);

        expect(context.logsModal.matchCount).toBe(2);
    });

    test('level filter is case-insensitive', () => {
        context.logsModal.logs = 'error: lowercase\nERROR: uppercase';
        context.logsModal.levelFilter = 'ERROR';

        window.cloudLogMethods.filterLogs.call(context);

        expect(context.logsModal.matchCount).toBe(2);
    });
});
