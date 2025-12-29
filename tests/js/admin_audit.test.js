/**
 * Tests for admin audit module.
 *
 * Tests audit log methods including chain verification.
 */

// Load the audit module
require('../../simpletuner/static/js/modules/admin/audit.js');

describe('adminAuditMethods', () => {
    let context;

    beforeEach(() => {
        // Reset fetch mock
        fetch.mockClear();

        // Create a fresh context with required state
        context = {
            audit: {
                loading: false,
                loaded: false,
                entries: [],
                stats: null,
                eventTypes: [],
                limit: 50,
                offset: 0,
                hasMore: false,
                eventTypeFilter: '',
                actorSearch: '',
                sinceDate: '',
                untilDate: '',
                securityOnly: false,
                verifyResult: null,
                verifying: false,
            },
        };

        // Bind methods to context
        Object.keys(window.adminAuditMethods).forEach((key) => {
            if (typeof window.adminAuditMethods[key] === 'function') {
                context[key] = window.adminAuditMethods[key].bind(context);
            }
        });
    });

    describe('verifyAuditChain', () => {
        test('uses GET method for verification request', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ valid: true, entries_checked: 100 }),
            });

            await context.verifyAuditChain();

            expect(fetch).toHaveBeenCalledWith('/api/cloud/audit/verify');
            // Verify no method override (GET is default)
            expect(fetch).toHaveBeenCalledTimes(1);
            const callArgs = fetch.mock.calls[0];
            // Should only have URL, no options with method
            expect(callArgs.length).toBe(1);
        });

        test('sets verifying state during request', async () => {
            let verifyingDuringRequest = false;

            fetch.mockImplementationOnce(() => {
                verifyingDuringRequest = context.audit.verifying;
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ valid: true }),
                });
            });

            await context.verifyAuditChain();

            expect(verifyingDuringRequest).toBe(true);
            expect(context.audit.verifying).toBe(false);
        });

        test('stores successful verification result', async () => {
            const verifyResult = {
                valid: true,
                entries_checked: 150,
                first_entry_id: 1,
                last_entry_id: 150,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(verifyResult),
            });

            await context.verifyAuditChain();

            expect(context.audit.verifyResult).toEqual(verifyResult);
            expect(context.audit.verifyResult.valid).toBe(true);
        });

        test('handles verification failure from server', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Chain integrity compromised at entry 42' }),
            });

            await context.verifyAuditChain();

            expect(context.audit.verifyResult.valid).toBe(false);
            expect(context.audit.verifyResult.error).toBe('Chain integrity compromised at entry 42');
        });

        test('handles verification failure with unparseable error', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.reject(new Error('Invalid JSON')),
            });

            await context.verifyAuditChain();

            expect(context.audit.verifyResult.valid).toBe(false);
            expect(context.audit.verifyResult.error).toBe('Verification request failed');
        });

        test('handles network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));

            await context.verifyAuditChain();

            expect(context.audit.verifyResult.valid).toBe(false);
            expect(context.audit.verifyResult.error).toBe('Network error');
            expect(context.audit.verifying).toBe(false);
        });

        test('clears previous result before new verification', async () => {
            context.audit.verifyResult = { valid: true, old: true };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ valid: true, new: true }),
            });

            await context.verifyAuditChain();

            expect(context.audit.verifyResult.old).toBeUndefined();
            expect(context.audit.verifyResult.new).toBe(true);
        });
    });

    describe('loadAuditEntries', () => {
        test('builds query parameters correctly', async () => {
            context.audit.limit = 25;
            context.audit.offset = 50;
            context.audit.eventTypeFilter = 'auth.login';
            context.audit.actorSearch = '123';
            context.audit.securityOnly = true;

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ entries: [], has_more: false }),
            });

            await context.loadAuditEntries();

            const url = fetch.mock.calls[0][0];
            expect(url).toContain('limit=25');
            expect(url).toContain('offset=50');
            expect(url).toContain('event_type=auth.login');
            expect(url).toContain('actor_id=123');
            expect(url).toContain('security_only=true');
        });

        test('sets loaded flag on success', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ entries: [{ id: 1 }], has_more: true }),
            });

            await context.loadAuditEntries();

            expect(context.audit.loaded).toBe(true);
            expect(context.audit.entries).toHaveLength(1);
            expect(context.audit.hasMore).toBe(true);
        });
    });

    describe('pagination', () => {
        test('auditNextPage increments offset', () => {
            context.audit.hasMore = true;
            context.audit.offset = 0;
            context.audit.limit = 50;

            // Mock loadAuditEntries
            context.loadAuditEntries = jest.fn();

            context.auditNextPage();

            expect(context.audit.offset).toBe(50);
            expect(context.loadAuditEntries).toHaveBeenCalled();
        });

        test('auditNextPage does nothing without hasMore', () => {
            context.audit.hasMore = false;
            context.audit.offset = 0;

            context.loadAuditEntries = jest.fn();

            context.auditNextPage();

            expect(context.audit.offset).toBe(0);
            expect(context.loadAuditEntries).not.toHaveBeenCalled();
        });

        test('auditPrevPage decrements offset', () => {
            context.audit.offset = 100;
            context.audit.limit = 50;

            context.loadAuditEntries = jest.fn();

            context.auditPrevPage();

            expect(context.audit.offset).toBe(50);
            expect(context.loadAuditEntries).toHaveBeenCalled();
        });

        test('auditPrevPage stops at zero', () => {
            context.audit.offset = 25;
            context.audit.limit = 50;

            context.loadAuditEntries = jest.fn();

            context.auditPrevPage();

            expect(context.audit.offset).toBe(0);
        });
    });

    describe('formatAuditTimestamp', () => {
        test('formats timestamp correctly', () => {
            const result = context.formatAuditTimestamp('2024-12-28T10:30:45Z');
            // Should contain date and time parts
            expect(result).toMatch(/\d+.*\d+:\d+:\d+/);
        });

        test('returns empty string for null', () => {
            expect(context.formatAuditTimestamp(null)).toBe('');
            expect(context.formatAuditTimestamp(undefined)).toBe('');
        });
    });

    describe('clearAuditFilters', () => {
        test('resets all filters and reloads', () => {
            context.audit.eventTypeFilter = 'auth.login';
            context.audit.actorSearch = '123';
            context.audit.sinceDate = '2024-01-01';
            context.audit.untilDate = '2024-12-31';
            context.audit.securityOnly = true;
            context.audit.offset = 100;

            context.loadAuditEntries = jest.fn();

            context.clearAuditFilters();

            expect(context.audit.eventTypeFilter).toBe('');
            expect(context.audit.actorSearch).toBe('');
            expect(context.audit.sinceDate).toBe('');
            expect(context.audit.untilDate).toBe('');
            expect(context.audit.securityOnly).toBe(false);
            expect(context.audit.offset).toBe(0);
            expect(context.loadAuditEntries).toHaveBeenCalled();
        });
    });
});
