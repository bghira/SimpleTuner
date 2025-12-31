/**
 * Tests for UIHelpers utility.
 *
 * UIHelpers consolidates badge classes, icons, and formatters used across the UI.
 * These tests ensure consistent behavior for error handling, date formatting, and styling.
 */

// Load the module
require('../../simpletuner/static/js/utils/ui-helpers.js');

describe('UIHelpers', () => {
    describe('extractErrorMessage', () => {
        test('returns fallback for null data', () => {
            expect(window.UIHelpers.extractErrorMessage(null)).toBe('An error occurred');
            expect(window.UIHelpers.extractErrorMessage(null, 'Custom fallback')).toBe('Custom fallback');
        });

        test('returns fallback for undefined data', () => {
            expect(window.UIHelpers.extractErrorMessage(undefined)).toBe('An error occurred');
        });

        test('extracts string detail', () => {
            expect(window.UIHelpers.extractErrorMessage({ detail: 'Permission denied' })).toBe('Permission denied');
        });

        test('extracts array detail (Pydantic validation errors)', () => {
            const data = {
                detail: [
                    { msg: 'Field required', loc: ['body', 'name'] },
                    { msg: 'Invalid email', loc: ['body', 'email'] },
                ],
            };
            expect(window.UIHelpers.extractErrorMessage(data)).toBe('Field required, Invalid email');
        });

        test('handles array detail with message property', () => {
            const data = {
                detail: [{ message: 'Error 1' }, { message: 'Error 2' }],
            };
            expect(window.UIHelpers.extractErrorMessage(data)).toBe('Error 1, Error 2');
        });

        test('handles array detail without msg or message (converts to string)', () => {
            const data = {
                detail: [{ code: 'ERR1' }, { code: 'ERR2' }],
            };
            const result = window.UIHelpers.extractErrorMessage(data);
            expect(result).toContain('[object Object]');
        });

        test('extracts message property', () => {
            expect(window.UIHelpers.extractErrorMessage({ message: 'Something went wrong' }))
                .toBe('Something went wrong');
        });

        test('extracts error property', () => {
            expect(window.UIHelpers.extractErrorMessage({ error: 'Network error' }))
                .toBe('Network error');
        });

        test('prioritizes detail over message over error', () => {
            expect(window.UIHelpers.extractErrorMessage({
                detail: 'Detail message',
                message: 'Message property',
                error: 'Error property',
            })).toBe('Detail message');

            expect(window.UIHelpers.extractErrorMessage({
                message: 'Message property',
                error: 'Error property',
            })).toBe('Message property');
        });
    });

    describe('getEventTypeBadgeClass', () => {
        test('returns bg-secondary for null/undefined', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass(null)).toBe('bg-secondary');
            expect(window.UIHelpers.getEventTypeBadgeClass(undefined)).toBe('bg-secondary');
            expect(window.UIHelpers.getEventTypeBadgeClass('')).toBe('bg-secondary');
        });

        test('returns bg-danger for failure events', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('AUTH_LOGIN_FAILED')).toBe('bg-danger');
            expect(window.UIHelpers.getEventTypeBadgeClass('access_denied')).toBe('bg-danger');
            expect(window.UIHelpers.getEventTypeBadgeClass('validation_error')).toBe('bg-danger');
        });

        test('returns bg-warning for security events', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('security_alert')).toBe('bg-warning text-dark');
            expect(window.UIHelpers.getEventTypeBadgeClass('suspicious_activity')).toBe('bg-warning text-dark');
            expect(window.UIHelpers.getEventTypeBadgeClass('rate_limit_exceeded')).toBe('bg-warning text-dark');
        });

        test('returns bg-success for creation/login events', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('JOB_CREATED')).toBe('bg-success');
            expect(window.UIHelpers.getEventTypeBadgeClass('user_login')).toBe('bg-success');
            expect(window.UIHelpers.getEventTypeBadgeClass('operation_success')).toBe('bg-success');
        });

        test('returns bg-danger for delete events', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('JOB_DELETED')).toBe('bg-danger');
            expect(window.UIHelpers.getEventTypeBadgeClass('user_removed')).toBe('bg-danger');
        });

        test('returns bg-info for update events', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('config_updated')).toBe('bg-info');
            expect(window.UIHelpers.getEventTypeBadgeClass('settings_update')).toBe('bg-info');
            // Note: 'modified' doesn't contain 'modify' - test 'modify' directly
            expect(window.UIHelpers.getEventTypeBadgeClass('user_modify_role')).toBe('bg-info');
        });

        test('is case-insensitive', () => {
            expect(window.UIHelpers.getEventTypeBadgeClass('AUTH_LOGIN_FAILED'))
                .toBe(window.UIHelpers.getEventTypeBadgeClass('auth_login_failed'));
        });
    });

    describe('getRoleBadgeClass', () => {
        test('returns bg-danger for admin', () => {
            expect(window.UIHelpers.getRoleBadgeClass('admin')).toBe('bg-danger');
        });

        test('returns bg-warning for lead', () => {
            expect(window.UIHelpers.getRoleBadgeClass('lead')).toBe('bg-warning text-dark');
        });

        test('returns bg-secondary for member and unknown roles', () => {
            expect(window.UIHelpers.getRoleBadgeClass('member')).toBe('bg-secondary');
            expect(window.UIHelpers.getRoleBadgeClass('unknown')).toBe('bg-secondary');
            expect(window.UIHelpers.getRoleBadgeClass('')).toBe('bg-secondary');
        });
    });

    describe('getSeverityBadgeClass', () => {
        test('returns correct class for each severity level', () => {
            expect(window.UIHelpers.getSeverityBadgeClass('debug')).toBe('bg-secondary');
            expect(window.UIHelpers.getSeverityBadgeClass('info')).toBe('bg-info');
            expect(window.UIHelpers.getSeverityBadgeClass('warning')).toBe('bg-warning text-dark');
            expect(window.UIHelpers.getSeverityBadgeClass('error')).toBe('bg-danger');
            expect(window.UIHelpers.getSeverityBadgeClass('critical')).toBe('bg-dark');
        });

        test('returns bg-secondary for unknown severity', () => {
            expect(window.UIHelpers.getSeverityBadgeClass('unknown')).toBe('bg-secondary');
        });
    });

    describe('getQuotaActionBadgeClass', () => {
        test('returns correct class for each action', () => {
            expect(window.UIHelpers.getQuotaActionBadgeClass('block')).toBe('bg-danger');
            expect(window.UIHelpers.getQuotaActionBadgeClass('warn')).toBe('bg-warning text-dark');
            expect(window.UIHelpers.getQuotaActionBadgeClass('require_approval')).toBe('bg-info');
        });

        test('is case-insensitive', () => {
            expect(window.UIHelpers.getQuotaActionBadgeClass('BLOCK')).toBe('bg-danger');
            expect(window.UIHelpers.getQuotaActionBadgeClass('Warn')).toBe('bg-warning text-dark');
        });

        test('handles null/undefined gracefully', () => {
            expect(window.UIHelpers.getQuotaActionBadgeClass(null)).toBe('bg-secondary');
            expect(window.UIHelpers.getQuotaActionBadgeClass(undefined)).toBe('bg-secondary');
        });
    });

    describe('formatEventType', () => {
        test('converts snake_case to Title Case', () => {
            expect(window.UIHelpers.formatEventType('job_created')).toBe('Job Created');
            expect(window.UIHelpers.formatEventType('user_login_failed')).toBe('User Login Failed');
        });

        test('converts dot.notation to Title Case', () => {
            expect(window.UIHelpers.formatEventType('auth.login')).toBe('Auth Login');
        });

        test('handles mixed separators', () => {
            expect(window.UIHelpers.formatEventType('auth.login_success')).toBe('Auth Login Success');
        });

        test('returns empty string for null/undefined', () => {
            expect(window.UIHelpers.formatEventType(null)).toBe('');
            expect(window.UIHelpers.formatEventType(undefined)).toBe('');
            expect(window.UIHelpers.formatEventType('')).toBe('');
        });
    });

    describe('formatDuration', () => {
        test('returns -- for null/undefined/zero', () => {
            expect(window.UIHelpers.formatDuration(null)).toBe('--');
            expect(window.UIHelpers.formatDuration(undefined)).toBe('--');
        });

        test('returns < 1m for very short durations', () => {
            expect(window.UIHelpers.formatDuration(30)).toBe('< 1m');
            expect(window.UIHelpers.formatDuration(59)).toBe('< 1m');
        });

        test('formats minutes correctly', () => {
            expect(window.UIHelpers.formatDuration(60)).toBe('1m');
            expect(window.UIHelpers.formatDuration(120)).toBe('2m');
            expect(window.UIHelpers.formatDuration(1800)).toBe('30m');
        });

        test('formats hours and minutes correctly', () => {
            expect(window.UIHelpers.formatDuration(3600)).toBe('1h 0m');
            expect(window.UIHelpers.formatDuration(3660)).toBe('1h 1m');
            expect(window.UIHelpers.formatDuration(5400)).toBe('1h 30m');
            expect(window.UIHelpers.formatDuration(7200)).toBe('2h 0m');
        });

        test('handles 0 seconds correctly', () => {
            expect(window.UIHelpers.formatDuration(0)).toBe('< 1m');
        });
    });

    describe('formatWaitTime', () => {
        test('returns N/A for null/undefined', () => {
            expect(window.UIHelpers.formatWaitTime(null)).toBe('N/A');
            expect(window.UIHelpers.formatWaitTime(undefined)).toBe('N/A');
        });

        test('returns < 1 min for short times', () => {
            expect(window.UIHelpers.formatWaitTime(0)).toBe('< 1 min');
            expect(window.UIHelpers.formatWaitTime(30)).toBe('< 1 min');
            expect(window.UIHelpers.formatWaitTime(59)).toBe('< 1 min');
        });

        test('formats minutes with ~ prefix', () => {
            expect(window.UIHelpers.formatWaitTime(60)).toBe('~1 min');
            expect(window.UIHelpers.formatWaitTime(300)).toBe('~5 min');
        });

        test('formats hours and minutes with ~ prefix', () => {
            expect(window.UIHelpers.formatWaitTime(3600)).toBe('~1h 0m');
            expect(window.UIHelpers.formatWaitTime(5400)).toBe('~1h 30m');
        });
    });

    describe('formatRelativeTime', () => {
        beforeEach(() => {
            // Mock Date.now to a fixed time
            jest.useFakeTimers();
            jest.setSystemTime(new Date('2024-06-15T12:00:00Z'));
        });

        afterEach(() => {
            jest.useRealTimers();
        });

        test('returns fallback for null/undefined', () => {
            expect(window.UIHelpers.formatRelativeTime(null)).toBe('');
            expect(window.UIHelpers.formatRelativeTime(null, { fallback: 'N/A' })).toBe('N/A');
        });

        test('returns "Just now" for very recent times', () => {
            const recent = new Date('2024-06-15T11:59:30Z');
            expect(window.UIHelpers.formatRelativeTime(recent)).toBe('Just now');
        });

        test('returns compact "now" when compact option is true', () => {
            const recent = new Date('2024-06-15T11:59:30Z');
            expect(window.UIHelpers.formatRelativeTime(recent, { compact: true })).toBe('now');
        });

        test('formats minutes ago', () => {
            const fiveMinAgo = new Date('2024-06-15T11:55:00Z');
            expect(window.UIHelpers.formatRelativeTime(fiveMinAgo)).toBe('5m ago');

            const thirtyMinAgo = new Date('2024-06-15T11:30:00Z');
            expect(window.UIHelpers.formatRelativeTime(thirtyMinAgo)).toBe('30m ago');
        });

        test('formats hours ago', () => {
            const twoHoursAgo = new Date('2024-06-15T10:00:00Z');
            expect(window.UIHelpers.formatRelativeTime(twoHoursAgo)).toBe('2h ago');
        });

        test('formats days ago', () => {
            const twoDaysAgo = new Date('2024-06-13T12:00:00Z');
            expect(window.UIHelpers.formatRelativeTime(twoDaysAgo)).toBe('2d ago');
        });

        test('falls back to absolute date for older times', () => {
            const twoWeeksAgo = new Date('2024-06-01T12:00:00Z');
            const result = window.UIHelpers.formatRelativeTime(twoWeeksAgo);
            // Should be a locale date string, not relative
            expect(result).not.toContain('ago');
        });

        test('respects relativeDays option', () => {
            const threeDaysAgo = new Date('2024-06-12T12:00:00Z');

            // Default 7 days - should be relative
            expect(window.UIHelpers.formatRelativeTime(threeDaysAgo)).toBe('3d ago');

            // Custom 2 days - should be absolute
            const result = window.UIHelpers.formatRelativeTime(threeDaysAgo, { relativeDays: 2 });
            expect(result).not.toContain('ago');
        });

        test('handles invalid date gracefully', () => {
            expect(window.UIHelpers.formatRelativeTime('not-a-date')).toBe('');
            expect(window.UIHelpers.formatRelativeTime('not-a-date', { fallback: 'Invalid' })).toBe('Invalid');
        });

        test('handles ISO string input', () => {
            const isoString = '2024-06-15T11:55:00Z';
            expect(window.UIHelpers.formatRelativeTime(isoString)).toBe('5m ago');
        });
    });

    describe('formatDateTime', () => {
        test('returns fallback for null/undefined', () => {
            expect(window.UIHelpers.formatDateTime(null)).toBe('--');
            expect(window.UIHelpers.formatDateTime(null, { fallback: 'N/A' })).toBe('N/A');
        });

        test('formats valid date with time', () => {
            const date = new Date('2024-06-15T10:30:45Z');
            const result = window.UIHelpers.formatDateTime(date);
            // Should contain both date and time parts
            expect(result).toMatch(/\d+.*\d+:\d+/);
        });

        test('includes seconds when option is set', () => {
            const date = new Date('2024-06-15T10:30:45Z');
            const result = window.UIHelpers.formatDateTime(date, { seconds: true });
            // Should have three colon-separated time components
            expect(result).toMatch(/\d+:\d+:\d+/);
        });

        test('handles ISO string input', () => {
            const result = window.UIHelpers.formatDateTime('2024-06-15T10:30:45Z');
            expect(result).toMatch(/\d+.*\d+:\d+/);
        });

        test('handles invalid date gracefully', () => {
            expect(window.UIHelpers.formatDateTime('invalid')).toBe('--');
        });
    });

    describe('formatTimestamp', () => {
        test('returns - for null/undefined', () => {
            expect(window.UIHelpers.formatTimestamp(null)).toBe('-');
            expect(window.UIHelpers.formatTimestamp(undefined)).toBe('-');
        });

        test('formats timestamp with month, day, and time including seconds', () => {
            const date = new Date('2024-06-15T10:30:45Z');
            const result = window.UIHelpers.formatTimestamp(date);
            // Should include seconds
            expect(result).toMatch(/\d+:\d+:\d+/);
        });
    });

    describe('navigateToTab', () => {
        beforeEach(() => {
            document.body.innerHTML = `
                <div data-tab="settings" class="tab-button"></div>
                <div data-tab="jobs" class="tab-button"></div>
            `;
        });

        afterEach(() => {
            document.body.innerHTML = '';
        });

        test('clicks the matching tab element', () => {
            const settingsTab = document.querySelector('[data-tab="settings"]');
            const clickSpy = jest.spyOn(settingsTab, 'click');

            const result = window.UIHelpers.navigateToTab('settings');

            expect(result).toBe(true);
            expect(clickSpy).toHaveBeenCalled();
        });

        test('returns false when tab not found', () => {
            const result = window.UIHelpers.navigateToTab('nonexistent');
            expect(result).toBe(false);
        });
    });

    describe('navigateToTabAndClose', () => {
        beforeEach(() => {
            document.body.innerHTML = '<div data-tab="settings"></div>';
        });

        afterEach(() => {
            document.body.innerHTML = '';
        });

        test('calls closeCallback before navigating', () => {
            const closeCallback = jest.fn();

            window.UIHelpers.navigateToTabAndClose('settings', closeCallback);

            expect(closeCallback).toHaveBeenCalled();
        });

        test('handles missing closeCallback gracefully', () => {
            expect(() => {
                window.UIHelpers.navigateToTabAndClose('settings', null);
            }).not.toThrow();
        });

        test('handles non-function closeCallback gracefully', () => {
            expect(() => {
                window.UIHelpers.navigateToTabAndClose('settings', 'not a function');
            }).not.toThrow();
        });
    });
});

describe('UIHelpers icon helpers', () => {
    beforeEach(() => {
        require('../../simpletuner/static/js/utils/ui-helpers.js');
    });

    describe('getChannelIcon', () => {
        test('returns correct icons for known channels', () => {
            expect(window.UIHelpers.getChannelIcon('email')).toBe('fas fa-envelope');
            expect(window.UIHelpers.getChannelIcon('webhook')).toBe('fas fa-plug');
            expect(window.UIHelpers.getChannelIcon('slack')).toBe('fab fa-slack');
        });

        test('returns default bell icon for unknown channels', () => {
            expect(window.UIHelpers.getChannelIcon('unknown')).toBe('fas fa-bell');
        });
    });

    describe('getStatusIcon', () => {
        test('returns correct icons for known statuses', () => {
            expect(window.UIHelpers.getStatusIcon('delivered')).toBe('fa-check');
            expect(window.UIHelpers.getStatusIcon('failed')).toBe('fa-times');
            expect(window.UIHelpers.getStatusIcon('pending')).toBe('fa-clock');
        });

        test('returns question mark for unknown status', () => {
            expect(window.UIHelpers.getStatusIcon('unknown')).toBe('fa-question');
        });
    });

    describe('getSeverityIcon', () => {
        test('returns correct icons with color classes', () => {
            expect(window.UIHelpers.getSeverityIcon('success')).toBe('fas fa-check-circle text-success');
            expect(window.UIHelpers.getSeverityIcon('warning')).toBe('fas fa-exclamation-triangle text-warning');
            expect(window.UIHelpers.getSeverityIcon('error')).toBe('fas fa-exclamation-circle text-danger');
            expect(window.UIHelpers.getSeverityIcon('danger')).toBe('fas fa-exclamation-circle text-danger');
        });

        test('returns default info icon for unknown severity', () => {
            expect(window.UIHelpers.getSeverityIcon('unknown')).toBe('fas fa-info-circle text-secondary');
        });
    });
});
