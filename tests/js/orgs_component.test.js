/**
 * Tests for organizations component.
 *
 * Tests the orgsComponent Alpine.js component methods including
 * role management and member operations.
 */

// Mock window.showToast
window.showToast = jest.fn();

// Mock window.HintMixin
window.HintMixin = {
    createSingleHint: () => ({
        showHeroCTA: false,
        loadHeroCTAState: jest.fn(),
        dismissHeroCTA: jest.fn(),
        restoreHeroCTA: jest.fn(),
    }),
};

// Mock window.UIHelpers
window.UIHelpers = {
    getRoleBadgeClass: (role) => {
        const classes = { admin: 'bg-danger', lead: 'bg-primary', member: 'bg-secondary' };
        return classes[role] || 'bg-secondary';
    },
};

// Mock bootstrap.Modal
window.bootstrap = {
    Modal: jest.fn().mockImplementation(() => ({
        show: jest.fn(),
        hide: jest.fn(),
    })),
};

// Load the component
require('../../simpletuner/static/js/orgs_component.js');

describe('orgsComponent', () => {
    let component;

    beforeEach(() => {
        // Reset mocks
        fetch.mockReset();
        window.showToast.mockReset();

        // Create component instance
        component = window.orgsComponent();

        // Mock $nextTick and $refs
        component.$nextTick = (fn) => fn();
        component.$refs = {
            orgModal: document.createElement('div'),
            teamModal: document.createElement('div'),
            addMemberModal: document.createElement('div'),
            deleteModal: document.createElement('div'),
        };

        // Set up required state
        component.selectedOrg = { id: 1, name: 'Test Org' };
        component.selectedTeam = { id: 10, name: 'Test Team' };
        component.teamMembers = [
            { id: 100, username: 'alice', role: 'admin' },
            { id: 101, username: 'bob', role: 'lead' },
            { id: 102, username: 'charlie', role: 'member' },
        ];
    });

    describe('updateMemberRole', () => {
        test('does nothing when role unchanged', async () => {
            const member = { id: 100, role: 'admin' };
            await component.updateMemberRole(member, 'admin');

            expect(fetch).not.toHaveBeenCalled();
        });

        test('calls PATCH endpoint with new role', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true }),
            });

            const member = { id: 102, role: 'member' };
            await component.updateMemberRole(member, 'lead');

            expect(fetch).toHaveBeenCalledWith(
                '/api/cloud/orgs/1/teams/10/members/102',
                expect.objectContaining({
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ role: 'lead' }),
                })
            );
        });

        test('updates member role on success', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true }),
            });

            const member = { id: 102, role: 'member' };
            await component.updateMemberRole(member, 'lead');

            expect(member.role).toBe('lead');
            expect(window.showToast).toHaveBeenCalledWith('Role updated to lead', 'success');
        });

        test('shows error toast on failure', async () => {
            fetch.mockResolvedValueOnce({
                ok: false,
                json: () => Promise.resolve({ detail: 'Permission denied' }),
            });

            // Mock loadTeamMembers to prevent additional fetch
            component.loadTeamMembers = jest.fn();

            const member = { id: 102, role: 'member' };
            await component.updateMemberRole(member, 'admin');

            expect(window.showToast).toHaveBeenCalledWith('Permission denied', 'error');
            expect(component.loadTeamMembers).toHaveBeenCalled();
        });

        test('reloads members on network error', async () => {
            fetch.mockRejectedValueOnce(new Error('Network error'));
            component.loadTeamMembers = jest.fn();

            const member = { id: 102, role: 'member' };
            await component.updateMemberRole(member, 'lead');

            expect(window.showToast).toHaveBeenCalledWith('Network error', 'error');
            expect(component.loadTeamMembers).toHaveBeenCalledWith(10);
        });
    });

    describe('getRoleBadgeClass', () => {
        test('returns correct class for each role', () => {
            expect(component.getRoleBadgeClass('admin')).toBe('bg-danger');
            expect(component.getRoleBadgeClass('lead')).toBe('bg-primary');
            expect(component.getRoleBadgeClass('member')).toBe('bg-secondary');
        });

        test('returns default for unknown role', () => {
            expect(component.getRoleBadgeClass('unknown')).toBe('bg-secondary');
        });
    });

    describe('addMember', () => {
        test('does nothing without user_id', async () => {
            component.memberForm = { user_id: '', role: 'member' };
            await component.addMember();

            expect(fetch).not.toHaveBeenCalled();
        });

        test('calls POST endpoint with member data', async () => {
            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({ success: true }),
            });
            component.loadTeamMembers = jest.fn();
            component.addMemberModal = { hide: jest.fn() };

            component.memberForm = { user_id: '105', role: 'lead' };
            await component.addMember();

            expect(fetch).toHaveBeenCalledWith(
                '/api/cloud/orgs/1/teams/10/members',
                expect.objectContaining({
                    method: 'POST',
                    body: JSON.stringify({ user_id: 105, role: 'lead' }),
                })
            );
        });
    });

    describe('removeMember', () => {
        beforeEach(() => {
            // Mock confirm
            window.confirm = jest.fn(() => true);
        });

        test('aborts if user cancels confirm', async () => {
            window.confirm.mockReturnValueOnce(false);

            const member = { id: 102, username: 'charlie' };
            await component.removeMember(member);

            expect(fetch).not.toHaveBeenCalled();
        });

        test('calls DELETE endpoint', async () => {
            fetch.mockResolvedValueOnce({ ok: true, status: 204 });
            component.loadTeamMembers = jest.fn();

            const member = { id: 102, username: 'charlie' };
            await component.removeMember(member);

            expect(fetch).toHaveBeenCalledWith(
                '/api/cloud/orgs/1/teams/10/members/102',
                { method: 'DELETE' }
            );
            expect(component.loadTeamMembers).toHaveBeenCalledWith(10);
        });
    });
});
