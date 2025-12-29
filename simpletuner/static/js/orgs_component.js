/**
 * Organizations & Teams Alpine.js Component
 *
 * Manages organizations, teams, and memberships.
 */

window.orgsComponent = function() {
    return {
        // State
        loading: false,
        saving: false,
        deleting: false,
        organizations: [],
        teams: [],
        teamMembers: [],
        orgMembers: [],
        availableUsers: [],

        // Selection
        selectedOrg: null,
        selectedTeam: null,

        // Loading states
        teamsLoading: false,
        membersLoading: false,
        orgMembersLoading: false,

        // Hero CTA state (using HintMixin)
        ...(window.HintMixin?.createSingleHint({
            useApi: true,
            hintKey: 'orgs_hero',
        }) || { heroDismissed: false, loadHeroCTAState() {}, dismissHeroCTA() {}, restoreHeroCTA() {} }),

        // Modal state
        editingOrg: null,
        editingTeam: null,
        deleteTarget: null,
        deleteType: null,
        deleteMessage: '',

        // Modals
        orgModal: null,
        teamModal: null,
        addMemberModal: null,
        deleteModal: null,

        // Forms
        orgForm: {
            name: '',
            slug: '',
            description: '',
        },
        teamForm: {
            name: '',
            slug: '',
            description: '',
        },
        memberForm: {
            user_id: '',
            role: 'member',
        },

        init() {
            this.loadHeroCTAState();
            this.loadOrganizations();
            this.loadAvailableUsers();

            this.$nextTick(() => {
                if (typeof bootstrap !== 'undefined') {
                    if (this.$refs.orgModal) {
                        this.orgModal = new bootstrap.Modal(this.$refs.orgModal);
                    }
                    if (this.$refs.teamModal) {
                        this.teamModal = new bootstrap.Modal(this.$refs.teamModal);
                    }
                    if (this.$refs.addMemberModal) {
                        this.addMemberModal = new bootstrap.Modal(this.$refs.addMemberModal);
                    }
                    if (this.$refs.deleteModal) {
                        this.deleteModal = new bootstrap.Modal(this.$refs.deleteModal);
                    }
                }
            });
        },

        // Hero CTA methods provided by HintMixin spread above

        get totalTeams() {
            return this.organizations.reduce((acc, org) => acc + (org.team_count || 0), 0);
        },

        get totalMembers() {
            return this.organizations.reduce((acc, org) => acc + (org.member_count || 0), 0);
        },

        async loadOrganizations() {
            this.loading = true;
            try {
                const response = await fetch('/api/orgs');
                if (response.ok) {
                    const data = await response.json();
                    // Server returns enriched orgs with team_count and member_count
                    this.organizations = data.organizations || [];
                }
            } catch (error) {
                console.error('Failed to load organizations:', error);
            } finally {
                this.loading = false;
            }
        },

        async loadAvailableUsers() {
            try {
                const response = await fetch('/api/users');
                if (response.ok) {
                    const data = await response.json();
                    this.availableUsers = data.users || [];
                }
            } catch (error) {
                console.error('Failed to load users:', error);
            }
        },

        async selectOrg(org) {
            this.selectedOrg = org;
            this.selectedTeam = null;
            this.teamMembers = [];
            await this.loadTeams(org.id);
            await this.loadOrgMembers(org.id);
        },

        async loadTeams(orgId) {
            this.teamsLoading = true;
            try {
                const response = await fetch(`/api/orgs/${orgId}/teams`);
                if (response.ok) {
                    const data = await response.json();
                    this.teams = data.teams || [];
                }
            } catch (error) {
                console.error('Failed to load teams:', error);
            } finally {
                this.teamsLoading = false;
            }
        },

        async loadOrgMembers(orgId) {
            this.orgMembersLoading = true;
            try {
                const response = await fetch(`/api/orgs/${orgId}/members`);
                if (response.ok) {
                    const data = await response.json();
                    this.orgMembers = data.members || [];
                }
            } catch (error) {
                console.error('Failed to load org members:', error);
            } finally {
                this.orgMembersLoading = false;
            }
        },

        async selectTeam(team) {
            this.selectedTeam = team;
            await this.loadTeamMembers(team.id);
        },

        async loadTeamMembers(teamId) {
            if (!this.selectedOrg) return;

            this.membersLoading = true;
            try {
                const response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${teamId}/members`);
                if (response.ok) {
                    const data = await response.json();
                    this.teamMembers = data.members || [];
                }
            } catch (error) {
                console.error('Failed to load team members:', error);
            } finally {
                this.membersLoading = false;
            }
        },

        // Organization CRUD
        showOrgModal() {
            this.editingOrg = null;
            this.orgForm = { name: '', slug: '', description: '' };
            if (this.orgModal) {
                this.orgModal.show();
            }
        },

        editOrg(org) {
            this.editingOrg = org;
            this.orgForm = {
                name: org.name,
                slug: org.slug,
                description: org.description || '',
            };
            if (this.orgModal) {
                this.orgModal.show();
            }
        },

        async saveOrg() {
            this.saving = true;
            try {
                let response;
                if (this.editingOrg) {
                    response = await fetch(`/api/orgs/${this.editingOrg.id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            name: this.orgForm.name,
                            description: this.orgForm.description,
                        }),
                    });
                } else {
                    response = await fetch('/api/orgs', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.orgForm),
                    });
                }

                if (response.ok) {
                    if (this.orgModal) {
                        this.orgModal.hide();
                    }
                    await this.loadOrganizations();
                    if (window.showToast) {
                        window.showToast(
                            this.editingOrg ? 'Organization updated' : 'Organization created',
                            'success'
                        );
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to save organization');
                }
            } catch (error) {
                console.error('Failed to save organization:', error);
                if (window.showToast) {
                    window.showToast(error.message, 'error');
                }
            } finally {
                this.saving = false;
            }
        },

        confirmDeleteOrg(org) {
            this.deleteTarget = org;
            this.deleteType = 'org';
            this.deleteMessage = `Are you sure you want to delete "${org.name}"? This will also delete all teams within this organization.`;
            if (this.deleteModal) {
                this.deleteModal.show();
            }
        },

        // Team CRUD
        showTeamModal() {
            if (!this.selectedOrg) return;
            this.editingTeam = null;
            this.teamForm = { name: '', slug: '', description: '' };
            if (this.teamModal) {
                this.teamModal.show();
            }
        },

        editTeam(team) {
            this.editingTeam = team;
            this.teamForm = {
                name: team.name,
                slug: team.slug,
                description: team.description || '',
            };
            if (this.teamModal) {
                this.teamModal.show();
            }
        },

        async saveTeam() {
            if (!this.selectedOrg) return;

            this.saving = true;
            try {
                let response;
                if (this.editingTeam) {
                    response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${this.editingTeam.id}`, {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            name: this.teamForm.name,
                            description: this.teamForm.description,
                        }),
                    });
                } else {
                    response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(this.teamForm),
                    });
                }

                if (response.ok) {
                    if (this.teamModal) {
                        this.teamModal.hide();
                    }
                    await this.loadTeams(this.selectedOrg.id);
                    if (window.showToast) {
                        window.showToast(
                            this.editingTeam ? 'Team updated' : 'Team created',
                            'success'
                        );
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to save team');
                }
            } catch (error) {
                console.error('Failed to save team:', error);
                if (window.showToast) {
                    window.showToast(error.message, 'error');
                }
            } finally {
                this.saving = false;
            }
        },

        confirmDeleteTeam(team) {
            this.deleteTarget = team;
            this.deleteType = 'team';
            this.deleteMessage = `Are you sure you want to delete the team "${team.name}"?`;
            if (this.deleteModal) {
                this.deleteModal.show();
            }
        },

        async confirmDelete() {
            this.deleting = true;
            try {
                let url;
                if (this.deleteType === 'org') {
                    url = `/api/orgs/${this.deleteTarget.id}`;
                } else if (this.deleteType === 'team') {
                    url = `/api/orgs/${this.selectedOrg.id}/teams/${this.deleteTarget.id}`;
                } else {
                    return;
                }

                const response = await fetch(url, { method: 'DELETE' });

                if (response.ok || response.status === 204) {
                    if (this.deleteModal) {
                        this.deleteModal.hide();
                    }

                    if (this.deleteType === 'org') {
                        if (this.selectedOrg?.id === this.deleteTarget.id) {
                            this.selectedOrg = null;
                            this.teams = [];
                        }
                        await this.loadOrganizations();
                    } else if (this.deleteType === 'team') {
                        if (this.selectedTeam?.id === this.deleteTarget.id) {
                            this.selectedTeam = null;
                            this.teamMembers = [];
                        }
                        await this.loadTeams(this.selectedOrg.id);
                    }

                    if (window.showToast) {
                        window.showToast(`${this.deleteType === 'org' ? 'Organization' : 'Team'} deleted`, 'success');
                    }
                } else {
                    throw new Error('Failed to delete');
                }
            } catch (error) {
                console.error('Failed to delete:', error);
                if (window.showToast) {
                    window.showToast('Failed to delete', 'error');
                }
            } finally {
                this.deleting = false;
                this.deleteTarget = null;
                this.deleteType = null;
            }
        },

        // Team Membership
        showAddMemberModal() {
            if (!this.selectedTeam) return;
            this.memberForm = { user_id: '', role: 'member' };
            if (this.addMemberModal) {
                this.addMemberModal.show();
            }
        },

        showAddOrgMemberModal() {
            if (!this.selectedOrg) return;
            this.memberForm = { user_id: '', role: 'member' };
            if (this.addMemberModal) {
                this.addMemberModal.show();
            }
        },

        async addMember() {
            if (!this.memberForm.user_id) return;

            this.saving = true;
            try {
                const response = await fetch(
                    `/api/orgs/${this.selectedOrg.id}/teams/${this.selectedTeam.id}/members`,
                    {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            user_id: parseInt(this.memberForm.user_id),
                            role: this.memberForm.role,
                        }),
                    }
                );

                if (response.ok) {
                    if (this.addMemberModal) {
                        this.addMemberModal.hide();
                    }
                    await this.loadTeamMembers(this.selectedTeam.id);
                    if (window.showToast) {
                        window.showToast('Member added', 'success');
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.message || 'Failed to add member');
                }
            } catch (error) {
                console.error('Failed to add member:', error);
                if (window.showToast) {
                    window.showToast(error.message, 'error');
                }
            } finally {
                this.saving = false;
            }
        },

        async removeMember(member) {
            if (!confirm(`Remove ${member.display_name || member.username} from this team?`)) return;

            try {
                const response = await fetch(
                    `/api/orgs/${this.selectedOrg.id}/teams/${this.selectedTeam.id}/members/${member.id}`,
                    { method: 'DELETE' }
                );

                if (response.ok || response.status === 204) {
                    await this.loadTeamMembers(this.selectedTeam.id);
                    if (window.showToast) {
                        window.showToast('Member removed', 'success');
                    }
                }
            } catch (error) {
                console.error('Failed to remove member:', error);
            }
        },

        async updateMemberRole(member, newRole) {
            if (member.role === newRole) return;

            try {
                const response = await fetch(
                    `/api/orgs/${this.selectedOrg.id}/teams/${this.selectedTeam.id}/members/${member.id}`,
                    {
                        method: 'PATCH',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ role: newRole }),
                    }
                );

                if (response.ok) {
                    member.role = newRole;
                    if (window.showToast) {
                        window.showToast(`Role updated to ${newRole}`, 'success');
                    }
                } else {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to update role');
                }
            } catch (error) {
                console.error('Failed to update member role:', error);
                if (window.showToast) {
                    window.showToast(error.message, 'error');
                }
                await this.loadTeamMembers(this.selectedTeam.id);
            }
        },

        async removeOrgMember(member) {
            if (!confirm(`Remove ${member.display_name || member.username} from this organization?`)) return;

            try {
                const response = await fetch(
                    `/api/orgs/${this.selectedOrg.id}/members/${member.id}`,
                    { method: 'DELETE' }
                );

                if (response.ok || response.status === 204) {
                    await this.loadOrgMembers(this.selectedOrg.id);
                    if (window.showToast) {
                        window.showToast('Member removed from organization', 'success');
                    }
                }
            } catch (error) {
                console.error('Failed to remove org member:', error);
            }
        },

        // Helpers
        getRoleBadgeClass(role) {
            return window.UIHelpers?.getRoleBadgeClass(role) || 'bg-secondary';
        },
    };
};
