/**
 * Admin Panel - Organizations & Teams Module
 *
 * Handles organization and team CRUD plus entity quotas.
 */

window.adminOrgTeamMethods = {
    async loadOrganizations() {
        this.orgsLoading = true;
        try {
            const response = await fetch('/api/orgs');
            if (response.ok) {
                const data = await response.json();
                this.orgs = data.organizations || [];
                this.filterOrgs();
            }
        } catch (error) {
            console.error('Failed to load organizations:', error);
        } finally {
            this.orgsLoading = false;
        }
    },

    filterOrgs() {
        const query = (this.orgSearch || '').toLowerCase().trim();
        if (!query) {
            this.filteredOrgs = this.orgs;
            return;
        }
        this.filteredOrgs = this.orgs.filter(org => {
            return (org.name || '').toLowerCase().includes(query) ||
                   (org.slug || '').toLowerCase().includes(query);
        });
    },

    showCreateOrgForm() {
        this.editingOrg = null;
        this.orgForm = {
            name: '',
            slug: '',
            description: '',
            is_active: true,
        };
        this.orgFormOpen = true;
    },

    showEditOrgForm(org) {
        this.editingOrg = org;
        this.orgForm = {
            name: org.name,
            slug: org.slug,
            description: org.description || '',
            is_active: org.is_active,
        };
        this.orgFormOpen = true;
    },

    async saveOrg() {
        this.saving = true;
        try {
            let response;
            if (this.editingOrg) {
                response = await fetch(`/api/orgs/${this.editingOrg.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.orgForm),
                });
            } else {
                response = await fetch('/api/orgs', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.orgForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingOrg ? 'Organization updated' : 'Organization created', 'success');
                }
                this.orgFormOpen = false;
                await this.loadOrganizations();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save', 'error');
            }
        } catch (error) {
            console.error('Failed to save organization:', error);
            if (window.showToast) window.showToast('Failed to save organization', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteOrg(org) {
        if (!confirm(`Delete organization "${org.name}"? All teams and quotas will also be deleted.`)) return;

        try {
            const response = await fetch(`/api/orgs/${org.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Organization deleted', 'success');
                if (this.selectedOrg?.id === org.id) {
                    this.selectedOrg = null;
                    this.teams = [];
                }
                await this.loadOrganizations();
            } else {
                if (window.showToast) window.showToast('Failed to delete organization', 'error');
            }
        } catch (error) {
            console.error('Failed to delete organization:', error);
            if (window.showToast) window.showToast('Failed to delete organization', 'error');
        }
    },

    async openOrgTeams(org) {
        this.selectedOrg = org;
        await this.loadTeams(org.id);
    },

    async openOrgQuotas(org) {
        this.quotaTargetType = 'organization';
        this.quotaTargetId = org.id;
        this.quotaTargetName = org.name;
        this.entityQuotaOpen = true;
        await this.loadEntityQuotas();
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

    showCreateTeamForm() {
        if (!this.selectedOrg) return;
        this.editingTeam = null;
        this.teamForm = {
            name: '',
            slug: '',
            description: '',
            is_active: true,
        };
        this.teamFormOpen = true;
    },

    showEditTeamForm(team) {
        this.editingTeam = team;
        this.teamForm = {
            name: team.name,
            slug: team.slug,
            description: team.description || '',
            is_active: team.is_active,
        };
        this.teamFormOpen = true;
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
                    body: JSON.stringify(this.teamForm),
                });
            } else {
                response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.teamForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingTeam ? 'Team updated' : 'Team created', 'success');
                }
                this.teamFormOpen = false;
                await this.loadTeams(this.selectedOrg.id);
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save', 'error');
            }
        } catch (error) {
            console.error('Failed to save team:', error);
            if (window.showToast) window.showToast('Failed to save team', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteTeam(team) {
        if (!confirm(`Delete team "${team.name}"?`)) return;
        if (!this.selectedOrg) return;

        try {
            const response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${team.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Team deleted', 'success');
                await this.loadTeams(this.selectedOrg.id);
            } else {
                if (window.showToast) window.showToast('Failed to delete team', 'error');
            }
        } catch (error) {
            console.error('Failed to delete team:', error);
            if (window.showToast) window.showToast('Failed to delete team', 'error');
        }
    },

    async openTeamQuotas(team) {
        this.quotaTargetType = 'team';
        this.quotaTargetId = team.id;
        this.quotaTargetName = team.name;
        this.entityQuotaOpen = true;
        await this.loadEntityQuotas();
    },

    async openTeamMembers(team) {
        this.selectedTeam = team;
        this.teamMembersOpen = true;
        await this.loadTeamMembers(team.id);
        await this.loadAvailableTeamUsers();
    },

    async loadTeamMembers(teamId) {
        if (!this.selectedOrg) return;
        this.teamMembersLoading = true;
        try {
            const response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${teamId}/members`);
            if (response.ok) {
                const data = await response.json();
                this.teamMembers = data.members || [];
            }
        } catch (error) {
            console.error('Failed to load team members:', error);
        } finally {
            this.teamMembersLoading = false;
        }
    },

    async loadAvailableTeamUsers() {
        if (!this.selectedOrg) return;
        try {
            // Get users in the same org who aren't in this team
            const response = await fetch(`/api/orgs/${this.selectedOrg.id}/members`);
            if (response.ok) {
                const data = await response.json();
                const teamMemberIds = new Set(this.teamMembers.map(m => m.id));
                this.availableTeamUsers = (data.members || []).filter(u => !teamMemberIds.has(u.id));
            }
        } catch (error) {
            console.error('Failed to load available users:', error);
        }
    },

    async addTeamMember() {
        if (!this.selectedTeam || !this.addMemberUserId || !this.selectedOrg) return;
        this.saving = true;
        try {
            const response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${this.selectedTeam.id}/members`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: this.addMemberUserId, role: 'member' }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Member added to team', 'success');
                await this.loadTeamMembers(this.selectedTeam.id);
                await this.loadAvailableTeamUsers();
                await this.loadTeams(this.selectedOrg.id); // Refresh member counts
                this.addMemberUserId = null;
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to add member', 'error');
            }
        } catch (error) {
            console.error('Failed to add team member:', error);
            if (window.showToast) window.showToast('Failed to add member', 'error');
        } finally {
            this.saving = false;
        }
    },

    async removeTeamMember(member) {
        if (!this.selectedTeam || !this.selectedOrg) return;
        if (!confirm(`Remove ${member.display_name || member.username} from ${this.selectedTeam.name}?`)) return;

        try {
            const response = await fetch(`/api/orgs/${this.selectedOrg.id}/teams/${this.selectedTeam.id}/members/${member.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Member removed from team', 'success');
                await this.loadTeamMembers(this.selectedTeam.id);
                await this.loadAvailableTeamUsers();
                await this.loadTeams(this.selectedOrg.id); // Refresh member counts
            } else {
                if (window.showToast) window.showToast('Failed to remove member', 'error');
            }
        } catch (error) {
            console.error('Failed to remove team member:', error);
            if (window.showToast) window.showToast('Failed to remove member', 'error');
        }
    },

    closeTeamMembers() {
        this.teamMembersOpen = false;
        this.selectedTeam = null;
        this.teamMembers = [];
        this.availableTeamUsers = [];
        this.addMemberUserId = null;
    },

    getEntityQuotaUrl() {
        // Build the correct URL based on target type
        if (this.quotaTargetType === 'organization') {
            return `/api/orgs/${this.quotaTargetId}/quotas`;
        } else if (this.quotaTargetType === 'team' && this.selectedOrg) {
            return `/api/orgs/${this.selectedOrg.id}/teams/${this.quotaTargetId}/quotas`;
        }
        return null;
    },

    async loadQuotaTypes() {
        // Only load once
        if (this.quotaTypes.length > 0) return;

        try {
            const response = await fetch('/api/quotas/types');
            if (response.ok) {
                const data = await response.json();
                this.quotaTypes = data.quota_types || [];
                this.quotaActions = data.actions || [];
            }
        } catch (error) {
            console.error('Failed to load quota types:', error);
            // Fallback to defaults if API fails
            this.quotaTypes = [
                { value: 'concurrent_jobs', label: 'Concurrent Jobs', description: 'Maximum running jobs at once' },
                { value: 'jobs_per_day', label: 'Jobs per Day', description: 'Maximum job submissions per day' },
                { value: 'jobs_per_hour', label: 'Jobs per Hour', description: 'Maximum job submissions per hour' },
                { value: 'cost_daily', label: 'Daily Cost ($)', description: 'Maximum USD spend per day' },
                { value: 'cost_monthly', label: 'Monthly Cost ($)', description: 'Maximum USD spend per month' },
            ];
            this.quotaActions = [
                { value: 'block', label: 'Block', description: 'Block action when quota exceeded' },
                { value: 'warn', label: 'Warn', description: 'Allow but show warning' },
                { value: 'require_approval', label: 'Require Approval', description: 'Require admin approval' },
            ];
        }
    },

    async loadEntityQuotas() {
        const url = this.getEntityQuotaUrl();
        if (!url) return;

        // Load quota types if needed
        await this.loadQuotaTypes();

        try {
            const response = await fetch(url);
            if (response.ok) {
                const data = await response.json();
                this.entityQuotas = data.quotas || [];
            }
        } catch (error) {
            console.error('Failed to load entity quotas:', error);
        }
    },

    async addEntityQuota() {
        const url = this.getEntityQuotaUrl();
        if (!url) return;

        this.saving = true;
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.newEntityQuota),
            });

            if (response.ok) {
                if (window.showToast) window.showToast('Quota added', 'success');
                await this.loadEntityQuotas();
                this.newEntityQuota = {
                    quota_type: 'concurrent_jobs',
                    limit_value: 10,
                    action: 'block',
                };
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to add quota', 'error');
            }
        } catch (error) {
            console.error('Failed to add entity quota:', error);
            if (window.showToast) window.showToast('Failed to add quota', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteEntityQuota(quota) {
        if (!confirm('Delete this quota ceiling?')) return;

        const url = this.getEntityQuotaUrl();
        if (!url) return;

        try {
            const response = await fetch(`${url}/${quota.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Quota deleted', 'success');
                await this.loadEntityQuotas();
            } else {
                if (window.showToast) window.showToast('Failed to delete quota', 'error');
            }
        } catch (error) {
            console.error('Failed to delete entity quota:', error);
            if (window.showToast) window.showToast('Failed to delete quota', 'error');
        }
    },
};
