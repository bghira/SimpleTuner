/**
 * Admin Panel - Users Management Module
 *
 * Handles user CRUD, filtering, and credential checks.
 */

window.adminUserMethods = {
    async loadUsers() {
        this.usersLoading = true;
        try {
            const response = await fetch('/api/users/');
            if (response.ok) {
                const data = await response.json();
                // API returns array directly or {users: [...]}
                this.users = Array.isArray(data) ? data : (data.users || []);
                this.filterUsers();
                this.updateStats();
            }
        } catch (error) {
            console.error('Failed to load users:', error);
            if (window.showToast) window.showToast('Failed to load users', 'error');
        } finally {
            this.usersLoading = false;
        }
    },

    updateStats() {
        this.stats.userCount = this.users.length;
        this.stats.activeUserCount = this.users.filter(u => u.is_active).length;
        this.stats.externalUserCount = this.users.filter(u => u.auth_provider && u.auth_provider !== 'local').length;
        this.stats.levelCount = this.levels?.length || 0;
        this.stats.ruleCount = this.rules?.length || 0;
        this.stats.quotaCount = this.quotas?.length || 0;
        this.stats.approvalRuleCount = this.approvalRules?.length || 0;
        this.stats.pendingApprovals = this.pendingApprovals?.length || 0;
        this.stats.authProvidersConfigured = this.authProviders?.filter(p => p.enabled).length || 0;
        this.stats.orgCount = this.orgs?.length || 0;
    },

    filterUsers() {
        const query = (this.userSearch || '').toLowerCase().trim();
        if (!query) {
            this.filteredUsers = this.users;
            return;
        }
        this.filteredUsers = this.users.filter(user => {
            return (user.username || '').toLowerCase().includes(query) ||
                   (user.email || '').toLowerCase().includes(query) ||
                   (user.display_name || '').toLowerCase().includes(query);
        });
    },

    showCreateUserForm() {
        this.editingUser = null;
        this.userForm = {
            email: '',
            username: '',
            display_name: '',
            password: '',
            is_admin: false,
            is_active: true,
            level_names: [],
        };
        this.userFormOpen = true;
    },

    showEditUserForm(user) {
        this.editingUser = user;
        this.userForm = {
            email: user.email,
            username: user.username,
            display_name: user.display_name || '',
            password: '',
            is_admin: user.is_admin,
            is_active: user.is_active,
            level_names: (user.level_names || []).slice(),
        };
        this.userFormOpen = true;
    },

    toggleUserLevel(levelName) {
        const idx = this.userForm.level_names.indexOf(levelName);
        if (idx >= 0) {
            this.userForm.level_names.splice(idx, 1);
        } else {
            this.userForm.level_names.push(levelName);
        }
    },

    async saveUser() {
        this.saving = true;
        this.error = null;
        try {
            let response;
            if (this.editingUser) {
                const payload = {
                    display_name: this.userForm.display_name || null,
                    is_admin: this.userForm.is_admin,
                    is_active: this.userForm.is_active,
                    level_names: this.userForm.level_names,
                };
                if (this.userForm.password) {
                    payload.password = this.userForm.password;
                }
                response = await fetch(`/api/users/${this.editingUser.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            } else {
                response = await fetch('/api/users', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: this.userForm.email,
                        username: this.userForm.username,
                        password: this.userForm.password,
                        display_name: this.userForm.display_name || null,
                        is_admin: this.userForm.is_admin,
                        level_names: this.userForm.level_names,
                    }),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingUser ? 'User updated' : 'User created', 'success');
                }
                this.userFormOpen = false;
                await this.loadUsers();
            } else {
                const data = await response.json();
                this.error = window.UIHelpers?.extractErrorMessage(data, 'Failed to save user') || 'Failed to save user';
                if (window.showToast) window.showToast(this.error, 'error');
            }
        } catch (error) {
            console.error('Failed to save user:', error);
            this.error = 'Network error';
            if (window.showToast) window.showToast('Failed to save user', 'error');
        } finally {
            this.saving = false;
        }
    },

    confirmDeleteUser(user) {
        if (user.id === this.currentUser?.id) {
            if (window.showToast) window.showToast('Cannot delete yourself', 'error');
            return;
        }
        this.deletingUser = user;
        this.deleteUserOpen = true;
    },

    async deleteUser() {
        if (!this.deletingUser) return;
        this.saving = true;
        try {
            const response = await fetch(`/api/users/${this.deletingUser.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('User deleted', 'success');
                this.deleteUserOpen = false;
                this.deletingUser = null;
                await this.loadUsers();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to delete user', 'error');
            }
        } catch (error) {
            console.error('Failed to delete user:', error);
            if (window.showToast) window.showToast('Failed to delete user', 'error');
        } finally {
            this.saving = false;
        }
    },

    // ==================== Permission Overrides ====================

    async openPermissionOverridesModal(user) {
        this.permissionOverridesUser = user;
        this.permissionOverridesModalOpen = true;
        this.newOverride = { permission: '', granted: true };
        await this.loadUserPermissionOverrides();
    },

    closePermissionOverridesModal() {
        this.permissionOverridesModalOpen = false;
        this.permissionOverridesUser = null;
        this.userPermissionOverrides = [];
    },

    async loadUserPermissionOverrides() {
        if (!this.permissionOverridesUser) return;

        this.permissionOverridesLoading = true;
        try {
            // Get user with full permissions to extract overrides
            const response = await fetch(`/api/users/${this.permissionOverridesUser.id}`);
            if (response.ok) {
                const data = await response.json();
                // Extract permission overrides from user data
                this.userPermissionOverrides = data.user?.permission_overrides || [];
            }
        } catch (error) {
            console.error('Failed to load permission overrides:', error);
        } finally {
            this.permissionOverridesLoading = false;
        }
    },

    async addPermissionOverride() {
        if (!this.permissionOverridesUser || !this.newOverride.permission) return;

        this.saving = true;
        try {
            const response = await fetch(`/api/users/${this.permissionOverridesUser.id}/permissions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    permission_name: this.newOverride.permission,
                    granted: this.newOverride.granted,
                }),
            });

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(
                        `Permission ${this.newOverride.granted ? 'granted' : 'denied'}: ${this.newOverride.permission}`,
                        'success'
                    );
                }
                this.newOverride = { permission: '', granted: true };
                await this.loadUserPermissionOverrides();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to add override', 'error');
            }
        } catch (error) {
            console.error('Failed to add permission override:', error);
            if (window.showToast) window.showToast('Failed to add override', 'error');
        } finally {
            this.saving = false;
        }
    },

    async removePermissionOverride(permissionName) {
        if (!this.permissionOverridesUser) return;

        try {
            const response = await fetch(
                `/api/users/${this.permissionOverridesUser.id}/permissions/${encodeURIComponent(permissionName)}`,
                { method: 'DELETE' }
            );

            if (response.ok) {
                if (window.showToast) window.showToast(`Override removed: ${permissionName}`, 'success');
                await this.loadUserPermissionOverrides();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to remove override', 'error');
            }
        } catch (error) {
            console.error('Failed to remove permission override:', error);
            if (window.showToast) window.showToast('Failed to remove override', 'error');
        }
    },

    getAvailablePermissionsForOverride() {
        // Filter out permissions that already have overrides
        const existingOverrides = new Set(this.userPermissionOverrides.map(o => o.permission_name));
        return (this.allPermissions || []).filter(p => !existingOverrides.has(p.name));
    },
};
