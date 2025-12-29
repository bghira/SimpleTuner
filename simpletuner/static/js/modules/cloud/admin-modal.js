/**
 * Cloud Dashboard - Admin Modal Module
 *
 * Handles the admin modal for users, levels, and resource rules management.
 */

window.cloudAdminModalMethods = {
    async openAdminModal() {
        this.adminModal.open = true;
        this.adminModal.tab = 'users';
        await this.loadUsers();
        await this.loadLevels();
    },

    closeAdminModal() {
        this.adminModal.open = false;
    },

    async loadUsers() {
        this.adminModal.usersLoading = true;
        try {
            const response = await fetch('/api/users/');
            if (response.ok) {
                const data = await response.json();
                this.adminModal.users = data.users || [];
                this.filterUsers();
            }
        } catch (error) {
            console.error('Failed to load users:', error);
            if (window.showToast) window.showToast('Failed to load users', 'error');
        } finally {
            this.adminModal.usersLoading = false;
        }
    },

    filterUsers() {
        const query = (this.adminModal.userSearch || '').toLowerCase().trim();
        if (!query) {
            this.adminModal.filteredUsers = this.adminModal.users;
            return;
        }
        this.adminModal.filteredUsers = this.adminModal.users.filter(user => {
            return (user.username || '').toLowerCase().includes(query) ||
                   (user.email || '').toLowerCase().includes(query) ||
                   (user.display_name || '').toLowerCase().includes(query);
        });
    },

    async loadLevels() {
        this.adminModal.levelsLoading = true;
        try {
            const response = await fetch('/api/users/levels');
            if (response.ok) {
                const data = await response.json();
                this.adminModal.levels = data.levels || [];
                // Load rules for each level
                for (const level of this.adminModal.levels) {
                    await this.loadLevelRulesForLevel(level);
                }
            }
        } catch (error) {
            console.error('Failed to load levels:', error);
        } finally {
            this.adminModal.levelsLoading = false;
        }
    },

    async loadLevelRulesForLevel(level) {
        try {
            const response = await fetch(`/api/users/levels/${level.id}/rules`);
            if (response.ok) {
                level.rules = await response.json();
            }
        } catch (error) {
            console.warn(`Failed to load rules for level ${level.id}:`, error);
            level.rules = [];
        }
    },

    async loadResourceRules() {
        this.adminModal.rulesLoading = true;
        try {
            const response = await fetch('/api/users/resource-rules');
            if (response.ok) {
                this.adminModal.rules = await response.json();
                this.filterRules();
            }
        } catch (error) {
            console.error('Failed to load resource rules:', error);
            if (window.showToast) window.showToast('Failed to load resource rules', 'error');
        } finally {
            this.adminModal.rulesLoading = false;
        }
    },

    filterRules() {
        const typeFilter = this.adminModal.ruleTypeFilter;
        if (!typeFilter) {
            this.adminModal.filteredRules = this.adminModal.rules;
            return;
        }
        this.adminModal.filteredRules = this.adminModal.rules.filter(
            rule => rule.resource_type === typeFilter
        );
    },

    showCreateRuleForm() {
        this.adminModal.editingRule = null;
        this.adminModal.ruleForm = {
            name: '',
            resource_type: 'config',
            pattern: '',
            action: 'allow',
            priority: 0,
            description: '',
        };
        this.adminModal.ruleFormOpen = true;
    },

    editRule(rule) {
        this.adminModal.editingRule = rule;
        this.adminModal.ruleForm = {
            name: rule.name,
            resource_type: rule.resource_type,
            pattern: rule.pattern,
            action: rule.action,
            priority: rule.priority,
            description: rule.description || '',
        };
        this.adminModal.ruleFormOpen = true;
    },

    async saveRule() {
        this.adminModal.saving = true;
        try {
            const form = this.adminModal.ruleForm;
            let response;
            if (this.adminModal.editingRule) {
                response = await fetch(`/api/users/resource-rules/${this.adminModal.editingRule.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(form),
                });
            } else {
                response = await fetch('/api/users/resource-rules', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(form),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.adminModal.editingRule ? 'Rule updated' : 'Rule created', 'success');
                }
                this.adminModal.ruleFormOpen = false;
                await this.loadResourceRules();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save rule', 'error');
            }
        } catch (error) {
            console.error('Failed to save rule:', error);
            if (window.showToast) window.showToast('Failed to save rule', 'error');
        } finally {
            this.adminModal.saving = false;
        }
    },

    async confirmDeleteRule(rule) {
        if (!confirm(`Delete rule "${rule.name}"? This cannot be undone.`)) return;

        try {
            const response = await fetch(`/api/users/resource-rules/${rule.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Rule deleted', 'success');
                await this.loadResourceRules();
            } else {
                if (window.showToast) window.showToast('Failed to delete rule', 'error');
            }
        } catch (error) {
            console.error('Failed to delete rule:', error);
            if (window.showToast) window.showToast('Failed to delete rule', 'error');
        }
    },

    editLevelRules(level) {
        this.adminModal.editingLevel = level;
        this.adminModal.selectedLevelRules = (level.rules || []).map(r => r.id);
        this.adminModal.levelRulesOpen = true;
    },

    toggleLevelRule(ruleId) {
        const idx = this.adminModal.selectedLevelRules.indexOf(ruleId);
        if (idx >= 0) {
            this.adminModal.selectedLevelRules.splice(idx, 1);
        } else {
            this.adminModal.selectedLevelRules.push(ruleId);
        }
    },

    async saveLevelRules() {
        if (!this.adminModal.editingLevel) return;
        this.adminModal.saving = true;
        try {
            const response = await fetch(`/api/users/levels/${this.adminModal.editingLevel.id}/rules`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rule_ids: this.adminModal.selectedLevelRules }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Level rules updated', 'success');
                this.adminModal.levelRulesOpen = false;
                await this.loadLevels();
            } else {
                if (window.showToast) window.showToast('Failed to update level rules', 'error');
            }
        } catch (error) {
            console.error('Failed to save level rules:', error);
            if (window.showToast) window.showToast('Failed to save level rules', 'error');
        } finally {
            this.adminModal.saving = false;
        }
    },

    showCreateUserForm() {
        this.adminModal.editingUser = null;
        this.adminModal.userForm = {
            email: '',
            username: '',
            display_name: '',
            password: '',
            is_admin: false,
            is_active: true,
            level_names: ['researcher'],
        };
        this.adminModal.userFormOpen = true;
    },

    editUser(user) {
        this.adminModal.editingUser = user;
        this.adminModal.userForm = {
            email: user.email,
            username: user.username,
            display_name: user.display_name || '',
            password: '',
            is_admin: user.is_admin,
            is_active: user.is_active,
            level_names: (user.levels || []).map(l => l.name),
        };
        this.adminModal.userFormOpen = true;
    },

    toggleUserLevel(levelName) {
        const idx = this.adminModal.userForm.level_names.indexOf(levelName);
        if (idx >= 0) {
            this.adminModal.userForm.level_names.splice(idx, 1);
        } else {
            this.adminModal.userForm.level_names.push(levelName);
        }
    },

    async saveUser() {
        this.adminModal.saving = true;
        try {
            const form = this.adminModal.userForm;
            let response;

            if (this.adminModal.editingUser) {
                const updates = {
                    display_name: form.display_name || null,
                    is_admin: form.is_admin,
                };
                if (form.password) {
                    updates.password = form.password;
                }

                response = await fetch(`/api/users/${this.adminModal.editingUser.id}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updates),
                });

                if (response.ok) {
                    await this.updateUserLevels(this.adminModal.editingUser.id, form.level_names);

                    if (form.is_active !== this.adminModal.editingUser.is_active) {
                        const endpoint = form.is_active ? 'activate' : 'deactivate';
                        await fetch(`/api/users/${this.adminModal.editingUser.id}/${endpoint}`, {
                            method: 'POST',
                        });
                    }
                }
            } else {
                response = await fetch('/api/users', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        email: form.email,
                        username: form.username,
                        password: form.password,
                        display_name: form.display_name || null,
                        is_admin: form.is_admin,
                        level_names: form.level_names,
                    }),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.adminModal.editingUser ? 'User updated' : 'User created', 'success');
                }
                this.adminModal.userFormOpen = false;
                await this.loadUsers();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save user', 'error');
            }
        } catch (error) {
            console.error('Failed to save user:', error);
            if (window.showToast) window.showToast('Failed to save user', 'error');
        } finally {
            this.adminModal.saving = false;
        }
    },

    async updateUserLevels(userId, levelNames) {
        const user = this.adminModal.users.find(u => u.id === userId);
        if (!user) return;

        const currentLevels = new Set((user.levels || []).map(l => l.name));
        const targetLevels = new Set(levelNames);

        for (const level of currentLevels) {
            if (!targetLevels.has(level)) {
                await fetch(`/api/users/${userId}/levels/${level}`, { method: 'DELETE' });
            }
        }

        for (const level of targetLevels) {
            if (!currentLevels.has(level)) {
                await fetch(`/api/users/${userId}/levels`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ level_name: level }),
                });
            }
        }
    },

    confirmDeleteUser(user) {
        if (user.is_admin) {
            if (window.showToast) window.showToast('Cannot delete admin users', 'error');
            return;
        }
        this.adminModal.deletingUser = user;
        this.adminModal.deleteUserOpen = true;
    },

    async deleteUser() {
        if (!this.adminModal.deletingUser) return;
        this.adminModal.saving = true;
        try {
            const response = await fetch(`/api/users/${this.adminModal.deletingUser.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('User deleted', 'success');
                this.adminModal.deleteUserOpen = false;
                this.adminModal.deletingUser = null;
                await this.loadUsers();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to delete user', 'error');
            }
        } catch (error) {
            console.error('Failed to delete user:', error);
            if (window.showToast) window.showToast('Failed to delete user', 'error');
        } finally {
            this.adminModal.saving = false;
        }
    },

    viewLevelDetails(level) {
        const permissions = (level.permissions || []).join(', ') || 'None';
        const ruleCount = (level.rules || []).length;
        alert(`Level: ${level.name}\n\nDescription: ${level.description || 'None'}\nPriority: ${level.priority}\nPermissions: ${permissions}\nResource Rules: ${ruleCount}`);
    },

    formatDate(dateStr) {
        return window.UIHelpers?.formatDateTime(dateStr) || dateStr;
    },
};
