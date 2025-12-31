/**
 * Admin Panel - Levels Management Module
 *
 * Handles access level CRUD and rule/permission assignment.
 */

window.adminLevelMethods = {
    async loadLevels() {
        this.levelsLoading = true;
        try {
            const response = await fetch('/api/users/meta/levels');
            if (response.ok) {
                const data = await response.json();
                this.levels = data.levels || [];
            }
        } catch (error) {
            console.error('Failed to load levels:', error);
        } finally {
            this.levelsLoading = false;
        }
    },

    async loadPermissions() {
        try {
            const response = await fetch('/api/users/meta/permissions');
            if (response.ok) {
                const data = await response.json();
                this.allPermissions = data.permissions || [];
            }
        } catch (error) {
            console.error('Failed to load permissions:', error);
        }
    },

    showCreateLevelForm() {
        this.editingLevel = null;
        this.levelForm = {
            name: '',
            description: '',
            priority: 0,
            permission_names: [],
        };
        this.levelFormOpen = true;
    },

    showEditLevelForm(level) {
        this.editingLevel = level;
        this.levelForm = {
            name: level.name,
            description: level.description || '',
            priority: level.priority,
            permission_names: (level.permission_names || []).slice(),
        };
        this.levelFormOpen = true;
    },

    async saveLevel() {
        this.saving = true;
        try {
            let response;
            if (this.editingLevel) {
                response = await fetch(`/api/users/levels/${this.editingLevel.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.levelForm),
                });
            } else {
                response = await fetch('/api/users/levels', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.levelForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingLevel ? 'Level updated' : 'Level created', 'success');
                }
                this.levelFormOpen = false;
                await this.loadLevels();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save level', 'error');
            }
        } catch (error) {
            console.error('Failed to save level:', error);
            if (window.showToast) window.showToast('Failed to save level', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteLevel(level) {
        if (!confirm(`Delete level "${level.name}"? Users with only this level will lose access.`)) return;

        try {
            const response = await fetch(`/api/users/levels/${level.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Level deleted', 'success');
                await this.loadLevels();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to delete level', 'error');
            }
        } catch (error) {
            console.error('Failed to delete level:', error);
            if (window.showToast) window.showToast('Failed to delete level', 'error');
        }
    },

    openLevelRules(level) {
        this.editingLevel = level;
        this.selectedLevelRules = (level.rule_ids || []).slice();
        this.levelRulesOpen = true;
    },

    openLevelPermissions(level) {
        this.editingLevel = level;
        this.selectedPermissions = (level.permission_names || []).slice();
        this.levelPermissionsOpen = true;
    },

    toggleLevelRule(ruleId) {
        const idx = this.selectedLevelRules.indexOf(ruleId);
        if (idx >= 0) {
            this.selectedLevelRules.splice(idx, 1);
        } else {
            this.selectedLevelRules.push(ruleId);
        }
    },

    togglePermission(permName) {
        const idx = this.selectedPermissions.indexOf(permName);
        if (idx >= 0) {
            this.selectedPermissions.splice(idx, 1);
        } else {
            this.selectedPermissions.push(permName);
        }
    },

    async saveLevelRules() {
        if (!this.editingLevel) return;
        this.saving = true;
        try {
            const response = await fetch(`/api/users/levels/${this.editingLevel.id}/rules`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ rule_ids: this.selectedLevelRules }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Level rules updated', 'success');
                this.levelRulesOpen = false;
                await this.loadLevels();
            } else {
                if (window.showToast) window.showToast('Failed to update rules', 'error');
            }
        } catch (error) {
            console.error('Failed to save level rules:', error);
            if (window.showToast) window.showToast('Failed to save rules', 'error');
        } finally {
            this.saving = false;
        }
    },

    async saveLevelPermissions() {
        if (!this.editingLevel) return;
        this.saving = true;
        try {
            const response = await fetch(`/api/users/levels/${this.editingLevel.id}/permissions`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ permission_names: this.selectedPermissions }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Level permissions updated', 'success');
                this.levelPermissionsOpen = false;
                await this.loadLevels();
            } else {
                if (window.showToast) window.showToast('Failed to update permissions', 'error');
            }
        } catch (error) {
            console.error('Failed to save level permissions:', error);
            if (window.showToast) window.showToast('Failed to save permissions', 'error');
        } finally {
            this.saving = false;
        }
    },
};
