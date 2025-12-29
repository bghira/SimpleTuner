/**
 * Admin Panel - Resource Rules Module
 *
 * Handles resource rule CRUD and filtering.
 */

window.adminRuleMethods = {
    async loadResourceRules() {
        this.rulesLoading = true;
        try {
            const response = await fetch('/api/users/resource-rules');
            if (response.ok) {
                const data = await response.json();
                this.rules = data.rules || data || [];
                this.filterRules();
            }
        } catch (error) {
            console.error('Failed to load resource rules:', error);
            if (window.showToast) window.showToast('Failed to load rules', 'error');
        } finally {
            this.rulesLoading = false;
        }
    },

    filterRules() {
        if (!this.ruleTypeFilter) {
            this.filteredRules = this.rules;
            return;
        }
        this.filteredRules = this.rules.filter(rule => rule.resource_type === this.ruleTypeFilter);
    },

    showCreateRuleForm() {
        this.editingRule = null;
        this.ruleForm = {
            name: '',
            resource_type: 'config',
            action: 'allow',
            pattern: '',
            priority: 0,
            description: '',
        };
        this.ruleFormOpen = true;
    },

    showEditRuleForm(rule) {
        this.editingRule = rule;
        this.ruleForm = {
            name: rule.name,
            resource_type: rule.resource_type,
            action: rule.action,
            pattern: rule.pattern,
            priority: rule.priority,
            description: rule.description || '',
        };
        this.ruleFormOpen = true;
    },

    async saveRule() {
        this.saving = true;
        try {
            let response;
            if (this.editingRule) {
                response = await fetch(`/api/users/resource-rules/${this.editingRule.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.ruleForm),
                });
            } else {
                response = await fetch('/api/users/resource-rules', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.ruleForm),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingRule ? 'Rule updated' : 'Rule created', 'success');
                }
                this.ruleFormOpen = false;
                await this.loadResourceRules();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save rule', 'error');
            }
        } catch (error) {
            console.error('Failed to save rule:', error);
            if (window.showToast) window.showToast('Failed to save rule', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteRule(rule) {
        if (!confirm(`Delete rule "${rule.name}"? This cannot be undone.`)) return;

        try {
            const response = await fetch(`/api/users/resource-rules/${rule.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Rule deleted', 'success');
                await this.loadResourceRules();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to delete rule', 'error');
            }
        } catch (error) {
            console.error('Failed to delete rule:', error);
            if (window.showToast) window.showToast('Failed to delete rule', 'error');
        }
    },

    getTypeBadgeClass(resourceType) {
        return window.UIHelpers?.getResourceTypeBadgeClass(resourceType) || 'bg-secondary';
    },

    getActionBadgeClass(action) {
        return window.UIHelpers?.getActionBadgeClass(action) || 'bg-secondary';
    },
};
