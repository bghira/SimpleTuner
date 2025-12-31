/**
 * Admin Panel - Approval Rules Module
 *
 * Handles approval rule CRUD.
 */

window.adminApprovalMethods = {
    async loadApprovalRules() {
        this.approvalRulesLoading = true;
        try {
            const response = await fetch('/api/approvals/rules');
            if (response.ok) {
                const data = await response.json();
                this.approvalRules = data.rules || [];
            }
        } catch (error) {
            console.error('Failed to load approval rules:', error);
        } finally {
            this.approvalRulesLoading = false;
        }
    },

    async loadApprovalConditions() {
        this.approvalConditionsLoading = true;
        try {
            const response = await fetch('/api/approvals/conditions');
            if (response.ok) {
                const data = await response.json();
                this.approvalConditions = data.conditions || [];
            }
        } catch (error) {
            console.error('Failed to load approval conditions:', error);
        } finally {
            this.approvalConditionsLoading = false;
        }
    },

    async loadPendingApprovals() {
        this.pendingApprovalsLoading = true;
        try {
            const response = await fetch('/api/approvals/requests?pending_only=true');
            if (response.ok) {
                const data = await response.json();
                this.pendingApprovals = data.requests || [];
            }
        } catch (error) {
            console.error('Failed to load pending approvals:', error);
        } finally {
            this.pendingApprovalsLoading = false;
        }
    },

    async approveRequest(requestId, notes = '') {
        try {
            const response = await fetch(`/api/approvals/requests/${requestId}/approve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ notes }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Request approved', 'success');
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to approve', 'error');
            }
        } catch (error) {
            console.error('Failed to approve request:', error);
            if (window.showToast) window.showToast('Failed to approve request', 'error');
        }
    },

    async rejectRequest(requestId, reason = '') {
        try {
            const response = await fetch(`/api/approvals/requests/${requestId}/reject`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ reason }),
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Request rejected', 'success');
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to reject', 'error');
            }
        } catch (error) {
            console.error('Failed to reject request:', error);
            if (window.showToast) window.showToast('Failed to reject request', 'error');
        }
    },

    toggleApprovalSelection(requestId) {
        const idx = this.selectedApprovalIds.indexOf(requestId);
        if (idx === -1) {
            this.selectedApprovalIds.push(requestId);
        } else {
            this.selectedApprovalIds.splice(idx, 1);
        }
    },

    toggleSelectAllApprovals() {
        if (this.selectedApprovalIds.length === this.pendingApprovals.length) {
            this.selectedApprovalIds = [];
        } else {
            this.selectedApprovalIds = this.pendingApprovals.map(r => r.id);
        }
    },

    async bulkApproveRequests() {
        if (this.selectedApprovalIds.length === 0) return;

        const notes = prompt('Optional notes for all approvals:') || '';
        this.bulkProcessing = true;

        try {
            const response = await fetch('/api/approvals/requests/bulk-approve', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_ids: this.selectedApprovalIds,
                    notes: notes,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(`Approved ${data.succeeded} of ${data.total} requests`, 'success');
                }
                this.selectedApprovalIds = [];
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Bulk approve failed', 'error');
            }
        } catch (error) {
            console.error('Failed to bulk approve:', error);
            if (window.showToast) window.showToast('Bulk approve failed', 'error');
        } finally {
            this.bulkProcessing = false;
        }
    },

    async bulkRejectRequests() {
        if (this.selectedApprovalIds.length === 0) return;

        const reason = prompt('Rejection reason (required):');
        if (!reason) {
            if (window.showToast) window.showToast('Rejection reason is required', 'warning');
            return;
        }

        this.bulkProcessing = true;

        try {
            const response = await fetch('/api/approvals/requests/bulk-reject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    request_ids: this.selectedApprovalIds,
                    reason: reason,
                }),
            });

            if (response.ok) {
                const data = await response.json();
                if (window.showToast) {
                    window.showToast(`Rejected ${data.succeeded} of ${data.total} requests`, 'success');
                }
                this.selectedApprovalIds = [];
                await this.loadPendingApprovals();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Bulk reject failed', 'error');
            }
        } catch (error) {
            console.error('Failed to bulk reject:', error);
            if (window.showToast) window.showToast('Bulk reject failed', 'error');
        } finally {
            this.bulkProcessing = false;
        }
    },

    showCreateApprovalRuleForm() {
        this.editingApprovalRule = null;
        this.approvalRuleForm = {
            name: '',
            trigger_type: 'cost_threshold',
            trigger_value: '',
            required_level: 'lead',
            priority: 0,
            enabled: true,
        };
        this.approvalRuleFormOpen = true;
    },

    showEditApprovalRuleForm(rule) {
        this.editingApprovalRule = rule;
        this.approvalRuleForm = {
            name: rule.name,
            trigger_type: rule.trigger_type,
            trigger_value: String(rule.trigger_value),
            required_level: rule.required_level,
            priority: rule.priority,
            enabled: rule.enabled,
        };
        this.approvalRuleFormOpen = true;
    },

    async saveApprovalRule() {
        this.saving = true;
        try {
            const payload = {
                ...this.approvalRuleForm,
                trigger_value: this.approvalRuleForm.trigger_type === 'cost_threshold'
                    ? parseFloat(this.approvalRuleForm.trigger_value)
                    : this.approvalRuleForm.trigger_value,
            };

            let response;
            if (this.editingApprovalRule) {
                response = await fetch(`/api/approvals/rules/${this.editingApprovalRule.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            } else {
                response = await fetch('/api/approvals/rules', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
            }

            if (response.ok) {
                if (window.showToast) {
                    window.showToast(this.editingApprovalRule ? 'Rule updated' : 'Rule created', 'success');
                }
                this.approvalRuleFormOpen = false;
                await this.loadApprovalRules();
            } else {
                const data = await response.json();
                if (window.showToast) window.showToast(data.detail || 'Failed to save rule', 'error');
            }
        } catch (error) {
            console.error('Failed to save approval rule:', error);
            if (window.showToast) window.showToast('Failed to save rule', 'error');
        } finally {
            this.saving = false;
        }
    },

    async deleteApprovalRule(rule) {
        if (!confirm(`Delete approval rule "${rule.name}"?`)) return;

        try {
            const response = await fetch(`/api/approvals/rules/${rule.id}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                if (window.showToast) window.showToast('Rule deleted', 'success');
                await this.loadApprovalRules();
            } else {
                if (window.showToast) window.showToast('Failed to delete rule', 'error');
            }
        } catch (error) {
            console.error('Failed to delete approval rule:', error);
            if (window.showToast) window.showToast('Failed to delete rule', 'error');
        }
    },

    getTriggerTypeName(triggerType) {
        // Use dynamic conditions if loaded
        if (this.approvalConditions && this.approvalConditions.length > 0) {
            const cond = this.approvalConditions.find(c => c.value === triggerType);
            if (cond) return cond.label || cond.description;
        }
        // Fallback to static names
        const names = {
            'cost_threshold': 'Cost > $',
            'hardware_type': 'Hardware',
            'first_job': 'First Job',
            'config_pattern': 'Config Pattern',
        };
        return names[triggerType] || triggerType;
    },
};
