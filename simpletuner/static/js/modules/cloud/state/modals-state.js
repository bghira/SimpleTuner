/**
 * Modals State Factory
 *
 * State for logs modal, approval modal, and admin modal.
 */

window.cloudModalsStateFactory = function() {
    return {
        logsModal: {
            open: false,
            jobId: null,
            logs: '',
            loading: false,
            searchQuery: '',
            levelFilter: '',
            autoScroll: true,
            streaming: false,
            filteredLogsHtml: '',
            matchCount: 0,
            lineCount: 0,
            eventSource: null,
        },
        approvalModal: {
            open: false,
            action: null,
            job: null,
            reason: '',
            notes: '',
            processing: false,
        },
        adminModal: {
            open: false,
            tab: 'users',
            users: [],
            filteredUsers: [],
            usersLoading: false,
            userSearch: '',
            levels: [],
            levelsLoading: false,
            rules: [],
            filteredRules: [],
            rulesLoading: false,
            ruleTypeFilter: '',
            ruleFormOpen: false,
            editingRule: null,
            ruleForm: {
                name: '',
                resource_type: 'config',
                pattern: '',
                action: 'allow',
                priority: 0,
                description: '',
            },
            levelRulesOpen: false,
            editingLevel: null,
            selectedLevelRules: [],
            userFormOpen: false,
            editingUser: null,
            userForm: {
                email: '',
                username: '',
                display_name: '',
                password: '',
                is_admin: false,
                is_active: true,
                level_names: ['researcher'],
            },
            deleteUserOpen: false,
            deletingUser: null,
            saving: false,
        },
    };
};
