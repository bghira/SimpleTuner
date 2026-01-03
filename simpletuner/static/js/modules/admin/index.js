/**
 * Admin Panel - Main Component
 *
 * Combines all admin module methods into a single Alpine.js component.
 * Individual modules are loaded before this file and attach methods
 * to window.admin*Methods objects.
 */

window.adminPanelComponent = function() {
    // State definition
    const state = {
        // Current user info
        currentUser: null,

        // Active tab
        activeTab: 'users',

        // Hints (dismissable guidance)
        hints: {
            overview: true,
            users: true,
            levels: true,
            rules: true,
            quotas: true,
            approvals: true,
            auth: true,
            orgs: true,
            notifications: true,
            registration: true,
        },
        hintsLoading: false,

        // System stats
        stats: {
            userCount: 0,
            activeUserCount: 0,
            externalUserCount: 0,
            levelCount: 0,
            ruleCount: 0,
            quotaCount: 0,
            approvalRuleCount: 0,
            pendingApprovals: 0,
            authProvidersConfigured: 0,
            orgCount: 0,
        },

        // Users management
        users: [],
        filteredUsers: [],
        usersLoading: false,
        userSearch: '',

        // Levels management
        levels: [],
        levelsLoading: false,
        allPermissions: [],

        // Resource rules management
        rules: [],
        filteredRules: [],
        rulesLoading: false,
        ruleTypeFilter: '',

        // Quotas management
        quotas: [],
        quotasLoading: false,
        quotaTargetFilter: '',
        // User quota status lookup
        quotaLookupUserId: '',
        quotaLookupLoading: false,
        quotaLookupResult: null,
        quotaLookupDetailOpen: false,

        // Approval rules management
        approvalRules: [],
        approvalRulesLoading: false,
        pendingApprovals: [],
        pendingApprovalsLoading: false,
        approvalConditions: [],
        approvalConditionsLoading: false,
        // Bulk approval selection
        selectedApprovalIds: [],
        bulkProcessing: false,

        // External auth providers
        authProviders: [],
        authProvidersLoading: false,

        // Organizations and Teams
        orgs: [],
        filteredOrgs: [],
        orgsLoading: false,
        orgSearch: '',
        selectedOrg: null,
        teams: [],
        teamsLoading: false,

        // Form states
        userFormOpen: false,
        editingUser: null,
        userForm: {
            email: '',
            username: '',
            display_name: '',
            password: '',
            is_admin: false,
            is_active: true,
            level_names: [],
        },

        levelFormOpen: false,
        editingLevel: null,
        levelForm: {
            name: '',
            description: '',
            priority: 0,
            permission_names: [],
        },

        ruleFormOpen: false,
        editingRule: null,
        ruleForm: {
            name: '',
            resource_type: 'config',
            action: 'allow',
            pattern: '',
            priority: 0,
            description: '',
        },

        quotaFormOpen: false,
        editingQuota: null,
        quotaForm: {
            quota_type: 'COST_MONTHLY',
            target_type: 'global',
            target_id: null,
            limit_value: 0,
            action: 'WARN',
        },

        approvalRuleFormOpen: false,
        editingApprovalRule: null,
        approvalRuleForm: {
            name: '',
            trigger_type: 'cost_threshold',
            trigger_value: '',
            required_level: 'lead',
            priority: 0,
            enabled: true,
        },

        authProviderFormOpen: false,
        editingAuthProvider: null,
        authProviderForm: {
            provider_type: 'oidc',
            name: '',
            enabled: true,
            config: {},
        },

        // Organization form
        orgFormOpen: false,
        editingOrg: null,
        orgForm: {
            name: '',
            slug: '',
            description: '',
            is_active: true,
        },

        // Team form
        teamFormOpen: false,
        editingTeam: null,
        teamForm: {
            name: '',
            slug: '',
            description: '',
            is_active: true,
        },

        // Entity quota management (for orgs/teams)
        entityQuotaOpen: false,
        quotaTargetType: 'organization',
        quotaTargetId: null,
        quotaTargetName: '',
        entityQuotas: [],
        quotaTypes: [],  // Loaded from API
        quotaActions: [], // Loaded from API
        newEntityQuota: {
            quota_type: 'concurrent_jobs',
            limit_value: 10,
            action: 'block',
        },

        // Level rules/permissions assignment
        levelRulesOpen: false,
        levelPermissionsOpen: false,
        selectedLevelRules: [],
        selectedPermissions: [],

        // Delete confirmation
        deleteUserOpen: false,
        deletingUser: null,

        // Audit log state
        audit: {
            entries: [],
            loading: false,
            loaded: false,
            stats: { total_entries: 0, entries_24h: 0, failed_logins_24h: 0, permission_denials_24h: 0 },
            eventTypes: [],
            // Filters
            eventTypeFilter: '',
            actorSearch: '',
            sinceDate: '',
            untilDate: '',
            securityOnly: false,
            // Pagination
            limit: 50,
            offset: 0,
            hasMore: false,
            // Chain verification
            verifyResult: null,
            verifying: false,
        },

        // Credential management state
        credentialsModalOpen: false,
        credentialsLoading: false,
        credentialsUser: null,
        userCredentials: [],
        rotateConfirmOpen: false,
        rotatingCredential: null,
        rotating: false,
        bulkStaleCheckLoading: false,
        staleCredentialsThreshold: 90,
        credentialEarlyWarningEnabled: false,
        credentialEarlyWarningPercent: 75,
        // Credential security settings (for Auth tab UI)
        credentialSettings: {
            stale_threshold_days: 90,
            early_warning_enabled: false,
            early_warning_percent: 75,
        },
        savingCredentialSettings: false,

        // Registration settings state
        registrationLoading: false,
        registrationSettings: {
            enabled: false,
            default_level: 'researcher',
            email_configured: false,
            email_required_warning: null,
        },

        // Permission overrides state
        permissionOverridesModalOpen: false,
        permissionOverridesLoading: false,
        permissionOverridesUser: null,
        userPermissionOverrides: [],
        newOverride: {
            permission: '',
            granted: true,
        },

        // General state
        saving: false,
        error: null,

        // Enterprise onboarding state (progressive disclosure CTA)
        enterpriseOnboarding: {
            auth_configured: false,
            auth_skipped: false,
            org_created: false,
            org_skipped: false,
            team_created: false,
            team_skipped: false,
            quotas_configured: false,
            quotas_skipped: false,
            credentials_configured: false,
            credentials_skipped: false,
            all_skipped: false,
            created_org_id: null,
            created_org_name: '',
        },

        // Quick forms for enterprise onboarding CTA
        quickOrgForm: {
            name: '',
            slug: '',
            saving: false,
            error: null,
        },
        quickTeamForm: {
            name: '',
            slug: '',
            saving: false,
            error: null,
        },
        quickQuotaForm: {
            monthly_cost_limit: 0,
            concurrent_jobs: 0,
            saving: false,
            error: null,
        },
        quickCredentialForm: {
            stale_days: 90,
            early_warning: false,
            warning_percent: 75,
            saving: false,
            error: null,
        },
        quickAuthForm: {
            provider_type: '',  // 'oidc' or 'ldap'
            name: '',
            saving: false,
            error: null,
            oidc: {
                issuer_url: '',
                client_id: '',
                client_secret: '',
                scopes: 'openid email profile',
                auto_provision: true,
                default_level: 'researcher',
            },
            ldap: {
                server_url: '',
                bind_dn: '',
                bind_password: '',
                base_dn: '',
                user_filter: '(uid={username})',
                auto_provision: true,
                default_level: 'researcher',
            },
        },

        // Notifications state
        notifications: {
            channels: [],
            presets: [],
            preferences: [],
            history: [],
            status: null,
            loading: false,
            loaded: false,
            testing: null,
            testResult: null,
            skipped: false,
        },

        // Channel form for create/edit
        channelFormOpen: false,
        editingChannel: null,
        channelForm: {
            channel_type: 'email',
            name: '',
            is_enabled: true,
            // Email-specific
            smtp_host: '',
            smtp_port: 587,
            smtp_username: '',
            smtp_password: '',
            smtp_use_tls: true,
            smtp_from_address: '',
            // IMAP for response handling
            imap_enabled: false,
            imap_host: '',
            imap_port: 993,
            imap_username: '',
            imap_password: '',
            imap_use_ssl: true,
            // Webhook-specific
            webhook_url: '',
            webhook_secret: '',
            // Slack-specific
            slack_webhook_url: '',
        },

        // Notification preferences (event routing)
        eventTypes: [],
        preferenceFormOpen: false,
        editingPreference: null,
        preferenceForm: {
            event_type: '',
            channel_id: null,
            is_enabled: true,
            recipients: '',
            min_severity: 'info',
        },

        // Team members management
        teamMembersOpen: false,
        selectedTeam: null,
        teamMembers: [],
        teamMembersLoading: false,
        availableTeamUsers: [],
        addMemberUserId: null,
    };

    // Core methods that orchestrate initialization
    const coreMethods = {
        async init() {
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                return;
            }

            this.loadCurrentUser();
            this.loadHints();
            this.loadEnterpriseOnboardingState();
            this.loadCredentialSecuritySettings();
            this.loadRegistrationSettings();
            this.loadUsers();
            this.loadLevels();
            this.loadResourceRules();
            this.loadQuotas();
            this.loadApprovalRules();
            this.loadApprovalConditions();
            this.loadPendingApprovals();
            this.loadAuthProviders();
            this.loadPermissions();
            this.loadOrganizations();
        },

        setActiveTab(tabName) {
            this.activeTab = tabName;
            // Lazy-load audit entries when tab is selected
            if (tabName === 'audit') {
                this.loadAuditEntriesIfNeeded();
            }
            // Lazy-load notifications when tab is selected
            if (tabName === 'notifications') {
                this.loadNotificationsIfNeeded();
            }
            // Refresh registration settings when tab is selected
            if (tabName === 'registration') {
                this.loadRegistrationSettings();
            }
        },

        formatDate(isoString) {
            return window.UIHelpers?.formatDate(isoString) || '';
        },
    };

    // Combine state with all module methods
    return {
        ...state,
        ...coreMethods,
        ...(window.adminHintMethods || {}),
        ...(window.adminUserMethods || {}),
        ...(window.adminLevelMethods || {}),
        ...(window.adminRuleMethods || {}),
        ...(window.adminQuotaMethods || {}),
        ...(window.adminApprovalMethods || {}),
        ...(window.adminAuthProviderMethods || {}),
        ...(window.adminOrgTeamMethods || {}),
        ...(window.adminAuditMethods || {}),
        ...(window.adminCredentialMethods || {}),
        ...(window.adminEnterpriseOnboardingMethods || {}),
        ...(window.adminNotificationMethods || {}),
        ...(window.adminRegistrationMethods || {}),

        // Getters must be defined here, not in spread modules, to avoid evaluation issues
        get anyHintsDismissed() {
            return this.hints && Object.values(this.hints).some(v => !v);
        },
    };
};
