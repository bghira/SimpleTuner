/**
 * First-Run Setup State Factory
 *
 * State for the initial admin setup wizard and login.
 */

window.cloudSetupStateFactory = function() {
    return {
        loading: true,
        needsSetup: false,
        needsLogin: false,  // True when auth is required but user not logged in
        hasAdmin: true,
        currentUser: null,  // Current logged-in user
        submitting: false,
        error: null,
        showPassword: false,
        // First-run setup form
        form: {
            email: '',
            username: '',
            displayName: '',
            password: '',
            confirmPassword: '',
        },
        // Login form
        loginForm: {
            username: '',
            password: '',
            rememberMe: false,
        },
    };
};
