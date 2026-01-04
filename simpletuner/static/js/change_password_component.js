/**
 * Change Password Component
 *
 * Alpine.js component for changing user password.
 */

window.changePasswordComponent = function() {
    return {
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
        saving: false,
        error: '',
        success: false,

        init() {
            const modal = document.getElementById('changePasswordModal');
            if (modal) {
                modal.addEventListener('show.bs.modal', () => this.reset());
                modal.addEventListener('hidden.bs.modal', () => this.reset());
            }
        },

        reset() {
            this.currentPassword = '';
            this.newPassword = '';
            this.confirmPassword = '';
            this.saving = false;
            this.error = '';
            this.success = false;
        },

        validate() {
            if (!this.currentPassword) {
                this.error = 'Current password is required';
                return false;
            }
            if (!this.newPassword) {
                this.error = 'New password is required';
                return false;
            }
            if (this.newPassword.length < 8) {
                this.error = 'New password must be at least 8 characters';
                return false;
            }
            if (this.newPassword !== this.confirmPassword) {
                this.error = 'Passwords do not match';
                return false;
            }
            if (this.currentPassword === this.newPassword) {
                this.error = 'New password must be different from current password';
                return false;
            }
            return true;
        },

        async changePassword() {
            this.error = '';
            this.success = false;

            if (!this.validate()) {
                return;
            }

            this.saving = true;

            try {
                const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content;
                const response = await fetch('/api/users/me/password', {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        ...(csrfToken && { 'X-CSRF-Token': csrfToken })
                    },
                    credentials: 'include',
                    body: JSON.stringify({
                        current_password: this.currentPassword,
                        new_password: this.newPassword
                    })
                });

                if (!response.ok) {
                    const data = await response.json();
                    throw new Error(data.detail || 'Failed to change password');
                }

                this.success = true;
                this.currentPassword = '';
                this.newPassword = '';
                this.confirmPassword = '';

                if (window.showToast) {
                    window.showToast('Password changed successfully', 'success');
                }

                // Close modal after short delay
                setTimeout(() => {
                    const modal = bootstrap.Modal.getInstance(document.getElementById('changePasswordModal'));
                    if (modal) modal.hide();
                }, 1500);

            } catch (err) {
                this.error = err.message;
            } finally {
                this.saving = false;
            }
        }
    };
};
