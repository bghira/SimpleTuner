/**
 * User Profile Component
 *
 * Alpine.js component for managing user profile and avatar.
 */

window.userProfileComponent = function() {
    return {
        loading: true,
        saving: false,
        uploading: false,
        error: null,

        profile: {
            id: null,
            email: '',
            username: '',
            display_name: '',
            avatar_url: null,
            auth_provider: 'local',
            is_admin: false,
            created_at: '',
        },

        form: {
            display_name: '',
        },

        originalDisplayName: '',

        async init() {
            // Load profile when modal is shown
            const modal = document.getElementById('userProfileModal');
            if (modal) {
                modal.addEventListener('show.bs.modal', () => this.loadProfile());
            }
        },

        async loadProfile() {
            this.loading = true;
            this.error = null;

            try {
                const response = await fetch('/api/users/me/profile');
                if (response.ok) {
                    this.profile = await response.json();
                    this.form.display_name = this.profile.display_name || '';
                    this.originalDisplayName = this.form.display_name;
                } else {
                    const data = await response.json();
                    this.error = data.detail || 'Failed to load profile';
                }
            } catch (error) {
                console.error('Failed to load profile:', error);
                this.error = 'Failed to load profile';
            } finally {
                this.loading = false;
            }
        },

        hasChanges() {
            return this.form.display_name !== this.originalDisplayName;
        },

        async saveProfile() {
            if (!this.hasChanges()) return;

            this.saving = true;
            this.error = null;

            try {
                const response = await fetch('/api/users/me/profile', {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ display_name: this.form.display_name }),
                });

                if (response.ok) {
                    this.profile = await response.json();
                    this.originalDisplayName = this.form.display_name;
                    if (window.showToast) window.showToast('Profile updated', 'success');

                    // Update the topbar avatar display name
                    this.updateTopbarUserInfo();
                } else {
                    const data = await response.json();
                    this.error = data.detail || 'Failed to save profile';
                }
            } catch (error) {
                console.error('Failed to save profile:', error);
                this.error = 'Failed to save profile';
            } finally {
                this.saving = false;
            }
        },

        async uploadAvatar(event) {
            const file = event.target.files?.[0];
            if (!file) return;

            // Validate file type
            const validTypes = ['image/png', 'image/jpeg', 'image/gif', 'image/webp'];
            if (!validTypes.includes(file.type)) {
                this.error = 'Please upload a PNG, JPEG, GIF, or WebP image';
                return;
            }

            // Validate file size (512KB max)
            if (file.size > 512 * 1024) {
                this.error = 'Image must be smaller than 512KB';
                return;
            }

            this.uploading = true;
            this.error = null;

            try {
                // Convert to data URL
                const dataUrl = await this.fileToDataUrl(file);

                const response = await fetch('/api/users/me/avatar/upload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ data_url: dataUrl }),
                });

                if (response.ok) {
                    const data = await response.json();
                    this.profile.avatar_url = data.avatar_url + '?t=' + Date.now();
                    if (window.showToast) window.showToast('Avatar uploaded', 'success');

                    // Update the topbar avatar
                    this.updateTopbarUserInfo();
                } else {
                    const data = await response.json();
                    this.error = data.detail || 'Failed to upload avatar';
                }
            } catch (error) {
                console.error('Failed to upload avatar:', error);
                this.error = 'Failed to upload avatar';
            } finally {
                this.uploading = false;
                // Clear the file input
                event.target.value = '';
            }
        },

        async removeAvatar() {
            if (!confirm('Remove your avatar?')) return;

            this.saving = true;
            this.error = null;

            try {
                const response = await fetch('/api/users/me/avatar', {
                    method: 'DELETE',
                });

                if (response.ok) {
                    this.profile.avatar_url = null;
                    if (window.showToast) window.showToast('Avatar removed', 'success');

                    // Update the topbar avatar
                    this.updateTopbarUserInfo();
                } else {
                    const data = await response.json();
                    this.error = data.detail || 'Failed to remove avatar';
                }
            } catch (error) {
                console.error('Failed to remove avatar:', error);
                this.error = 'Failed to remove avatar';
            } finally {
                this.saving = false;
            }
        },

        fileToDataUrl(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        },

        getInitials() {
            const name = this.profile.display_name || this.profile.username || 'U';
            const parts = name.split(/\s+/);
            if (parts.length >= 2) {
                return (parts[0][0] + parts[1][0]).toUpperCase();
            }
            return name.substring(0, 2).toUpperCase();
        },

        formatDate(isoString) {
            return window.UIHelpers?.formatDate(isoString, { fallback: 'Unknown' }) || 'Unknown';
        },

        updateTopbarUserInfo() {
            // Dispatch event for topbar to update
            window.dispatchEvent(new CustomEvent('user-profile-updated', {
                detail: {
                    display_name: this.profile.display_name,
                    avatar_url: this.profile.avatar_url,
                    username: this.profile.username,
                },
            }));
        },
    };
};
