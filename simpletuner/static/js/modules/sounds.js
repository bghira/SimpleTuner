/**
 * UI Sound effects module using freedesktop sound theme.
 * Provides audio feedback for notifications and UI events.
 */

const SoundManager = (function() {
    'use strict';

    const SOUND_FILES = {
        success: '/api/sounds/completion-success.oga',
        error: '/api/sounds/dialog-error.oga',
        warning: '/api/sounds/dialog-warning.oga',
        info: '/api/sounds/dialog-information.oga',
        retroHover: '/api/sounds/retro-hover.wav',
    };

    let settings = {
        enabled: true,
        volume: 0.5,
        successEnabled: true,
        errorEnabled: true,
        warningEnabled: true,
        infoEnabled: true,
        retroHoverEnabled: false,
    };

    let hoverListenersAttached = false;

    const audioCache = {};
    let initialized = false;

    function init(initialSettings) {
        if (initialSettings) {
            updateSettings(initialSettings);
        }
        preloadSounds();
        initialized = true;
    }

    function updateSettings(newSettings) {
        if (newSettings.sounds_enabled !== undefined) {
            settings.enabled = Boolean(newSettings.sounds_enabled);
        }
        if (newSettings.sounds_volume !== undefined) {
            const vol = parseInt(newSettings.sounds_volume, 10);
            settings.volume = Number.isFinite(vol) ? Math.max(0, Math.min(100, vol)) / 100 : 0.5;
        }
        if (newSettings.sounds_success_enabled !== undefined) {
            settings.successEnabled = Boolean(newSettings.sounds_success_enabled);
        }
        if (newSettings.sounds_error_enabled !== undefined) {
            settings.errorEnabled = Boolean(newSettings.sounds_error_enabled);
        }
        if (newSettings.sounds_warning_enabled !== undefined) {
            settings.warningEnabled = Boolean(newSettings.sounds_warning_enabled);
        }
        if (newSettings.sounds_info_enabled !== undefined) {
            settings.infoEnabled = Boolean(newSettings.sounds_info_enabled);
        }
        if (newSettings.sounds_retro_hover_enabled !== undefined) {
            settings.retroHoverEnabled = Boolean(newSettings.sounds_retro_hover_enabled);
            if (settings.retroHoverEnabled) {
                attachNavHoverListeners();
            }
        }
    }

    function preloadSounds() {
        Object.entries(SOUND_FILES).forEach(([type, src]) => {
            if (!audioCache[type]) {
                const audio = new Audio();
                audio.preload = 'auto';
                audio.src = src;
                audioCache[type] = audio;
            }
        });
    }

    function isCategoryEnabled(type) {
        switch (type) {
            case 'success':
                return settings.successEnabled;
            case 'error':
                return settings.errorEnabled;
            case 'warning':
                return settings.warningEnabled;
            case 'info':
                return settings.infoEnabled;
            default:
                return true;
        }
    }

    function play(type) {
        if (!settings.enabled) {
            return;
        }

        if (!isCategoryEnabled(type)) {
            return;
        }

        const src = SOUND_FILES[type];
        if (!src) {
            return;
        }

        try {
            // Create a new audio instance each time to allow overlapping sounds
            const audio = new Audio(src);
            audio.volume = settings.volume;
            audio.play().catch(err => {
                // Autoplay may be blocked until user interaction
                console.debug('Sound playback blocked:', err.message);
            });
        } catch (err) {
            console.debug('Sound playback error:', err.message);
        }
    }

    function playSuccess() {
        play('success');
    }

    function playError() {
        play('error');
    }

    function playWarning() {
        play('warning');
    }

    function playInfo() {
        play('info');
    }

    function setEnabled(enabled) {
        settings.enabled = Boolean(enabled);
    }

    function setVolume(volume) {
        const vol = parseInt(volume, 10);
        settings.volume = Number.isFinite(vol) ? Math.max(0, Math.min(100, vol)) / 100 : 0.5;
    }

    function setCategoryEnabled(category, enabled) {
        const key = category + 'Enabled';
        if (key in settings) {
            settings[key] = Boolean(enabled);
        }
    }

    function getSettings() {
        return { ...settings };
    }

    function testSound(type) {
        const src = SOUND_FILES[type || 'info'];
        if (!src) {
            return;
        }
        try {
            const audio = new Audio(src);
            audio.volume = settings.volume;
            audio.play().catch(err => {
                console.debug('Test sound blocked:', err.message);
            });
        } catch (err) {
            console.debug('Test sound error:', err.message);
        }
    }

    function playHover() {
        if (!settings.enabled || !settings.retroHoverEnabled) {
            return;
        }
        try {
            const audio = new Audio(SOUND_FILES.retroHover);
            audio.volume = settings.volume * 0.6; // slightly quieter for hover
            audio.play().catch(() => {});
        } catch (err) {
            // silently ignore
        }
    }

    function attachNavHoverListeners() {
        if (hoverListenersAttached) {
            return;
        }
        hoverListenersAttached = true;

        // Use event delegation with mouseenter simulation
        // We track which elements we've already played for to avoid duplicates
        let lastHoveredElement = null;

        document.addEventListener('mouseover', function(e) {
            if (!settings.enabled || !settings.retroHoverEnabled) {
                return;
            }
            // Only target tab buttons in the nav
            const tabBtn = e.target.closest('.tab-btn');
            if (tabBtn && tabBtn !== lastHoveredElement) {
                lastHoveredElement = tabBtn;
                playHover();
            }
        });

        document.addEventListener('mouseout', function(e) {
            const tabBtn = e.target.closest('.tab-btn');
            if (tabBtn && tabBtn === lastHoveredElement) {
                // Check if we're leaving to a non-tab-btn element
                const relatedTarget = e.relatedTarget;
                if (!relatedTarget || !relatedTarget.closest('.tab-btn')) {
                    lastHoveredElement = null;
                }
            }
        });
    }

    function setRetroHoverEnabled(enabled) {
        settings.retroHoverEnabled = Boolean(enabled);
        if (settings.retroHoverEnabled) {
            attachNavHoverListeners();
        }
    }

    return {
        init,
        updateSettings,
        play,
        playSuccess,
        playError,
        playWarning,
        playInfo,
        playHover,
        setEnabled,
        setVolume,
        setCategoryEnabled,
        setRetroHoverEnabled,
        getSettings,
        testSound,
    };
})();

// Export for module systems and attach to window for global access
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SoundManager;
}
window.SoundManager = SoundManager;
