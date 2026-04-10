(function() {
    const prefixes = {
        library: 'user_prompt_library',
        librariesDir: 'validation_prompt_libraries',
    };
    const selectors = {
        modal: '#promptLibraryModal',
        rows: '#prompt-library-rows',
        nameInput: '#prompt-library-name',
        filenamePreview: '[data-filename-name]',
        saveButton: '#prompt-library-save',
        errorBlock: '#prompt-library-modal-error',
    };

    const state = {
        libraries: [],
        loadingList: false,
        modalElements: null,
        modalReady: false,
        previousFilename: null,
        targetInput: null,
        globalWidth: 1024,
        globalHeight: 1024,
        activeBboxPanel: null,
        bboxEnabled: false,
        isVideoModel: false,
        numVideoFrames: 25,
    };

    const showModal = () => {
        const elements = getModalElements();
        if (!elements) return;
        elements.modal.style.display = '';
        document.body.style.overflow = 'hidden';
    };

    const hideModal = () => {
        const elements = getModalElements();
        if (!elements) return;
        elements.modal.style.display = 'none';
        document.body.style.overflow = '';
        state.targetInput = null;
        state.previousFilename = null;
        clearError();
        setBboxEnabled(false);
        resetRows();
        if (elements.nameInput) {
            elements.nameInput.value = '';
        }
        updateFilenamePreview();
    };

    const getModalElements = () => {
        if (state.modalElements) {
            return state.modalElements;
        }
        const modal = document.querySelector(selectors.modal);
        if (!modal) {
            return null;
        }
        const elements = {
            modal,
            rowsContainer: modal.querySelector(selectors.rows),
            nameInput: modal.querySelector(selectors.nameInput),
            filenamePreview: modal.querySelector(selectors.filenamePreview),
            saveButton: modal.querySelector(selectors.saveButton),
            errorBlock: modal.querySelector(selectors.errorBlock),
            closeButton: modal.querySelector('#prompt-library-close'),
            cancelButton: modal.querySelector('#prompt-library-cancel'),
            bboxToggle: modal.querySelector('#prompt-library-bbox-toggle'),
            resolutionContainer: modal.querySelector('#prompt-library-resolution-container'),
            resWidthInput: modal.querySelector('#prompt-library-res-width'),
            resHeightInput: modal.querySelector('#prompt-library-res-height'),
            arLabel: modal.querySelector('#prompt-library-ar-label'),
            colHeaders: modal.querySelector('#prompt-library-col-headers'),
            videoFramesContainer: document.querySelector('#prompt-library-video-frames-container'),
            videoFramesInput: document.querySelector('#prompt-library-video-frames'),
        };
        state.modalElements = elements;
        return elements;
    };

    const updateFilenamePreview = () => {
        const elements = getModalElements();
        if (!elements) {
            return;
        }
        const value = elements.nameInput?.value.trim() || '';
        const filename = value ? `${prefixes.library}-${value}.json` : `${prefixes.library}.json`;
        if (elements.filenamePreview) {
            elements.filenamePreview.textContent = filename;
        }
    };

    const clearError = () => {
        const elements = getModalElements();
        if (!elements || !elements.errorBlock) {
            return;
        }
        elements.errorBlock.textContent = '';
        elements.errorBlock.classList.add('d-none');
    };

    const showError = (message) => {
        const elements = getModalElements();
        if (elements && elements.errorBlock) {
            elements.errorBlock.textContent = message;
            elements.errorBlock.classList.remove('d-none');
            return;
        }
        if (window.showToast) {
            window.showToast(message, 'error');
        }
    };

    const handleRowInput = (event) => {
        if (!event || !event.target) {
            return;
        }
        event.target.classList.remove('is-invalid');
        clearError();
        ensureTrailingRow();
    };

    // -- Bbox mode management --

    const setBboxEnabled = (enabled) => {
        state.bboxEnabled = enabled;
        const elements = getModalElements();
        if (!elements) return;

        if (elements.bboxToggle) {
            elements.bboxToggle.checked = enabled;
        }

        // Show/hide resolution inputs
        if (elements.resolutionContainer) {
            elements.resolutionContainer.classList.toggle('d-none', !enabled);
        }

        // Show/hide video frames input
        if (elements.videoFramesContainer) {
            elements.videoFramesContainer.classList.toggle('d-none', !(enabled && state.isVideoModel));
        }

        // Update column widths and show/hide bbox buttons on all rows
        updateRowLayout();
    };

    const updateRowLayout = () => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) return;

        // Update column headers
        if (elements.colHeaders) {
            const cols = elements.colHeaders.children;
            if (state.bboxEnabled) {
                cols[0]?.setAttribute('class', 'col-md-3');
                cols[1]?.setAttribute('class', 'col-md-7');
                cols[2]?.setAttribute('class', 'col-md-2 text-end');
            } else {
                cols[0]?.setAttribute('class', 'col-md-3');
                cols[1]?.setAttribute('class', 'col-md-8');
                cols[2]?.setAttribute('class', 'col-md-1 text-end');
            }
        }

        // Update each row
        elements.rowsContainer.querySelectorAll('.prompt-library-row').forEach((row) => {
            const shortCol = row.querySelector('.prompt-library-shortname')?.parentElement;
            const promptCol = row.querySelector('.prompt-library-prompt')?.parentElement;
            const actionCol = row.querySelector('.prompt-library-actions');
            const bboxBtn = row.querySelector('.prompt-library-bbox-btn');

            if (state.bboxEnabled) {
                shortCol?.setAttribute('class', 'col-md-3');
                promptCol?.setAttribute('class', 'col-md-7');
                actionCol?.setAttribute('class', 'col-md-2 d-flex align-items-start justify-content-end prompt-library-actions');
                if (bboxBtn) bboxBtn.classList.remove('d-none');
            } else {
                shortCol?.setAttribute('class', 'col-md-3');
                promptCol?.setAttribute('class', 'col-md-8');
                actionCol?.setAttribute('class', 'col-md-1 d-flex align-items-start justify-content-end prompt-library-actions');
                if (bboxBtn) bboxBtn.classList.add('d-none');
                // Close any open panel
                if (row._bboxPanel && !row._bboxPanel.classList.contains('d-none')) {
                    closeBboxPanel(row);
                }
            }
        });
    };

    const closeBboxPanel = (row) => {
        if (row._bboxPanel) {
            row._bboxPanel.classList.add('d-none');
        }
        if (state.activeBboxPanel === row) {
            state.activeBboxPanel = null;
        }
    };

    const toggleBboxPanel = (row, button) => {
        if (state.activeBboxPanel && state.activeBboxPanel !== row) {
            closeBboxPanel(state.activeBboxPanel);
        }

        if (row._bboxPanel) {
            if (row._bboxPanel.classList.contains('d-none')) {
                row._bboxPanel.classList.remove('d-none');
                state.activeBboxPanel = row;
            } else {
                closeBboxPanel(row);
            }
            return;
        }

        const panel = document.createElement('div');
        panel.className = 'prompt-library-bbox-panel border rounded p-3 mb-2 bg-body-tertiary';

        const header = document.createElement('div');
        header.className = 'd-flex justify-content-between align-items-center mb-2';
        const title = document.createElement('small');
        title.className = 'fw-semibold text-body-secondary text-uppercase';
        title.style.fontSize = '0.7rem';
        title.style.letterSpacing = '0.5px';
        title.textContent = 'Bounding Boxes';
        header.appendChild(title);
        const hint = document.createElement('small');
        hint.className = 'text-body-secondary';
        hint.style.fontSize = '0.75rem';
        hint.innerHTML = 'Draw to create &middot; Click to select &middot; Double-click to rename &middot; <kbd>Del</kbd> or <span style="color: #e53935; font-weight: bold;">&times;</span> to remove';
        header.appendChild(hint);
        panel.appendChild(header);

        const canvasContainer = document.createElement('div');
        canvasContainer.className = 'bbox-canvas-container';
        panel.appendChild(canvasContainer);

        const canvas = new BboxCanvas(canvasContainer, {
            width: state.globalWidth,
            height: state.globalHeight,
            onChange: (entities) => {
                if (!state.isVideoModel) {
                    row._bboxEntities = entities.length > 0 ? entities : null;
                }
                row._updateBboxBadge?.();
            },
        });

        if (row._bboxEntities) {
            canvas.setBoxes(row._bboxEntities);
        }

        row._bboxPanel = panel;
        row._bboxCanvas = canvas;

        // Add keyframe timeline for video models
        if (state.isVideoModel && typeof BboxKeyframeTimeline !== 'undefined') {
            const timelineContainer = document.createElement('div');
            timelineContainer.className = 'bbox-keyframe-timeline-container mt-2';
            panel.appendChild(timelineContainer);

            const timeline = new BboxKeyframeTimeline(timelineContainer, {
                canvas: canvas,
                numFrames: state.numVideoFrames,
                onChange: (keyframes) => {
                    row._bboxKeyframes = keyframes.length > 0 ? keyframes : null;
                    row._updateBboxBadge?.();
                },
            });

            if (row._bboxKeyframes && row._bboxKeyframes.length > 0) {
                timeline.setKeyframes(row._bboxKeyframes);
            }

            row._keyframeTimeline = timeline;
        }

        row.after(panel);
        state.activeBboxPanel = row;
    };

    const createRow = (entry = { shortname: '', prompt: '', bbox_entities: null, bbox_keyframes: null }) => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) {
            return;
        }
        const row = document.createElement('div');
        row.className = 'row g-2 align-items-start prompt-library-row';

        row._bboxEntities = entry.bbox_entities || null;
        row._bboxKeyframes = entry.bbox_keyframes || null;

        const createInput = (tag, classes, attrs = {}) => {
            const elm = document.createElement(tag);
            elm.className = classes;
            Object.entries(attrs).forEach(([key, value]) => elm.setAttribute(key, value));
            return elm;
        };

        const shortCol = document.createElement('div');
        shortCol.className = state.bboxEnabled ? 'col-md-3' : 'col-md-3';
        const shortInput = createInput('input', 'form-control form-control-sm prompt-library-shortname', {
            type: 'text',
            placeholder: 'Shortname',
            value: entry.shortname || '',
        });
        shortInput.addEventListener('input', handleRowInput);
        shortCol.appendChild(shortInput);

        const promptCol = document.createElement('div');
        promptCol.className = state.bboxEnabled ? 'col-md-7' : 'col-md-8';
        const promptInput = createInput('textarea', 'form-control form-control-sm prompt-library-prompt', {
            rows: '2',
            placeholder: 'Validation prompt',
        });
        promptInput.value = entry.prompt || '';
        promptInput.addEventListener('input', handleRowInput);
        promptCol.appendChild(promptInput);

        const actionCol = document.createElement('div');
        actionCol.className = (state.bboxEnabled ? 'col-md-2' : 'col-md-1') + ' d-flex align-items-start justify-content-end prompt-library-actions';

        // Bbox button (visible only when bbox mode is on)
        const bboxButton = document.createElement('button');
        bboxButton.type = 'button';
        bboxButton.className = 'btn btn-outline-secondary btn-sm prompt-library-bbox-btn' + (state.bboxEnabled ? '' : ' d-none');
        bboxButton.setAttribute('aria-label', 'Edit bounding boxes');
        bboxButton.addEventListener('click', () => toggleBboxPanel(row, bboxButton));

        const updateBboxBadge = () => {
            if (state.isVideoModel && row._bboxKeyframes && row._bboxKeyframes.length > 0) {
                bboxButton.textContent = 'Bounding boxes (' + row._bboxKeyframes.length + ' keyframes)';
            } else {
                const count = (row._bboxEntities || []).length;
                if (count > 0) {
                    bboxButton.textContent = 'Bounding boxes (' + count + ')';
                } else {
                    bboxButton.textContent = 'Bounding boxes';
                }
            }
        };
        row._updateBboxBadge = updateBboxBadge;
        updateBboxBadge();

        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'btn btn-link btn-sm text-danger px-1 py-0';
        removeButton.setAttribute('aria-label', 'Remove prompt entry');
        removeButton.innerHTML = '<i class="fas fa-times"></i>';
        removeButton.addEventListener('click', () => {
            if (row._keyframeTimeline) row._keyframeTimeline.destroy();
            if (row._bboxCanvas) row._bboxCanvas.destroy();
            if (row._bboxPanel) row._bboxPanel.remove();
            if (state.activeBboxPanel === row) state.activeBboxPanel = null;
            row.remove();
            if (!elements.rowsContainer?.querySelector('.prompt-library-row')) {
                createRow();
            }
            ensureTrailingRow();
        });

        actionCol.appendChild(bboxButton);
        actionCol.appendChild(removeButton);

        row.appendChild(shortCol);
        row.appendChild(promptCol);
        row.appendChild(actionCol);
        elements.rowsContainer.appendChild(row);
    };

    const ensureTrailingRow = () => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) {
            return;
        }
        const rows = Array.from(elements.rowsContainer.querySelectorAll('.prompt-library-row'));
        const lastRow = rows[rows.length - 1];
        if (!lastRow) {
            createRow();
            return;
        }
        const shortInput = lastRow.querySelector('.prompt-library-shortname');
        const promptInput = lastRow.querySelector('.prompt-library-prompt');
        const hasShort = shortInput && shortInput.value.trim();
        const hasPrompt = promptInput && promptInput.value.trim();
        if (hasShort && hasPrompt) {
            createRow();
        }
    };

    const resetRows = (seedEmptyRow = true) => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) {
            return;
        }
        elements.rowsContainer.querySelectorAll('.prompt-library-row').forEach((row) => {
            if (row._keyframeTimeline) row._keyframeTimeline.destroy();
            if (row._bboxCanvas) row._bboxCanvas.destroy();
            if (row._bboxPanel) row._bboxPanel.remove();
        });
        state.activeBboxPanel = null;
        elements.rowsContainer.innerHTML = '';
        if (seedEmptyRow) {
            createRow();
        }
    };

    const collectEntries = () => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) {
            throw { message: 'Prompt rows are not available.' };
        }
        const rows = Array.from(elements.rowsContainer.querySelectorAll('.prompt-library-row'));
        const seen = new Set();
        const entries = [];
        rows.forEach((row) => {
            const shortInput = row.querySelector('.prompt-library-shortname');
            const promptInput = row.querySelector('.prompt-library-prompt');
            const shortname = shortInput ? shortInput.value.trim() : '';
            const promptValue = promptInput ? promptInput.value : '';
            if (!shortname && !promptValue.trim()) {
                return;
            }
            if (!shortname) {
                shortInput?.classList.add('is-invalid');
                throw { message: 'Each prompt entry requires a shortname.', element: shortInput };
            }
            if (!promptValue.trim()) {
                promptInput?.classList.add('is-invalid');
                throw { message: 'Each prompt entry requires a validation prompt.', element: promptInput };
            }
            if (seen.has(shortname)) {
                shortInput?.classList.add('is-invalid');
                throw { message: 'Prompt IDs must be unique.', element: shortInput };
            }
            seen.add(shortname);
            entries.push({
                shortname,
                prompt: promptValue,
                bbox_entities: state.bboxEnabled ? (row._bboxEntities || null) : null,
                bbox_keyframes: (state.bboxEnabled && state.isVideoModel) ? (row._bboxKeyframes || null) : null,
            });
        });
        if (!entries.length) {
            throw { message: 'Add at least one prompt before saving the library.' };
        }
        return entries;
    };

    const fetchLibrary = async (filename) => {
        if (!filename) {
            throw new Error('Filename is required to load a prompt library.');
        }
        const response = await fetch(`/api/prompt-libraries/${encodeURIComponent(filename)}`);
        if (!response.ok) {
            const payload = await response.json().catch(() => ({}));
            throw new Error(payload.detail || 'Failed to load prompt library.');
        }
        return response.json();
    };

    const refreshList = async (force = false) => {
        if (state.loadingList && !force) {
            return state.libraries;
        }
        state.loadingList = true;
        try {
            const response = await fetch('/api/prompt-libraries');
            if (!response.ok) {
                throw new Error('Unable to fetch prompt libraries.');
            }
            const data = await response.json();
            const libs = Array.isArray(data.libraries) ? data.libraries : [];
            state.libraries = libs;
            if (typeof window !== 'undefined') {
                window.__promptLibraries = libs.slice();
                window.dispatchEvent(new CustomEvent('promptLibrariesRefreshed', { detail: { libraries: libs.slice() } }));
            }
            return state.libraries;
        } catch (error) {
            console.error('Failed to refresh prompt libraries:', error);
            return state.libraries;
        } finally {
            state.loadingList = false;
        }
    };

    const setTargetInputValue = (value) => {
        if (!state.targetInput) {
            return;
        }
        state.targetInput.value = value;
        state.targetInput.dispatchEvent(new Event('input', { bubbles: true }));
        state.targetInput.dispatchEvent(new Event('change', { bubbles: true }));
        if (window.Alpine && typeof Alpine.store === 'function') {
            const trainerStore = Alpine.store('trainer');
            if (trainerStore) {
                if (typeof trainerStore.markFormDirty === 'function') {
                    trainerStore.markFormDirty();
                } else {
                    trainerStore.formDirty = true;
                    trainerStore._skipNextClean = true;
                }
            }
        }
    };

    const _refreshVideoModelState = () => {
        const trainer = window.Alpine?.store?.('trainer');
        const ctx = trainer?.modelContext || {};
        state.isVideoModel = Boolean(ctx.isVideoModel || ctx.supportsVideo);
    };

    const openModal = async (options = {}) => {
        const elements = getModalElements();
        if (!elements) {
            return;
        }
        if (!state.modalReady) {
            initModal();
        }
        clearError();
        state.targetInput = options.targetInput || null;
        state.previousFilename = options.filename || null;

        // Re-check video model status from current Alpine store
        _refreshVideoModelState();

        // Reset bbox mode
        setBboxEnabled(false);
        resetRows();

        if (options.filename) {
            try {
                const payload = await fetchLibrary(options.filename);
                const entries = payload.entries && typeof payload.entries === 'object'
                    ? Object.entries(payload.entries)
                    : [];
                resetRows(false);

                // Check if any entry has bbox_entities or bbox_keyframes to auto-enable bbox mode
                let hasBbox = false;
                entries.forEach(([shortname, value]) => {
                    if (typeof value === 'string') {
                        createRow({ shortname, prompt: value, bbox_entities: null, bbox_keyframes: null });
                    } else if (typeof value === 'object' && value !== null) {
                        if ((value.bbox_entities && value.bbox_entities.length > 0) ||
                            (value.bbox_keyframes && value.bbox_keyframes.length > 0)) {
                            hasBbox = true;
                        }
                        createRow({
                            shortname,
                            prompt: value.prompt || '',
                            bbox_entities: value.bbox_entities || null,
                            bbox_keyframes: value.bbox_keyframes || null,
                        });
                    }
                });

                if (hasBbox) {
                    setBboxEnabled(true);
                }

                ensureTrailingRow();
                elements.nameInput.value = extractLibraryName(options.filename);
                state.previousFilename = payload.library?.filename || options.filename;
            } catch (error) {
                showError(error.message || 'Unable to load the selected prompt library.');
                elements.nameInput.value = extractLibraryName(options.filename);
            }
        } else {
            elements.nameInput.value = '';
            state.previousFilename = null;
            ensureTrailingRow();
        }
        updateFilenamePreview();
        showModal();
    };

    const extractLibraryName = (filename) => {
        if (!filename) {
            return '';
        }
        const match = filename.match(/^user_prompt_library(?:-([A-Za-z0-9._-]+))?\.json$/i);
        return match && match[1] ? match[1] : '';
    };

    const dispatchLibrariesSaved = () => {
        window.dispatchEvent(new CustomEvent('promptLibrarySaved', { detail: { libraries: state.libraries.slice() } }));
    };

    const saveLibrary = async () => {
        const elements = getModalElements();
        if (!elements || !elements.saveButton) {
            return;
        }
        clearError();
        let entries;
        try {
            entries = collectEntries();
        } catch (error) {
            if (error?.element) {
                error.element.focus();
            }
            showError(error?.message || 'Invalid entries.');
            return;
        }
        const libraryName = elements.nameInput.value.trim();
        if (!libraryName) {
            elements.nameInput.classList.add('is-invalid');
            elements.nameInput.focus();
            showError('A library name is required.');
            return;
        }
        if (!/^[A-Za-z0-9._-]+$/.test(libraryName)) {
            elements.nameInput.classList.add('is-invalid');
            showError('Library name may only contain letters, numbers, ".", "_", or "-".');
            return;
        }
        elements.nameInput.classList.remove('is-invalid');
        const filename = `${prefixes.library}-${libraryName}.json`;
        const payload = {
            entries: entries.reduce((acc, entry) => {
                const hasBbox = entry.bbox_entities && entry.bbox_entities.length > 0;
                const hasKeyframes = entry.bbox_keyframes && entry.bbox_keyframes.length > 0;
                if (hasBbox || hasKeyframes) {
                    const obj = { prompt: entry.prompt };
                    if (hasBbox) obj.bbox_entities = entry.bbox_entities;
                    if (hasKeyframes) obj.bbox_keyframes = entry.bbox_keyframes;
                    acc[entry.shortname] = obj;
                } else {
                    acc[entry.shortname] = entry.prompt;
                }
                return acc;
            }, {}),
            previous_filename: state.previousFilename || undefined,
        };
        const originalText = elements.saveButton.textContent;
        elements.saveButton.disabled = true;
        elements.saveButton.textContent = 'Saving...';
        try {
            const response = await fetch(`/api/prompt-libraries/${encodeURIComponent(filename)}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(data.detail || data.message || 'Failed to save prompt library.');
            }
            state.previousFilename = data.library?.filename || filename;
            const targetPath = data.library?.absolute_path || data.library?.relative_path;
            if (targetPath) {
                setTargetInputValue(targetPath);
            }
            await refreshList(true);
            dispatchLibrariesSaved();
            if (window.showToast) {
                window.showToast('Prompt library saved', 'success');
            }
            hideModal();
        } catch (error) {
            showError(error.message || 'Failed to save prompt library.');
        } finally {
            elements.saveButton.disabled = false;
            elements.saveButton.textContent = originalText;
        }
    };

    const initModal = () => {
        const elements = getModalElements();
        if (!elements || state.modalReady) {
            return;
        }
        elements.saveButton?.addEventListener('click', saveLibrary);
        elements.closeButton?.addEventListener('click', hideModal);
        elements.cancelButton?.addEventListener('click', hideModal);
        elements.modal.addEventListener('click', (e) => {
            if (e.target === elements.modal) hideModal();
        });
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && elements.modal.style.display !== 'none') {
                hideModal();
            }
        });
        elements.nameInput?.addEventListener('input', () => {
            elements.nameInput.classList.remove('is-invalid');
            updateFilenamePreview();
        });

        // Bbox toggle
        elements.bboxToggle?.addEventListener('change', () => {
            setBboxEnabled(elements.bboxToggle.checked);
        });

        // Resolution inputs
        const onResChange = () => {
            const w = parseInt(elements.resWidthInput?.value) || 1024;
            const h = parseInt(elements.resHeightInput?.value) || 1024;
            state.globalWidth = w;
            state.globalHeight = h;
            const gcd = (a, b) => b ? gcd(b, a % b) : a;
            const d = gcd(w, h);
            if (elements.arLabel) {
                elements.arLabel.textContent = `${w / d}:${h / d}`;
            }
            if (state.activeBboxPanel?._bboxCanvas) {
                state.activeBboxPanel._bboxCanvas.setResolution(w, h);
            }
        };
        elements.resWidthInput?.addEventListener('input', onResChange);
        elements.resHeightInput?.addEventListener('input', onResChange);

        // Video model detection: listen for context updates
        window.addEventListener('trainer-model-context-updated', (event) => {
            const ctx = event.detail || {};
            state.isVideoModel = Boolean(ctx.isVideoModel || ctx.supportsVideo);
        });

        // Video frames input
        if (elements.videoFramesInput) {
            elements.videoFramesInput.addEventListener('input', () => {
                const n = parseInt(elements.videoFramesInput.value, 10) || 25;
                state.numVideoFrames = Math.max(1, n);
                // Update all active keyframe timelines
                elements.rowsContainer?.querySelectorAll('.prompt-library-row').forEach((row) => {
                    if (row._keyframeTimeline) {
                        row._keyframeTimeline.setNumFrames(state.numVideoFrames);
                    }
                });
            });
        }

        state.modalReady = true;
    };

    const updateButtonLabel = (button, input) => {
        if (!button) {
            return;
        }
        const label = button.querySelector('[data-prompt-library-button-label]');
        if (!label || !input) {
            return;
        }
        const hasValue = Boolean(input.value && input.value.trim());
        label.textContent = hasValue ? 'Edit prompt library' : 'Create prompt library';
    };

    const attachTrigger = (button) => {
        if (!button || button.dataset.promptLibraryInit) {
            return;
        }
        const targetId = button.dataset.promptLibraryTarget;
        if (!targetId) {
            return;
        }
        const input = document.getElementById(targetId);
        if (!input) {
            return;
        }
        button.addEventListener('click', () => {
            const value = input.value.trim();
            const segments = value ? value.split(/[/\\]+/) : [];
            const filename = segments.length ? segments[segments.length - 1] : undefined;
            openModal({ targetInput: input, filename });
        });
        input.addEventListener('input', () => updateButtonLabel(button, input));
        updateButtonLabel(button, input);
        button.dataset.promptLibraryInit = 'true';
    };

    const initFieldTriggers = (root) => {
        const scope = root || document;
        const buttons = scope.querySelectorAll('[data-prompt-library-trigger]');
        buttons.forEach((button) => attachTrigger(button));
    };

    const manager = {
        openModal,
        refreshList,
        getLibraries: () => state.libraries,
        async init() {
            initModal();
            initFieldTriggers();
            // Wait for auth before making any API calls
            const canProceed = await window.waitForAuthReady();
            if (!canProceed) {
                // User needs to login - skip API-dependent initialization
                return;
            }
            refreshList().catch((error) => console.debug('Prompt library preload failed:', error));
        },
        initTriggers: (root) => initFieldTriggers(root),
    };

    document.addEventListener('DOMContentLoaded', () => manager.init());
    document.addEventListener('htmx:load', (event) => manager.initTriggers(event.detail?.elt || event.target));
    document.addEventListener('htmx:afterSwap', () => manager.initTriggers());
    window.promptLibraryManager = manager;
})();
