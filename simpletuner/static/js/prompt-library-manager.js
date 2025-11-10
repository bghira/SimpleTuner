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
        modalInstance: null,
        modalReady: false,
        previousFilename: null,
        targetInput: null,
    };

    const attachModalToBody = (modal) => {
        if (modal && modal.parentElement !== document.body) {
            document.body.appendChild(modal);
        }
    };

    const getModalElements = () => {
        if (state.modalElements) {
            return state.modalElements;
        }
        const modal = document.querySelector(selectors.modal);
        if (!modal) {
            return null;
        }
        attachModalToBody(modal);
        const elements = {
            modal,
            rowsContainer: modal.querySelector(selectors.rows),
            nameInput: modal.querySelector(selectors.nameInput),
            filenamePreview: modal.querySelector(selectors.filenamePreview),
            saveButton: modal.querySelector(selectors.saveButton),
            errorBlock: modal.querySelector(selectors.errorBlock),
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

    const createRow = (entry = { shortname: '', prompt: '' }) => {
        const elements = getModalElements();
        if (!elements || !elements.rowsContainer) {
            return;
        }
        const row = document.createElement('div');
        row.className = 'row g-2 align-items-start prompt-library-row';

        const createInput = (tag, classes, attrs = {}) => {
            const elm = document.createElement(tag);
            elm.className = classes;
            Object.entries(attrs).forEach(([key, value]) => elm.setAttribute(key, value));
            return elm;
        };

        const shortCol = document.createElement('div');
        shortCol.className = 'col-md-3';
        const shortInput = createInput('input', 'form-control form-control-sm prompt-library-shortname', {
            type: 'text',
            placeholder: 'Shortname',
            value: entry.shortname || '',
        });
        shortInput.addEventListener('input', handleRowInput);
        shortCol.appendChild(shortInput);

        const promptCol = document.createElement('div');
        promptCol.className = 'col-md-8';
        const promptInput = createInput('textarea', 'form-control form-control-sm prompt-library-prompt', {
            rows: '2',
            placeholder: 'Validation prompt',
        });
        promptInput.value = entry.prompt || '';
        promptInput.addEventListener('input', handleRowInput);
        promptCol.appendChild(promptInput);

        const actionCol = document.createElement('div');
        actionCol.className = 'col-md-1 d-flex align-items-start justify-content-end';
        const removeButton = document.createElement('button');
        removeButton.type = 'button';
        removeButton.className = 'btn btn-link btn-sm text-danger px-1 py-0';
        removeButton.setAttribute('aria-label', 'Remove prompt entry');
        removeButton.innerHTML = '<i class="fas fa-times"></i>';
        removeButton.addEventListener('click', () => {
            row.remove();
            if (!elements.rowsContainer?.querySelector('.prompt-library-row')) {
                createRow();
            }
            ensureTrailingRow();
        });
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
            entries.push({ shortname, prompt: promptValue });
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
        resetRows();
        if (options.filename) {
            try {
                const payload = await fetchLibrary(options.filename);
                const entries = payload.entries && typeof payload.entries === 'object'
                    ? Object.entries(payload.entries)
                    : [];
                resetRows(false);
                entries.forEach(([shortname, prompt]) => {
                    createRow({ shortname, prompt });
                });
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
        state.modalInstance?.show();
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
        if (libraryName && !/^[A-Za-z0-9._-]+$/.test(libraryName)) {
            elements.nameInput.classList.add('is-invalid');
            showError('Library name may only contain letters, numbers, ".", "_", or "-".');
            return;
        }
        elements.nameInput.classList.remove('is-invalid');
        const filename = libraryName ? `${prefixes.library}-${libraryName}.json` : `${prefixes.library}.json`;
        const payload = {
            entries: entries.reduce((acc, entry) => ({ ...acc, [entry.shortname]: entry.prompt }), {}),
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
            state.modalInstance?.hide();
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
        if (typeof bootstrap === 'undefined') {
            return;
        }
        state.modalInstance = new bootstrap.Modal(elements.modal, { backdrop: 'static', keyboard: false });
        elements.saveButton?.addEventListener('click', saveLibrary);
        elements.nameInput?.addEventListener('input', () => {
            elements.nameInput.classList.remove('is-invalid');
            updateFilenamePreview();
        });
        elements.modal.addEventListener('hidden.bs.modal', () => {
            state.targetInput = null;
            state.previousFilename = null;
            clearError();
            resetRows();
            if (elements.nameInput) {
                elements.nameInput.value = '';
            }
            updateFilenamePreview();
        });
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
        init() {
            initModal();
            initFieldTriggers();
            refreshList().catch((error) => console.debug('Prompt library preload failed:', error));
        },
        initTriggers: (root) => initFieldTriggers(root),
    };

    document.addEventListener('DOMContentLoaded', () => manager.init());
    document.addEventListener('htmx:load', (event) => manager.initTriggers(event.detail?.elt || event.target));
    document.addEventListener('htmx:afterSwap', () => manager.initTriggers());
    window.promptLibraryManager = manager;
})();
