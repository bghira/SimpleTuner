/**
 * Search component for fuzzy searching tabs and fields.
 * Provides intelligent search with keyboard navigation and result highlighting.
 */

function searchComponent() {
    return {
        query: '',
        results: { tabs: [], fields: [] },
        showResults: false,
        highlightedIndex: -1,
        searchTimeout: null,
        isLoading: false,

        init() {
            // Hide results when clicking outside
            document.addEventListener('click', (e) => {
                if (!this.$el.contains(e.target)) {
                    this.showResults = false;
                    this.highlightedIndex = -1;
                }
            });

            // Hide results on escape key
            document.addEventListener('keydown', (e) => {
                if (e.key === 'Escape' && this.showResults) {
                    this.showResults = false;
                    this.highlightedIndex = -1;
                    e.preventDefault();
                }
            });

            // Show results when search input gets focused (if there are results)
            this.$el.querySelector('.search-input')?.addEventListener('focus', () => {
                if (this.hasResults) {
                    this.showResults = true;
                }
            });

            // Initialize flags and observer
            this.isApplyingHighlighting = false;
            this.tabChangeTimeout = null;
            this.hasPrimaryHighlight = false; // Track if primary highlight was set by search result click
            this.primarySelectedField = null; // Store persistent primary field info
            this.setupTabChangeObserver();
        },

        setupTabChangeObserver() {
            // Watch for tab changes to apply search highlighting
            this.tabObserver = new MutationObserver((mutations) => {
                // Prevent infinite loop - don't process if we're already applying highlighting
                if (this.isApplyingHighlighting) return;

                mutations.forEach((mutation) => {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'class') {
                        const target = mutation.target;
                        if (target.classList.contains('tab-btn') && target.classList.contains('active')) {
                            // Clear any existing timeout
                            if (this.tabChangeTimeout) {
                                clearTimeout(this.tabChangeTimeout);
                            }

                            // Tab changed, apply search highlighting if we have results
                            this.tabChangeTimeout = setTimeout(() => {
                                this.applySearchHighlighting();
                            }, 100); // Small delay to ensure content is loaded
                        }
                    }
                });
            });

            // Observe all tab buttons for class changes
            document.querySelectorAll('.tab-btn').forEach(tab => {
                this.tabObserver.observe(tab, { attributes: true, attributeFilter: ['class'] });
            });
        },

        async handleSearch() {
            // Clear previous timeout
            if (this.searchTimeout) {
                clearTimeout(this.searchTimeout);
            }

            // Don't search for very short queries
            if (this.query.length < 2) {
                // Clear highlights and filtering but keep results for potential re-showing
                this.clearAllHighlights();
                this.clearFiltering();
                this.showResults = false;
                this.highlightedIndex = -1;
                return;
            }

            // Debounce search with 300ms delay
            this.searchTimeout = setTimeout(async () => {
                await this.performSearch();
            }, 300);
        },

        resetSearchState() {
            // Clear all highlights and filtering
            this.clearAllHighlights();
            this.clearFiltering();

            // Reset search results and UI state
            this.results = { tabs: [], fields: [] };
            this.showResults = false;
            this.highlightedIndex = -1;

            // Clear any active highlights tracking
            this.activeHighlights = [];

            console.log('Search state reset completed');
        },

        async performSearch() {
            if (this.isLoading) return;

            this.isLoading = true;

            try {
                console.log('🔍 SEARCHING FOR:', this.query);
                const response = await fetch(`/web/trainer/search?q=${encodeURIComponent(this.query)}&limit=10`);

                if (!response.ok) {
                    throw new Error('Search failed');
                }

                const data = await response.json();
                this.results = data.results || { tabs: [], fields: [] };

                console.log('📋 RAW SEARCH RESULTS:', data);
                console.log('📋 PROCESSED RESULTS:', this.results);
                console.log('📊 TABS FOUND:', this.results.tabs.length, this.results.tabs);
                console.log('📊 FIELDS FOUND:', this.results.fields.length, this.results.fields);

                // Log all field details for debugging
                this.results.fields.forEach((field, index) => {
                    console.log(`  📝 FIELD ${index + 1}: ${field.name} (${field.title}) in ${field.context?.tab}`);
                });

                this.showResults = true;
                this.highlightedIndex = -1;

                // No result limiting - keep all fields since we don't have that many
                const totalTabs = this.results.tabs.length;
                const totalFields = this.results.fields.length;

                console.log(`📊 TOTAL RESULTS: ${totalTabs} tabs + ${totalFields} fields = ${totalTabs + totalFields} total (no limiting)`);
                console.log(`� KEEPING ALL ${totalTabs} tabs and ${totalFields} fields`);

            } catch (error) {
                console.error('Search error:', error);
                this.results = { tabs: [], fields: [] };
                this.showResults = false;
            } finally {
                this.isLoading = false;
            }
        },

        clearSearch() {
            console.log('🧹 CLEARING SEARCH - removing all highlights and tab styling');

            this.query = '';
            this.results = { tabs: [], fields: [] };
            this.showResults = false;
            this.highlightedIndex = -1;

            // Clear all highlights and reset state
            this.clearAllHighlights();
            this.clearFiltering();

            // Clear tab highlighting
            this.clearTabHighlighting();

            // Focus the input with error handling
            try {
                const searchInput = this.$el.querySelector('.search-input');
                if (searchInput && searchInput.focus) {
                    searchInput.focus();
                }
            } catch (error) {
                console.warn('Could not focus search input:', error);
            }

            console.log('✅ SEARCH CLEARED - all highlights and tab styling removed');
        },

        highlightNext() {
            if (!this.showResults) return;

            const totalResults = this.results.tabs.length + this.results.fields.length;
            if (totalResults === 0) return;

            this.highlightedIndex = (this.highlightedIndex + 1) % totalResults;
            this.scrollToHighlighted();
        },

        highlightPrevious() {
            if (!this.showResults) return;

            const totalResults = this.results.tabs.length + this.results.fields.length;
            if (totalResults === 0) return;

            this.highlightedIndex = this.highlightedIndex <= 0
                ? totalResults - 1
                : this.highlightedIndex - 1;
            this.scrollToHighlighted();
        },

        scrollToHighlighted() {
            const highlightedElement = this.$el.querySelector('.search-result-item.highlighted');
            if (highlightedElement) {
                highlightedElement.scrollIntoView({
                    block: 'nearest',
                    behavior: 'smooth'
                });
            }
        },

        selectHighlighted() {
            if (!this.showResults || this.highlightedIndex === -1) return;

            const totalTabs = this.results.tabs.length;

            if (this.highlightedIndex < totalTabs) {
                // Select tab
                const tab = this.results.tabs[this.highlightedIndex];
                this.selectTab(tab.name);
            } else {
                // Select field
                const fieldIndex = this.highlightedIndex - totalTabs;
                const field = this.results.fields[fieldIndex];
                this.selectField(field);
            }
        },

        selectTab(tabName) {
            // Use simple tab switching - let the existing system handle it
            const tabButton = document.querySelector(`.tab-btn[data-tab="${tabName}"]`);
            if (tabButton) {
                tabButton.click();
            }

            // Hide search results after selection
            this.showResults = false;
            this.highlightedIndex = -1;
        },

        selectField(fieldResult) {
            // Store persistent primary field info (non-disruptive)
            console.log('🔍 SEARCH RESULT CLICKED:', fieldResult);
            console.log('  - Tab:', fieldResult.context.tab);
            console.log('  - Field Name:', fieldResult.name);
            console.log('  - Title:', fieldResult.title);

            this.primarySelectedField = {
                tabName: fieldResult.context.tab,
                fieldName: fieldResult.name,
                title: fieldResult.title
            };

            console.log('✅ STORED PRIMARY FIELD:', this.primarySelectedField);

            // First switch to the correct tab
            this.selectTab(fieldResult.context.tab);

            // Then focus and highlight the specific field after a short delay
            setTimeout(() => {
                console.log('🔍 LOOKING FOR FIELD ELEMENTS WITH NAME:', fieldResult.name);
                const fieldElements = this.findFieldElements(fieldResult.name);
                console.log('  - Found', fieldElements.length, 'elements');

                if (fieldElements.length > 0) {
                    // Set flag to indicate primary highlight was set by search result click
                    this.hasPrimaryHighlight = true;

                    // Scroll to and highlight the field
                    this.scrollToElement(fieldElements[0]);
                    this.highlightPrimaryField(fieldElements[0]);

                    // Focus the field
                    try {
                        this.focusField(fieldElements[0]);
                    } catch (focusError) {
                        console.warn('Failed to focus field:', focusError);
                    }
                } else {
                    console.warn('❌ NO FIELD ELEMENTS FOUND FOR:', fieldResult.name);
                }
            }, 200);

            // Hide search results after selection
            this.showResults = false;
            this.highlightedIndex = -1;
        },

        // Apply search highlighting to current tab (for when user switches tabs manually)
        applySearchHighlighting() {
            // Prevent infinite loop - don't process if we're already applying highlighting
            if (this.isApplyingHighlighting || !this.hasResults) return;

            // Set flag to prevent re-entrant calls
            this.isApplyingHighlighting = true;

            try {
                // Temporarily disconnect observer to prevent infinite loop
                if (this.tabObserver) {
                    this.tabObserver.disconnect();
                }

                // Get current active tab
                const activeTab = document.querySelector('.tab-btn.active');
                if (!activeTab) return;

                const tabName = activeTab.getAttribute('data-tab');
                if (!tabName) return;

                // Check if we have persistent primary field for this tab
                const hasPersistentPrimary = this.primarySelectedField &&
                                       this.primarySelectedField.tabName === tabName;

                // Check if we have primary highlights to preserve
                const hasPrimaryHighlights = document.querySelectorAll('.search-highlighted-field-primary').length > 0;

                if (hasPersistentPrimary || hasPrimaryHighlights) {
                    // Preserve primary highlights, only clear secondary ones
                    this.clearSecondaryHighlights();

                    // Try to restore persistent primary highlight
                    if (hasPersistentPrimary) {
                        this.restorePrimaryHighlight();
                    }
                } else {
                    // No primary highlights, clear everything
                    this.clearAllHighlights();
                }

                // Highlight tabs based on search results
                this.highlightTabs(this.results.tabs.concat(this.results.fields));

                // Highlight matching fields on current tab
                this.highlightMatchingFields(tabName);

            } finally {
                // Clear flag and reconnect observer
                this.isApplyingHighlighting = false;
                if (this.tabObserver) {
                    // Reconnect observer to all tab buttons
                    document.querySelectorAll('.tab-btn').forEach(tab => {
                        this.tabObserver.observe(tab, { attributes: true, attributeFilter: ['class'] });
                    });
                }
            }
        },

        clearFiltering() {
            // Clear all highlights
            this.clearAllHighlights();
        },

        highlightTabs(results) {
            const allTabs = document.querySelectorAll('.tab-btn');

            allTabs.forEach(tab => {
                const tabName = tab.getAttribute('data-tab');
                const isActiveTab = tab.classList.contains('active');

                // Check if tab has direct tab results OR field results (case-insensitive)
                const hasTabResults = results.some(result =>
                    result.type === 'tab' && result.name && result.name.toLowerCase() === tabName.toLowerCase()
                );
                const hasFieldResults = results.some(result =>
                    result.type === 'field' && result.context && result.context.tab &&
                    result.context.tab.toLowerCase() === tabName.toLowerCase()
                );

                const hasResults = hasTabResults || hasFieldResults;

                // Remove all previous states
                tab.classList.remove('search-highlighted-tab', 'search-dimmed-tab', 'search-highlighted-active-tab');

                if (hasResults) {
                    if (isActiveTab) {
                        // Active tab with results gets special highlighting
                        tab.classList.add('search-highlighted-active-tab');
                    } else {
                        // Inactive tab with results gets regular highlighting
                        tab.classList.add('search-highlighted-tab');
                    }
                } else {
                    // Tabs without results get dimmed
                    tab.classList.add('search-dimmed-tab');
                }
            });
        },

        findFieldElements(fieldName) {
            const strategies = [
                // Strategy 1: Direct name match
                () => document.querySelectorAll(`[name="${fieldName}"]`),

                // Strategy 2: ID match
                () => document.querySelectorAll(`#${fieldName}`),

                // Strategy 3: Label text match
                () => this.findFieldsByLabel(fieldName),

                // Strategy 4: Data attribute match
                () => document.querySelectorAll(`[data-field-name="${fieldName}"]`),

                // Strategy 5: Partial name match
                () => document.querySelectorAll(`[name*="${fieldName}"]`),

                // Strategy 6: Try with -- prefix for command line arguments
                () => document.querySelectorAll(`[name="--${fieldName}"]`),
                () => document.querySelectorAll(`[data-field-name="--${fieldName}"]`)
            ];

            for (const strategy of strategies) {
                const elements = strategy();
                if (elements.length > 0) {
                    return Array.from(elements);
                }
            }

            return [];
        },

        findFieldsByLabel(fieldName) {
            const labels = document.querySelectorAll('label');
            const matchingFields = [];

            for (const label of labels) {
                const labelText = label.textContent.toLowerCase();
                const fieldNameLower = fieldName.toLowerCase();

                if (labelText.includes(fieldNameLower) || fieldNameLower.includes(labelText)) {
                    const field = label.querySelector('input, select, textarea') ||
                                 label.parentElement.querySelector('input, select, textarea') ||
                                 label.nextElementSibling;
                    if (field) {
                        matchingFields.push(field);
                    }
                }
            }

            return matchingFields;
        },

        highlightField(element) {
            // Add field-specific highlighting (secondary highlight)
            element.classList.add('search-highlighted-field-secondary');

            // Also highlight the parent section for context
            const section = element.closest('.form-section, .field-group, .config-section');
            if (section) {
                section.classList.add('search-highlighted-section-secondary');
            }

            // Store reference for manual removal
            if (!this.activeHighlights) {
                this.activeHighlights = [];
            }
            this.activeHighlights.push({ element, section, type: 'secondary' });
        },

        highlightPrimaryField(element) {
            // Add primary field-specific highlighting (stronger highlight)
            element.classList.add('search-highlighted-field-primary');

            // Also highlight the parent section for context
            const section = element.closest('.form-section, .field-group, .config-section');
            if (section) {
                section.classList.add('search-highlighted-section-primary');
            }

            // Store reference for manual removal
            if (!this.activeHighlights) {
                this.activeHighlights = [];
            }
            this.activeHighlights.push({ element, section, type: 'primary' });
        },

        scrollToElement(element) {
            element.scrollIntoView({
                behavior: 'smooth',
                block: 'center'
            });
        },

        focusField(element) {
            if (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA' || element.tagName === 'SELECT') {
                element.focus();

                // For text-based inputs, move cursor to end (only if selection is supported)
                const textInputTypes = ['text', 'email', 'password', 'search', 'tel', 'url'];
                if (textInputTypes.includes(element.type) || element.tagName === 'TEXTAREA') {
                    try {
                        element.setSelectionRange(element.value.length, element.value.length);
                    } catch (error) {
                        // Silently handle selection errors (e.g., for inputs that don't support selection)
                        console.debug('Could not set selection range:', error);
                    }
                }
            }
        },

        clearAllHighlights() {
            // Remove field highlights
            const fieldHighlights = document.querySelectorAll('.search-highlighted-field');
            fieldHighlights.forEach(element => {
                element.classList.remove('search-highlighted-field');
            });

            // Remove section highlights
            const sectionHighlights = document.querySelectorAll('.search-highlighted-section');
            sectionHighlights.forEach(element => {
                element.classList.remove('search-highlighted-section');
            });

            // Remove old search highlights
            const oldHighlights = document.querySelectorAll('.search-highlighted');
            oldHighlights.forEach(element => {
                element.classList.remove('search-highlighted');
            });

            // Remove secondary and primary highlights
            const secondaryHighlights = document.querySelectorAll('.search-highlighted-field-secondary');
            secondaryHighlights.forEach(element => {
                element.classList.remove('search-highlighted-field-secondary');
            });

            const primaryHighlights = document.querySelectorAll('.search-highlighted-field-primary');
            primaryHighlights.forEach(element => {
                element.classList.remove('search-highlighted-field-primary');
            });

            const secondarySectionHighlights = document.querySelectorAll('.search-highlighted-section-secondary');
            secondarySectionHighlights.forEach(element => {
                element.classList.remove('search-highlighted-section-secondary');
            });

            const primarySectionHighlights = document.querySelectorAll('.search-highlighted-section-primary');
            primarySectionHighlights.forEach(element => {
                element.classList.remove('search-highlighted-section-primary');
            });

            // Clear active highlights tracking
            this.activeHighlights = [];

            // Reset primary highlight flag when clearing all highlights
            this.hasPrimaryHighlight = false;

            // Clear persistent primary field storage
            this.primarySelectedField = null;
        },

        clearSecondaryHighlights() {
            // Remove only secondary highlights, preserve primary ones
            const secondaryHighlights = document.querySelectorAll('.search-highlighted-field-secondary');
            secondaryHighlights.forEach(element => {
                element.classList.remove('search-highlighted-field-secondary');
            });

            const secondarySectionHighlights = document.querySelectorAll('.search-highlighted-section-secondary');
            secondarySectionHighlights.forEach(element => {
                element.classList.remove('search-highlighted-section-secondary');
            });

            // Clear active highlights tracking for secondary highlights only
            if (this.activeHighlights) {
                this.activeHighlights = this.activeHighlights.filter(highlight => highlight.type !== 'secondary');
            }
        },

        clearTabHighlighting() {
            // Remove all search-related styling from navigation tabs
            const highlightedTabs = document.querySelectorAll('.search-highlighted-tab, .search-highlighted-active-tab, .search-dimmed-tab');
            highlightedTabs.forEach(tab => {
                tab.classList.remove('search-highlighted-tab', 'search-highlighted-active-tab', 'search-dimmed-tab');
            });

            console.log('🧹 TAB HIGHLIGHTING CLEARED - navigation tabs restored to normal styling');
        },

        restorePrimaryHighlight() {
            if (!this.primarySelectedField) return;

            console.log('Restoring primary highlight for:', this.primarySelectedField.fieldName);

            // Find the field using existing field finding strategies
            const fieldElements = this.findFieldElements(this.primarySelectedField.fieldName);

            if (fieldElements.length > 0) {
                // Re-apply primary highlight
                this.highlightPrimaryField(fieldElements[0]);
                this.hasPrimaryHighlight = true;
                console.log('Successfully restored primary highlight');
            } else {
                console.warn('Could not find field to restore primary highlight:', this.primarySelectedField.fieldName);
                // If field not found, clear the persistent storage to prevent future attempts
                this.primarySelectedField = null;
            }
        },

        async highlightMatchingFields(tabName) {
            // Get all field results for the current tab (case-insensitive comparison)
            const matchingFields = this.results.fields.filter(field =>
                field.context && field.context.tab && field.context.tab.toLowerCase() === tabName.toLowerCase()
            );

            if (matchingFields.length === 0) {
                console.log('No matching fields found for tab:', tabName);
                return;
            }

            console.log(`🎯 HIGHLIGHTING ${matchingFields.length} MATCHING FIELDS IN TAB:`, tabName);
            console.log('📋 MATCHING FIELD DETAILS:', matchingFields);

            // Highlight ALL matching fields with secondary highlighting first
            let highlightedCount = 0;
            for (const fieldResult of matchingFields) {
                console.log(`🔍 PROCESSING FIELD: ${fieldResult.name} (${fieldResult.title})`);
                const fieldElements = this.findFieldElements(fieldResult.name);

                console.log(`  📍 FOUND ${fieldElements.length} ELEMENTS FOR FIELD ${fieldResult.name}:`, fieldElements);

                if (fieldElements.length > 0) {
                    fieldElements.forEach(element => {
                        console.log(`  ✨ HIGHLIGHTING ELEMENT (SECONDARY):`, element);
                        console.log(`    - Element classes before:`, element.className);
                        this.highlightField(element);
                        console.log(`    - Element classes after:`, element.className);
                        highlightedCount++;
                    });
                } else {
                    console.warn(`  ❌ NO ELEMENTS FOUND FOR FIELD: ${fieldResult.name}`);
                }
            }

            console.log(`📊 HIGHLIGHTED ${highlightedCount} FIELDS OUT OF ${matchingFields.length} MATCHING FIELDS`);

            // If we have a primary selected field, re-apply primary highlighting
            if (this.primarySelectedField && this.primarySelectedField.tabName === tabName) {
                console.log(`🎯 RE-APPLYING PRIMARY HIGHLIGHT TO: ${this.primarySelectedField.fieldName}`);

                const primaryFieldElements = this.findFieldElements(this.primarySelectedField.fieldName);
                if (primaryFieldElements.length > 0) {
                    console.log(`📍 FOUND ${primaryFieldElements.length} PRIMARY FIELD ELEMENTS`);

                    // Remove secondary highlight from primary field elements first
                    primaryFieldElements.forEach(element => {
                        console.log(`  🗑 REMOVING SECONDARY HIGHLIGHT FROM:`, element);
                        element.classList.remove('search-highlighted-field-secondary');
                        const section = element.closest('.form-section, .field-group, .config-section');
                        if (section) {
                            section.classList.remove('search-highlighted-section-secondary');
                        }
                    });

                    // Apply primary highlighting
                    primaryFieldElements.forEach(element => {
                        console.log(`  ⭐ APPLYING PRIMARY HIGHLIGHT TO:`, element);
                        this.highlightPrimaryField(element);
                    });

                    console.log('✅ SUCCESSFULLY RE-APPLIED PRIMARY HIGHLIGHT');
                } else {
                    console.warn(`❌ COULD NOT FIND PRIMARY FIELD ELEMENTS FOR: ${this.primarySelectedField.fieldName}`);
                }
            } else {
                console.log(`ℹ️ NO PRIMARY FIELD STORED FOR TAB: ${tabName}`);
            }

            // Count total highlighted elements
            const secondaryElements = document.querySelectorAll('.search-highlighted-field-secondary');
            const primaryElements = document.querySelectorAll('.search-highlighted-field-primary');
            console.log(`🎨 TOTAL HIGHLIGHTED ELEMENTS: ${secondaryElements.length} secondary + ${primaryElements.length} primary = ${secondaryElements.length + primaryElements.length}`);

            // Scroll to primary field if it exists, otherwise first secondary field
            const primaryField = document.querySelector('.search-highlighted-field-primary');
            const firstField = primaryField || document.querySelector('.search-highlighted-field-secondary');
            if (firstField) {
                console.log(`🎯 SCROLLING TO:`, firstField);
                this.scrollToElement(firstField);
            }
        },

        get totalResults() {
            return this.results.tabs.length + this.results.fields.length;
        },

        get hasResults() {
            return this.totalResults > 0;
        }
    };
}

// Make the component available globally
window.searchComponent = searchComponent;

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { searchComponent };
}
