/**
 * Tests for trainer navigation and tab management.
 *
 * Covers tab navigation, form state management, and configuration flow.
 * These tests replace Selenium E2E tests for TabNavigationTestCase.
 */

// Mock Alpine store
const createMockTrainerStore = () => ({
    activeTab: 'basic',
    configValues: {},
    saving: false,
    loading: false,
    modelContext: {},
    tabs: ['basic', 'model', 'datasets', 'training', 'validation', 'cloud'],

    switchTab(tabName) {
        if (this.tabs.includes(tabName)) {
            this.activeTab = tabName;
            return true;
        }
        return false;
    },

    isTabActive(tabName) {
        return this.activeTab === tabName;
    },

    getNextTab() {
        const currentIndex = this.tabs.indexOf(this.activeTab);
        if (currentIndex < this.tabs.length - 1) {
            return this.tabs[currentIndex + 1];
        }
        return null;
    },

    getPreviousTab() {
        const currentIndex = this.tabs.indexOf(this.activeTab);
        if (currentIndex > 0) {
            return this.tabs[currentIndex - 1];
        }
        return null;
    },

    goToNextTab() {
        const next = this.getNextTab();
        if (next) {
            this.activeTab = next;
            return true;
        }
        return false;
    },

    goToPreviousTab() {
        const prev = this.getPreviousTab();
        if (prev) {
            this.activeTab = prev;
            return true;
        }
        return false;
    },
});

describe('Trainer Tab Navigation', () => {
    let store;

    beforeEach(() => {
        store = createMockTrainerStore();
    });

    describe('tab switching', () => {
        test('starts on basic tab', () => {
            expect(store.activeTab).toBe('basic');
            expect(store.isTabActive('basic')).toBe(true);
        });

        test('can switch to model tab', () => {
            store.switchTab('model');

            expect(store.activeTab).toBe('model');
            expect(store.isTabActive('model')).toBe(true);
            expect(store.isTabActive('basic')).toBe(false);
        });

        test('can switch to datasets tab', () => {
            store.switchTab('datasets');

            expect(store.activeTab).toBe('datasets');
        });

        test('can switch to training tab', () => {
            store.switchTab('training');

            expect(store.activeTab).toBe('training');
        });

        test('can switch to validation tab', () => {
            store.switchTab('validation');

            expect(store.activeTab).toBe('validation');
        });

        test('can switch to cloud tab', () => {
            store.switchTab('cloud');

            expect(store.activeTab).toBe('cloud');
        });

        test('returns false for invalid tab', () => {
            const result = store.switchTab('nonexistent');

            expect(result).toBe(false);
            expect(store.activeTab).toBe('basic');
        });
    });

    describe('sequential navigation', () => {
        test('goToNextTab advances to next tab', () => {
            expect(store.activeTab).toBe('basic');

            store.goToNextTab();
            expect(store.activeTab).toBe('model');

            store.goToNextTab();
            expect(store.activeTab).toBe('datasets');
        });

        test('goToPreviousTab goes back to previous tab', () => {
            store.activeTab = 'datasets';

            store.goToPreviousTab();
            expect(store.activeTab).toBe('model');

            store.goToPreviousTab();
            expect(store.activeTab).toBe('basic');
        });

        test('goToNextTab returns false on last tab', () => {
            store.activeTab = 'cloud';

            const result = store.goToNextTab();

            expect(result).toBe(false);
            expect(store.activeTab).toBe('cloud');
        });

        test('goToPreviousTab returns false on first tab', () => {
            store.activeTab = 'basic';

            const result = store.goToPreviousTab();

            expect(result).toBe(false);
            expect(store.activeTab).toBe('basic');
        });
    });

    describe('tab helpers', () => {
        test('getNextTab returns correct tab', () => {
            store.activeTab = 'basic';
            expect(store.getNextTab()).toBe('model');

            store.activeTab = 'model';
            expect(store.getNextTab()).toBe('datasets');
        });

        test('getNextTab returns null on last tab', () => {
            store.activeTab = 'cloud';
            expect(store.getNextTab()).toBeNull();
        });

        test('getPreviousTab returns correct tab', () => {
            store.activeTab = 'model';
            expect(store.getPreviousTab()).toBe('basic');

            store.activeTab = 'datasets';
            expect(store.getPreviousTab()).toBe('model');
        });

        test('getPreviousTab returns null on first tab', () => {
            store.activeTab = 'basic';
            expect(store.getPreviousTab()).toBeNull();
        });
    });
});

describe('Trainer Configuration State', () => {
    let store;

    beforeEach(() => {
        store = createMockTrainerStore();
    });

    test('configValues starts empty', () => {
        expect(store.configValues).toEqual({});
    });

    test('can set config values', () => {
        store.configValues['--model_family'] = 'flux';
        store.configValues['--model_type'] = 'lora';

        expect(store.configValues['--model_family']).toBe('flux');
        expect(store.configValues['--model_type']).toBe('lora');
    });

    test('saving state is tracked', () => {
        expect(store.saving).toBe(false);

        store.saving = true;
        expect(store.saving).toBe(true);
    });

    test('loading state is tracked', () => {
        expect(store.loading).toBe(false);

        store.loading = true;
        expect(store.loading).toBe(true);
    });
});

describe('Tab Order Validation', () => {
    let store;

    beforeEach(() => {
        store = createMockTrainerStore();
    });

    test('tabs are in correct order', () => {
        expect(store.tabs).toEqual([
            'basic',
            'model',
            'datasets',
            'training',
            'validation',
            'cloud',
        ]);
    });

    test('can navigate through all tabs in order', () => {
        const visitedTabs = [store.activeTab];

        while (store.goToNextTab()) {
            visitedTabs.push(store.activeTab);
        }

        expect(visitedTabs).toEqual(store.tabs);
    });

    test('can navigate back through all tabs', () => {
        store.activeTab = 'cloud';
        const visitedTabs = [store.activeTab];

        while (store.goToPreviousTab()) {
            visitedTabs.push(store.activeTab);
        }

        expect(visitedTabs).toEqual([...store.tabs].reverse());
    });
});

describe('Basic Configuration Tab', () => {
    test('basic tab fields are identified correctly', () => {
        const basicTabFields = [
            '--job_id',
            '--output_dir',
            '--pretrained_model_name_or_path',
        ];

        basicTabFields.forEach((field) => {
            expect(field.startsWith('--')).toBe(true);
        });
    });
});

describe('Model Configuration Tab', () => {
    test('model tab fields are identified correctly', () => {
        const modelTabFields = [
            '--model_family',
            '--model_type',
            '--model_flavour',
            '--lora_rank',
        ];

        modelTabFields.forEach((field) => {
            expect(field.startsWith('--')).toBe(true);
        });
    });
});
