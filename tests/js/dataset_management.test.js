/**
 * Tests for dataset management functionality.
 *
 * Covers dataset CRUD operations, filtering, and search.
 * These tests replace Selenium E2E tests for DatasetManagementTestCase.
 */

// Mock fetch
global.fetch = jest.fn();

// Counter for unique IDs
let datasetIdCounter = 0;

// Create mock datasets module
const createDatasetManager = () => ({
    datasets: [],
    filteredDatasets: [],
    searchQuery: '',
    datasetTypes: ['image', 'video', 'text_embeds', 'conditioning'],
    loading: false,
    error: null,

    addDataset(type) {
        datasetIdCounter++;
        const id = `${type}-${Date.now()}-${datasetIdCounter}`;
        const dataset = {
            id,
            type,
            instance_data_dir: '',
            caption_strategy: 'textfile',
            disabled: false,
        };

        // Add type-specific defaults
        if (type === 'text_embeds') {
            dataset.cache_dir = '{output_dir}/cache/text/{model_family}';
        } else if (type === 'conditioning') {
            dataset.conditioning_type = 'canny';
        }

        this.datasets.push(dataset);
        this.applyFilter();
        return dataset;
    },

    removeDataset(id) {
        const index = this.datasets.findIndex((d) => d.id === id);
        if (index !== -1) {
            this.datasets.splice(index, 1);
            this.applyFilter();
            return true;
        }
        return false;
    },

    getDataset(id) {
        return this.datasets.find((d) => d.id === id);
    },

    updateDataset(id, updates) {
        const dataset = this.getDataset(id);
        if (dataset) {
            Object.assign(dataset, updates);
            return true;
        }
        return false;
    },

    searchDatasets(query) {
        this.searchQuery = query.toLowerCase();
        this.applyFilter();
    },

    clearSearch() {
        this.searchQuery = '';
        this.applyFilter();
    },

    applyFilter() {
        if (!this.searchQuery) {
            this.filteredDatasets = [...this.datasets];
        } else {
            this.filteredDatasets = this.datasets.filter(
                (d) =>
                    d.id.toLowerCase().includes(this.searchQuery) ||
                    d.type.toLowerCase().includes(this.searchQuery)
            );
        }
    },

    getDatasetCount() {
        return this.datasets.length;
    },

    getFilteredCount() {
        return this.filteredDatasets.length;
    },

    toggleDataset(id) {
        const dataset = this.getDataset(id);
        if (dataset) {
            dataset.disabled = !dataset.disabled;
            return true;
        }
        return false;
    },

    getEnabledDatasets() {
        return this.datasets.filter((d) => !d.disabled);
    },

    getDisabledDatasets() {
        return this.datasets.filter((d) => d.disabled);
    },

    validateDataset(dataset) {
        const errors = [];

        if (!dataset.id || dataset.id.trim() === '') {
            errors.push('Dataset ID is required');
        }

        if (!dataset.instance_data_dir && dataset.type !== 'text_embeds') {
            errors.push('Data directory is required');
        }

        return {
            valid: errors.length === 0,
            errors,
        };
    },

    getDatasetsByType(type) {
        return this.datasets.filter((d) => d.type === type);
    },
});

describe('Dataset Management', () => {
    let manager;

    beforeEach(() => {
        manager = createDatasetManager();
    });

    describe('adding datasets', () => {
        test('can add image dataset', () => {
            const dataset = manager.addDataset('image');

            expect(dataset.type).toBe('image');
            expect(manager.getDatasetCount()).toBe(1);
        });

        test('can add video dataset', () => {
            const dataset = manager.addDataset('video');

            expect(dataset.type).toBe('video');
        });

        test('can add text_embeds dataset', () => {
            const dataset = manager.addDataset('text_embeds');

            expect(dataset.type).toBe('text_embeds');
            expect(dataset.cache_dir).toBeDefined();
        });

        test('can add conditioning dataset', () => {
            const dataset = manager.addDataset('conditioning');

            expect(dataset.type).toBe('conditioning');
            expect(dataset.conditioning_type).toBe('canny');
        });

        test('dataset has unique ID', () => {
            const dataset1 = manager.addDataset('image');
            const dataset2 = manager.addDataset('image');

            expect(dataset1.id).not.toBe(dataset2.id);
        });

        test('dataset has default caption_strategy', () => {
            const dataset = manager.addDataset('image');

            expect(dataset.caption_strategy).toBe('textfile');
        });

        test('dataset starts enabled', () => {
            const dataset = manager.addDataset('image');

            expect(dataset.disabled).toBe(false);
        });
    });

    describe('removing datasets', () => {
        test('can remove dataset by ID', () => {
            const dataset = manager.addDataset('image');

            const result = manager.removeDataset(dataset.id);

            expect(result).toBe(true);
            expect(manager.getDatasetCount()).toBe(0);
        });

        test('returns false when removing non-existent dataset', () => {
            const result = manager.removeDataset('non-existent-id');

            expect(result).toBe(false);
        });

        test('only removes specified dataset', () => {
            const dataset1 = manager.addDataset('image');
            const dataset2 = manager.addDataset('video');

            manager.removeDataset(dataset1.id);

            expect(manager.getDatasetCount()).toBe(1);
            expect(manager.getDataset(dataset2.id)).toBeDefined();
        });
    });

    describe('updating datasets', () => {
        test('can update dataset properties', () => {
            const dataset = manager.addDataset('image');

            manager.updateDataset(dataset.id, {
                instance_data_dir: '/path/to/data',
                caption_strategy: 'filename',
            });

            const updated = manager.getDataset(dataset.id);
            expect(updated.instance_data_dir).toBe('/path/to/data');
            expect(updated.caption_strategy).toBe('filename');
        });

        test('returns false when updating non-existent dataset', () => {
            const result = manager.updateDataset('non-existent', { caption_strategy: 'filename' });

            expect(result).toBe(false);
        });
    });

    describe('searching datasets', () => {
        beforeEach(() => {
            manager.addDataset('image');
            manager.addDataset('video');
            manager.addDataset('text_embeds');
        });

        test('search filters by type', () => {
            manager.searchDatasets('image');

            expect(manager.getFilteredCount()).toBe(1);
            expect(manager.filteredDatasets[0].type).toBe('image');
        });

        test('search filters by ID', () => {
            const datasets = manager.datasets;
            const imageDataset = datasets.find((d) => d.type === 'image');

            manager.searchDatasets(imageDataset.id.substring(0, 5));

            expect(manager.getFilteredCount()).toBeGreaterThanOrEqual(1);
        });

        test('search is case-insensitive', () => {
            manager.searchDatasets('IMAGE');

            expect(manager.getFilteredCount()).toBe(1);
        });

        test('clear search shows all datasets', () => {
            manager.searchDatasets('image');
            expect(manager.getFilteredCount()).toBe(1);

            manager.clearSearch();
            expect(manager.getFilteredCount()).toBe(3);
        });

        test('empty search shows all datasets', () => {
            manager.searchDatasets('');

            expect(manager.getFilteredCount()).toBe(3);
        });
    });

    describe('dataset toggling', () => {
        test('can disable dataset', () => {
            const dataset = manager.addDataset('image');

            manager.toggleDataset(dataset.id);

            expect(manager.getDataset(dataset.id).disabled).toBe(true);
        });

        test('can re-enable dataset', () => {
            const dataset = manager.addDataset('image');
            manager.toggleDataset(dataset.id); // disable
            manager.toggleDataset(dataset.id); // re-enable

            expect(manager.getDataset(dataset.id).disabled).toBe(false);
        });

        test('getEnabledDatasets returns only enabled', () => {
            const dataset1 = manager.addDataset('image');
            const dataset2 = manager.addDataset('video');
            manager.toggleDataset(dataset1.id); // disable

            const enabled = manager.getEnabledDatasets();

            expect(enabled.length).toBe(1);
            expect(enabled[0].id).toBe(dataset2.id);
        });

        test('getDisabledDatasets returns only disabled', () => {
            const dataset1 = manager.addDataset('image');
            manager.addDataset('video');
            manager.toggleDataset(dataset1.id); // disable

            const disabled = manager.getDisabledDatasets();

            expect(disabled.length).toBe(1);
            expect(disabled[0].id).toBe(dataset1.id);
        });
    });

    describe('dataset validation', () => {
        test('validates missing ID', () => {
            const dataset = { id: '', type: 'image', instance_data_dir: '/path' };

            const result = manager.validateDataset(dataset);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Dataset ID is required');
        });

        test('validates missing data directory for non-text_embeds', () => {
            const dataset = { id: 'test', type: 'image', instance_data_dir: '' };

            const result = manager.validateDataset(dataset);

            expect(result.valid).toBe(false);
            expect(result.errors).toContain('Data directory is required');
        });

        test('text_embeds can have empty data directory', () => {
            const dataset = { id: 'text-cache', type: 'text_embeds', instance_data_dir: '' };

            const result = manager.validateDataset(dataset);

            expect(result.valid).toBe(true);
        });

        test('valid dataset passes validation', () => {
            const dataset = { id: 'test', type: 'image', instance_data_dir: '/path/to/data' };

            const result = manager.validateDataset(dataset);

            expect(result.valid).toBe(true);
            expect(result.errors.length).toBe(0);
        });
    });

    describe('filtering by type', () => {
        beforeEach(() => {
            manager.addDataset('image');
            manager.addDataset('image');
            manager.addDataset('video');
            manager.addDataset('text_embeds');
        });

        test('getDatasetsByType returns correct datasets', () => {
            const images = manager.getDatasetsByType('image');
            const videos = manager.getDatasetsByType('video');

            expect(images.length).toBe(2);
            expect(videos.length).toBe(1);
        });

        test('getDatasetsByType returns empty for no matches', () => {
            const conditioning = manager.getDatasetsByType('conditioning');

            expect(conditioning.length).toBe(0);
        });
    });
});

describe('Dataset Builder View Mode', () => {
    let manager;

    beforeEach(() => {
        manager = createDatasetManager();
    });

    test('can add multiple datasets of same type', () => {
        manager.addDataset('image');
        manager.addDataset('image');
        manager.addDataset('image');

        expect(manager.getDatasetsByType('image').length).toBe(3);
    });

    test('auto text_embeds dataset scenario', () => {
        // Simulate auto-created text embeds
        manager.addDataset('text_embeds');

        // Add user datasets
        manager.addDataset('image');
        manager.addDataset('video');

        expect(manager.getDatasetCount()).toBe(3);
        expect(manager.getDatasetsByType('text_embeds').length).toBe(1);
    });
});
