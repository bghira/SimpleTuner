require('../../simpletuner/static/js/dataloader-section-component.js');


describe('Dataloader train batch size eligibility', () => {
    let component;

    beforeEach(() => {
        component = window.dataloaderSectionComponent();
        component.markAsUnsaved = jest.fn();
    });

    test.each(['image', 'video', 'audio', 'caption'])(
        'allows independently sampled %s datasets',
        (datasetType) => {
            expect(component.supportsTrainBatchSize({ dataset_type: datasetType })).toBe(true);
        }
    );

    test.each(['conditioning', 'eval', 'text_embeds', 'image_embeds'])(
        'rejects %s datasets',
        (datasetType) => {
            expect(component.supportsTrainBatchSize({ dataset_type: datasetType })).toBe(false);
        }
    );

    test('removes an existing override when the dataset type becomes unsupported', () => {
        const dataset = { dataset_type: 'conditioning', train_batch_size: 4 };

        component.onDatasetTypeChange(dataset);

        expect(dataset).not.toHaveProperty('train_batch_size');
        expect(component.markAsUnsaved).toHaveBeenCalledTimes(1);
    });

    test('preserves an existing override when the dataset type remains supported', () => {
        const dataset = { dataset_type: 'caption', train_batch_size: 4 };

        component.onDatasetTypeChange(dataset);

        expect(dataset.train_batch_size).toBe(4);
        expect(component.markAsUnsaved).toHaveBeenCalledTimes(1);
    });
});
