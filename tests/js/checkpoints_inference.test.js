window.HintMixin = {
    createMultiHint: () => ({
        hints: { hero: false },
        loadHints: jest.fn(),
        dismissHint: jest.fn(),
        showHint: jest.fn(),
    }),
};
window.ApiClient = { fetch: jest.fn() };
global.ApiClient = window.ApiClient;
window.checkpointsManager = undefined;

require('../../simpletuner/static/js/modules/checkpoints.js');

describe('checkpoint inference manager', () => {
    let manager;

    beforeEach(() => {
        manager = window.checkpointsManager();
        manager.environment = 'test-environment';
        manager.checkpoints = [
            { id: 'checkpoint-100', name: 'checkpoint-100', size: 100 },
            { id: 'checkpoint-200', name: 'checkpoint-200', size: 200 },
        ];
        window.ApiClient.fetch.mockReset();
    });

    test('tracks inference selection separately and disables persistent mode for batches', () => {
        manager.selectedCheckpoint = manager.checkpoints[0];
        manager.toggleInferenceCheckpoint('checkpoint-100');
        manager.inference.form.keep_loaded = true;
        manager.toggleInferenceCheckpoint('checkpoint-200');

        expect(manager.inferenceSelection).toEqual(['checkpoint-100', 'checkpoint-200']);
        expect(manager.selectedCheckpoint.name).toBe('checkpoint-100');
        expect(manager.inference.form.keep_loaded).toBe(false);
    });

    test('counts configured, built-in, library, and custom prompts', () => {
        manager.inferenceSelection = ['checkpoint-100', 'checkpoint-200'];
        manager.inference.promptSources = {
            configured_prompt: 'configured',
            builtin_count: 3,
            configured_user_library: null,
            user_libraries: [{ filename: 'user_prompt_library-test.json', prompt_count: 2 }],
        };
        Object.assign(manager.inference.form, {
            use_configured_prompt: true,
            use_builtin_library: true,
            user_library_filename: 'user_prompt_library-test.json',
            custom_prompts: 'first custom\n\nsecond custom',
            validation_resolution: '512,768x512',
        });

        expect(manager.inferencePromptCount()).toBe(8);
        expect(manager.inferenceRunCount()).toBe(32);
    });

    test('warns when configured validation multigpu modes are unsupported', () => {
        manager.inference.promptSources.unsupported_multigpu_modes = ['batch-parallel', 'context-parallel'];

        expect(manager.inferenceMultigpuWarning()).toBe(
            "Checkpoint inference currently runs on one process and will not use this environment's configured batch parallel and context parallel validation modes."
        );

        manager.inference.promptSources.unsupported_multigpu_modes = [];
        expect(manager.inferenceMultigpuWarning()).toBe('');
    });

    test('submits the selected checkpoint and prompt settings', async () => {
        manager.inferenceSelection = ['checkpoint-100'];
        Object.assign(manager.inference.form, {
            use_configured_prompt: false,
            custom_prompts: 'a custom prompt',
            filename_style: 'content-hash',
            keep_loaded: true,
            streaming_preview: true,
            idle_timeout_minutes: 9,
            seed: '42',
            validation_resolution: '512,768x512',
        });
        window.ApiClient.fetch.mockResolvedValue({
            ok: true,
            json: async () => ({ session_id: 'session-one', status: 'loading' }),
        });
        jest.spyOn(manager, 'scheduleInferenceStatusPoll').mockImplementation(() => {});

        await manager.startInference();

        const [url, options] = window.ApiClient.fetch.mock.calls[0];
        expect(url).toBe('/api/checkpoints/inference/start');
        expect(JSON.parse(options.body)).toEqual(expect.objectContaining({
            environment: 'test-environment',
            checkpoint_names: ['checkpoint-100'],
            custom_prompts: ['a custom prompt'],
            filename_style: 'content-hash',
            keep_loaded: true,
            streaming_preview: true,
            idle_timeout_minutes: 9,
            settings: { seed: 42, validation_resolution: '512,768x512' },
        }));
        expect(manager.inference.session.session_id).toBe('session-one');
    });

    test('shows the newest streaming or completed output in setup', () => {
        manager.inference.session = {
            streaming_preview: true,
            preview: {
                prompt: 'preview',
                streaming: true,
                updated_at: '2026-01-01T00:00:02Z',
            },
            latest_output: {
                prompt: 'completed',
                streaming: false,
                created_at: '2026-01-01T00:00:01Z',
            },
        };
        expect(manager.inferenceDisplayOutput().prompt).toBe('preview');

        manager.inference.session.latest_output.created_at = '2026-01-01T00:00:03Z';
        expect(manager.inferenceDisplayOutput().prompt).toBe('completed');

        manager.inference.session.streaming_preview = false;
        manager.inference.session.latest_output = null;
        expect(manager.inferenceDisplayOutput()).toBeNull();
    });

    test('loads active environment inference defaults without replacing user overrides', async () => {
        window.ApiClient.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    configured_prompt: null,
                    configured_user_library: null,
                    user_libraries: [],
                    inference_defaults: {
                        num_inference_steps: 24,
                        guidance_scale: 5.5,
                        validation_resolution: '512,768x512',
                    },
                }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    configured_prompt: null,
                    configured_user_library: null,
                    user_libraries: [],
                    inference_defaults: {
                        num_inference_steps: 30,
                        guidance_scale: 6.5,
                        validation_resolution: '1024x1024',
                    },
                }),
            });

        await manager.loadInferencePromptSources();
        expect(manager.inference.form.num_inference_steps).toBe(24);
        expect(manager.inference.form.guidance_scale).toBe(5.5);
        expect(manager.inference.form.validation_resolution).toBe('512,768x512');

        manager.inference.form.num_inference_steps = '12';
        await manager.loadInferencePromptSources();

        expect(manager.inference.form.num_inference_steps).toBe('12');
        expect(manager.inference.form.guidance_scale).toBe(6.5);
        expect(manager.inference.form.validation_resolution).toBe('1024x1024');
    });

    test('loads and saves persisted inference options', async () => {
        window.ApiClient.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ filename_style: 'prompt', keep_loaded: true, streaming_preview: true }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({
                    filename_style: 'content-hash',
                    keep_loaded: false,
                    streaming_preview: true,
                }),
            });

        await manager.loadInferenceSettings();
        expect(manager.inference.form.filename_style).toBe('prompt');
        expect(manager.inference.form.keep_loaded).toBe(true);
        expect(manager.inference.form.streaming_preview).toBe(true);

        manager.inference.form.filename_style = 'content-hash';
        manager.inference.form.keep_loaded = false;
        await manager.saveInferenceSettings();

        expect(window.ApiClient.fetch).toHaveBeenLastCalledWith(
            '/api/webui/ui-state/checkpoint-inference',
            expect.objectContaining({
                method: 'POST',
                body: JSON.stringify({
                    filename_style: 'content-hash',
                    keep_loaded: false,
                    streaming_preview: true,
                }),
            })
        );
    });

    test('selects and deletes inference history samples', async () => {
        manager.inference.history = [
            { media_path: 'session-one/checkpoint-100/one.png' },
            { media_path: 'session-one/checkpoint-100/two.png' },
        ];
        manager.inference.historyTotal = 2;
        global.confirm = jest.fn(() => true);
        window.ApiClient.fetch
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ deleted_count: 2 }),
            })
            .mockResolvedValueOnce({
                ok: true,
                json: async () => ({ items: [], page: 1, total: 0 }),
            });

        manager.toggleInferenceHistoryPage();
        expect(manager.inference.historySelection).toEqual([
            'session-one/checkpoint-100/one.png',
            'session-one/checkpoint-100/two.png',
        ]);

        await manager.deleteInferenceHistorySelection();

        expect(global.confirm).toHaveBeenCalledWith('Delete 2 selected generations? This cannot be undone.');
        expect(window.ApiClient.fetch).toHaveBeenNthCalledWith(
            1,
            '/api/checkpoints/inference/history',
            expect.objectContaining({
                method: 'DELETE',
                body: JSON.stringify({
                    environment: 'test-environment',
                    media_paths: [
                        'session-one/checkpoint-100/one.png',
                        'session-one/checkpoint-100/two.png',
                    ],
                }),
            })
        );
        expect(manager.inference.history).toEqual([]);
        expect(manager.inference.historySelection).toEqual([]);
    });
});
