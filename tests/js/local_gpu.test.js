/**
 * Tests for local GPU concurrency management.
 *
 * Tests GPU status display, concurrency settings, and local job submission.
 * Uses consolidated endpoints: /api/system/status, /api/queue/stats, /api/queue/concurrency, /api/queue/submit
 */

describe('GPU Status via System API', () => {
    beforeEach(() => {
        fetch.mockReset();
        fetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({}),
        });
    });

    describe('GET /api/system/status?include_allocation=true', () => {
        test('fetches system status with GPU allocation', async () => {
            const mockStatus = {
                timestamp: 1704067200.0,
                load_avg_5min: 2.5,
                memory_percent: 45.2,
                gpus: [],
                gpu_inventory: {
                    backend: 'cuda',
                    count: 4,
                    capabilities: {},
                },
                gpu_allocation: {
                    allocated_gpus: [0, 1],
                    available_gpus: [2, 3],
                    running_local_jobs: 1,
                    devices: [
                        { index: 0, name: 'GPU 0', memory_gb: 24, allocated: true, job_id: 'job-123' },
                        { index: 1, name: 'GPU 1', memory_gb: 24, allocated: true, job_id: 'job-123' },
                        { index: 2, name: 'GPU 2', memory_gb: 24, allocated: false, job_id: null },
                        { index: 3, name: 'GPU 3', memory_gb: 24, allocated: false, job_id: null },
                    ],
                },
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStatus),
            });

            const response = await fetch('/api/system/status?include_allocation=true');
            const data = await response.json();

            expect(fetch).toHaveBeenCalledWith('/api/system/status?include_allocation=true');
            expect(data.gpu_inventory.count).toBe(4);
            expect(data.gpu_allocation.allocated_gpus).toEqual([0, 1]);
            expect(data.gpu_allocation.available_gpus).toEqual([2, 3]);
            expect(data.gpu_allocation.running_local_jobs).toBe(1);
            expect(data.gpu_allocation.devices).toHaveLength(4);
        });

        test('status without allocation when not requested', async () => {
            const mockStatus = {
                timestamp: 1704067200.0,
                load_avg_5min: 2.5,
                memory_percent: 45.2,
                gpus: [],
                gpu_inventory: {
                    backend: 'cuda',
                    count: 4,
                    capabilities: {},
                },
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStatus),
            });

            const response = await fetch('/api/system/status');
            const data = await response.json();

            expect(data.gpu_inventory.count).toBe(4);
            expect(data.gpu_allocation).toBeUndefined();
        });
    });

    describe('GET /api/queue/stats', () => {
        test('fetches queue stats with local GPU info', async () => {
            const mockStats = {
                queue_depth: 3,
                running: 2,
                max_concurrent: 5,
                local_gpu_max_concurrent: 6,
                local_job_max_concurrent: 2,
                local: {
                    running_jobs: 1,
                    pending_jobs: 0,
                    allocated_gpus: [0, 1],
                    available_gpus: [2, 3],
                    total_gpus: 4,
                    max_concurrent_gpus: 6,
                    max_concurrent_jobs: 2,
                },
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStats),
            });

            const response = await fetch('/api/queue/stats');
            const data = await response.json();

            expect(data.local.running_jobs).toBe(1);
            expect(data.local.allocated_gpus).toEqual([0, 1]);
            expect(data.local.available_gpus).toEqual([2, 3]);
            expect(data.local_gpu_max_concurrent).toBe(6);
            expect(data.local_job_max_concurrent).toBe(2);
        });

        test('handles null local GPU limit (unlimited)', async () => {
            const mockStats = {
                queue_depth: 0,
                running: 0,
                max_concurrent: 5,
                local_gpu_max_concurrent: null,
                local_job_max_concurrent: 1,
                local: {
                    running_jobs: 0,
                    pending_jobs: 0,
                    allocated_gpus: [],
                    available_gpus: [0, 1, 2, 3],
                    total_gpus: 4,
                    max_concurrent_gpus: null,
                    max_concurrent_jobs: 1,
                },
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockStats),
            });

            const response = await fetch('/api/queue/stats');
            const data = await response.json();

            expect(data.local_gpu_max_concurrent).toBeNull();
            expect(data.local.max_concurrent_gpus).toBeNull();
        });
    });
});

describe('Queue Concurrency Settings', () => {
    beforeEach(() => {
        fetch.mockReset();
    });

    describe('POST /api/queue/concurrency', () => {
        test('updates local GPU concurrency limits', async () => {
            const newLimits = {
                local_gpu_max_concurrent: 6,
                local_job_max_concurrent: 3,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    success: true,
                    max_concurrent: 5,
                    user_max_concurrent: 2,
                    ...newLimits,
                }),
            });

            const response = await fetch('/api/queue/concurrency', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newLimits),
            });

            expect(fetch).toHaveBeenCalledWith(
                '/api/queue/concurrency',
                expect.objectContaining({
                    method: 'POST',
                    body: JSON.stringify(newLimits),
                })
            );

            const data = await response.json();
            expect(data.success).toBe(true);
            expect(data.local_gpu_max_concurrent).toBe(6);
            expect(data.local_job_max_concurrent).toBe(3);
        });

        test('updates mixed cloud and local limits', async () => {
            const mixedUpdate = {
                max_concurrent: 10,
                local_gpu_max_concurrent: 8,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve({
                    success: true,
                    max_concurrent: 10,
                    user_max_concurrent: 2,
                    local_gpu_max_concurrent: 8,
                    local_job_max_concurrent: 1,
                }),
            });

            const response = await fetch('/api/queue/concurrency', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(mixedUpdate),
            });

            const data = await response.json();
            expect(data.max_concurrent).toBe(10);
            expect(data.local_gpu_max_concurrent).toBe(8);
        });
    });
});

describe('Local Job Submission API', () => {
    beforeEach(() => {
        fetch.mockReset();
    });

    describe('POST /api/queue/submit', () => {
        test('submits job successfully when GPUs available', async () => {
            const mockResponse = {
                success: true,
                job_id: 'abc123',
                status: 'running',
                allocated_gpus: [0, 1],
                queue_position: null,
                reason: null,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            const response = await fetch('/api/queue/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config_name: 'test-config',
                    no_wait: false,
                    any_gpu: false,
                }),
            });

            const data = await response.json();
            expect(data.success).toBe(true);
            expect(data.status).toBe('running');
            expect(data.allocated_gpus).toEqual([0, 1]);
        });

        test('job is queued when GPUs unavailable', async () => {
            const mockResponse = {
                success: true,
                job_id: 'def456',
                status: 'queued',
                allocated_gpus: null,
                queue_position: 3,
                reason: 'Waiting for 2 GPU(s) to become available',
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            const response = await fetch('/api/queue/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config_name: 'test-config',
                }),
            });

            const data = await response.json();
            expect(data.success).toBe(true);
            expect(data.status).toBe('queued');
            expect(data.queue_position).toBe(3);
            expect(data.reason).toContain('GPU');
        });

        test('job is rejected with no_wait when GPUs unavailable', async () => {
            const mockResponse = {
                success: false,
                job_id: null,
                status: 'rejected',
                error: 'Required GPUs unavailable and --no-wait specified',
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            const response = await fetch('/api/queue/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config_name: 'test-config',
                    no_wait: true,
                }),
            });

            const data = await response.json();
            expect(data.success).toBe(false);
            expect(data.status).toBe('rejected');
        });

        test('any_gpu parameter uses available GPUs', async () => {
            const mockResponse = {
                success: true,
                job_id: 'ghi789',
                status: 'running',
                allocated_gpus: [2, 3],  // Different from configured
                queue_position: null,
            };

            fetch.mockResolvedValueOnce({
                ok: true,
                json: () => Promise.resolve(mockResponse),
            });

            const response = await fetch('/api/queue/submit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    config_name: 'test-config',
                    any_gpu: true,
                }),
            });

            const data = await response.json();
            expect(data.success).toBe(true);
            expect(data.allocated_gpus).toEqual([2, 3]);
        });
    });
});

describe('GPU Status Display Helpers', () => {
    test('formats GPU list for display', () => {
        function formatGPUList(gpus) {
            if (!gpus || gpus.length === 0) return 'None';
            return gpus.join(', ');
        }

        expect(formatGPUList([0, 1, 2])).toBe('0, 1, 2');
        expect(formatGPUList([0])).toBe('0');
        expect(formatGPUList([])).toBe('None');
        expect(formatGPUList(null)).toBe('None');
    });

    test('calculates GPU utilization percentage', () => {
        function calculateUtilization(allocated, total) {
            if (total === 0) return 0;
            return Math.round((allocated.length / total) * 100);
        }

        expect(calculateUtilization([0, 1], 4)).toBe(50);
        expect(calculateUtilization([0, 1, 2, 3], 4)).toBe(100);
        expect(calculateUtilization([], 4)).toBe(0);
        expect(calculateUtilization([], 0)).toBe(0);
    });

    test('determines GPU availability status', () => {
        function getAvailabilityStatus(available, required) {
            if (available >= required) {
                return { canStart: true, message: 'GPUs available' };
            }
            return {
                canStart: false,
                message: `Need ${required} GPUs, only ${available} available`,
            };
        }

        expect(getAvailabilityStatus(4, 2)).toEqual({
            canStart: true,
            message: 'GPUs available',
        });

        expect(getAvailabilityStatus(1, 4)).toEqual({
            canStart: false,
            message: 'Need 4 GPUs, only 1 available',
        });
    });

    test('formats backend name for display', () => {
        function formatBackend(backend) {
            const names = {
                cuda: 'NVIDIA CUDA',
                rocm: 'AMD ROCm',
                mps: 'Apple Metal',
                cpu: 'CPU Only',
            };
            return names[backend] || backend;
        }

        expect(formatBackend('cuda')).toBe('NVIDIA CUDA');
        expect(formatBackend('rocm')).toBe('AMD ROCm');
        expect(formatBackend('mps')).toBe('Apple Metal');
        expect(formatBackend('cpu')).toBe('CPU Only');
        expect(formatBackend('unknown')).toBe('unknown');
    });
});

describe('Local GPU Concurrency UI State', () => {
    let context;

    beforeEach(() => {
        fetch.mockReset();
        fetch.mockResolvedValue({
            ok: true,
            json: () => Promise.resolve({}),
        });

        context = {
            localGPUStatus: null,
            localConcurrencySettings: {
                local_gpu_max_concurrent: null,
                local_job_max_concurrent: 1,
            },
            loadingGPUStatus: false,
            savingConcurrency: false,
        };
    });

    test('refreshLocalGPUStatus updates state from system status', async () => {
        const mockStatus = {
            timestamp: 1704067200.0,
            gpu_allocation: {
                allocated_gpus: [0],
                available_gpus: [1, 2, 3],
                running_local_jobs: 1,
                devices: [],
            },
        };

        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve(mockStatus),
        });

        async function refreshLocalGPUStatus() {
            if (context.loadingGPUStatus) return;
            context.loadingGPUStatus = true;
            try {
                const response = await fetch('/api/system/status?include_allocation=true');
                const data = await response.json();
                context.localGPUStatus = data.gpu_allocation;
            } finally {
                context.loadingGPUStatus = false;
            }
        }

        await refreshLocalGPUStatus.call(context);

        expect(context.localGPUStatus).toEqual(mockStatus.gpu_allocation);
        expect(context.loadingGPUStatus).toBe(false);
    });

    test('updateLocalConcurrency sends settings to API', async () => {
        const newSettings = {
            local_gpu_max_concurrent: 4,
            local_job_max_concurrent: 2,
        };

        fetch.mockResolvedValueOnce({
            ok: true,
            json: () => Promise.resolve({ success: true, ...newSettings }),
        });

        window.showToast = jest.fn();

        async function updateLocalConcurrency(settings) {
            if (context.savingConcurrency) return;
            context.savingConcurrency = true;
            try {
                const response = await fetch('/api/queue/concurrency', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(settings),
                });
                const data = await response.json();
                if (data.success) {
                    context.localConcurrencySettings = {
                        local_gpu_max_concurrent: data.local_gpu_max_concurrent,
                        local_job_max_concurrent: data.local_job_max_concurrent,
                    };
                    window.showToast('GPU concurrency settings updated', 'success');
                }
            } finally {
                context.savingConcurrency = false;
            }
        }

        await updateLocalConcurrency.call(context, newSettings);

        expect(context.localConcurrencySettings).toEqual(newSettings);
        expect(window.showToast).toHaveBeenCalledWith(
            'GPU concurrency settings updated',
            'success'
        );

        delete window.showToast;
    });

    test('prevents concurrent updates', async () => {
        context.savingConcurrency = true;

        async function updateLocalConcurrency(settings) {
            if (context.savingConcurrency) return;
            await fetch('/api/queue/concurrency', {
                method: 'POST',
                body: JSON.stringify(settings),
            });
        }

        await updateLocalConcurrency.call(context, { local_gpu_max_concurrent: 4 });

        expect(fetch).not.toHaveBeenCalled();
    });
});

describe('Dry Run GPU Display', () => {
    test('displays GPU requirements section', () => {
        function formatDryRunGPUInfo(requirements, status) {
            const lines = [];
            lines.push('GPU Requirements:');
            lines.push(`  num_processes: ${requirements.num_processes}`);
            if (requirements.device_ids) {
                lines.push(`  configured_gpus: [${requirements.device_ids.join(', ')}]`);
            }
            lines.push('');
            lines.push('Available GPUs:');
            status.devices.forEach((gpu) => {
                const state = gpu.allocated
                    ? `[allocated - job ${gpu.job_id}]`
                    : '[available]';
                lines.push(`  GPU ${gpu.index}: ${gpu.name} ${state}`);
            });
            return lines.join('\n');
        }

        const requirements = {
            num_processes: 2,
            device_ids: [0, 1],
        };

        const status = {
            devices: [
                { index: 0, name: 'A100', allocated: false, job_id: null },
                { index: 1, name: 'A100', allocated: true, job_id: 'abc123' },
            ],
        };

        const output = formatDryRunGPUInfo(requirements, status);

        expect(output).toContain('num_processes: 2');
        expect(output).toContain('configured_gpus: [0, 1]');
        expect(output).toContain('GPU 0: A100 [available]');
        expect(output).toContain('GPU 1: A100 [allocated - job abc123]');
    });

    test('shows job would be queued when insufficient GPUs', () => {
        function getDryRunStatus(availableCount, requiredCount) {
            if (availableCount >= requiredCount) {
                return {
                    action: 'start',
                    message: `Job would start immediately using ${requiredCount} GPU(s)`,
                };
            }
            return {
                action: 'queue',
                message: `Job would be queued (need ${requiredCount} GPUs, only ${availableCount} available)`,
            };
        }

        expect(getDryRunStatus(4, 2)).toEqual({
            action: 'start',
            message: 'Job would start immediately using 2 GPU(s)',
        });

        expect(getDryRunStatus(1, 4)).toEqual({
            action: 'queue',
            message: 'Job would be queued (need 4 GPUs, only 1 available)',
        });
    });
});
