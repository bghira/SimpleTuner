window.workerGroupSelection = function () {
    return {
        workers: [],
        selected: [],
        requireMultinode: false,
        loading: false,
        error: '',
        lookup(id) {
            return this.workers.find(worker => worker.id === id);
        },
        compatible(worker) {
            if (!['idle', 'busy'].includes(worker.status) || (this.requireMultinode && !worker.multinode_gpus)) return false;
            if (!this.selected.length) return true;
            const leader = this.lookup(this.selected[0]);
            return Boolean(leader?.multinode_gpus && worker.multinode_gpus && leader.gpu === worker.gpu);
        },
        toggle(id, checked) {
            this.setSelection(checked ? [...this.selected, id] : this.selected.filter(item => item !== id));
        },
        setSelection(ids) {
            this.selected = ids;
        },
        groupSummary() {
            if (this.selected.length < 2) return '';
            const count = Math.min(...this.selected.map(id => this.lookup(id)?.multinode_gpus || 0));
            return count ? this.selected.length + ' workers · ' + count + ' GPUs per worker · ' +
                (count * this.selected.length) + ' GPUs total' : 'Review the unavailable workers.';
        },
        async loadWorkers(environment, includeOffline = false) {
            this.loading = true;
            this.error = '';
            try {
                const query = new URLSearchParams({environment, include_offline: String(includeOffline)});
                const response = await ApiClient.fetch('/api/training/launch-workers?' + query);
                if (!response.ok) throw new Error('Could not load this environment’s workers. Refresh and retry.');
                const group = await response.json();
                if ((this.$store.trainer.activeEnvironment || 'default') !== environment) return;
                this.workers = group.workers;
                return group;
            } catch (error) {
                this.error = error.message;
                window.showToast?.(this.error, 'error');
            } finally {
                this.loading = false;
            }
        },
    };
};

window.hardwareWorkerGroup = function () {
    return {
        ...window.workerGroupSelection(),
        check: null,
        checking: false,
        checkTimer: null,
        disposed: false,
        destroy() {
            this.disposed = true;
            clearTimeout(this.checkTimer);
        },
        get canCheck() {
            return this.selected.length >= 2 && this.selected.every(id => {
                const w = this.lookup(id);
                return w?.status === 'idle' && w.group_check && w.gpu === this.lookup(this.selected[0])?.gpu;
            });
        },
        async testGroup() {
            if (!this.canCheck || this.checking) return;
            this.checking = true;
            this.check = null;
            this.error = '';
            try {
                const response = await ApiClient.fetch('/api/training/worker-group-checks', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({worker_ids: this.selected, port: Number(this.setting('main_process_port') || 29500)}),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Invalid group check settings.');
                await this.readCheck(data.job_id);
            } catch (error) {
                this.error = error.message;
                this.checking = false;
            }
        },
        async readCheck(id) {
            if (this.disposed) return;
            try {
                const response = await ApiClient.fetch('/api/training/worker-group-checks/' + encodeURIComponent(id));
                if (!response.ok) throw new Error('Could not read the NCCL result.');
                this.check = await response.json();
                this.checking = !this.check.terminal;
                if (this.checking && !this.disposed) this.checkTimer = setTimeout(() => this.readCheck(id), 1000);
            } catch (error) {
                this.error = error.message;
                this.checking = false;
            }
        },
        async refresh() {
            const environment = this.$store.trainer.activeEnvironment || 'default';
            const group = await this.loadWorkers(environment, true);
            if (!group) return;
            const draft = this.$store.trainer.workerGroupDraft;
            this.selected = draft?.environment === environment ? [...draft.ids] : [...group.default_worker_ids];
        },
        requireMultinode: true,
        setSelection(ids) {
            this.selected = ids;
            const trainer = this.$store.trainer;
            trainer.workerGroupDraft = {environment: trainer.activeEnvironment || 'default', ids: [...ids]};
            trainer.markFormDirty();
        },
        move(index, direction) {
            const ids = [...this.selected];
            const other = index + direction;
            [ids[index], ids[other]] = [ids[other], ids[index]];
            this.setSelection(ids);
        },
        setting(key) {
            const trainer = this.$store.trainer;
            const canonical = trainer.canonicalizeKey(key);
            return trainer.formValueStore?.[canonical]?.value ?? trainer.configValues?.[canonical] ?? trainer.activeEnvironmentConfig?.[canonical];
        },
    };
};
