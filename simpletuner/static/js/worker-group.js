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
