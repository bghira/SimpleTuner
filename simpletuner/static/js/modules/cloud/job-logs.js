/**
 * Cloud Dashboard - Job Logs Module
 *
 * Handles log viewing, filtering, formatting, and download.
 */

window.cloudLogMethods = {
    async viewLogs(jobId) {
        this.logsModal.open = true;
        this.logsModal.jobId = jobId;
        this.logsModal.logs = '';
        this.logsModal.loading = true;
        this.logsModal.searchQuery = '';
        this.logsModal.levelFilter = '';
        this.logsModal.filteredLogsHtml = '';
        this.logsModal.matchCount = 0;
        this.logsModal.lineCount = 0;
        this.logsModal.streaming = false;

        try {
            const response = await fetch(`/api/cloud/jobs/${jobId}/logs`);
            if (response.ok) {
                const data = await response.json();
                this.logsModal.logs = data.logs || '(No logs available)';
                this.updateLogStats();
                this.filterLogs();
            }
        } catch (error) {
            this.logsModal.logs = `Error: ${error.message}`;
        }
        this.logsModal.loading = false;

        if (this.logsModal.autoScroll) {
            this.$nextTick(() => this.scrollLogsToBottom());
        }
    },

    async refreshLogs() {
        if (!this.logsModal.jobId) return;
        this.logsModal.loading = true;
        try {
            const response = await fetch(`/api/cloud/jobs/${this.logsModal.jobId}/logs`);
            if (response.ok) {
                const data = await response.json();
                this.logsModal.logs = data.logs || '(No logs available)';
                this.updateLogStats();
                this.filterLogs();
                if (this.logsModal.autoScroll) {
                    this.$nextTick(() => this.scrollLogsToBottom());
                }
            }
        } catch (error) {
            console.error('Failed to refresh logs:', error);
        }
        this.logsModal.loading = false;
    },

    closeLogsModal() {
        this.logsModal.open = false;
        this.logsModal.jobId = null;
        this.logsModal.logs = '';
        this.logsModal.filteredLogsHtml = '';
        this.logsModal.streaming = false;
        if (this.logsModal.eventSource) {
            this.logsModal.eventSource.close();
            this.logsModal.eventSource = null;
        }
    },

    filterLogs() {
        const logs = this.logsModal.logs || '';
        const query = this.logsModal.searchQuery.toLowerCase();
        const levelFilter = this.logsModal.levelFilter;

        if (!query && !levelFilter) {
            this.logsModal.filteredLogsHtml = this.formatLogs(logs);
            this.logsModal.matchCount = logs.split('\n').length;
            return;
        }

        const lines = logs.split('\n');
        let matchCount = 0;
        const filteredLines = lines.filter(line => {
            if (levelFilter) {
                const hasLevel = line.toUpperCase().includes(levelFilter);
                if (!hasLevel) return false;
            }
            if (query) {
                const matches = line.toLowerCase().includes(query);
                if (!matches) return false;
            }
            matchCount++;
            return true;
        });

        this.logsModal.matchCount = matchCount;
        const filteredText = filteredLines.join('\n');
        this.logsModal.filteredLogsHtml = this.formatLogs(filteredText, query);
    },

    formatLogs(text, highlight = '') {
        let html = this.escapeHtml(text);

        html = html.replace(/^(.*\bERROR\b.*)$/gm, '<span class="log-error">$1</span>');
        html = html.replace(/^(.*\bWARN(?:ING)?\b.*)$/gm, '<span class="log-warn">$1</span>');
        html = html.replace(/^(.*\bINFO\b.*)$/gm, '<span class="log-info">$1</span>');
        html = html.replace(/^(.*\bDEBUG\b.*)$/gm, '<span class="log-debug">$1</span>');

        if (highlight) {
            const regex = new RegExp(`(${this.escapeRegex(highlight)})`, 'gi');
            html = html.replace(regex, '<span class="log-highlight">$1</span>');
        }

        html = html.replace(/(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)/g,
            '<span class="log-timestamp">$1</span>');

        return html;
    },

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    escapeRegex(str) {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    },

    updateLogStats() {
        const logs = this.logsModal.logs || '';
        this.logsModal.lineCount = logs ? logs.split('\n').length : 0;
    },

    scrollLogsToBottom() {
        const viewer = this.$refs.logViewer;
        if (viewer) {
            viewer.scrollTop = viewer.scrollHeight;
        }
    },

    downloadLogs() {
        const logs = this.logsModal.logs;
        if (!logs) return;

        const blob = new Blob([logs], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `logs-${this.logsModal.jobId}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },
};
