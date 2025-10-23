// Event handling system with improved CORS and connection management
class EventHandler {
    constructor() {
        this.lastEventIndex = 0;
        this.eventList = document.getElementById('eventList');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.callbackUrl = null; // Will be set after config detection
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 30; // More attempts with shorter delays
        this.reconnectDelay = 500; // Start with 500ms for faster reconnection
        this.maxReconnectDelay = 5000; // Max 5 seconds
        this.abortController = null;
        this.fastReconnectThreshold = 5; // First 5 attempts are fast
        this.reconnectTimeout = null; // Store timeout for cancellation
        this.pollTimeout = null; // Store poll timeout for cleanup
        this.isReconnecting = false; // Track reconnection state
        this.hasConnectedBefore = false; // Track if we've ever connected
        this.isFetching = false; // Prevent concurrent fetch loops
        this.healthCheckInterval = null;
        this.healthCheckInFlight = false;
        this.websocketReconnectTimeout = null;
        this.isActive = false;
        this.sseRegistered = false;

        // WebSocket support
        this.websocket = null;
        this.useWebSocket = true; // Enable WebSocket by default

        // Training state tracking
        this.currentStep = 0;
        this.totalSteps = 0;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.currentLoss = 0;
        this.learningRate = 0;
        this.lastKnownJobId = null;
        this.lastReportedStatus = null;

        // Dynamic row limit tracking
        this.currentMaxEvents = 500;
        this.resizeObserver = null;

        this.init();
    }

    async init() {
        // Wait for server configuration to be ready
        await window.ServerConfig.waitForReady();

        // Set the callback URL based on detected configuration
        this.callbackUrl = window.ServerConfig.callbackUrl;
        // EventHandler configured with callback URL

        // Set up resize observer for dynamic row calculation
        this.setupResizeObserver();

        // Subscribe to SSE notifications once available
        this.subscribeToSSE();

        await this.startFetching();
        // Clean up on page unload
        window.addEventListener('beforeunload', () => this.cleanup());
    }

    subscribeToSSE() {
        if (this.sseRegistered) {
            return;
        }

        const attemptRegistration = () => {
            if (this.sseRegistered) {
                return;
            }

            if (window.sseManager && typeof window.sseManager.addEventListener === 'function') {
                window.sseManager.addEventListener('notification', (payload) => {
                    if (!payload) {
                        return;
                    }

                    const message = payload.message || payload.headline || payload.body;
                    if (!message) {
                        return;
                    }

                    const severity = String(payload.severity || '').toLowerCase();
                    let messageType = 'info';
                    if (severity === 'error' || severity === 'critical' || severity === 'fatal') {
                        messageType = 'fatal_error';
                    } else if (severity === 'warning') {
                        messageType = 'warning';
                    } else if (severity === 'success') {
                        messageType = 'success';
                    }

                    this.updateEventList([
                        {
                            timestamp: payload.timestamp || new Date().toISOString(),
                            message_type: messageType,
                            message,
                        },
                    ]);
                });

                this.sseRegistered = true;
            } else {
                setTimeout(attemptRegistration, 500);
            }
        };

        attemptRegistration();
    }

    setupResizeObserver() {
        if (!this.eventList) return;

        // Create resize observer to monitor container size changes
        this.resizeObserver = new ResizeObserver((entries) => {
            for (const entry of entries) {
                if (entry.target === this.eventList) {
                    // Recalculate max events when container is resized
                    const newMaxEvents = this.calculateDynamicMaxEvents();
                    if (newMaxEvents !== this.currentMaxEvents) {
                        this.currentMaxEvents = newMaxEvents;
                        // If we have more events than the new limit, remove excess events
                        while (this.eventList.children.length > newMaxEvents) {
                            this.eventList.removeChild(this.eventList.lastChild);
                        }
                    }
                }
            }
        });

        // Start observing the event list container
        this.resizeObserver.observe(this.eventList);

        // Also observe the event dock body to catch resize changes
        const eventDockBody = this.eventList.closest('.event-dock-body');
        if (eventDockBody) {
            this.resizeObserver.observe(eventDockBody);
        }
    }

    async checkServerHealth() {
        try {
            const response = await ApiClient.fetch("/health", {
                method: 'GET',
                mode: 'cors',
                credentials: 'omit',
                cache: 'no-cache',
                signal: AbortSignal.timeout(2000) // 2 second timeout
            }, { forceApi: true });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    async fetchBroadcastEvents() {
        if (!this.isActive || this.isFetching) {
            return;
        }
        this.isFetching = true;
        try {
            // Create new abort controller for this request
            this.abortController = new AbortController();

            // Try WebSocket connection first
            if (this.useWebSocket && !this.websocket) {
                this.connectWebSocket();
                return; // Let WebSocket handle events
            }

            // Fall back to polling if WebSocket not available
            const response = await ApiClient.fetch(
                `/api/training/events?since_index=${this.lastEventIndex}`,
                {
                    signal: this.abortController.signal,
                    headers: {
                        'Accept': 'application/json',
                    },
                    credentials: 'include', // Include credentials for CORS
                },
                { forceCallback: true }
            );

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (!this.isActive) {
                return;
            }

            // Reset reconnect attempts on successful connection
            if (!this.isConnected) {
                this.isReconnecting = false; // Clear reconnecting state
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                // Connected to event server
                this.handleConnectionStateChange(true);
            } else {
                // Update connection status without message
                this.setConnectionStatus(true, null, false);
            }

            // Process process keeper events
            if (data.events && data.events.length > 0) {
                this.processProcessKeeperEvents(data.events);
            }
            this.lastEventIndex = data.next_index || this.lastEventIndex + (data.events ? data.events.length : 0);

            // Update connection status (without showing message on every poll)
            this.setConnectionStatus(true, null, false);

            // Continue fetching
            if (this.pollTimeout) {
                clearTimeout(this.pollTimeout);
            }
            this.pollTimeout = setTimeout(() => {
                this.pollTimeout = null;
                if (!this.isActive) {
                    return;
                }
                this.fetchBroadcastEvents();
            }, 1000); // Poll every second
        } catch (error) {
            if (error.name === 'AbortError') {
                // Fetch aborted
                return;
            }

            console.error('Error fetching events:', error);

            if (!this.isActive) {
                return;
            }

            // Handle disconnection
            this.handleConnectionStateChange(false);

            // Start reconnection mode
            if (!this.isReconnecting) {
                this.isReconnecting = true;
                this.startHealthCheckPolling(); // Start polling for server availability
            }

            // Show reconnecting status with attempt info
            const reconnectMsg = this.reconnectAttempts <= this.fastReconnectThreshold ?
                'Reconnecting...' :
                `Reconnecting (attempt ${this.reconnectAttempts})...`;
            this.setConnectionStatus(false, reconnectMsg, false);

            // Implement smart reconnection strategy
            this.reconnectAttempts++;
            if (this.reconnectAttempts <= this.maxReconnectAttempts) {
                let delay;
                if (this.reconnectAttempts <= this.fastReconnectThreshold) {
                    // Fast reconnection for first few attempts (500ms)
                    delay = this.reconnectDelay;
                    // Fast reconnection attempt
                } else {
                    // Slower exponential backoff after fast attempts
                    const backoffAttempt = this.reconnectAttempts - this.fastReconnectThreshold;
                    delay = Math.min(1000 * Math.pow(1.5, backoffAttempt - 1), this.maxReconnectDelay);
                    // Reconnecting with backoff
                }

                // Clear any existing reconnect timeout
                if (this.reconnectTimeout) {
                    clearTimeout(this.reconnectTimeout);
                }

                // Schedule reconnection attempt
                this.reconnectTimeout = setTimeout(() => {
                    this.reconnectTimeout = null;
                    // Only proceed if still in reconnecting state
                    if (this.isReconnecting && this.isActive) {
                        this.fetchBroadcastEvents();
                    }
                }, delay);
            } else {
                console.error('Max reconnection attempts reached');
                this.setConnectionStatus(false, 'Max reconnection attempts reached', false);
                // Reset and try again after a longer delay
                this.reconnectTimeout = setTimeout(() => {
                    this.reconnectTimeout = null;
                    this.reconnectAttempts = 0;
                    if (this.isActive) {
                        this.fetchBroadcastEvents();
                    }
                }, 10000); // Try again after 10 seconds
            }
        } finally {
            this.isFetching = false;
            if (this.abortController) {
                this.abortController = null;
            }
        }
    }

    updateEventList(events) {
        if (!this.eventList) return;

        // Calculate dynamic max events based on container height
        const maxEvents = this.calculateDynamicMaxEvents();

        events.forEach(event => {
            // Parse structured data for specific event types
            this.parseStructuredData(event);

            // Skip events without messages (pure data events)
            if (!event.message && !this.shouldDisplayEvent(event)) return;

            // Remove oldest event if we've reached the limit (rotation system)
            if (this.eventList.children.length >= maxEvents) {
                this.eventList.removeChild(this.eventList.lastChild);
            }

            const eventItem = document.createElement('div');
            eventItem.className = 'event-item';

            // Apply specific styles based on event type
            switch (event.message_type) {
                case 'error':
                case 'fatal_error':
                case 'exit':
                    eventItem.classList.add('event-item-error');
                    break;
                case 'train':
                case 'init_resume_checkpoint':
                case '_train_initial_msg':
                    eventItem.classList.add('event-item-train');
                    break;
                case 'info':
                    eventItem.classList.add('event-item-info');
                    break;
                case 'validation':
                case 'checkpoint':
                    eventItem.classList.add('event-item-success');
                    break;
                default:
                    eventItem.classList.add('event-item-default');
                    break;
            }

            // Format timestamp
            const timestamp = event.timestamp ?
                new Date(event.timestamp).toLocaleTimeString() :
                new Date().toLocaleTimeString();

            // Create structured content with enhanced data display
            const messageContent = this.formatEnhancedMessage(event);

            eventItem.innerHTML = `
                <div class="event-header">
                    <span class="timestamp">[${timestamp}]</span>
                    <span class="event-type">${this.formatEventType(event.message_type)}</span>
                </div>
                <div class="event-message">${messageContent}</div>
            `;

            if (this.eventList.firstChild) {
                this.eventList.insertBefore(eventItem, this.eventList.firstChild);
            } else {
                this.eventList.appendChild(eventItem);
            }

            // Handle special events
            this.handleSpecialEvents(event);
        });

        // Auto-scroll to latest only if user is near the top already
        const nearTop = this.eventList.scrollTop <= 5;
        if (nearTop) {
            this.eventList.scrollTop = 0;
        }

        if (this.eventList.children.length > 0) {
            this.eventList.dataset.hydrated = 'true';
        }
    }

    calculateDynamicMaxEvents() {
        if (!this.eventList) return 500; // Fallback to original limit

        // Get the available height of the event list container
        const containerHeight = this.eventList.clientHeight;
        if (containerHeight <= 0) return 500; // Fallback if container not visible

        // Calculate approximate event item height (including padding and margins)
        const eventItemHeight = 30; // Approximate height in pixels for each event item

        // Calculate how many events can fit in the available space
        // Reserve some space for padding and scrollbar
        const availableHeight = containerHeight - 20; // Reserve 20px for padding/scrollbar
        const calculatedMaxEvents = Math.floor(availableHeight / eventItemHeight);

        // Ensure we have a reasonable range (minimum 3, maximum 1000)
        const dynamicMaxEvents = Math.max(3, Math.min(1000, calculatedMaxEvents));

        return dynamicMaxEvents;
    }

    formatEventType(type) {
        // Format event type for display
        return type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    formatMessage(message) {
        // Add any message formatting logic here
        // For example, highlighting numbers, paths, etc.
        return message;
    }

    parseStructuredData(event) {
        // Parse and store training state from structured events
        switch (event.message_type) {
            case 'configure_webhook':
                // Reset training state when new training starts
                this.resetTrainingState();
                break;

            case 'train':
                if (event.state) {
                    this.currentStep = event.state.global_step || this.currentStep;
                    this.currentEpoch = event.state.current_epoch || this.currentEpoch;
                }
                if (event.loss !== undefined) {
                    this.currentLoss = event.loss;
                }
                if (event.learning_rate !== undefined) {
                    this.learningRate = event.learning_rate;
                }
                if (event.final_epoch !== undefined) {
                    this.totalEpochs = event.final_epoch;
                }
                this.updateProgressDisplay();
                break;

            case 'progress_update':
            case 'lifecycle.stage': {
                const stageInfo = event.stage || event.message;
                if (stageInfo) {
                    const progressInfo = stageInfo.progress || stageInfo;
                    const progressType = stageInfo.progress_type || stageInfo.key || stageInfo.label || 'progress';
                    let currentValue = Number(progressInfo.current ?? progressInfo.current_estimated_index);
                    let totalValue = Number(progressInfo.total ?? progressInfo.total_elements);
                    if (!Number.isFinite(currentValue) || !Number.isFinite(totalValue) || totalValue <= 0) {
                        const percentValue = Number(progressInfo.percent ?? progressInfo.progress ?? 0);
                        currentValue = Number.isFinite(percentValue) ? percentValue : 0;
                        totalValue = 100;
                    }
                    this.updateProgressBar(
                        String(progressType),
                        currentValue,
                        totalValue,
                        stageInfo.readable_type || stageInfo.label || this.prettifyLabel(progressType)
                    );
                }
                break;
            }

            case 'training_config':
                if (event.total_num_steps) {
                    this.totalSteps = event.total_num_steps;
                }
                if (event.total_num_epochs) {
                    this.totalEpochs = event.total_num_epochs;
                }
                break;

            case 'training_status':
                const trainingStatus = String(event.status || '').toLowerCase();
                if (['failed', 'error', 'cancelled', 'stopped', 'completed'].includes(trainingStatus)) {
                    this.resetTrainingState();
                }
                break;

            case 'exit':
            case 'fatal_error':
                // Clear progress displays on completion/error
                this.clearProgressDisplays();
                break;
        }
    }

    resetTrainingState() {
        // Reset all training state variables
        this.currentStep = 0;
        this.totalSteps = 0;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.currentLoss = 0;
        this.learningRate = 0;

        // Clear any existing progress displays
        this.clearProgressDisplays();
    }

    clearProgressDisplays() {
        const progressContainer = document.getElementById('trainingProgress');
        if (progressContainer) {
            // Use dynamic removal instead of innerHTML swap to avoid full redraw
            while (progressContainer.firstChild) {
                progressContainer.removeChild(progressContainer.firstChild);
            }
        }

        const progressBars = document.getElementById('progressBars');
        if (progressBars) {
            // Use dynamic removal instead of innerHTML swap to avoid full redraw
            while (progressBars.firstChild) {
                progressBars.removeChild(progressBars.firstChild);
            }
        }
    }

    shouldDisplayEvent(event) {
        // Determine if an event without a message should still be displayed
        return ['train', 'validation', 'checkpoint'].includes(event.message_type);
    }

    extractJobId(event, data) {
        const candidates = [
            event?.job_id,
            data?.job_id,
            data?.jobID,
            data?.jobId,
            data?.job_id_current,
            this.lastKnownJobId,
        ];

        for (const candidate of candidates) {
            if (typeof candidate === 'string' && candidate.trim()) {
                return candidate.trim();
            }
        }
        return null;
    }

    getTrainerActionsInstance() {
        if (window.TrainerActionsInstance && typeof window.TrainerActionsInstance.updateButtonStates === 'function') {
            return window.TrainerActionsInstance;
        }
        if (window.trainerMain && window.trainerMain.actions && typeof window.trainerMain.actions.updateButtonStates === 'function') {
            return window.trainerMain.actions;
        }
        return null;
    }

    notifyTrainingState(status, data = {}, options = {}) {
        const normalizedStatus = String(status || '').toLowerCase();
        if (!normalizedStatus) {
            return;
        }

        const jobId = this.extractJobId({ job_id: data?.job_id }, data) || this.lastKnownJobId || null;

        if (jobId) {
            this.lastKnownJobId = jobId;
        }

        const terminalStatuses = new Set(['failed', 'error', 'fatal', 'fatal_error', 'cancelled', 'canceled', 'stopped', 'completed', 'terminated']);
        const activeStatuses = new Set(['running', 'starting', 'initializing', 'initialising', 'configuring']);
        const isTraining = options.forceTraining === true || activeStatuses.has(normalizedStatus);
        const shouldResetProgress = options.resetProgress === true || terminalStatuses.has(normalizedStatus);

        if (this.lastReportedStatus === normalizedStatus && !options.force && jobId === this.lastKnownJobId && !shouldResetProgress) {
            return;
        }

        this.lastReportedStatus = normalizedStatus;

        const trainerActions = this.getTrainerActionsInstance();
        if (trainerActions) {
            trainerActions.updateButtonStates(isTraining);
        }

        const body = document?.body;
        if (body) {
            body.dataset.trainingActive = isTraining ? 'true' : 'false';
        }

        const jobIdInput = document.getElementById('job_id');
        if (jobIdInput) {
            if (isTraining && jobId) {
                jobIdInput.value = jobId;
            } else if (!isTraining && (!jobId || jobIdInput.value === jobId)) {
                jobIdInput.value = '';
            }
        }

        const trainerStore = window.Alpine && typeof window.Alpine.store === 'function' ? window.Alpine.store('trainer') : null;
        if (trainerStore) {
            trainerStore.isTraining = isTraining;
            if (shouldResetProgress) {
                trainerStore.trainingProgress = {};
            }
        }

        const statusDetail = {
            status: normalizedStatus,
            job_id: jobId,
            source: options.source || 'event-handler',
        };
        if (data && typeof data === 'object') {
            const possibleMessage = data.message || data.error || data.detail;
            if (possibleMessage) {
                statusDetail.message = possibleMessage;
            }
        }
        window.dispatchEvent(new CustomEvent('training-status', { detail: statusDetail }));

        if (shouldResetProgress) {
            window.dispatchEvent(new CustomEvent('training-progress', { detail: { reset: true, job_id: jobId } }));
            this.resetTrainingState();
        }
    }

    formatEnhancedMessage(event) {
        let message = event.message || '';

        // For training events, show structured data even without message
        if (event.message_type === 'train') {
            const parts = [];
            if (event.state && event.state.global_step !== undefined) {
                parts.push(`Step: ${event.state.global_step}${this.totalSteps ? '/' + this.totalSteps : ''}`);
            }
            if (event.loss !== undefined) {
                parts.push(`Loss: ${event.loss.toFixed(4)}`);
            }
            if (event.learning_rate !== undefined) {
                parts.push(`LR: ${event.learning_rate.toExponential(2)}`);
            }
            if (event.epoch !== undefined) {
                parts.push(`Epoch: ${event.epoch}/${event.final_epoch || '?'}`);
            }

            if (parts.length > 0) {
                message = parts.join(' | ');
            }
        }

        // For progress updates, format the progress info
        else if (event.message_type === 'progress_update' || event.message_type === 'lifecycle.stage') {
            const stageInfo = event.stage || event.message || {};
            const progress = stageInfo.progress || stageInfo;
            const label = stageInfo.readable_type || stageInfo.label || stageInfo.progress_type || stageInfo.key;
            let currentVal = Number(progress.current_estimated_index ?? progress.current);
            if (!Number.isFinite(currentVal)) {
                currentVal = NaN;
            }
            let totalVal = Number(progress.total_elements ?? progress.total);
            if (!Number.isFinite(totalVal)) {
                totalVal = NaN;
            }
            let percent = Number(progress.percent ?? progress.progress);
            if (!Number.isFinite(percent)) {
                percent = NaN;
            }
            if (!Number.isFinite(currentVal) || !Number.isFinite(totalVal) || totalVal <= 0) {
                currentVal = Number.isFinite(percent) ? percent : 0;
                totalVal = 100;
            } else if (!Number.isFinite(percent)) {
                percent = (currentVal / totalVal) * 100;
            }
            if (Number.isFinite(percent)) {
                message = `${this.prettifyLabel(label)}: ${currentVal}/${totalVal} (${percent.toFixed(1)}%)`;
            } else {
                message = `${this.prettifyLabel(label)}: ${currentVal}/${totalVal}`;
            }
        }

        const escaped = this.escapeHtml(message || '');
        return escaped.replace(/\n/g, '<br>');
    }

    updateProgressDisplay() {
        // Update any global progress displays
        const progressContainer = document.getElementById('trainingProgress');
        if (progressContainer && this.totalSteps > 0) {
            const progress = (this.currentStep / this.totalSteps) * 100;
            progressContainer.innerHTML = `
                <div class="progress mb-2">
                    <div class="progress-bar" role="progressbar"
                         style="width: ${progress}%"
                         aria-valuenow="${this.currentStep}"
                         aria-valuemin="0"
                         aria-valuemax="${this.totalSteps}">
                        ${this.currentStep}/${this.totalSteps} steps
                    </div>
                </div>
                <div class="training-stats">
                    <span>Epoch: ${this.currentEpoch}/${this.totalEpochs}</span>
                    <span>Loss: ${this.currentLoss.toFixed(4)}</span>
                    <span>LR: ${this.learningRate.toExponential(2)}</span>
                </div>
            `;
        }
    }

    prettifyLabel(value) {
        if (!value) return 'Progress';
        return String(value)
            .replace(/[_-]+/g, ' ')
            .replace(/\b\w/g, function(chr) {
                return chr.toUpperCase();
            });
    }

    updateProgressBar(type, current, total, label) {
        // Update specific progress bars for different operations
        const safeType = type.replace(/[^a-z0-9]/gi, '-').toLowerCase();
        const progressId = `progress-${safeType}`;
        let progressElement = document.getElementById(progressId);

        if (!progressElement) {
            // Create progress element if it doesn't exist
            const progressContainer = document.getElementById('progressBars');
            if (progressContainer) {
                const progressDiv = document.createElement('div');
                progressDiv.id = progressId;
                progressDiv.className = 'progress-item mb-2';
                progressDiv.dataset.type = type; // Store original type
                progressContainer.appendChild(progressDiv);
                progressElement = progressDiv;
            }
        }

        if (progressElement) {
            const safeTotal = Number.isFinite(Number(total)) && Number(total) > 0 ? Number(total) : 100;
            const safeCurrent = Number.isFinite(Number(current)) ? Number(current) : 0;
            const percent = Math.min(100, (safeCurrent / safeTotal) * 100);
            const formattedType = (label || type).replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

            progressElement.innerHTML = `
                <label class="progress-label">${formattedType}</label>
                <div class="progress">
                    <div class="progress-bar progress-bar-striped progress-bar-animated"
                         role="progressbar"
                         style="width: ${percent}%"
                         aria-valuenow="${current}"
                         aria-valuemin="0"
                         aria-valuemax="${total}">
                        ${current}/${total} (${percent.toFixed(1)}%)
                    </div>
                </div>
            `;

            // Update data attributes for potential future reference
            progressElement.dataset.current = current;
            progressElement.dataset.total = total;

            // Remove completed progress bars after a delay
            if (current >= total && percent >= 99.9) {
                progressElement.classList.add('progress-complete');
                setTimeout(() => {
                    progressElement.style.transition = 'opacity 0.5s';
                    progressElement.style.opacity = '0';
                    setTimeout(() => {
                        if (progressElement && progressElement.parentNode) {
                            progressElement.remove();
                        }
                    }, 500);
                }, 3000);
            }
        }
    }

    handleSpecialEvents(event) {
        // Handle special event types that might update UI state
        switch (event.message_type) {
            case 'training_complete':
                window.trainerUI?.showToast('Training completed successfully!', 'success');
                break;
            case 'fatal_error':
                window.trainerUI?.showToast(this.formatEnhancedMessage(event) || 'Fatal error occurred during training!', 'error');
                break;
            case 'checkpoint_saved':
                window.trainerUI?.showToast('Checkpoint saved', 'info');
                break;
        }
    }

    limitEventDisplay(maxEvents = 1000) {
        // Remove old events if we have too many
        while (this.eventList.children.length > maxEvents) {
            this.eventList.removeChild(this.eventList.lastChild);
        }
    }

    handleExit() {
        // Don't reset the event list, just add the exit message
        this.updateEventList([{
            message_type: 'info',
            message: 'Training session ended. Ready for new configuration.',
            timestamp: new Date().toISOString()
        }]);

        // Clear training state but don't reset event index
        this.clearProgressDisplays();

        // Reset status displays
        const statusContainers = ['submitStatus', 'runStatus', 'cancelStatus'];
        statusContainers.forEach(id => {
            const element = document.getElementById(id);
            if (element) element.innerHTML = '';
        });

        // Continue listening for new events (no need to restart from 0)
        // The fetch loop is already running
    }

    resetEventList() {
        if (this.eventList) {
            // Use dynamic removal instead of innerHTML swap to avoid full redraw
            while (this.eventList.firstChild) {
                this.eventList.removeChild(this.eventList.firstChild);
            }
        }
        // Don't reset lastEventIndex here - only do it when explicitly needed
    }

    setConnectionStatus(connected, message = null, showMessage = false) {
        const connectionStatus = document.getElementById('connectionStatus');
        const eventStatus = document.getElementById('eventStatus');

        const statusText = connected ? 'Connected' : (message || 'Disconnected');
        const statusIcon = connected ? 'fa-circle text-success' : 'fa-circle text-danger';
        const statusContent = `<i class="fas ${statusIcon}"></i> ${statusText}`;

        if (connectionStatus) {
            connectionStatus.innerHTML = statusContent;
            // Add pulsing animation for disconnected state
            if (!connected) {
                connectionStatus.classList.add('connection-error');
            } else {
                connectionStatus.classList.remove('connection-error');
            }
        }

        // Also update event display status
        if (eventStatus) {
            eventStatus.innerHTML = statusContent;
        }

        // Only show message in event list if requested
        if (showMessage) {
            this.updateEventList([{
                message_type: 'info',
                message: connected ? 'Connected to callback server.' : 'Disconnected from callback server.',
                timestamp: new Date().toISOString()
            }]);
        }
    }

    async startFetching() {
        // Add initial connection message
        this.isActive = true;
        if (this.pollTimeout) {
            clearTimeout(this.pollTimeout);
            this.pollTimeout = null;
        }
        this.updateEventList([{
            message_type: 'info',
            message: 'Connecting to callback server...',
            timestamp: new Date().toISOString()
        }]);

        // Check if server is already healthy
        const isHealthy = await this.checkServerHealth();
        if (isHealthy) {
            // Server is already up, mark as connected immediately
            this.handleConnectionStateChange(true);
        }

        this.fetchBroadcastEvents();
    }

    cleanup() {
        this.isActive = false;
        this.isFetching = false;
        // Abort any pending requests
        if (this.abortController) {
            this.abortController.abort();
            this.abortController = null;
        }
        this.isReconnecting = false;
        // Clear reconnect timeout
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        if (this.pollTimeout) {
            clearTimeout(this.pollTimeout);
            this.pollTimeout = null;
        }
        // Clear health check interval
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
            this.healthCheckInterval = null;
        }
        this.healthCheckInFlight = false;
        // Close WebSocket if open
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }
        if (this.websocketReconnectTimeout) {
            clearTimeout(this.websocketReconnectTimeout);
            this.websocketReconnectTimeout = null;
        }
        // Clean up resize observer
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
    }

    startHealthCheckPolling() {
        // Poll health endpoint while reconnecting without flooding the server
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }

        this.healthCheckInterval = setInterval(async () => {
            if (!this.isReconnecting || !this.isActive) {
                clearInterval(this.healthCheckInterval);
                this.healthCheckInterval = null;
                return;
            }

            if (this.healthCheckInFlight) {
                return;
            }

            this.healthCheckInFlight = true;

            let isHealthy = false;
            try {
                isHealthy = await this.checkServerHealth();
            } finally {
                this.healthCheckInFlight = false;
            }
            if (isHealthy && this.isReconnecting) {
                // Health check passed - server is back online
                clearInterval(this.healthCheckInterval);
                this.healthCheckInterval = null;
                this.isReconnecting = false;
                // Clear any pending reconnect timeout
                if (this.reconnectTimeout) {
                    clearTimeout(this.reconnectTimeout);
                    this.reconnectTimeout = null;
                }
                // Reset attempts and connect immediately
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;

                // Mark as connected immediately since health check passed
                this.handleConnectionStateChange(true);
                // Resuming broadcast polling

                // Resume broadcast polling
                this.fetchBroadcastEvents();
            }
        }, 750); // Limit health checks while reconnecting
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    connectWebSocket() {
        if (!this.isActive) {
            return;
        }
        try {
            if (this.pollTimeout) {
                clearTimeout(this.pollTimeout);
                this.pollTimeout = null;
            }
            const wsUrl = ApiClient.resolveWebsocket('/api/training/events/stream', { forceCallback: true });
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected for event streaming');
                this.handleConnectionStateChange(true);
                this.setConnectionStatus(true, null, false);
            };

            this.websocket.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.error) {
                    console.error('WebSocket error:', data.error);
                    return;
                }

                if (data.event) {
                    // Process single event from WebSocket
                    this.processProcessKeeperEvents([data.event]);
                }
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.websocket = null;
                this.useWebSocket = false; // Fall back to polling
                this.fetchBroadcastEvents(); // Start polling
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.websocket = null;
                if (!this.isActive) {
                    return;
                }

                this.handleConnectionStateChange(false);

                // Try to reconnect after a delay
                if (this.websocketReconnectTimeout) {
                    clearTimeout(this.websocketReconnectTimeout);
                }
                this.websocketReconnectTimeout = setTimeout(() => {
                    this.websocketReconnectTimeout = null;
                    if (this.useWebSocket && !this.websocket && this.isActive) {
                        this.connectWebSocket();
                    }
                }, 2000);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.useWebSocket = false;
            this.fetchBroadcastEvents(); // Fall back to polling
        }
    }

    processProcessKeeperEvents(events) {
        // Convert process keeper events to the format expected by updateEventList
        const formattedEvents = events.map(event => {
            const jobId = this.extractJobId(event, event.data);
            const baseEvent = {
                timestamp: event.timestamp || new Date().toISOString(),
                message_type: event.type || 'info'
            };

            // Handle different event types from process keeper
            if (event.type === 'state') {
                const status = event.data?.status;
                baseEvent.message_type = this.mapStatusToEventType(status);
                baseEvent.message = this.formatStateMessage(status, event.data);
                if (status) {
                    this.notifyTrainingState(status, { ...event.data, job_id: jobId }, { resetProgress: ['failed', 'error', 'fatal', 'cancelled', 'stopped', 'terminated', 'completed'].includes(String(status).toLowerCase()) });
                }

                // Update training state if relevant
                if (event.data?.config) {
                    this.parseTrainingConfig(event.data.config);
                }
            } else if (event.type === 'error') {
                baseEvent.message_type = 'error';
                baseEvent.message = event.data?.message || 'Unknown error';
                this.notifyTrainingState('error', { ...event.data, job_id: jobId }, { resetProgress: true });
            } else if (event.type === 'webhook') {
                baseEvent.message_type = 'info';
                baseEvent.message = event.data?.message || '';
            } else if (event.type === 'training.status') {
                const statusValue = (event.data?.status || event.status || '').toLowerCase();
                baseEvent.message_type = this.mapStatusToEventType(statusValue);
                baseEvent.message = event.message || `Training status updated: ${statusValue}`;
                if (statusValue) {
                    this.notifyTrainingState(statusValue, { ...event.data, job_id: jobId }, { resetProgress: ['failed', 'error', 'fatal', 'cancelled', 'stopped', 'completed'].includes(statusValue) });
                }
            } else if (event.type === 'notification') {
                const severity = String(event.severity || '').toLowerCase();
                if (severity === 'error' || severity === 'fatal') {
                    baseEvent.message_type = 'fatal_error';
                    this.notifyTrainingState('error', { ...event.data, job_id: jobId, message: event.message || event.title }, { resetProgress: true });
                } else if (severity === 'warning') {
                    baseEvent.message_type = 'warning';
                } else if (severity === 'success') {
                    baseEvent.message_type = 'success';
                } else {
                    baseEvent.message_type = 'info';
                }
                baseEvent.message = event.message || event.title || '';
            } else if (event.type === 'progress') {
                baseEvent.message_type = 'train';
                baseEvent.state = event.data;
                baseEvent.loss = event.data?.loss;
                baseEvent.learning_rate = event.data?.learning_rate;

                // Don't show empty progress events
                if (!event.data || Object.keys(event.data).length === 0) {
                    return null;
                }
            } else if (event.type === 'exit') {
                this.notifyTrainingState('idle', { job_id: jobId }, { resetProgress: true, force: true });
            }

            return baseEvent;
        }).filter(event => event !== null);

        if (formattedEvents.length > 0) {
            this.updateEventList(formattedEvents);
        }
    }

    mapStatusToEventType(status) {
        const statusMap = {
            'starting': 'init_resume_checkpoint',
            'running': '_train_initial_msg',
            'completed': 'training_complete',
            'failed': 'fatal_error',
            'aborted': 'exit',
            'paused': 'info',
            'resumed': 'info',
            'config_updated': 'info'
        };
        return statusMap[status] || 'info';
    }

    formatStateMessage(status, data) {
        const detailRaw = data?.message || data?.detail || data?.error;
        const detail = typeof detailRaw === 'string' ? detailRaw.trim() : detailRaw;
        const messages = {
            'starting': 'Training process starting...',
            'running': 'Training is running',
            'completed': 'Training completed successfully',
            'failed': detail ? `Training failed: ${detail}` : 'Training failed',
            'aborted': detail ? `Training was aborted: ${detail}` : 'Training was aborted',
            'aborting': 'Aborting training...',
            'paused': 'Training paused',
            'resumed': 'Training resumed',
            'config_updated': 'Configuration updated'
        };
        return messages[status] || `Status: ${status}`;
    }

    parseTrainingConfig(config) {
        // Extract training parameters from config
        if (config['--num_train_epochs']) {
            this.totalEpochs = parseInt(config['--num_train_epochs']);
        }
        if (config['--max_train_steps']) {
            this.totalSteps = parseInt(config['--max_train_steps']);
        }
    }

    handleConnectionStateChange(newState) {
        const wasConnected = this.isConnected;
        this.isConnected = newState;

        if (newState && !wasConnected) {
            // We just connected
            if (!this.hasConnectedBefore) {
                // First time connection
                this.hasConnectedBefore = true;
                this.updateEventList([{
                    message_type: 'info',
                    message: 'Connected to server',
                    timestamp: new Date().toISOString()
                }]);
            } else {
                // Reconnection
                this.updateEventList([{
                    message_type: 'info',
                    message: 'Reconnected to server',
                    timestamp: new Date().toISOString()
                }]);
            }
            this.setConnectionStatus(true, null, true);
        } else if (!newState && wasConnected) {
            // We just disconnected
            this.updateEventList([{
                message_type: 'error',
                message: 'Disconnected from server',
                timestamp: new Date().toISOString()
            }]);
            this.setConnectionStatus(false, null, true);
        }
    }

    // Public methods for external control
    pause() {
        this.cleanup();
    }

    resume() {
        this.startFetching();
    }
}

// Initialize event handler when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.eventHandler = new EventHandler();
});

// Add CSS for connection error animation and progress bars
const style = document.createElement('style');
style.textContent = `
    .connection-error {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    .event-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.25rem;
    }

    .event-type {
        font-weight: 600;
        color: #6c757d;
    }

    .event-message {
        padding-left: 1rem;
    }

    .event-item-success {
        background-color: #d4edda;
        color: #155724;
        border-left: 4px solid #28a745;
    }

    .event-item-state {
        background-color: #d1ecf1;
        color: #0c5460;
        border-left: 4px solid #17a2b8;
    }

    .event-item-progress {
        background-color: #e7f3ff;
        color: #004085;
        border-left: 4px solid #0066cc;
    }

    .progress-item {
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: rgba(0, 0, 0, 0.03);
        border-radius: 0.25rem;
    }

    .progress-label {
        font-weight: 500;
        margin-bottom: 0.25rem;
        display: block;
        color: #495057;
    }

    .progress-complete {
        background-color: rgba(40, 167, 69, 0.1);
    }

    .training-stats {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
        font-size: 0.9em;
        color: #6c757d;
    }

    .training-stats span {
        padding: 0.25rem 0.5rem;
        background-color: rgba(0, 0, 0, 0.05);
        border-radius: 0.25rem;
    }

    .training-progress-section {
        margin-top: 1.5rem;
    }

    .event-display {
        margin-top: 1.5rem;
    }
`;
document.head.appendChild(style);
