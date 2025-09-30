// Wrap in IIFE to wait for ServerConfig
(async function() {
    // Wait for server configuration to be ready
    await window.ServerConfig.waitForReady();

    let lastEventIndex = 0;

    // Function to gather the payload from form inputs
function getPayload() {
    const formData = new FormData(document.getElementById('configForm'));
    const payload = {
        trainer_config: {},
        dataloader_config: [],
        webhook_config: {},
        job_id: formData.get('job_id')
    };

    // Convert FormData into respective configuration objects
    formData.forEach((value, key) => {
        if (key.startsWith('--')) {
            payload.trainer_config[key] = value;
        } else if (key === 'dataloader_config') {
            try {
                payload.dataloader_config = JSON.parse(value);
            } catch (error) {
                document.getElementById('submitStatus').innerHTML = `<p class='submit-error'>Error: Invalid JSON for Dataloader Config</p>`;
                return null;
            }
        } else if (key === 'webhook_config') {
            try {
                payload.webhook_config = JSON.parse(value);
            } catch (error) {
                document.getElementById('submitStatus').innerHTML = `<p class='submit-error'>Error: Invalid JSON for Webhook Config</p>`;
                return null;
            }
        }
    });

    return payload;
}

// Function to update event list based on received events
function updateEventList(events) {
    const eventList = document.getElementById('eventList');
    events.forEach(event => {
        if (!event.message) return; // Skip events without a message

        const eventItem = document.createElement('div');
        eventItem.className = 'event-item';

        // Apply specific styles based on event type
        switch (event.message_type) {
            case 'error':
            case 'fatal_error':
            case 'exit':
                eventItem.classList.add('event-item-error'); // Red background for errors
                break;
            case 'train':
            case 'init_resume_checkpoint':
            case '_train_initial_msg':
                eventItem.classList.add('event-item-train'); // Highlighted for special train messages
                break;
            case 'info':
                eventItem.classList.add('event-item-info'); // Blue background for informational messages
                break;
            default:
                eventItem.classList.add('event-item-default'); // Default style for other messages
                break;
        }

        // Set the content of the event item
        eventItem.textContent = `${event.message_type}: ${event.message}`;

        // Append the new event to the list
        eventList.appendChild(eventItem);
    });

    // Auto-scroll to the latest event
    eventList.scrollTop = eventList.scrollHeight;
}


// Function to reset the event list and index
function resetEventList() {
    const eventList = document.getElementById('eventList');
    eventList.innerHTML = ''; // Clear the existing events
    lastEventIndex = 0; // Reset the index
}

// Function to fetch and display broadcast events
async function fetchBroadcastEvents() {
    try {
        const response = await fetch(`${window.ServerConfig.callbackUrl}/broadcast?last_event_index=${lastEventIndex}`);
        const data = await response.json();

        // Check if there's an "exit" event in the data
        const exitEvent = data.events.find(event => event.message_type === "exit");
        if (exitEvent) {
            resetEventList(); // Clear events if "exit" is detected
            updateEventList([]); // Optionally show the exit event message
            lastEventIndex = 0; // Reset the index
            // clear the button statuses
            document.getElementById('submitStatus').innerHTML = '';
            document.getElementById('runStatus').innerHTML = '';
            document.getElementById('cancelStatus').innerHTML = '';
            // Continue fetching new events
            setTimeout(fetchBroadcastEvents, 1000);
            return; // Stop fetching further until user action resets
        }

        // Update the display with new events
        updateEventList(data.events);

        // Update the last event index for the next fetch
        lastEventIndex = data.next_index;

        // Continue fetching new events
        fetchBroadcastEvents();
    } catch (error) {
        console.error('Error fetching events:', error);
        // Retry fetching after a delay
        setTimeout(fetchBroadcastEvents, 1000);
    }
}

// Start fetching broadcast events
fetchBroadcastEvents();

// Function to show toast notification
function showToast(message, type = 'success') {
    let toastEl;
    if (type === 'success') {
        toastEl = document.getElementById('successToast');
        toastEl.querySelector('.toast-body').textContent = message || 'Operation Successful!';
    } else if (type === 'error') {
        toastEl = document.getElementById('errorToast');
        toastEl.querySelector('.toast-body').textContent = message || 'An error occurred!';
    } else {
        toastEl = document.getElementById('infoToast');
        toastEl.querySelector('.toast-body').textContent = message || 'Information!';
    }
    const toast = new bootstrap.Toast(toastEl);
    toast.show();
}

function buttonUsesHTMX(button) {
    return button && (button.hasAttribute('hx-post') || button.hasAttribute('data-hx-post'));
}

// Legacy button handlers (only when HTMX is not wiring the buttons)
const legacyValidateBtn = document.getElementById('validateBtn');
if (legacyValidateBtn && !buttonUsesHTMX(legacyValidateBtn)) {
    legacyValidateBtn.addEventListener('click', function () {
        const payload = getPayload();
        if (!payload) return;

        fetch('http://localhost:8000/training/configuration/check', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
            .then(response => response.json())
            .then(data => {
                if (data.detail) {
                    document.getElementById('submitStatus').innerHTML = `<p class='submit-error'>Error: ${data.detail}</p>`;
                    showToast(data.detail, 'error');
                } else if (data.result) {
                    document.getElementById('submitStatus').innerHTML = `<p class='submit-success'>${data.result}</p>`;
                    showToast(data.result, 'success');
                } else {
                    document.getElementById('submitStatus').innerHTML = `<p class='submit-error'>Unexpected response format</p>`;
                    showToast('Unexpected response format', 'error');
                }
            })
            .catch(error => {
                document.getElementById('submitStatus').innerHTML = `<p class='submit-error'>Error: ${error.message}</p>`;
                showToast(error.message, 'error');
            });
    });
}

const legacyRunBtn = document.getElementById('runBtn');
if (legacyRunBtn && !buttonUsesHTMX(legacyRunBtn)) {
    legacyRunBtn.addEventListener('click', function () {
        const payload = getPayload();
        if (!payload) return;

        fetch('http://localhost:8001/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
            .then(response => response.json())
            .then(data => {
                if (data.detail) {
                    document.getElementById('runStatus').innerHTML = `<p class='submit-error'>Error: ${data.detail}</p>`;
                    showToast(data.detail, 'error');
                } else if (data.result) {
                    document.getElementById('runStatus').innerHTML = `<p class='submit-success'>${data.result}</p>`;
                    showToast(data.result, 'success');
                } else {
                    document.getElementById('runStatus').innerHTML = `<p class='submit-error'>Unexpected response format</p>`;
                    showToast('Unexpected response format', 'error');
                }
            })
            .catch(error => {
                document.getElementById('runStatus').innerHTML = `<p class='submit-error'>Error: ${error.message}</p>`;
                showToast(error.message, 'error');
            });
    });
}

const legacyCancelBtn = document.getElementById('cancelBtn');
if (legacyCancelBtn && !buttonUsesHTMX(legacyCancelBtn)) {
    legacyCancelBtn.addEventListener('click', function () {
        const payload = { job_id: document.getElementById('job_id').value };
        showToast('Cancelling may take a while.', 'success');

        fetch('http://localhost:8000/training/cancel', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
            .then(response => response.json())
            .then(data => {
                if (data.detail) {
                    document.getElementById('cancelStatus').innerHTML = `<p class='submit-error'>Error: ${data.detail}</p>`;
                    showToast(data.detail, 'error');
                } else if (data.result) {
                    document.getElementById('cancelStatus').innerHTML = `<p class='submit-success'>${data.result}</p>`;
                    showToast(data.result, 'success');
                } else {
                    document.getElementById('cancelStatus').innerHTML = `<p class='submit-error'>Unexpected response format</p>`;
                    showToast('Unexpected response format', 'error');
                }
            })
            .catch(error => {
                document.getElementById('cancelStatus').innerHTML = `<p class='submit-error'>Error: ${error.message}</p>`;
                showToast(error.message, 'error');
            });
    });
}

})(); // End of IIFE
