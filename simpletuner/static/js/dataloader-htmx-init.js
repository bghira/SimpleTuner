// HTMX-specific initialization for dataloader builder
(function() {
    let skipLogged = false;

    // Function to initialize or reinitialize the dataloader builder
    function initDataloaderBuilder() {
        const section = document.getElementById('dataloaderSection');

        // Bail out early if the Alpine-based builder is mounted
        if (section?.dataset?.builder && section.dataset.builder !== 'legacy') {
            if (!skipLogged) {
                // Dataloader builder initialization skipped: Alpine-managed datasets UI detected.
                skipLogged = true;
            }
            return;
        }

        // Check for dataset elements

        // Check if we're on the datasets tab
        const datasetList = document.getElementById('datasetList');
        const jsonEditor = document.getElementById('dataloader_config');

        if (!datasetList) {
            // datasetList element not found
            return;
        }
        if (!jsonEditor) {
            // dataloader_config textarea not found
            return;
        }

        // Dataset elements found, initializing DataloaderBuilder

        try {
            // Clean up any existing instance
            if (window.dataloaderBuilderInstance) {
                // Cleaning up existing DataloaderBuilder instance
                // Clear the dataset list to avoid duplicates
                if (datasetList) {
                    datasetList.innerHTML = '';
                }
            }

            // Create new instance
            if (typeof DataloaderBuilder !== 'undefined') {
                window.dataloaderBuilderInstance = new DataloaderBuilder();
                // DataloaderBuilder instance created successfully

                // Trigger a render to ensure UI is updated
                if (window.dataloaderBuilderInstance && window.dataloaderBuilderInstance.render) {
                    window.dataloaderBuilderInstance.render();
                }
            } else {
                console.error('DataloaderBuilder class not found - make sure dataloader-builder.js is loaded');
            }
        } catch (error) {
            console.error('Error initializing DataloaderBuilder:', error);
        }
    }

    // Listen for HTMX events
    document.addEventListener('htmx:afterSwap', function(evt) {
        // Check if we just loaded content that might contain dataset elements
        if (evt.detail.target.id === 'tab-content' ||
            evt.detail.xhr.responseURL.includes('datasets')) {
            // Give the DOM a moment to settle
            setTimeout(initDataloaderBuilder, 200);
        }
    });

    // Also listen for manual tab switches
    document.addEventListener('dataset-tab-loaded', initDataloaderBuilder);

    // Initialize on page load if elements are already present
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initDataloaderBuilder);
    } else {
        // DOM is already loaded
        setTimeout(initDataloaderBuilder, 100);
    }
})();
