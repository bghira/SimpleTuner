require('../../simpletuner/static/js/dataset-captioning.js');

describe('datasetCaptioningComponent log scrolling', () => {
    let component;
    let originalRequestAnimationFrame;

    beforeEach(() => {
        originalRequestAnimationFrame = window.requestAnimationFrame;
        window.requestAnimationFrame = (callback) => {
            callback();
            return 1;
        };
        component = window.datasetCaptioningComponent();
        component.$refs = {
            captioningLogViewer: {
                scrollTop: 0,
                scrollHeight: 1000,
                clientHeight: 200,
            },
        };
        component.$nextTick = (callback) => callback();
    });

    afterEach(() => {
        window.requestAnimationFrame = originalRequestAnimationFrame;
    });

    test('auto-scroll intent is the source of truth before log updates', () => {
        component.autoScrollLogs = true;
        component.$refs.captioningLogViewer.scrollTop = 0;

        expect(component.shouldAutoScrollLogs()).toBe(true);
    });

    test('user scroll position disables auto-scroll when away from bottom', () => {
        component.$refs.captioningLogViewer.scrollTop = 100;
        component.$refs.captioningLogViewer.scrollHeight = 1000;
        component.$refs.captioningLogViewer.clientHeight = 200;

        component.handleLogScroll();

        expect(component.autoScrollLogs).toBe(false);
    });

    test('scrolls to bottom after Alpine and browser layout updates', () => {
        component.scrollLogsAfterUpdate(true);

        expect(component.$refs.captioningLogViewer.scrollTop).toBe(1000);
    });

    test('does not scroll when auto-scroll was disabled', () => {
        component.scrollLogsAfterUpdate(false);

        expect(component.$refs.captioningLogViewer.scrollTop).toBe(0);
    });
});
