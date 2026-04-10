/**
 * BboxKeyframeTimeline — keyframe timeline for per-frame video bounding boxes.
 *
 * Wraps an existing BboxCanvas and adds a frame slider with inline keyframe
 * markers, add/remove buttons, and an interpolation preview.
 *
 * Usage:
 *   const timeline = new BboxKeyframeTimeline(containerEl, {
 *       canvas: bboxCanvasInstance,
 *       numFrames: 80,
 *       onChange: (keyframes) => { ... },
 *   });
 *
 * Public API:
 *   setNumFrames(n)
 *   setKeyframes(keyframes)
 *   getKeyframes()
 *   destroy()
 */
class BboxKeyframeTimeline {
    // Thumb width for the range input (used to offset marker positions so
    // they line up with the thumb center at min/max).  Browser default for
    // Bootstrap .form-range is roughly 16px.
    static THUMB_W = 16;

    constructor(container, opts = {}) {
        this._container = container;
        this._canvas = opts.canvas;
        this._numFrames = opts.numFrames || 25;
        this._onChange = opts.onChange || null;

        this.keyframes = []; // [{frame, entities: [{label, bbox}]}]
        this.currentFrame = 0;
        this._playing = false;
        this._playTimer = null;
        this._editableFrame = false;

        this._buildDOM();
        this._injectStyles();
        this._bindEvents();
        this._updateUI();
    }

    /* ------------------------------------------------------------------ */
    /*  DOM construction                                                   */
    /* ------------------------------------------------------------------ */

    _buildDOM() {
        this._el = document.createElement('div');
        this._el.className = 'bbox-keyframe-timeline mt-2';

        // -- Slider row: label, slider-with-overlays, frame counter ------
        const sliderRow = document.createElement('div');
        sliderRow.className = 'd-flex align-items-center gap-2 mb-2';

        const sliderLabel = document.createElement('small');
        sliderLabel.className = 'text-body-secondary fw-semibold';
        sliderLabel.textContent = 'Frame:';
        sliderRow.appendChild(sliderLabel);

        // Wrapper that holds the <input range> plus the overlay layer
        this._sliderWrap = document.createElement('div');
        this._sliderWrap.className = 'bbox-kf-slider-wrap position-relative flex-grow-1';
        this._sliderWrap.style.height = '28px';

        // Overlay DIV for markers — sits behind the thumb but in front of
        // the track.  pointer-events:none so dragging passes through.
        this._overlay = document.createElement('div');
        this._overlay.className = 'bbox-kf-overlay';
        this._sliderWrap.appendChild(this._overlay);

        this._slider = document.createElement('input');
        this._slider.type = 'range';
        this._slider.className = 'form-range bbox-kf-range';
        this._slider.min = '0';
        this._slider.max = String(Math.max(this._numFrames - 1, 0));
        this._slider.value = '0';
        this._slider.step = '1';
        this._sliderWrap.appendChild(this._slider);

        sliderRow.appendChild(this._sliderWrap);

        this._frameLabel = document.createElement('small');
        this._frameLabel.className = 'text-body-secondary';
        this._frameLabel.style.minWidth = '60px';
        this._frameLabel.style.textAlign = 'right';
        sliderRow.appendChild(this._frameLabel);

        this._el.appendChild(sliderRow);

        // -- Button row --------------------------------------------------
        const btnRow = document.createElement('div');
        btnRow.className = 'd-flex align-items-center gap-2';

        this._addBtn = document.createElement('button');
        this._addBtn.type = 'button';
        this._addBtn.className = 'btn btn-outline-success btn-sm';
        this._addBtn.textContent = 'Add keyframe';
        btnRow.appendChild(this._addBtn);

        this._removeBtn = document.createElement('button');
        this._removeBtn.type = 'button';
        this._removeBtn.className = 'btn btn-outline-danger btn-sm';
        this._removeBtn.textContent = 'Remove keyframe';
        this._removeBtn.disabled = true;
        btnRow.appendChild(this._removeBtn);

        const spacer = document.createElement('div');
        spacer.className = 'flex-grow-1';
        btnRow.appendChild(spacer);

        this._playBtn = document.createElement('button');
        this._playBtn.type = 'button';
        this._playBtn.className = 'btn btn-outline-info btn-sm';
        this._playBtn.innerHTML = '&#9654; Preview';
        btnRow.appendChild(this._playBtn);

        this._stopBtn = document.createElement('button');
        this._stopBtn.type = 'button';
        this._stopBtn.className = 'btn btn-outline-secondary btn-sm d-none';
        this._stopBtn.innerHTML = '&#9632; Stop';
        btnRow.appendChild(this._stopBtn);

        this._el.appendChild(btnRow);

        this._container.appendChild(this._el);
    }

    /** Inject scoped CSS once per page. */
    _injectStyles() {
        if (document.getElementById('bbox-kf-timeline-styles')) return;
        const style = document.createElement('style');
        style.id = 'bbox-kf-timeline-styles';
        style.textContent = `
/* -- Keyframe timeline slider ----------------------------------------- */
.bbox-kf-slider-wrap {
    display: flex;
    align-items: center;
}
/* Overlay sits on top of the track area, behind the thumb */
.bbox-kf-overlay {
    position: absolute;
    /* Inset by half-thumb so overlay aligns with the thumb travel range */
    left: ${BboxKeyframeTimeline.THUMB_W / 2}px;
    right: ${BboxKeyframeTimeline.THUMB_W / 2}px;
    top: 0; bottom: 0;
    pointer-events: none;
    z-index: 0;
}
/* Make the range sit on top so the thumb is draggable */
.bbox-kf-range {
    position: relative;
    z-index: 1;
    width: 100%;
    margin: 0;
    /* Transparent track so overlay shows through */
}
/* Webkit: style the track */
.bbox-kf-range::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.12);
}
.bbox-kf-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px;
    height: 16px;
    margin-top: -6px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid rgba(255,255,255,0.4);
    box-shadow: 0 0 4px rgba(0,0,0,0.4);
    cursor: pointer;
    position: relative;
    z-index: 3;
}
.bbox-kf-range::-webkit-slider-thumb:hover {
    background: #e0e0e0;
    transform: scale(1.1);
}
/* Firefox */
.bbox-kf-range::-moz-range-track {
    height: 6px;
    border-radius: 3px;
    background: rgba(255,255,255,0.1);
    border: 1px solid rgba(255,255,255,0.12);
}
.bbox-kf-range::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #fff;
    border: 2px solid rgba(255,255,255,0.4);
    box-shadow: 0 0 4px rgba(0,0,0,0.4);
    cursor: pointer;
}
/* Keyframe diamond marker */
.bbox-kf-diamond {
    position: absolute;
    top: 50%;
    width: 10px; height: 10px;
    transform: translate(-50%, -50%) rotate(45deg);
    border: 1px solid rgba(255,255,255,0.5);
    pointer-events: none;
    z-index: 2;
}
.bbox-kf-diamond--active {
    width: 12px; height: 12px;
    box-shadow: 0 0 6px rgba(255,152,0,0.6);
}
/* Interpolation segment between keyframes */
.bbox-kf-segment {
    position: absolute;
    top: 50%;
    height: 4px;
    transform: translateY(-50%);
    background: rgba(76,175,80,0.35);
    border-radius: 2px;
    pointer-events: none;
}
/* Tick marks */
.bbox-kf-tick {
    position: absolute;
    top: 50%;
    width: 1px; height: 10px;
    transform: translate(-50%, -50%);
    background: rgba(255,255,255,0.1);
    pointer-events: none;
}
/* Empty hint */
.bbox-kf-hint {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: none;
    color: rgba(255,255,255,0.25);
    font-size: 0.7rem;
}
`;
        document.head.appendChild(style);
    }

    /* ------------------------------------------------------------------ */
    /*  Events                                                             */
    /* ------------------------------------------------------------------ */

    _bindEvents() {
        this._slider.addEventListener('input', () => {
            this.currentFrame = parseInt(this._slider.value, 10);
            this._onFrameChange();
        });

        this._addBtn.addEventListener('click', () => this._addKeyframe());
        this._removeBtn.addEventListener('click', () => this._removeKeyframe());
        this._playBtn.addEventListener('click', () => this._startPreview());
        this._stopBtn.addEventListener('click', () => this._stopPreview());
    }

    /* ------------------------------------------------------------------ */
    /*  Keyframe management                                                */
    /* ------------------------------------------------------------------ */

    _addKeyframe() {
        const frame = this.currentFrame;
        const entities = this._canvas.getBoxes();
        const existing = this.keyframes.findIndex(kf => kf.frame === frame);
        if (existing >= 0) {
            this.keyframes[existing].entities = JSON.parse(JSON.stringify(entities));
        } else {
            this.keyframes.push({ frame, entities: JSON.parse(JSON.stringify(entities)) });
            this.keyframes.sort((a, b) => a.frame - b.frame);
        }
        this._editableFrame = true;
        this._updateUI();
        this._fireChange();
    }

    _removeKeyframe() {
        const idx = this.keyframes.findIndex(kf => kf.frame === this.currentFrame);
        if (idx >= 0) {
            this.keyframes.splice(idx, 1);
            this._editableFrame = false;
            this._onFrameChange();
            this._fireChange();
        }
    }

    _onFrameChange() {
        const isKeyframe = this.keyframes.some(kf => kf.frame === this.currentFrame);
        this._editableFrame = isKeyframe;

        if (isKeyframe) {
            const kf = this.keyframes.find(kf => kf.frame === this.currentFrame);
            this._canvas.setBoxes(kf.entities);
        } else {
            const interpolated = this._interpolateFrame(this.currentFrame);
            this._canvas.setBoxes(interpolated);
        }
        this._updateUI();
    }

    /* ------------------------------------------------------------------ */
    /*  Interpolation (JS-side, mirrors Python module for preview)         */
    /* ------------------------------------------------------------------ */

    _interpolateFrame(frameIdx) {
        if (this.keyframes.length === 0) return [];

        const labelTimelines = {};
        for (const kf of this.keyframes) {
            const f = Math.min(kf.frame, this._numFrames - 1);
            for (const ent of kf.entities) {
                if (!labelTimelines[ent.label]) labelTimelines[ent.label] = [];
                labelTimelines[ent.label].push({ frame: f, bbox: ent.bbox.slice() });
            }
        }

        const result = [];
        for (const label of Object.keys(labelTimelines).sort()) {
            const tl = labelTimelines[label];
            const bbox = this._interpAt(tl, frameIdx);
            result.push({ label, bbox });
        }
        return result;
    }

    _interpAt(timeline, t) {
        if (timeline.length === 1) return timeline[0].bbox.slice();
        if (t <= timeline[0].frame) return timeline[0].bbox.slice();
        if (t >= timeline[timeline.length - 1].frame) return timeline[timeline.length - 1].bbox.slice();

        for (let i = 0; i < timeline.length - 1; i++) {
            const f0 = timeline[i].frame, b0 = timeline[i].bbox;
            const f1 = timeline[i + 1].frame, b1 = timeline[i + 1].bbox;
            if (f0 <= t && t <= f1) {
                if (f0 === f1) return b1.slice();
                const alpha = (t - f0) / (f1 - f0);
                return b0.map((v, j) => v + alpha * (b1[j] - v));
            }
        }
        return timeline[timeline.length - 1].bbox.slice();
    }

    /* ------------------------------------------------------------------ */
    /*  Playback preview                                                   */
    /* ------------------------------------------------------------------ */

    _startPreview() {
        if (this._playing) return;
        this._playing = true;
        this._playBtn.classList.add('d-none');
        this._stopBtn.classList.remove('d-none');
        this._slider.disabled = true;
        this._addBtn.disabled = true;
        this._removeBtn.disabled = true;

        let frame = 0;
        const tick = () => {
            if (!this._playing) return;
            this.currentFrame = frame;
            this._slider.value = String(frame);
            const interpolated = this._interpolateFrame(frame);
            this._canvas.setBoxes(interpolated);
            this._updateFrameLabel();
            this._renderOverlay();

            frame++;
            if (frame >= this._numFrames) frame = 0;
            this._playTimer = setTimeout(tick, 100);
        };
        tick();
    }

    _stopPreview() {
        this._playing = false;
        if (this._playTimer) {
            clearTimeout(this._playTimer);
            this._playTimer = null;
        }
        this._playBtn.classList.remove('d-none');
        this._stopBtn.classList.add('d-none');
        this._slider.disabled = false;
        this._addBtn.disabled = false;
        this._onFrameChange();
    }

    /* ------------------------------------------------------------------ */
    /*  UI updates                                                         */
    /* ------------------------------------------------------------------ */

    _updateUI() {
        this._updateFrameLabel();
        this._renderOverlay();

        const isKeyframe = this.keyframes.some(kf => kf.frame === this.currentFrame);
        this._removeBtn.disabled = !isKeyframe;
        this._addBtn.textContent = isKeyframe ? 'Update keyframe' : 'Add keyframe';
    }

    _updateFrameLabel() {
        this._frameLabel.textContent = `${this.currentFrame + 1} / ${this._numFrames}`;
    }

    /** Render keyframe diamonds + interpolation segments onto the slider overlay. */
    _renderOverlay() {
        this._overlay.innerHTML = '';
        const max = Math.max(this._numFrames - 1, 1);
        const pct = (f) => `${(f / max) * 100}%`;

        // Interpolation segments between consecutive keyframes
        for (let i = 0; i < this.keyframes.length - 1; i++) {
            const f0 = Math.min(this.keyframes[i].frame, max);
            const f1 = Math.min(this.keyframes[i + 1].frame, max);
            const seg = document.createElement('div');
            seg.className = 'bbox-kf-segment';
            seg.style.left = pct(f0);
            seg.style.width = `${((f1 - f0) / max) * 100}%`;
            this._overlay.appendChild(seg);
        }

        // Tick marks
        const tickInterval = this._numFrames <= 50 ? 10 : this._numFrames <= 200 ? 25 : 50;
        for (let f = 0; f <= max; f += tickInterval) {
            const tick = document.createElement('div');
            tick.className = 'bbox-kf-tick';
            tick.style.left = pct(f);
            this._overlay.appendChild(tick);
        }

        // Keyframe diamonds
        for (const kf of this.keyframes) {
            const isCurrent = kf.frame === this.currentFrame;
            const diamond = document.createElement('div');
            diamond.className = 'bbox-kf-diamond' + (isCurrent ? ' bbox-kf-diamond--active' : '');
            diamond.style.left = pct(kf.frame);
            diamond.style.background = isCurrent ? '#FF9800' : '#4CAF50';
            if (!isCurrent) {
                diamond.style.boxShadow = '0 0 4px rgba(76,175,80,0.5)';
            }
            diamond.title = `Keyframe @ frame ${kf.frame + 1}`;
            this._overlay.appendChild(diamond);
        }

        // Empty-state hint
        if (this.keyframes.length === 0) {
            const hint = document.createElement('div');
            hint.className = 'bbox-kf-hint';
            hint.textContent = 'Draw boxes on the canvas, then click "Add keyframe"';
            this._overlay.appendChild(hint);
        }
    }

    /* ------------------------------------------------------------------ */
    /*  Public API                                                         */
    /* ------------------------------------------------------------------ */

    _fireChange() {
        if (this._onChange) {
            this._onChange(JSON.parse(JSON.stringify(this.keyframes)));
        }
    }

    setNumFrames(n) {
        this._numFrames = Math.max(1, n);
        this._slider.max = String(this._numFrames - 1);
        if (this.currentFrame >= this._numFrames) {
            this.currentFrame = this._numFrames - 1;
            this._slider.value = String(this.currentFrame);
        }
        this._updateUI();
    }

    setKeyframes(keyframes) {
        this.keyframes = JSON.parse(JSON.stringify(keyframes || []));
        this.keyframes.sort((a, b) => a.frame - b.frame);
        this._onFrameChange();
    }

    getKeyframes() {
        return JSON.parse(JSON.stringify(this.keyframes));
    }

    destroy() {
        this._stopPreview();
        if (this._el && this._el.parentNode) {
            this._el.parentNode.removeChild(this._el);
        }
    }
}
