/**
 * BboxCanvas — self-contained bounding-box drawing widget.
 *
 * Usage:
 *   const canvas = new BboxCanvas(containerElement, {
 *       width: 1024,    // target resolution width
 *       height: 1024,   // target resolution height
 *       onChange: (entities) => { ... },
 *   });
 *
 * Public API:
 *   setResolution(w, h)
 *   setBoxes([{label, bbox: [x1,y1,x2,y2]}])
 *   getBoxes() → [{label, bbox}]
 *   destroy()
 */
class BboxCanvas {
    static COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0', '#00BCD4'];
    static HANDLE_SIZE = 6;
    static DELETE_RADIUS = 9;

    constructor(container, opts = {}) {
        this._container = container;
        this._targetW = opts.width || 1024;
        this._targetH = opts.height || 1024;
        this._onChange = opts.onChange || null;

        this.boxes = [];
        this._mode = 'idle';
        this._selectedIdx = -1;
        this._activeHandle = null;
        this._dragStart = null;
        this._dragBox = null;
        this._drawRect = null;
        this._deleteBtn = null;

        this._canvas = document.createElement('canvas');
        this._canvas.style.cursor = 'crosshair';
        this._canvas.style.display = 'block';
        this._canvas.style.margin = '0 auto';
        this._canvas.style.borderRadius = '6px';
        this._container.appendChild(this._canvas);
        this._ctx = this._canvas.getContext('2d');

        this._labelInput = null;
        this._pendingBoxIdx = -1;

        this._resizeCanvas();
        this._bindEvents();
    }

    // -- Public API --

    setResolution(w, h) {
        this._targetW = w;
        this._targetH = h;
        this._resizeCanvas();
        this._draw();
    }

    setBoxes(entities) {
        this.boxes = (entities || []).map((e, i) => ({
            label: e.label || '',
            x1: e.bbox[0],
            y1: e.bbox[1],
            x2: e.bbox[2],
            y2: e.bbox[3],
            color: BboxCanvas.COLORS[i % BboxCanvas.COLORS.length],
        }));
        this._selectedIdx = -1;
        this._draw();
    }

    getBoxes() {
        return this.boxes.map(b => ({
            label: b.label,
            bbox: [b.x1, b.y1, b.x2, b.y2],
        }));
    }

    destroy() {
        this._removeLabelInput();
        this._canvas.removeEventListener('mousedown', this._onMouseDown);
        this._canvas.removeEventListener('mousemove', this._onMouseMove);
        this._canvas.removeEventListener('dblclick', this._onDblClick);
        window.removeEventListener('mouseup', this._onMouseUp);
        window.removeEventListener('keydown', this._onKeyDown);
        this._canvas.remove();
    }

    // -- Internal: sizing --

    _resizeCanvas() {
        const containerW = this._container.clientWidth || 400;
        const ar = this._targetH / this._targetW;
        const displayW = containerW;
        const displayH = Math.round(containerW * ar);
        const dpr = window.devicePixelRatio || 1;

        this._canvas.style.width = displayW + 'px';
        this._canvas.style.height = displayH + 'px';
        this._canvas.width = Math.round(displayW * dpr);
        this._canvas.height = Math.round(displayH * dpr);
        this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this._displayW = displayW;
        this._displayH = displayH;
    }

    // -- Internal: coordinate helpers --

    _canvasPos(e) {
        const rect = this._canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    }

    _toNorm(px) {
        return {
            x: Math.max(0, Math.min(1, px.x / this._displayW)),
            y: Math.max(0, Math.min(1, px.y / this._displayH)),
        };
    }

    _toPixel(norm) {
        return {
            x: norm.x * this._displayW,
            y: norm.y * this._displayH,
        };
    }

    // -- Internal: event binding --

    _bindEvents() {
        this._onMouseDown = this._handleMouseDown.bind(this);
        this._onMouseMove = this._handleMouseMove.bind(this);
        this._onMouseUp = this._handleMouseUp.bind(this);
        this._onKeyDown = this._handleKeyDown.bind(this);
        this._onDblClick = this._handleDblClick.bind(this);

        this._canvas.addEventListener('mousedown', this._onMouseDown);
        this._canvas.addEventListener('mousemove', this._onMouseMove);
        this._canvas.addEventListener('dblclick', this._onDblClick);
        window.addEventListener('mouseup', this._onMouseUp);
        window.addEventListener('keydown', this._onKeyDown);
    }

    _handleMouseDown(e) {
        const pos = this._canvasPos(e);
        const norm = this._toNorm(pos);

        // Check for delete button hit on selected box
        if (this._selectedIdx >= 0 && this._deleteBtn) {
            const dx = pos.x - this._deleteBtn.x;
            const dy = pos.y - this._deleteBtn.y;
            if (dx * dx + dy * dy <= this._deleteBtn.r * this._deleteBtn.r) {
                this.boxes.splice(this._selectedIdx, 1);
                this._recolorBoxes();
                this._selectedIdx = -1;
                this._deleteBtn = null;
                this._draw();
                this._fireChange();
                e.preventDefault();
                return;
            }
        }

        // Check for handle hit on selected box
        if (this._selectedIdx >= 0) {
            const handle = this._hitHandle(this._selectedIdx, pos);
            if (handle) {
                this._mode = 'resizing';
                this._activeHandle = handle;
                this._dragStart = norm;
                this._dragBox = { ...this.boxes[this._selectedIdx] };
                e.preventDefault();
                return;
            }
        }

        // Check for box hit (reverse order so top-drawn boxes are hit first)
        for (let i = this.boxes.length - 1; i >= 0; i--) {
            if (this._hitBox(i, norm)) {
                this._selectedIdx = i;
                const handle = this._hitHandle(i, pos);
                if (handle) {
                    this._mode = 'resizing';
                    this._activeHandle = handle;
                    this._dragStart = norm;
                    this._dragBox = { ...this.boxes[i] };
                } else {
                    this._mode = 'moving';
                    this._dragStart = norm;
                    this._dragBox = { ...this.boxes[i] };
                }
                this._draw();
                e.preventDefault();
                return;
            }
        }

        // Empty area — start drawing
        this._selectedIdx = -1;
        this._mode = 'drawing';
        this._dragStart = norm;
        this._drawRect = { x1: norm.x, y1: norm.y, x2: norm.x, y2: norm.y };
        this._draw();
        e.preventDefault();
    }

    _handleMouseMove(e) {
        const pos = this._canvasPos(e);
        const norm = this._toNorm(pos);

        if (this._mode === 'drawing') {
            this._drawRect.x2 = norm.x;
            this._drawRect.y2 = norm.y;
            this._draw();
            return;
        }

        if (this._mode === 'moving' && this._selectedIdx >= 0) {
            const dx = norm.x - this._dragStart.x;
            const dy = norm.y - this._dragStart.y;
            const b = this._dragBox;
            const bw = b.x2 - b.x1;
            const bh = b.y2 - b.y1;
            let nx1 = b.x1 + dx;
            let ny1 = b.y1 + dy;
            nx1 = Math.max(0, Math.min(1 - bw, nx1));
            ny1 = Math.max(0, Math.min(1 - bh, ny1));
            this.boxes[this._selectedIdx].x1 = nx1;
            this.boxes[this._selectedIdx].y1 = ny1;
            this.boxes[this._selectedIdx].x2 = nx1 + bw;
            this.boxes[this._selectedIdx].y2 = ny1 + bh;
            this._draw();
            return;
        }

        if (this._mode === 'resizing' && this._selectedIdx >= 0 && this._activeHandle) {
            this._applyResize(norm);
            this._draw();
            return;
        }

        // Update cursor
        if (this._selectedIdx >= 0 && this._deleteBtn) {
            const dx = pos.x - this._deleteBtn.x;
            const dy = pos.y - this._deleteBtn.y;
            if (dx * dx + dy * dy <= this._deleteBtn.r * this._deleteBtn.r) {
                this._canvas.style.cursor = 'pointer';
                return;
            }
        }
        if (this._selectedIdx >= 0) {
            const handle = this._hitHandle(this._selectedIdx, pos);
            if (handle) {
                this._canvas.style.cursor = this._handleCursor(handle);
                return;
            }
        }
        for (let i = this.boxes.length - 1; i >= 0; i--) {
            if (this._hitBox(i, norm)) {
                this._canvas.style.cursor = 'move';
                return;
            }
        }
        this._canvas.style.cursor = 'crosshair';
    }

    _handleMouseUp(e) {
        if (this._mode === 'drawing' && this._drawRect) {
            const r = this._drawRect;
            const x1 = Math.min(r.x1, r.x2);
            const y1 = Math.min(r.y1, r.y2);
            const x2 = Math.max(r.x1, r.x2);
            const y2 = Math.max(r.y1, r.y2);
            const minSize = 0.02;
            if ((x2 - x1) > minSize && (y2 - y1) > minSize) {
                const idx = this.boxes.length;
                this.boxes.push({
                    label: '',
                    x1, y1, x2, y2,
                    color: BboxCanvas.COLORS[idx % BboxCanvas.COLORS.length],
                });
                this._selectedIdx = idx;
                this._drawRect = null;
                this._mode = 'idle';
                this._draw();
                this._showLabelInput(idx);
                return;
            }
        }

        if (this._mode === 'moving' || this._mode === 'resizing') {
            this._fireChange();
        }

        this._mode = 'idle';
        this._drawRect = null;
        this._activeHandle = null;
        this._dragStart = null;
        this._dragBox = null;
        this._draw();
    }

    _handleKeyDown(e) {
        if (this._labelInput) return;
        if ((e.key === 'Delete' || e.key === 'Backspace') && this._selectedIdx >= 0) {
            this.boxes.splice(this._selectedIdx, 1);
            this._recolorBoxes();
            this._selectedIdx = -1;
            this._deleteBtn = null;
            this._draw();
            this._fireChange();
            e.preventDefault();
        }
    }

    _handleDblClick(e) {
        const pos = this._canvasPos(e);
        const norm = this._toNorm(pos);
        for (let i = this.boxes.length - 1; i >= 0; i--) {
            if (this._hitBox(i, norm)) {
                this._selectedIdx = i;
                this._draw();
                this._showLabelInput(i);
                e.preventDefault();
                return;
            }
        }
    }

    // -- Internal: hit testing --

    _hitBox(idx, norm) {
        const b = this.boxes[idx];
        return norm.x >= b.x1 && norm.x <= b.x2 && norm.y >= b.y1 && norm.y <= b.y2;
    }

    _getHandles(idx) {
        const b = this.boxes[idx];
        const mx = (b.x1 + b.x2) / 2;
        const my = (b.y1 + b.y2) / 2;
        return [
            { id: 'tl', x: b.x1, y: b.y1 },
            { id: 'tc', x: mx,   y: b.y1 },
            { id: 'tr', x: b.x2, y: b.y1 },
            { id: 'ml', x: b.x1, y: my   },
            { id: 'mr', x: b.x2, y: my   },
            { id: 'bl', x: b.x1, y: b.y2 },
            { id: 'bc', x: mx,   y: b.y2 },
            { id: 'br', x: b.x2, y: b.y2 },
        ];
    }

    _hitHandle(idx, pxPos) {
        const hs = BboxCanvas.HANDLE_SIZE;
        const handles = this._getHandles(idx);
        for (const h of handles) {
            const hp = this._toPixel(h);
            if (Math.abs(pxPos.x - hp.x) <= hs && Math.abs(pxPos.y - hp.y) <= hs) {
                return h.id;
            }
        }
        return null;
    }

    _handleCursor(id) {
        const map = {
            tl: 'nwse-resize', br: 'nwse-resize',
            tr: 'nesw-resize', bl: 'nesw-resize',
            tc: 'ns-resize',   bc: 'ns-resize',
            ml: 'ew-resize',   mr: 'ew-resize',
        };
        return map[id] || 'pointer';
    }

    // -- Internal: resize logic --

    _applyResize(norm) {
        const b = this.boxes[this._selectedIdx];
        const db = this._dragBox;
        const h = this._activeHandle;

        if (h.includes('l')) b.x1 = Math.max(0, Math.min(norm.x, db.x2 - 0.02));
        if (h.includes('r')) b.x2 = Math.min(1, Math.max(norm.x, db.x1 + 0.02));
        if (h.includes('t')) b.y1 = Math.max(0, Math.min(norm.y, db.y2 - 0.02));
        if (h.includes('b')) b.y2 = Math.min(1, Math.max(norm.y, db.y1 + 0.02));
    }

    // -- Internal: label input --

    _showLabelInput(idx) {
        this._removeLabelInput();
        const b = this.boxes[idx];
        const px = this._toPixel({ x: b.x1, y: b.y1 });
        const canvasRect = this._canvas.getBoundingClientRect();
        const containerRect = this._container.getBoundingClientRect();
        const offsetX = canvasRect.left - containerRect.left;
        const offsetY = canvasRect.top - containerRect.top;

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'form-control form-control-sm';
        input.placeholder = 'Label (required)';
        input.value = b.label;
        input.style.position = 'absolute';
        input.style.left = (offsetX + px.x) + 'px';
        input.style.top = (offsetY + px.y - 30) + 'px';
        input.style.width = '140px';
        input.style.zIndex = '10';
        input.style.fontSize = '12px';

        const commit = () => {
            const val = input.value.trim();
            if (val) {
                this.boxes[idx].label = val;
                this._fireChange();
            } else {
                this.boxes.splice(idx, 1);
                this._recolorBoxes();
                if (this._selectedIdx === idx) this._selectedIdx = -1;
                this._fireChange();
            }
            this._removeLabelInput();
            this._draw();
        };

        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); commit(); }
            if (e.key === 'Escape') { e.preventDefault(); this._removeLabelInput(); this._draw(); }
        });
        input.addEventListener('blur', commit);

        this._container.style.position = 'relative';
        this._container.appendChild(input);
        this._labelInput = input;
        this._pendingBoxIdx = idx;
        input.focus();
        input.select();
    }

    _removeLabelInput() {
        if (this._labelInput) {
            this._labelInput.removeEventListener('blur', this._labelInput._commitFn);
            this._labelInput.remove();
            this._labelInput = null;
            this._pendingBoxIdx = -1;
        }
    }

    // -- Internal: drawing --

    _draw() {
        const ctx = this._ctx;
        const w = this._displayW;
        const h = this._displayH;

        // Background
        ctx.clearRect(0, 0, w, h);
        const isDark = document.documentElement.getAttribute('data-bs-theme') === 'dark'
            || window.matchMedia?.('(prefers-color-scheme: dark)').matches;
        ctx.fillStyle = isDark ? '#1a1d21' : '#f8f9fa';
        ctx.fillRect(0, 0, w, h);

        // Grid
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
        ctx.lineWidth = 0.5;
        const step = Math.max(w, h) / 8;
        for (let x = step; x < w; x += step) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
        }
        for (let y = step; y < h; y += step) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        }

        // Crosshair at center
        ctx.strokeStyle = isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 6]);
        ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
        ctx.setLineDash([]);

        // Empty-state hint
        if (this.boxes.length === 0 && this._mode !== 'drawing') {
            ctx.fillStyle = isDark ? 'rgba(255,255,255,0.2)' : 'rgba(0,0,0,0.2)';
            ctx.font = '14px system-ui, -apple-system, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('Click and drag to draw a bounding box', w / 2, h / 2);
            ctx.textAlign = 'start';
        }

        // Reset delete button reference
        this._deleteBtn = null;

        // Draw boxes
        this.boxes.forEach((b, i) => {
            const p1 = this._toPixel({ x: b.x1, y: b.y1 });
            const p2 = this._toPixel({ x: b.x2, y: b.y2 });
            const bw = p2.x - p1.x;
            const bh = p2.y - p1.y;
            const isSelected = i === this._selectedIdx;

            // Fill
            ctx.fillStyle = b.color + '26';
            ctx.fillRect(p1.x, p1.y, bw, bh);

            // Stroke
            ctx.strokeStyle = b.color;
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.strokeRect(p1.x, p1.y, bw, bh);

            // Label pill
            if (b.label) {
                ctx.font = '12px system-ui, -apple-system, sans-serif';
                const tw = ctx.measureText(b.label).width;
                const pillW = tw + 12;
                const pillH = 20;
                const pillX = p1.x;
                const pillY = p1.y - pillH - 3;

                ctx.fillStyle = b.color;
                ctx.beginPath();
                ctx.roundRect(pillX, pillY, pillW, pillH, 4);
                ctx.fill();

                ctx.fillStyle = '#fff';
                ctx.textBaseline = 'middle';
                ctx.fillText(b.label, pillX + 6, pillY + pillH / 2);
            }

            // Handles and delete button for selected box
            if (isSelected) {
                const handles = this._getHandles(i);
                const hs = BboxCanvas.HANDLE_SIZE;
                handles.forEach(handle => {
                    const hp = this._toPixel(handle);
                    ctx.fillStyle = '#fff';
                    ctx.fillRect(hp.x - hs, hp.y - hs, hs * 2, hs * 2);
                    ctx.strokeStyle = b.color;
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(hp.x - hs, hp.y - hs, hs * 2, hs * 2);
                });

                // Delete button
                const delR = BboxCanvas.DELETE_RADIUS;
                const delX = p2.x + delR + 2;
                const delY = p1.y - delR - 2;
                // Clamp to canvas bounds
                const clampedX = Math.min(delX, w - delR - 1);
                const clampedY = Math.max(delY, delR + 1);
                ctx.fillStyle = '#e53935';
                ctx.beginPath();
                ctx.arc(clampedX, clampedY, delR, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = '#fff';
                ctx.font = 'bold 13px system-ui, -apple-system, sans-serif';
                ctx.textBaseline = 'middle';
                ctx.textAlign = 'center';
                ctx.fillText('\u00d7', clampedX, clampedY);
                ctx.textAlign = 'start';

                this._deleteBtn = { x: clampedX, y: clampedY, r: delR };
            }
        });

        // Rubber-band rect while drawing
        if (this._mode === 'drawing' && this._drawRect) {
            const r = this._drawRect;
            const p1 = this._toPixel({ x: Math.min(r.x1, r.x2), y: Math.min(r.y1, r.y2) });
            const p2 = this._toPixel({ x: Math.max(r.x1, r.x2), y: Math.max(r.y1, r.y2) });
            ctx.setLineDash([5, 5]);
            ctx.strokeStyle = '#1976D2';
            ctx.lineWidth = 2;
            ctx.strokeRect(p1.x, p1.y, p2.x - p1.x, p2.y - p1.y);
            ctx.setLineDash([]);
        }
    }

    // -- Internal: helpers --

    _recolorBoxes() {
        this.boxes.forEach((b, i) => {
            b.color = BboxCanvas.COLORS[i % BboxCanvas.COLORS.length];
        });
    }

    _fireChange() {
        if (this._onChange) {
            this._onChange(this.getBoxes());
        }
    }
}

if (typeof window !== 'undefined') {
    window.BboxCanvas = BboxCanvas;
}
