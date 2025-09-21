"""NiceGUI theme helpers for legacy SimpleTuner styling."""
from __future__ import annotations

from nicegui import ui

_INITIALISED = False


def ensure_legacy_theme() -> None:
    """Inject Bootstrap/FontAwesome assets and legacy CSS, once per process."""
    global _INITIALISED
    if _INITIALISED:
        return

    ui.add_head_html(
        """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
<link href="/static/css/base.css" rel="stylesheet">
<link href="/static/css/trainer.css" rel="stylesheet">
<link href="/static/css/trainer-polish.css" rel="stylesheet">
<style>
    body.dark-theme {
        background-color: #0a0a0a;
        color: #e5e7eb;
    }
    .nicegui-content {
        background-color: transparent;
    }
    .q-field.legacy-input .q-field__control,
    .q-select.legacy-input .q-field__control,
    .q-input.legacy-input .q-field__control {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 0.75rem;
        background-color: rgba(16,18,27,0.9);
        min-height: 44px;
        padding: 0 0.75rem;
        color: #f8fafc;
    }
    .q-field.legacy-input .q-field__native,
    .q-field.legacy-input input,
    .q-field.legacy-input textarea {
        color: #f8fafc;
        font-size: 0.95rem;
    }
    .q-field.legacy-input .q-field__bottom {
        display: none;
    }
    .q-field.legacy-input .q-field__label {
        display: none;
    }
    .q-checkbox.legacy-checkbox .q-checkbox__bg {
        border-radius: 6px;
        background-color: rgba(16,18,27,0.6);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .q-checkbox.legacy-checkbox .q-checkbox__inner--truthy .q-checkbox__bg {
        background-color: #38bdf8;
    }
    .legacy-form-check {
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .legacy-form-check .q-checkbox {
        margin-bottom: 0;
    }
    .dashboard-wrapper {
        min-height: calc(100vh - 64px);
    }
    .legacy-select-popup {
        background-color: rgba(16,18,27,0.95);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 0.75rem;
        color: #f8fafc;
    }
    .legacy-select-popup .q-item {
        padding: 0.45rem 0.9rem;
    }
    .legacy-select-popup .q-item__section--main {
        font-size: 0.9rem;
    }
    .dataset-summary-card {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 0.9rem;
        padding: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    .dataset-summary-card .dataset-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #e2e8f0;
        font-weight: 600;
    }
    .dataset-summary-card .dataset-meta {
        color: #94a3b8;
        font-weight: 500;
    }
    .dataset-summary-card .dataset-row {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        font-size: 0.82rem;
        color: #cbd5f5;
    }
    .dataset-summary-card .dataset-row span:first-child {
        color: #94a3b8;
    }
    .status-message {
        padding: 0.65rem 0.9rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(29,35,48,0.65);
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .status-success {
        background: rgba(21,128,61,0.45);
        border-color: rgba(34,197,94,0.4);
        color: #bbf7d0;
    }
    .status-error {
        background: rgba(127,29,29,0.45);
        border-color: rgba(248,113,113,0.4);
        color: #fecaca;
    }
    .status-warning {
        background: rgba(113,63,18,0.45);
        border-color: rgba(250,204,21,0.35);
        color: #fef08a;
    }
    .status-info {
        background: rgba(30,41,59,0.6);
        border-color: rgba(148,163,184,0.25);
    }
</style>
"""
    )
    ui.add_head_html(
        """
<script>
    document.addEventListener('DOMContentLoaded', () => {
        document.documentElement.classList.add('dark-theme');
        document.body.classList.add('dark-theme');
    });
</script>
"""
    )
    ui.add_body_html(
        """
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    window.__simpletunerInitSidebar = function() {
        const toggles = document.querySelectorAll('.sidebar-toggle');
        const sidebar = document.getElementById('sidebar');
        if (!sidebar) return;
        const toggleSidebar = () => sidebar.classList.toggle('show');
        toggles.forEach(btn => {
            btn.removeEventListener('click', toggleSidebar);
            btn.addEventListener('click', toggleSidebar);
        });
        document.addEventListener('click', (event) => {
            if (!sidebar.contains(event.target) && !event.target.closest('.sidebar-toggle')) {
                sidebar.classList.remove('show');
            }
        });
    }
</script>
"""
    )

    _INITIALISED = True
