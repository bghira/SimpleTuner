# SimpleTuner WebUI implementation details

## Design Overview

SimpleTuner's API is built with flexibility in mind.

In `trainer` mode, a single port is opened for FastAPI integration into other services.
With `unified` mode, an additional port is opened for the WebUI to receive callback events from a remote `trainer` process.

## Web Framework

A WebUI is built and served using FastAPI;

- Alpine.js is used for reactive components
- HTMX is used for dynamic content loading and interactivity
- FastAPI with Starlette and SSE-Starlette are used to serve a data-centric API with server-sent events (SSE) for real-time updates
- Jinja2 is used for HTML templating

Alpine was chosen for its simplicity and ease of integration - it keeps NodeJS out of the stack, making it easier to deploy and maintain.

HTMX had simple and lightweight syntax that paired well with Alpine. It provides extensive capabilities for dynamic content loading, form handling, and interactivity without the need for a full frontend framework.

I've selected Starlette and SSE-Starlette because I wanted to keep code duplication to a minimum; a lot of refactoring was required to move from a very ad-hoc scattering of procedural code to a more declarative approach.

### Data Flow

Historically the FastAPI app doubled as a “service worker” inside job clusters: the trainer booted, exposed a narrow callback surface, and remote orchestrators streamed status updates back over HTTP. The WebUI simply reuses that same callback bus. In unified mode we run both the trainer and the interface in-process, while trainer-only deployments can still push events into `/callbacks` and let a separate WebUI instance consume them over SSE. No new queue or broker needed – we are leaning on the infrastructure that already ships with headless deployments.

## Backend Architecture

The trainer UI rides on top of the core SDK that now exposes well-defined services instead of loose procedural helpers. FastAPI still terminates every request, but most routes are thin delegators into service objects. This keeps the HTTP layer dumb and maximises reusability for the CLI, config wizard, and future APIs.

### Route handlers

`simpletuner/simpletuner_sdk/server/routes/web.py` wires the `/web/trainer` surface. There are only two interesting endpoints:

- `trainer_page` – renders the outer chrome (navigation, config selector, tabs list). It asks `TabService` for metadata and pipes everything into the `trainer_htmx.html` template.
- `render_tab` – a generic HTMX target. Every tab button hits this endpoint with the tab name; the route resolves the matching layout through `TabService.render_tab` and returns the HTML chunk.

The rest of the HTTP router set lives under `simpletuner/simpletuner_sdk/server/routes/` and follows the same pattern: business logic sits in a service module, the route extracts params, calls the service, and turns the result into JSON or HTML.

### TabService

`TabService` is the central orchestrator for the training form. It defines:

- static metadata for each tab (title, icon, template, optional context hook)
- `render_tab()` which
  1. grabs the tab config (template, description)
  2. asks `FieldService` for the field bundle belonging to the tab/section
  3. injects any tab-specific context (datasets list, GPU inventory, onboarding state)
  4. returns a Jinja render of `form_tab.html`, `datasets_tab.html`, etc.

By pushing this logic into a class we can reuse the exact same rendering for HTMX, the CLI wizard, and tests. Nothing in the template reaches into global state anymore – everything is supplied explicitly through context.

### FieldService and FieldRegistry

`FieldService` converts registry entries into template-ready dictionaries. Responsibilities:

- filter fields by platform/model context (eg. hide CUDA-only knobs on MPS machines)
- evaluate dependency rules (`FieldDependency`) so the UI can disable or hide controls (for example Dynamo extras remain greyed out until a backend is selected)
- enrich fields with hints, dynamic choices, display formatting, and column classes

It delegates the raw catalog of fields to `FieldRegistry`, which is a declarative listing under `simpletuner/simpletuner_sdk/server/services/field_registry`. Each `ConfigField` describes CLI flag names, validation rules, importance ordering, dependency metadata, and default UI copy. This arrangement lets other layers (CLI parser, API, documentation generator) share the same source of truth while presenting it in their own format.

### State persistence and onboarding

The WebUI stores lightweight preferences via `WebUIStateStore`. It reads defaults from `$SIMPLETUNER_WEB_UI_CONFIG` (or an XDG path) and exposes:

- theme, dataset root, output dir defaults
- onboarding checklist state per feature
- cached Accelerate overrides (only whitelisted keys such as `--num_processes`, `--dynamo_backend`)

Those values are injected into the page during the initial `/web/trainer` render so Alpine stores can bootstrap without extra round-trips.

### HTMX + Alpine interaction

Every settings panel is just a chunk of HTML with `x-data` for Alpine behaviour. Tab buttons trigger HTMX GETs against `/web/trainer/tabs/{tab}`; the server responds with the rendered form and Alpine keeps the existing component state. A small helper (`trainer-form.js`) replays saved value changes so the user doesn’t lose in-progress edits when switching tabs.

Server updates (training status, GPU telemetry) flow through SSE endpoints (`sse_manager.js`) and drop data into Alpine stores that drive toasts, progress bars, and system banners.

### File layout cheatsheet

- `templates/` – Jinja templates; `partials/form_field.html` renders individual controls. `partials/form_field_htmx.html` is the HTMX-friendly variant used when a wizard needs two-way binding.
- `static/js/modules/` – Alpine component scripts (trainer form, hardware inventory, dataset browser).
- `static/js/services/` – shared helpers (dependency evaluation, SSE manager, event bus).
- `simpletuner/simpletuner_sdk/server/services/` – backend service layer (fields, tabs, configs, datasets, maintenance, events).

Together this keeps the WebUI stateless on the server side, with stateful bits (form data, toasts) living in the browser. The backend sticks to pure data transforms which makes testing easier and avoids threading issues when the trainer and web server run in the same process.
