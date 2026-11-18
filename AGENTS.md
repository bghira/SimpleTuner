# AGENTS.md

## Project Dependencies

- Venv location: `.venv`
- Python version: `3.13`
- Test framework: `unittest` (NOT `pytest`)
- Test command: `.venv/bin/python -m unittest -v -f`
- Test average runtime: ~300 seconds
- Preferred diff tool: `difft --display=inline` (difftastic)
- Never commit or push on your own.

## Worktree management

- At task start, check `git status --short --branch` and `git worktree list`. Use an existing checkout for read-only reviews, CI checks, and GPU monitoring.
- Use one worktree per active change. Reuse it for follow-ups, tests, and turning an experiment into a PR. Create a new worktree only when there is no suitable checkout for the change or when independent work or an incompatible base requires isolation. Fetch the target remote before creating a new branch and worktree from the target branch, outside the main checkout.
- Run edits, builds, tests that write files, and Git mutations in the task worktree. Verify the path and branch before making changes. Keep the main checkout free of task changes. Preserve pre-existing changes in every checkout. If task changes land in the main checkout, move only your own changes and verify the main checkout matches its initial state. Ask if ownership is unclear.
- Keep experiment results, recovery scripts, and other files needed beyond the checkout in a private directory outside the repository and its worktrees. Record the path in the handoff. Before cleanup, move existing task-owned artifacts there and verify the archive is complete and readable. Leave other people's files in place.
- Clean up completed task worktrees at handoff, including finished experiments and closed PRs. For a merged PR, fetch the target and verify the merge includes the work, including squash and rebase merges. For work without a merge, verify that any unique commits remain on a local branch and that any uncommitted changes are preserved. Keep a worktree while further local changes or tests are pending. A fully pushed PR can keep its branch without keeping a checkout once local work is done.
- Before removal, check staged and unstaged changes, untracked and ignored files, and any processes that use the checkout. Remove only worktrees belonging to the task or identified by the user. Use `git worktree remove` without force. If cleanup would discard files or edits, keep the worktree and report the blocker. Prune verified stale metadata as needed. Delete a local branch only after verifying its work is merged or it has no unique commits. Leave remote branches in place unless the user authorizes deletion.
- During long tasks and at handoff, recheck `git worktree list` and `git status --short --branch` in the main checkout and any task worktrees still present. Verify the main checkout matches its initial state. Report any retained task worktree and the reason it still needs to exist.

## Code style

- Defensive programming should be justified and expected by users, not masking bugs
- DO not hide or mute import failures unless the logic actually requires it
- Use type: ignore only when absolutely necessary
- NEVER add a code fallback path unless it is explicit to the requirements
- Do not make assumptions if confusion arises. Instead, stop working, and request clarification.
- Let's not add wandering, rambling comments in notes. Be concise and to the point or leave no comment at all since the code should be self-explanatory.

## Plan inspection guidelines

- Always inspect a provided plan for deficiencies
- If a plan is for problem solving, it should contain a clear root cause analysis that can be verified
- A plan containing vague instructions should be rejected until it is thorough enough as to contain line numbers and proposed function names, potential pitfalls and known edge cases
- If the plan is proposing new code infrastructure, it should contain justification for which existing code paths are similar but insufficient enough that new code is required
- All proposals for new code should have as well, proposal for code removal or refactoring of existing code to avoid bloat and duplication - in general, our codebase should be kept as minimal as possible, so that it is easy to maintain, review, and understand for newcomers
- If fallbacks or defensive programming practices are proposed, they must be clearly justified: the fallback should be necessary and reasonable for a user to expect, not just "in case something goes wrong".
- Plans shouldn't "do too much", the user should be forced to break down complex problems into smaller steps that can be solved iteratively to avoid ambiguity.
- If a dataloader/dataset configuration option is being added, it should have an update in the corresponding Dataset template for the WebUI.

## File preservation

- Preserve untracked files. You may move task-owned files to a verified private archive during worktree cleanup. Delete untracked files only when explicitly instructed.

## Public text privacy

Never publish local machine identity in public text. This includes commit messages, PR titles, PR bodies, PR comments, issue comments, release notes, model cards, Hub commit messages, generated metadata, and validation summaries.

Forbidden public text includes:
- Local absolute home-directory paths
- Local account names or workstation usernames
- Co-author trailers (e.g., `Co-authored-by:`) that include personal names/emails
- Temp/cache paths that include local identity
- Raw terminal output containing local identity
- Pod/workspace paths unless they are intentionally public infrastructure paths
- Co-author, reviewer, or attribution trailers that expose local identity

Use repo-relative paths and generic commands instead:
- Good: `.venv/bin/python -m unittest -v -f tests.test_model_card`
- Good: `simpletuner/helpers/publishing/huggingface.py`
- Bad: any absolute local path from this machine

Before creating a commit, PR, PR comment, issue comment, Hub commit, model card, or any other public text, scan the exact text that will be sent. If the scan cannot be performed, stop and ask before publishing.

When a privacy guard blocks content, do not print the matched secret, username, path, or offending line. Report only: `Blocked: local machine identity was found in public text.`

Do not copy validation commands directly from shell history if they contain absolute paths. Rewrite them to repo-relative form first.

GitHub Actions are too late for PR body/comment privacy because notifications may already be sent. Client-side scanning is required before using `gh pr create`, `gh pr edit`, `gh pr comment`, connector PR tools, or any API that publishes text.

## Publishing workflow

Before any push or PR creation:
1. Inspect the staged diff.
2. Inspect the commit message.
3. Inspect the exact PR title/body/comment text.
4. Confirm none contain local machine identity.
5. Only then push or publish.

Do not include absolute local paths in validation notes. Use repo-relative commands.

## Problem solving

- It's always tempting to jump right into declaring an answer, but the best solutions come from carefully-developed understanding of the root cause
- Problems should always be provable through tests, logging, or other means
- Generally speaking, it's fine to run the full application end-to-end to verify a fix, "it's heavy" is not a valid excuse to avoid verification - we're on a ML development workstation that's designed to allow running these workloads
- For the most part, things should not be marked as CUDA-only unless it relies on third-party compiled CUDA kernels or similar. Don't be afraid to use the available accelerator eg. mps, cuda, if available on the system opportunistically.
- Do not assume checkpoint resume supports changing topology or dataset settings. Changing DDP/FSDP/CP layout, world size, batch sizing, gradient accumulation, repeats, bucket splitting, shuffling, or dataloader configuration between save and resume can produce unexpected behavior; avoid adding compatibility fallbacks unless support for that specific change is explicitly required and verified.

## Frontend testing

### Test types and their limitations

- **Jest/JSDOM tests** (`tests/js/`): Unit tests for JavaScript logic. They mock Alpine.js and DOM APIs. They do NOT test:
  - Alpine event modifiers (`.stop`, `.prevent`, `.self`)
  - DOM event bubbling/propagation
  - Real browser rendering or timing
  - Integration between Alpine templates and store logic

- **Selenium E2E tests** (`tests/test_webui_e2e.py`): Integration tests with real browsers. These ARE required for:
  - Event propagation behavior
  - Form dirty state tracking
  - Alpine template ↔ store integration
  - Any bug involving "it works after X but not on direct load"

### When to write E2E tests

Frontend changes involving event flow, form state, or Alpine reactivity MUST include Selenium E2E tests. Jest tests are insufficient - they test isolated functions, not the wired-up behavior.

Run E2E tests: `.venv/bin/python -m unittest tests.test_webui_e2e -v`

### Form dirty state testing checklist

The `formDirty` → save button flow has multiple failure modes. E2E tests must verify:
1. Direct page load → edit Easy Mode field → save button enables
2. Direct page load → edit main form field → save button enables
3. Tab switch → edit field → save button enables
4. Save → dirty state clears → new edit → save button re-enables

## Documentation

- We're using mkDocs, so be sure to update the custom index templates.
- Update all translations for any existing documentation when we're making changes.
- Create translations for zh, ja, pt-BR, es, and hindi when creating *new* documentation.
- If there's a new option being added, we should add it to the OPTIONS.md and all of its translations.
- If a new dataloader setting is being added, it should be in DATALOADER.md and all of its translations.
- Dataloader dataset keys are not necessarily CLI options. Discover dataset config fields from `simpletuner/helpers/data_backend/config/`, `simpletuner/simpletuner_sdk/server/data/dataset_blueprints.py`, and `simpletuner/templates/components/dataloader/` before deciding whether `OPTIONS.md` is relevant. For dataclass fields, use `.venv/bin/python -c "from dataclasses import fields; from simpletuner.helpers.data_backend.config import ImageBackendConfig; print([f.name for f in fields(ImageBackendConfig)])"`.
