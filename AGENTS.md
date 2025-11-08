# AGENTS.md

## Project Dependencies

- Venv location: `.venv`
- Python version: `3.12`
- Test framework: `unittest` (NOT `pytest`)

## Code style

- Defensive programming should be justified and expected by users, not masking bugs
- DO not hide or mute import failures unless the logic actually requires it
- Use type: ignore only when absolutely necessary
- NEVER add a code fallback path unless it is explicit to the requirements
- Do not make assumptions if confusion arises. Instead, stop working, and request clarification.

## Plan inspection guidelines

- Always inspect a provided plan for deficiencies
- If a plan is for problem solving, it should contain a clear root cause analysis that can be verified
- A plan containing vague instructions should be rejected until it is thorough enough as to contain line numbers and proposed function names, potential pitfalls and known edge cases
- If the plan is proposing new code infrastructure, it should contain justification for which existing code paths are similar but insufficient enough that new code is required
- All proposals for new code should have as well, proposal for code removal or refactoring of existing code to avoid bloat and duplication - in general, our codebase should be kept as minimal as possible, so that it is easy to maintain, review, and understand for newcomers
- If fallbacks or defensive programming practices are proposed, they must be clearly justified: the fallback should be necessary and reasonable for a user to expect, not just "in case something goes wrong".
- Plans shouldn't "do too much", the user should be forced to break down complex problems into smaller steps that can be solved iteratively to avoid ambiguity.

## File preservation

- Do not remove untracked files from the repository unless explicitly instructed to do so
