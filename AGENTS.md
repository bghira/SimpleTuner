# AGENTS.md

## Project Dependencies

- Venv location: `.venv`
- Python version: `3.12`
- Test framework: `unittest` (NOT `pytest`)
- Test command: `.venv/bin/python -m unittest -v -f`
- Test average runtime: ~300 seconds

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

## File preservation

- Do not remove untracked files from the repository unless explicitly instructed to do so

## Problem solving

- It's always tempting to jump right into declaring an answer, but the best solutions come from carefully-developed understanding of the root cause
- Problems should always be provable through tests, logging, or other means
- Generally speaking, it's fine to run the full application end-to-end to verify a fix, "it's heavy" is not a valid excuse to avoid verification - we're on a ML development workstation that's designed to allow running these workloads
- For the most part, things should not be marked as CUDA-only unless it relies on third-party compiled CUDA kernels or similar. Don't be afraid to use the available accelerator eg. mps, cuda, if available on the system opportunistically.
