# Contributing

- **Branching** — See [docs/BRANCHES.md](BRANCHES.md) for `main`, `develop`, `production`, and `feature/*`.
- **Own the diff.** Whether you wrote it by hand, refactored with a copilot, or mixed both, you are responsible for tests, style, and review feedback. No “the model did it” as an excuse in review.
- **Keep changes reviewable** — scope work in **branches and pull requests**: one main concern per PR (or a clearly split series) so the diff and discussion stay focused. Local commits are for your workflow; use rebase, fixups, or squash as needed so **merged history** on the default branch stays easy to follow.
- **Run tests** before you open or update a PR: `pytest tests/ -q` (or `make check` if the Makefile defines it).
- **Don’t add noise:** avoid filler comments that restate the code, cargo-cult docstrings, or README churn unless it helps the next maintainer.
- Traces and eval artifacts under `traces/` / `evaluation/results/` are usually local-only; do not commit secrets or API keys.

If you use AI tools, treat them like an advanced linter or sparring partner: you still need to read the patch and understand it.
