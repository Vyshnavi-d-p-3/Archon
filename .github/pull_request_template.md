## Summary

What changed and why (1–3 sentences).

## Branches

Target branch follows [docs/BRANCHES.md](docs/BRANCHES.md) (`main` / `develop` / `production` / `feature/*`).

## Checklist

- [ ] `pytest tests/ -q` passes (CI runs on PRs to `main`, `develop`, `production`)
- [ ] No secrets or local-only paths (e.g. `traces/`, `evaluation/results/`) in the diff
