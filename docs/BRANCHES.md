# Branches: main, develop, production, and features

## Overview

| Branch | Role |
|--------|------|
| **`main`** | Default branch. Stable, review-ready code. All PRs that should “ship” merge here. |
| **`develop`** | Integration line for work that is not yet ready to promote to `main` (optional; useful for teams or batched work). If you are solo, you can work directly on `main` and still keep `develop` in sync, or use `develop` as the daily integration target. |
| **`production`** | What you **deploy** or call “live.” Often tracks `main` on each release, or lags by one step until you cut a release. You can fast-forward it to `main` when a version is going to production. |
| `feature/*` | Short-lived branches: `feature/x`, e.g. `feature/dashboard-ops`. Branch from `develop` (or from `main` if you are not using `develop` yet) and open a **PR** into that target. |

**Existing examples in this repo:** `feature/initial-dashboard`, `feature/dashboard`, `feature/observability` (milestone or topic pointers).

## Typical flow (when using all three)

1. `git checkout develop && git pull`
2. `git checkout -b feature/my-task`
3. Work, commit, push: `git push -u origin feature/my-task`
4. Open a PR: **→ `develop`** (or **→ `main`** if you skip `develop`)
5. After merge, periodically or on release: merge `develop` → `main`, then (if you use it) `main` → `production` or fast-forward `production` to the release tag.

Solo, minimal flow: branch `feature/…` from `main`, PR to `main`, and keep `develop` and `production` **fast-forwarded to `main`** so they do not fall behind if you are not using them yet.

## GitHub “activity” graph / sparkline

The small activity graph on your profile and repo list is driven by **commits on the default branch** of each repository (and your [contribution](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/managing-contribution-graphs-on-your-profile/why-are-my-contributions-not-showing-up-on-my-profile) graph uses your **verified** commit email and several other rules). To help a graph show up over time:

- Set **`user.email` (and `user.name`)** in git to an address that is **added and verified** on GitHub (Settings → Emails).
- Push work to the **default branch** of the repo (usually `main`); the sparkline is based on that branch’s **commit history and dates**.
- New or **force-rewritten** histories can take **hours** to show a stable spark; continued **regular pushes** make the line clearer.
- A single massive push in one day can make the sparkline look **flat** until more days have activity.

**Changing the default branch** (e.g. to `develop`) is under **Settings → General → Default branch** on GitHub. The sparkline for that repository follows the **default** branch you set.

## One-time setup: create and publish branches (local)

From a clean `main`:

```bash
git checkout main
git pull origin main
git branch develop
git branch production
git push -u origin develop production
# Optional: first-time feature branch
# git checkout -b feature/your-topic
# git push -u origin feature/your-topic
```

Keep `develop` and `production` current by merging or fast-forwarding from `main` when you are ready, or set branch protection in GitHub for `main` / `production` as needed.
