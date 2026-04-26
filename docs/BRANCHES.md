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

### Still no graph? Checklist

**A. Profile: green “contributions” calendar**

1. **Author email must be on your GitHub account** — The email in each commit’s **Author** field must be listed under [GitHub → Settings → Emails](https://github.com/settings/emails) (and verified, if GitHub required it). Check locally:
   ```bash
   git log -1 --format='%ae  %an'   # should match a verified address (or your GitHub noreply).
   ```
   To use GitHub’s private address: `USERNAME@users.noreply.github.com` (see **Settings → Emails** — GitHub shows the exact value).

2. **Backdated or rewritten history** — If you used tools that set **author dates** in the **past** (e.g. all commits dated in 2025), those contributions appear on **that year’s** calendar, not the current year. Open your profile’s contribution graph and **go back a year** (or the year of your commit dates) to see squares.

3. **Current year looks empty** — You need at least one **new** commit, authored **this year** with a **valid email**, and pushed to the **default branch** of a **non-fork** repo (or a merged PR) for squares to show in the **current** year. Make a small doc or chore commit, push to `main`, wait up to **24 hours**.

4. **Private contributions** — Under [Profile → Settings](https://github.com/settings/profile), enable **“Include private contributions on my profile”** if the repo is private and you want them counted.

5. **Forks** — Commits on a **fork** do not add to your profile until they are **merged** into the parent (or in some cases the default rules — see [GitHub’s help](https://docs.github.com/en/account-and-profile/setting-up-and-managing-your-github-profile/managing-contribution-graphs-on-your-profile/why-are-my-contributions-not-showing-up-on-my-profile)).

**B. Repo list: small “activity” line next to the repo name**

- Tied to the **default branch**; can lag **several hours** after pushes.
- **One huge push in a single day** can look like a very **flat** line until you have more days of pushes.
- If you **rebased / force-pushed** the whole history recently, the UI can take time to catch up.

**C. Quick test**

```bash
git config user.name "Your Name"
git config user.email "YOUR_VERIFIED_EMAIL_OR_NOREPLY"
# tiny commit on main, push — then recheck the graph after a day
```

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

**CI** — [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs `pytest` on **push and pull request** to `main`, `develop`, and `production` (Python 3.11 and 3.12).

**Automation** — [Dependabot](https://docs.github.com/en/code-security/dependabot) updates pip and GitHub Actions (see [`.github/dependabot.yml`](../.github/dependabot.yml)). New PRs use the template in [`.github/pull_request_template.md`](../.github/pull_request_template.md).
