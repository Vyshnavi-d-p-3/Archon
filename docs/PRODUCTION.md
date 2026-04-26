# Archon production deployment

This document closes common gaps between the built-in `ThreadingHTTPServer` dashboard and a production deployment. Treat it as a checklist, not a guarantee; tune it to your threat model and SLOs.

## What the app already provides

- Trace-driven **metrics** and **RAG Studio** (ingest, query, session reset) with server-side size limits and optional **bearer** auth for `/api/rag/*`.
- **Request IDs** (`X-Request-Id`) and **security headers** on HTTP responses.
- **Sliding-window rate limiting** for RAG routes (per process, keyed by client IP; honor `X-Forwarded-For` from your edge proxy).
- **Durable JSONL** append to `<traces-dir>/rag_store/ingests.jsonl` with POSIX `flock` on append (good for a single node; not a distributed store).
- **`GET /api/health`** for **liveness** (process up; includes version, RAG limiter configuration, and filesystem `checks`).
- **`GET /api/info`** for **service discovery** (version, `traces_dir`, and documented HTTP entry points; safe to cache briefly).
- **`GET /api/ready`** for **readiness** — **200** when `traces/` and `rag_store/` are writable, **503** otherwise (use for Kubernetes `readinessProbe`).
- **SIGTERM** triggers a graceful `server.shutdown()` (with **SIGINT** still stopping the process as before).
- **Optional CORS** via `ARCHON_CORS_ORIGIN` for browser clients on another origin (lock down to a single origin in production).
- **JSON audit lines** (default on): one line per request on the `archon.audit` logger with method, path (truncated), status, `request_id`, client IP, and duration — **never** request bodies or `Authorization` headers. Disable with `ARCHON_AUDIT_JSON=0`. Point your log shipper at process stdout.

## What you should add at the edge

1. **TLS** — Terminate HTTPS with nginx, Caddy, or Traefik; do not expose the app server directly on the public internet.
2. **Forward real client IP** — Set `X-Forwarded-For` (or `X-Real-IP`) at the reverse proxy so rate limits and any future logs reflect the client, not the proxy.
3. **Stronger RAG token handling** — Prefer server-side session cookies (HttpOnly, Secure) or a proper identity provider for anything beyond internal ops; browser `localStorage` + bearer is convenient but is sensitive to XSS.
4. **Distributed limits and WAF** — The built-in limiter is per process. Use your proxy, API gateway, or cloud WAF for global rate limits and DDoS protection.
5. **Secrets** — Set `ARCHON_DASHBOARD_TOKEN` via a secret manager, not committed files. Rotate on schedule or on incident.

## Environment reference

| Variable | Purpose |
|----------|---------|
| `ARCHON_DASHBOARD_TOKEN` | If set, `/api/rag/*` require `Authorization: Bearer …`. |
| `ARCHON_RAG_MAX_REQUEST_BYTES` | Max JSON body size (default `200000`). |
| `ARCHON_RAG_MAX_INGEST_CHARS` | Max characters per ingest `text` (default `50000`). |
| `ARCHON_RAG_RATE_MAX` | Max RAG API requests per IP per window (default `120`). |
| `ARCHON_RAG_RATE_WINDOW_SEC` | Sliding window length in seconds (default `60`). |
| `ARCHON_LOG_LEVEL` | e.g. `INFO`, `DEBUG` (default `INFO`). |
| `ARCHON_AUDIT_JSON` | `1` (default) enables one JSON audit line per request; `0` disables. |
| `ARCHON_CORS_ORIGIN` | If set, enables CORS for `/api/*` and `OPTIONS` preflight. Prefer one explicit origin. |
| `ARCHON_CORS_MAX_AGE` | Preflight `Access-Control-Max-Age` (default `86400`). |
| `ARCHON_DASHBOARD_METRICS_MAX` | Cap on **newest** trace JSON files read for `summary` KPIs (default `10000`, max `100000`). The HTTP `?limit=` still only caps the `traces` list. |

## RAG data model (current)

- **In-process** vector index and **on-disk** JSONL log. Suitable for a **single** dashboard instance with backup of `traces/` (and `traces/rag_store/` if used).
- For **HA / horizontal scale**, plan a **shared vector or SQL store** (e.g. pgvector, Qdrant, S3 + dedicated indexer) and replace the in-memory `RAGPipeline` wiring.

## Suggested `Dockerfile` / Kubernetes

- **Health check**: `GET /api/health` (expect 200, JSON with `"status": "ok"`).
- **Command**: `python main.py dashboard --host 0.0.0.0 --port 8787 --traces-dir /data/traces` with a **volume** for traces and RAG store.
- **User**: Run as **non-root** (image already uses `agent` in the default `Dockerfile`).
- **CD workflow**: pushes to `production` build/push a GHCR image via [`.github/workflows/cd.yml`](../.github/workflows/cd.yml). For Render, set repository secret `RENDER_DEPLOY_HOOK_URL` (from your Render service) to auto-trigger deploys after image publish.

## Render deployment (recommended)

Use [render.yaml](../render.yaml) as the blueprint baseline (service type, health check, env defaults, and Docker start command).

1. Create a **Web Service** in Render from this repository (Docker environment).
2. Set start command:
   - `python main.py dashboard --host 0.0.0.0 --port $PORT --traces-dir /data/traces`
3. Configure health check path:
   - `/api/health`
4. In GitHub repo secrets, add:
   - `RENDER_DEPLOY_HOOK_URL` = Render Deploy Hook URL (Render Dashboard → Settings → Deploy Hook)
5. Merge/push to `production`:
   - CD publishes GHCR image and triggers Render deploy hook automatically.

## Remaining product gaps (optional roadmap)

- **Log shipping** — forward `archon.audit` / process stdout to your SIEM; add alerts on 401/429 spikes or health check failures.
- **CORS** only if the UI is on another origin; keep it deny-by-default.
- **Citations + LLM answer** path (optional) for governed answers, not just retrieval snippets.

For repository hosting, see the main [README](../README.md) and your deployment platform’s hardening guide.
