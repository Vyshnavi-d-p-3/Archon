"""Lightweight dashboard server for Archon trace visualization."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import time
import tomllib
import uuid
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent.state import FailureCategory
from scripts.dashboard_reliability import SlidingWindowRateLimiter, append_jsonl_line

# Reflector / metrics failure labels that are treated as AI safety & trust in aggregates.
_SAFETY_FAILURE_KEYS: frozenset[str] = frozenset(
    c.value
    for c in (
        FailureCategory.POLICY_VIOLATION,
        FailureCategory.UNSAFE_OUTPUT,
        FailureCategory.PROMPT_INJECTION,
        FailureCategory.PII_OR_SECRETS_RISK,
        FailureCategory.UNGROUNDED_CLAIM,
    )
)

_log = logging.getLogger("archon.dashboard")
_audit = logging.getLogger("archon.audit")

HTML_PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Archon — Operations</title>
    <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
    <style>
      *, *::before, *::after { box-sizing: border-box; }
      button, input, select, textarea { font: inherit; }
      button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
        outline: 2px solid #2563eb; outline-offset: 2px;
      }
      @media (max-width: 720px) {
        .archon-explore-grid { grid-template-columns: 1fr !important; }
      }
    </style>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  </head>
  <body style="margin:0;background:#f1f5f9;">
    <div id="root"></div>
    <script type="text/babel">
      const { useEffect, useState, useMemo } = React;
      const ThemeCtx = React.createContext("light");

      const Light = {
        bg: "#f1f5f9", card: "#ffffff", text: "#0f172a", subtext: "#64748b", text2: "#334155", muted: "#94a3b8", border: "#e2e8f0",
        primary: "#2563eb", success: "#16a34a", warning: "#d97706", danger: "#dc2626", info: "#0d9488", violet: "#7c3aed",
        inputBg: "#ffffff", shellGrad: "radial-gradient(1200px 600px at 20% 0%, #eef2ff 0%, transparent 50%), linear-gradient(180deg, #f8fafc 0%, #f1f5f9 45%)",
        cardShadow: "0 1px 2px rgba(15, 23, 42, 0.04), 0 4px 16px rgba(15, 23, 42, 0.05)", trustInner: "linear-gradient(145deg, #ffffff 0%, #f8fafc 100%)",
        statBg: "rgba(248, 250, 252, 0.8)", statBg2: "#fafbfc", barTrack: "rgba(15, 23, 42, 0.07)", timelineBtn: "linear-gradient(180deg, #fff, #fafbfc)", recItem: "linear-gradient(90deg, #f8fafc, #fff)", segBar: "rgba(241, 245, 249, 0.9)", kpiPanel: "#ffffff",
        chip: {
          default: { bg: "rgba(255,255,255,0.8)", b: "rgba(226, 232, 240, 0.9)", c: "#475569" },
          warn: { bg: "rgba(255, 251, 235, 0.95)", b: "rgba(251, 191, 36, 0.45)", c: "#a16207" },
          bad: { bg: "rgba(254, 242, 242, 0.9)", b: "rgba(252, 165, 165, 0.5)", c: "#b91c1c" },
          good: { bg: "rgba(220, 252, 231, 0.85)", b: "rgba(134, 239, 172, 0.5)", c: "#166534" },
        },
      };
      const Dark = {
        bg: "#0a0d14", card: "#111827", text: "#f1f5f9", subtext: "#94a3b8", text2: "#cbd5e1", muted: "#64748b", border: "rgba(51, 65, 85, 0.55)",
        primary: "#60a5fa", success: "#4ade80", warning: "#fbbf24", danger: "#f87171", info: "#2dd4bf", violet: "#a78bfa",
        inputBg: "#0f172a", shellGrad: "radial-gradient(900px 500px at 20% 0%, rgba(99, 102, 241, 0.2) 0%, transparent 55%), linear-gradient(180deg, #0f172a 0%, #0a0d14 50%)",
        cardShadow: "0 4px 24px rgba(0, 0, 0, 0.45)", trustInner: "linear-gradient(145deg, #1e293b 0%, #0f172a 100%)",
        statBg: "rgba(15, 23, 42, 0.55)", statBg2: "rgba(30, 41, 59, 0.4)", barTrack: "rgba(148, 163, 184, 0.12)", timelineBtn: "linear-gradient(180deg, #1e293b, #0f172a)", recItem: "linear-gradient(90deg, #0f172a, #1e293b)", segBar: "rgba(15, 23, 42, 0.85)", kpiPanel: "rgba(17, 24, 39, 0.65)",
        chip: {
          default: { bg: "rgba(30, 41, 59, 0.6)", b: "rgba(51, 65, 85, 0.6)", c: "#cbd5e1" },
          warn: { bg: "rgba(120, 53, 15, 0.35)", b: "rgba(245, 158, 11, 0.35)", c: "#fde68a" },
          bad: { bg: "rgba(127, 29, 29, 0.35)", b: "rgba(248, 113, 113, 0.4)", c: "#fecaca" },
          good: { bg: "rgba(20, 83, 45, 0.35)", b: "rgba(74, 222, 128, 0.35)", c: "#bbf7d0" },
        },
      };

      function useP() { return React.useContext(ThemeCtx) === "dark" ? Dark : Light; }
      function cardS(p) {
        return { background: p.card, borderRadius: 16, border: `1px solid ${p.border}`, boxShadow: p.cardShadow, padding: 20 };
      }
      function inSm(p) {
        return { border: `1px solid ${p.border}`, borderRadius: 10, padding: "8px 12px", background: p.inputBg, fontSize: 13, color: p.text, outline: "none" };
      }
      function btnPrimary() {
        return { border: "none", color: "#fff", background: "linear-gradient(180deg, #3b82f6, #2563eb)", borderRadius: 10, padding: "8px 16px", cursor: "pointer", fontSize: 13, fontWeight: 700, boxShadow: "0 1px 2px rgba(37, 99, 235, 0.35)" };
      }
      function btnSecondary(p) {
        return { border: `1px solid ${p.border}`, background: p.inputBg, color: p.text2, borderRadius: 10, padding: "7px 14px", cursor: "pointer", fontSize: 13, fontWeight: 600 };
      }
      function btnDanger(p) {
        return { border: "1px solid rgba(220, 38, 38, 0.45)", background: p.inputBg, color: p === Dark ? "#fca5a5" : "#b91c1c", borderRadius: 10, padding: "7px 14px", cursor: "pointer", fontSize: 13, fontWeight: 600 };
      }
      function btnTeal() {
        return { border: "1px solid #0d9488", color: "#fff", background: "linear-gradient(180deg, #14b8a6, #0d9488)", borderRadius: 10, padding: "8px 16px", cursor: "pointer", fontSize: 13, fontWeight: 700 };
      }
      function shellS(p) {
        return { minHeight: "100vh", background: p.shellGrad, color: p.text, fontFamily: "Inter, system-ui, -apple-system, Segoe UI, sans-serif", WebkitFontSmoothing: "antialiased" };
      }

      function MetaChip({ children, tone = "default" }) {
        const p = useP();
        const t = p.chip[tone] || p.chip.default;
        return <span style={{ display: "inline-flex", alignItems: "center", fontSize: 11, fontWeight: 600, color: t.c, background: t.bg, border: `1px solid ${t.b}`, borderRadius: 999, padding: "5px 11px" }}>{children}</span>;
      }

      /** All values are direct aggregates from loaded traces (no composite scores). */
      function MeasuredOutcomesCard({ stepCompletionRate, tracePassRate, retries, p95 }) {
        const p = useP();
        const c = stepCompletionRate;
        const statusLabel = c >= 90 ? "Healthy steps" : c >= 70 ? "Some step issues" : "Many step issues";
        const tone = c >= 90 ? p.success : c >= 70 ? p.warning : p.danger;
        return (
          <div
            style={{
              display: "grid",
              gap: 14,
              background: p.trustInner,
              border: `1px solid ${p.border}`,
              borderRadius: 14,
              padding: 18,
              minHeight: "100%",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
              <div>
                <div style={{ color: p.muted, fontSize: 10, fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}>Step completion rate</div>
                <div style={{ fontSize: 32, fontWeight: 800, letterSpacing: "-0.03em", lineHeight: 1.1, marginTop: 4, color: p.text }}>{stepCompletionRate.toFixed(1)}<span style={{ color: p.muted, fontSize: 18, fontWeight: 600 }}>%</span></div>
                <div style={{ color: p.subtext, fontSize: 11, marginTop: 4 }}>Share of steps with status &quot;completed&quot; in the metrics window</div>
              </div>
              <div style={{ padding: "5px 11px", borderRadius: 999, fontSize: 11, fontWeight: 700, color: tone, background: `${tone}20`, border: `1px solid ${tone}40` }}>{statusLabel}</div>
            </div>
            <div style={{ height: 6, borderRadius: 999, background: p.barTrack }}>
              <div style={{ height: 6, borderRadius: 999, width: `${Math.min(100, c)}%`, background: `linear-gradient(90deg, ${p.primary}, ${tone})` }} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
              <div style={{ padding: "8px 0 0" }}><div style={{ color: p.muted, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>Trace pass</div><div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: p.text }}>{tracePassRate.toFixed(1)}%</div></div>
              <div style={{ padding: "8px 0 0" }}><div style={{ color: p.muted, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>Avg retries / trace</div><div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: p.text }}>{retries.toFixed(2)}</div></div>
              <div style={{ padding: "8px 0 0" }}><div style={{ color: p.muted, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.06em" }}>P95 step</div><div style={{ fontWeight: 700, fontSize: 15, marginTop: 2, color: p.text }}>{p95.toFixed(0)}ms</div></div>
            </div>
          </div>
        );
      }

      function StatCard({ title, value, sub, accent, compact }) {
        const p = useP();
        return (
          <div
            style={{
              background: compact ? p.statBg : p.statBg2,
              border: `1px solid ${p.border}`,
              borderLeft: `3px solid ${accent}`,
              borderRadius: 12,
              padding: compact ? "10px 12px" : "14px 16px",
            }}
          >
            <div style={{ color: p.muted, fontSize: 10, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em" }}>{title}</div>
            <div style={{ fontSize: compact ? 20 : 24, fontWeight: 800, marginTop: 4, letterSpacing: "-0.02em", color: p.text }}>{value}</div>
            {sub && <div style={{ color: p.subtext, fontSize: 11, marginTop: 4, lineHeight: 1.35 }}>{sub}</div>}
          </div>
        );
      }

      function SectionTitle({ title, subtitle, eyebrow }) {
        const p = useP();
        return (
          <div style={{ marginBottom: 14 }}>
            {eyebrow && <div style={{ color: p.muted, fontSize: 10, fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6 }}>{eyebrow}</div>}
            <div style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.02em", color: p.text }}>{title}</div>
            {subtitle && <div style={{ color: p.subtext, fontSize: 13, marginTop: 4, lineHeight: 1.45 }}>{subtitle}</div>}
          </div>
        );
      }

      function BarRows({ rows, color, valueSuffix = "", emptyText = "No data" }) {
        const p = useP();
        const max = Math.max(1, ...rows.map((r) => r.value));
        return (
          <div style={{ display: "grid", gap: 8 }}>
            {rows.length === 0 && <div style={{ color: p.subtext, fontSize: 12 }}>{emptyText}</div>}
            {rows.map((r) => (
              <div key={r.label}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
                  <span style={{ color: p.text2, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 260, fontWeight: 500 }}>{r.label}</span>
                  <span style={{ color: p.text, fontWeight: 700, fontSize: 12 }}>{r.value}{valueSuffix}</span>
                </div>
                <div style={{ height: 5, borderRadius: 999, background: p.barTrack }}>
                  <div style={{ height: 5, borderRadius: 999, width: `${(r.value / max) * 100}%`, background: color }} />
                </div>
              </div>
            ))}
          </div>
        );
      }

      function ModelComparison({ rows }) {
        const p = useP();
        return (
          <div style={cardS(p)}>
            <SectionTitle
              eyebrow="Traces"
              title="By model"
              subtitle="Trace-level pass rate and share of steps that completed in the window (direct counts, no index)"
            />
            <div style={{ display: "grid", gap: 12 }}>
              {rows.length === 0 && <div style={{ color: p.subtext, fontSize: 12 }}>No model comparison data yet</div>}
              {rows.map((m) => (
                <div key={m.model}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 5 }}>
                    <span style={{ fontWeight: 600, color: p.text }}>{m.model}</span>
                    <span style={{ color: p.text2 }}>Traces pass {m.success_rate.toFixed(0)}% · Steps done {m.step_completion_rate.toFixed(0)}%</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: p.muted, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>Trace pass</div>
                      <div style={{ height: 6, borderRadius: 999, background: p.barTrack }}>
                        <div style={{ width: `${m.success_rate}%`, background: p.primary, height: 6, borderRadius: 999 }} />
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 10, fontWeight: 700, color: p.muted, textTransform: "uppercase", letterSpacing: "0.05em", marginBottom: 4 }}>Step completion</div>
                      <div style={{ height: 6, borderRadius: 999, background: p.barTrack }}>
                        <div style={{ width: `${m.step_completion_rate}%`, background: p.info, height: 6, borderRadius: 999 }} />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      }

      function RecommendationPanel({ items }) {
        const p = useP();
        return (
          <div style={cardS(p)}>
            <SectionTitle eyebrow="Next steps" title="Recommendations" subtitle="Prioritized from aggregate signals" />
            <div style={{ display: "grid", gap: 6 }}>
              {items.length === 0 && <div style={{ color: p.subtext, fontSize: 13, padding: "4px 0" }}>No urgent actions. Aggregate signals look healthy.</div>}
              {items.map((text, i) => (
                <div key={i} style={{ border: `1px solid ${p.border}`, background: p.recItem, borderRadius: 10, padding: "12px 14px", fontSize: 13, color: p.text2, lineHeight: 1.5 }}>
                  <span style={{ fontWeight: 800, color: p.primary, marginRight: 8, fontSize: 11, opacity: 0.9 }}>{String(i + 1).padStart(2, "0")}</span>{text}
                </div>
              ))}
            </div>
          </div>
        );
      }

      function DistributionPills({ title, rows, tone }) {
        const p = useP();
        return (
          <div style={cardS(p)}>
            <SectionTitle eyebrow="Distribution" title={title} />
            <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
              {rows.length === 0 && <span style={{ color: p.subtext, fontSize: 12 }}>No data</span>}
              {rows.map((r) => (
                <span key={r.label} style={{ fontSize: 11, fontWeight: 600, padding: "6px 10px", borderRadius: 8, background: `${tone}16`, color: tone, border: `1px solid ${tone}30` }}>
                  {r.label} <span style={{ opacity: 0.85 }}>· {r.value}</span>
                </span>
              ))}
            </div>
          </div>
        );
      }

      function Timeline({ steps, onSelectStep }) {
        const p = useP();
        const maxLatency = Math.max(1, ...(steps || []).map((s) => s.latency || 0));
        return (
          <div style={{ display: "grid", gap: 8 }}>
            {(steps || []).map((s, i) => (
              <button
                key={s.id || i}
                onClick={() => onSelectStep && onSelectStep(s)}
                type="button"
                style={{ padding: "12px 14px", border: `1px solid ${p.border}`, borderRadius: 12, background: p.timelineBtn, color: p.text, width: "100%", textAlign: "left", cursor: "pointer", boxShadow: p === Dark ? "0 1px 2px rgba(0,0,0,0.2)" : "0 1px 1px rgba(15, 23, 42, 0.03)" }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontSize: 12, color: p.muted, fontWeight: 700, marginBottom: 4 }}>Step {i + 1}</div>
                    <div style={{ fontSize: 14, fontWeight: 600, lineHeight: 1.35, color: p.text }}>{s.desc}</div>
                    <div style={{ color: p.subtext, fontSize: 12, marginTop: 4 }}><span style={{ color: p.text2, fontWeight: 500 }}>{s.tool}</span> · {s.status} · {s.verdict}{s.failure ? ` · ${s.failure}` : ""}</div>
                  </div>
                  <div style={{ fontSize: 11, fontWeight: 700, color: p.text2, background: p === Dark ? "rgba(124, 58, 237, 0.2)" : "rgba(124, 58, 237, 0.08)", padding: "4px 8px", borderRadius: 8, flexShrink: 0 }}>{s.latency}ms</div>
                </div>
                <div style={{ height: 3, borderRadius: 999, background: p.barTrack, marginTop: 10 }}>
                  <div style={{ height: 3, borderRadius: 999, width: `${(s.latency / maxLatency) * 100}%`, background: "linear-gradient(90deg, #7c3aed, #4f46e5)" }} />
                </div>
              </button>
            ))}
          </div>
        );
      }

      function App() {
        const DASH_TRACE_LIMIT = 500;
        const [data, setData] = useState({ traces: [], summary: {}, meta: {} });
        const [selected, setSelected] = useState(0);
        const [error, setError] = useState("");
        const [initialLoading, setInitialLoading] = useState(true);
        const [activeView, setActiveView] = useState("overview");
        const [autoRefresh, setAutoRefresh] = useState(true);
        const [refreshMs, setRefreshMs] = useState(5000);
        const [modelFilter, setModelFilter] = useState("all");
        const [statusFilter, setStatusFilter] = useState("all");
        const [ragOnly, setRagOnly] = useState(false);
        const [searchQuery, setSearchQuery] = useState("");
        const [sortBy, setSortBy] = useState("recent");
        const [incidentOnly, setIncidentOnly] = useState(false);
        const [selectedStep, setSelectedStep] = useState(null);
        const [ragSource, setRagSource] = useState("dashboard_input");
        const [ragText, setRagText] = useState("");
        const [ragQuestion, setRagQuestion] = useState("");
        const [ragSessionId, setRagSessionId] = useState("");
        const [ragTopK, setRagTopK] = useState(3);
        const [ragBusy, setRagBusy] = useState(false);
        const [ragResult, setRagResult] = useState(null);
        const [ragSessionStats, setRagSessionStats] = useState(null);
        const [ragToken, setRagToken] = useState("");
        const [ragTokenVisible, setRagTokenVisible] = useState(false);
        const [ragAuthProbe, setRagAuthProbe] = useState(null);
        const [traceFiltersOpen, setTraceFiltersOpen] = useState(() => {
          try { return localStorage.getItem("archon_trace_filters_open") !== "0"; } catch (e) { return true; }
        });
        const [theme, setTheme] = useState(() => {
          try { return localStorage.getItem("archon_theme") === "dark" ? "dark" : "light"; } catch (e) { return "light"; }
        });
        const [helpOpen, setHelpOpen] = useState(false);
        const [health, setHealth] = useState(null);
        const [lastDataAt, setLastDataAt] = useState(null);

        const p = useMemo(() => (theme === "dark" ? Dark : Light), [theme]);

        useEffect(() => {
          try { localStorage.setItem("archon_trace_filters_open", traceFiltersOpen ? "1" : "0"); } catch (e) {}
        }, [traceFiltersOpen]);
        useEffect(() => {
          try { localStorage.setItem("archon_theme", theme); } catch (e) {}
          const pal = theme === "dark" ? Dark : Light;
          document.body.style.background = pal.bg;
          document.body.style.color = pal.text;
          try { document.documentElement.style.colorScheme = theme === "dark" ? "dark" : "light"; } catch (e) {}
        }, [theme]);

        const load = async (opts = {}) => {
          const { isInitial = false } = opts;
          try {
            if (isInitial) setInitialLoading(true);
            const res = await fetch(`/api/dashboard?limit=${DASH_TRACE_LIMIT}`);
            if (!res.ok) throw new Error("Failed to fetch dashboard data");
            const payload = await res.json();
            if (!payload.meta) payload.meta = {};
            setData(payload);
            setError("");
            setLastDataAt(new Date().toISOString());
          } catch (e) {
            setError(String(e));
          } finally {
            if (isInitial) setInitialLoading(false);
          }
        };

        const probeHealth = async () => {
          try {
            const res = await fetch("/api/health");
            if (!res.ok) {
              setHealth({ ok: false, ready: false, version: "?", checks: null });
              return;
            }
            const j = await res.json();
            setHealth({ ok: j.status === "ok", ready: !!(j.checks && j.checks.ready), version: j.version || "?", checks: j.checks || null });
          } catch (e) {
            setHealth({ ok: false, ready: false, version: "?", checks: null, err: String(e) });
          }
        };

        const ingestRag = async () => {
          const text = ragText.trim();
          if (!text) return;
          if (!ragSessionId.trim()) {
            setError("Session ID is required for RAG ingest.");
            return;
          }
          try {
            setRagBusy(true);
            const headers = { "Content-Type": "application/json" };
            if (ragToken.trim()) headers["Authorization"] = `Bearer ${ragToken.trim()}`;
            const res = await fetch("/api/rag/ingest", {
              method: "POST",
              headers,
              body: JSON.stringify({
                text,
                source: ragSource || "dashboard_input",
                session_id: ragSessionId.trim(),
              }),
            });
            if (!res.ok) throw new Error("Failed to ingest document");
            const payload = await res.json();
            setRagResult({ mode: "ingest", payload });
            setRagSessionStats(payload.session_stats || null);
            await load();
          } catch (e) {
            setError(String(e));
          } finally {
            setRagBusy(false);
          }
        };

        const askRag = async () => {
          const question = ragQuestion.trim();
          if (!question) return;
          if (!ragSessionId.trim()) {
            setError("Session ID is required for RAG query.");
            return;
          }
          try {
            setRagBusy(true);
            const headers = { "Content-Type": "application/json" };
            if (ragToken.trim()) headers["Authorization"] = `Bearer ${ragToken.trim()}`;
            const res = await fetch("/api/rag/ask", {
              method: "POST",
              headers,
              body: JSON.stringify({
                question,
                top_k: Number(ragTopK || 3),
                session_id: ragSessionId.trim(),
              }),
            });
            if (!res.ok) throw new Error("Failed to query RAG");
            const payload = await res.json();
            setRagResult({ mode: "ask", payload });
            setRagSessionStats(payload.session_stats || null);
          } catch (e) {
            setError(String(e));
          } finally {
            setRagBusy(false);
          }
        };

        const refreshSessionStats = async () => {
          const sid = ragSessionId.trim();
          if (!sid) return;
          try {
            const url = `/api/rag/session?session_id=${encodeURIComponent(sid)}`;
            const headers = {};
            if (ragToken.trim()) headers["Authorization"] = `Bearer ${ragToken.trim()}`;
            const res = await fetch(url, { headers });
            if (!res.ok) throw new Error("Failed to fetch session stats");
            const payload = await res.json();
            setRagSessionStats(payload);
          } catch (e) {
            setError(String(e));
          }
        };

        const probeRagAuth = async () => {
          const headers = {};
          if (ragToken.trim()) headers["Authorization"] = `Bearer ${ragToken.trim()}`;
          try {
            const res = await fetch("/api/rag/auth-check", { headers });
            if (!res.ok) {
              setRagAuthProbe({ ok: false, auth_configured: true, message: "Auth check failed" });
              return;
            }
            const j = await res.json();
            setRagAuthProbe(j);
          } catch (e) {
            setRagAuthProbe({ ok: false, auth_configured: true, message: String(e) });
          }
        };

        const resetSession = async () => {
          const sid = ragSessionId.trim();
          if (!sid) return;
          try {
            setRagBusy(true);
            const headers = { "Content-Type": "application/json" };
            if (ragToken.trim()) headers["Authorization"] = `Bearer ${ragToken.trim()}`;
            const res = await fetch("/api/rag/reset", {
              method: "POST",
              headers,
              body: JSON.stringify({ session_id: sid }),
            });
            if (!res.ok) throw new Error("Failed to reset RAG session");
            const payload = await res.json();
            setRagResult({ mode: "reset", payload });
            setRagSessionStats(payload.session_stats || null);
          } catch (e) {
            setError(String(e));
          } finally {
            setRagBusy(false);
          }
        };

        const exportCurrentTrace = () => {
          if (!current) return;
          const blob = new Blob([JSON.stringify(current, null, 2)], { type: "application/json;charset=utf-8" });
          const url = URL.createObjectURL(blob);
          const a = document.createElement("a");
          a.href = url;
          a.download = `archon-trace-${String(current.trace_id || "export").replace(/[^a-zA-Z0-9._-]/g, "_")}.json`;
          document.body.appendChild(a);
          a.click();
          a.remove();
          URL.revokeObjectURL(url);
        };

        const copyTraceId = async () => {
          if (!current || !current.trace_id) return;
          try {
            await navigator.clipboard.writeText(String(current.trace_id));
          } catch (e) {
            setError(String(e));
          }
        };

        useEffect(() => { load({ isInitial: true }); }, []);
        useEffect(() => { probeHealth(); }, []);
        useEffect(() => {
          if (!autoRefresh) return undefined;
          const id = setInterval(probeHealth, 120000);
          return () => clearInterval(id);
        }, [autoRefresh]);
        useEffect(() => {
          const key = "archon_rag_session_id";
          const fromStorage = window.localStorage.getItem(key);
          if (fromStorage) {
            setRagSessionId(fromStorage);
            return;
          }
          const created = `sess_${Date.now()}`;
          window.localStorage.setItem(key, created);
          setRagSessionId(created);
        }, []);
        useEffect(() => {
          const token = window.localStorage.getItem("archon_rag_api_token");
          if (token) setRagToken(token);
        }, []);
        useEffect(() => { probeRagAuth(); }, [ragToken]);
        useEffect(() => {
          if (ragSessionId.trim()) {
            window.localStorage.setItem("archon_rag_session_id", ragSessionId.trim());
          }
        }, [ragSessionId]);
        useEffect(() => {
          if (ragToken.trim()) {
            window.localStorage.setItem("archon_rag_api_token", ragToken.trim());
            return;
          }
          window.localStorage.removeItem("archon_rag_api_token");
        }, [ragToken]);
        useEffect(() => { refreshSessionStats(); }, [ragSessionId]);
        useEffect(() => {
          if (!autoRefresh) return undefined;
          const id = setInterval(load, refreshMs);
          return () => clearInterval(id);
        }, [autoRefresh, refreshMs]);

        const traces = data.traces || [];
        const summary = data.summary || {};
        const models = [...new Set(traces.map((t) => t.model))];
        const filtered = traces.filter((t) => {
          if (modelFilter !== "all" && t.model !== modelFilter) return false;
          if (statusFilter === "pass" && !t.success) return false;
          if (statusFilter === "fail" && t.success) return false;
          if (ragOnly && !(t.steps || []).some((s) => ["rag_ingest","rag_search","rag_context"].includes(s.tool))) return false;
          if (incidentOnly && t.success && (t.retries || 0) === 0 && (t.replans || 0) === 0) return false;
          if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            const inTask = String(t.task || "").toLowerCase().includes(q);
            const inTools = (t.steps || []).some((s) => String(s.tool || "").toLowerCase().includes(q));
            if (!inTask && !inTools) return false;
          }
          return true;
        });
        const sorted = useMemo(() => {
          const base = [...filtered];
          if (sortBy === "recent") return base;
          return base.sort((a, b) => {
            if (sortBy === "success") return Number(b.success) - Number(a.success);
            if (sortBy === "steps") return (b.total_steps || 0) - (a.total_steps || 0);
            if (sortBy === "latency") return (b.wall_time || 0) - (a.wall_time || 0);
            if (sortBy === "retries") return (b.retries || 0) - (a.retries || 0);
            return 0;
          });
        }, [filtered, sortBy]);
        const current = sorted[selected] || null;
        useEffect(() => {
          if (selected >= sorted.length) setSelected(0);
        }, [selected, sorted.length]);
        useEffect(() => {
          setSelectedStep(null);
        }, [selected, modelFilter, statusFilter, ragOnly, incidentOnly, searchQuery, sortBy]);

        const topTools = (summary.top_tools || []).map(([label, value]) => ({ label, value }));
        const failures = (summary.failure_taxonomy || []).map(([label, value]) => ({ label, value }));
        const modelRows = summary.model_comparison || [];
        const statusDist = (summary.status_distribution || []).map(([label, value]) => ({ label, value }));
        const verdictDist = (summary.verdict_distribution || []).map(([label, value]) => ({ label, value }));
        const recommendations = summary.recommendations || [];
        const stepCompletionRate = summary.step_completion_rate != null ? summary.step_completion_rate : 0;
        const totalStepCount = summary.total_step_count != null ? summary.total_step_count : 0;
        const safetyFailureSteps = summary.safety_failure_steps != null ? summary.safety_failure_steps : 0;
        const safetyTaggedStepRate = summary.safety_tagged_step_rate != null ? summary.safety_tagged_step_rate : 0;
        const meta = data.meta || {};
        const hasTraces = traces.length > 0;
        const showKpi = activeView === "overview" || activeView === "explorer";
        const kpiN = summary.trace_count != null ? summary.trace_count : meta.metrics_files_read != null ? meta.metrics_files_read : null;
        const tableN = meta.traces_in_response != null ? meta.traces_in_response : meta.traces_loaded;
        const segBtn = (id, label) => {
          const on = activeView === id;
          return (
            <button
              type="button"
              onClick={() => setActiveView(id)}
              style={{
                border: "none",
                background: on ? p.inputBg : "transparent",
                color: on ? p.primary : p.muted,
                borderRadius: 8,
                padding: "7px 14px",
                fontSize: 13,
                fontWeight: 700,
                cursor: "pointer",
                boxShadow: on ? (theme === "dark" ? "0 1px 4px rgba(0,0,0,0.45)" : "0 1px 3px rgba(15, 23, 42, 0.12)") : "none",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              {label}
            </button>
          );
        };

        return (
        <ThemeCtx.Provider value={theme}>
          <div style={shellS(p)}>
            <div style={{ maxWidth: 1280, margin: "0 auto", padding: "20px 22px 40px" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 20, marginBottom: 22, paddingBottom: 20, borderBottom: "1px solid rgba(148, 163, 184, 0.25)" }}>
                <div style={{ minWidth: 0, flex: "1 1 320px" }}>
                  <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8, marginBottom: 8 }}>
                    <span style={{ fontSize: 11, fontWeight: 800, color: p.muted, letterSpacing: "0.12em", textTransform: "uppercase" }}>Archon</span>
                    {meta.version && <MetaChip>v{meta.version}</MetaChip>}
                  </div>
                  <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0, letterSpacing: "-0.035em", lineHeight: 1.15, color: p.text }}>Operations</h1>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 12 }}>
                    {meta.server_time && <MetaChip>Server {meta.server_time}</MetaChip>}
                    {meta.traces_on_disk != null && kpiN != null && (
                      <MetaChip>KPIs: {kpiN} of {meta.traces_on_disk} files</MetaChip>
                    )}
                    {meta.metrics_omit_older && <MetaChip tone="warn">Newest {meta.metrics_file_cap || ""} only</MetaChip>}
                    {meta.traces_on_disk != null && <MetaChip>Table: {tableN} (max {DASH_TRACE_LIMIT})</MetaChip>}
                    {health && (
                      <MetaChip tone={health.ok && health.ready ? "good" : health.ok ? "warn" : "bad"}>
                        {health.ok ? (health.ready ? "Storage OK" : "Storage issue") : "API unreachable"}
                      </MetaChip>
                    )}
                    {lastDataAt && <MetaChip>Data: {new Date(lastDataAt).toLocaleString()}</MetaChip>}
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 10 }}>
                  <div style={{ display: "inline-flex", padding: 3, background: p.segBar, border: `1px solid ${p.border}`, borderRadius: 10 }}>
                    {segBtn("overview", "Overview")}
                    {segBtn("explorer", "Traces")}
                    {segBtn("rag", "RAG")}
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", justifyContent: "flex-end" }}>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: p.subtext, fontWeight: 600, cursor: "pointer" }}><input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />Auto-refresh</label>
                    <select value={String(refreshMs)} onChange={(e) => setRefreshMs(Number(e.target.value))} style={{ ...inSm(p), minWidth: 80 }}>
                      <option value="2000">2s</option><option value="5000">5s</option><option value="10000">10s</option>
                    </select>
                    <button type="button" onClick={() => { load({}); probeHealth(); }} style={btnPrimary()}>Refresh</button>
                    <button type="button" onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))} style={{ ...btnSecondary(p), fontSize: 12, padding: "6px 12px" }} title="Color theme" aria-pressed={theme === "dark"}>
                      {theme === "dark" ? "Light" : "Dark"}
                    </button>
                    <button type="button" onClick={() => setHelpOpen((v) => !v)} style={{ ...btnSecondary(p), fontSize: 12, padding: "6px 12px" }} title="Help & API" aria-expanded={helpOpen}>
                      {helpOpen ? "Close" : "Help"}
                    </button>
                  </div>
                </div>
              </div>
              {helpOpen && (
                <div style={{ ...cardS(p), marginBottom: 16, padding: "16px 18px" }}>
                  <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>Run agent → traces → dashboard</div>
                  <div style={{ color: p.text2, fontSize: 13, lineHeight: 1.6, maxWidth: 900 }}>
                    <p style={{ margin: "0 0 10px" }}><strong>CLI</strong> — <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, color: p.text, fontSize: 12 }}>python main.py run &quot;Your task&quot; --trace-output traces/run.json</code> then refresh this page.</p>
                    <p style={{ margin: "0 0 10px" }}><strong>APIs</strong> — <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, fontSize: 12 }}>GET /api/health</code> (liveness + checks), <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, fontSize: 12 }}>GET /api/ready</code> (readiness for k8s), <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, fontSize: 12 }}>GET /api/info</code> (discovery), <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, fontSize: 12 }}>GET /api/dashboard?limit=500</code>.</p>
                    <p style={{ margin: 0 }}>Use <strong>Overview</strong> for aggregate charts, <strong>Traces</strong> to inspect runs, <strong>RAG</strong> for retrieval without trace files. Set <code style={{ background: p.inputBg, padding: "2px 6px", borderRadius: 6, fontSize: 12 }}>ARCHON_DASHBOARD_TOKEN</code> to protect RAG routes.</p>
                  </div>
                </div>
              )}
              {error && (
                <div style={{
                  ...cardS(p),
                  borderColor: theme === "dark" ? "rgba(248, 113, 113, 0.4)" : "rgba(252, 165, 165, 0.6)",
                  color: theme === "dark" ? "#fecaca" : "#991b1b",
                  marginBottom: 16,
                  background: theme === "dark" ? "linear-gradient(90deg, rgba(69, 10, 10, 0.4), rgba(15, 23, 42, 0.9))" : "linear-gradient(90deg, rgba(254, 242, 242, 0.95), #fff)",
                  display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12,
                }}>
                  <div style={{ lineHeight: 1.5, fontSize: 14, fontWeight: 500, flex: 1, minWidth: 0 }}>{error}</div>
                  <button type="button" onClick={() => setError("")} style={{ ...btnSecondary(p), flexShrink: 0, fontSize: 12, padding: "4px 10px" }}>Dismiss</button>
                </div>
              )}

              {initialLoading && (
                <div style={{ ...cardS(p), marginBottom: 16, padding: 28, textAlign: "center" }}>
                  <div style={{ fontSize: 14, fontWeight: 600, color: p.subtext, marginBottom: 18 }}>Loading…</div>
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 8 }}>
                    {[0,1,2,3,4,5,6,7].map((i) => (
                      <div key={i} style={{ height: 56, borderRadius: 12, background: theme === "dark" ? "linear-gradient(90deg,#1e293b 0%,#0f172a 50%,#1e293b 100%)" : "linear-gradient(90deg,#e2e8f0 0%,#f8fafc 50%,#e2e8f0 100%)", backgroundSize: "200% 100%", animation: "pulse 1.2s ease-in-out infinite" }} />
                    ))}
                  </div>
                  <style>{"@keyframes pulse { 0% { background-position: 0% 0; } 100% { background-position: -200% 0; } }"}</style>
                </div>
              )}

              {!initialLoading && !hasTraces && (activeView === "overview" || activeView === "explorer") && (
                <div style={{ ...cardS(p), padding: 36, textAlign: "center", maxWidth: 520, margin: "0 auto 20px" }}>
                  <div style={{ width: 48, height: 48, margin: "0 auto 16px", borderRadius: 14, background: theme === "dark" ? "linear-gradient(145deg, #1e1b4b, #0f172a)" : "linear-gradient(145deg, #e0e7ff, #f1f5f9)", border: `1px solid ${p.border}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, color: p.muted }}>◆</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: p.text, marginBottom: 8, letterSpacing: "-0.02em" }}>No trace files yet</div>
                  <div style={{ color: p.subtext, fontSize: 14, lineHeight: 1.6 }}>
                    Run the agent to emit JSON under your traces directory, then refresh. The <strong>RAG</strong> tab works without traces. Example:
                    <pre style={{ textAlign: "left", background: p.inputBg, border: `1px solid ${p.border}`, borderRadius: 10, padding: 14, fontSize: 12, marginTop: 14, overflow: "auto", lineHeight: 1.45, color: p.text2 }}>python main.py run "Your task" --trace-output traces/run.json
python main.py run --mock "demo task" --trace-output traces/mock.json</pre>
                  </div>
                </div>
              )}

              {hasTraces && activeView === "explorer" && (
              <div style={{ marginBottom: 18, background: p.card, border: `1px solid ${p.border}`, borderRadius: 16, boxShadow: p.cardShadow, padding: "12px 16px" }}>
                <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>List</div>
                    <div style={{ fontSize: 12, color: p.subtext, fontWeight: 600 }}>{sorted.length} / {traces.length} after filters</div>
                    <button type="button" onClick={() => setTraceFiltersOpen((v) => !v)} style={{ ...btnSecondary(p), fontSize: 12, padding: "4px 10px" }}>{traceFiltersOpen ? "Hide filters" : "Show filters"}</button>
                  </div>
                </div>
                {traceFiltersOpen && (
                <div style={{ marginTop: 12, paddingTop: 12, borderTop: `1px solid ${p.border}` }}>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}>
                    <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)} style={{ ...inSm(p), minWidth: 120 }}>
                      <option value="all">All models</option>
                      {models.map((m) => <option key={m} value={m}>{m}</option>)}
                    </select>
                    <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ ...inSm(p), minWidth: 100 }}>
                      <option value="all">All status</option><option value="pass">Pass</option><option value="fail">Fail</option>
                    </select>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 6, border: `1px solid ${p.border}`, borderRadius: 10, padding: "6px 10px", background: p.statBg, fontSize: 12, color: p.text2 }}>
                      <input type="checkbox" checked={ragOnly} onChange={(e) => setRagOnly(e.target.checked)} />RAG
                    </label>
                    <label style={{ display: "inline-flex", alignItems: "center", gap: 6, border: `1px solid ${p.border}`, borderRadius: 10, padding: "6px 10px", background: p.statBg, fontSize: 12, color: p.text2 }}>
                      <input type="checkbox" checked={incidentOnly} onChange={(e) => setIncidentOnly(e.target.checked)} />Incidents
                    </label>
                    <input
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="Search task or tool…"
                      style={{ ...inSm(p), minWidth: 200, flex: "1 1 180px" }}
                    />
                    <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} style={{ ...inSm(p), minWidth: 150 }}>
                      <option value="recent">Recent</option>
                      <option value="latency">Wall time</option>
                      <option value="steps">Steps</option>
                      <option value="retries">Retries</option>
                      <option value="success">Success</option>
                    </select>
                    <button type="button" onClick={() => { setModelFilter("all"); setStatusFilter("all"); setRagOnly(false); setIncidentOnly(false); setSearchQuery(""); setSortBy("recent"); }} style={{ ...btnSecondary(p), fontSize: 12, color: p.primary, borderColor: "rgba(37, 99, 235, 0.25)" }}>Reset</button>
                  </div>
                </div>
                )}
                <div style={{ fontSize: 11, color: p.muted, marginTop: 8 }}>Charts: <strong>Overview</strong> · browse runs here</div>
              </div>
              )}

              {hasTraces && showKpi && !initialLoading && (
              <div style={{ background: p.kpiPanel, border: `1px solid ${p.border}`, borderRadius: 20, boxShadow: p.cardShadow, padding: "22px 22px 20px", marginBottom: 20 }}>
                <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 14 }}>Aggregate KPIs</div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 14 }}>
                  <MeasuredOutcomesCard
                    stepCompletionRate={stepCompletionRate}
                    tracePassRate={summary.success_rate || 0}
                    retries={summary.avg_retries || 0}
                    p95={summary.p95_step_latency_ms || 0}
                  />
                  <StatCard title="Success rate" value={`${(summary.success_rate || 0).toFixed(1)}%`} sub={`${summary.trace_count || 0} traces in window`} accent={p.success} />
                  <StatCard title="Avg wall time" value={`${(summary.avg_wall_time || 0).toFixed(2)}s`} sub="Mean per trace" accent={p.primary} />
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
                  <StatCard
                    compact
                    title="Safety-tagged steps"
                    value={totalStepCount > 0 ? `${safetyTaggedStepRate.toFixed(1)}%` : "—"}
                    sub={totalStepCount > 0 ? `${safetyFailureSteps} of ${totalStepCount} steps (reflector category)` : "No steps in window"}
                    accent={safetyFailureSteps > 0 ? p.danger : p.success}
                  />
                  <StatCard compact title="Avg steps" value={(summary.avg_steps || 0).toFixed(2)} sub="Depth" accent={p.violet} />
                  <StatCard compact title="Avg retries" value={(summary.avg_retries || 0).toFixed(2)} sub="Stability" accent={p.warning} />
                  <StatCard compact title="RAG step share" value={`${(summary.rag_step_share || 0).toFixed(1)}%`} sub="Of all steps" accent={p.info} />
                  <StatCard compact title="RAG success" value={`${(summary.rag_success_rate || 0).toFixed(1)}%`} sub={`${summary.rag_chunks_ingested || 0} chunks in`} accent={p.info} />
                </div>
              </div>
              )}

              {!initialLoading && activeView === "rag" && (
              <div style={{ ...cardS(p), marginBottom: 20 }}>
                <SectionTitle eyebrow="Retrieval" title="RAG studio" subtitle="Ingest text into the session index and ask questions with citations" />
                <div style={{ border: `1px solid ${p.border}`, borderRadius: 12, background: theme === "dark" ? "linear-gradient(180deg, #0f172a, #1e293b)" : "linear-gradient(180deg, #f8fafc, #f1f5f9)", padding: 14, marginBottom: 14 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, textTransform: "uppercase", letterSpacing: "0.08em" }}>API access</div>
                    {ragAuthProbe && (
                      <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
                        <span style={{
                          fontSize: 11,
                          fontWeight: 600,
                          padding: "4px 8px",
                          borderRadius: 999,
                          background: ragAuthProbe.ok ? "#dcfce7" : "#fee2e2",
                          color: ragAuthProbe.ok ? "#166534" : "#b91c1c",
                          border: `1px solid ${ragAuthProbe.ok ? "#bbf7d0" : "#fecaca"}`,
                        }}>{ragAuthProbe.ok ? "Auth OK" : "Auth issue"}</span>
                        {ragAuthProbe.auth_configured === false && (
                          <span style={{ fontSize: 11, color: p.muted }}>Server has no token lock</span>
                        )}
                        <button type="button" onClick={probeRagAuth} style={{ ...btnSecondary(p), fontSize: 11, padding: "4px 10px" }}>Recheck</button>
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      type={ragTokenVisible ? "text" : "password"}
                      value={ragToken}
                      onChange={(e) => setRagToken(e.target.value)}
                      placeholder="Bearer token (set ARCHON_DASHBOARD_TOKEN server-side)"
                      style={{ ...inSm(p), flex: 1, minWidth: 120 }}
                    />
                    <button type="button" onClick={() => setRagTokenVisible((v) => !v)} style={{ ...btnSecondary(p), padding: "7px 12px" }}>
                      {ragTokenVisible ? "Hide" : "Show"}
                    </button>
                    <button type="button" onClick={() => setRagToken("")} style={{ ...btnSecondary(p), padding: "7px 12px" }}>
                      Clear
                    </button>
                  </div>
                  <div style={{ color: p.muted, fontSize: 11, marginTop: 6 }}>
                    Token is stored in browser localStorage and attached to RAG API requests.
                    {ragAuthProbe && ragAuthProbe.message && (
                      <span style={{ display: "block", marginTop: 4, color: ragAuthProbe.ok ? p.text2 : p.danger }}>{ragAuthProbe.message}</span>
                    )}
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 320px), 1fr))", gap: 14 }}>
                  <div style={{ border: `1px solid ${p.border}`, borderRadius: 14, padding: 14, background: p.card, boxShadow: p.cardShadow }}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>Ingest</div>
                    <input
                      value={ragSessionId}
                      onChange={(e) => setRagSessionId(e.target.value)}
                      placeholder="Session id (e.g. prod_ops_team)"
                      style={{ width: "100%", ...inSm(p), marginBottom: 8 }}
                    />
                    <input
                      value={ragSource}
                      onChange={(e) => setRagSource(e.target.value)}
                      placeholder="Source id (e.g. report_q2.txt)"
                      style={{ width: "100%", ...inSm(p), marginBottom: 8 }}
                    />
                    <textarea
                      value={ragText}
                      onChange={(e) => setRagText(e.target.value)}
                      placeholder="Paste text to ingest into RAG memory..."
                      rows={5}
                      style={{ width: "100%", ...inSm(p), resize: "vertical", lineHeight: 1.45 }}
                    />
                    <div style={{ marginTop: 8, display: "flex", flexWrap: "wrap", gap: 8 }}>
                      <button type="button" disabled={ragBusy} onClick={ingestRag} style={{ ...btnTeal(), opacity: ragBusy ? 0.55 : 1 }}>
                        {ragBusy ? "Working…" : "Ingest"}
                      </button>
                      <button type="button" disabled={ragBusy} onClick={resetSession} style={{ ...btnDanger(p), opacity: ragBusy ? 0.55 : 1 }}>
                        Reset session
                      </button>
                    </div>
                    {ragSessionStats && (
                      <div style={{ marginTop: 10, border: `1px solid ${p.border}`, borderRadius: 8, background: p.inputBg, padding: 8, fontSize: 12, color: p.text2 }}>
                        Session stats: {ragSessionStats.session_id} · chunks {ragSessionStats.total_chunks || 0} · ingests {ragSessionStats.ingest_count || 0}
                      </div>
                    )}
                  </div>

                  <div style={{ border: `1px solid ${p.border}`, borderRadius: 14, padding: 14, background: p.card, boxShadow: p.cardShadow }}>
                    <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>Ask</div>
                    <input
                      value={ragQuestion}
                      onChange={(e) => setRagQuestion(e.target.value)}
                      placeholder="Ask a question over ingested data…"
                      style={{ width: "100%", ...inSm(p), marginBottom: 8 }}
                    />
                    <div style={{ display: "flex", gap: 8, marginBottom: 8, alignItems: "center" }}>
                      <label style={{ color: p.subtext, fontSize: 12, fontWeight: 600 }}>Top K</label>
                      <select value={String(ragTopK)} onChange={(e) => setRagTopK(Number(e.target.value))} style={{ ...inSm(p) }}>
                        <option value="1">1</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="8">8</option>
                      </select>
                      <button type="button" disabled={ragBusy} onClick={askRag} style={{ marginLeft: "auto", ...btnPrimary(), opacity: ragBusy ? 0.55 : 1 }}>
                        {ragBusy ? "Working…" : "Ask"}
                      </button>
                    </div>
                    {ragResult && ragResult.mode === "ask" && (
                      <div style={{ border: `1px solid ${theme === "dark" ? "rgba(59, 130, 246, 0.35)" : "#bfdbfe"}`, background: theme === "dark" ? "rgba(30, 58, 138, 0.25)" : "#eff6ff", borderRadius: 10, padding: 10, fontSize: 12 }}>
                        <div style={{ fontWeight: 700, color: p.primary, marginBottom: 6 }}>Answer preview</div>
                        <div style={{ color: p.text2, marginBottom: 8 }}>{ragResult.payload.answer}</div>
                        <div style={{ color: p.muted, fontFamily: "monospace", marginBottom: 4 }}>Sources: {(ragResult.payload.sources || []).join(", ") || "none"}</div>
                        <div style={{ color: p.muted, fontFamily: "monospace", marginBottom: 8 }}>Session: {ragResult.payload.session_id || "unknown"} · Chunks: {ragResult.payload.total_chunks || 0}</div>
                        <div style={{ display: "grid", gap: 6, marginBottom: 8 }}>
                          {(ragResult.payload.results || []).slice(0, 5).map((r, idx) => (
                            <div key={idx} style={{ border: `1px solid ${p.border}`, borderRadius: 8, background: p.inputBg, padding: 8 }}>
                              <div style={{ color: p.primary, fontWeight: 600, marginBottom: 3 }}>
                                [{idx + 1}] {r.chunk?.source || "unknown"} · score {Number(r.score || 0).toFixed(3)} · {r.confidence || "low"}
                              </div>
                              <div style={{ color: p.text2 }}>{String(r.chunk?.content || "").slice(0, 240)}</div>
                            </div>
                          ))}
                        </div>
                        <details>
                          <summary style={{ cursor: "pointer", color: p.primary }}>Show context</summary>
                          <pre style={{ whiteSpace: "pre-wrap", marginTop: 6, color: p.text2, fontSize: 11 }}>{ragResult.payload.context || "(no context)"}</pre>
                        </details>
                      </div>
                    )}
                    {ragResult && ragResult.mode === "ingest" && (
                      <div style={{ border: `1px solid ${theme === "dark" ? "rgba(34, 197, 94, 0.35)" : "#bbf7d0"}`, background: theme === "dark" ? "rgba(6, 78, 59, 0.35)" : "#f0fdf4", borderRadius: 10, padding: 10, fontSize: 12, color: theme === "dark" ? "#bbf7d0" : "#166534" }}>
                        Ingested {ragResult.payload.chunks_added || 0} chunks from source "{ragResult.payload.source || "unknown"}" in session "{ragResult.payload.session_id || "unknown"}" (total chunks: {ragResult.payload.total_chunks || 0}).
                      </div>
                    )}
                    {ragResult && ragResult.mode === "reset" && (
                      <div style={{ border: `1px solid ${theme === "dark" ? "rgba(244, 63, 94, 0.4)" : "#fecaca"}`, background: theme === "dark" ? "rgba(88, 28, 28, 0.35)" : "#fff1f2", borderRadius: 10, padding: 10, fontSize: 12, color: theme === "dark" ? "#fecdd3" : "#9f1239" }}>
                        Session "{ragResult.payload.session_id || "unknown"}" reset successfully.
                      </div>
                    )}
                  </div>
                </div>
              </div>
              )}

              {hasTraces && activeView === "overview" && !initialLoading && (
              <div style={{ display: "grid", gap: 18, marginBottom: 20 }}>
                <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>Failure taxonomy and AI safety</div>
                <div style={cardS(p)}>
                  <SectionTitle
                    eyebrow="Priority"
                    title="Failure taxonomy"
                    subtitle="Reflects ordered categories from trace metrics; safety-related labels are sorted first, then by frequency"
                  />
                  <BarRows rows={failures} color={p.danger} emptyText="No failures" />
                </div>
                <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>Models and operations</div>
                <ModelComparison rows={modelRows} />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
                  <RecommendationPanel items={recommendations} />
                  <div style={cardS(p)}>
                    <SectionTitle eyebrow="Usage" title="Top tools" />
                    <BarRows rows={topTools} color={p.primary} emptyText="No tool data" />
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0,1fr))", gap: 16 }}>
                  <DistributionPills title="Step status" rows={statusDist} tone={p.primary} />
                  <DistributionPills title="Reflection verdicts" rows={verdictDist} tone={p.violet} />
                </div>
              </div>
              )}

              {hasTraces && activeView === "explorer" && !initialLoading && (
              <div className="archon-explore-grid" style={{ display: "grid", gridTemplateColumns: "minmax(min(100%, 320px), 400px) minmax(0, 1fr)", gap: 16, alignItems: "start" }}>
                <div style={cardS(p)}>
                  <SectionTitle eyebrow="Browse" title="Runs" subtitle="Newest in table window" />
                  <div style={{ display: "grid", gap: 6, maxHeight: "min(72vh, 780px)", overflow: "auto", paddingRight: 4, marginTop: 4 }}>
                    {sorted.map((t, i) => (
                      <button
                        type="button"
                        key={String(t.trace_id) + "-" + i}
                        onClick={() => setSelected(i)}
                        style={{
                          textAlign: "left",
                          borderRadius: 12,
                          cursor: "pointer",
                          color: p.text,
                          border: `1px solid ${i === selected ? "rgba(59, 130, 246, 0.5)" : p.border}`,
                          background: i === selected ? (theme === "dark" ? "linear-gradient(90deg, rgba(30, 58, 138, 0.4), #0f172a)" : "linear-gradient(90deg, #eff6ff, #fff)") : p.inputBg,
                          padding: "11px 12px",
                          boxShadow: i === selected ? "0 1px 2px rgba(37, 99, 235, 0.12)" : "none",
                        }}
                      >
                        <div style={{ fontWeight: 600, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", lineHeight: 1.35 }}>{t.task}</div>
                        <div style={{ color: p.muted, fontSize: 11, marginTop: 6, display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
                          <span style={{ fontWeight: 600, color: p.text2 }}>{t.model}</span>
                          <span>·</span>
                          <span>{t.total_steps} steps</span>
                          <span
                            style={{
                              marginLeft: "auto",
                              fontWeight: 800,
                              fontSize: 10,
                              textTransform: "uppercase",
                              letterSpacing: "0.04em",
                              color: t.success ? "#166534" : "#b91c1c",
                              background: t.success ? "rgba(22, 101, 52, 0.1)" : "rgba(185, 28, 28, 0.1)",
                              padding: "2px 6px",
                              borderRadius: 6,
                            }}
                          >{t.success ? "Pass" : "Fail"}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                <div style={cardS(p)}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10, marginBottom: 4, flexWrap: "wrap" }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: 10, fontWeight: 800, color: p.muted, letterSpacing: "0.1em", textTransform: "uppercase" }}>Selected run</div>
                      <div style={{ fontSize: 17, fontWeight: 700, marginTop: 6, lineHeight: 1.35, letterSpacing: "-0.02em", color: p.text }}>{current ? current.task : "Select a run"}</div>
                      {current && current.trace_id && (
                        <div style={{ fontSize: 11, color: p.muted, fontFamily: "ui-monospace,monospace", marginTop: 6 }}>{current.trace_id}</div>
                      )}
                    </div>
                    {current && (
                      <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                        <button type="button" onClick={copyTraceId} style={{ ...btnSecondary(p), fontSize: 12, padding: "7px 12px" }}>Copy ID</button>
                        <button type="button" onClick={exportCurrentTrace} style={{ ...btnPrimary(), fontSize: 12, padding: "7px 14px" }}>Export</button>
                      </div>
                    )}
                  </div>
                  {current && (
                    <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginBottom: 14 }}>
                      <MetaChip>{current.model}</MetaChip>
                      <MetaChip>{(current.total_steps || 0)} steps</MetaChip>
                      <MetaChip>{(current.wall_time != null ? Number(current.wall_time).toFixed(2) : "—")}s wall</MetaChip>
                      <MetaChip tone={current.success ? "good" : "bad"}>{current.success ? "Passed" : "Failed"}</MetaChip>
                    </div>
                  )}
                  <SectionTitle eyebrow="Execution" title="Step timeline" subtitle="Click a step for detail" />
                  <Timeline steps={current ? current.steps : []} onSelectStep={setSelectedStep} />
                  {selectedStep && (
                    <div style={{ marginTop: 14, border: `1px solid ${p.border}`, background: theme === "dark" ? "rgba(30, 58, 138, 0.2)" : "linear-gradient(180deg, #f8fbff, #eff6ff)", borderRadius: 12, padding: 14 }}>
                      <div style={{ fontSize: 10, fontWeight: 800, color: p.primary, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>Step detail</div>
                      <div style={{ fontSize: 13, color: p.text2, lineHeight: 1.55 }}>
                        <div><span style={{ color: p.muted, fontSize: 11, fontWeight: 600 }}>ID</span> {selectedStep.id}</div>
                        <div style={{ marginTop: 4 }}><span style={{ color: p.muted, fontSize: 11, fontWeight: 600 }}>Tool</span> {selectedStep.tool}</div>
                        <div style={{ marginTop: 4 }}><span style={{ color: p.muted, fontSize: 11, fontWeight: 600 }}>Status & verdict</span> {selectedStep.status} · {selectedStep.verdict} · {selectedStep.retries} retries</div>
                        {selectedStep.failure && <div style={{ marginTop: 6, color: p.danger, fontWeight: 600 }}>{selectedStep.failure}</div>}
                      </div>
                    </div>
                  )}
                </div>
              </div>
              )}

              <div style={{ marginTop: 28, paddingTop: 18, borderTop: "1px solid rgba(148, 163, 184, 0.25)", textAlign: "center", fontSize: 12, color: p.muted, lineHeight: 1.6 }}>
                <div>Archon operations · v{meta.version || "—"}</div>
                <div style={{ marginTop: 4 }}><span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>GET /api/health</span> · <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>GET /api/info</span> · <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>GET /api/dashboard</span></div>
              </div>
            </div>
          </div>
        </ThemeCtx.Provider>
        );
      }

      ReactDOM.createRoot(document.getElementById("root")).render(<App />);
    </script>
  </body>
</html>
"""


@dataclass
class TraceDashboardData:
    traces: list[dict[str, Any]]
    summary: dict[str, Any]
    meta: dict[str, Any]


def _package_version() -> str:
    try:
        from config.version import package_version

        return package_version()
    except Exception:
        try:
            root = Path(__file__).resolve().parents[1]
            data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
            return str((data.get("project") or {}).get("version", "0.0.0"))
        except Exception:
            return "unknown"


def _readiness_check(traces_dir: Path) -> dict[str, Any]:
    """Filesystem checks for routing traffic (readiness) vs process up (liveness)."""
    out: dict[str, Any] = {
        "traces_dir": str(traces_dir),
        "traces_dir_exists": False,
        "traces_dir_writable": False,
        "rag_store_writable": False,
    }
    try:
        resolved = traces_dir.resolve()
        out["traces_dir"] = str(resolved)
        out["traces_dir_exists"] = resolved.is_dir()
        if out["traces_dir_exists"]:
            out["traces_dir_writable"] = os.access(resolved, os.W_OK)
        rag = resolved / "rag_store"
        rag.mkdir(parents=True, exist_ok=True)
        out["rag_store_writable"] = os.access(rag, os.W_OK)
    except OSError as exc:
        out["error"] = str(exc)
    out["ready"] = bool(
        out.get("traces_dir_exists")
        and out.get("traces_dir_writable")
        and out.get("rag_store_writable")
    )
    return out


def _api_info_payload(traces_dir: Path) -> dict[str, Any]:
    """Small discovery document for clients and operators."""
    return {
        "application": "archon",
        "component": "archon-dashboard",
        "version": _package_version(),
        "traces_dir": str(traces_dir.resolve()),
        "endpoints": {
            "info": "GET /api/info",
            "health": "GET /api/health",
            "ready": "GET /api/ready",
            "dashboard": "GET /api/dashboard?limit=<1-2000>",
        },
    }


# Minimal tab icon (layers / stack) — no external assets.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#0f172a"/>'
    '<path fill="#60a5fa" d="M6 20h20v3H6zM8 15h16v3H8zM10 10h12v3H10z"/>'
    "</svg>"
)


def _normalize_trace(raw: dict[str, Any]) -> dict[str, Any]:
    plans = raw.get("plans") or []
    steps: list[dict[str, Any]] = []
    for plan in plans:
        for idx, step in enumerate(plan.get("steps") or []):
            tool_call = step.get("tool_call") or {}
            tool_result = step.get("tool_result") or {}
            reflection = step.get("reflection") or {}
            steps.append(
                {
                    "id": step.get("step_id", f"s{idx+1}"),
                    "desc": step.get("description", "unknown"),
                    "tool": tool_call.get("tool_name") or step.get("expected_tool") or "n/a",
                    "status": step.get("status", "unknown"),
                    "latency": round(float(tool_result.get("latency_ms", 0) or 0)),
                    "retries": int(step.get("retries", 0) or 0),
                    "verdict": reflection.get("verdict", "n/a"),
                    "failure": reflection.get("failure_category"),
                    "output": tool_result.get("output"),
                }
            )

    return {
        "trace_id": raw.get("trace_id", "unknown"),
        "task": raw.get("task_description", "unknown task"),
        "model": raw.get("model_name", "unknown-model"),
        "success": bool(raw.get("success", False)),
        "wall_time": float(raw.get("wall_time_seconds", 0) or 0),
        "total_steps": len(steps),
        "retries": int(raw.get("total_retries", sum(s["retries"] for s in steps)) or 0),
        "replans": int(raw.get("total_replans", 0) or 0),
        "steps": steps,
    }


def _summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate *observable* fields from normalized traces. No composite scores: rates and counts
    are direct ratios (e.g. step completion = completed steps / total steps in window)."""
    if not traces:
        return {
            "trace_count": 0,
            "success_rate": 0.0,
            "total_step_count": 0,
            "step_completion_rate": 0.0,
            "avg_wall_time": 0.0,
            "avg_steps": 0.0,
            "avg_retries": 0.0,
            "rag_step_share": 0.0,
            "rag_success_rate": 0.0,
            "rag_chunks_ingested": 0,
            "top_tools": [],
            "failure_taxonomy": [],
            "safety_failure_steps": 0,
            "safety_tagged_step_rate": 0.0,
            "model_comparison": [],
            "status_distribution": [],
            "verdict_distribution": [],
            "p50_step_latency_ms": 0.0,
            "p95_step_latency_ms": 0.0,
            "recommendations": [],
        }

    all_steps = [step for trace in traces for step in trace["steps"]]
    rag_steps = [s for s in all_steps if s["tool"] in {"rag_ingest", "rag_search", "rag_context"}]

    tool_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    verdict_counts: dict[str, int] = {}
    rag_chunks = 0

    for step in all_steps:
        tool_counts[step["tool"]] = tool_counts.get(step["tool"], 0) + 1
        status_counts[step["status"]] = status_counts.get(step["status"], 0) + 1
        verdict_counts[step["verdict"]] = verdict_counts.get(step["verdict"], 0) + 1
        if step["failure"]:
            failure_counts[step["failure"]] = failure_counts.get(step["failure"], 0) + 1
        if step["tool"] == "rag_ingest" and isinstance(step.get("output"), dict):
            rag_chunks += int(step["output"].get("chunks_added", 0) or 0)

    trace_count = len(traces)
    total_step_count = len(all_steps)
    success_rate = (sum(1 for t in traces if t["success"]) / trace_count) * 100
    completed_steps = sum(1 for s in all_steps if s.get("status") == "completed")
    step_completion_rate = (completed_steps / total_step_count) * 100.0 if total_step_count else 0.0
    avg_wall = sum(t["wall_time"] for t in traces) / trace_count
    avg_steps = len(all_steps) / trace_count
    avg_retries = sum(t["retries"] for t in traces) / trace_count

    rag_step_share = (len(rag_steps) / len(all_steps)) * 100 if all_steps else 0.0
    rag_success_rate = (
        sum(1 for s in rag_steps if s["status"] == "completed") / len(rag_steps) * 100
        if rag_steps
        else 0.0
    )
    latencies = sorted(float(s.get("latency", 0) or 0) for s in all_steps)
    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95_idx = int(max(0, round((len(latencies) - 1) * 0.95))) if latencies else 0
    p95 = latencies[p95_idx] if latencies else 0.0

    model_groups: dict[str, list[dict[str, Any]]] = {}
    for trace in traces:
        model_groups.setdefault(trace["model"], []).append(trace)

    model_rows = []
    for model, rows in model_groups.items():
        model_success = sum(1 for t in rows if t["success"]) / len(rows) * 100
        msteps = [s for t in rows for s in t["steps"]]
        m_total = len(msteps)
        m_done = sum(1 for s in msteps if s.get("status") == "completed")
        m_step_done_rate = (m_done / m_total) * 100.0 if m_total else 0.0
        model_rows.append(
            {
                "model": model,
                "success_rate": round(model_success, 1),
                "step_completion_rate": round(m_step_done_rate, 1),
            }
        )

    safety_fail_steps = sum(
        failure_counts.get(k, 0) for k in _SAFETY_FAILURE_KEYS
    )
    safety_tagged_step_rate = (
        (safety_fail_steps / total_step_count) * 100.0 if total_step_count else 0.0
    )

    def _fail_sort(t: tuple[str, int]) -> tuple[int, int, str]:
        label, n = t
        is_s = 1 if label in _SAFETY_FAILURE_KEYS else 0
        return (is_s, n, label)

    ordered_failures = sorted(failure_counts.items(), key=_fail_sort, reverse=True)

    def _one_rec(failure: str, count: int) -> str:
        if failure in _SAFETY_FAILURE_KEYS:
            return (
                f"[AI safety] Address '{failure}' ({count} steps): review policy, output filters, and grounding."
            )
        if failure == "tool_arg_schema_violation":
            return f"Reduce schema errors ({count}) by adding stricter tool-call examples in executor prompt."
        if failure == "tool_execution_failure":
            return f"Address execution failures ({count}) with fallback paths and stronger retry guards."
        if failure == "hallucinated_tool":
            return f"Reduce hallucinated tools ({count}) by emphasizing allowed tools in planner/executor prompts."
        if failure == "output_parse_error":
            return f"Fix parse issues ({count}) with tighter JSON-output constraints and validation cues."
        return f"Investigate recurring failure '{failure}' ({count}) to improve reliability."

    recommendations: list[str] = []
    top_failures = sorted(ordered_failures, key=lambda x: x[1], reverse=True)
    for failure, count in top_failures[:3]:
        recommendations.append(_one_rec(failure, count))
    if avg_retries > 0.8:
        recommendations.append("Retry volume is elevated; review correction quality and tighten stop/replan thresholds.")
    if rag_success_rate < 70 and rag_steps:
        recommendations.append("RAG success is low; evaluate chunking strategy and query formulation.")
    if safety_fail_steps and not any("AI safety" in r for r in recommendations[:3]):
        recommendations.insert(0, f"[AI safety] {safety_fail_steps} step(s) tagged with safety failure categories; audit prompts and allowlists first.")

    return {
        "trace_count": trace_count,
        "success_rate": success_rate,
        "total_step_count": total_step_count,
        "step_completion_rate": round(step_completion_rate, 1),
        "avg_wall_time": avg_wall,
        "avg_steps": avg_steps,
        "avg_retries": avg_retries,
        "rag_step_share": rag_step_share,
        "rag_success_rate": rag_success_rate,
        "rag_chunks_ingested": rag_chunks,
        "top_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:8],
        "failure_taxonomy": ordered_failures[:12],
        "safety_failure_steps": safety_fail_steps,
        "safety_tagged_step_rate": round(safety_tagged_step_rate, 2),
        "model_comparison": model_rows,
        "status_distribution": sorted(status_counts.items(), key=lambda x: x[1], reverse=True),
        "verdict_distribution": sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True),
        "p50_step_latency_ms": p50,
        "p95_step_latency_ms": p95,
        "recommendations": recommendations,
    }


def _load_dashboard_data(traces_dir: Path, list_limit: int) -> TraceDashboardData:
    """Load dashboard data: KPIs/aggregates over the newest *metrics* window; `traces` is a sub-list for the UI.

    - `?limit=` caps how many **newest** files appear in the `traces` array (and client table), max 2000.
    - `ARCHON_DASHBOARD_METRICS_MAX` caps how many **newest** files are read to compute `summary` (default 10_000, max 100_000).
    - When the disk has more files than the metrics cap, `meta.metrics_omit_older` is true: KPIs still reflect
      the full metrics window, not the shorter table list.
    """
    raw_metrics = os.getenv("ARCHON_DASHBOARD_METRICS_MAX", "10000")
    try:
        metrics_max = int(raw_metrics)
    except (TypeError, ValueError):
        metrics_max = 10_000
    metrics_max = max(1, min(metrics_max, 100_000))

    paths: list[Path] = []
    if traces_dir.exists():
        paths = sorted(traces_dir.glob("*.json"), reverse=True)
    total_on_disk = len(paths)

    paths_for_metrics = paths[:metrics_max]
    for_metrics: list[dict[str, Any]] = []
    for path in paths_for_metrics:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            for_metrics.append(_normalize_trace(raw))
        except Exception:
            continue

    summary = _summarize(for_metrics)
    list_cap = max(1, min(int(list_limit), 2000))
    traces = for_metrics[:list_cap]

    meta: dict[str, Any] = {
        "version": _package_version(),
        "traces_on_disk": total_on_disk,
        "traces_loaded": len(traces),
        "traces_in_response": len(traces),
        "limit": list_cap,
        "list_limit": list_cap,
        "metrics_files_read": len(for_metrics),
        "metrics_file_cap": metrics_max,
        "metrics_omit_older": total_on_disk > metrics_max,
        "server_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if meta["metrics_omit_older"]:
        meta["metrics_note"] = (
            f"KPIs use the newest {metrics_max} of {total_on_disk} trace files on disk. "
            "Set ARCHON_DASHBOARD_METRICS_MAX to include more files in aggregates."
        )
    return TraceDashboardData(traces=traces, summary=summary, meta=meta)


def serve_dashboard(host: str, port: int, traces_dir: Path) -> None:
    from tools.rag_pipeline import RAGPipeline

    rag_store_dir = traces_dir / "rag_store"
    rag_store_dir.mkdir(parents=True, exist_ok=True)
    rag_store_file = rag_store_dir / "ingests.jsonl"
    rag_pipelines: dict[str, RAGPipeline] = {}
    loaded_sessions: set[str] = set()
    rag_lock = Lock()
    rag_api_token = os.getenv("ARCHON_DASHBOARD_TOKEN", "").strip()
    max_request_bytes = int(os.getenv("ARCHON_RAG_MAX_REQUEST_BYTES", "200000"))
    max_ingest_chars = int(os.getenv("ARCHON_RAG_MAX_INGEST_CHARS", "50000"))
    rag_rate_max = int(os.getenv("ARCHON_RAG_RATE_MAX", "120"))
    rag_rate_window = float(os.getenv("ARCHON_RAG_RATE_WINDOW_SEC", "60"))
    _logging_level = os.getenv("ARCHON_LOG_LEVEL", "INFO").upper()
    if not logging.root.handlers:
        logging.basicConfig(
            level=getattr(logging, _logging_level, logging.INFO),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    _log.setLevel(getattr(logging, _logging_level, logging.INFO))
    rag_limiter = SlidingWindowRateLimiter(
        max_events=rag_rate_max, window_sec=rag_rate_window
    )
    rate_meta = {
        "enabled": True,
        "max_events_per_window": rag_rate_max,
        "window_sec": rag_rate_window,
    }
    audit_json_enabled = os.getenv("ARCHON_AUDIT_JSON", "1").lower() in (
        "1",
        "true",
        "yes",
    )
    if audit_json_enabled and not _audit.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        _audit.addHandler(h)
        _audit.setLevel(logging.INFO)
        _audit.propagate = False

    cors_origin = os.getenv("ARCHON_CORS_ORIGIN", "").strip()
    cors_max_age = int(os.getenv("ARCHON_CORS_MAX_AGE", "86400"))

    def _normalize_session_id(raw_session: str) -> str:
        session = (raw_session or "").strip()
        safe = "".join(ch for ch in session if ch.isalnum() or ch in {"-", "_"})
        return safe[:64] or "default"

    def _get_pipeline(session_id: str) -> RAGPipeline:
        pipeline = rag_pipelines.get(session_id)
        if pipeline is None:
            pipeline = RAGPipeline()
            rag_pipelines[session_id] = pipeline
        if session_id not in loaded_sessions and rag_store_file.exists():
            try:
                for line in rag_store_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("session_id") != session_id:
                        continue
                    if record.get("event") == "reset":
                        pipeline = RAGPipeline()
                        rag_pipelines[session_id] = pipeline
                        continue
                    text = str(record.get("text", "") or "")
                    if not text:
                        continue
                    source = str(record.get("source", "dashboard_input") or "dashboard_input")
                    pipeline.ingest(text=text, source=source)
            except Exception:
                # Keep dashboard resilient even if persistence file is malformed.
                pass
            loaded_sessions.add(session_id)
        return pipeline

    def _append_event_record(event: str, session_id: str, source: str = "", text: str = "") -> None:
        record = {
            "event": event,
            "session_id": session_id,
            "source": source,
            "text": text,
            "ts": int(time.time()),
        }
        append_jsonl_line(rag_store_file, record, use_flock=True)

    def _confidence_label(score: float) -> str:
        if score >= 0.65:
            return "high"
        if score >= 0.35:
            return "medium"
        return "low"

    def _session_stats(session_id: str) -> dict[str, Any]:
        pipeline = rag_pipelines.get(session_id)
        chunk_count = pipeline.document_count if pipeline else 0
        ingest_count = 0
        if rag_store_file.exists():
            try:
                for line in rag_store_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("session_id") == session_id and (
                        record.get("event", "ingest") == "ingest"
                    ):
                        ingest_count += 1
            except Exception:
                pass
        return {
            "session_id": session_id,
            "total_chunks": chunk_count,
            "ingest_count": ingest_count,
        }

    class Handler(BaseHTTPRequestHandler):
        def _client_ip(self) -> str:
            xff = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
            if xff:
                return xff[:100]
            return self.client_address[0] if self.client_address else "unknown"

        def _ensure_request_id(self) -> str:
            got = (self.headers.get("X-Request-Id") or "").strip()
            if got and len(got) < 200:
                return got
            return uuid.uuid4().hex

        def _common_security_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Permissions-Policy", "interest-cohort=()")

        def _cors_headers(self, *, is_preflight: bool = False) -> None:
            if not cors_origin:
                return
            self.send_header("Access-Control-Allow-Origin", cors_origin)
            if cors_origin != "*":
                self.send_header("Vary", "Origin")
            if is_preflight:
                self.send_header(
                    "Access-Control-Allow-Methods", "GET, POST, OPTIONS"
                )
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Authorization, Content-Type, X-Request-Id",
                )
                self.send_header("Access-Control-Max-Age", str(cors_max_age))

        def _check_rag_rate(self) -> bool:
            if rag_limiter.allow(f"ip:{self._client_ip()}"):
                return True
            _log.warning("rate_limited_rag path=%s ip=%s", self.path, self._client_ip())
            self._send_json(
                {
                    "error": "rate_limited",
                    "retry_after_sec": int(rag_rate_window),
                },
                code=429,
                request_id=self._request_id,
                retry_after_sec=int(rag_rate_window),
            )
            return False

        def _require_rag_bearer(self) -> bool:
            if not rag_api_token:
                return True
            auth_header = self.headers.get("Authorization", "")
            if auth_header != f"Bearer {rag_api_token}":
                self._send_json({"error": "unauthorized"}, code=401)
                return False
            return True

        def _send_json(
            self,
            payload: dict[str, Any],
            code: int = 200,
            request_id: str | None = None,
            *,
            retry_after_sec: int | None = None,
        ) -> None:
            rid = request_id or getattr(self, "_request_id", None) or self._ensure_request_id()
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-Id", rid)
            if code == 429 and retry_after_sec is not None:
                self.send_header("Retry-After", str(retry_after_sec))
            self._common_security_headers()
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str, code: int = 200) -> None:
            rid = getattr(self, "_request_id", None) or self._ensure_request_id()
            body = html.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-Id", rid)
            self._common_security_headers()
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_svg(self, svg: str, code: int = 200) -> None:
            rid = getattr(self, "_request_id", None) or self._ensure_request_id()
            body = svg.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("X-Request-Id", rid)
            self._common_security_headers()
            self._cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def send_response(self, code: int, message: str | None = None) -> None:  # noqa: N802
            self._response_begun = True
            self._response_status = int(code)
            super().send_response(code, message)  # type: ignore[call-arg]

        def _emit_audit(self, method: str, path: str, t0: float) -> None:
            if not audit_json_enabled:
                return
            rec: dict[str, Any] = {
                "component": "archon.dashboard",
                "method": method,
                "path": path[:512],
                "status": getattr(self, "_response_status", None),
                "request_id": getattr(self, "_request_id", None),
                "client": self._client_ip(),
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
            _audit.info(json.dumps(rec, separators=(",", ":")))

        def do_GET(self) -> None:  # noqa: N802
            self._request_id = self._ensure_request_id()
            self._response_status = None
            self._response_begun = False
            t0 = time.perf_counter()
            try:
                try:
                    parsed = urlparse(self.path)
                    if parsed.path == "/api/ready":
                        rc = _readiness_check(traces_dir)
                        code = 200 if rc.get("ready") else 503
                        self._send_json(
                            {
                                "status": "ready" if rc.get("ready") else "not_ready",
                                "checks": rc,
                            },
                            code=code,
                        )
                        return
                    if parsed.path == "/api/health":
                        self._send_json(
                            {
                                "status": "ok",
                                "service": "archon-dashboard",
                                "version": _package_version(),
                                "traces_dir": str(traces_dir),
                                "rag_bearer_required": bool(rag_api_token),
                                "rate_limiting": rate_meta,
                                "checks": _readiness_check(traces_dir),
                            }
                        )
                        return
                    if parsed.path == "/api/info":
                        self._send_json(_api_info_payload(traces_dir))
                        return
                    if parsed.path == "/favicon.svg":
                        self._send_svg(FAVICON_SVG)
                        return
                    if parsed.path.startswith("/api/rag/") and not self._check_rag_rate():
                        return
                    if parsed.path == "/api/rag/auth-check":
                        if not rag_api_token:
                            self._send_json(
                                {
                                    "auth_configured": False,
                                    "ok": True,
                                    "message": "Server has no ARCHON_DASHBOARD_TOKEN; RAG endpoints are not protected by bearer auth.",
                                }
                            )
                            return
                        auth = self.headers.get("Authorization", "")
                        valid = auth == f"Bearer {rag_api_token}"
                        self._send_json(
                            {
                                "auth_configured": True,
                                "ok": valid,
                                "message": "Bearer token is valid."
                                if valid
                                else "Missing or invalid bearer token. Use the value of ARCHON_DASHBOARD_TOKEN.",
                            }
                        )
                        return
                    if parsed.path.startswith("/api/rag/"):
                        if not self._require_rag_bearer():
                            return
                    if parsed.path == "/api/dashboard":
                        qd = parse_qs(parsed.query or "")
                        try:
                            lim = int((qd.get("limit") or ["200"])[0])
                        except (TypeError, ValueError):
                            lim = 200
                        data = _load_dashboard_data(traces_dir, lim)
                        self._send_json(
                            {
                                "traces": data.traces,
                                "summary": data.summary,
                                "meta": data.meta,
                            }
                        )
                        return
                    if parsed.path == "/api/rag/session":
                        session_param = parse_qs(parsed.query or "").get("session_id", ["default"])[0]
                        session_id = _normalize_session_id(session_param)
                        with rag_lock:
                            _get_pipeline(session_id)
                            stats = _session_stats(session_id)
                        self._send_json(stats)
                        return
                    if parsed.path in {"/", "/dashboard"}:
                        self._send_html(HTML_PAGE)
                        return
                    _log.info("not_found method=GET path=%s request_id=%s", self.path, self._request_id)
                    self._send_json({"error": "not found"}, code=404, request_id=self._request_id)
                except Exception:
                    _log.exception("do_GET failed")
                    if not getattr(self, "_response_begun", False):
                        self._send_json(
                            {
                                "error": "internal_error",
                                "request_id": self._request_id,
                            },
                            code=500,
                            request_id=self._request_id,
                        )
            finally:
                self._emit_audit("GET", self.path, t0)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._request_id = self._ensure_request_id()
            self._response_status = None
            self._response_begun = False
            t0 = time.perf_counter()
            try:
                if not cors_origin:
                    self.send_error(404)
                    return
                parsed = urlparse(self.path)
                if not parsed.path.startswith("/api/"):
                    self.send_error(404)
                    return
                self.send_response(204)
                self._common_security_headers()
                self._cors_headers(is_preflight=True)
                self.end_headers()
            except Exception:
                _log.exception("do_OPTIONS failed")
            finally:
                self._emit_audit("OPTIONS", self.path, t0)

        def do_POST(self) -> None:  # noqa: N802
            self._request_id = self._ensure_request_id()
            self._response_status = None
            self._response_begun = False
            t0 = time.perf_counter()
            try:
                try:
                    parsed = urlparse(self.path)
                    if parsed.path.startswith("/api/rag/") and not self._check_rag_rate():
                        return
                    length = int(self.headers.get("Content-Length", "0") or 0)
                    if length > max_request_bytes:
                        self._send_json({"error": "payload_too_large"}, code=413)
                        return
                    if rag_api_token and parsed.path.startswith("/api/rag/"):
                        auth_header = self.headers.get("Authorization", "")
                        expected = f"Bearer {rag_api_token}"
                        if auth_header != expected:
                            self._send_json({"error": "unauthorized"}, code=401)
                            return
                    raw_body = self.rfile.read(length) if length > 0 else b"{}"
                    try:
                        body = json.loads(raw_body.decode("utf-8"))
                    except Exception:
                        self._send_json({"error": "invalid_json"}, code=400)
                        return

                    if parsed.path == "/api/rag/ingest":
                        text = str(body.get("text", "") or "").strip()
                        source = str(body.get("source", "dashboard_input") or "dashboard_input")
                        session_id = _normalize_session_id(
                            str(body.get("session_id", "default") or "default")
                        )
                        if not text:
                            self._send_json({"error": "text_required"}, code=400)
                            return
                        if len(text) > max_ingest_chars:
                            self._send_json({"error": "text_too_large"}, code=413)
                            return
                        with rag_lock:
                            pipeline = _get_pipeline(session_id)
                            chunks_added = pipeline.ingest(text=text, source=source)
                            total = pipeline.document_count
                            _append_event_record(
                                event="ingest", session_id=session_id, source=source, text=text
                            )
                            stats = _session_stats(session_id)
                        self._send_json(
                            {
                                "status": "ingested",
                                "chunks_added": chunks_added,
                                "total_chunks": total,
                                "source": source,
                                "session_id": session_id,
                                "session_stats": stats,
                            }
                        )
                        return

                    if parsed.path == "/api/rag/ask":
                        question = str(body.get("question", "") or "").strip()
                        top_k = int(body.get("top_k", 3) or 3)
                        session_id = _normalize_session_id(
                            str(body.get("session_id", "default") or "default")
                        )
                        if not question:
                            self._send_json({"error": "question_required"}, code=400)
                            return
                        bounded_top_k = max(1, min(top_k, 10))
                        with rag_lock:
                            pipeline = _get_pipeline(session_id)
                            results = pipeline.retrieve(question, top_k=bounded_top_k)
                            context = pipeline.retrieve_as_context(question, top_k=bounded_top_k)
                            total = pipeline.document_count
                            stats = _session_stats(session_id)
                        if not results:
                            self._send_json(
                                {
                                    "answer": "No relevant context found in the RAG index. Ingest data first, then ask again.",
                                    "sources": [],
                                    "context": context,
                                    "results": [],
                                    "session_id": session_id,
                                    "total_chunks": total,
                                    "session_stats": stats,
                                }
                            )
                            return

                        sources = []
                        for r in results:
                            src = r.chunk.source or "unknown"
                            if src not in sources:
                                sources.append(src)
                        preview_snippets = [r.chunk.content[:180].strip() for r in results[:2]]
                        answer = " ".join(preview_snippets)
                        self._send_json(
                            {
                                "answer": answer,
                                "sources": sources,
                                "context": context,
                                "results": [
                                    {
                                        **r.to_dict(),
                                        "confidence": _confidence_label(float(r.score)),
                                    }
                                    for r in results
                                ],
                                "session_id": session_id,
                                "total_chunks": total,
                                "session_stats": stats,
                            }
                        )
                        return

                    if parsed.path == "/api/rag/reset":
                        session_id = _normalize_session_id(
                            str(body.get("session_id", "default") or "default")
                        )
                        with rag_lock:
                            rag_pipelines[session_id] = RAGPipeline()
                            loaded_sessions.add(session_id)
                            _append_event_record(event="reset", session_id=session_id)
                            stats = _session_stats(session_id)
                        self._send_json(
                            {
                                "status": "reset",
                                "session_id": session_id,
                                "session_stats": stats,
                            }
                        )
                        return

                    self._send_json({"error": "not_found"}, code=404, request_id=self._request_id)
                except Exception:
                    _log.exception("do_POST failed")
                    if not getattr(self, "_response_begun", False):
                        self._send_json(
                            {
                                "error": "internal_error",
                                "request_id": self._request_id,
                            },
                            code=500,
                            request_id=self._request_id,
                        )
            finally:
                self._emit_audit("POST", self.path, t0)

        def log_message(self, format: str, *args: Any) -> None:
            _log.debug("%s - %s", self.address_string(), format % args)

    server = ThreadingHTTPServer((host, port), Handler)

    def _handle_signal(_signum: int, _frame: Any) -> None:
        _log.info("Received signal %s; shutting down dashboard.", _signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _handle_signal)
        except (OSError, ValueError):
            pass
    try:
        signal.signal(signal.SIGINT, _handle_signal)
    except (OSError, ValueError):
        pass

    print(f"Archon dashboard running at http://{host}:{port}")
    print(f"Reading traces from: {traces_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("Keyboard interrupt, shutting down dashboard.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Archon dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--traces-dir", default="traces")
    args = parser.parse_args()
    traces_dir = Path(args.traces_dir)
    if not traces_dir.is_absolute():
        traces_dir = Path(os.getcwd()) / traces_dir
    serve_dashboard(args.host, args.port, traces_dir)


if __name__ == "__main__":
    main()
