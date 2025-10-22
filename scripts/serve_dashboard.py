"""Lightweight dashboard server for Archon trace visualization."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
import tomllib
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qs, urlparse

from scripts.dashboard_reliability import SlidingWindowRateLimiter, append_jsonl_line

_log = logging.getLogger("archon.dashboard")
_audit = logging.getLogger("archon.audit")

HTML_PAGE = """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Archon Dashboard</title>
    <script crossorigin src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  </head>
  <body style="margin:0;background:#f3f6fb;">
    <div id="root"></div>
    <script type="text/babel">
      const { useEffect, useState } = React;

      const PALETTE = {
        bg: "#f3f6fb",
        card: "#ffffff",
        text: "#0f172a",
        subtext: "#64748b",
        border: "#e2e8f0",
        primary: "#2563eb",
        success: "#16a34a",
        warning: "#d97706",
        danger: "#dc2626",
        info: "#0d9488",
        violet: "#7c3aed",
      };

      const shell = {
        minHeight: "100vh",
        background: "linear-gradient(180deg,#f9fbff 0%,#f3f6fb 100%)",
        color: PALETTE.text,
        fontFamily: "Inter, system-ui, -apple-system, Segoe UI, sans-serif",
      };
      const card = {
        background: PALETTE.card,
        borderRadius: 16,
        border: `1px solid ${PALETTE.border}`,
        boxShadow: "0 8px 24px rgba(15,23,42,0.06)",
        padding: 16,
      };

      function TrustCard({ score, successRate, retries, p95 }) {
        const trustLabel = score >= 85 ? "High confidence" : score >= 70 ? "Moderate confidence" : "Needs attention";
        const tone = score >= 85 ? PALETTE.success : score >= 70 ? PALETTE.warning : PALETTE.danger;
        return (
          <div style={{ ...card, display: "grid", gap: 12 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <div>
                <div style={{ color: PALETTE.subtext, fontSize: 12 }}>System Trust Index</div>
                <div style={{ fontSize: 30, fontWeight: 800 }}>{score.toFixed(0)}/100</div>
              </div>
              <div style={{ padding: "6px 10px", borderRadius: 999, fontSize: 12, fontWeight: 600, color: tone, background: `${tone}18` }}>{trustLabel}</div>
            </div>
            <div style={{ height: 10, borderRadius: 999, background: "#e2e8f0" }}>
              <div style={{ height: 10, borderRadius: 999, width: `${score}%`, background: `linear-gradient(90deg, ${PALETTE.primary}, ${tone})` }} />
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 8 }}>
              <div><div style={{ color: PALETTE.subtext, fontSize: 11 }}>Success</div><div style={{ fontWeight: 700 }}>{successRate.toFixed(1)}%</div></div>
              <div><div style={{ color: PALETTE.subtext, fontSize: 11 }}>Avg Retries</div><div style={{ fontWeight: 700 }}>{retries.toFixed(2)}</div></div>
              <div><div style={{ color: PALETTE.subtext, fontSize: 11 }}>P95 Step</div><div style={{ fontWeight: 700 }}>{p95.toFixed(0)}ms</div></div>
            </div>
          </div>
        );
      }

      function StatCard({ title, value, sub, accent }) {
        return (
          <div style={{ ...card, borderTop: `3px solid ${accent}` }}>
            <div style={{ color: PALETTE.subtext, fontSize: 12, fontWeight: 500 }}>{title}</div>
            <div style={{ fontSize: 28, fontWeight: 700, marginTop: 6 }}>{value}</div>
            <div style={{ color: PALETTE.subtext, fontSize: 12, marginTop: 4 }}>{sub}</div>
          </div>
        );
      }

      function SectionTitle({ title, subtitle }) {
        return (
          <div style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 18, fontWeight: 750 }}>{title}</div>
            {subtitle && <div style={{ color: PALETTE.subtext, fontSize: 12, marginTop: 2 }}>{subtitle}</div>}
          </div>
        );
      }

      function BarRows({ rows, color, valueSuffix = "", emptyText = "No data" }) {
        const max = Math.max(1, ...rows.map((r) => r.value));
        return (
          <div style={{ display: "grid", gap: 10 }}>
            {rows.length === 0 && <div style={{ color: PALETTE.subtext, fontSize: 12 }}>{emptyText}</div>}
            {rows.map((r) => (
              <div key={r.label}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 4 }}>
                  <span style={{ color: "#334155", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", maxWidth: 240 }}>{r.label}</span>
                  <span style={{ color: PALETTE.text, fontWeight: 600 }}>{r.value}{valueSuffix}</span>
                </div>
                <div style={{ height: 8, borderRadius: 999, background: "#e2e8f0" }}>
                  <div style={{ height: 8, borderRadius: 999, width: `${(r.value / max) * 100}%`, background: color }} />
                </div>
              </div>
            ))}
          </div>
        );
      }

      function ModelComparison({ rows }) {
        return (
          <div style={card}>
            <SectionTitle
              title="Model Comparison"
              subtitle="Outcome quality by model from observed traces"
            />
            <div style={{ display: "grid", gap: 12 }}>
              {rows.length === 0 && <div style={{ color: PALETTE.subtext, fontSize: 12 }}>No model comparison data yet</div>}
              {rows.map((m) => (
                <div key={m.model}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 5 }}>
                    <span style={{ fontWeight: 600 }}>{m.model}</span>
                    <span style={{ color: "#334155" }}>Success {m.success_rate.toFixed(0)}% · Reliability {m.reliability.toFixed(0)}%</span>
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                    <div style={{ height: 8, borderRadius: 999, background: "#e2e8f0", position: "relative" }}>
                      <div style={{ width: `${m.success_rate}%`, background: "#2563eb", height: 8, borderRadius: 999 }} />
                    </div>
                    <div style={{ height: 8, borderRadius: 999, background: "#e2e8f0", position: "relative" }}>
                      <div style={{ width: `${m.reliability}%`, background: "#14b8a6", height: 8, borderRadius: 999 }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      }

      function RecommendationPanel({ items }) {
        return (
          <div style={card}>
            <SectionTitle title="Recommended Actions" subtitle="What to do next to improve reliability" />
            <div style={{ display: "grid", gap: 8 }}>
              {items.length === 0 && <div style={{ color: PALETTE.subtext, fontSize: 12 }}>No urgent actions. System looks healthy.</div>}
              {items.map((text, i) => (
                <div key={i} style={{ border: "1px solid #e2e8f0", background: "#f8fafc", borderRadius: 12, padding: 10, fontSize: 13, color: "#334155" }}>
                  <span style={{ fontWeight: 700, color: PALETTE.primary, marginRight: 8 }}>#{i + 1}</span>{text}
                </div>
              ))}
            </div>
          </div>
        );
      }

      function DistributionPills({ title, rows, tone }) {
        return (
          <div style={card}>
            <SectionTitle title={title} />
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {rows.length === 0 && <span style={{ color: PALETTE.subtext, fontSize: 12 }}>No data</span>}
              {rows.map((r) => (
                <span key={r.label} style={{ fontSize: 12, padding: "7px 10px", borderRadius: 999, background: `${tone}18`, color: tone, border: `1px solid ${tone}33` }}>
                  {r.label}: <strong>{r.value}</strong>
                </span>
              ))}
            </div>
          </div>
        );
      }

      function Timeline({ steps, onSelectStep }) {
        const maxLatency = Math.max(1, ...(steps || []).map((s) => s.latency || 0));
        return (
          <div style={{ display: "grid", gap: 10 }}>
            {(steps || []).map((s, i) => (
              <button
                key={s.id || i}
                onClick={() => onSelectStep && onSelectStep(s)}
                style={{ padding: 12, border: "1px solid #e2e8f0", borderRadius: 12, background: "#fff", width: "100%", textAlign: "left", cursor: "pointer" }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 10 }}>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{i + 1}. {s.desc}</div>
                    <div style={{ color: "#64748b", fontSize: 12 }}>{s.tool} · {s.status} · {s.verdict}{s.failure ? ` · ${s.failure}` : ""}</div>
                  </div>
                  <div style={{ fontSize: 12, color: "#334155", fontWeight: 600 }}>{s.latency}ms</div>
                </div>
                <div style={{ height: 6, borderRadius: 999, background: "#e2e8f0", marginTop: 8 }}>
                  <div style={{ height: 6, borderRadius: 999, width: `${(s.latency / maxLatency) * 100}%`, background: "#7c3aed" }} />
                </div>
              </button>
            ))}
          </div>
        );
      }

      function App() {
        const [data, setData] = useState({ traces: [], summary: {} });
        const [selected, setSelected] = useState(0);
        const [error, setError] = useState("");
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

        const load = async () => {
          try {
            const res = await fetch("/api/dashboard");
            if (!res.ok) throw new Error("Failed to fetch dashboard data");
            const payload = await res.json();
            setData(payload);
            setError("");
          } catch (e) {
            setError(String(e));
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

        useEffect(() => { load(); }, []);
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
        const sorted = [...filtered].sort((a, b) => {
          if (sortBy === "success") return Number(b.success) - Number(a.success);
          if (sortBy === "steps") return (b.total_steps || 0) - (a.total_steps || 0);
          if (sortBy === "latency") return (b.wall_time || 0) - (a.wall_time || 0);
          if (sortBy === "retries") return (b.retries || 0) - (a.retries || 0);
          return 0;
        });
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
        const trustIndex = summary.trust_index || 0;

        return (
          <div style={shell}>
            <div style={{ maxWidth: 1400, margin: "0 auto", padding: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 28, fontWeight: 800, letterSpacing: "-0.02em" }}>Archon Intelligence Dashboard</div>
                  <div style={{ color: "#64748b", fontSize: 13 }}>Operational reliability, RAG behavior, and model-level performance</div>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <label style={{ fontSize: 12, color: "#475569" }}><input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} /> Auto</label>
                  <select value={String(refreshMs)} onChange={(e) => setRefreshMs(Number(e.target.value))} style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "7px 10px", background: "#fff" }}>
                    <option value="2000">2s</option><option value="5000">5s</option><option value="10000">10s</option>
                  </select>
                  <button onClick={load} style={{ border: "1px solid #2563eb", color: "#fff", background: "#2563eb", borderRadius: 10, padding: "8px 12px", cursor: "pointer" }}>Refresh</button>
                </div>
              </div>
              {error && <div style={{ ...card, borderColor: "#fecaca", color: "#b91c1c", marginBottom: 12 }}>{error}</div>}

              <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                <select value={modelFilter} onChange={(e) => setModelFilter(e.target.value)} style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff" }}>
                  <option value="all">All models</option>
                  {models.map((m) => <option key={m} value={m}>{m}</option>)}
                </select>
                <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)} style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff" }}>
                  <option value="all">All status</option><option value="pass">Pass only</option><option value="fail">Fail only</option>
                </select>
                <label style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff", fontSize: 13 }}>
                  <input type="checkbox" checked={ragOnly} onChange={(e) => setRagOnly(e.target.checked)} /> RAG only
                </label>
                <label style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff", fontSize: 13 }}>
                  <input type="checkbox" checked={incidentOnly} onChange={(e) => setIncidentOnly(e.target.checked)} /> Incident mode
                </label>
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search task or tool..."
                  style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff", minWidth: 220 }}
                />
                <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} style={{ border: "1px solid #cbd5e1", borderRadius: 10, padding: "8px 10px", background: "#fff" }}>
                  <option value="recent">Sort: Default</option>
                  <option value="latency">Sort: Wall Time</option>
                  <option value="steps">Sort: Steps</option>
                  <option value="retries">Sort: Retries</option>
                  <option value="success">Sort: Success</option>
                </select>
                <div style={{ marginLeft: "auto", alignSelf: "center", color: "#64748b", fontSize: 12 }}>Showing {sorted.length}/{traces.length} traces</div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr 1fr", gap: 10, marginBottom: 12 }}>
                <TrustCard
                  score={trustIndex}
                  successRate={summary.success_rate || 0}
                  retries={summary.avg_retries || 0}
                  p95={summary.p95_step_latency_ms || 0}
                />
                <StatCard title="Success Rate" value={`${(summary.success_rate || 0).toFixed(1)}%`} sub={`${summary.trace_count || 0} total traces`} accent="#16a34a" />
                <StatCard title="Avg Wall Time" value={`${(summary.avg_wall_time || 0).toFixed(2)}s`} sub="per trace" accent="#2563eb" />
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,minmax(0,1fr))", gap: 10, marginBottom: 12 }}>
                <StatCard title="Avg Steps" value={(summary.avg_steps || 0).toFixed(2)} sub="execution depth" accent={PALETTE.violet} />
                <StatCard title="Avg Retries" value={(summary.avg_retries || 0).toFixed(2)} sub="stability signal" accent={PALETTE.warning} />
                <StatCard title="RAG Step Share" value={`${(summary.rag_step_share || 0).toFixed(1)}%`} sub="RAG usage footprint" accent={PALETTE.info} />
                <StatCard title="RAG Success" value={`${(summary.rag_success_rate || 0).toFixed(1)}%`} sub={`chunks ingested: ${summary.rag_chunks_ingested || 0}`} accent="#0f766e" />
              </div>

              <div style={{ ...card, marginBottom: 12 }}>
                <SectionTitle title="RAG Studio" subtitle="Ingest context and ask retrieval-backed questions directly from the dashboard" />
                <div style={{ border: "1px solid #e2e8f0", borderRadius: 10, background: "#f8fafc", padding: 10, marginBottom: 10 }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 6 }}>
                    <div style={{ fontSize: 13, fontWeight: 600 }}>RAG API Access</div>
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
                          <span style={{ fontSize: 11, color: "#64748b" }}>Server has no token lock</span>
                        )}
                        <button type="button" onClick={probeRagAuth} style={{ fontSize: 11, border: "1px solid #cbd5e1", background: "#fff", borderRadius: 6, padding: "3px 8px", cursor: "pointer" }}>Recheck</button>
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <input
                      type={ragTokenVisible ? "text" : "password"}
                      value={ragToken}
                      onChange={(e) => setRagToken(e.target.value)}
                      placeholder="Bearer token (set ARCHON_DASHBOARD_TOKEN server-side)"
                      style={{ flex: 1, border: "1px solid #cbd5e1", borderRadius: 8, padding: "8px 10px", background: "#fff" }}
                    />
                    <button onClick={() => setRagTokenVisible((v) => !v)} style={{ border: "1px solid #cbd5e1", background: "#fff", borderRadius: 8, padding: "8px 10px", cursor: "pointer" }}>
                      {ragTokenVisible ? "Hide" : "Show"}
                    </button>
                    <button onClick={() => setRagToken("")} style={{ border: "1px solid #cbd5e1", background: "#fff", borderRadius: 8, padding: "8px 10px", cursor: "pointer" }}>
                      Clear
                    </button>
                  </div>
                  <div style={{ color: "#64748b", fontSize: 11, marginTop: 6 }}>
                    Token is stored in browser localStorage and attached to RAG API requests.
                    {ragAuthProbe && ragAuthProbe.message && (
                      <span style={{ display: "block", marginTop: 4, color: ragAuthProbe.ok ? "#334155" : "#b91c1c" }}>{ragAuthProbe.message}</span>
                    )}
                  </div>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#fff" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Ingest Document</div>
                    <input
                      value={ragSessionId}
                      onChange={(e) => setRagSessionId(e.target.value)}
                      placeholder="session id (e.g. prod_ops_team)"
                      style={{ width: "100%", border: "1px solid #cbd5e1", borderRadius: 8, padding: "8px 10px", marginBottom: 8 }}
                    />
                    <input
                      value={ragSource}
                      onChange={(e) => setRagSource(e.target.value)}
                      placeholder="source id (e.g. report_q2.txt)"
                      style={{ width: "100%", border: "1px solid #cbd5e1", borderRadius: 8, padding: "8px 10px", marginBottom: 8 }}
                    />
                    <textarea
                      value={ragText}
                      onChange={(e) => setRagText(e.target.value)}
                      placeholder="Paste text to ingest into RAG memory..."
                      rows={5}
                      style={{ width: "100%", border: "1px solid #cbd5e1", borderRadius: 8, padding: "8px 10px", resize: "vertical" }}
                    />
                    <div style={{ marginTop: 8 }}>
                      <button disabled={ragBusy} onClick={ingestRag} style={{ border: "1px solid #0f766e", color: "#fff", background: "#0f766e", borderRadius: 8, padding: "8px 12px", cursor: "pointer", opacity: ragBusy ? 0.6 : 1, marginRight: 8 }}>
                        {ragBusy ? "Working..." : "Ingest"}
                      </button>
                      <button disabled={ragBusy} onClick={resetSession} style={{ border: "1px solid #dc2626", color: "#dc2626", background: "#fff", borderRadius: 8, padding: "8px 12px", cursor: "pointer", opacity: ragBusy ? 0.6 : 1 }}>
                        Reset Session
                      </button>
                    </div>
                    {ragSessionStats && (
                      <div style={{ marginTop: 10, border: "1px solid #e2e8f0", borderRadius: 8, background: "#f8fafc", padding: 8, fontSize: 12, color: "#334155" }}>
                        Session stats: {ragSessionStats.session_id} · chunks {ragSessionStats.total_chunks || 0} · ingests {ragSessionStats.ingest_count || 0}
                      </div>
                    )}
                  </div>

                  <div style={{ border: "1px solid #e2e8f0", borderRadius: 12, padding: 10, background: "#fff" }}>
                    <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>Ask Question</div>
                    <input
                      value={ragQuestion}
                      onChange={(e) => setRagQuestion(e.target.value)}
                      placeholder="Ask a question over ingested data..."
                      style={{ width: "100%", border: "1px solid #cbd5e1", borderRadius: 8, padding: "8px 10px", marginBottom: 8 }}
                    />
                    <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                      <label style={{ color: "#64748b", fontSize: 12, alignSelf: "center" }}>Top K</label>
                      <select value={String(ragTopK)} onChange={(e) => setRagTopK(Number(e.target.value))} style={{ border: "1px solid #cbd5e1", borderRadius: 8, padding: "7px 10px", background: "#fff" }}>
                        <option value="1">1</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="8">8</option>
                      </select>
                      <button disabled={ragBusy} onClick={askRag} style={{ marginLeft: "auto", border: "1px solid #2563eb", color: "#fff", background: "#2563eb", borderRadius: 8, padding: "8px 12px", cursor: "pointer", opacity: ragBusy ? 0.6 : 1 }}>
                        {ragBusy ? "Working..." : "Ask"}
                      </button>
                    </div>
                    {ragResult && ragResult.mode === "ask" && (
                      <div style={{ border: "1px solid #dbeafe", background: "#eff6ff", borderRadius: 10, padding: 10, fontSize: 12 }}>
                        <div style={{ fontWeight: 700, color: "#1d4ed8", marginBottom: 6 }}>Answer Preview</div>
                        <div style={{ color: "#334155", marginBottom: 8 }}>{ragResult.payload.answer}</div>
                        <div style={{ color: "#64748b", fontFamily: "monospace", marginBottom: 4 }}>Sources: {(ragResult.payload.sources || []).join(", ") || "none"}</div>
                        <div style={{ color: "#64748b", fontFamily: "monospace", marginBottom: 8 }}>Session: {ragResult.payload.session_id || "unknown"} · Chunks: {ragResult.payload.total_chunks || 0}</div>
                        <div style={{ display: "grid", gap: 6, marginBottom: 8 }}>
                          {(ragResult.payload.results || []).slice(0, 5).map((r, idx) => (
                            <div key={idx} style={{ border: "1px solid #bfdbfe", borderRadius: 8, background: "#fff", padding: 8 }}>
                              <div style={{ color: "#1e3a8a", fontWeight: 600, marginBottom: 3 }}>
                                [{idx + 1}] {r.chunk?.source || "unknown"} · score {Number(r.score || 0).toFixed(3)} · {r.confidence || "low"}
                              </div>
                              <div style={{ color: "#334155" }}>{String(r.chunk?.content || "").slice(0, 240)}</div>
                            </div>
                          ))}
                        </div>
                        <details>
                          <summary style={{ cursor: "pointer", color: "#2563eb" }}>Show context</summary>
                          <pre style={{ whiteSpace: "pre-wrap", marginTop: 6, color: "#334155", fontSize: 11 }}>{ragResult.payload.context || "(no context)"}</pre>
                        </details>
                      </div>
                    )}
                    {ragResult && ragResult.mode === "ingest" && (
                      <div style={{ border: "1px solid #bbf7d0", background: "#f0fdf4", borderRadius: 10, padding: 10, fontSize: 12, color: "#166534" }}>
                        Ingested {ragResult.payload.chunks_added || 0} chunks from source "{ragResult.payload.source || "unknown"}" in session "{ragResult.payload.session_id || "unknown"}" (total chunks: {ragResult.payload.total_chunks || 0}).
                      </div>
                    )}
                    {ragResult && ragResult.mode === "reset" && (
                      <div style={{ border: "1px solid #fee2e2", background: "#fff1f2", borderRadius: 10, padding: 10, fontSize: 12, color: "#9f1239" }}>
                        Session "{ragResult.payload.session_id || "unknown"}" reset successfully.
                      </div>
                    )}
                  </div>
                </div>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "340px 1fr 360px", gap: 12 }}>
                <div style={card}>
                  <SectionTitle title="Trace Explorer" subtitle="What happened in each run?" />
                  <div style={{ display: "grid", gap: 8, maxHeight: 700, overflow: "auto", paddingRight: 4 }}>
                    {sorted.map((t, i) => (
                      <button key={t.trace_id} onClick={() => setSelected(i)} style={{
                        textAlign: "left", borderRadius: 12, cursor: "pointer",
                        border: `1px solid ${i === selected ? "#93c5fd" : "#e2e8f0"}`,
                        background: i === selected ? "#eff6ff" : "#fff", padding: 10
                      }}>
                        <div style={{ fontWeight: 600, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{t.task}</div>
                        <div style={{ color: "#64748b", fontSize: 12, marginTop: 4 }}>{t.model} · {t.total_steps} steps · {t.success ? "PASS" : "FAIL"}</div>
                      </button>
                    ))}
                  </div>
                </div>

                <div style={{ display: "grid", gap: 12 }}>
                  <ModelComparison rows={modelRows} />
                  <div style={card}>
                    <SectionTitle title={current ? current.task : "No trace selected"} subtitle="Why this run succeeded or failed" />
                    <Timeline steps={current ? current.steps : []} onSelectStep={setSelectedStep} />
                    {selectedStep && (
                      <div style={{ marginTop: 12, border: "1px solid #dbeafe", background: "#eff6ff", borderRadius: 12, padding: 10 }}>
                        <div style={{ fontSize: 12, color: "#1d4ed8", fontWeight: 700, marginBottom: 4 }}>Step Detail</div>
                        <div style={{ fontSize: 12, color: "#334155" }}><strong>ID:</strong> {selectedStep.id}</div>
                        <div style={{ fontSize: 12, color: "#334155" }}><strong>Tool:</strong> {selectedStep.tool}</div>
                        <div style={{ fontSize: 12, color: "#334155" }}><strong>Status:</strong> {selectedStep.status} · <strong>Verdict:</strong> {selectedStep.verdict}</div>
                        <div style={{ fontSize: 12, color: "#334155" }}><strong>Retries:</strong> {selectedStep.retries}</div>
                        {selectedStep.failure && <div style={{ fontSize: 12, color: "#b91c1c" }}><strong>Failure:</strong> {selectedStep.failure}</div>}
                      </div>
                    )}
                  </div>
                </div>

                <div style={{ display: "grid", gap: 12 }}>
                  <RecommendationPanel items={recommendations} />
                  <div style={card}>
                    <SectionTitle title="Top Tools" />
                    <BarRows rows={topTools} color="#2563eb" emptyText="Run traces to see tool usage" />
                  </div>
                  <div style={card}>
                    <SectionTitle title="Failure Taxonomy" />
                    <BarRows rows={failures} color="#dc2626" emptyText="No failures captured in current traces" />
                  </div>
                  <DistributionPills title="Step Status Mix" rows={statusDist} tone={PALETTE.primary} />
                  <DistributionPills title="Reflection Verdict Mix" rows={verdictDist} tone={PALETTE.violet} />
                </div>
              </div>
            </div>
          </div>
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


def _package_version() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        return str((data.get("project") or {}).get("version", "0.0.0"))
    except Exception:
        return "unknown"


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
    if not traces:
        return {
            "trace_count": 0,
            "success_rate": 0.0,
            "avg_wall_time": 0.0,
            "avg_steps": 0.0,
            "avg_retries": 0.0,
            "rag_step_share": 0.0,
            "rag_success_rate": 0.0,
            "rag_chunks_ingested": 0,
            "top_tools": [],
            "failure_taxonomy": [],
            "model_comparison": [],
            "status_distribution": [],
            "verdict_distribution": [],
            "p50_step_latency_ms": 0.0,
            "p95_step_latency_ms": 0.0,
            "trust_index": 0.0,
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
    success_rate = (sum(1 for t in traces if t["success"]) / trace_count) * 100
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
        model_retry_avg = sum(t["retries"] for t in rows) / len(rows)
        reliability = max(0.0, min(100.0, 100.0 - (model_retry_avg * 15.0)))
        model_rows.append(
            {
                "model": model,
                "success_rate": round(model_success, 1),
                "reliability": round(reliability, 1),
            }
        )

    # Trust index: weighted blend favoring outcomes and stability.
    retry_penalty = min(25.0, avg_retries * 12.0)
    latency_penalty = min(20.0, (p95 / 1000.0) * 8.0)
    failure_penalty = min(35.0, (len(failure_counts) * 4.0))
    trust_index = max(0.0, min(100.0, success_rate - retry_penalty - latency_penalty - failure_penalty + 25.0))

    recommendations: list[str] = []
    top_failures = sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)
    for failure, count in top_failures[:3]:
        if failure == "tool_arg_schema_violation":
            recommendations.append(f"Reduce schema errors ({count}) by adding stricter tool-call examples in executor prompt.")
        elif failure == "tool_execution_failure":
            recommendations.append(f"Address execution failures ({count}) with fallback paths and stronger retry guards.")
        elif failure == "hallucinated_tool":
            recommendations.append(f"Reduce hallucinated tools ({count}) by emphasizing allowed tools in planner/executor prompts.")
        elif failure == "output_parse_error":
            recommendations.append(f"Fix parse issues ({count}) with tighter JSON-output constraints and validation cues.")
        else:
            recommendations.append(f"Investigate recurring failure '{failure}' ({count}) to improve reliability.")
    if avg_retries > 0.8:
        recommendations.append("Retry volume is elevated; review correction quality and tighten stop/replan thresholds.")
    if rag_success_rate < 70 and rag_steps:
        recommendations.append("RAG success is low; evaluate chunking strategy and query formulation.")

    return {
        "trace_count": trace_count,
        "success_rate": success_rate,
        "avg_wall_time": avg_wall,
        "avg_steps": avg_steps,
        "avg_retries": avg_retries,
        "rag_step_share": rag_step_share,
        "rag_success_rate": rag_success_rate,
        "rag_chunks_ingested": rag_chunks,
        "top_tools": sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:8],
        "failure_taxonomy": sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)[:8],
        "model_comparison": model_rows,
        "status_distribution": sorted(status_counts.items(), key=lambda x: x[1], reverse=True),
        "verdict_distribution": sorted(verdict_counts.items(), key=lambda x: x[1], reverse=True),
        "p50_step_latency_ms": p50,
        "p95_step_latency_ms": p95,
        "trust_index": trust_index,
        "recommendations": recommendations,
    }


def _load_dashboard_data(traces_dir: Path) -> TraceDashboardData:
    traces: list[dict[str, Any]] = []
    if traces_dir.exists():
        for path in sorted(traces_dir.glob("*.json"), reverse=True):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                traces.append(_normalize_trace(raw))
            except Exception:
                continue
    return TraceDashboardData(traces=traces, summary=_summarize(traces))


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
            self.end_headers()
            self.wfile.write(body)

        def send_response(self, code: int, message: str | None = None) -> None:  # noqa: N802
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
            t0 = time.perf_counter()
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/api/health":
                    self._send_json(
                        {
                            "status": "ok",
                            "service": "archon-dashboard",
                            "version": _package_version(),
                            "traces_dir": str(traces_dir),
                            "rag_bearer_required": bool(rag_api_token),
                            "rate_limiting": rate_meta,
                        }
                    )
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
                    data = _load_dashboard_data(traces_dir)
                    self._send_json({"traces": data.traces, "summary": data.summary})
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
            finally:
                self._emit_audit("GET", self.path, t0)

        def do_POST(self) -> None:  # noqa: N802
            self._request_id = self._ensure_request_id()
            self._response_status = None
            t0 = time.perf_counter()
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
            finally:
                self._emit_audit("POST", self.path, t0)

        def log_message(self, format: str, *args: Any) -> None:
            _log.debug("%s - %s", self.address_string(), format % args)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Archon dashboard running at http://{host}:{port}")
    print(f"Reading traces from: {traces_dir}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
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
