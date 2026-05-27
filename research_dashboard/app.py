from __future__ import annotations

import argparse
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .artifacts import preview_artifact
from .snapshot import build_snapshot, resolve_project_root


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vesuvius AutoResearch Control Center</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: dark;
      --bg: #070913;
      --card-bg: rgba(15, 23, 42, 0.55);
      --card-border: rgba(51, 65, 85, 0.45);
      --card-hover: rgba(30, 41, 59, 0.8);
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #06b6d4;
      --accent-glow: rgba(6, 182, 212, 0.12);
      --accent-purple: #a855f7;
      --accent-purple-glow: rgba(168, 85, 247, 0.12);
      --success: #10b981;
      --warning: #f59e0b;
      --error: #ef4444;
      --line: #1e293b;
      --font-sans: 'Inter', ui-sans-serif, system-ui, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 0;
      font-family: var(--font-sans);
      background: radial-gradient(circle at top left, #0e172a, #070913 60%);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }
    
    header {
      padding: 1rem 1.5rem;
      border-bottom: 1px solid var(--line);
      background: rgba(15, 23, 42, 0.4);
      backdrop-filter: blur(8px);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    
    header h1 {
      font-size: 1.25rem;
      font-weight: 800;
      margin: 0;
      letter-spacing: -0.025em;
      background: linear-gradient(to right, #f8fafc, #06b6d4);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    
    header p {
      margin: 0.15rem 0 0;
      font-size: 0.75rem;
      color: var(--muted);
      font-family: var(--font-mono);
    }
    
    .ctrl-group {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }
    
    select, button {
      background: rgba(30, 41, 59, 0.6);
      color: var(--text);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 0.45rem 0.75rem;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      transition: all 0.15s ease;
      font-family: var(--font-sans);
    }
    
    select:hover, button:hover {
      border-color: var(--accent);
      background: rgba(30, 41, 59, 0.9);
      box-shadow: 0 0 10px var(--accent-glow);
    }
    
    button.primary {
      background: var(--accent);
      color: #070913;
      border-color: var(--accent);
      font-weight: 700;
    }
    button.primary:hover {
      background: #22d3ee;
      border-color: #22d3ee;
      box-shadow: 0 0 15px rgba(6, 182, 212, 0.4);
    }
    
    main {
      padding: 1.25rem;
      max-width: 1600px;
      margin: 0 auto;
      display: grid;
      gap: 1.25rem;
    }
    
    .telemetry-row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
    }
    
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 1.25rem;
      backdrop-filter: blur(12px);
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    
    .panel:hover {
      border-color: rgba(6, 182, 212, 0.35);
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.3);
    }
    
    .panel h2 {
      font-size: 0.8rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin: 0 0 0.5rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    
    .panel-val {
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text);
      letter-spacing: -0.03em;
      margin: 0 0 0.25rem;
      font-family: var(--font-mono);
    }
    
    .panel-detail {
      font-size: 0.72rem;
      color: var(--muted);
      margin: 0;
      display: flex;
      align-items: center;
      gap: 0.35rem;
    }
    
    .indicator-badge {
      display: inline-flex;
      align-items: center;
      padding: 0.15rem 0.45rem;
      border-radius: 9999px;
      font-size: 0.65rem;
      font-weight: 700;
      text-transform: uppercase;
      border: 1px solid currentColor;
    }
    .badge-success { color: var(--success); background: rgba(16, 185, 129, 0.08); }
    .badge-warning { color: var(--warning); background: rgba(245, 158, 11, 0.08); }
    .badge-error { color: var(--error); background: rgba(239, 68, 68, 0.08); }
    
    .split-layout {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(0, 1fr);
      gap: 1.25rem;
    }
    
    @media (max-width: 1024px) {
      .split-layout {
        grid-template-columns: 1fr;
      }
    }
    
    .chart-container {
      position: relative;
      height: 190px;
      width: 100%;
      border: 1px solid rgba(255,255,255,0.03);
      border-radius: 8px;
      padding: 0.5rem;
      background: rgba(7, 9, 19, 0.3);
      margin-top: 0.5rem;
    }
    
    .chart-point {
      cursor: pointer;
      transition: r 0.15s ease, stroke-width 0.15s ease;
    }
    .chart-point:hover {
      r: 6;
      stroke: var(--text);
      stroke-width: 1.5;
    }
    
    .matrix-grid {
      display: grid;
      gap: 0.25rem;
      margin-top: 0.5rem;
      overflow-x: auto;
      padding-bottom: 0.5rem;
    }
    
    .matrix-cell {
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 0.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.15s ease;
      min-width: 60px;
    }
    .matrix-cell:hover {
      transform: scale(1.05);
      z-index: 10;
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(6,182,212,0.2);
    }
    .matrix-header {
      font-size: 0.65rem;
      font-weight: 700;
      color: var(--muted);
      text-align: center;
      padding: 0.25rem;
      font-family: var(--font-mono);
    }
    .matrix-row-title {
      font-size: 0.65rem;
      font-weight: 700;
      color: var(--muted);
      display: flex;
      align-items: center;
      padding: 0.5rem;
      font-family: var(--font-mono);
    }
    
    .table-container {
      width: 100%;
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(7, 9, 19, 0.25);
    }
    
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.78rem;
      text-align: left;
    }
    
    th {
      background: rgba(30, 41, 59, 0.35);
      padding: 0.65rem 0.85rem;
      color: var(--muted);
      font-weight: 600;
      border-bottom: 1px solid var(--line);
      text-transform: uppercase;
      font-size: 0.68rem;
      letter-spacing: 0.05em;
    }
    
    td {
      padding: 0.65rem 0.85rem;
      border-bottom: 1px solid var(--line);
      color: #cbd5e1;
    }
    
    tr {
      transition: background-color 0.15s ease;
    }
    tr:hover {
      background-color: var(--card-hover);
    }
    
    tr.selected {
      background-color: rgba(6, 182, 212, 0.08) !important;
      border-left: 3px solid var(--accent);
    }
    
    .run-pill {
      font-family: var(--font-mono);
      font-size: 0.7rem;
      padding: 0.15rem 0.45rem;
      border-radius: 4px;
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid var(--line);
    }
    
    .drawer {
      border-top: 1px solid var(--line);
      background: rgba(10, 16, 28, 0.9);
      padding: 1rem;
      animation: slideDown 0.2s ease-out;
    }
    
    @keyframes slideDown {
      from { opacity: 0; transform: translateY(-5px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    .pill {
      display: inline-flex;
      align-items: center;
      background: rgba(30, 41, 59, 0.45);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0.25rem 0.5rem;
      font-size: 0.7rem;
      color: #cbd5e1;
      margin: 0.15rem;
      font-family: var(--font-mono);
      transition: all 0.15s ease;
    }
    .pill:hover {
      border-color: var(--accent);
      color: var(--text);
    }
    
    .pill.available {
      border-color: rgba(16, 185, 129, 0.4);
      background: rgba(16, 185, 129, 0.04);
    }
    .pill.missing {
      border-color: rgba(239, 68, 68, 0.3);
      background: rgba(239, 68, 68, 0.04);
      opacity: 0.75;
    }
    
    .diff-list {
      display: grid;
      gap: 0.35rem;
      max-height: 250px;
      overflow-y: auto;
      padding-right: 0.25rem;
    }
    .diff-item {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
      padding: 0.4rem 0.5rem;
      background: rgba(7, 9, 19, 0.4);
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 0.72rem;
    }
    .diff-key {
      font-family: var(--font-mono);
      font-weight: 600;
      color: var(--accent);
    }
    .diff-val-change {
      display: flex;
      align-items: center;
      gap: 0.5rem;
      font-family: var(--font-mono);
    }
    .diff-before { color: var(--error); text-decoration: line-through; }
    .diff-arrow { color: var(--muted); }
    .diff-after { color: var(--success); font-weight: 600; }
    
    pre {
      font-family: var(--font-mono);
      font-size: 0.72rem;
      background: #04060b;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0.75rem;
      color: #cbd5e1;
      overflow: auto;
      margin: 0;
      white-space: pre-wrap;
      word-break: break-all;
    }
    
    .terminal-header {
      background: rgba(30, 41, 59, 0.3);
      border: 1px solid var(--line);
      border-bottom: none;
      border-radius: 8px 8px 0 0;
      padding: 0.5rem 0.75rem;
      font-family: var(--font-mono);
      font-size: 0.7rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .terminal-body {
      border-radius: 0 0 8px 8px;
      max-height: 250px;
    }
    
    .console-input {
      background: rgba(7, 9, 19, 0.6);
      border: 1px solid var(--line);
      border-radius: 4px;
      color: var(--text);
      padding: 0.2rem 0.5rem;
      font-size: 0.7rem;
      font-family: var(--font-sans);
      width: 140px;
    }
    .console-input:focus {
      outline: none;
      border-color: var(--accent);
    }
    
    .milestone-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.5rem 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
      font-size: 0.76rem;
    }
    .milestone-item:last-child { border: none; }
    .milestone-label { display: flex; align-items: center; gap: 0.5rem; font-weight: 500; }
    
    .circle-indicator {
      width: 8px;
      height: 8px;
      border-radius: 9999px;
      display: inline-block;
    }
    .circle-done { background: var(--success); box-shadow: 0 0 6px var(--success); }
    .circle-warning { background: var(--warning); box-shadow: 0 0 6px var(--warning); }
    .circle-active { background: var(--accent); box-shadow: 0 0 6px var(--accent); animation: pulse 1.5s infinite; }
    .circle-pending { background: var(--line); }
    
    @keyframes pulse {
      0% { transform: scale(0.9); opacity: 0.6; }
      50% { transform: scale(1.1); opacity: 1; }
      100% { transform: scale(0.9); opacity: 0.6; }
    }
    
    .proc-item {
      padding: 0.5rem;
      background: rgba(30, 41, 59, 0.25);
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-bottom: 0.35rem;
      font-size: 0.72rem;
    }
    .proc-title { font-family: var(--font-mono); font-weight: 600; color: var(--accent); margin-bottom: 0.15rem; }
    
    .feature-card {
      padding: 0.65rem 0.85rem;
      background: rgba(15, 23, 42, 0.4);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 0.5rem;
      transition: all 0.2s ease;
    }
    .feature-card:hover {
      border-color: rgba(6, 182, 212, 0.25);
      background: rgba(30, 41, 59, 0.3);
    }
    .feature-title { font-size: 0.8rem; font-weight: 700; margin: 0 0 0.15rem; display: flex; justify-content: space-between; align-items: center; }
    .feature-desc { font-size: 0.72rem; color: var(--muted); margin: 0 0 0.4rem; }
    
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: var(--card-bg);
      border: 1px solid var(--accent);
      padding: 0.75rem 1.25rem;
      border-radius: 8px;
      font-size: 0.78rem;
      color: var(--text);
      z-index: 1000;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
      animation: fadeInUp 0.2s ease-out;
      backdrop-filter: blur(10px);
    }
    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Vesuvius AutoResearch Control Center</h1>
      <p id="meta-info">Loading snapshot...</p>
    </div>
    <div class="ctrl-group">
      <select id="poll-interval" onchange="setupPolling()">
        <option value="0">Manual Refresh</option>
        <option value="5000">Poll: 5s</option>
        <option value="10000" selected>Poll: 10s</option>
        <option value="30000">Poll: 30s</option>
      </select>
      <button class="primary" onclick="loadDashboard()">Refresh</button>
    </div>
  </header>

  <main>
    <!-- Scorecard row -->
    <section class="telemetry-row">
      <div class="panel">
        <h2>Peak Score</h2>
        <div class="panel-val" id="best-f1">-</div>
        <p class="panel-detail" id="best-f1-detail">Peak run ID</p>
      </div>
      <div class="panel">
        <h2>Robust Champion</h2>
        <div class="panel-val" id="robust-f1">-</div>
        <p class="panel-detail" id="robust-detail">Robust candidate</p>
      </div>
      <div class="panel">
        <h2>Promotion Eligible</h2>
        <div class="panel-val" id="promote-f1">-</div>
        <p class="panel-detail" id="promote-detail">Promotion gate</p>
      </div>
      <div class="panel">
        <h2>Decision Brief</h2>
        <div class="panel-val" id="ops-lock-status">-</div>
        <p class="panel-detail" id="ops-process-detail">Next action</p>
      </div>
    </section>

    <!-- Main Workspace Split -->
    <div class="split-layout">
      
      <!-- Left Column: Metrics, Heatmaps, ledger -->
      <section style="display:grid; gap:1.25rem;">
        
        <!-- F1 / AP Line Graph -->
        <div class="panel">
          <h2>Chronological Performance Trends <span style="font-size:0.7rem;font-weight:normal;color:var(--muted);text-transform:none;"><span style="color:var(--accent);margin-right:1rem;">● Validation F1</span> <span style="color:var(--accent-purple);">◆ Average Precision</span></span></h2>
          <div class="chart-container" id="trend-chart-container">
            <div style="color:var(--muted);text-align:center;padding:4.5rem 0;font-size:0.8rem;">Initialising chart rendering engine...</div>
          </div>
        </div>

        <!-- 2D Validation Segment heat-map matrix -->
        <div class="panel">
          <h2>Validation Fold Matrix (Train → Validate F1 Cross-grid)</h2>
          <div id="segment-matrix-container">
            <div style="color:var(--muted);text-align:center;padding:2rem;font-size:0.8rem;">Discovering cross-fold matrix variables...</div>
          </div>
        </div>

        <!-- Table Ledger of Runs -->
        <div class="panel">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
            <h2 style="margin:0;">Experiment Runs Ledger <span id="ledger-filter-badge" style="display:none;margin-left:0.5rem;" class="indicator-badge badge-warning">Filter Active</span></h2>
            <button id="clear-filter-btn" style="display:none; padding:0.25rem 0.5rem; font-size:0.7rem;" onclick="clearSegmentFilter()">Clear Filter</button>
          </div>
          <div class="table-container">
            <table>
              <thead>
                <tr>
                  <th>Run ID</th>
                  <th>Model Architecture</th>
                  <th>F1 Score</th>
                  <th>Avg Precision</th>
                  <th>Ink Prevalence Ratio</th>
                  <th>Val Split</th>
                </tr>
              </thead>
              <tbody id="runs-table-body">
                <tr><td colspan="6" style="text-align:center;color:var(--muted);padding:2rem;">Acquiring experiment SQLite parameters...</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- Dynamic inline artifact previews -->
        <div class="panel" id="preview-panel" style="display:none;">
          <h2>Artifact Reader: <span id="preview-file-name" style="font-family:var(--font-mono);font-size:0.7rem;color:var(--accent);">file_name.json</span></h2>
          <div style="margin-top:0.5rem;" id="preview-display-wrapper"></div>
        </div>
      </section>

      <!-- Right Column: Process monitoring, milestons, configs, commands, logs -->
      <section style="display:grid; gap:1.25rem; align-content:start;">
        
        <!-- Live processes -->
        <div class="panel">
          <h2>Active Processes</h2>
          <div id="process-monitor-list">
            <div style="color:var(--muted);font-size:0.75rem;padding:0.25rem 0;">Scanning background training states...</div>
          </div>
        </div>

        <!-- Readiness Checklist -->
        <div class="panel">
          <h2 id="foundation-readiness-title">Milestone Readiness</h2>
          <div id="milestone-checklist">
            <div style="color:var(--muted);font-size:0.75rem;padding:0.25rem 0;">Evaluating pipeline validation gates...</div>
          </div>
        </div>

        <!-- Candidate-linked evidence and next actions -->
        <div class="panel">
          <h2>Candidate Evidence</h2>
          <div id="candidate-evidence-panel">
            <div style="color:var(--muted);font-size:0.75rem;padding:0.25rem 0;">Resolving linked LOO and full-tile diagnostics...</div>
          </div>
        </div>

        <!-- Selected Run Config Diff Panel -->
        <div class="panel">
          <h2>Hyperparameter Config Diffs <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--accent);text-transform:none;" id="diff-comparison-badge">Best vs Latest</span></h2>
          <div style="display:flex; gap:0.35rem; margin-bottom:0.5rem;">
            <select id="diff-base-select" style="padding:0.25rem 0.5rem; font-size:0.7rem; flex:1;" onchange="updateDiffView()">
              <option value="best" selected>Compare vs Champion Best</option>
              <option value="baseline">Compare vs Baseline Run</option>
              <option value="previous">Compare vs Previous Run</option>
            </select>
          </div>
          <div class="diff-list" id="diff-visual-list">
            <div style="color:var(--muted);font-size:0.75rem;padding:1rem;text-align:center;">Select a run to compare configuration trees.</div>
          </div>
        </div>

        <!-- Discovery Config Catalog badges -->
        <div class="panel">
          <h2>Config files</h2>
          <div id="configs-badge-container"></div>
        </div>

        <!-- Script Command accordion execution center -->
        <div class="panel">
          <h2>Feature Command center</h2>
          <div id="commands-accordion-container" style="display:grid; gap:0.5rem; margin-top:0.5rem;"></div>
        </div>

        <!-- Tailable regex console log -->
        <div class="panel">
          <div class="terminal-header">
            <span>Terminal tail: autoresearch.log</span>
            <input type="text" id="log-search" class="console-input" placeholder="Regex filter..." oninput="filterLogs()">
          </div>
          <pre class="terminal-body" id="log-tail-console">Streaming workspace logtail lines...</pre>
        </div>
      </section>
    </div>
  </main>

  <div id="toast-wrapper"></div>

  <script>
    let rawData = null;
    let selectedRunId = null;
    let pollIntervalId = null;
    let selectedTrainSegment = null;
    let selectedValSegment = null;
    let fullLogs = [];

    const fmt = (v, d = 4) => {
      const num = Number(v);
      return Number.isFinite(num) ? num.toFixed(d) : 'n/a';
    };
    
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[c]));

    const dashboardToken = new URLSearchParams(window.location.search).get('token') || sessionStorage.getItem('vesuvius_dashboard_token') || '';
    if (dashboardToken) sessionStorage.setItem('vesuvius_dashboard_token', dashboardToken);
    function apiUrl(path) {
      if (!dashboardToken) return path;
      const url = new URL(path, window.location.origin);
      url.searchParams.set('token', dashboardToken);
      return url.pathname + url.search;
    }

    function showToast(message) {
      const wrapper = document.getElementById('toast-wrapper');
      const t = document.createElement('div');
      t.className = 'toast';
      t.textContent = message;
      wrapper.appendChild(t);
      setTimeout(() => t.remove(), 3500);
    }

    async function loadDashboard() {
      try {
        const response = await fetch(apiUrl('/api/research'));
        if (!response.ok) throw new Error(`Snapshot fetch failure: ${response.statusText}`);
        rawData = await response.json();
        renderDashboard();
      } catch (err) {
        showToast(`Sync Failed: ${err.message}`);
        console.error(err);
      }
    }

    function setupPolling() {
      if (pollIntervalId) clearInterval(pollIntervalId);
      const val = parseInt(document.getElementById('poll-interval').value);
      if (val > 0) {
        pollIntervalId = setInterval(loadDashboard, val);
      }
    }

    function renderDashboard() {
      if (!rawData) return;
      
      // 1. Meta-information
      document.getElementById('meta-info').textContent = `${esc(rawData.project.root)} · Schema version: ${esc(rawData.schema_version)} · Gen: ${new Date(rawData.generated_at).toLocaleTimeString()}`;
      
      // 2. Scorecard Telemetry
      const score = rawData.progress.scorecard || {};
      const summary = rawData.progress.summary || {};
      const sanity = rawData.progress.sanity || {};
      const decision = rawData.research_summary?.decision || rawData.experiments.decision || {};
      const champions = rawData.research_summary?.champions || rawData.experiments.champions || {};
      const peak = champions.peak_score || rawData.experiments.best || {};
      const robust = champions.robust_candidate || {};
      const promotable = champions.promotion_eligible || {};
      const blockerCounts = decision.blocker_counts || summary.blocker_counts || {};
      const blockerText = Object.entries(blockerCounts).slice(0, 2).map(([k, v]) => `${k}:${v}`).join(', ') || 'none';
      
      document.getElementById('best-f1').textContent = fmt(peak.metrics?.val_f1 ?? score.best);
      document.getElementById('best-f1-detail').innerHTML = `<span class="run-pill" style="cursor:pointer;" onclick="selectRun('${peak.run_id || score.best_run_id}')">${esc((peak.run_id || score.best_run_id || 'n/a').slice(0, 8))}</span> · gain ${fmt(score.gain_vs_baseline, 4)}`;
      
      document.getElementById('robust-f1').textContent = fmt(robust.metrics?.val_f1 ?? robust.main_metric);
      document.getElementById('robust-detail').innerHTML = robust.run_id ? `<span class="run-pill" style="cursor:pointer;" onclick="selectRun('${robust.run_id}')">${esc(robust.run_id.slice(0, 8))}</span> · ${esc(robust.promotion_status || 'unknown')}` : 'No robust candidate';

      document.getElementById('promote-f1').textContent = fmt(promotable.metrics?.val_f1 ?? promotable.main_metric);
      document.getElementById('promote-f1').className = `panel-val ${promotable.run_id ? 'badge-success' : 'badge-warning'}`;
      document.getElementById('promote-detail').innerHTML = promotable.run_id ? `<span class="run-pill" style="cursor:pointer;" onclick="selectRun('${promotable.run_id}')">${esc(promotable.run_id.slice(0, 8))}</span> · eligible` : `blocked · ${esc(blockerText)}`;
      
      const ops = rawData.operations || {};
      document.getElementById('ops-lock-status').textContent = ops.lock_active ? "LOCKED" : esc(decision.status || summary.status || 'IDLE').toUpperCase().slice(0, 14);
      document.getElementById('ops-lock-status').className = `panel-val ${ops.lock_active || decision.status === 'blocked' ? 'badge-warning' : 'badge-success'}`;
      document.getElementById('ops-process-detail').textContent = `${(ops.processes || []).length} active runs · next: ${decision.next_action || summary.next_action || 'review'}`;
      
      // Auto-select latest run initially if nothing selected
      if (!selectedRunId && score.latest_run_id) {
        selectedRunId = score.latest_run_id;
      }

      // 3. Render Custom Components
      renderMilestones(rawData.progress.milestones, summary.foundation_readiness);
      renderCandidateEvidence(decision.candidate_evidence || rawData.research_summary?.candidate_evidence || {});
      renderTrendChart(rawData.experiments.metric_trends);
      renderFoldMatrix(rawData.experiments.validation_matrix);
      renderRecentRuns(rawData.experiments.recent);
      renderActiveProcesses(ops.processes);
      renderConfigBadges(rawData.configs);
      renderCommands(rawData.inventory.features);
      renderLogTail(ops.logs);
      updateDiffView();
    }

    function renderCandidateEvidence(evidence) {
      const container = document.getElementById('candidate-evidence-panel');
      if (!container) return;
      if (!evidence || !evidence.candidate_run_id) {
        container.innerHTML = '<div style="color:var(--muted);font-size:0.75rem;">No robust candidate evidence selected.</div>';
        return;
      }
      const loo = evidence.loo || {};
      const weak = evidence.weak_fold_full_tile || {};
      const full = evidence.full_tile || {};
      const looFull = evidence.loo_full_tile || {};
      const actions = evidence.promotion_actions || [];
      const action = actions[0] || {};
      const cmd = action.command_text || weak.command_text;
      container.innerHTML = `
        <div class="milestone-item"><span class="milestone-label">Candidate</span><span class="run-pill" style="cursor:pointer;" onclick="selectRun('${esc(evidence.candidate_run_id)}')">${esc(String(evidence.candidate_run_id).slice(0, 8))}</span></div>
        <div class="milestone-item"><span class="milestone-label">LOO weak fold</span><span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);">${esc(loo.worst_fold_id || 'unknown')} · F1 ${fmt(loo.worst_fold_val_f1, 4)}</span></div>
        <div class="milestone-item"><span class="milestone-label">Full-tile coverage</span><span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);">${esc((full.segments_covered || []).join(', ') || 'none')}</span></div>
        <div class="milestone-item"><span class="milestone-label">LOO tile panel</span><span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--muted);">${esc((looFull.segments_covered || []).join(', ') || 'none')} · ${esc(String(looFull.coverage_count || 0))} runs</span></div>
        <div class="milestone-item"><span class="milestone-label">Weak-fold tile</span><span class="indicator-badge ${weak.status === 'done' ? 'badge-success' : 'badge-warning'}">${esc(weak.status || 'unknown')}</span></div>
        <div style="margin-top:0.5rem;color:var(--text);font-size:0.75rem;">${esc(action.label || 'Review candidate evidence')}</div>
        ${cmd ? `<button style="margin-top:0.5rem;width:100%;font-size:0.68rem;" onclick="copyToClipboard('${esc(cmd).replace(/'/g, '&#39;')}')">Copy next command</button>` : ''}
      `;
    }

    // A. Milestones Readiness
    function renderMilestones(milestones, readinessPercent) {
      const readinessClass = readinessPercent >= 100 ? 'badge-success' : 'badge-warning';
      document.getElementById('foundation-readiness-title').innerHTML = `Promotion Gate <span class="indicator-badge ${readinessClass}" style="float:right;">${readinessPercent}%</span>`;
      const container = document.getElementById('milestone-checklist');
      if (!milestones) return;
      container.innerHTML = milestones.map(m => {
        let stateClass = "circle-pending";
        if (m.state === "done") stateClass = "circle-done";
        else if (m.state === "warning") stateClass = "circle-warning";
        else if (m.state === "active") stateClass = "circle-active";
        
        return `
          <div class="milestone-item">
            <span class="milestone-label"><span class="circle-indicator ${stateClass}"></span> ${esc(m.label)}</span>
            <span style="font-size:0.68rem; color:var(--muted); font-family:var(--font-mono);">${esc(m.detail)}</span>
          </div>
        `;
      }).join('');
    }

    // B. Interactive SVG Chart
    function renderTrendChart(trends) {
      const container = document.getElementById('trend-chart-container');
      if (!trends || trends.length === 0) {
        container.innerHTML = '<div style="color:var(--muted);text-align:center;padding:4.5rem 0;font-size:0.8rem;">No training runs logged in experiment ledger</div>';
        return;
      }
      
      const width = container.clientWidth || 700;
      const height = 180;
      const paddingLeft = 32;
      const paddingRight = 10;
      const paddingTop = 12;
      const paddingBottom = 20;
      
      let maxVal = 0.0;
      let minVal = 1.0;
      
      trends.forEach(t => {
        const f1 = parseFloat(t.val_f1 ?? t.main_metric ?? 0);
        const ap = parseFloat(t.average_precision ?? 0);
        maxVal = Math.max(maxVal, f1, ap);
        minVal = Math.min(minVal, f1, ap);
      });
      
      maxVal = Math.min(1.0, maxVal + 0.05);
      minVal = Math.max(0.0, minVal - 0.05);
      if (maxVal === minVal) { maxVal = 1.0; minVal = 0.0; }
      
      const chartW = width - paddingLeft - paddingRight;
      const chartH = height - paddingTop - paddingBottom;
      
      const pointsF1 = [];
      const pointsAP = [];
      trends.forEach((t, i) => {
        const x = paddingLeft + (i / (trends.length - 1 || 1)) * chartW;
        const f1 = parseFloat(t.val_f1 ?? t.main_metric ?? 0);
        const ap = parseFloat(t.average_precision ?? 0);
        
        const yF1 = paddingTop + chartH - ((f1 - minVal) / (maxVal - minVal)) * chartH;
        const yAP = paddingTop + chartH - ((ap - minVal) / (maxVal - minVal)) * chartH;
        
        pointsF1.push({ x, y: yF1, runId: t.run_id, val: f1 });
        pointsAP.push({ x, y: yAP, runId: t.run_id, val: ap });
      });
      
      let pathF1 = '';
      let pathAP = '';
      if (pointsF1.length > 0) {
        pathF1 = `M ${pointsF1[0].x} ${pointsF1[0].y} ` + pointsF1.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ');
      }
      if (pointsAP.length > 0) {
        pathAP = `M ${pointsAP[0].x} ${pointsAP[0].y} ` + pointsAP.slice(1).map(p => `L ${p.x} ${p.y}`).join(' ');
      }
      
      let gridLines = '';
      for (let steps = 0; steps <= 4; steps++) {
        const fraction = steps / 4;
        const val = minVal + fraction * (maxVal - minVal);
        const y = paddingTop + chartH - fraction * chartH;
        gridLines += `<line x1="${paddingLeft}" y1="${y}" x2="${width - paddingRight}" y2="${y}" stroke="rgba(255,255,255,0.04)" stroke-dasharray="2,4" />`;
        gridLines += `<text x="${paddingLeft - 6}" y="${y + 3}" fill="var(--muted)" font-size="8" font-family="var(--font-mono)" text-anchor="end">${val.toFixed(2)}</text>`;
      }
      
      let interactivePoints = '';
      pointsF1.forEach((p, idx) => {
        const isSelected = p.runId === selectedRunId;
        interactivePoints += `
          <circle cx="${p.x}" cy="${p.y}" r="${isSelected ? 6 : 3.5}" 
                  fill="${isSelected ? '#ffffff' : 'var(--accent)'}" 
                  stroke="var(--accent)" stroke-width="${isSelected ? 2 : 0}"
                  class="chart-point" 
                  onclick="selectRun('${p.runId}')">
            <title>F1: ${p.val.toFixed(4)} (Run: ${p.runId.slice(0,8)})</title>
          </circle>
        `;
      });
      
      pointsAP.forEach((p, idx) => {
        const isSelected = p.runId === selectedRunId;
        interactivePoints += `
          <rect x="${p.x - 3.5}" y="${p.y - 3.5}" width="${isSelected ? 9 : 7}" height="${isSelected ? 9 : 7}"
                fill="${isSelected ? '#ffffff' : 'var(--accent-purple)'}" 
                stroke="var(--accent-purple)" stroke-width="${isSelected ? 1.5 : 0}"
                class="chart-point"
                onclick="selectRun('${p.runId}')">
            <title>AP: ${p.val.toFixed(4)} (Run: ${p.runId.slice(0,8)})</title>
          </rect>
        `;
      });
      
      container.innerHTML = `
        <svg width="100%" height="${height}" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" style="overflow:visible;">
          ${gridLines}
          <path d="${pathF1}" fill="none" stroke="var(--accent)" stroke-width="2.5" />
          <path d="${pathAP}" fill="none" stroke="var(--accent-purple)" stroke-width="1.5" stroke-dasharray="3,3" />
          ${interactivePoints}
        </svg>
      `;
    }

    // C. 2D Fold Matrix Heatmap Grid
    function renderFoldMatrix(matrix) {
      const container = document.getElementById('segment-matrix-container');
      if (!matrix || matrix.length === 0) {
        container.innerHTML = '<div style="color:var(--muted);text-align:center;padding:1.5rem;font-size:0.75rem;">No cross-segment validation fold mappings discovered</div>';
        return;
      }
      
      // Discover all unique Train segments (Rows) and Val segments (Columns)
      const trains = [...new Set(matrix.map(cell => cell.train_segment_id))].sort();
      const vals = [...new Set(matrix.map(cell => cell.val_segment_id))].sort();
      
      let html = `<div class="matrix-grid" style="grid-template-columns: 85px repeat(${vals.length}, 1fr);">`;
      
      // Matrix Header Row
      html += `<div></div>`;
      vals.forEach(v => {
        html += `<div class="matrix-header">Val: ${esc(v)}</div>`;
      });
      
      // Matrix Rows
      trains.forEach(t => {
        html += `<div class="matrix-row-title">Train: ${esc(t)}</div>`;
        vals.forEach(v => {
          const cell = matrix.find(c => c.train_segment_id === t && c.val_segment_id === v);
          if (cell) {
            const f1 = parseFloat(cell.best_val_f1 ?? cell.best_main_metric ?? 0.0);
            
            // Color grade cells based on F1
            let hue = 15; // default bad (red/orange)
            let sat = 75;
            let light = 32;
            
            if (f1 > 0.85) {
              hue = 145; // healthy emerald green
              light = 30;
            } else if (f1 > 0.70) {
              hue = 85;  // olive/sage green
              light = 28;
            } else if (f1 > 0.50) {
              hue = 40;  // amber/yellow
              light = 28;
            }
            
            const isCurrentFilter = selectedTrainSegment === t && selectedValSegment === v;
            const style = `background: hsl(${hue}, ${sat}%, ${light}%); border-color: ${isCurrentFilter ? 'var(--text)' : 'var(--line)'}; font-family:var(--font-mono);`;
            
            html += `
              <div class="matrix-cell" style="${style}" onclick="toggleSegmentFilter('${esc(t)}', '${esc(v)}')" title="F1 Score: ${f1.toFixed(4)} over ${cell.run_count} runs. Click to filter ledger.">
                <div style="font-size:0.75rem; font-weight:700; color:#fff;">${f1.toFixed(3)}</div>
                <div style="font-size:0.55rem; color:rgba(255,255,255,0.7);">runs: ${cell.run_count}</div>
              </div>
            `;
          } else {
            html += `<div class="matrix-cell" style="background:rgba(255,255,255,0.01);opacity:0.3;cursor:not-allowed;"><div style="font-size:0.7rem;color:var(--muted);">-</div></div>`;
          }
        });
      });
      
      html += `</div>`;
      container.innerHTML = html;
    }

    // Toggle segment ledger filtering
    function toggleSegmentFilter(train, val) {
      if (selectedTrainSegment === train && selectedValSegment === val) {
        clearSegmentFilter();
      } else {
        selectedTrainSegment = train;
        selectedValSegment = val;
        document.getElementById('ledger-filter-badge').style.display = 'inline-flex';
        document.getElementById('clear-filter-btn').style.display = 'inline-block';
        renderRecentRuns(rawData.experiments.recent);
        showToast(`Table Filtered: Segment ${train} → ${val}`);
      }
    }

    function clearSegmentFilter() {
      selectedTrainSegment = null;
      selectedValSegment = null;
      document.getElementById('ledger-filter-badge').style.display = 'none';
      document.getElementById('clear-filter-btn').style.display = 'none';
      renderRecentRuns(rawData.experiments.recent);
      renderFoldMatrix(rawData.experiments.validation_matrix);
      showToast("Ledger filters cleared");
    }

    // D. Runs Ledger Table
    function renderRecentRuns(recent) {
      const tbody = document.getElementById('runs-table-body');
      let filtered = recent || [];
      
      if (selectedTrainSegment && selectedValSegment) {
        filtered = filtered.filter(run => 
          String(run.validation_setup?.train_segment_id) === selectedTrainSegment &&
          String(run.validation_setup?.val_segment_id) === selectedValSegment
        );
      }
      
      if (filtered.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--muted);padding:2rem;">No runs match current filter segment coordinates.</td></tr>`;
        return;
      }
      
      tbody.innerHTML = filtered.map(run => {
        const isSelected = run.run_id === selectedRunId;
        const m = run.metrics || {};
        const s = run.validation_setup || {};
        const ratioVal = m.pred_positive_rate && m.val_positive_rate ? (parseFloat(m.pred_positive_rate) / Math.max(parseFloat(m.val_positive_rate), 1e-12)) : 1.0;
        
        let drawerHtml = '';
        if (isSelected) {
          const files = run.artifacts || [];
          drawerHtml = `
            <tr class="drawer-row">
              <td colspan="6" style="padding:0;">
                <div class="drawer">
                  <div style="font-weight:700; font-size:0.75rem; color:var(--accent); margin-bottom:0.4rem;">Run Directory Artifact Explorer</div>
                  <div style="margin-bottom:0.75rem;">
                    ${files.map(f => `
                      <span class="pill ${f.kind === 'pt' ? 'missing' : 'available'}" style="cursor:pointer;" onclick="previewArtifact('${esc(f.path)}', '${esc(f.name)}')">
                        📁 ${esc(f.name)} <span style="color:var(--muted);font-size:0.6rem;margin-left:0.25rem;">(${(f.size_bytes / 1024).toFixed(1)}k)</span>
                      </span>
                    `).join('') || '<span style="color:var(--muted);font-size:0.7rem;">No run artifacts discovered.</span>'}
                  </div>
                  <div style="font-size:0.68rem;color:var(--muted);font-family:var(--font-mono);">Path: ${esc(run.artifact_dir)}</div>
                </div>
              </td>
            </tr>
          `;
        }
        
        return `
          <tr class="${isSelected ? 'selected' : ''}" style="cursor:pointer;" onclick="selectRun('${run.run_id}', event)">
            <td><code class="run-pill">${esc(run.run_id.slice(0, 8))}</code></td>
            <td style="font-weight:500;">${esc(m.model_name || run.config?.model?.name || 'unknown')}</td>
            <td style="font-family:var(--font-mono);font-weight:700;color:var(--accent);">${fmt(m.val_f1)}</td>
            <td style="font-family:var(--font-mono);">${fmt(m.average_precision)}</td>
            <td style="font-family:var(--font-mono);" class="${ratioVal > 3.0 || ratioVal < 0.3 ? 'warn':''}">${ratioVal.toFixed(2)}x</td>
            <td><span class="run-pill">${esc(s.train_segment_id)} → ${esc(s.val_segment_id)}</span></td>
          </tr>
          ${drawerHtml}
        `;
      }).join('');
    }

    function selectRun(runId, event) {
      if (event && event.target.closest('.pill')) return; // ignore clicks inside drawer pills
      selectedRunId = runId;
      renderRecentRuns(rawData.experiments.recent);
      renderTrendChart(rawData.experiments.metric_trends);
      updateDiffView();
      showToast(`Selected Run: ${runId.slice(0,8)}`);
    }

    // E. Dynamic Artifact Inline Previewer
    async function previewArtifact(filePath, fileName) {
      const panel = document.getElementById('preview-panel');
      const nameSpan = document.getElementById('preview-file-name');
      const display = document.getElementById('preview-display-wrapper');
      
      panel.style.display = 'block';
      nameSpan.textContent = fileName;
      display.innerHTML = '<div style="color:var(--muted);font-size:0.75rem;font-family:var(--font-mono);padding:1rem;">Fetching remote file bytes...</div>';
      
      // scroll panel into view smoothly
      panel.scrollIntoView({ behavior: 'smooth' });

      try {
        const response = await fetch(apiUrl(`/api/artifact?path=${encodeURIComponent(filePath)}`));
        if (!response.ok) throw new Error(response.statusText);
        const data = await response.json();
        
        if (data.preview_error) {
          display.innerHTML = `<div style="color:var(--error);font-size:0.75rem;font-family:var(--font-mono);">${esc(data.preview_error)}</div>`;
          return;
        }

        const isImg = data.kind && ['png','jpg','jpeg','webp','gif'].includes(data.kind.toLowerCase());
        if (isImg && data.preview) {
          display.innerHTML = `
            <div style="background:#000; padding:1rem; border-radius:8px; border:1px solid var(--line); display:flex; justify-content:center;">
              <img src="${data.preview}" style="max-width:100%; max-height:400px; border-radius:4px; box-shadow:0 0 20px rgba(0,0,0,0.5);" alt="Run Visual Prediction">
            </div>
          `;
        } else if (data.kind === 'json' && typeof data.preview === 'object') {
          display.innerHTML = `<pre>${esc(JSON.stringify(data.preview, null, 2))}</pre>`;
        } else {
          // Syntax highlight text logs / scripts
          let textContent = esc(data.preview || '');
          if (data.truncated) textContent += '\\n\\n[FILE TRUNCATED AT 24,000 CHARACTERS]';
          
          const highlighted = textContent
            .replace(/(\\[INFO\\]|INFO:)/g, '<span style="color:var(--success); font-weight:bold;">$1</span>')
            .replace(/(\\[WARNING\\]|WARNING:|WARN:)/g, '<span style="color:var(--warning); font-weight:bold;">$1</span>')
            .replace(/(\\[ERROR\\]|ERROR:|CRITICAL:)/g, '<span style="color:var(--error); font-weight:bold;">$1</span>');
            
          display.innerHTML = `<pre>${highlighted}</pre>`;
        }
      } catch (err) {
        display.innerHTML = `<div style="color:var(--error);font-size:0.75rem;font-family:var(--font-mono);">Sync Error: ${esc(err.message)}</div>`;
      }
    }

    // F. Update Configuration Diff Panel
    function updateDiffView() {
      const container = document.getElementById('diff-visual-list');
      const badge = document.getElementById('diff-comparison-badge');
      if (!rawData || !selectedRunId) return;
      
      const comparator = document.getElementById('diff-base-select').value;
      const targetRun = rawData.experiments.recent.find(r => r.run_id === selectedRunId);
      
      if (!targetRun) {
        container.innerHTML = '<div style="color:var(--muted);font-size:0.7rem;text-align:center;padding:1rem;">Selected run config is missing from snapshot.</div>';
        return;
      }
      
      badge.textContent = `Selected vs ${comparator.toUpperCase()}`;
      
      let diffArray = [];
      const diffs = rawData.experiments.config_diffs || {};
      
      const isLatest = selectedRunId === (rawData.progress.scorecard?.latest_run_id);
      
      if (isLatest) {
        // If selecting the latest run, we can use the pre-calculated diff arrays from the server
        if (comparator === 'best') diffArray = diffs.latest_vs_best || [];
        else if (comparator === 'baseline') diffArray = diffs.latest_vs_baseline || [];
        else if (comparator === 'previous') diffArray = diffs.latest_vs_previous || [];
      } else {
        // Compute dynamically on client side for arbitrary selected runs
        let baseRun = null;
        if (comparator === 'best') baseRun = rawData.experiments.best;
        else if (comparator === 'baseline') baseRun = rawData.experiments.recent[rawData.experiments.recent.length - 1];
        else if (comparator === 'previous') {
          const idx = rawData.experiments.recent.findIndex(r => r.run_id === selectedRunId);
          if (idx !== -1 && idx + 1 < rawData.experiments.recent.length) {
            baseRun = rawData.experiments.recent[idx + 1];
          }
        }
        
        diffArray = computeClientDiff(baseRun ? baseRun.config : {}, targetRun.config);
      }
      
      if (diffArray.length === 0) {
        container.innerHTML = '<div style="color:var(--success);font-size:0.75rem;text-align:center;padding:1.5rem;font-weight:600;">✓ Hyperparameters are identical (100% match)</div>';
        return;
      }
      
      container.innerHTML = diffArray.map(item => `
        <div class="diff-item">
          <span class="diff-key">${esc(item.path)}</span>
          <div class="diff-val-change">
            <span class="diff-before">${esc(item.before === null ? 'null' : item.before)}</span>
            <span class="diff-arrow">→</span>
            <span class="diff-after">${esc(item.after === null ? 'null' : item.after)}</span>
          </div>
        </div>
      `).join('');
    }

    function computeClientDiff(base, target) {
      function flatten(obj, prefix = '', out = {}) {
        if (obj && typeof obj === 'object') {
          for (const k in obj) {
            flatten(obj[k], prefix ? `${prefix}.${k}` : k, out);
          }
        } else {
          out[prefix] = obj;
        }
        return out;
      }
      const left = flatten(base);
      const right = flatten(target);
      const keys = new Set([...Object.keys(left), ...Object.keys(right)]);
      const diffs = [];
      
      for (const k of keys) {
        const lStr = JSON.stringify(left[k]);
        const rStr = JSON.stringify(right[k]);
        if (lStr !== rStr) {
          diffs.push({ path: k, before: left[k], after: right[k] });
        }
      }
      return diffs.sort((a,b) => a.path.localeCompare(b.path));
    }

    // G. Config Discovery Badges
    function renderConfigBadges(configs) {
      const container = document.getElementById('configs-badge-container');
      if (!configs || configs.length === 0) {
        container.innerHTML = '<div style="color:var(--muted);font-size:0.7rem;">No hyperparameter configs discovered</div>';
        return;
      }
      container.innerHTML = configs.slice(0, 18).map(c => `
        <span class="pill" style="cursor:help;" title="YAML hyperparameter options available inside configs directory">${esc(c.name)}</span>
      `).join('');
    }

    // H. Live Processes Clocks
    function renderActiveProcesses(processes) {
      const container = document.getElementById('process-monitor-list');
      if (!processes || processes.length === 0) {
        container.innerHTML = '<div style="color:var(--success);font-size:0.74rem;padding:0.25rem 0;font-weight:600;">✓ Pipeline Idle (No background runs active)</div>';
        return;
      }
      
      container.innerHTML = processes.map(p => {
        const min = Math.floor(p.age_seconds / 60);
        const sec = p.age_seconds % 60;
        const clock = `${min}m ${sec}s`;
        
        return `
          <div class="proc-item">
            <div class="proc-title">PID ${p.pid} · <span style="color:#fff;">running ${clock}</span></div>
            <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--muted);word-break:break-all;">${esc(p.command)}</div>
          </div>
        `;
      }).join('');
    }

    // I. Feature Script CommandsAccordion
    function renderCommands(features) {
      const container = document.getElementById('commands-accordion-container');
      if (!features || features.length === 0) {
        container.innerHTML = '<div style="color:var(--muted);font-size:0.7rem;">No interactive automation features provided.</div>';
        return;
      }
      
      container.innerHTML = features.map(f => `
        <div class="feature-card">
          <div class="feature-title">
            <span>${esc(f.title)}</span>
            <span class="indicator-badge ${f.available ? 'badge-success':'badge-error'}">${f.available ? 'Ready':'Unavailable'}</span>
          </div>
          <p class="feature-desc">${esc(f.description)}</p>
          <div style="display:flex; justify-content:space-between; align-items:center; gap:0.5rem; flex-wrap:wrap;">
            <div>
              ${(f.tags || []).map(t => `<span class="pill" style="font-size:0.6rem;padding:0.1rem 0.35rem;margin:0.05rem;">${esc(t)}</span>`).join('')}
            </div>
            <button style="padding:0.25rem 0.5rem;font-size:0.68rem;" onclick="copyToClipboard('${esc(f.command_text)}')">Copy Command</button>
          </div>
        </div>
      `).join('');
    }

    function copyToClipboard(text) {
      navigator.clipboard.writeText(text);
      showToast("Command copied to clipboard!");
    }

    // J. Monospace Log tail & search
    function renderLogTail(logs) {
      const consoleNode = document.getElementById('log-tail-console');
      if (!logs || !logs.lines) return;
      
      fullLogs = logs.lines || [];
      filterLogs();
    }

    function filterLogs() {
      const query = document.getElementById('log-search').value.toLowerCase();
      const consoleNode = document.getElementById('log-tail-console');
      
      let lines = fullLogs;
      if (query.trim() !== '') {
        lines = fullLogs.filter(line => line.toLowerCase().includes(query));
      }
      
      if (lines.length === 0) {
        consoleNode.innerHTML = `<span style="color:var(--muted);font-style:italic;">No log lines matched filter query.</span>`;
        return;
      }
      
      const mapped = lines.map(line => {
        let escLine = esc(line);
        
        // Highlight search queries in orange
        if (query.trim() !== '') {
          const regex = new RegExp(`(${query})`, 'gi');
          escLine = escLine.replace(regex, '<span style="background:rgba(245, 158, 11, 0.35);color:#fff;border-radius:2px;padding:0 2px;">$1</span>');
        }
        
        // Custom syntax highlight standard python log tags
        return escLine
          .replace(/(running experiment|propos(al|e)| proposing)/gi, '<span style="color:var(--accent); font-weight:bold;">$1</span>')
          .replace(/(success|completed|canonical)/gi, '<span style="color:var(--success); font-weight:bold;">$1</span>')
          .replace(/(warning|warning:|failed to read|ignoring)/gi, '<span style="color:var(--warning); font-weight:bold;">$1</span>')
          .replace(/(error|exception|critical|crash)/gi, '<span style="color:var(--error); font-weight:bold;">$1</span>');
      }).join('\\n');
      
      consoleNode.innerHTML = mapped;
      consoleNode.scrollTop = consoleNode.scrollHeight;
    }

    // Initial launch
    loadDashboard().catch(console.error);
    setupPolling();
    
    // Auto-refresh when window resizes to ensure SVG is perfectly sized
    window.addEventListener('resize', () => {
      if (rawData) renderTrendChart(rawData.experiments.metric_trends);
    });
  </script>
</body>
</html>"""



def make_handler(project_root: Path, auth_token: str | None = None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                if not self._authorized(parsed):
                    self._send(401, "Unauthorized\n", "text/plain; charset=utf-8", {"WWW-Authenticate": "Bearer"})
                    return
                if parsed.path == "/":
                    self._send(200, HTML, "text/html; charset=utf-8")
                elif parsed.path == "/health":
                    self._json(200, {"ok": True, "project_root": str(project_root)})
                elif parsed.path == "/api/research":
                    self._json(200, build_snapshot(project_root))
                elif parsed.path == "/api/artifact":
                    path = parse_qs(parsed.query).get("path", [""])[0]
                    self._json(200, preview_artifact(project_root, path))
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, fmt, *args):
            return

        def _json(self, status: int, payload: dict):
            self._send(status, json.dumps(payload, sort_keys=True, default=str).encode(), "application/json")

        def _authorized(self, parsed) -> bool:
            if not auth_token:
                return True
            query_token = parse_qs(parsed.query).get("token", [""])[0]
            header = self.headers.get("Authorization", "")
            bearer = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
            return hmac.compare_digest(query_token, auth_token) or hmac.compare_digest(bearer, auth_token)

        def _send(self, status: int, body, content_type: str, extra_headers: dict[str, str] | None = None):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            if extra_headers:
                for key, value in extra_headers.items():
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the standalone Vesuvius AutoResearch dashboard")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--auth-token", default=os.getenv("VESUVIUS_DASHBOARD_TOKEN"), help="Require this bearer/query token for all dashboard routes")
    args = parser.parse_args(argv)
    root = resolve_project_root(args.repo_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root, args.auth_token))
    auth_note = " with token auth" if args.auth_token else " without auth"
    print(f"Vesuvius dashboard running at http://{args.host}:{args.port} for {root}{auth_note}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
