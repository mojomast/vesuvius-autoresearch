from __future__ import annotations

import argparse
import json
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
  <title>Vesuvius AutoResearch Dashboard</title>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --card:#141b2f; --muted:#94a3b8; --text:#e5e7eb; --line:#243047; --accent:#38bdf8; --good:#22c55e; --warn:#f59e0b; --bad:#ef4444; }
    body { margin:0; font-family: ui-sans-serif, system-ui, sans-serif; background:radial-gradient(circle at top left,#16213c,#0b1020 42rem); color:var(--text); }
    header { padding:1.2rem clamp(1rem,3vw,2rem); border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center; }
    main { padding:1rem clamp(1rem,3vw,2rem) 2rem; display:grid; gap:1rem; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem; }
    .card { background:rgba(20,27,47,.9); border:1px solid var(--line); border-radius:14px; padding:1rem; box-shadow:0 18px 40px rgba(0,0,0,.18); }
    h1,h2,h3 { margin:.1rem 0 .5rem; } h1 { font-size:1.4rem; } h2 { font-size:1rem; } p { color:var(--muted); line-height:1.4; }
    .pill { display:inline-flex; border:1px solid var(--line); border-radius:999px; padding:.18rem .5rem; color:var(--muted); font-size:.75rem; margin:.12rem; }
    .stat { font-size:1.6rem; font-weight:800; color:var(--accent); }
    table { width:100%; border-collapse:collapse; font-size:.82rem; } th,td { padding:.42rem .5rem; border-bottom:1px solid var(--line); text-align:left; } th { color:var(--muted); }
    pre { white-space:pre-wrap; overflow:auto; color:#cbd5e1; background:#080d1a; border:1px solid var(--line); border-radius:10px; padding:.65rem; font-size:.76rem; }
    button { background:#1e293b; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:.42rem .65rem; cursor:pointer; } button:hover { border-color:var(--accent); }
    .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
  </style>
</head>
<body>
  <header><div><h1>Vesuvius AutoResearch Dashboard</h1><p id="project"></p></div><button onclick="loadDashboard()">Refresh</button></header>
  <main id="app"><div class="card">Loading...</div></main>
  <script>
    const fmt = (v, d=4) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : 'n/a';
    const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    function commandCard(item) { return `<div class="card"><h3>${esc(item.title)}</h3><p>${esc(item.description)}</p><div>${(item.tags||[]).map(t=>`<span class="pill">${esc(t)}</span>`).join('')}</div><pre>${esc(item.command_text)}</pre></div>`; }
    function runRow(run) { const m=run.metrics||{}, s=run.validation_setup||{}; return `<tr><td><code>${esc(run.run_id)}</code></td><td>${esc(m.model_name || run.config?.model?.name || '')}</td><td>${fmt(m.val_f1)}</td><td>${fmt(m.average_precision)}</td><td>${fmt(m.pred_positive_rate,3)}</td><td>${esc(s.mode||'')}</td><td>${esc(s.train_segment_id||'?')} → ${esc(s.val_segment_id||'?')}</td></tr>`; }
    async function loadDashboard() {
      const data = await fetch('/api/research').then(r => r.json());
      document.getElementById('project').textContent = `${data.project.root} · ${data.schema_version}`;
      const score = data.progress.scorecard || {}, summary = data.progress.summary || {}, sanity = data.progress.sanity || {};
      const latest = data.experiments.latest || null;
      const features = data.inventory.features || [];
      const configs = data.configs || [];
      const prepared = data.datasets.prepared || [];
      const foldMaps = data.datasets.fold_maps || [];
      document.getElementById('app').innerHTML = `
        <section class="grid">
          <div class="card"><h2>Best F1</h2><div class="stat">${fmt(score.best)}</div><p>Best run ${esc(score.best_run_id || 'n/a')} · gain ${fmt(score.gain_vs_baseline)}</p></div>
          <div class="card"><h2>Latest F1</h2><div class="stat">${fmt(score.latest)}</div><p>${esc(summary.current_step || '')}</p></div>
          <div class="card"><h2>Validation Sanity</h2><div class="stat ${sanity.status === 'ok' ? 'good':'warn'}">${esc(sanity.status || 'unknown')}</div><p>${esc((sanity.warnings || []).join(' · ') || 'No warnings')}</p></div>
          <div class="card"><h2>Operations</h2><div class="stat ${data.operations.lock_active ? 'warn':'good'}">${data.operations.lock_active ? 'Running':'Idle'}</div><p>${(data.operations.processes||[]).length} live processes · run controls ${data.capabilities.enable_runs ? 'enabled':'disabled'}</p></div>
        </section>
        <section class="card"><h2>Recent Runs</h2><table><thead><tr><th>Run</th><th>Model</th><th>F1</th><th>AP</th><th>Pred+</th><th>Mode</th><th>Train → Val</th></tr></thead><tbody>${(data.experiments.recent||[]).slice(0,25).map(runRow).join('') || '<tr><td colspan="7">No runs</td></tr>'}</tbody></table></section>
        <section><h2>Feature Inventory</h2><div class="grid">${features.map(commandCard).join('')}</div></section>
        <section class="grid"><div class="card"><h2>Champion Configs</h2>${(data.inventory.champion_configs||[]).map(c=>`<span class="pill">${esc(c.name)} ${c.available?'':'missing'}</span>`).join('')}</div><div class="card"><h2>Configs</h2><p>${configs.length} config files</p>${configs.slice(0,10).map(c=>`<span class="pill">${esc(c.name)}</span>`).join('')}</div><div class="card"><h2>Prepared Data</h2><p>${prepared.length} metadata records</p>${prepared.slice(0,8).map(d=>`<span class="pill">${esc(d.metadata?.segment_id || d.path)}</span>`).join('')}</div><div class="card"><h2>Fold Maps</h2>${foldMaps.map(f=>`<p><code>${esc(f.path)}</code> · ${esc(f.folds)} folds</p>`).join('') || '<p>No fold maps discovered.</p>'}</div></section>
        <section class="card"><h2>AutoResearch Log Tail</h2><pre>${esc((data.operations.logs.lines||[]).join('\n') || 'No log lines.')}</pre></section>
      `;
    }
    loadDashboard().catch(err => { document.getElementById('app').innerHTML = `<div class="card bad">${esc(err)}</div>`; });
  </script>
</body>
</html>"""


def make_handler(project_root: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
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

        def _send(self, status: int, body, content_type: str):
            if isinstance(body, str):
                body = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the standalone Vesuvius AutoResearch dashboard")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    root = resolve_project_root(args.repo_root)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(root))
    print(f"Vesuvius dashboard running at http://{args.host}:{args.port} for {root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
