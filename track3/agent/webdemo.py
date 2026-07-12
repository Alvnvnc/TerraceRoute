"""A local web console for TerraceGate — the demo you self-host.

Run it next to your local Ollama (the two model families live on your AMD GPU),
then expose it with TerraceGate itself:

    python3 -m agent.webdemo --port 8787          # serve the console locally
    python3 -m agent.cli expose --host amd.<zone> --port 8787   # tunnel it out

The page is **plan + gate only**: you type a request, both local models plan it
independently, and the gate shows its verdict. It deliberately has **no execute
path**, so it is safe to expose publicly — a visitor can never drive a real
change, and the Cloudflare token never leaves the machine. A single-inference
lock keeps the GPU from being thrashed by concurrent requests.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .brain.planner import DEFAULT_PLANNER, DEFAULT_VERIFIER, dual_plan
from .types import BlastRadius, GateDecision

# One inference at a time — protects the GPU when the page is public.
_GPU_LOCK = threading.Lock()
_VLM_TOKEN = os.environ.get("VLM_PROXY_TOKEN", "")
_VLM_MODELS = {"gemma3:12b"}
_VLM_MAX_BODY = 12 * 1024 * 1024

_DECISION_META = {
    GateDecision.AUTO_APPLY: ("AUTO-APPLY", "#17915a", "the two models agree and the blast radius is safe"),
    GateDecision.CONFIRM: ("CONFIRM", "#c8890a", "hold for one human confirmation before applying"),
    GateDecision.CONFIRM_PER_ITEM: ("CONFIRM EACH", "#c8890a", "confirm every item — destructive but explicit"),
    GateDecision.REFUSE: ("REFUSE", "#1211CA", "refuse: risky and the two model families diverge"),
}

_EXAMPLES = [
    "expose my grafana on stats.alvnvnc.site port 3000",
    "hapus semua DNS yang kelihatannya tidak dipakai",
    "bikin git.alvnvnc.site online port 3000 terus hapus semua yang lain",
    "is my media tunnel healthy?",
]


def _here(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), name)


def _fonts() -> str:
    try:
        with open(_here("console_fonts.css"), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _plan_view(res) -> dict:
    p = res.plan
    return {
        "model": res.model,
        "op": (p.op if p else "—"),
        "hostname": (p.hostname if p else ""),
        "port": (p.port if p else 0),
        "scheme": (p.service_scheme if p else ""),
        "reasoning": (p.reasoning if p else ""),
        "tok_s": round(res.tokens_per_s, 1),
    }


def analyze(text: str) -> dict:
    dp = dual_plan(text)
    g = dp.gate
    label, color, blurb = _DECISION_META.get(g.decision, ("—", "#101010", ""))
    return {
        "text": text,
        "planner": _plan_view(dp.planner),
        "verifier": _plan_view(dp.verifier),
        "gate": {
            "decision": g.decision.value,
            "label": label,
            "color": color,
            "blurb": blurb,
            "blast_radius": BlastRadius(g.blast_radius).name.replace("_", "-").lower(),
            "disagreement": round(g.disagreement, 2),
            "rationale": g.rationale,
            "conflicts": list(g.conflicts),
        },
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "TerraceGate/1.0"

    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path in ("/healthz", "/ready"):
            self._send(200, "ok", "text/plain")
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path == "/v1/chat/completions":
            self._proxy_vlm()
            return
        if path != "/api/plan":
            self._send(404, json.dumps({"error": "not found"}))
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            text = (payload.get("text") or "").strip()
        except (ValueError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "bad request"}))
            return
        if not text:
            self._send(400, json.dumps({"error": "empty request"}))
            return
        if len(text) > 500:
            self._send(400, json.dumps({"error": "request too long"}))
            return
        if not _GPU_LOCK.acquire(blocking=False):
            self._send(429, json.dumps({"busy": True,
                       "error": "the GPU is busy with another request — try again in a moment"}))
            return
        try:
            self._send(200, json.dumps(analyze(text)))
        except Exception as e:  # never leak a stack trace to the public page
            self._send(500, json.dumps({"error": f"inference failed: {type(e).__name__}"}))
        finally:
            _GPU_LOCK.release()

    def _proxy_vlm(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query)
        supplied = query.get("token", [""])[0]
        if not _VLM_TOKEN or not hmac.compare_digest(supplied, _VLM_TOKEN):
            self._send(401, json.dumps({"error": "unauthorized"}))
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            length = 0
        if length <= 0 or length > _VLM_MAX_BODY:
            self._send(413, json.dumps({"error": "invalid request size"}))
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (ValueError, json.JSONDecodeError):
            self._send(400, json.dumps({"error": "bad request"}))
            return
        if payload.get("model") not in _VLM_MODELS:
            self._send(403, json.dumps({"error": "model not allowed"}))
            return
        payload["stream"] = False
        payload["max_tokens"] = min(max(int(payload.get("max_tokens", 400)), 1), 800)
        request = urllib.request.Request(
            "http://127.0.0.1:11434/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                self._send(response.status, response.read())
        except urllib.error.HTTPError as exc:
            self._send(exc.code, exc.read() or json.dumps({"error": "upstream error"}))
        except (OSError, urllib.error.URLError, TimeoutError):
            self._send(502, json.dumps({"error": "upstream unavailable"}))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TerraceGate · live console</title>
<style>
__FONTS__
:root{--blue:#1211CA;--amber:#F9B314;--ink:#101010;--ink2:#2D262A;--muted:#77838D;
 --line:#E6E8EE;--disp:'Montserrat',system-ui,sans-serif;--body:'Poppins',system-ui,sans-serif;
 --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:var(--body);color:var(--ink2);background:#fff;-webkit-font-smoothing:antialiased;line-height:1.5}
.wrap{max-width:1080px;margin:0 auto;padding:40px 26px 64px}
.brand{font-family:var(--body);font-weight:600;font-size:22px;color:var(--ink)}
.brand::after{content:"";display:block;width:150px;height:5px;background:var(--amber);margin-top:10px;border-radius:2px}
.top{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:14px}
.env{font-family:var(--mono);font-size:12.5px;color:var(--muted);text-align:right;line-height:1.7}
.env b{color:var(--blue)}
h1{font-family:var(--disp);font-weight:800;font-size:40px;line-height:1.05;letter-spacing:-.5px;margin:34px 0 6px}
h1 .b{color:var(--amber)}
.sub{font-size:18px;color:var(--ink2);max-width:720px}
.inrow{display:flex;gap:12px;margin-top:26px}
#q{flex:1;font-family:var(--body);font-size:17px;padding:15px 18px;border:1.6px solid var(--line);
 border-radius:12px;outline:none}
#q:focus{border-color:var(--blue)}
button.go{font-family:var(--body);font-weight:600;font-size:16px;color:#fff;background:var(--blue);
 border:none;border-radius:12px;padding:0 26px;cursor:pointer}
button.go:disabled{opacity:.5;cursor:default}
.chips{display:flex;gap:9px;flex-wrap:wrap;margin-top:14px}
.chip{font-size:13.5px;color:var(--ink2);border:1.4px solid var(--line);border-radius:999px;
 padding:7px 14px;cursor:pointer;background:#fff}
.chip:hover{border-color:var(--blue);color:var(--blue)}
#out{margin-top:30px}
.verdict{border-radius:16px;padding:22px 24px;color:#fff;display:flex;align-items:center;gap:22px;flex-wrap:wrap}
.verdict .lab{font-family:var(--disp);font-weight:800;font-size:30px;letter-spacing:.5px}
.verdict .meta{font-size:14px;opacity:.92;font-family:var(--mono)}
.verdict .blurb{font-size:15px;opacity:.95;max-width:520px}
.cards{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:18px}
.card{border:1.5px solid var(--line);border-radius:14px;padding:18px 20px}
.card .role{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--muted);font-weight:600}
.card .model{font-family:var(--mono);font-size:13px;color:var(--muted);margin-top:2px}
.card .op{font-family:var(--disp);font-weight:800;font-size:26px;margin-top:10px}
.card .args{font-family:var(--mono);font-size:14px;color:var(--ink2);margin-top:4px}
.card .why{font-size:13.5px;color:var(--muted);margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
.card .tps{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:8px}
.op-expose{color:#17915a}.op-unexpose{color:#c0392b}.op-status,.op-diagnose{color:var(--blue)}.op-heal{color:#c8890a}
.reasons{margin-top:16px;border-left:3px solid var(--amber);padding:6px 0 6px 16px}
.reasons .h{font-weight:600;font-size:14px;color:var(--ink)}
.reasons li{font-size:14px;color:var(--ink2);margin-top:4px;list-style:none}
.reasons li::before{content:"– ";color:var(--muted)}
.note{margin-top:34px;font-size:13px;color:var(--muted);font-family:var(--mono);border-top:1px solid var(--line);padding-top:16px}
.note b{color:var(--blue)}
.spin{display:inline-block;width:16px;height:16px;border:2.5px solid rgba(255,255,255,.4);
 border-top-color:#fff;border-radius:50%;animation:s .7s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
.err{background:#fff4f2;border:1.5px solid #f0b3a8;color:#a5352a;border-radius:12px;padding:14px 18px;font-size:15px}
@media(max-width:680px){.cards{grid-template-columns:1fr}h1{font-size:31px}}
</style></head><body><div class="wrap">
<div class="top">
  <div class="brand">TerraceGate</div>
  <div class="env">live on your <b>AMD Radeon</b> · gemma3:12b + qwen2.5:3b<br>
    two local families · the Cloudflare token never left this machine</div>
</div>
<h1>Type a request. Two local models plan it.<br><span class="b">The gate decides.</span></h1>
<p class="sub">Natural language in — planner and verifier run independently on the GPU, and the
  safety gate reads their <b>disagreement</b> (plus a deterministic intent guard). This console
  only plans &amp; gates; it never executes a change.</p>
<div class="inrow">
  <input id="q" placeholder="expose my grafana on stats.alvnvnc.site port 3000" autocomplete="off">
  <button class="go" id="go">Plan</button>
</div>
<div class="chips" id="chips"></div>
<div id="out"></div>
<div class="note">Plan &amp; gate only — <b>no execute path</b> on this public page. Reasoning runs
  100% locally on AMD ROCm; nothing is sent to a cloud LLM.</div>
</div>
<script>
var EX=__EXAMPLES__;
var chips=document.getElementById('chips');
EX.forEach(function(t){var c=document.createElement('div');c.className='chip';c.textContent=t;
  c.onclick=function(){document.getElementById('q').value=t;run();};chips.appendChild(c);});
var q=document.getElementById('q'),go=document.getElementById('go'),out=document.getElementById('out');
q.addEventListener('keydown',function(e){if(e.key==='Enter')run();});
go.onclick=run;
function esc(s){return (s||'').replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function opcls(op){return 'op op-'+esc(op);}
function card(role,d){
  var args=d.hostname?(esc(d.hostname)+' :'+d.port+'/'+esc(d.scheme)):'—';
  return '<div class="card"><div class="role">'+role+'</div><div class="model">'+esc(d.model)+'</div>'+
    '<div class="'+opcls(d.op)+'">'+esc(d.op)+'</div><div class="args">'+args+'</div>'+
    (d.reasoning?'<div class="why">'+esc(d.reasoning)+'</div>':'')+
    '<div class="tps">'+d.tok_s+' tok/s</div></div>';
}
function run(){
  var text=q.value.trim(); if(!text)return;
  go.disabled=true; go.innerHTML='<span class="spin"></span>';
  out.innerHTML='<div class="card" style="text-align:center;color:#77838d">two models thinking on the GPU…</div>';
  fetch('/api/plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})})
  .then(function(r){return r.json().then(function(j){return {ok:r.ok,j:j};});})
  .then(function(res){
    go.disabled=false; go.textContent='Plan';
    if(!res.ok){out.innerHTML='<div class="err">'+esc(res.j.error||'error')+'</div>';return;}
    var j=res.j,g=j.gate;
    var reasons=(g.conflicts&&g.conflicts.length)?
      '<div class="reasons"><div class="h">why</div><ul>'+g.conflicts.map(function(c){return '<li>'+esc(c)+'</li>';}).join('')+'</ul></div>':'';
    out.innerHTML=
      '<div class="verdict" style="background:'+g.color+'">'+
        '<div class="lab">'+esc(g.label)+'</div>'+
        '<div><div class="meta">blast radius: '+esc(g.blast_radius)+' · disagreement: '+g.disagreement+'</div>'+
        '<div class="blurb">'+esc(g.blurb)+'</div></div></div>'+
      '<div class="cards">'+card('Planner',j.planner)+card('Verifier',j.verifier)+'</div>'+reasons;
  })
  .catch(function(){go.disabled=false;go.textContent='Plan';
    out.innerHTML='<div class="err">network error</div>';});
}
</script></body></html>"""

PAGE = PAGE.replace("__FONTS__", _fonts()).replace("__EXAMPLES__", json.dumps(_EXAMPLES))


def main() -> None:
    ap = argparse.ArgumentParser(description="TerraceGate live console (plan + gate only).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"TerraceGate console on http://{args.host}:{args.port}  "
          f"(planner={DEFAULT_PLANNER}, verifier={DEFAULT_VERIFIER})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()


if __name__ == "__main__":
    main()
