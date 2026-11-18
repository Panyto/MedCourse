"""金标标注页：只绑 127.0.0.1，凭口令访问，标注写进 labels.jsonl。页面不引任何外部资源，服务不记请求日志。

python -B -m gold.serve --dir 金标目录 [--port 8790]

口令写在 金标目录/url.txt（600），老板在自己电脑上开隧道后打开：
  ssh -N -L 8790:127.0.0.1:8790 ys      然后浏览器打开 url.txt 里的地址
ys 是多人共用的机器，本机端口别的用户也连得上，所以每个请求都要口令。同一题可以改，汇总取最后一次。

0.2（PREREG S2G v2）：类型和状态的名字、定义直接从 medcourse.extract.schema 读，和抽取提示词同一份（v1 的提示词把"体征"算进症状，
标注页只写"症状"，对不上）。说明里列出全部定义。不成立的题多问一句为什么（原文没有、不是这个病人、不是事件、类型错→应该是哪类），
只多一次点击，给下一版找原因用。状态的选项不含"不是这个病人"（那是不成立的一种）。
"""
import argparse
import hmac
import json
import os
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from medcourse.contract import CORE_TYPES
from medcourse.extract.schema import ASSERT_DEFS, TYPE_DEFS
from medcourse.safe import guarded, log

TYPES = {t: TYPE_DEFS[t][0] for t in CORE_TYPES}
TYPES_ALL = {t: v[0] for t, v in TYPE_DEFS.items()}
ASSERT = {a: v[0] for a, v in ASSERT_DEFS.items() if a != "not_patient"}
REASONS = {"absent": "原文没有这件事", "not_patient": "说的不是这个病人", "not_event": "不是事件（病史条目、护理、笼统说法）", "wrong_type": "类型错"}


def defs_html():
    """标注说明里的定义表，和抽取提示词同一份。"""
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731
    rows = "".join(f"<tr><td>{esc(cn)}</td><td>{esc(d)}</td></tr>" for t, (cn, d) in TYPE_DEFS.items())
    arows = "".join(f"<tr><td>{esc(cn)}</td><td>{esc(d)}</td></tr>" for a, (cn, d) in ASSERT_DEFS.items())
    return f"<p><b>类型</b>（和抽取提示词同一份定义；体征、七类以外只是给模型分流用，不出题）</p><table>{rows}</table><p><b>状态</b></p><table>{arows}</table>"

PAGE = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>S2 抽取金标</title><style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mute:#6b6b66;--line:#ddd9cf;--hl:#ffe8a3;--hl2:#e3ecff;--ok:#1f7a4d;--bad:#b3261e;--btn:#fff}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.7 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
header{display:flex;justify-content:space-between;align-items:baseline;gap:12px;flex-wrap:wrap}
h1{font-size:18px;margin:0}.mute{color:var(--mute);font-size:14px}
.box{background:#fff;border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:12px 0}
.text{white-space:pre-wrap;word-break:break-all}
mark{background:var(--hl);padding:0 1px}mark.b{background:var(--hl2)}
table{border-collapse:collapse;width:100%}td{border-bottom:1px solid var(--line);padding:4px 6px;vertical-align:top}td:first-child{color:var(--mute);width:6em}
.q{margin:10px 0}.q b{display:block;margin-bottom:4px}
button{font:inherit;padding:6px 14px;margin:2px 6px 2px 0;border:1px solid var(--line);border-radius:6px;background:var(--btn);cursor:pointer}
button.on{border-color:var(--fg);background:var(--fg);color:#fff}
.nav{display:flex;gap:8px;justify-content:space-between;align-items:center;margin-top:12px}
.grid{display:grid;grid-template-columns:7em 1fr;gap:4px 8px;align-items:center}
details{margin:8px 0}summary{cursor:pointer;color:var(--mute)}
.warn{color:var(--bad)}
</style></head><body><main>
<header><h1>S2 抽取金标</h1><span class="mute" id="prog"></span></header>
<details><summary>标注说明</summary><div class="box mute">
<p><b>A 部分（一条事件）</b>：黄色是模型给的原文依据。先判"成立"：原文里确实有这件事、说的是这个病人，而且类型对（按下面的定义）。不成立的选一个原因，类型错的再选应该是哪类。成立再判名称、数值（有数值才问）、状态。</p>
<p><b>B 部分（一段原文）</b>：浅蓝是这段里已经抽到的七类事件。按类型数一数<b>漏抽了几条</b>有临床意义的事件（同一件事写两遍算一条）。查体所见、护理这类不算七类，不用数。没有漏抽就全选 0。</p>
__DEFS__
<p>快捷键：← → 翻题；改了会自动保存。中途可以关掉，下次接着标。</p></div></details>
<div id="card"></div>
<div class="nav"><button id="prev">← 上一题</button><span class="mute" id="saved"></span><button id="next">下一题 →</button></div>
<p class="mute">只在 ys 本机运行；页面不连外网。</p>
</main><script>
const T=__TYPES__, A=__ASSERT__, TA=__TYPES_ALL__, R=__REASONS__;
let items=[], labels={}, cur=0;
async function api(path, body){const r=await fetch(path,{method:body?"POST":"GET",headers:{"Content-Type":"application/json"},body:body?JSON.stringify(body):undefined,credentials:"same-origin"});if(!r.ok)throw new Error(r.status);return r.json();}
function el(tag, attrs, ...kids){const e=document.createElement(tag);for(const[k,v]of Object.entries(attrs||{})){if(k==="cls")e.className=v;else if(k==="on")e.onclick=v;else e.setAttribute(k,v);}for(const c of kids){if(c==null)continue;e.append(c instanceof Node?c:document.createTextNode(String(c)));}return e;}
function marked(text, spans, cls){const out=el("div",{cls:"text"});const s=[...spans].sort((a,b)=>a[0]-b[0]);let p=0;for(const[a,b]of s){if(a<p)continue;out.append(text.slice(p,a));out.append(el("mark",cls?{cls}:{},text.slice(a,b)));p=b;}out.append(text.slice(p));return out;}
function choice(label, key, opts, lab, onchange){const q=el("div",{cls:"q"},el("b",{},label));for(const[v,t]of opts){q.append(el("button",{cls:lab[key]===v?"on":"",on:()=>{lab[key]=v;onchange();}},t));}return q;}
async function save(){const it=items[cur];const lab=labels[it.item_id];if(!lab)return;await api("/api/label",{item_id:it.item_id,labels:lab});document.getElementById("saved").textContent="已保存";render();}
function done(it){const l=labels[it.item_id];if(!l)return false;if(it.part==="B")return Object.keys(T).every(k=>l["miss_"+k]!==undefined);if(l.valid==="no")return !!l.reason&&(l.reason!=="wrong_type"||!!l.type_true);if(l.valid!=="yes"||!l.name||!l.assertion)return false;if(it.event.value!=null&&!l.value)return false;return l.assertion==="ok"||!!l.assertion_true;}
function render(){const it=items[cur];if(!it)return;const n=items.filter(done).length;document.getElementById("prog").textContent=`第 ${cur+1} / ${items.length} 题 · 已标完 ${n} 题 · ${it.part==="A"?"A 事件":"B 漏抽"}`;
const lab=labels[it.item_id]=labels[it.item_id]||{};const card=document.getElementById("card");card.replaceChildren();
card.append(el("div",{cls:"mute"},`${it.doc_type} · ${it.section||""}`));
if(it.part==="A"){const e=it.event;card.append(el("div",{cls:"box"},marked(it.text,it.hl)));
const t=el("table");for(const[k,v]of[["类型",T[e.event_type]||e.event_type],["名称",e.name],["数值",e.value==null?"（无）":`${e.value}${e.unit||""}`],["状态",A[e.assertion]||"（未判）"],["时间",e.time_text||"（无）"]])t.append(el("tr",{},el("td",{},k),el("td",{},v)));card.append(el("div",{cls:"box"},t));
const ch=()=>save();card.append(choice("这条事件成立吗（原文里有这件事、说的是这个病人，类型也对）","valid",[["yes","成立"],["no","不成立"]],lab,ch));
if(lab.valid==="no"){card.append(choice("为什么不成立","reason",Object.entries(R),lab,ch));
if(lab.reason==="wrong_type")card.append(choice("应该是哪类","type_true",Object.entries(TA).filter(([k])=>k!==e.event_type),lab,ch));}
if(lab.valid==="yes"){card.append(choice("名称对吗","name",[["ok","对"],["bad","错"]],lab,ch));
if(e.value!=null)card.append(choice("数值和单位对吗","value",[["ok","对"],["bad","错"]],lab,ch));
card.append(choice("状态对吗","assertion",[["ok","对"],["bad","错"]],lab,ch));
if(lab.assertion==="bad")card.append(choice("应该是","assertion_true",Object.entries(A).filter(([k])=>k!==e.assertion),lab,ch));}}
else{card.append(el("div",{cls:"box"},marked(it.text,it.hl,"b")));const lst=el("div",{cls:"box mute"},el("b",{},`这段已抽到 ${it.events.length} 条：`));
for(const e of it.events)lst.append(el("div",{},`${T[e.event_type]||e.event_type}｜${e.name}${e.value!=null?"｜"+e.value+(e.unit||""):""}｜${A[e.assertion]||"未判"}`));card.append(lst);
const g=el("div",{cls:"grid"});for(const[k,t]of Object.entries(T)){const row=el("div",{});for(const v of["0","1","2","3+"])row.append(el("button",{cls:lab["miss_"+k]===v?"on":"",on:()=>{lab["miss_"+k]=v;save();}},v));g.append(el("div",{},`漏抽${t}`),row);}
card.append(el("div",{cls:"q"},el("b",{},"这段里各类漏抽了几条"),g));
card.append(el("button",{on:()=>{for(const k of Object.keys(T))if(lab["miss_"+k]===undefined)lab["miss_"+k]="0";save();}},"其余都没漏（未选的记 0）"));}}
function go(d){cur=Math.max(0,Math.min(items.length-1,cur+d));document.getElementById("saved").textContent="";render();window.scrollTo(0,0);}
document.getElementById("prev").onclick=()=>go(-1);document.getElementById("next").onclick=()=>go(1);
document.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")go(-1);if(e.key==="ArrowRight")go(1);});
(async()=>{const d=await api("/api/state");items=d.items;labels=d.labels;cur=Math.max(0,items.findIndex(x=>!done(x)));if(cur<0)cur=0;render();})().catch(e=>{document.getElementById("card").replaceChildren(el("p",{cls:"warn"},"读不到题目："+e.message));});
</script></body></html>"""


def load_labels(path):
    out = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    x = json.loads(line)
                    out[x["item_id"]] = x["labels"]
    return out


def make_handler(gold_dir, token):
    with open(os.path.join(gold_dir, "items.jsonl"), encoding="utf-8") as f:
        items = [json.loads(x) for x in f if x.strip()]
    ids = {x["item_id"] for x in items}
    labels_path = os.path.join(gold_dir, "labels.jsonl")
    lock = threading.Lock()
    page = (PAGE.replace("__TYPES_ALL__", json.dumps(TYPES_ALL, ensure_ascii=False)).replace("__TYPES__", json.dumps(TYPES, ensure_ascii=False))
            .replace("__ASSERT__", json.dumps(ASSERT, ensure_ascii=False)).replace("__REASONS__", json.dumps(REASONS, ensure_ascii=False))
            .replace("__DEFS__", defs_html())).encode("utf-8")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):    # 不记请求日志
            pass

        def _authed(self):
            q = parse_qs(urlparse(self.path).query).get("t", [""])[0]
            cookie = dict(x.strip().split("=", 1) for x in (self.headers.get("Cookie") or "").split(";") if "=" in x).get("gold_t", "")
            return hmac.compare_digest(q.encode(), token.encode()) or hmac.compare_digest(cookie.encode(), token.encode())

        def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if not self._authed():
                return self._send(403, b'{"error":"token"}')
            path = urlparse(self.path).path
            if path == "/":
                return self._send(200, page, "text/html; charset=utf-8", {"Set-Cookie": f"gold_t={token}; HttpOnly; SameSite=Strict; Path=/"})
            if path == "/api/state":
                with lock:
                    labels = load_labels(labels_path)
                return self._send(200, json.dumps({"items": items, "labels": labels}, ensure_ascii=False).encode("utf-8"))
            return self._send(404, b'{"error":"path"}')

        def do_POST(self):
            if not self._authed() or urlparse(self.path).path != "/api/label":
                return self._send(403, b'{"error":"token"}')
            n = int(self.headers.get("Content-Length") or 0)
            if n > 20000:
                return self._send(413, b'{"error":"size"}')
            try:
                x = json.loads(self.rfile.read(n))
                assert x["item_id"] in ids and isinstance(x["labels"], dict)
                lab = {str(k)[:32]: str(v)[:16] for k, v in x["labels"].items()}    # 只收短标签，不收自由文本
            except Exception:  # noqa: BLE001
                return self._send(400, b'{"error":"bad"}')
            with lock, open(labels_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"item_id": x["item_id"], "ts": round(time.time()), "labels": lab}, ensure_ascii=False) + "\n")
            return self._send(200, b'{"ok":true}')

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    os.umask(0o077)
    token = secrets.token_hex(16)
    with open(os.path.join(a.dir, "url.txt"), "w", encoding="utf-8") as f:
        f.write(f"http://127.0.0.1:{a.port}/?t={token}\n")
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(a.dir, token))
    log("gold_serve_ready", port=a.port, url_file=os.path.join(a.dir, "url.txt"))
    srv.serve_forever()


if __name__ == "__main__":
    guarded(main)
