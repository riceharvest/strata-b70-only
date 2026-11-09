// serve/web/app.js - the Strata web app: Chat, Monitor, About. No framework, no network beyond this server.
// The Monitor tab rebuilds PR #22's dashboard idea (code-martin) on the server's own /metrics.
"use strict";

const $ = (id) => document.getElementById(id);
const SPRITE = "web/sprite.svg";
const icon = (name, cls = "st-icon") => `<svg class="${cls}" aria-hidden="true"><use href="${SPRITE}#i-${name}"/></svg>`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (n, d = 0) => (n == null || Number.isNaN(n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const kfmt = (n) => (n == null ? "–" : n >= 1000 ? `${fmt(n / 1000, n >= 10000 ? 0 : 1)}k` : fmt(n));
// a context size: 32768 -> "32K" (powers of two), else like kfmt
const ctxfmt = (n) => (n && n % 1024 === 0 ? `${fmt(n / 1024)}K` : kfmt(n));
const gb = (b, d = 1) => (b == null ? "–" : fmt(b / 1073741824, d));   // memory: binary GB, as Windows shows it

// Cloud baseline: Alibaba Cloud Int. rates for Qwen-Flash
const QWEN_FLASH_PRICING = {
  provider: "Alibaba Cloud Int.",
  model: "Qwen-Flash",
  inputPerM: 0.15,
  outputPerM: 0.47,
  cacheReadPerM: 0.016,
  cacheCreate5mPerM: 0.20,
  cacheRead5mPerM: 0.016,
};
function calcCost(promptTokens = 0, reusedTokens = 0, outputTokens = 0) {
  const p = Number(promptTokens) || 0;
  const r = Math.min(p, Math.max(0, Number(reusedTokens) || 0));
  const f = Math.max(0, p - r);
  const o = Math.max(0, Number(outputTokens) || 0);
  return (f * QWEN_FLASH_PRICING.inputPerM + r * QWEN_FLASH_PRICING.cacheReadPerM + o * QWEN_FLASH_PRICING.outputPerM) / 1e6;
}
function fmtCost(n, precision) {
  if (n == null || Number.isNaN(n)) return "–";
  const v = Number(n);
  if (v === 0) return "$0.00";
  if (v >= 1e6) return `$${fmt(v / 1e6, 2)}M`;
  if (v >= 1e4) return `$${fmt(v / 1e3, 1)}k`;
  if (v >= 100) return `$${v.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
  if (v >= 1) return `$${v.toFixed(precision != null ? precision : 2)}`;
  if (v >= 0.01) return `$${v.toFixed(precision != null ? precision : 2)}`;
  if (v >= 0.001) return `$${v.toFixed(precision != null ? precision : 3)}`;
  if (v >= 0.0001) return `$${v.toFixed(precision != null ? precision : 4)}`;
  return `$${v.toFixed(precision != null ? precision : 4)}`;
}

const store = {
  get(k, d) { try { const v = localStorage.getItem("strata." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("strata." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

// ------------------------------------------------------------------ toasts
function toast(kind, title, text = "", ms = 3500, action = null) {
  const names = {info: "info", success: "check", warn: "warning", error: "error"};
  const el = document.createElement("div");
  el.className = `st-toast st-toast--${kind}`;
  el.innerHTML = `${icon(names[kind] || "info")}<div><div class="st-toast__title"></div><div class="t-text"></div></div>`;
  el.querySelector(".st-toast__title").textContent = title;
  el.querySelector(".t-text").textContent = text;
  if (action) {
    const b = document.createElement("button");
    b.className = "st-btn st-btn--secondary";
    b.style.height = "32px";
    b.style.marginLeft = "auto";
    b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.appendChild(b);
  }
  $("toasts").appendChild(el);
  setTimeout(() => el.remove(), ms);
}

async function copyText(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {                                   // http on another host: no async clipboard
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  if (btn) {
    const use = btn.querySelector("use");
    use.setAttribute("href", `${SPRITE}#i-check`);
    setTimeout(() => use.setAttribute("href", `${SPRITE}#i-copy`), 1500);
  }
  toast("success", "Copied to clipboard", "", 1800);
}

// ------------------------------------------------------------------ theme and tabs
// the system's theme until the user picks one (only a click is saved)
function setTheme(t, save) {
  document.documentElement.dataset.theme = t;
  if (save) try { localStorage.setItem("strata.theme", t); } catch (e) { /* ignore */ }
  $("theme-icon").setAttribute("href", `${SPRITE}#i-${t === "dark" ? "sun" : "moon"}`);
  $("dark-toggle").setAttribute("aria-checked", String(t === "dark"));
}
const flipTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
$("theme-btn").onclick = flipTheme;
$("dark-toggle").onclick = flipTheme;
setTheme(document.documentElement.dataset.theme || "light", false);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
  let saved = null;
  try { saved = localStorage.getItem("strata.theme"); } catch (err) { /* ignore */ }
  if (!saved) setTheme(e.matches ? "dark" : "light", false);
});

const TABS = ["chat", "images", "monitor", "requests", "about"];
const TITLES = {chat: ["Chat", ""], images: ["Images", ""], monitor: ["Dashboard", ""], requests: ["Requests", "every request the server keeps"],
                about: ["Model and settings", ""]};
let tab = "chat";
function showTab(name) {
  tab = TABS.includes(name) ? name : "chat";
  for (const b of document.querySelectorAll(".st-tab")) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
  for (const v of TABS) $(`view-${v}`).hidden = v !== tab;
  $("page-title").textContent = TITLES[tab][0];
  $("page-sub").textContent = TITLES[tab][1];
  document.body.dataset.tab = tab;
  if (tab === "images") $("img-prompt").focus();
  if (location.hash.slice(1) !== tab) history.replaceState(null, "", tab === "chat" ? location.pathname : `#${tab}`);
  if (tab === "chat") $("input").focus();
  if (tab === "monitor") loadMcp();
  if (tab === "about") loadConfig();
  if (wantsAll()) allReqsAt = 0;                      // the next poll reads every kept request
  if (lastMetrics) render(lastMetrics);
}
for (const b of document.querySelectorAll(".st-tab")) b.onclick = () => showTab(b.dataset.tab);
window.addEventListener("hashchange", () => showTab(location.hash.slice(1)));

// ------------------------------------------------------------------ server access
let apiKey = store.get("apikey", "");
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}
$("api-key").value = apiKey;
function saveApiKey(notify = false) {
  apiKey = $("api-key").value.trim();                // usable even when browser storage is blocked or full
  const saved = store.set("apikey", apiKey);
  if (notify) {
    if (saved) toast("success", "API key saved", "Kept in this browser only.");
    else toast("warn", "API key used for this page only", "Browser storage is unavailable. Enter it again after reloading.");
  }
}
$("api-key").oninput = () => saveApiKey();
$("api-key").onchange = () => saveApiKey(true);

let health = {model: "strata", images: false, max_context: 0};
async function loadHealth() {
  try {
    health = await (await fetch("health")).json();
    $("attach-btn").title = health.images ? "Attach a text file or a picture (or drop it here)"
                                          : "Attach a text file (or drop it here)";
    $("chat-empty-sub").textContent = `${health.model} runs on this PC. Nothing leaves it.`;
    $("tab-btn-images").hidden = !health.image_generation;
    if (location.hash === "#images" && health.image_generation) showTab("images");
  } catch (e) {
    setTimeout(loadHealth, 2000);
  }
}

// ------------------------------------------------------------------ Dashboard
// Hardware tiles: a number and a sparkline each (the server keeps a minute of history per series)
const METRICS = [
  {key: "gpu", label: "GPU load", icon: "gpu", unit: "%", series: "gpu_util", max: 100},
  {key: "vram", label: "VRAM", icon: "layers", unit: "GB", series: "gpu_mem_used"},
  {key: "temp", label: "GPU temp", icon: "thermometer", unit: "°C", series: "gpu_temp", tone: "warn"},
  {key: "power", label: "Power", icon: "bolt", unit: "W", series: "gpu_power"},
  {key: "pcie", label: "PCIe", icon: "link", unit: "", series: "gpu_pcie_rx_mb", tone: "info"},
  {key: "cpu", label: "CPU", icon: "cpu", unit: "%", series: "cpu", max: 100},
  {key: "disk", label: "Disk read", icon: "disk", unit: "MB/s", series: "disk_read_mb", tone: "info"},
];
$("metrics").innerHTML = METRICS.map((m) => `
  <div class="hwtile"><span class="hwtile__label">${icon(m.icon, "st-icon st-icon--sm")}${esc(m.label)}</span>
    <span class="hwtile__value" id="mv-${m.key}">–</span><span class="hwtile__sub" id="ms-${m.key}"></span>
    <svg class="spark" id="sp-${m.key}" viewBox="0 0 100 32" preserveAspectRatio="none"${m.tone ? ` data-tone="${m.tone}"` : ""}>
      <path class="area" fill="currentColor" opacity=".12"/><path class="line" fill="none" stroke="currentColor"
      stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg></div>`).join("");

function sparkPath(v, max, w = 100, h = 32, pad = 3) {
  v = (v || []).map((x) => (x == null || Number.isNaN(x) ? 0 : x));
  if (v.length < 2) return ["", ""];
  const top = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => [(i / (v.length - 1)) * w, h - pad - (x / top) * (h - 2 * pad)]);
  const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join("");
  return [line, `${line}L${w},${h}L0,${h}Z`];
}
function spark(id, values, max) {
  const svg = $(id);
  const [line, area] = sparkPath(values, max);
  svg.querySelector(".line").setAttribute("d", line);
  svg.querySelector(".area").setAttribute("d", area);
}
function setMetric(key, value, unit, sub) {
  $(`mv-${key}`).closest(".hwtile").hidden = value == null && !sub;    // a reading this PC cannot give is not shown as a dash
  $(`mv-${key}`).innerHTML = value == null ? "–" : `${esc(value)}${unit ? `<small>${esc(unit)}</small>` : ""}`;
  $(`ms-${key}`).textContent = sub || "";
}

let lastMetrics = null, metricsFailures = 0, keyWarned = false, mcpTick = 0;
let allReqs = [], allReqsAt = 0, polling = false;      // every request the server keeps, fetched at most every 5 s
const wantsAll = () => tab === "monitor" || tab === "requests";
// After Load or Unload returns, a poll that started earlier can still say the opposite. Hold the POST's state until a later
// poll agrees, or until the first poll that started after the POST arrives. (PR #3550 by sakuumomo)
let metricsEpoch = 0, metricsHold = null;
function metricsAgree(live, state) {
  const s = (live && live.state) || "";
  if (state === "unloaded") return s === "unloaded";
  return s !== "" && s !== "unloaded";
}
async function poll() {
  if (polling) return;
  polling = true;
  const epoch = ++metricsEpoch;
  try {
    const full = wantsAll() && Date.now() - allReqsAt > 5000;
    const r = await fetch(full ? "metrics?requests=all" : "metrics", {headers: headers()});
    if (r.status === 401) {
      setPill("error", "API key needed");
      if (!keyWarned) { keyWarned = true; toast("warn", "API key needed", "This server needs a key: add it under Model and settings.", 6000); }
    } else if (r.ok) {
      const m = await r.json();
      if (metricsHold && epoch <= metricsHold.after && !metricsAgree(m.live, metricsHold.state)) {
        polling = false;
        setTimeout(poll, 1000);
        return;
      }
      if (metricsHold && (metricsAgree(m.live, metricsHold.state) || epoch > metricsHold.after)) metricsHold = null;
      lastMetrics = m;
      if (full) { allReqs = lastMetrics.requests || []; allReqsAt = Date.now(); }
      else if (allReqs.length) {                         // the newest 12 are always fresh; the rest comes from the last full read
        const ids = new Set((lastMetrics.requests || []).map((x) => x.id));
        allReqs = (lastMetrics.requests || []).concat(allReqs.filter((x) => !ids.has(x.id))).slice(0, 500);
      }
      metricsFailures = 0;
      render(lastMetrics);
    } else {
      throw new Error(`HTTP ${r.status}`);
    }
  } catch (e) {
    if (++metricsFailures === 3) setPill("error", "Server not reachable");
  }
  polling = false;
  if (tab === "monitor" && ++mcpTick % 10 === 0) loadMcp();       // server states change rarely: every 10 s
  setTimeout(poll, 1000);
}

function setPill(state, text) {
  $("pill").dataset.state = state === "error" ? "queued" : state;
  $("pill-text").textContent = text;
  // a poll re-sets the sentence every second; refit only when its text changed
  if (text !== lastPillText) { lastPillText = text; fitHeader(); }
}
let lastPillText = null;
// The pill's sentence is the first thing the top bar can spare when it runs out of room.
function fitHeader() {
  const bar = document.querySelector(".topbar");
  if (!bar) return;
  bar.classList.remove("topbar--clip");
  if (bar.scrollWidth > bar.clientWidth) bar.classList.add("topbar--clip");
}
window.addEventListener("resize", fitHeader);

// ------------------------------------------------------------------ Load / Unload (POST /load, /unload: the top bar button)
// Metrics stay "unloaded" until the engine is READY, so a turn or an image job on this page has to paint Loading itself.
// modelOp covers the button's own POST, which returns before the next poll can agree. (PR #3550 by sakuumomo)
let modelOp = null;
let imageBusy = false;
function pageBusy() { return busy != null || imageBusy; }
function imageLive(live) {
  if (live && live.phase === "generating an image") return true;
  const state = live && live.state;
  if (state === "reading" || state === "generating") return false;
  return imageBusy;
}
function paintLive() { if (lastMetrics) render(lastMetrics); }
function turnBlocksUnload(live) {
  const state = (live && live.state) || "";
  return pageBusy() || state === "reading" || state === "generating" || (live && live.queued > 0);
}
function setModelButton(text, title, disabled) {
  const btn = $("unload-btn");
  btn.textContent = text; btn.title = title; btn.disabled = disabled;
  fitHeader();
}
function modelButton(live) {
  if (modelOp === "load") return setModelButton("Loading\u2026", "The model is loading", true);
  if (modelOp === "unload") return setModelButton("Unloading\u2026", "The model is unloading", true);
  const state = (live && live.state) || "";
  const metricsBusy = state === "reading" || state === "generating" || (live && live.queued > 0);
  const imageNow = imageLive(live);
  if (state === "unloaded") {
    if (imageNow) return setModelButton("Loading\u2026", "An image is already loading the model", true);
    if (pageBusy()) return setModelButton("Loading\u2026", "A reply is already loading the model", true);
    return setModelButton("Load", "Load the model now. A message also loads it.", false);
  }
  if (imageNow) return setModelButton("Unload", "An image is generating. It finishes on its own.", true);
  if (!state) return setModelButton("\u2026", "Waiting for the server", true);
  if (metricsBusy || pageBusy()) return setModelButton("Unload", "Stop the reply, then unload", true);
  return setModelButton("Unload", "Unload the model and free the GPU", false);
}
const busyCopy = (sample, stillLoading) => stillLoading
  ? (imageLive(sample) ? "An image is already loading the model." : "A reply is already loading the model.")
  : (imageLive(sample) ? "An image is generating. It finishes on its own." : "Stop the reply, then unload.");
$("unload-btn").onclick = async () => {
  if (modelOp) return;
  const live = (lastMetrics && lastMetrics.live) || {};
  const loading = live.state === "unloaded";
  if (loading ? pageBusy() : turnBlocksUnload(live)) { toast("warn", "Model is busy", busyCopy(live, loading)); return; }
  if (!loading && !confirm("Unload the model and free the GPU? Load, or the next message, brings it back.")) return;
  // confirm() blocks the poll. Yield so a sample that arrived during the dialog is applied. A confirmed Unload stays an unload.
  if (!loading) {
    modelOp = "unload";
    modelButton(live);
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  const now = (lastMetrics && lastMetrics.live) || {};
  const loadingNow = loading;
  if (loadingNow ? pageBusy() : turnBlocksUnload(now)) {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton(now);
    toast("warn", "Model is busy", busyCopy(now, loadingNow));
    return;
  }
  modelOp = loadingNow ? "load" : "unload";
  modelButton(now);
  if (loadingNow) toast("info", "Loading the model", "This can take a few minutes.");
  const remember = (state) => {
    if (!lastMetrics) lastMetrics = {};
    lastMetrics.live = Object.assign({}, lastMetrics.live, {state, queued: 0});
    metricsHold = {state, after: metricsEpoch};
  };
  const failTitle = loadingNow ? "Load failed" : "Unload failed";
  try {
    const r = await fetch(loadingNow ? "load" : "unload", {method: "POST", headers: headers(true), body: "{}"});
    let body = {};
    try { body = await r.json(); } catch (e) { /* the connection dropped */ }
    const status = body.status || "";
    const msg = (body.error && body.error.message) ||
                (r.status === 401 ? "API key needed: add it under Model and settings." : `The server answered HTTP ${r.status}.`);
    if (loadingNow) {
      if (r.ok && status === "loaded") { remember("idle"); toast("success", "Model loaded", "The GPU is in use."); }
      else toast("error", failTitle, msg, 8000);
    } else if (status === "unloaded" || status === "not loaded") {
      remember("unloaded");
      toast(status === "unloaded" ? "success" : "info", status === "unloaded" ? "Model unloaded" : "Already unloaded",
            "The GPU is free. Load, or the next message, brings it back.");
    } else if (r.status === 409 || status === "busy") {
      toast("warn", "Model is busy", busyCopy(live, false));
    } else if (status === "unsupported") {
      toast("error", "Unload is not available", "");
    } else {
      toast("error", failTitle, msg, 8000);
    }
  } catch (e) {
    toast("error", failTitle, "The server did not answer.");
  } finally {
    modelOp = null;
    if (lastMetrics) render(lastMetrics); else modelButton({});
  }
};

function render(m) {
  const live = m.live || {}, hw = m.hardware || {}, st = m.hardware_static || {}, eng = m.engine || {}, h = m.history || {};
  const reqs = wantsAll() && allReqs.length ? allReqs : (m.requests || []);
  const last = (m.requests || [])[0];
  // the header pill. modelOp and a local load win over "N queued": queued stays up through ensure_loaded(), and
  // metrics stay "unloaded" until the engine is READY. An image POST in that window is still a load.
  const localLoad = live.state === "unloaded" && pageBusy();
  if (modelOp === "load" || localLoad) {
    setPill("idle", "Loading\u2026");
  } else if (modelOp === "unload") {
    setPill("idle", "Unloading\u2026");
  } else if (imageLive(live)) {
    setPill("reading", "Generating image\u2026");
  } else if (live.state === "reading") {
    const pct = live.prompt_total ? Math.round((100 * live.prompt_read) / live.prompt_total) : null;
    setPill("reading", pct != null ? `Reading prompt · ${pct}%` : "Reading prompt");
  } else if (live.state === "generating") {
    setPill("generating", `Generating · ${fmt(live.tok_s, 1)} tok/s`);
  } else {
    setPill("idle", live.state === "unloaded" ? "Unloaded" : "Idle");
  }
  modelButton(live);
  if (!modelOp && !localLoad && live.queued > 0) setPill("queued", `${live.queued} queued`);
  $("side-model-name").textContent = eng.model || "–";
  $("side-model-sub").textContent = [eng.max_context ? `${ctxfmt(eng.max_context)} context` : null,
                                     live.state === "unloaded" ? "not loaded" : "loaded"].filter(Boolean).join(" · ");
  $("side-model").title = eng.model || "";
  if (tab === "monitor") {
    renderDash(live, hw, st, eng, h, last, reqs, m);
    renderConvCache(m.conversation_cache);
  }
  if (tab === "requests") renderRequests(reqs, m);
  if (tab === "about") renderAbout(eng, hw, st, m);
}

// #596: the conversation cache - the prompt's state the engine keeps between requests (always), and the whole
// conversations it parks in RAM when "--conversation-cache-mib N" is in the run config's args (opt-in)
function since(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${fmt(s / 60)} min ago` : `${fmt(s / 3600, 1)} h ago`;
}
function renderConvCache(c) {
  $("cc-card").hidden = !c;
  if (!c) return;                                  // an older server
  const pct = (a, b) => (b ? `${Math.min(100, (100 * a) / b)}%` : "0%");
  $("cc-bars").hidden = !c.enabled;
  if (c.enabled) {
    $("cc-slots-text").textContent = `${fmt(c.parked)} / ${fmt(c.slots)}`;
    $("cc-slots-bar").style.width = pct(c.parked, c.slots);
    const budget = c.budget_mib * 1048576;
    $("cc-mem-text").textContent = `${gb(c.bytes)} / ${gb(budget)} GB`;
    $("cc-mem-bar").style.width = pct(c.bytes, budget);
  }
  $("cc-sum").textContent = c.requests ? `${fmt(c.requests_reused)} of ${fmt(c.requests)} requests reused part of their prompt` : "";
  const share = c.prompt_tokens ? ` (${fmt((100 * c.reused_tokens) / c.prompt_tokens)}% of all prompt tokens)` : "";
  const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens, ${since(c.last_at)}` : null;
  facts($("cc-facts"), [
    ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : null],
    ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens${share}` : null],
    ["Parked / restored", c.enabled ? `${fmt(c.parks)} / ${fmt(c.restores)}${c.evictions ? ` · ${fmt(c.evictions)} evicted` : ""}` : null],
    ["Last switch", c.enabled ? event : null],
  ]);
  $("cc-note").textContent = c.enabled
    ? "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
    : "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations (agents taking turns), set Parked conversations and the memory for them under Model and settings, then restart (docs/DETAILS.md).";
}

// ---- numbers behind the tiles and charts
const big = (n) => (n == null ? "–" : n >= 1e9 ? `${fmt(n / 1e9, 2)}B` : n >= 1e6 ? `${fmt(n / 1e6, n >= 1e7 ? 1 : 2)}M` : n >= 1e4 ? `${fmt(n / 1e3, 1)}k` : fmt(n));
const median = (a) => { const v = a.filter((x) => x != null && !Number.isNaN(x)).sort((x, y) => x - y); return v.length ? (v.length % 2 ? v[(v.length - 1) / 2] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : null; };
const quant = (a, q) => { const v = a.filter((x) => x != null && !Number.isNaN(x)).sort((x, y) => x - y); return v.length ? v[Math.min(v.length - 1, Math.floor(q * v.length))] : null; };
const freshOf = (r) => (r.prompt_tokens == null ? null : Math.max(0, r.prompt_tokens - (r.reused || 0)));
const prefillOf = (r) => { const f = freshOf(r); return f > 0 && r.prompt_ms > 0 ? f / (r.prompt_ms / 1000) : null; };
const ttftOf = (r) => (r.first_token_s != null ? r.first_token_s : r.first_engine_token_s > 0 ? r.first_engine_token_s : null);
const clock = (t) => new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
const clockShort = (t) => new Date(t * 1000).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
const dayLabel = (iso) => new Date(`${iso}T12:00:00`).toLocaleDateString([], {month: "short", day: "numeric"});
const FINISH = {stop: ["st-badge--generating", "Done"], length: ["st-badge--reading", "Max tokens"], cancel: ["st-badge--queued", "Stopped"],
                disconnect: ["st-badge--queued", "Closed"], error: ["st-badge--error", "Error"]};
const COLORS = {decode: "var(--c-decode)", prefill: "var(--c-prefill)", cached: "var(--c-cached)", fresh: "var(--c-fresh)", out: "var(--c-out)"};

// % change of the recent half against the one before it, as a small chip; better = "up" or "down"
function chip(cur, prev, better = "up", suffix = "") {
  if (cur == null || prev == null || prev === 0) return "";
  const d = (100 * (cur - prev)) / Math.abs(prev);
  if (!Number.isFinite(d)) return "";
  const good = Math.abs(d) < 0.5 ? "flat" : (d > 0) === (better === "up") ? "good" : "bad";
  return `<span class="delta delta--${good}">${d >= 0 ? "▲" : "▼"} ${fmt(Math.abs(d), Math.abs(d) < 10 ? 1 : 0)}%${suffix}</span>`;
}
function tileSpark(values, tone) {
  const [line, area] = sparkPath(values);
  if (!line) return `<svg class="spark spark--empty" viewBox="0 0 100 32" preserveAspectRatio="none"><path d="M0,29H100" stroke="currentColor" stroke-width="1" stroke-dasharray="2 3" fill="none" vector-effect="non-scaling-stroke" opacity=".4"/></svg>`;
  return `<svg class="spark" data-tone="${tone || ""}" viewBox="0 0 100 32" preserveAspectRatio="none"><path d="${area}" fill="currentColor" opacity=".12"/>` +
    `<path d="${line}" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/></svg>`;
}
function tile(label, value, unit, rows, spark, tone, hint) {
  return `<div class="tile"><div class="tile__head"><span>${esc(label)}</span>${hint ? `<span class="tile__hint" title="${esc(hint)}">?</span>` : ""}</div>` +
    `<div class="tile__value">${value}${unit ? `<small>${esc(unit)}</small>` : ""}</div>` +
    `<div class="tile__rows">${rows.filter(Boolean).join("")}</div>${spark ? tileSpark(spark, tone) : ""}</div>`;
}
const row = (text) => `<span class="tile__row">${text}</span>`;

// the lifetime counters by day, the last n days (zero-filled)
function lastDays(life, n = 14) {
  const days = (life && life.days) || {}, out = [];
  for (let i = n - 1; i >= 0; i--) {
    const d = new Date(); d.setDate(d.getDate() - i);
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    out.push({key, ...(days[key] || {requests: 0, prompt_tokens: 0, reused: 0, output_tokens: 0})});
  }
  return out;
}

// ---- tiny SVG charts (no library): a line chart and a stacked bar chart with a hover read-out
const chartState = {};
function niceMax(v) {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v)), f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}
function drawChart(id, spec) {
  const el = $(id);
  const w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  const n = spec.labels.length;
  if (n < 2 && spec.kind === "line" || n < 1) {
    el.innerHTML = `<div class="chart__empty muted">${esc(spec.empty || "Waiting for requests")}</div>`;
    chartState[id] = null;
    return;
  }
  const hasR = spec.series.some((s) => s.axis === "r");
  const L = 44, R = hasR ? 44 : 10, T = 8, B = 22, pw = w - L - R, ph = h - T - B;
  const stacked = spec.kind === "bars";
  const tops = spec.series[0].values.map((_, i) => spec.series.reduce((s, x) => s + (x.values[i] || 0), 0));
  if (stacked && !tops.some((t) => t > 0)) {   // (an all-zero axis would read 0,0,1,1,1)
    el.innerHTML = `<div class="chart__empty muted">${esc(spec.empty || "Waiting for requests")}</div>`;
    chartState[id] = null;
    return;
  }
  const maxOf = (list) => Math.max(...list.flatMap((s) => s.values.map((x) => x || 0)), 0);
  const rawMax = stacked ? Math.max(...tops, 0) : maxOf(spec.series.filter((s) => s.axis !== "r"));
  const maxR = hasR ? niceMax(maxOf(spec.series.filter((s) => s.axis === "r")) * 1.04) : 0;
  const max = niceMax(Math.max(rawMax, spec.ref || 0) * 1.04);
  const y = (v, axis) => T + ph - (v / (axis === "r" ? maxR : max)) * ph;
  const x = stacked ? (i) => L + (i + 0.5) * (pw / n) : (i) => L + (i / (n - 1)) * pw;
  let g = "";
  for (let k = 0; k <= 4; k++) {
    const v = (max * k) / 4, yy = y(v);
    g += `<line class="gl" x1="${L}" x2="${w - R}" y1="${yy.toFixed(1)}" y2="${yy.toFixed(1)}"/>` +
         `<text class="ytick" x="${L - 6}" y="${(yy + 3.5).toFixed(1)}" text-anchor="end">${spec.yfmt ? spec.yfmt(v) : big(v)}</text>`;
    if (hasR) g += `<text class="ytick ytick--r" x="${w - R + 6}" y="${(yy + 3.5).toFixed(1)}" text-anchor="start" style="fill:${spec.series.find((q) => q.axis === "r").color}">${big((maxR * k) / 4)}</text>`;
  }
  const want = Math.max(2, Math.floor(pw / 96));
  const step = Math.max(1, Math.ceil(n / want));
  for (let i = 0; i < n; i += step) {
    const anchor = i === 0 ? "start" : "middle";
    g += `<text class="xtick" x="${x(i).toFixed(1)}" y="${h - 6}" text-anchor="${anchor}">${esc(spec.xlabels ? spec.xlabels[i] : spec.labels[i])}</text>`;
  }
  if (stacked) {
    const bw = Math.max(2, Math.min(34, (pw / n) * 0.66));
    spec.series.forEach(() => {});
    for (let i = 0; i < n; i++) {
      let acc = 0;
      spec.series.forEach((s) => {
        const v = s.values[i] || 0;
        if (v > 0) {
          const y1 = y(acc + v), y0 = y(acc);
          g += `<rect class="bar" x="${(x(i) - bw / 2).toFixed(1)}" y="${y1.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(0.5, y0 - y1).toFixed(1)}" fill="${s.color}"/>`;
        }
        acc += v;
      });
    }
  } else {
    spec.series.forEach((s, si) => {
      let d = "", pen = false, first = null, lastX = 0;
      s.values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v, s.axis).toFixed(1)}`;
        if (first == null) first = x(i);
        lastX = x(i); pen = true;
      });
      if (si === 0 && spec.area && first != null) g += `<path d="${d}L${lastX.toFixed(1)},${T + ph}L${first.toFixed(1)},${T + ph}Z" fill="${s.color}" opacity=".10"/>`;
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`;
      if (n <= 30) s.values.forEach((v, i) => { if (v != null) g += `<circle cx="${x(i).toFixed(1)}" cy="${y(v, s.axis).toFixed(1)}" r="2.2" fill="${s.color}"/>`; });
    });
  }
  if (spec.ref != null) {
    g += `<line class="ref" x1="${L}" x2="${w - R}" y1="${y(spec.ref).toFixed(1)}" y2="${y(spec.ref).toFixed(1)}"/>` +
         `<text class="reftext" x="${w - R - 2}" y="${(y(spec.ref) - 4).toFixed(1)}" text-anchor="end">${esc(spec.reflabel || "")}</text>`;
  }
  el.innerHTML = `<svg viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" aria-hidden="true">${g}<line class="cross" y1="${T}" y2="${T + ph}" x1="-10" x2="-10"/></svg><div class="tip" hidden></div>`;
  chartState[id] = {spec, x, n, L, pw, stacked};
}
function chartHover(e) {
  const el = e.currentTarget, c = chartState[el.id];
  const tip = el.querySelector(".tip"), cross = el.querySelector(".cross");
  if (!c || !tip) return;
  const rect = el.getBoundingClientRect(), px = e.clientX - rect.left;
  const i = c.stacked ? Math.floor(((px - c.L) / c.pw) * c.n) : Math.round(((px - c.L) / c.pw) * (c.n - 1));
  if (i < 0 || i >= c.n) { chartLeave(e); return; }
  const cx = c.x(i);
  cross.setAttribute("x1", cx); cross.setAttribute("x2", cx); cross.style.opacity = 1;
  const s = c.spec;
  tip.innerHTML = `<b>${esc(s.tips ? s.tips[i] : s.labels[i])}</b>` + s.series.map((x) =>
    `<span class="tip__row"><i style="background:${x.color}"></i>${esc(x.name)}<em>${x.values[i] == null ? "–" : (x.fmt || fmt)(x.values[i])}</em></span>`).join("");
  tip.hidden = false;
  tip.style.left = `${Math.min(Math.max(cx + 12, 4), el.clientWidth - tip.offsetWidth - 4)}px`;
  tip.style.top = "6px";
  if (cx + 12 + tip.offsetWidth > el.clientWidth - 4) tip.style.left = `${Math.max(4, cx - tip.offsetWidth - 12)}px`;
}
function chartLeave(e) {
  const el = e.currentTarget, tip = el.querySelector(".tip"), cross = el.querySelector(".cross");
  if (tip) tip.hidden = true;
  if (cross) cross.style.opacity = 0;
}
for (const id of ["chart-tp", "chart-day", "chart-pr", "chart-ttft"]) {
  $(id).addEventListener("pointermove", chartHover);
  $(id).addEventListener("pointerleave", chartLeave);
}
const legend = (items) => items.map(([name, color]) => `<span><i style="background:${color}"></i>${esc(name)}</span>`).join("");
let chartSig = "";

function renderDash(live, hw, st, eng, h, last, reqs, m) {
  // the live strip
  const prog = $("state-progress");
  const on = live.queued > 0 ? "queued" : live.state;
  let label = live.state === "unloaded" ? "Model not loaded" : "Waiting for a request", detail = "", pct = 0;
  if (live.state === "reading") {
    label = "Reading prompt";
    prog.dataset.tone = "info";
    if (live.prompt_total) {
      pct = (100 * live.prompt_read) / live.prompt_total;
      detail = `${fmt(live.prompt_read)} / ${fmt(live.prompt_total)} tokens · ${fmt(pct)}%`;
    } else {
      detail = `${fmt(live.prompt_tokens)} tokens`;
    }
  } else if (live.state === "generating") {
    label = live.phase ? live.phase[0].toUpperCase() + live.phase.slice(1) : "Generating";
    delete prog.dataset.tone;
    pct = live.max_tokens ? Math.min(100, (100 * live.generated) / live.max_tokens) : 0;
    detail = `${fmt(live.generated)} tokens · ${fmt(live.tok_s, 1)} tok/s`;
  } else if (last) {
    delete prog.dataset.tone;
    detail = `last: ${fmt(last.output_tokens)} tokens${last.decode_tok_s ? ` at ${fmt(last.decode_tok_s, 1)} tok/s` : ""}`;
  }
  if (live.queued > 0) detail += `${detail ? " · " : ""}${live.queued} queued`;
  $("live-state").dataset.s = on;
  $("state-label").textContent = label;
  $("state-detail").textContent = detail;
  $("state-bar").style.width = `${pct}%`;
  $("page-sub").textContent = m.lifetime && m.lifetime.since ? `Lifetime counters since ${new Date(m.lifetime.since * 1000).toLocaleDateString([], {year: "numeric", month: "short", day: "numeric"})}` : "";

  // data: finished requests, oldest first
  const done = reqs.filter((r) => r.finish).slice().reverse();
  const recent = done.slice(-40);
  const decodes = recent.map((r) => r.decode_tok_s), prefills = recent.map(prefillOf), ttfts = recent.map(ttftOf);
  const L = m.lifetime || {requests: m.totals.requests, prompt_tokens: m.totals.prompt_tokens, reused: m.totals.reused,
                           output_tokens: m.totals.output_tokens, prompt_ms: m.totals.prompt_ms, decode_ms: m.totals.decode_ms, days: {}};
  const days = lastDays(L);
  const today = days[days.length - 1], yday = days[days.length - 2];
  const half = (arr, f) => { const v = arr.filter((x) => x != null); const k = Math.min(10, Math.floor(v.length / 2)); return k ? [f(v.slice(-k)), f(v.slice(-2 * k, -k))] : [null, null]; };
  const [dNow, dPrev] = half(decodes, median), [pNow, pPrev] = half(prefills, median), [tNow, tPrev] = half(ttfts, median);
  const lifeDecode = L.decode_ms > 0 ? L.output_tokens / (L.decode_ms / 1000) : null;
  const lifePrefill = L.prompt_ms > 0 && L.prompt_tokens > L.reused ? (L.prompt_tokens - L.reused) / (L.prompt_ms / 1000) : null;
  const ok = done.filter((r) => r.finish === "stop" || r.finish === "length").length;
  const errs = done.filter((r) => r.finish === "error").length, cut = done.length - ok - errs;
  const cacheShare = L.prompt_tokens ? (100 * L.reused) / L.prompt_tokens : null;
  const day = (f) => days.map(f);
  const nowSpeed = live.state === "generating" ? live.tok_s : dNow;
  const lifeFresh = Math.max(0, (L.prompt_tokens || 0) - (L.reused || 0));
  const lifeCost = calcCost(L.prompt_tokens, L.reused, L.output_tokens);
  const todayCost = calcCost(today.prompt_tokens, today.reused, today.output_tokens);
  const ydayCost = calcCost(yday.prompt_tokens, yday.reused, yday.output_tokens);
  const dayCost = (d) => calcCost(d.prompt_tokens, d.reused, d.output_tokens);
  const tiles = [
    tile("Cost saved", fmtCost(lifeCost), "", [
      row(`${chip(todayCost, ydayCost)} <span class="muted">${fmtCost(todayCost)} today</span>`),
      row(`<span class="muted">${fmtCost(calcCost(lifeFresh, 0, 0))} in · ${fmtCost(calcCost(0, 0, L.output_tokens))} out</span>`)], day(dayCost), "cost", "Estimated API cost saved vs Alibaba Cloud Qwen-Flash ($0.15/M input, $0.47/M output, $0.016/M cache read)"),
    tile("Input tokens", big(L.prompt_tokens), "", [
      row(`${chip(today.prompt_tokens, yday.prompt_tokens)} <span class="muted">${big(today.prompt_tokens)} today</span>`),
      row(`<span class="muted">${big(L.prompt_tokens - L.reused)} read · ${big(L.reused)} reused</span>`)], day((d) => d.prompt_tokens), "", "All prompt tokens the server has processed, lifetime"),
    tile("Output tokens", big(L.output_tokens), "", [
      row(`${chip(today.output_tokens, yday.output_tokens)} <span class="muted">${big(today.output_tokens)} today</span>`),
      row(`<span class="muted">${lifeDecode ? `${fmt(lifeDecode, 1)} tok/s lifetime avg` : "no decode timings yet"}</span>`)], day((d) => d.output_tokens), "out"),
    tile("Cached prompt tokens", big(L.reused), "", [
      row(`<span class="${cacheShare >= 50 ? "good" : ""}">${cacheShare == null ? "–" : `${fmt(cacheShare, cacheShare < 10 ? 1 : 0)}%`}</span> <span class="muted">of input reused</span>`),
      row(`<span class="muted">${big(today.reused)} today</span>`)], day((d) => d.reused), "cached", "Prompt tokens the engine did not have to read again because the conversation cache held them"),
    tile("Requests", big(L.requests), "", [
      row(`${chip(today.requests, yday.requests)} <span class="muted">${fmt(today.requests)} today</span>`),
      row(`<span class="muted">${fmt(m.totals.requests)} since this start</span>`)], day((d) => d.requests)),
    tile("Completed", done.length ? fmt((100 * ok) / done.length, ok === done.length ? 0 : 1) : "–", done.length ? "%" : "", [
      row(`<span class="${errs ? "bad" : "muted"}">${fmt(errs)} errors</span> <span class="muted">· ${fmt(cut)} cut short</span>`),
      row(`<span class="muted">of ${fmt(done.length)} kept requests</span>`)], null),
    tile("Decode speed", nowSpeed == null ? "–" : fmt(nowSpeed, 1), "tok/s", [
      row(`${chip(dNow, dPrev)} <span class="muted">${live.state === "generating" ? "live" : "median of last 10"}</span>`),
      row(`<span class="muted">${lifeDecode ? `${fmt(lifeDecode, 1)} lifetime avg` : "&nbsp;"}</span>`)], decodes, "decode"),
    tile("Prefill speed", pNow == null ? "–" : fmt(pNow), "tok/s", [
      row(`${chip(pNow, pPrev)} <span class="muted">median of last 10</span>`),
      row(`<span class="muted">${lifePrefill ? `${fmt(lifePrefill)} lifetime avg` : "&nbsp;"}</span>`)], prefills, "prefill", "New prompt tokens per second, not counting tokens served from the cache"),
    tile("Time to first token", tNow == null ? "–" : fmt(tNow, tNow < 10 ? 2 : 1), "s", [
      row(`${chip(tNow, tPrev, "down")} <span class="muted">median of last 10</span>`),
      row(`<span class="muted">p95 ${quant(ttfts, 0.95) == null ? "–" : `${fmt(quant(ttfts, 0.95), 1)} s`}</span>`)], ttfts, "ttft"),
  ].join("");
  if ($("tiles").dataset.sig !== tiles) { $("tiles").innerHTML = tiles; $("tiles").dataset.sig = tiles; }

  // charts: redraw only when their data changed (or the box did)
  const dim = `${$("chart-tp").clientWidth}x${$("chart-day").clientWidth}x${$("chart-pr").clientWidth}`;
  const sig = `${dim}|${done.length}|${done.length ? done[done.length - 1].id : ""}|${JSON.stringify(days.map((d) => d.prompt_tokens + d.output_tokens))}`;
  if (sig !== chartSig) {
    chartSig = sig;
    const pts = done.slice(-60).filter((r) => r.decode_tok_s || prefillOf(r));
    const lab = (r) => clockShort(r.time);
    drawChart("chart-tp", {kind: "line", area: true, labels: pts.map(lab), tips: pts.map((r) => `${new Date(r.time * 1000).toLocaleString()}`),
      series: [{name: "Decode", color: COLORS.decode, values: pts.map((r) => r.decode_tok_s), fmt: (v) => `${fmt(v, 1)} tok/s`},
               {name: "Prefill", color: COLORS.prefill, axis: "r", values: pts.map(prefillOf), fmt: (v) => `${fmt(v)} tok/s`}]});
    $("tp-legend").innerHTML = legend([["Decode (left axis)", COLORS.decode], ["Prefill (right axis)", COLORS.prefill]]);
    const dl = days.map((d) => dayLabel(d.key));
    drawChart("chart-day", {kind: "bars", labels: dl, xlabels: dl, empty: "No tokens counted yet",
      tips: dl.map((d, i) => `${days[i].key} · ${fmtCost(dayCost(days[i]))} saved`),
      series: [{name: "Cached input", color: COLORS.cached, values: days.map((d) => d.reused), fmt: big},
               {name: "Fresh input", color: COLORS.fresh, values: days.map((d) => d.prompt_tokens - d.reused), fmt: big},
               {name: "Output", color: COLORS.out, values: days.map((d) => d.output_tokens), fmt: big}]});
    $("day-legend").innerHTML = legend([["Cached input", COLORS.cached], ["Fresh input", COLORS.fresh], ["Output", COLORS.out]]);
    const rp = done.slice(-40);
    drawChart("chart-pr", {kind: "bars", labels: rp.map(lab), tips: rp.map((r) => new Date(r.time * 1000).toLocaleString()),
      series: [{name: "Cached", color: COLORS.cached, values: rp.map((r) => r.reused || 0), fmt: fmt},
               {name: "Fresh", color: COLORS.fresh, values: rp.map((r) => freshOf(r) || 0), fmt: fmt}]});
    $("pr-legend").innerHTML = legend([["Cached", COLORS.cached], ["Fresh", COLORS.fresh]]);
    const med = median(rp.map(ttftOf));
    drawChart("chart-ttft", {kind: "bars", labels: rp.map(lab), tips: rp.map((r) => new Date(r.time * 1000).toLocaleString()),
      yfmt: (v) => `${fmt(v, v < 10 ? 1 : 0)}s`, ref: med, reflabel: med == null ? "" : `median ${fmt(med, 2)} s`,
      series: [{name: "First token", color: COLORS.fresh, values: rp.map((r) => ttftOf(r) || 0), fmt: (v) => `${fmt(v, 2)} s`}]});
  }

  // how requests ended
  const cat = [["Completed", done.filter((r) => r.finish === "stop").length, "var(--c-cached)"],
               ["Hit max tokens", done.filter((r) => r.finish === "length").length, "var(--c-fresh)"],
               ["Stopped or closed", done.filter((r) => r.finish === "cancel" || r.finish === "disconnect").length, "var(--st-warn)"],
               ["Errors", errs, "var(--st-danger)"]];
  const fin = cat.map(([name, n, color]) => `<div class="fin__row"><i style="background:${color}"></i><span>${name}</span>` +
    `<b>${fmt(n)}</b><em>${done.length ? fmt((100 * n) / done.length, 0) : 0}%</em></div>`).join("");
  const stack = `<div class="fin__stack">${cat.map(([, n, c]) => (n ? `<span style="flex:${n};background:${c}"></span>` : "")).join("")}</div>`;
  const finHtml = stack + fin;
  if ($("fin").dataset.sig !== finHtml) { $("fin").innerHTML = finHtml; $("fin").dataset.sig = finHtml; }
  $("fin-sub").textContent = done.length ? `${fmt(done.length)} requests kept in memory` : "no requests yet";

  // context in use: the running request, else the last one
  const ctx = eng.max_context || 0;
  let used = 0;
  if (live.state !== "idle" && live.state !== "unloaded") used = (live.prompt_tokens || 0) + (live.generated || 0);
  else if (last) used = (last.prompt_tokens || 0) + (last.output_tokens || 0);
  const frac = ctx ? Math.min(1, used / ctx) : 0;
  $("ctx-bar").style.width = `${frac * 100}%`;
  $("ctx-sub").textContent = ctx ? `${kfmt(used)} / ${ctxfmt(ctx)} · ${fmt(frac * 100, frac < 0.1 ? 1 : 0)}%` : "–";
  $("ctx-note").textContent = live.state !== "idle" && live.state !== "unloaded" ? "the running request" : last ? "the last request" : "";

  // model and engine
  const loaded = live.state !== "unloaded";
  const lb = $("loaded-badge");
  lb.textContent = loaded ? "Loaded" : "Unloaded";
  lb.className = `st-badge${loaded ? " st-badge--generating" : ""}`;
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  const t = m.totals || {};
  facts($("facts-dash"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : (eng.model ? "built from source" : null)],
    ["Context size", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, ${fmt(eng.kv_resident)} positions per layer in VRAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP, up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} drafts${t.drafts_offered ? ` · ${fmt((100 * t.drafts_accepted) / t.drafts_offered)}% accepted` : ""}` : null],
    ["Parallel slots", live.parallel ? `${fmt(live.running || 0)} running of ${fmt(live.parallel)}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Since start", t.requests ? `${renderTotalsShort(t)}` : null],
  ]);

  // hardware
  const gen = hw.gpu_pcie_gen_max || hw.gpu_pcie_gen;
  const per = (f) => (hw.gpus || []).map((g) => `GPU ${g.index} ${f(g)}`).join(" · ");
  const multi = (hw.gpus || []).length > 1;
  setMetric("gpu", hw.gpu_util == null ? null : fmt(hw.gpu_util), "%",
            multi ? per((g) => (g.util == null ? "–" : `${fmt(g.util)}%`)) : st.gpu_name || (st.gpu_note ? "not available" : ""));
  spark("sp-gpu", h.gpu_util, 100);
  setMetric("vram", hw.gpu_mem_used == null ? null : gb(hw.gpu_mem_used), hw.gpu_mem_total ? `/ ${gb(hw.gpu_mem_total, 0)} GB` : "GB",
            multi ? per((g) => (g.mem_used == null ? "–" : `${gb(g.mem_used)} GB`)) : (st.gpu_note ? "not available on Windows AMD yet" : ""));
  spark("sp-vram", h.gpu_mem_used, hw.gpu_mem_total);
  setMetric("temp", hw.gpu_temp == null ? null : fmt(hw.gpu_temp), "°C", multi ? per((g) => (g.temp == null ? "–" : `${fmt(g.temp)}°`)) : "");
  spark("sp-temp", h.gpu_temp, 90);
  setMetric("power", hw.gpu_power == null ? null : fmt(hw.gpu_power), "W", hw.gpu_power_limit ? `of ${fmt(hw.gpu_power_limit)} W limit` : "");
  spark("sp-power", h.gpu_power, hw.gpu_power_limit);
  setMetric("pcie", gen ? `Gen${gen}` : null, hw.gpu_pcie_width ? `x${hw.gpu_pcie_width}` : "",
            hw.gpu_pcie_rx_mb == null ? "" : `to GPU ${fmt(hw.gpu_pcie_rx_mb, hw.gpu_pcie_rx_mb < 10 ? 1 : 0)} MB/s` +
            (hw.gpu_pcie_gen && gen && hw.gpu_pcie_gen < gen ? ` · idle Gen${hw.gpu_pcie_gen}` : ""));
  spark("sp-pcie", h.gpu_pcie_rx_mb);
  setMetric("cpu", hw.cpu == null ? null : fmt(hw.cpu), "%", st.threads ? `${st.cores ? `${st.cores} cores · ` : ""}${st.threads} threads` : "");
  spark("sp-cpu", h.cpu, 100);
  if (hw.disk_read_mb == null) {
    setMetric("disk", null, "", st.psutil ? "" : "needs psutil (setup installs it)");
  } else {
    const bigr = hw.disk_read_mb >= 1000;
    setMetric("disk", bigr ? fmt(hw.disk_read_mb / 1024, 2) : fmt(hw.disk_read_mb, hw.disk_read_mb < 10 ? 1 : 0), bigr ? "GB/s" : "MB/s",
              hw.disk_write_mb == null ? "" : `write ${fmt(hw.disk_write_mb, 1)} MB/s`);
  }
  spark("sp-disk", h.disk_read_mb);
  const cacheBytes = (eng.expert_cache_mib || 0) * 1048576;
  $("slots-text").parentElement.parentElement.hidden = !eng.expert_slots;
  $("temp-text").parentElement.parentElement.hidden = hw.gpu_temp == null;
  $("slots-text").textContent = eng.expert_slots ? `${fmt(eng.expert_slots)} · ${gb(cacheBytes)} GB` : "–";
  $("slots-bar").style.width = hw.gpu_mem_total ? `${Math.min(100, (100 * cacheBytes) / hw.gpu_mem_total)}%` : "0%";
  $("ram-text").textContent = hw.ram_total ? `${gb(hw.ram_used)} / ${gb(hw.ram_total, 0)} GB` : "–";
  const ramPct = hw.ram_total ? (100 * hw.ram_used) / hw.ram_total : 0;
  $("ram-bar").style.width = `${ramPct}%`;
  if (ramPct > 92) $("ram-progress").dataset.tone = "danger"; else delete $("ram-progress").dataset.tone;
  $("temp-text").textContent = hw.gpu_temp == null ? "–" : `${fmt(hw.gpu_temp)} °C`;
  $("temp-bar").style.width = hw.gpu_temp == null ? "0%" : `${Math.min(100, hw.gpu_temp)}%`;

  // the latest requests (compact)
  const body = $("dash-req-body");
  const html = reqs.length ? reqs.slice(0, 8).map((r) => reqCells(r, false)).join("") : `<tr><td colspan="10" class="muted">No requests yet</td></tr>`;
  if (body.dataset.sig !== html) { body.innerHTML = html; body.dataset.sig = html; }
}
$("goto-requests").onclick = () => showTab("requests");

function renderTotalsShort(t) {
  const read = t.prompt_tokens - t.reused;
  const cost = calcCost(t.prompt_tokens, t.reused, t.output_tokens);
  return `${fmt(t.requests)} requests · ${big(read)} read · ${big(t.output_tokens)} written · ${fmtCost(cost)} saved`;
}
function statusBadge(r) {
  const [cls, text] = FINISH[r.finish] || ["", r.finish || "–"];
  const proj = r.projection == null ? "" : ` <span class="st-badge${r.projection ? " st-badge--reading" : ""}" title="experimental speed projection ${r.projection ? "on" : "off"}">${r.projection ? "ESP" : "stock"}</span>`;
  return `<span class="st-badge ${cls}">${esc(text)}</span>${proj}`;
}
// one table row; the full table (the Requests page) adds the prompt bar, the expert hit rate and the drafts
function reqCells(r, full) {
  const fresh = freshOf(r), pf = prefillOf(r), tt = ttftOf(r);
  const t = clock(r.time);
  const cost = calcCost(r.prompt_tokens, r.reused, r.output_tokens);
  if (!full) {
    return `<tr><td>${esc(t)}</td><td>${statusBadge(r)}</td><td class="num">${fmt(fresh)}</td><td class="num">${fmt(r.reused)}</td>` +
      `<td class="num">${fmt(r.output_tokens)}</td><td class="num">${tt == null ? "–" : `${fmt(tt, 2)} s`}</td><td class="num">${fmt(pf)}</td>` +
      `<td class="num">${fmt(r.decode_tok_s, 1)}</td><td class="num">${fmt(r.duration_s, 1)} s</td>` +
      `<td class="num">${fmtCost(cost)}</td></tr>`;
  }
  const total = r.prompt_tokens || 0, share = total ? (100 * (r.reused || 0)) / total : 0;
  const hit = r.hit_rate == null ? "–" : `${(r.hit_rate * 100).toFixed(1)}%` +
    (r.pcie_share ? ` <span class="muted" title="routed experts the GPU read over PCIe (--pcie-frac) or another GPU computed">+${(r.pcie_share * 100).toFixed(1)}%</span>` : "");
  const drafts = r.drafts_offered ? `${fmt((100 * (r.drafts_accepted || 0)) / r.drafts_offered)}%` : "–";
  return `<tr title="${esc(r.id || "")}"><td>${esc(t)}</td><td>${statusBadge(r)}</td>` +
    `<td><div class="pcell"><span>${fmt(total)}</span><div class="pbar" title="${fmt(r.reused || 0)} reused of ${fmt(total)}"><i style="width:${share.toFixed(1)}%"></i></div>` +
    `<em>${share >= 1 ? `${fmt(share)}% cached` : "fresh"}</em></div></td>` +
    `<td class="num">${fmt(r.output_tokens)}</td><td class="num">${tt == null ? "–" : `${fmt(tt, 2)} s`}</td><td class="num">${fmt(pf)}</td>` +
    `<td class="num">${fmt(r.decode_tok_s, 1)}</td><td class="num">${hit}</td><td class="num">${drafts}</td><td class="num">${fmt(r.duration_s, 1)} s</td>` +
    `<td class="num">${fmtCost(cost)}</td></tr>`;
}
let reqFilter = "all";
for (const b of $("req-filter").children) b.onclick = () => {
  reqFilter = b.dataset.f;
  for (const x of $("req-filter").children) x.setAttribute("aria-checked", String(x === b));
  $("req-body").dataset.sig = "";
  if (lastMetrics) render(lastMetrics);
};
function renderRequests(reqs, m) {
  const keep = (r) => reqFilter === "all" || (reqFilter === "ok" ? r.finish === "stop" || r.finish === "length"
    : reqFilter === "error" ? r.finish === "error" : r.finish === "cancel" || r.finish === "disconnect");
  const shown = reqs.filter(keep);
  const sig = `${reqFilter}|${reqs.length}|${reqs[0] ? reqs[0].id : ""}`;
  const body = $("req-body");
  if (body.dataset.sig !== sig) {
    body.dataset.sig = sig;
    body.innerHTML = shown.length ? shown.map((r) => reqCells(r, true)).join("")
                                  : `<tr><td colspan="11" class="muted">${reqs.length ? "No request matches this filter" : "No requests yet"}</td></tr>`;
  }
  const kept = m.requests_kept == null ? reqs.length : m.requests_kept;
  $("req-count").textContent = `${fmt(shown.length)}${reqFilter === "all" ? "" : ` of ${fmt(reqs.length)}`} requests · the server keeps the last ${fmt(Math.max(kept, 500))}`;
  $("req-totals").textContent = renderTotals(m.totals);
}
function renderTotals(t) {
  if (!t || !t.requests) return "";
  const since = new Date(t.since * 1000).toLocaleString([], {weekday: "short", hour: "2-digit", minute: "2-digit"});
  const read = t.prompt_tokens - t.reused;
  const pSpeed = t.prompt_ms > 0 && read > 0 ? ` at ${fmt(read / (t.prompt_ms / 1000))} tok/s` : "";
  const oSpeed = t.decode_ms > 0 && t.output_tokens > 0 ? ` at ${fmt(t.output_tokens / (t.decode_ms / 1000), 1)} tok/s` : "";
  const cost = calcCost(t.prompt_tokens, t.reused, t.output_tokens);
  return `Since ${since}: ${fmt(t.requests)} requests · ${fmt(read)} prompt tokens read${pSpeed} (${fmt(t.reused)} reused) · ` +
         `${fmt(t.output_tokens)} written${oSpeed} · ${fmtCost(cost)} saved`;
}

function facts(el, rows) {
  el.innerHTML = rows.filter((r) => r[1] != null && r[1] !== "").map(([k, v, copy]) =>
    `<dt>${esc(k)}</dt><dd>${copy ? `<code>${esc(v)}</code><button class="st-btn st-btn--icon" data-copy="${esc(v)}" aria-label="Copy">${icon("copy")}</button>` : esc(v)}</dd>`).join("");
}
// INFO cvec=project:4-44[:singleL] | add:A-B | 0
function projectionText(c) {
  if (!c || c === "0" || c === 0) return null;
  const [mode, range, single] = String(c).split(":");
  const [a, b] = (range || "").split("-");
  return `${mode === "project" ? "Projection" : "Additive"} control vector on layers ${a}–${b}` +
         `${single ? ` (layer ${single.replace("single", "")}'s direction)` : ""}. Per chat in Sampling. Its package ` +
         "describes the vector as a refusal-direction projection; measure the speed yourself";
}
function renderAbout(eng, hw, st, m) {
  const kv = {int8: "8-bit", q4_0: "4-bit (Hadamard-rotated)", fp16: "16-bit"}[eng.kv] || eng.kv;
  const L = (m && m.lifetime) || (m && m.totals) || {};
  const days = lastDays(L);
  const today = days[days.length - 1] || {};
  const lifeCost = calcCost(L.prompt_tokens, L.reused, L.output_tokens);
  const todayCost = calcCost(today.prompt_tokens, today.reused, today.output_tokens);
  facts($("facts-engine"), [
    ["Model", eng.model],
    ["Engine", eng.version ? `v${eng.version}` : "built from source"],
    ["Context", eng.max_context ? `${fmt(eng.max_context)} tokens` : null],
    ["KV cache", kv ? `${kv}${eng.kv_resident ? `, streamed: ${fmt(eng.kv_resident)} positions per layer in VRAM, the rest in RAM` : ", all in VRAM"}` : null],
    ["Experts in VRAM", eng.expert_slots ? `${fmt(eng.expert_slots)} (${gb((eng.expert_cache_mib || 0) * 1048576)} GB)` : null],
    ["Speculation", eng.spec ? `MTP drafts up to ${Math.max(0, (eng.mtp_max || eng.spec) - 1)} tokens${eng.lookup ? ", prompt lookup on" : ""}` : null],
    ["Images", eng.images ? "on" : "off"],
    ["Experimental speed projection", projectionText(eng.cvec)],
    ["Cost saved", `${fmtCost(lifeCost)} lifetime (${fmtCost(todayCost)} today)`],
  ]);
  facts($("facts-hw"), [
    ["GPU", st.gpu_name ? `${st.gpu_name}${hw.gpu_mem_total ? `, ${gb(hw.gpu_mem_total, 0)} GB` : ""}` : (st.gpu_note || "not readable (NVML)")],
    ["CPU", st.cpu_name ? `${st.cpu_name}${st.threads ? `, ${st.threads} threads` : ""}` : null],
    ["RAM", hw.ram_total ? `${gb(hw.ram_total, 0)} GB` : null],
  ]);
  if ($("facts-cost")) {
    facts($("facts-cost"), [
      ["Provider", QWEN_FLASH_PRICING.provider],
      ["Model baseline", QWEN_FLASH_PRICING.model],
      ["Input /M", `$${QWEN_FLASH_PRICING.inputPerM.toFixed(2)}`],
      ["Output /M", `$${QWEN_FLASH_PRICING.outputPerM.toFixed(2)}`],
      ["Cache read /M", `$${QWEN_FLASH_PRICING.cacheReadPerM.toFixed(3)}`],
      ["Cache create (5m) /M", `$${QWEN_FLASH_PRICING.cacheCreate5mPerM.toFixed(2)}`],
      ["Cache read (5m) /M", `$${QWEN_FLASH_PRICING.cacheRead5mPerM.toFixed(3)}`],
      ["Lifetime saved", `${fmtCost(lifeCost)} (${big(L.prompt_tokens || 0)} input · ${big(L.output_tokens || 0)} output)`],
      ["Today saved", `${fmtCost(todayCost)} (${big(today.prompt_tokens || 0)} input · ${big(today.output_tokens || 0)} output)`],
    ]);
  }
  const base = location.origin;
  facts($("facts-api"), [
    ["OpenAI base URL", `${base}/v1`, true],
    ["Anthropic base URL", base, true],
    ["Model name", eng.model, true],
  ]);
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-copy]");
  if (b) copyText(b.dataset.copy, b);
});

// ------------------------------------------------------------------ MCP servers (GET /mcp)
// Tools from the MCP servers in the run config: the chat offers them to the model (opt-in per request,
// "strata_mcp": true, which only this page sends); the Monitor lists the servers and what they offer.
let mcpInfo = {servers: [], tools: 0}, mcpRetry = null;
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (!r.ok) return;
    mcpInfo = await r.json();
  } catch (e) { return; /* an older server: no MCP */ }
  renderMcp();
  clearTimeout(mcpRetry);                          // right after the start, servers may still be starting (npx downloads)
  if ((mcpInfo.servers || []).some((s) => s.status === "starting")) mcpRetry = setTimeout(loadMcp, 3000);
}
const MCP_STATE = {ready: ["st-badge--generating", "Connected"], starting: ["st-badge--reading", "Starting"],
                   failed: ["st-badge--error", "Failed"], stopped: ["st-badge--queued", "Stopped"], idle: ["", "Waiting"]};
function renderMcp() {
  const servers = mcpInfo.servers || [];
  $("mcp-card").hidden = !servers.length;
  $("mcp-row").hidden = !servers.length;
  const ready = servers.filter((s) => s.status === "ready" || s.status === "stopped");
  $("mcp-sum").textContent = servers.length ? `${fmt(mcpInfo.tools)} tools · ${ready.length} of ${servers.length} servers connected` : "";
  $("mcp-row-sub").textContent = mcpInfo.tools ? `${fmt(mcpInfo.tools)} tools from ${ready.map((s) => s.name).join(", ")}; the model calls them when it decides to`
                                               : "no server is connected yet (see the Monitor)";
  $("mcp-list").innerHTML = servers.map((s) => {
    const [cls, text] = MCP_STATE[s.status] || ["", s.status];
    const info = s.info && s.info.name ? ` · ${s.info.name}${s.info.version ? ` ${s.info.version}` : ""}` : "";
    return `<div class="mcp-server"><div class="mcp-server__head"><span class="st-badge ${cls}">${esc(text)}</span>` +
      `<strong>${esc(s.name)}</strong><span class="muted small">${esc(s.transport)} · ${fmt(s.tools.length)} tools${esc(info)}</span></div>` +
      (s.error ? `<div class="msg-error">${esc(s.error)}</div>` : "") +
      (s.tools.length ? `<div class="mcp-server__tools">${s.tools.map((t) => `<span class="chip" title="${esc(t.description || "")}">${esc(t.tool)}</span>`).join("")}</div>` : "") +
      `</div>`;
  }).join("");
}

// ------------------------------------------------------------------ Model settings (GET / POST /config, #564)
// A few documented keys of the run config (strata-<model>.json), for every client, from the next start on.  The
// server lists them, checks every value and keeps every other key of the file as it is.
let cfgKeys = [];
async function loadConfig() {
  let r;
  try { r = await fetch("config", {headers: headers()}); } catch (e) { return; }
  if (!r.ok) { $("cfg-card").hidden = true; return; }       // no run config, an older server, or no key yet
  const c = await r.json();
  cfgKeys = c.keys || [];
  const models = c.models || [];                     // the run configs next to this one: other models to switch to
  $("cfg-model-row").hidden = models.length < 2;
  $("cfg-model").innerHTML = models.map((m) =>
    `<option value="${esc(m.file)}"${m.current ? " selected" : ""}>${esc(m.model)} (${esc(m.file)})${m.current ? " - running" : ""}</option>`).join("");
  $("cfg-file").textContent = c.file || "";
  $("cfg-form").innerHTML = cfgKeys.map((k, i) => {
    const id = `cfg-${i}`, v = k.value;
    let input;
    if (k.kind === "bool" || k.kind === "enum") {
      const opts = k.kind === "bool" ? [["true", "on"], ["false", "off"]] : k.choices.map((x) => [x, x]);
      const cur = v == null ? "" : String(v);
      input = `<select class="st-input" id="${id}"><option value=""${cur === "" ? " selected" : ""}>default</option>` +
        opts.map(([val, text]) => `<option value="${esc(val)}"${cur === val ? " selected" : ""}>${esc(text)}</option>`).join("") + `</select>`;
    } else {
      const text = v == null ? "" : Array.isArray(v) ? v.join(", ") : String(v);
      input = `<input class="st-input" id="${id}" ${k.kind === "number" ? 'type="number" step="any" min="0"' : 'type="text"'} ` +
        `value="${esc(text)}" placeholder="default" autocomplete="off">`;
    }
    return `<label for="${id}" title="${esc(k.help)}">${esc(k.help)}<code>${esc(k.key)}</code></label>${input}`;
  }).join("");
  const pi = cfgKeys.findIndex((k) => k.key === "parallel"), bi = cfgKeys.findIndex((k) => k.key === "batch_mtp");
  if (pi >= 0 && bi >= 0) {                          // batch_mtp only means something with 2 or more parallel requests
    const sync = () => {
      const on = Number($(`cfg-${pi}`).value) >= 2, b = $(`cfg-${bi}`);
      b.disabled = !on;
      if (!on) b.value = "";
    };
    $(`cfg-${pi}`).addEventListener("input", sync);
    sync();
  }
  $("cfg-card").hidden = false;
}
function configValue(k, el) {
  const s = el.value.trim();
  if (s === "") return null;
  if (k.kind === "bool") return s === "true";
  if (k.kind === "number") return Number(s);
  return s;                                         // enum, or names (the server splits them at commas)
}
$("cfg-save").addEventListener("click", async () => {
  const set = {};
  cfgKeys.forEach((k, i) => {
    const v = configValue(k, $(`cfg-${i}`));
    const old = Array.isArray(k.value) ? k.value.join(", ") : k.value;
    if (JSON.stringify(v) !== JSON.stringify(old ?? null)) set[k.key] = v;
  });
  if (!Object.keys(set).length) { $("cfg-msg").textContent = "Nothing changed."; return; }
  try {
    const r = await fetch("config", {method: "POST", headers: headers(true), body: JSON.stringify({set})});
    const b = await r.json();
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
    $("cfg-msg").textContent = b.changed.length
      ? `Saved (${b.changed.join(", ")}); the earlier file is ${b.file}.bak. Restart to use it.` : "Nothing changed.";
    loadConfig();
  } catch (e) {
    $("cfg-msg").textContent = "";
    toast("error", "Not saved", String(e.message || e), 6000);
  }
});

// POST /v1/restart: the server unloads, stops cleanly and starts the same command again, so the whole run config is read
// again. The connection drops on the way; the page waits for /health and reloads itself.
// config: another run config next to this one (a model switch), else the same one read again
async function restartServer(config) {
  const live = (lastMetrics && lastMetrics.live) || {};
  const what = config ? `switch to ${config}` : "read the run config again";
  // a running or queued request (or one the engine hangs on) is ended by a forced restart, after asking
  const force = turnBlocksUnload(live);
  if (!confirm(force
    ? `A request is running (or the engine is stuck on it). Force restart? It ends that request, then Strata restarts to ${what} and loads the model (a few minutes).`
    : `Restart Strata now to ${what}? The model unloads and the model loads again (a few minutes).`)) return;
  if (force && busy) busy.controller.abort();
  $("cfg-restart").disabled = $("cfg-switch").disabled = true;
  try {
    const r = await fetch(force ? "v1/restart?force=1" : "v1/restart",
                          {method: "POST", headers: headers(true), body: JSON.stringify(config ? {config} : {})});
    let b = {};
    try { b = await r.json(); } catch (e) { /* no body */ }
    if (!r.ok) throw new Error((b.error || {}).message || `HTTP ${r.status}`);
  } catch (e) {
    $("cfg-restart").disabled = $("cfg-switch").disabled = false;
    toast("error", "Not restarted", String(e.message || e), 8000);
    return;
  }
  $("cfg-msg").textContent = "Restarting\u2026 this page reloads when Strata is back.";
  toast("info", "Restarting", "The page reloads when Strata is back.", 8000);
  const t0 = Date.now();
  await new Promise((resolve) => setTimeout(resolve, 3000));
  while (Date.now() - t0 < 30 * 60 * 1000) {
    try { if ((await fetch("health", {cache: "no-store"})).ok) { location.reload(); return; } } catch (e) { /* still down */ }
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
  $("cfg-restart").disabled = $("cfg-switch").disabled = false;
  $("cfg-msg").textContent = "Strata did not come back in 30 minutes: look at its window.";
}
$("cfg-restart").addEventListener("click", () => restartServer(null));
$("cfg-switch").addEventListener("click", () => {
  const sel = $("cfg-model");
  const cur = sel.querySelector("option[selected]");
  if (cur && sel.value === cur.value) { toast("info", "Already running", "Pick another model first."); return; }
  restartServer(sel.value);
});

// ------------------------------------------------------------------ Markdown (escaped first, then formatted)
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
  s = esc(s)
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
    .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
}
function codeBlock(lang, code) {
  return `<div class="st-code"><div class="st-code__head"><span>${esc(lang || "code")}</span>` +
    `<button class="st-btn st-btn--icon" data-code-copy aria-label="Copy code">${icon("copy")}</button></div>` +
    `<pre><code>${esc(code)}</code></pre></div>`;
}
function blocks(text) {
  const out = [], lines = text.split("\n");
  let para = [], list = null;
  const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
  const flushList = () => { if (list) out.push(`<${list.tag}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`); list = null; };
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    let m;
    if (!l.trim()) { flushPara(); flushList(); continue; }
    if ((m = l.match(/^(#{1,6})\s+(.*)$/))) { flushPara(); flushList(); out.push(`<${m[1].length <= 2 ? "h3" : "h4"}>${inline(m[2])}</${m[1].length <= 2 ? "h3" : "h4"}>`); continue; }
    if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
    if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
      flushPara(); flushList();
      const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
      i--;
      out.push(html + "</tbody></table>");
      continue;
    }
    if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
      flushPara();
      const tag = m[1] ? "ol" : "ul";
      if (!list || list.tag !== tag) { flushList(); list = {tag, items: []}; }
      list.items.push(m[2]);
      continue;
    }
    if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
    flushList();
    para.push(l);
  }
  flushPara(); flushList();
  return out.join("");
}
function markdown(text) {
  let html = "", rest = text;
  for (;;) {
    const m = rest.match(/(^|\n)```([^\n`]*)\n/);
    if (!m) { html += blocks(rest); break; }
    html += blocks(rest.slice(0, m.index));
    rest = rest.slice(m.index + m[0].length);
    const end = rest.match(/(^|\n)```[ \t]*(\n|$)/);
    if (!end) { html += codeBlock(m[2].trim(), rest); break; }         // still streaming
    html += codeBlock(m[2].trim(), rest.slice(0, end.index));
    rest = rest.slice(end.index + end[0].length);
  }
  return html;
}

// ------------------------------------------------------------------ Chat
// The sampling shown until the user changes it is the model card's thinking set. Untouched, the page sends NO sampling
// fields: the server picks the card's set for the request's thinking state (thinking 1.0 / 0.95 / 20, not thinking
// 0.7 / 0.8 / 20 with a presence penalty). Before 0.1.45 the page sent 0.6 itself; a stored 0.6 with the other two
// values untouched counts as "not set" too.
const DEFAULTS = {thinking: "high", temperature: 1.0, top_p: 0.95, top_k: 20, max: "", seed: "", show: true, esp: true, mcp: true};
const OLD_DEFAULT_TEMPERATURE = 0.6;
function samplingUntouched(s) {
  const t = +s.temperature;
  return (t === DEFAULTS.temperature || t === OLD_DEFAULT_TEMPERATURE) && +s.top_p === DEFAULTS.top_p && +s.top_k === DEFAULTS.top_k;
}
let settings = {...DEFAULTS, ...store.get("sampling", {})};
if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
let messages = store.get("chat", []);
let attachments = [];                 // {name, url}
let busy = null;                      // {controller, msg}

function saveChat() {
  store.set("chat", messages.map((m) => ({...m, images: (m.images || []).map((i) => ({name: i.name})),
                                           files: (m.files || []).map((f) => ({name: f.name}))})));
}
function timeStr(t) { return new Date(t).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"}); }

function msgEl(m, i) {
  const el = document.createElement("div");
  el.className = `st-msg st-msg--${m.role}`;
  el.dataset.i = i;
  if (m.role === "user") {
    if (m.files && m.files.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const f of m.files) {
        const c = document.createElement("span");
        c.className = "chip";
        c.innerHTML = icon("attach", "st-icon st-icon--sm");
        c.append(f.name);
        wrap.appendChild(c);
      }
      el.appendChild(wrap);
    }
    if (m.images && m.images.length) {
      const wrap = document.createElement("div");
      wrap.className = "msg-images";
      for (const im of m.images) {
        if (im.url) { const img = document.createElement("img"); img.src = im.url; img.alt = im.name || "image"; wrap.appendChild(img); }
        else { const c = document.createElement("span"); c.className = "chip"; c.innerHTML = icon("image", "st-icon st-icon--sm"); c.append(im.name || "image"); wrap.appendChild(c); }
      }
      el.appendChild(wrap);
    }
    const b = document.createElement("div");
    b.className = "st-bubble";
    b.textContent = m.text;
    el.appendChild(b);
    const meta = document.createElement("div");
    meta.className = "st-msg__meta";
    meta.textContent = `You · ${timeStr(m.time)}`;
    el.appendChild(meta);
  } else {
    el.innerHTML = `<details class="st-collapse think" hidden><summary>${icon("thinking", "st-icon st-icon--sm")}<span class="think-title"></span>` +
      `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body thinking"></div></details>` +
      `<div class="st-bubble"></div><div class="st-msg__meta"><span class="meta-text"></span>` +
      `<button class="st-btn st-btn--icon" data-msg-copy aria-label="Copy the answer" title="Copy">${icon("copy")}</button></div>`;
    updateAssistant(el, m, false);
  }
  return el;
}
// One MCP tool call in the answer: a compact block (name, state, a one-line preview) that opens to the arguments and
// the result as the model read it.  Its body is built only while open: a result can be 20,000 characters.
const TOOL_STATE = {writing: ["st-badge--reading", "Writing"], running: ["st-badge--generating", "Running"], done: ["", "Done"],
                    error: ["st-badge--error", "Error"], skipped: ["st-badge--queued", "Not run"]};
function toolHtml(t, k) {
  const [cls, label] = TOOL_STATE[t.state] || ["", t.state];
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  const preview = t.result != null ? t.result : args.replace(/\s+/g, " ");
  let body = "";
  if (t.open) {
    body = `<div class="tool-call__label">Arguments</div><pre class="tool-call__pre">${esc(args || "(being written)")}</pre>`;
    if (t.result != null) {
      body += `<div class="tool-call__label">${t.ok ? "Result" : "Error"}${t.chars ? ` · ${fmt(t.chars)} characters` : ""}` +
              `${t.truncated ? ", cut for the model" : ""}</div><pre class="tool-call__pre">${esc(t.result)}</pre>`;
    }
  }
  return `<details class="st-collapse tool-call" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}>` +
    `<summary>${icon("tool", "st-icon st-icon--sm")}<span class="tool-call__name" title="${esc(t.name || "")}">${esc(t.tool || t.name || "tool")}</span>` +
    (t.server ? `<span class="muted small">${esc(t.server)}</span>` : "") +
    `<span class="tool-call__preview muted">${esc(preview.slice(0, 200))}</span>` +
    `<span class="st-badge ${cls}">${esc(label)}</span>${t.ms != null && t.state !== "skipped" ? `<span class="muted small">${fmt(t.ms / 1000, 1)} s</span>` : ""}` +
    `${icon("chevron", "st-icon st-icon--sm st-chev")}</summary><div class="st-collapse__body">${body}</div></details>`;
}
// the answer's text with the tool blocks where the model called them
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return markdown(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += markdown(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + markdown(m.text.slice(pos));
}
// a tool event from the stream (the `strata_mcp` field of a chunk)
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms,
                      state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  const det = el.querySelector("details.think");
  if (m.reasoning) {
    det.hidden = false;
    const thinkingNow = streaming && !m.text;
    el.querySelector(".think-title").textContent = thinkingNow ? "Thinking…" :
      m.thinkSecs != null ? `Thought for ${fmt(m.thinkSecs, 1)} s` : "Thoughts";
    const body = el.querySelector(".thinking");
    if (det.open || thinkingNow) body.textContent = m.reasoning;
    else body.dataset.pending = "1";
    // open while it streams (if wanted), closed once the answer starts - unless the user toggled it themselves
    if (thinkingNow && settings.show && !det.dataset.touched && !det.open) { det._auto = true; det.open = true; }
    if (!thinkingNow && det.open && !det.dataset.touched) { det._auto = true; det.open = false; }
  }
  const bubble = el.querySelector(".st-bubble");
  if (m.error) {
    bubble.innerHTML = `<div class="msg-error"></div>`;
    bubble.firstChild.textContent = m.error;
  } else if (!m.text && streaming && !(m.tools && m.tools.length)) {
    bubble.innerHTML = m.reasoning ? `<span class="muted cursor">Writing</span>` : `<span class="cursor"></span>`;
  } else {
    bubble.innerHTML = answerHtml(m);
    if (streaming) bubble.classList.add("cursor"); else bubble.classList.remove("cursor");
  }
  el.querySelector(".meta-text").textContent = m.meta || (streaming ? "" : m.stopped ? "Stopped" : "");
  el.querySelector("[data-msg-copy]").hidden = streaming || !m.text;
}
function renderChat() {
  const chat = $("chat");
  chat.querySelectorAll(".st-msg").forEach((e) => e.remove());
  $("chat-empty").hidden = messages.length > 0;
  messages.forEach((m, i) => chat.appendChild(msgEl(m, i)));
  scrollDown(true);
}
function nearBottom() { const s = $("chat-scroll"); return s.scrollHeight - s.scrollTop - s.clientHeight < 120; }
function scrollDown(force) { const s = $("chat-scroll"); if (force || nearBottom()) s.scrollTop = s.scrollHeight; }

$("chat").addEventListener("click", (e) => {
  const cc = e.target.closest("[data-code-copy]");
  if (cc) { copyText(cc.closest(".st-code").querySelector("pre").textContent, cc); return; }
  const mc = e.target.closest("[data-msg-copy]");
  if (mc) { const i = +mc.closest(".st-msg").dataset.i; copyText(messages[i].text, mc); return; }
  // a tool block: its open state lives in the message (the answer is rebuilt while it streams), so the click sets it
  const sum = e.target.closest(".tool-call > summary");
  if (sum) {
    e.preventDefault();
    const el = sum.closest(".st-msg"), m = messages[+el.dataset.i], t = m && m.tools && m.tools[+sum.parentElement.dataset.tool];
    if (!t) return;
    t.open = !t.open;
    updateAssistant(el, m, !!busy && busy.msg === m);
  }
});
$("chat").addEventListener("toggle", (e) => {
  const d = e.target;
  if (d.tagName !== "DETAILS" || !d.classList.contains("think")) return;
  if (d._auto) { d._auto = false; return; }          // our own open/close, not the user's
  d.dataset.touched = "1";
  const body = d.querySelector(".thinking");
  if (d.open && body.dataset.pending) { body.textContent = messages[+d.closest(".st-msg").dataset.i].reasoning; delete body.dataset.pending; }
}, true);

function apiMessages() {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text},
        ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else if (!(busy && busy.msg === m)) {            // the answer being asked for now is not history yet
      out.push(...assistantMessages(m));
    }
  }
  return out;
}
// An answer that used MCP tools goes back as the model wrote it: per round the text before the calls, the calls and
// their results (as the model read them), then the rest - so the next question can build on what the tools found.
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  // #1392: a turn with no answer text (only reasoning, a stop before the first content token, or an error) still goes
  // back as an assistant turn, so the history keeps alternating and a reasoning model sees that it already answered.
  // The server leaves such a turn out of the prompt itself (serve/frontend.py, #843); reasoning_content rides along.
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}

function setBusy(on) {
  paintLive();                                       // the Load / Unload button and the pill follow a turn at once
  $("stop-btn").hidden = !on;
  $("send-btn").disabled = on;
  $("composer-hint").textContent = on ? "" : "Shift+Enter: new line";
}

async function send() {
  const text = $("input").value.trim();
  if ((!text && !attachments.length) || busy) return;
  messages.push({role: "user", text, images: attachments.filter((a) => a.kind !== "file"),
                 files: attachments.filter((a) => a.kind === "file"), time: Date.now()});
  attachments = [];
  renderAttachments();
  $("input").value = "";
  autosize();
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  messages.push(m);
  renderChat();
  const el = $("chat").lastElementChild;
  const controller = new AbortController();
  busy = {controller, msg: m};
  setBusy(true);

  const body = {model: health.model, messages: apiMessages(), stream: true,
                reasoning_effort: settings.thinking};
  if (!(settings.temperature > 0)) {
    body.temperature = 0;
  } else if (!samplingUntouched(settings)) {
    Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (projectionLoaded()) body.experimental_speed_projection = !!settings.esp;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;   // this server may run MCP tools for it

  let firstAt = null, thinkStart = null, usage = null, frame = 0;
  const paint = () => { frame = 0; updateAssistant(el, m, true); scrollDown(); };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body),
                                                   signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it under About > Settings.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;              // ": keep-alive" comments while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;   // a new round after a tool
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else { m.error = e.message || String(e); toast("error", "The request failed", m.error, 6000); }
  }
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  const n = usage ? usage.completion_tokens : null;
  if (n && firstAt) {
    const secs = (performance.now() - firstAt) / 1000;
    const pt = usage.prompt_tokens || 0;
    const reused = (usage.prompt_tokens_details && usage.prompt_tokens_details.cached_tokens) || 0;
    const cost = calcCost(pt, reused, n);
    m.meta = `${fmt(n)} tokens${secs > 0.25 ? ` · ${fmt(n / secs, 1)} tok/s` : ""}${cost > 0 ? ` · ${fmtCost(cost)} saved` : ""}${m.stopped ? " · stopped" : ""}` +
             (projectionLoaded() ? (settings.esp ? " · projection on" : " · projection off") : "");
  } else if (m.stopped) {
    m.meta = "Stopped";
  }
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) m.meta = `${m.meta ? `${m.meta} · ` : ""}${ran} tool call${ran > 1 ? "s" : ""}`;
  if (m.limit) m.meta = `${m.meta || ""} · stopped at the limit of ${m.limit} tool rounds (mcp.max_rounds)`;
  busy = null;
  setBusy(false);
  if (frame) cancelAnimationFrame(frame);
  updateAssistant(el, m, false);
  saveChat();
  scrollDown();
}

$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
});
function autosize() { const t = $("input"); t.style.height = "auto"; t.style.height = `${Math.min(t.scrollHeight, innerHeight * 0.4)}px`; }
$("input").addEventListener("input", autosize);

$("new-btn").onclick = () => {
  if (busy) { toast("warn", "Still writing", "Stop the answer first."); return; }
  if (!messages.length) return;
  const backup = messages;
  messages = [];
  saveChat();
  renderChat();
  toast("info", "New chat", "The last one was cleared.", 6000, {label: "Undo", run: () => { messages = backup; saveChat(); renderChat(); }});
};
$("export-btn").onclick = () => {
  if (!messages.length) { toast("info", "Nothing to save yet"); return; }
  const tools = (m) => (m.tools || []).filter((t) => t.result != null).map((t) =>
    `<details><summary>Tool ${t.server ? `${t.server} / ` : ""}${t.tool || t.name}${t.ok ? "" : " (error)"}</summary>\n\n` +
    `\`\`\`json\n${JSON.stringify(t.arguments || {}, null, 2)}\n\`\`\`\n\n\`\`\`\n${t.result}\n\`\`\`\n\n</details>\n\n`).join("");
  const md = messages.map((m) => m.role === "user" ? `## You\n\n${m.text}\n` :
    `## ${health.model}\n\n${m.reasoning ? `<details><summary>Thinking</summary>\n\n${m.reasoning}\n\n</details>\n\n` : ""}${tools(m)}${m.text || m.error || ""}\n`).join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([md], {type: "text/markdown"}));
  a.download = `strata-chat-${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.md`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
};

// pictures and text files: the attach button, dropping them on the chat, or pasting a picture (issue #30)
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
const MAX_TEXT_FILE = 512 * 1024;
function isTextFile(f) {
  return f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) ||
         TEXT_EXT.test(f.name) || /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
}
function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("warn", "Pictures are off", "This model was set up for text only."); continue; }
      if (f.size > 20e6) { toast("warn", "Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    if (!isTextFile(f)) { toast("warn", "Not a text file", `${f.name}: attach text files (code, notes, logs, data)${health.images ? " or pictures" : ""}.`); continue; }
    if (f.size > MAX_TEXT_FILE) { toast("warn", "File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("warn", "Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
// a file's text in the message, fenced with more backticks than it contains itself
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  const files = (m.files || []).filter((f) => f.text != null);
  return [m.text, ...files.map(fileBlock)].filter((s) => s).join("\n\n");
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.innerHTML = "";
  attachments.forEach((a, i) => {
    const c = document.createElement("span");
    c.className = "chip";
    c.innerHTML = icon(a.kind === "file" ? "attach" : "image", "st-icon st-icon--sm");
    c.append(a.name);
    const x = document.createElement("button");
    x.type = "button"; x.className = "st-btn st-btn--icon"; x.setAttribute("aria-label", "Remove");
    x.innerHTML = icon("trash");
    x.onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    c.appendChild(x);
    box.appendChild(c);
  });
}
$("attach-btn").onclick = () => $("file").click();
// drop files on the chat or the message box
for (const id of ["chat", "composer"]) {
  const el = $(id);
  el.addEventListener("dragover", (e) => {
    if (![...(e.dataTransfer || {}).types || []].includes("Files")) return;
    e.preventDefault();
    $("composer").classList.add("dragging");
  });
  el.addEventListener("dragleave", () => $("composer").classList.remove("dragging"));
  el.addEventListener("drop", (e) => {
    $("composer").classList.remove("dragging");
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    addFiles(e.dataTransfer.files);
    $("input").focus();
  });
}
$("file").onchange = () => { addFiles($("file").files); $("file").value = ""; };
$("input").addEventListener("paste", (e) => {
  if (!health.images) return;
  const files = [...(e.clipboardData || {}).files || []].filter((f) => f.type.startsWith("image/"));
  if (files.length) { e.preventDefault(); addFiles(files); }
});

// ------------------------------------------------------------------ the sampling drawer
function openDrawer(open) {
  $("drawer").dataset.open = String(open);
  $("drawer").setAttribute("aria-hidden", String(!open));
  $("scrim").hidden = !open;
  if (open) { loadDrawer(); loadShared(); loadMcp(); }
}
// the model's reasoning effort as a number (DeepSeek-V4.1: health.effort_max): a slider beside the words, which set it
const EFFORT_OF = {low: 50, medium: 62, high: 75};
function effortShown() { return !!(health && health.effort_max); }
function loadDrawer(s = settings) {
  for (const b of $("s-thinking").children) b.setAttribute("aria-checked", String(b.dataset.v === s.thinking));
  $("effort-row").hidden = !effortShown();
  $("s-effort").value = typeof s.thinking === "number" ? s.thinking : (EFFORT_OF[s.thinking] || 75);
  $("s-effort").disabled = s.thinking === "none";
  $("s-temp").value = s.temperature; $("s-topp").value = s.top_p; $("s-topk").value = s.top_k;
  $("s-max").value = s.max; $("s-seed").value = s.seed;
  $("s-show").setAttribute("aria-checked", String(!!s.show));
  $("s-esp").setAttribute("aria-checked", String(s.esp !== false));
  $("esp-row").hidden = !projectionLoaded();
  $("s-mcp").setAttribute("aria-checked", String(s.mcp !== false));
  $("s-share").setAttribute("aria-checked", String(sharedOn));
  outputs();
}
// "Use for other apps too": the server keeps these settings as every client's defaults (GET/POST /settings)
let sharedOn = false;
async function loadShared() {
  try {
    const r = await fetch("settings", {headers: headers()});
    if (r.ok) sharedOn = !!(await r.json()).shared;
  } catch (e) { /* an older server: the switch just stays off */ }
  $("s-share").setAttribute("aria-checked", String(sharedOn));
}
function sharedDefaults(s) {
  const d = {reasoning_effort: s.thinking};
  if (!(+s.temperature > 0)) d.temperature = 0;
  else if (!samplingUntouched(s)) Object.assign(d, {temperature: +s.temperature, top_p: +s.top_p, top_k: +s.top_k});
  if (s.seed) d.seed = +s.seed;
  if (s.max) d.max_tokens = +s.max;
  if (projectionLoaded()) d.experimental_speed_projection = s.esp !== false;
  return d;
}
async function saveShared(on, s) {
  const r = await fetch("settings", {method: "POST", headers: headers(true),
                                      body: JSON.stringify({defaults: on ? sharedDefaults(s) : null})});
  if (!r.ok) {
    let msg = `HTTP ${r.status}`;
    try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  sharedOn = !!(await r.json()).shared;
}
// the engine was started with the experimental-speed-projection control vector (INFO cvec=...)
function projectionLoaded() {
  const c = lastMetrics && lastMetrics.engine ? lastMetrics.engine.cvec : 0;
  return !!c && c !== "0";
}
function outputs() {
  const t = +$("s-temp").value;
  $("o-temp").textContent = t === 0 ? "0 · greedy" : t.toFixed(2);
  $("o-topp").textContent = (+$("s-topp").value).toFixed(2);
  $("o-topk").textContent = $("s-topk").value;
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  const word = sel ? {none: "answers right away", low: "short", medium: "medium", high: "thorough (default)"}[sel.dataset.v] : "";
  $("o-thinking").textContent = !effortShown() || (sel && sel.dataset.v === "none") ? word
                                    : `${word ? word + " · " : ""}effort ${$("s-effort").value}`;
  for (const id of ["s-topp", "s-topk"]) $(id).disabled = t === 0;
}
for (const b of $("s-thinking").children) b.onclick = () => {
  for (const x of $("s-thinking").children) x.setAttribute("aria-checked", String(x === b));
  if (EFFORT_OF[b.dataset.v]) $("s-effort").value = EFFORT_OF[b.dataset.v];   // (the words' numbers on the slider)
  $("s-effort").disabled = b.dataset.v === "none";
  outputs();
};
$("s-effort").oninput = () => {   // (a number of one's own: no word is selected then)
  for (const x of $("s-thinking").children) x.setAttribute("aria-checked", "false");
  outputs();
};
for (const id of ["s-temp", "s-topp", "s-topk"]) $(id).oninput = outputs;
$("s-show").onclick = () => $("s-show").setAttribute("aria-checked", String($("s-show").getAttribute("aria-checked") !== "true"));
$("s-esp").onclick = () => $("s-esp").setAttribute("aria-checked", String($("s-esp").getAttribute("aria-checked") !== "true"));
$("s-mcp").onclick = () => $("s-mcp").setAttribute("aria-checked", String($("s-mcp").getAttribute("aria-checked") !== "true"));
$("s-share").onclick = () => $("s-share").setAttribute("aria-checked", String($("s-share").getAttribute("aria-checked") !== "true"));
$("s-reset").onclick = () => loadDrawer(DEFAULTS);
$("s-apply").onclick = async () => {
  const sel = [...$("s-thinking").children].find((b) => b.getAttribute("aria-checked") === "true");
  settings = {thinking: sel ? sel.dataset.v : effortShown() ? +$("s-effort").value : "high", temperature: +$("s-temp").value, top_p: +$("s-topp").value,
              top_k: +$("s-topk").value, max: $("s-max").value.trim(), seed: $("s-seed").value.trim(),
              show: $("s-show").getAttribute("aria-checked") === "true",
              esp: $("s-esp").getAttribute("aria-checked") === "true",
              mcp: $("s-mcp").getAttribute("aria-checked") === "true"};
  store.set("sampling", settings);
  const share = $("s-share").getAttribute("aria-checked") === "true";
  openDrawer(false);
  if (share || sharedOn) {
    try {
      await saveShared(share, settings);
      toast("success", "Sampling saved", share ? "Other apps (omp, API clients) use these settings from their next request."
                                               : "Other apps use their own settings again.");
    } catch (e) {
      toast("error", "Saved here, but not for other apps", e.message, 6000);
    }
    return;
  }
  toast("success", "Sampling saved", settings.temperature === 0 ? "Greedy: the same question gives the same answer." : "");
};
$("sampling-btn").onclick = () => openDrawer(true);
$("drawer-close").onclick = () => openDrawer(false);
$("scrim").onclick = () => openDrawer(false);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && $("drawer").dataset.open === "true") openDrawer(false); });

// ------------------------------------------------------------------ start
setBusy(false);
renderChat();
const startQuestion = new URLSearchParams(location.search).get("q");   // /?q=... starts a chat (a shortcut)
if (startQuestion) history.replaceState(null, "", location.pathname + location.hash);
loadHealth().then(loadMcp).then(() => { if (startQuestion) { $("input").value = startQuestion; send(); } });
showTab(location.hash.slice(1) || "chat");
poll();

// ------------------------------------------------------------------ images (POST /v1/images/generations)
$("img-form").onsubmit = async (ev) => {
  ev.preventDefault();
  const prompt = $("img-prompt").value.trim();
  if (!prompt) return;
  const body = {prompt, size: $("img-size").value, response_format: "b64_json"};
  if ($("img-seed").value !== "") body.seed = Number($("img-seed").value);
  $("img-go").disabled = true;
  imageBusy = true;
  paintLive();
  const t0 = performance.now();
  $("img-hint").textContent = "Generating\u2026 (a minute or more on a small GPU)";
  try {
    const r = await fetch("v1/images/generations", {method: "POST", headers: headers(true), body: JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || r.statusText);
    for (const d of j.data) {
      const fig = document.createElement("figure");
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + d.b64_json;
      img.alt = prompt;
      const cap = document.createElement("figcaption");
      cap.textContent = `${j.size}, seed ${d.seed}, ${((performance.now() - t0) / 1000).toFixed(0)} s: ${prompt}`;
      fig.append(img, cap);
      $("img-results").prepend(fig);
    }
    $("img-hint").textContent = "Done. Right-click an image to save it.";
  } catch (e) {
    $("img-hint").textContent = "";
    toast("error", "No image", String(e.message || e), 8000);
  } finally {
    $("img-go").disabled = false;
    imageBusy = false;
    paintLive();
  }
};
