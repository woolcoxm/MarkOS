/* MarkOS web UI — vanilla JS, no framework, no CDN. Hash-routed SPA over the
   /api control plane. */
"use strict";

const $ = (sel, el = document) => el.querySelector(sel);
const view = $("#view");
let STATE = null;        // /api/state payload
let currentHash = "";

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function fmtBytes(b) {
  if (b == null) return "–";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"]; let i = 0; let v = b;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return i === 0 ? `${b} B` : `${v.toFixed(1)} ${u[i]}`;
}
function toast(msg, isErr) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.toggle("err", !!isErr);
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), 4200);
}

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    credentials: "same-origin",
    ...opts,
  });
  if (r.status === 401 && !path.startsWith("/api/login")) {
    showLogin(true);
    throw new Error("login required");
  }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error?.message || `HTTP ${r.status}`);
  return body;
}

/* ---------------- auth ---------------- */
function showLogin(show) {
  $("#login-overlay").classList.toggle("hidden", !show);
  if (show) $("#login-form").username.focus();
}
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  $("#login-err").textContent = "";
  try {
    await api("/api/login", { method: "POST", body: JSON.stringify({
      username: e.target.username.value, password: e.target.password.value }) });
    showLogin(false);
    await refreshState();
    route();
  } catch (err) {
    $("#login-err").textContent = err.message;
  }
});
async function logout() {
  await api("/api/logout", { method: "POST" }).catch(() => {});
  STATE = null;
  showLogin(true);
}

/* ---------------- state ---------------- */
async function refreshState() {
  try {
    STATE = await api("/api/state");
    const sys = STATE.system || {};
    $("#hdr-status").textContent =
      `${(STATE.metrics?.tokps_10s ?? 0).toFixed(1)} tok/s · ` +
      `${fmtBytes(sys.avail_mem)} free · ${sys.soc_temp_c != null ? sys.soc_temp_c.toFixed(0) + "°C" : "–"}` +
      (sys.throttling ? " · THROTTLING" : "");
    $("#hdr-status").classList.toggle("warn", !!sys.throttling);
    $("#hdr-user").innerHTML = "";
  } catch (e) { /* 401 handled */ }
}

/* ---------------- routing ---------------- */
const routes = { models: renderModels, monitor: renderMonitor, network: renderNetwork, users: renderUsers, system: renderSystem };
window.addEventListener("hashchange", route);
function route() {
  if (!STATE) return;
  const h = (location.hash.replace(/^#\/?/, "") || "models").split("/")[0];
  if (!routes[h]) location.hash = "#models";
  currentHash = h;
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === `#${h}`));
  routes[h]();
}

/* ---------------- models view ---------------- */
function modelBadge(m) {
  if (!m.on_disk) return `<span class="badge err">missing file</span>`;
  if (m.resident) return `<span class="badge ok">resident</span>`;
  const d = STATE.downloads[m.id];
  if (d && d.state === "running") return `<span class="badge idle">downloading…</span>`;
  if (d && d.state === "error") return `<span class="badge err" title="${esc(d.error)}">download failed</span>`;
  return `<span class="badge idle">on disk</span>`;
}

function renderModels() {
  const ms = STATE.models || [];
  view.innerHTML = `
  <h1>Models</h1>
  <div class="card">
    <h2>Add model</h2>
    <div class="grid2">
      <div>
        <label>HuggingFace GGUF repo <input id="add-repo" placeholder="bartowski/Llama-3.2-1B-Instruct-GGUF"></label>
        <label>…or direct URL <input id="add-url" placeholder="https://…/model.Q4_K_M.gguf"></label>
      </div>
      <div>
        <label>Quantization <select id="add-quant">
          <option>Q4_K_M</option><option>Q5_K_M</option><option>Q6_K</option><option>Q8_0</option>
          <option>Q4_K_S</option><option>Q3_K_M</option><option>F16</option><option value="">(in URL)</option>
        </select></label>
        <label class="inline"><input type="checkbox" id="add-autoload"> load automatically when it fits</label>
        <div class="row"><button class="btn primary" id="add-btn">Register &amp; download</button></div>
      </div>
    </div>
    <p class="muted">Disk usage: <span class="mono">${fmtBytes(STATE.disk?.models_bytes)}</span> · free: <span class="mono">${fmtBytes(STATE.disk?.data_free)}</span></p>
  </div>
  <div id="model-list">${ms.length ? ms.map(modelCard).join("") : `<div class="card muted">No models configured yet.</div>`}</div>`;
  $("#add-btn").onclick = addModel;
  ms.forEach(m => bindModelCard(m));
}

function modelCard(m) {
  return `
  <div class="card" id="model-${esc(m.id)}">
    <div class="model-head" onclick="this.nextElementSibling.classList.toggle('hidden')">
      <h3>${esc(m.name || m.id)}</h3>
      <span class="mono muted">${esc(m.quant)}</span>
      <span class="mono muted">${fmtBytes(m.size_bytes)}</span>
      ${modelBadge(m)}
      <span class="spacer"></span>
      <span class="muted">ctx ${m.n_ctx} · ${m.threads} thr · batch ${m.n_batch} · kv ${esc(m.kv_quant)}</span>
      <span class="muted">▾</span>
    </div>
    <div class="hidden">
      <div class="row" style="margin:8px 0">
        <button class="btn small ${m.resident ? "" : "primary"}" data-act="load" data-id="${esc(m.id)}">${m.resident ? "Reload" : "Load"}</button>
        <button class="btn small" data-act="unload" data-id="${esc(m.id)}" ${m.resident ? "" : "disabled"}>Unload</button>
        <button class="btn small" data-act="estimate" data-id="${esc(m.id)}">Estimate RAM</button>
        <button class="btn small danger" data-act="delete" data-id="${esc(m.id)}">Delete</button>
        <span class="spacer"></span>
        <label class="inline"><input type="checkbox" data-act="autoload" data-id="${esc(m.id)}" ${m.auto_load ? "checked" : ""}> auto-load at boot</label>
      </div>
      ${modelEditor(m)}
    </div>
  </div>`;
}

function modelEditor(m) {
  const profs = (m.profiles || []).map(p => `
    <fieldset data-prof="${esc(p.name)}">
      <legend>${esc(p.name)}${p.name === m.default_profile ? " (default)" : ""}</legend>
      <div class="grid2">
        <div>
          <label>Temperature <input data-p="temperature" type="number" step="0.05" min="0" max="2" value="${p.sampling.temperature}"></label>
          <label>top_p <input data-p="top_p" type="number" step="0.01" min="0" max="1" value="${p.sampling.top_p}"></label>
          <label>top_k <input data-p="top_k" type="number" min="0" value="${p.sampling.top_k}"></label>
          <label>min_p <input data-p="min_p" type="number" step="0.01" min="0" max="1" value="${p.sampling.min_p}"></label>
          <label>Repeat penalty <input data-p="repeat_penalty" type="number" step="0.01" min="1" max="2" value="${p.sampling.repeat_penalty}"></label>
        </div>
        <div>
          <label>Presence penalty <input data-p="presence_penalty" type="number" step="0.1" value="${p.sampling.presence_penalty}"></label>
          <label>Frequency penalty <input data-p="frequency_penalty" type="number" step="0.1" value="${p.sampling.frequency_penalty}"></label>
          <label>Mirostat <select data-p="mirostat">
            <option value="0" ${p.sampling.mirostat == 0 ? "selected" : ""}>off</option>
            <option value="1" ${p.sampling.mirostat == 1 ? "selected" : ""}>v1</option>
            <option value="2" ${p.sampling.mirostat == 2 ? "selected" : ""}>v2</option>
          </select></label>
          <label>Mirostat τ / η <span class="row">
            <input data-p="mirostat_tau" type="number" step="0.1" value="${p.sampling.mirostat_tau}">
            <input data-p="mirostat_eta" type="number" step="0.01" value="${p.sampling.mirostat_eta}">
          </span></label>
          <label>Max tokens / stops <span class="row">
            <input data-p="max_tokens" type="number" min="1" value="${p.max_tokens}">
            <input data-p="stop_csv" placeholder="stops, comma-sep" value="${esc((p.stop || []).join(","))}">
          </span></label>
        </div>
      </div>
    </fieldset>`).join("");
  return `
  <div class="grid2">
    <div>
      <fieldset><legend>Runtime</legend>
        <div class="row">
          <label style="flex:1">Context length <input data-m="n_ctx" type="number" min="128" step="128" value="${m.n_ctx}"></label>
          <label style="flex:1">Threads <input data-m="threads" type="number" min="1" max="8" value="${m.threads}"></label>
          <label style="flex:1">Batch <input data-m="n_batch" type="number" min="32" step="32" value="${m.n_batch}"></label>
          <label style="flex:1">KV cache <select data-m="kv_quant">
            <option ${m.kv_quant === "F16" ? "selected" : ""}>F16</option>
            <option ${m.kv_quant === "Q8_0" ? "selected" : ""}>Q8_0</option>
            <option ${m.kv_quant === "Q4_0" ? "selected" : ""}>Q4_0</option>
          </select></label>
        </div>
        <label>System prompt (default) <textarea data-m="system_prompt" rows="2">${esc(m.system_prompt || "")}</textarea></label>
        <label>Default profile
          <select data-m="default_profile">
            ${(m.profiles || []).map(p => `<option ${p.name === m.default_profile ? "selected" : ""}>${esc(p.name)}</option>`).join("")}
          </select>
        </label>
      </fieldset>
      <div class="row"><button class="btn primary small" data-act="save" data-id="${esc(m.id)}">Save runtime &amp; profiles</button>
      <span class="muted" id="guard-${esc(m.id)}"></span></div>
    </div>
    <div>
      <fieldset><legend>Chat template (Jinja2-style, overrides GGUF)</legend>
        <textarea data-m="chat_template" rows="7" placeholder="{% for m in messages %}…{% endfor %}">${esc(m.chat_template || "")}</textarea>
        <p class="muted">Leave empty to use the template embedded in the GGUF. Context vars: <code>messages</code> (role/content), <code>add_generation_prompt</code>.</p>
      </fieldset>
    </div>
  </div>
  <details><summary>Profiles (${(m.profiles || []).length}) — switch behavior without re-typing sampling params</summary>${profs}</details>`;
}

function bindModelCard(m) {
  const card = $(`#model-${CSS.escape(m.id)}`);
  if (!card) return;
  card.querySelectorAll("[data-act]").forEach(el => {
    el.addEventListener("click", () => modelAction(el.dataset.act, el.dataset.id, el));
    if (el.dataset.act === "autoload") el.addEventListener("change", () => modelAction("autoload", el.dataset.id, el));
  });
}

async function addModel() {
  const repo = $("#add-repo").value.trim();
  const url = $("#add-url").value.trim();
  const quant = $("#add-quant").value;
  try {
    const r = await api("/api/models", { method: "POST", body: JSON.stringify({
      repo: repo || null, url: url || null, quant: quant || null,
    }) });
    toast(`Registered ${r.id}; download started`);
    await refreshState(); renderModels();
  } catch (e) { toast(e.message, true); }
}

async function modelAction(act, id, el) {
  const m = STATE.models.find(x => x.id === id);
  try {
    if (act === "load") {
      el.disabled = true; el.textContent = "Loading…";
      // Guardrail preview first; block with numbers if it won't fit.
      const inv = await api("/api/models/estimate", { method: "POST", body: JSON.stringify({ id }) });
      if (!inv.fits) {
        throw new Error(`Guardrail: needs ${fmtBytes(inv.estimate.total_bytes)}, only ${fmtBytes(inv.available_bytes)} free — reduce context or free a resident model`);
      }
      await api(`/api/models/${id}/load`, { method: "POST" });
      toast(`${id} loaded`);
    } else if (act === "unload") {
      await api(`/api/models/${id}/unload`, { method: "POST" });
      toast(`${id} unloaded`);
    } else if (act === "delete") {
      if (!confirm(`Delete ${id} (config + GGUF file)?`)) return;
      await api(`/api/models/${id}`, { method: "DELETE" });
      toast(`${id} deleted`);
    } else if (act === "estimate") {
      const inv = await api("/api/models/estimate", { method: "POST", body: JSON.stringify({ id }) });
      const e = inv.estimate;
      const pct = Math.min(100, Math.round(e.total_bytes / inv.available_bytes * 100));
      toast(`${id} @ ctx ${inv.n_ctx}: ${fmtBytes(e.total_bytes)} of ${fmtBytes(inv.available_bytes)} free (${pct}%) — ${inv.fits ? "fits" : "DOES NOT FIT"}`, !inv.fits);
      return;
    } else if (act === "save" || act === "autoload") {
      const card = $(`#model-${CSS.escape(id)}`);
      const g = (sel) => card.querySelector(`[data-m="${sel}"]`);
      const profiles = (m.profiles || []).map(p => {
        const fs = card.querySelector(`fieldset[data-prof="${CSS.escape(p.name)}"]`);
        if (!fs) return p;
        const gp = (k) => fs.querySelector(`[data-p="${k}"]`)?.value;
        return { ...p, sampling: { ...p.sampling,
          temperature: +gp("temperature"), top_p: +gp("top_p"), top_k: +gp("top_k"),
          min_p: +gp("min_p"), repeat_penalty: +gp("repeat_penalty"),
          presence_penalty: +gp("presence_penalty"), frequency_penalty: +gp("frequency_penalty"),
          mirostat: +gp("mirostat"), mirostat_tau: +gp("mirostat_tau"), mirostat_eta: +gp("mirostat_eta"),
        }, max_tokens: +gp("max_tokens"),
           stop: gp("stop_csv").split(",").map(s => s.trim()).filter(Boolean) };
      });
      const body = {
        ...m,
        n_ctx: +g("n_ctx").value, threads: +g("threads").value, n_batch: +g("n_batch").value,
        kv_quant: g("kv_quant").value, chat_template: g("chat_template").value || null,
        system_prompt: g("system_prompt").value || null,
        default_profile: g("default_profile").value,
        profiles, auto_load: act === "autoload" ? el.checked : m.auto_load,
      };
      const r = await api(`/api/models/${id}/config`, { method: "POST", body: JSON.stringify(body) });
      // Guardrail preview for the new context length, surfaced inline.
      const inv = await api("/api/models/estimate", { method: "POST", body: JSON.stringify({ id }) });
      const note = $("#guard-" + CSS.escape(id));
      note.innerHTML = inv.fits
        ? `<span class="ok mono">est. ${fmtBytes(inv.estimate.total_bytes)} / ${fmtBytes(inv.available_bytes)} — fits</span>`
        : `<span class="err mono">est. ${fmtBytes(inv.estimate.total_bytes)} / ${fmtBytes(inv.available_bytes)} — WILL NOT FIT, reload will be refused</span>`;
      toast(r.note || "saved");
    }
    await refreshState(); renderModels();
  } catch (e) { toast(e.message, true); if (el) el.disabled = false; }
}

/* ---------------- monitor ---------------- */
let monitorEs = null;
function renderMonitor() {
  view.innerHTML = `
  <h1>Monitor</h1>
  <div class="stat-grid" id="mon-stats"></div>
  <div class="card"><h2>Request log (live)</h2><div id="mon-log" class="mono muted">waiting…</div></div>
  <div class="card"><h2>Resident slots</h2><div id="mon-slots"></div></div>`;
  drawMonitor(STATE);
  if (monitorEs) monitorEs.close();
  monitorEs = new EventSource("/api/metrics/stream");
  monitorEs.addEventListener("metrics", async (ev) => {
    const p = JSON.parse(ev.data);
    drawMonitor({ metrics: p.metrics, system: p.system, slots: p.slots, models: STATE.models, downloads: STATE.downloads });
    if (currentHash !== "monitor") { monitorEs.close(); monitorEs = null; }
  });
}
function drawMonitor({ metrics, system, slots }) {
  if (!metrics || currentHash !== "monitor") return;
  const s = system || {};
  const temp = s.soc_temp_c != null ? s.soc_temp_c.toFixed(1) + " °C" : "–";
  const risk = s.throttling ? `<span class="badge err">throttling</span>` :
    (s.soc_temp_c >= 70 ? `<span class="badge warn">hot — check cooling</span>` : `<span class="badge ok">nominal</span>`);
  $("#mon-stats").innerHTML = `
    <div class="stat"><div class="v">${(metrics.tokps_1s ?? 0).toFixed(1)}</div><div class="k">tok/s (1s)</div></div>
    <div class="stat"><div class="v">${(metrics.tokps_60s ?? 0).toFixed(1)}</div><div class="k">tok/s (60s)</div></div>
    <div class="stat"><div class="v">${metrics.active}</div><div class="k">active</div></div>
    <div class="stat"><div class="v">${metrics.queued}</div><div class="k">queued</div></div>
    <div class="stat"><div class="v">${fmtBytes(s.avail_mem)}</div><div class="k">RAM free</div></div>
    <div class="stat"><div class="v">${(s.load1 ?? 0).toFixed(2)}</div><div class="k">load (1m)</div></div>
    <div class="stat"><div class="v">${temp}</div><div class="k">SoC ${risk}</div></div>
    <div class="stat"><div class="v">${metrics.tokens_out}</div><div class="k">tokens total</div></div>
    <div class="stat"><div class="v">${metrics.models_loaded}/${STATE.engine?.max_resident ?? 2}</div><div class="k">models resident</div></div>`;
  fetch("/api/log?tail=40", { credentials: "same-origin" }).then(r => r.json()).then(j => {
    $("#mon-log").innerHTML = (j.entries || []).map(e => {
      const t = new Date(e.ts).toLocaleTimeString();
      const tp = e.tokps != null ? ` · ${e.tokps.toFixed(1)} tok/s` : "";
      return `<div>${t} [${e.status}] ${esc(e.kind)} ${esc(e.path || e.model || "")} ${tp} ${esc(e.message || "")}</div>`;
    }).join("") || "no events yet";
  }).catch(() => {});
  $("#mon-slots").innerHTML = `<table><tr><th>model</th><th>state</th><th>busy</th><th>idle</th></tr>` +
    (slots || []).map(x => `<tr><td class="mono">${esc(x.model_id)}</td><td>${esc(x.state)}</td>
      <td>${x.busy ? '<span class="badge warn">generating</span>' : '<span class="badge idle">idle</span>'}</td>
      <td class="mono">${x.idle_for_s}s</td></tr>`).join("") + "</table>";
}

/* ---------------- network ---------------- */
function renderNetwork() {
  const s = STATE.server;
  view.innerHTML = `
  <h1>Network &amp; exposure</h1>
  <div class="card"><h2>Inference API</h2>
    <div class="row">
      <label style="flex:1">Bind address:port <input id="n-api" value="${esc(s.api_bind)}"></label>
      <label class="inline"><input type="checkbox" id="n-key" ${s.api_key_required ? "checked" : ""}> require API key</label>
      <button class="btn small" id="n-setkey">Set API key…</button>
    </div>
    <label>IP allowlist (one per line, empty = allow all; CIDR ok)
      <textarea id="n-allow" rows="3">${esc((s.ip_allowlist || []).join("\n"))}</textarea></label>
  </div>
  <div class="card"><h2>Web UI</h2>
    <div class="row">
      <label style="flex:2">UI binds (one per line; 4444 is the IPv4LL rescue listener — keep it)
        <textarea id="n-ui" rows="3">${esc((s.ui_binds || []).join("\n"))}</textarea></label>
      <label class="inline" style="flex:1"><input type="checkbox" id="n-tls" ${s.tls_enabled ? "checked" : ""}> TLS (self-signed cert on first boot)</label>
    </div>
    <label>mDNS name <input id="n-mdns" value="${esc(s.mdns_name)}"></label>
    <p class="muted">Appliance firewall allows only SSH (if enabled), the API port, the UI ports and mDNS. Discovery: <span class="mono">${esc(s.mdns_name)}.local</span></p>
    <div class="row"><button class="btn primary" id="n-save">Apply network settings</button></div>
  </div>`;
  $("#n-setkey").onclick = async () => {
    const k = prompt("API key (stored hashed; clients send it as Authorization: Bearer …)\nLeave empty to cancel:");
    if (!k) return;
    await api("/api/server/apikey", { method: "POST", body: JSON.stringify({ key: k }) });
    toast("API key set");
  };
  $("#n-save").onclick = async () => {
    try {
      await api("/api/server", { method: "POST", body: JSON.stringify({
        api_bind: $("#n-api").value.trim(),
        ui_binds: $("#n-ui").value.split("\n").map(x => x.trim()).filter(Boolean),
        tls_enabled: $("#n-tls").checked,
        api_key_required: $("#n-key").checked,
        ip_allowlist: $("#n-allow").value.split("\n").map(x => x.trim()).filter(Boolean),
        mdns_name: $("#n-mdns").value.trim() || "pi-inference",
      }) });
      toast("saved — takes effect on restart");
    } catch (e) { toast(e.message, true); }
  };
}

/* ---------------- users ---------------- */
function renderUsers() {
  view.innerHTML = `
  <h1>Users</h1>
  <div class="card"><h2>Accounts</h2>
    <table><tr><th>User</th><th>Role</th><th></th></tr>
    ${STATE.users.map(u => `<tr><td>${esc(u.username)}</td><td>${esc(u.role)}</td>
      <td><button class="btn small danger" data-u="${esc(u.username)}">Remove</button></td></tr>`).join("")}
    </table></div>
  <div class="card"><h2>Add user</h2>
    <div class="row">
      <label>Username <input id="u-name"></label>
      <label>Password <input id="u-pass" type="password" placeholder="≥ 8 chars"></label>
      <label>Role <select id="u-role"><option>Viewer</option><option>Admin</option></select></label>
      <button class="btn primary" id="u-add" style="align-self:end">Add</button>
    </div>
    <p class="muted">Viewers can see models, monitoring and logs; admins can change configuration and manage models.</p>
  </div>`;
  view.querySelectorAll("[data-u]").forEach(b => b.onclick = async () => {
    try { await api(`/api/users/${encodeURIComponent(b.dataset.u)}`, { method: "DELETE" }); await refreshState(); renderUsers(); }
    catch (e) { toast(e.message, true); }
  });
  $("#u-add").onclick = async () => {
    try {
      await api("/api/users", { method: "POST", body: JSON.stringify({
        username: $("#u-name").value.trim(), password: $("#u-pass").value,
        role: $("#u-role").value === "Admin" ? "Admin" : "Viewer" }) });
      toast("user added");
      await refreshState(); renderUsers();
    } catch (e) { toast(e.message, true); }
  };
}

/* ---------------- system ---------------- */
function renderSystem() {
  const sys = STATE.system || {};
  view.innerHTML = `
  <h1>System</h1>
  <div class="grid2">
    <div class="card"><h2>Software</h2>
      <p>MarkOS engine <b class="mono">${esc(STATE.version)}</b> · uptime ${Math.floor(STATE.uptime_s / 3600)}h ${Math.floor(STATE.uptime_s % 3600 / 60)}m</p>
      <p class="muted">${esc(sys.os_name || "")} · ${sys.cores ?? 4} cores</p>
      <div class="row">
        <button class="btn" id="sys-check">Check for updates</button>
        <button class="btn" id="sys-restart">Restart engine</button>
      </div>
      <pre class="output" id="sys-out">Update checks are manual only — nothing phones home.</pre>
    </div>
    <div class="card"><h2>Recovery paths</h2>
      <ul id="sys-recovery" class="muted"><li>loading…</li></ul>
      <p class="muted">If the web UI or network is broken: plug a laptop directly into the Pi's Ethernet port and open <span class="mono">http://169.254.9.1:4444</span>. Hold the factory-reset button (GPIO26) during boot to restore the install-time snapshot. Serial console: UART GPIO14/15 @ 115200.</p>
    </div>
  </div>
  <div class="card"><h2>Guardrails</h2>
    <p class="muted">Total RAM <span class="mono">${fmtBytes(sys.total_mem)}</span> · usable for models <span class="mono">${fmtBytes((sys.total_mem || 0) - (sys.avail_mem || 0) < 0 ? 0 : sys.avail_mem)}</span> free right now.
    The engine refuses model loads that would exceed memory — the UI estimates usage before you apply anything. There is no GPU on this hardware; everything runs on the four A76 cores with NEON.</p>
  </div>`;
  $("#sys-check").onclick = async () => {
    $("#sys-out").textContent = "checking…";
    try {
      const r = await api("/api/system/update-check", { method: "POST", body: "{}" });
      $("#sys-out").textContent = JSON.stringify(r, null, 2);
    } catch (e) { $("#sys-out").textContent = e.message; }
  };
  $("#sys-restart").onclick = async () => {
    if (!confirm("Restart the inference engine now? Active generations will be cut off.")) return;
    try {
      await api("/api/system/restart", { method: "POST" });
      toast("engine restarting…");
    } catch (e) { /* expected: connection dies */ }
  };
  fetch("/api/recovery", { credentials: "same-origin" }).then(r => r.json()).then(r => {
    $("#sys-recovery").innerHTML = Object.entries(r).map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("");
  }).catch(() => {});
}

/* ---------------- boot ---------------- */
(async function init() {
  $("#nav").classList.remove("hidden");
  view.classList.remove("hidden");
  try {
    await refreshState();
    if (!$("#login-overlay").classList.contains("hidden")) showLogin(false);
    route();
  } catch (e) {
    showLogin(true);
  }
})();
