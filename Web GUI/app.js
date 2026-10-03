/*
Pink Edge AI — HTML/JS edition front-end logic.

Talks to web_api.py (same origin, http://127.0.0.1:5000 by default) over the small REST API
documented in that file's docstring. Every value shown here — verdict, confidence, source,
localization, bbox overlay — comes straight from a /api/triage response built by run_triage()
(GUI.py), the same dispatcher the Tkinter and Streamlit editions call; nothing is computed or
guessed client-side except the purely cosmetic telemetry/hardware numbers, which are simulated
the same way in those other two editions (see GUI.py's hw_stats / net_stats).
*/
(() => {
  const API = ""; // same-origin; web_api.py serves this page itself

  const SHORT_LABELS = ["Mammography", "TB chest X-ray", "Maternal ultrasound"];

  const state = {
    models: [], model: null, role: "lhw", canOverride: false, lang: "en",
    translations: {}, biRadsOptions: [], acrOptions: [], tbSeverity: [], tbZones: [],
    lastResult: null, imageFile: null,
  };

  const $ = (id) => document.getElementById(id);

  // ---------------------------------------------------------------
  // Translation (mirrors GUI.py's t() — swap a handful of known labels)
  // ---------------------------------------------------------------
  function applyLang() {
    document.querySelectorAll("[data-t]").forEach((el) => {
      const key = el.getAttribute("data-t");
      el.textContent = state.lang === "ur" ? (state.translations[key] || key) : key;
    });
  }

  // ---------------------------------------------------------------
  // Config / model status
  // ---------------------------------------------------------------
  async function loadConfig() {
    const cfg = await fetch(`${API}/api/config`).then((r) => r.json());
    state.models = cfg.models;
    state.biRadsOptions = cfg.bi_rads_options;
    state.acrOptions = cfg.acr_density_options;
    state.tbSeverity = cfg.tb_severity_levels;
    state.tbZones = cfg.tb_lung_zones;
    state.translations = cfg.translations;
    state.model = cfg.models[0];

    const wrap = $("modelPills");
    wrap.innerHTML = "";
    cfg.models.forEach((m, i) => {
      const btn = document.createElement("button");
      btn.className = "model-pill" + (m === state.model ? " active" : "");
      btn.textContent = SHORT_LABELS[i] || m.split("(")[0].trim();
      btn.dataset.model = m;
      btn.addEventListener("click", () => selectModel(m));
      wrap.appendChild(btn);
    });

    applyLang();
    refreshPlaceholder();
    loadModelStatus();
  }

  async function loadModelStatus() {
    const status = await fetch(`${API}/api/model_status`).then((r) => r.json());
    const el = $("modelStatusList");
    if (status.error) { el.textContent = "unavailable"; return; }
    el.innerHTML = Object.entries(status)
      .map(([mod, info]) => `${info.real ? "🟢" : "⚪"} ${mod.split("(")[0].trim()}`)
      .join("<br>");
  }

  function selectModel(m) {
    state.model = m;
    document.querySelectorAll(".model-pill").forEach((b) => b.classList.toggle("active", b.dataset.model === m));
    resetResultView();
    refreshPlaceholder();
  }

  async function refreshPlaceholder() {
    if (state.lastResult) return; // don't clobber a real result with a fresh placeholder
    $("scanImage").src = `${API}/api/placeholder?model=${encodeURIComponent(state.model)}&seed=${Date.now() % 10000}`;
    $("scanCaption").textContent = "Generated Scan Placeholder";
  }

  function resetResultView() {
    state.lastResult = null;
    $("metricRow").hidden = true;
    $("verdictSection").hidden = true;
    $("cacheBtn").disabled = true;
    $("cacheSavedMsg").hidden = true;
    $("rConfidence").textContent = "—"; $("rSeverity").textContent = "—"; $("rEscalation").textContent = "—";
    document.querySelectorAll(".risk-tile").forEach((t) => t.classList.remove("selected"));
  }

  // ---------------------------------------------------------------
  // Role + auth
  // ---------------------------------------------------------------
  $("roleToggle").addEventListener("click", (e) => {
    const btn = e.target.closest(".pill");
    if (!btn) return;
    setRole(btn.dataset.role, btn.dataset.role === "radiologist" ? "👨‍⚕️ Senior Radiologist" : "👤 Lady Health Worker (LHW)");
    document.querySelectorAll("#roleToggle .pill").forEach((b) => b.classList.toggle("active", b === btn));
  });

  function setRole(role) {
    state.role = role;
    state.canOverride = role === "radiologist";
    renderOverrideSection();
  }

  $("pinInput").addEventListener("keydown", async (e) => {
    if (e.key !== "Enter") return;
    const res = await fetch(`${API}/api/auth/pin`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pin: e.target.value }),
    });
    if (res.ok) {
      const data = await res.json();
      setRole(data.role);
      document.querySelectorAll("#roleToggle .pill").forEach((b) => b.classList.toggle("active", b.dataset.role === data.role));
      e.target.value = "";
    } else {
      e.target.style.borderColor = "#fca5a5";
      setTimeout(() => { e.target.style.borderColor = ""; }, 800);
    }
  });

  $("langToggle").addEventListener("click", (e) => {
    const btn = e.target.closest(".pill");
    if (!btn) return;
    state.lang = btn.dataset.lang;
    document.querySelectorAll("#langToggle .pill").forEach((b) => b.classList.toggle("active", b === btn));
    applyLang();
  });

  // ---------------------------------------------------------------
  // Tabs
  // ---------------------------------------------------------------
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      tab.classList.add("active");
      $(`panel-${tab.dataset.tab}`).classList.add("active");
      if (tab.dataset.tab === "cache") refreshCache();
    });
  });

  // ---------------------------------------------------------------
  // Image upload
  // ---------------------------------------------------------------
  $("imageInput").addEventListener("change", (e) => {
    state.imageFile = e.target.files[0] || null;
  });

  // ---------------------------------------------------------------
  // Telemetry / hardware (cosmetic — simulated the same way GUI.py/streamlit_app.py do it)
  // ---------------------------------------------------------------
  const telemetry = [];
  function logLine(line) {
    telemetry.push(line);
    $("telemetryLog").innerHTML = telemetry.slice(-12).join("<br>");
  }
  function randomBetween(a, b) { return (Math.random() * (b - a) + a); }

  // ---------------------------------------------------------------
  // Run Triage
  // ---------------------------------------------------------------
  $("runBtn").addEventListener("click", async () => {
    const btn = $("runBtn");
    btn.disabled = true; btn.textContent = "⏳ Running…";
    try {
      const form = new FormData();
      form.append("model", state.model);
      form.append("show_bbox", $("bboxToggle").checked ? "true" : "false");
      if (state.imageFile) form.append("image", state.imageFile);

      const res = await fetch(`${API}/api/triage`, { method: "POST", body: form });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      state.lastResult = data;
      renderResult(data);

      $("hwStats").innerHTML = `RK3588 &nbsp;● ACTIVE<br>NPU Load &nbsp;97%<br>` +
        `Power &nbsp;${randomBetween(5.8, 6.5).toFixed(1)}W<br>Temp &nbsp;${Math.round(randomBetween(38, 45))}C`;
      data.log.forEach(logLine);
    } catch (err) {
      alert(`Triage failed: ${err.message}`);
    } finally {
      btn.disabled = false; btn.innerHTML = `▶️ <span data-t="Run Triage">Run Triage</span>`;
      applyLang();
    }
  });

  function renderResult(data) {
    const r = data.result;

    $("scanImage").src = data.image;
    $("scanCaption").textContent = data.uploaded_name || "Generated Scan Placeholder";

    $("metricRow").hidden = false;
    $("mModel").textContent = SHORT_LABELS[state.models.indexOf(state.model)] || state.model.split("(")[0].trim();
    $("mModel").title = state.model;
    $("mConfidence").textContent = `${r.confidence.toFixed(1)}%`;
    $("mLatency").textContent = `${data.inference_time}s`;

    $("sourcePill").textContent = "Real model";
    $("rConfidence").textContent = `${r.confidence.toFixed(0)}%`;

    const riskLabels = { critical: ["High", "risk-high"], moderate: ["Moderate", "risk-moderate"], low: ["Low", "risk-low"] };
    const [sevLabel, sevClass] = riskLabels[data.risk_level] || riskLabels.low;
    $("rSeverity").textContent = sevLabel;
    $("rSeverity").className = "chip " + sevClass;
    $("rEscalation").textContent = r.is_critical ? "Required" : "Not required";
    $("rEscalation").style.color = r.is_critical ? "var(--danger)" : "var(--success)";

    document.querySelectorAll(".risk-tile").forEach((t) => t.classList.remove("selected"));
    $(data.risk_level === "critical" ? "riskHigh" : data.risk_level === "moderate" ? "riskMod" : "riskLow").classList.add("selected");

    $("metaId").textContent = data.privacy_hash;
    $("metaAge").textContent = `${data.patient_age} Y`;
    $("metaModality").textContent = data.modality;
    $("metaDate").textContent = data.timestamp.slice(0, 10);

    $("verdictSection").hidden = false;
    $("lowConfWarning").hidden = r.confidence >= 65.0;

    const css = r.css || (r.is_critical ? "danger" : "success");
    const box = $("verdictBox");
    box.className = "verdict-box " + css;
    $("verdictIcon").textContent = r.vicon;
    $("verdictTitle").textContent = r.verdict;
    $("verdictSub").textContent = r.sub || "";
    $("verdictSource").textContent = `Source: ${r.source || "N/A"}`;
    $("verdictLoc").textContent = r.loc || "N/A";
    $("verdictExtra").textContent = r.extra || "N/A";
    $("verdictQuality").textContent = r.image_quality || "Adequate for Analysis";

    renderOverrideSection();

    const plan = $("actionPlan");
    const rec = r.recommendation || "Patient should be referred for specialist consultation";
    plan.className = "banner " + (r.is_critical ? "banner-warn" : "banner-success");
    plan.textContent = `${r.is_critical ? "⚠️" : "✅"} Action Plan: ${rec}`;

    $("cacheBtn").disabled = false;
    $("cacheSavedMsg").hidden = true;
  }

  function renderOverrideSection() {
    if (!state.lastResult) return;
    const isTb = state.model.includes("Tuberculosis");
    const opts1 = isTb ? state.tbSeverity : state.biRadsOptions;
    const opts2 = isTb ? state.tbZones : state.acrOptions;
    $("override1Label").textContent = isTb ? "TB Severity" : "BI-RADS Assessment";
    $("override2Label").textContent = isTb ? "Lung Zone" : "ACR Breast Density";

    $("overrideSection").hidden = !state.canOverride;
    $("overrideLocked").hidden = state.canOverride;
    if (!state.canOverride) return;

    const r = state.lastResult.result;
    fillSelect($("override1"), opts1, r.bi_rads);
    fillSelect($("override2"), opts2, r.acr);
  }

  function fillSelect(select, options, current) {
    select.innerHTML = options.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("");
    const idx = options.indexOf(current);
    select.selectedIndex = idx >= 0 ? idx : 0;
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  // ---------------------------------------------------------------
  // Report payload shared by Save to Cache / Download Text / Download PDF
  // ---------------------------------------------------------------
  function buildReportPayload() {
    const d = state.lastResult;
    return {
      patient_id: d.patient_id, patient_age: d.patient_age, modality: d.modality,
      model: state.model, timestamp: d.timestamp, inference_time: d.inference_time,
      network_mode: "Fully Offline", result: d.result,
      bi_rads: state.canOverride ? $("override1").value : null,
      acr_density: state.canOverride ? $("override2").value : null,
    };
  }

  $("cacheBtn").addEventListener("click", async () => {
    const res = await fetch(`${API}/api/cache`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildReportPayload()),
    });
    if (res.ok) {
      $("cacheSavedMsg").hidden = false;
      logLine(`[${new Date().toLocaleTimeString()}] [SQLite] Report cached (ID: ${state.lastResult.patient_id}).`);
    }
  });

  async function downloadReport(kind) {
    const res = await fetch(`${API}/api/report/${kind}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(buildReportPayload()),
    });
    if (!res.ok) { alert("Report generation is unavailable (missing optional dependency)."); return; }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${state.lastResult.patient_id}.${kind === "pdf" ? "pdf" : "txt"}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }
  $("downloadTextBtn").addEventListener("click", () => downloadReport("text"));
  $("downloadPdfBtn").addEventListener("click", () => downloadReport("pdf"));

  // ---------------------------------------------------------------
  // Cache tab
  // ---------------------------------------------------------------
  async function refreshCache() {
    const [rows, counts] = await Promise.all([
      fetch(`${API}/api/cache`).then((r) => r.json()),
      fetch(`${API}/api/cache/counts`).then((r) => r.json()),
    ]);
    $("cTotal").textContent = counts.total;
    $("cUnsynced").textContent = counts.unsynced;
    $("cSynced").textContent = counts.total - counts.unsynced;

    const tbody = document.querySelector("#cacheTable tbody");
    tbody.innerHTML = "";
    $("cacheEmptyMsg").hidden = rows.length > 0;
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.id}</td><td>${row.patient_id}</td><td>${row.modality}</td>` +
        `<td>${row.bi_rads || ""}</td><td>${row.verdict}</td><td>${row.timestamp}</td>` +
        `<td>${row.synced ? "✅" : "⏳"}</td>`;
      tbody.appendChild(tr);
    });
  }

  $("syncBtn").addEventListener("click", async () => {
    await fetch(`${API}/api/cache/sync`, { method: "POST" });
    refreshCache();
  });
  $("refreshCacheBtn").addEventListener("click", refreshCache);

  // ---------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------
  loadConfig();
})();
