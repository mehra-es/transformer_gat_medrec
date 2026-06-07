/**
 * Transformer-GAT MedRec Dashboard
 * Clinical decision support UI — not for autonomous prescribing.
 */

const API = "";
const charts = {};

function $(sel) {
  return document.querySelector(sel);
}

function destroyChart(id) {
  if (charts[id]) {
    charts[id].destroy();
    delete charts[id];
  }
}

async function fetchJSON(path) {
  const res = await fetch(`${API}${path}`);
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  return res.json();
}

function patientIndex() {
  return parseInt($("#patient-select").value, 10) || 0;
}

let pipelinePollTimer = null;

function setupNav() {
  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
      btn.classList.add("active");
      $(`#view-${btn.dataset.view}`).classList.add("active");
      if (btn.dataset.view === "metrics") loadMetrics();
      if (btn.dataset.view === "pipeline") loadPipeline();
    });
  });
}

function updateStatusBadge(status) {
  const el = $("#status-badge");
  if (status.model_loaded) {
    el.textContent = `Model ready · ${status.device}`;
    el.className = "badge badge-ok";
  } else {
    el.textContent = "No checkpoint — train first";
    el.className = "badge badge-warn";
  }
}

async function loadPatients() {
  const { patients } = await fetchJSON("/api/patients");
  const sel = $("#patient-select");
  sel.innerHTML = "";
  patients.forEach((p) => {
    const opt = document.createElement("option");
    opt.value = p.index;
    opt.textContent = `${p.patient_id} (#${p.index})`;
    sel.appendChild(opt);
  });
}

function renderVisitTable(visits) {
  const tbody = $("#visit-table tbody");
  tbody.innerHTML = "";
  visits.forEach((v) => {
    const tr = document.createElement("tr");
    const ids = `Dx: ${v.diagnosis_ids.join(",") || "—"} · Rx: ${v.medication_ids.join(",") || "—"}`;
    tr.innerHTML = `
      <td>${v.visit}</td>
      <td>${v.num_diagnoses}</td>
      <td>${v.num_medications}</td>
      <td>${v.lab_mean.toFixed(2)}</td>
      <td><code>${ids}</code></td>
    `;
    tbody.appendChild(tr);
  });
}

function chartVisits(visits) {
  destroyChart("chart-visits");
  const labels = visits.map((v) => `V${v.visit}`);
  charts["chart-visits"] = new Chart($("#chart-visits"), {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Diagnoses",
          data: visits.map((v) => v.num_diagnoses),
          backgroundColor: "rgba(3, 105, 161, 0.7)",
          borderRadius: 4,
        },
        {
          label: "Medications",
          data: visits.map((v) => v.num_medications),
          backgroundColor: "rgba(13, 148, 136, 0.7)",
          borderRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "bottom" } },
      scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
    },
  });
}

function chartLabs(visits) {
  destroyChart("chart-labs");
  charts["chart-labs"] = new Chart($("#chart-labs"), {
    type: "line",
    data: {
      labels: visits.map((v) => `V${v.visit}`),
      datasets: [
        {
          label: "Lab mean",
          data: visits.map((v) => v.lab_mean),
          borderColor: "#0369a1",
          backgroundColor: "rgba(3, 105, 161, 0.1)",
          fill: true,
          tension: 0.35,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

async function loadPatient() {
  const idx = patientIndex();
  const data = await fetchJSON(`/api/patient/${idx}`);
  $("#patient-meta").textContent = `${data.patient_id} · ${data.num_visits} visits · ground-truth meds: [${data.ground_truth_meds.join(", ") || "none"}]`;
  renderVisitTable(data.visits);
  chartVisits(data.visits);
  chartLabs(data.visits);
}

function renderRecKpis(pred) {
  const overlap = pred.overlap.length;
  const gt = pred.ground_truth.length;
  const jacc = gt + pred.predicted_at_threshold.length - overlap > 0
    ? (overlap / (gt + pred.predicted_at_threshold.length - overlap)).toFixed(3)
    : "—";
  $("#rec-kpis").innerHTML = `
    <div class="kpi"><div class="label">Predicted (≥ threshold)</div><div class="value">${pred.predicted_at_threshold.length}</div></div>
    <div class="kpi"><div class="label">Ground truth</div><div class="value">${gt}</div></div>
    <div class="kpi"><div class="label">Overlap</div><div class="value good">${overlap}</div></div>
    <div class="kpi"><div class="label">Approx. Jaccard</div><div class="value">${jacc}</div></div>
    <div class="kpi"><div class="label">Threshold</div><div class="value">${pred.threshold}</div></div>
  `;
}

function chartDrugs(recs) {
  destroyChart("chart-drugs");
  const top = recs.slice(0, 12);
  charts["chart-drugs"] = new Chart($("#chart-drugs"), {
    type: "bar",
    data: {
      labels: top.map((r) => `Drug ${r.drug_id}`),
      datasets: [
        {
          label: "Probability",
          data: top.map((r) => r.probability),
          backgroundColor: top.map((r) =>
            r.selected ? "rgba(13, 148, 136, 0.85)" : "rgba(148, 163, 184, 0.5)"
          ),
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { x: { max: 1, min: 0 } },
    },
  });
}

function renderRecTable(recs) {
  const tbody = $("#rec-table tbody");
  tbody.innerHTML = "";
  recs.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${r.drug_id}</code></td>
      <td>${(r.probability * 100).toFixed(1)}%</td>
      <td>${r.selected ? "✓ Yes" : "—"}</td>
    `;
    tbody.appendChild(tr);
  });
}

async function loadPredictions() {
  const idx = patientIndex();
  const pred = await fetchJSON(`/api/predict/${idx}?top_k=20`);
  renderRecKpis(pred);
  chartDrugs(pred.recommendations);
  renderRecTable(pred.recommendations);
  chartFusion(pred.fusion);
  renderDDI(pred.ddi_pairs_in_prediction);
  return pred;
}

function chartFusion(fusion) {
  destroyChart("chart-fusion");
  const pw = Math.round(fusion.patient_weight * 100);
  const dw = Math.round(fusion.drug_weight * 100);
  charts["chart-fusion"] = new Chart($("#chart-fusion"), {
    type: "doughnut",
    data: {
      labels: ["Patient (Transformer)", "Drug graph (GAT)"],
      datasets: [
        {
          data: [pw, dw],
          backgroundColor: ["#0d9488", "#0369a1"],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
      },
    },
  });
}

function renderDDI(pairs) {
  const ul = $("#ddi-list");
  ul.innerHTML = "";
  if (!pairs.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No DDI edges among predicted drug pairs at current threshold.";
    ul.appendChild(li);
    return;
  }
  pairs.forEach((p) => {
    const li = document.createElement("li");
    li.textContent = `Drug ${p.drug_a} ↔ Drug ${p.drug_b} · severity ${p.severity.toFixed(2)}`;
    ul.appendChild(li);
  });
}

async function runExplain() {
  const idx = patientIndex();
  const loading = $("#explain-loading");
  const btn = $("#btn-explain");
  loading.classList.remove("hidden");
  btn.disabled = true;
  try {
    const data = await fetchJSON(`/api/explain/${idx}`);
    const feats = data.top_shap_features || [];
    destroyChart("chart-shap");
    charts["chart-shap"] = new Chart($("#chart-shap"), {
      type: "bar",
      data: {
        labels: feats.map((f) => f.name),
        datasets: [
          {
            label: "Mean |SHAP|",
            data: feats.map((f) => f.shap_importance),
            backgroundColor: "rgba(13, 148, 136, 0.75)",
            borderRadius: 4,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: { legend: { display: false } },
      },
    });
    const tags = $("#explain-drugs");
    tags.innerHTML = "";
    (data.recommended_drugs || []).slice(0, 8).forEach((d) => {
      const prob = data.probabilities?.[d] ?? data.probabilities?.[String(d)];
      const span = document.createElement("span");
      span.className = "tag";
      span.textContent = `Drug ${d}: ${prob != null ? (prob * 100).toFixed(1) + "%" : "—"}`;
      tags.appendChild(span);
    });
  } catch (e) {
    alert(`SHAP failed: ${e.message}`);
  } finally {
    loading.classList.add("hidden");
    btn.disabled = false;
  }
}

async function loadMetrics() {
  const wrap = $("#metrics-kpis");
  wrap.innerHTML = "<p class='lead'>Loading metrics…</p>";
  try {
    const { metrics, error } = await fetchJSON("/api/metrics");
    if (error || !metrics) {
      wrap.innerHTML = `<p class="lead">${error || "Metrics unavailable"}</p>`;
      return;
    }
    const items = [
      ["Jaccard", metrics.jaccard, "good"],
      ["F1 (micro)", metrics.f1_micro, ""],
      ["F1 (macro)", metrics.f1_macro, ""],
      ["PRAUC", metrics.prauc, "good"],
      ["DDI rate", metrics.ddi_rate, metrics.ddi_rate > 0.1 ? "warn" : ""],
    ];
    wrap.innerHTML = items
      .map(
        ([label, val, cls]) => `
      <div class="kpi">
        <div class="label">${label}</div>
        <div class="value ${cls}">${typeof val === "number" ? val.toFixed(4) : val}</div>
      </div>`
      )
      .join("");

    destroyChart("chart-metrics");
    charts["chart-metrics"] = new Chart($("#chart-metrics"), {
      type: "radar",
      data: {
        labels: ["Jaccard", "F1 μ", "F1 M", "PRAUC", "1 − DDI"],
        datasets: [
          {
            label: "Test set",
            data: [
              metrics.jaccard,
              metrics.f1_micro,
              metrics.f1_macro,
              metrics.prauc,
              1 - metrics.ddi_rate,
            ],
            backgroundColor: "rgba(13, 148, 136, 0.2)",
            borderColor: "#0d9488",
            pointBackgroundColor: "#0d9488",
          },
        ],
      },
      options: {
        responsive: true,
        scales: { r: { min: 0, max: 1 } },
      },
    });
  } catch (e) {
    wrap.innerHTML = `<p class="lead">Error: ${e.message}</p>`;
  }
}

async function refreshAll() {
  await loadPatient();
  await loadPredictions();
}

async function postJSON(path, body = {}) {
  const res = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || data.error || res.statusText);
  return data;
}

function formatTime(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleTimeString();
}

function renderPipelineSteps(data) {
  const order = data.step_order || [];
  const byId = Object.fromEntries(data.steps.map((s) => [s.id, s]));
  const ol = $("#pipeline-steps");
  ol.innerHTML = "";
  order.forEach((id, i) => {
    const s = byId[id];
    if (!s) return;
    const li = document.createElement("li");
    li.className = `pipeline-step ${s.status}`;
    const canRun = id !== "explore" && !data.busy;
    li.innerHTML = `
      <div class="step-num">${i + 1}</div>
      <div class="step-body">
        <h4>${s.title}</h4>
        <p>${s.description}</p>
        <code class="step-cmd">${s.command}</code>
        <div class="step-meta">${s.last_message || ""} ${s.duration_sec != null ? ` · ${s.duration_sec}s` : ""} ${s.last_run_at ? ` · ${formatTime(s.last_run_at)}` : ""}</div>
      </div>
      <div class="step-actions">
        <span class="status-pill ${s.status}">${s.status}</span>
        ${id === "explore"
          ? `<button class="btn btn-secondary btn-sm" type="button" data-goto="overview">Open overview</button>`
          : `<button class="btn btn-secondary btn-sm" type="button" data-run="${id}" ${canRun ? "" : "disabled"}>Run</button>`
        }
      </div>
    `;
    ol.appendChild(li);
  });

  ol.querySelectorAll("[data-run]").forEach((btn) => {
    btn.addEventListener("click", () => runPipelineStep(btn.dataset.run));
  });
  ol.querySelectorAll("[data-goto]").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelector('.nav-item[data-view="overview"]').click();
    });
  });

  const sel = $("#log-step-select");
  const prev = sel.value;
  sel.innerHTML = order
    .filter((id) => id !== "explore")
    .map((id) => `<option value="${id}">${byId[id].title}</option>`)
    .join("");
  if (data.current_job && data.current_job !== "all") {
    sel.value = data.current_job;
  } else if (prev && [...sel.options].some((o) => o.value === prev)) {
    sel.value = prev;
  }
}

function renderSetupBanner(data) {
  const el = $("#pipeline-setup-banner");
  const s = data.setup || {};
  if (s.ready) {
    el.className = "setup-banner ok";
    el.textContent = "Environment ready: virtualenv and core dependencies (PyTorch, PyG) are installed.";
  } else {
    el.className = "setup-banner warn";
    el.textContent =
      "Setup not complete — run Step 1 (Environment setup) or ./run.sh setup before training.";
  }
}

async function loadPipelineLog(stepId) {
  try {
    const { log } = await fetchJSON(`/api/pipeline/logs/${stepId}?tail=400`);
    $("#pipeline-log").textContent = log || "(no output yet)";
    const pre = $("#pipeline-log");
    pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    $("#pipeline-log").textContent = `Log error: ${e.message}`;
  }
}

async function loadPipeline() {
  const data = await fetchJSON("/api/pipeline");
  renderSetupBanner(data);
  renderPipelineSteps(data);
  const busyEl = $("#pipeline-busy");
  if (data.busy) {
    busyEl.classList.remove("hidden");
    document.querySelectorAll("[data-run]").forEach((b) => (b.disabled = true));
    $("#btn-run-all").disabled = true;
  } else {
    busyEl.classList.add("hidden");
    $("#btn-run-all").disabled = false;
  }
  const logStep = $("#log-step-select").value || data.current_job || "train";
  await loadPipelineLog(data.busy && data.current_job && data.current_job !== "all" ? data.current_job : logStep);

  if (data.busy && !pipelinePollTimer) {
    pipelinePollTimer = setInterval(pipelinePoll, 2000);
  }
  if (!data.busy && pipelinePollTimer) {
    clearInterval(pipelinePollTimer);
    pipelinePollTimer = null;
    const status = await fetchJSON("/api/status");
    updateStatusBadge(status);
    await loadPatients();
    await refreshAll();
  }
}

async function pipelinePoll() {
  try {
    await loadPipeline();
  } catch (e) {
    console.warn("pipeline poll", e);
  }
}

async function runPipelineStep(stepId) {
  const skip = $("#chk-skip-train").checked;
  try {
    await postJSON(`/api/pipeline/run/${stepId}`, {
      skip_train_if_checkpoint: skip,
      patient_idx: patientIndex(),
    });
    $("#log-step-select").value = stepId;
    pipelinePollTimer = setInterval(pipelinePoll, 1500);
    await loadPipeline();
  } catch (e) {
    alert(`Could not start ${stepId}: ${e.message}`);
  }
}

async function runPipelineAll() {
  const skip = $("#chk-skip-train").checked;
  try {
    await postJSON("/api/pipeline/run/all", { skip_train_if_checkpoint: skip });
    pipelinePollTimer = setInterval(pipelinePoll, 1500);
    await loadPipeline();
  } catch (e) {
    alert(`Could not start pipeline: ${e.message}`);
  }
}

async function cancelPipeline() {
  await postJSON("/api/pipeline/cancel");
  await loadPipeline();
}

async function reloadModel() {
  const status = await postJSON("/api/pipeline/reload-model");
  updateStatusBadge(status);
  await loadPatients();
  await refreshAll();
  alert("Model reloaded from checkpoint.");
}

function setupPipelineUI() {
  $("#btn-run-all").addEventListener("click", runPipelineAll);
  $("#btn-cancel-pipeline").addEventListener("click", cancelPipeline);
  $("#btn-reload-model").addEventListener("click", reloadModel);
  $("#log-step-select").addEventListener("change", () => {
    loadPipelineLog($("#log-step-select").value);
  });
}

async function init() {
  setupNav();
  setupPipelineUI();
  const status = await fetchJSON("/api/status");
  updateStatusBadge(status);
  await loadPatients();
  $("#patient-select").addEventListener("change", refreshAll);
  $("#btn-refresh").addEventListener("click", refreshAll);
  $("#btn-explain").addEventListener("click", runExplain);
  await loadPipeline();
  await refreshAll();
}

init().catch((e) => {
  console.error(e);
  $("#status-badge").textContent = "API offline";
  alert("Could not connect to API. Start the server with: .venv/bin/python ui/server.py");
});
