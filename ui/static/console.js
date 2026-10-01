const KIND_COLOR = {
  event: "#7aa2ff",
  venue: "#3ee0c6",
  athlete: "#e0b15a",
  games: "#c4a0ff",
  sport: "#6ec8e8",
  document: "#9aacbf",
  chunk: "#6d7f93",
  fact: "#d7c39a",
  entity: "#8ea0b5",
};

const HISTORY_KEY = "og-console-history";
const MAX_HISTORY = 20;

const state = {
  bootstrap: null,
  view: "investigate",
  pipeline: "agentic_graphrag",
  investigation: null,
  compare: null,
  selectedEvidence: null,
  graph: { scale: 1, x: 0, y: 0 },
  busy: false,
  startedAt: 0,
  timer: null,
  controller: null,
  history: loadHistory(),
};

async function boot() {
  const res = await fetch("/api/bootstrap");
  state.bootstrap = await res.json();
  renderShell();
  renderBenchmark();
  renderSystem();
  renderHistory();
  renderReports();
  loadHealth();
  renderInvestigate();
}

function renderShell() {
  const caps = (state.bootstrap && state.bootstrap.capabilities) || {};
  const mode = caps.mode === "live" ? "live" : "unavailable";
  document.getElementById("top-caps").innerHTML = [
    capPill("RAG", caps.rag),
    capPill("Graph", caps.graphrag),
    capPill(caps.generator === "semantic" ? "Semantic" : "Deterministic", caps.llm_configured || caps.generator === "deterministic"),
    `<span class="chip ${mode}">${mode === "live" ? "LIVE" : "UNAVAILABLE"}</span>`,
  ].join("");
  document.getElementById("runtime-mode").textContent = mode === "live" ? "LIVE" : "UNAVAILABLE";
  document.getElementById("runtime-mode").className = `chip ${mode}`;
  const example = (state.bootstrap && state.bootstrap.example_placeholder) || "";
  document.getElementById("example-placeholder").textContent = example
    ? `Example shape (type your own question): ${example}`
    : "Type any question. This console executes the real backend.";
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => showView(tab.dataset.view));
  });
  document.getElementById("run-btn").addEventListener("click", () => runCurrent());
  document.getElementById("cancel-btn").addEventListener("click", cancelRun);
  document.getElementById("clear-btn").addEventListener("click", clearWorkspace);
  document.getElementById("copy-btn").addEventListener("click", () => copyPayload(exportInvestigation()));
  document.getElementById("download-btn").addEventListener("click", () => downloadPayload(exportInvestigation(), "investigation"));
  document.getElementById("copy-compare-btn").addEventListener("click", () => copyPayload(exportComparison()));
  document.getElementById("download-compare-btn").addEventListener("click", () => downloadPayload(exportComparison(), "comparison"));
  document.getElementById("history-clear").addEventListener("click", () => {
    state.history = [];
    saveHistory();
    renderHistory();
  });
  document.getElementById("graph-reset").addEventListener("click", () => {
    state.graph = { scale: 1, x: 0, y: 0 };
    renderGraph();
  });
}

function capPill(label, ok) {
  return `<span class="pill ${ok ? "ok" : "warn"}">${esc(label)} ${ok ? "ready" : "offline"}</span>`;
}

function selectedPipeline() {
  const picked = document.querySelector("input[name=pipeline]:checked");
  return picked ? picked.value : "agentic_graphrag";
}

function showView(name) {
  state.view = name;
  document.querySelectorAll(".view").forEach((node) => {
    const on = node.id === `view-${name}`;
    node.classList.toggle("active", on);
    node.hidden = !on;
  });
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.view === name));
  if (name === "system") loadHealth();
  if (name === "history") renderHistory();
  if (name === "reports") renderReports();
  if (name === "compare") renderCompare();
}

function currentQuestion() {
  return document.getElementById("question-input").value.trim();
}

function setBusy(busy, label) {
  state.busy = busy;
  document.getElementById("run-btn").disabled = busy;
  document.getElementById("cancel-btn").disabled = !busy;
  const runState = document.getElementById("run-state");
  runState.textContent = label || (busy ? "RUNNING" : "Idle");
  runState.className = `chip ${busy ? "running" : "mute"}`;
  if (busy) {
    state.startedAt = Date.now();
    tickElapsed();
    state.timer = setInterval(tickElapsed, 200);
    document.getElementById("answer-text").textContent = "Investigation in progress…";
    document.getElementById("answer-text").classList.add("placeholder");
    document.getElementById("answer-status").textContent = "RUNNING";
    document.getElementById("answer-status").className = "chip running";
    document.getElementById("timeline").innerHTML = `<li>Investigation running — waiting for the live backend.</li>`;
    document.getElementById("step-count").textContent = "running";
  } else if (state.timer) {
    clearInterval(state.timer);
    state.timer = null;
  }
}

function tickElapsed() {
  const seconds = state.startedAt ? (Date.now() - state.startedAt) / 1000 : 0;
  document.getElementById("elapsed").textContent = `${seconds.toFixed(1)}s`;
}

function cancelRun() {
  if (state.controller) state.controller.abort();
}

function clearWorkspace() {
  state.investigation = null;
  state.compare = null;
  state.selectedEvidence = null;
  document.getElementById("question-input").value = "";
  banner("", "");
  const caps = (state.bootstrap && state.bootstrap.capabilities) || {};
  const mode = caps.mode === "live" ? "live" : "unavailable";
  document.getElementById("runtime-mode").textContent = mode === "live" ? "LIVE" : "UNAVAILABLE";
  document.getElementById("runtime-mode").className = `chip ${mode}`;
  renderInvestigate();
  renderCompare();
  renderReports();
}

async function runCurrent() {
  const question = currentQuestion();
  if (!question) {
    banner("Enter a question.", "warn");
    return;
  }
  const pipeline = selectedPipeline();
  if (pipeline === "compare") {
    await runCompare(question);
    return;
  }
  await runInvestigate(question, pipeline);
}

async function runInvestigate(question, pipeline) {
  setBusy(true, `RUNNING ${pipelineLabel(pipeline)}`);
  banner(`Live investigation · ${pipelineLabel(pipeline)}`, "");
  state.controller = new AbortController();
  try {
    const res = await fetch("/api/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, pipeline }),
      signal: state.controller.signal,
    });
    const payload = await res.json();
    applyInvestigation(payload);
    const failed = !!(payload.errors && payload.errors.length) || payload.status === "unavailable" || payload.status === "error";
    banner(failed ? (payload.warnings || payload.errors || []).join(" ") : "Live result", failed ? "warn" : "");
  } catch (err) {
    if (err.name === "AbortError") {
      banner("Request stopped in the browser. The backend call may still finish.", "warn");
    } else {
      banner(`Request failed: ${err.message}`, "danger");
    }
  } finally {
    state.controller = null;
    setBusy(false, "Idle");
  }
}

async function runCompare(question) {
  setBusy(true, "RUNNING COMPARE ALL");
  banner("Running RAG, Fixed GraphRAG, and Agentic GraphRAG…", "");
  state.controller = new AbortController();
  try {
    const res = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: state.controller.signal,
    });
    const payload = await res.json();
    state.compare = payload;
    const preferred =
      (payload.results || []).find((row) => row.pipeline === "agentic_graphrag") || (payload.results || [])[0];
    if (preferred) applyInvestigation(preferred, { skipHistory: true });
    remember({
      question,
      pipeline: "compare",
      status: "compare",
      status_label: "COMPARE",
      latency_ms: Math.max(0, ...((payload.comparison || []).map((row) => Number(row.latency_ms) || 0))),
      investigation: preferred,
      compare: payload,
    });
    renderCompare();
    const failed = (payload.results || []).some((row) => (row.errors || []).length || row.status === "unavailable");
    banner(failed ? "Compare-all finished with pipeline errors or unavailable backends." : "Compare-all live results", failed ? "warn" : "");
    showView("compare");
  } catch (err) {
    if (err.name === "AbortError") banner("Compare stopped in the browser.", "warn");
    else banner(`Compare failed: ${err.message}`, "danger");
  } finally {
    state.controller = null;
    setBusy(false, "Idle");
  }
}

function applyInvestigation(payload, opts) {
  state.investigation = payload;
  state.selectedEvidence = null;
  renderInvestigate();
  renderReports();
  if (opts && opts.replay) {
    banner("Replaying a session investigation stored in this browser tab.", "");
  }
  if (!opts || !opts.skipHistory) {
    remember({
      question: payload.question,
      pipeline: payload.pipeline,
      status: payload.status,
      status_label: payload.status_label,
      latency_ms: payload.latency_ms,
      investigation: payload,
    });
  }
}

function banner(text, kind) {
  const node = document.getElementById("run-banner");
  node.textContent = text || "";
  node.classList.toggle("hidden", !text);
  node.className = `banner ${kind || "mute"}${text ? "" : " hidden"}`;
}

function renderInvestigate() {
  const inv = state.investigation;
  const empty = !inv;
  document.getElementById("asked-question").textContent = empty ? "" : inv.question || "";
  const status = empty ? "" : inv.status || "";
  const statusEl = document.getElementById("answer-status");
  statusEl.textContent = empty ? "—" : inv.status_label || status || "—";
  statusEl.className = `chip ${status || "mute"}`;
  const mode = empty ? "" : inv.runtime_mode || "";
  if (mode) {
    document.getElementById("runtime-mode").textContent = inv.runtime_mode_label || mode.toUpperCase();
    document.getElementById("runtime-mode").className = `chip ${mode}`;
  }
  const answerText = displayAnswer(inv);
  const answerEl = document.getElementById("answer-text");
  answerEl.textContent = answerText;
  answerEl.classList.toggle("placeholder", empty || answerText !== (inv && inv.answer));
  document.getElementById("step-count").textContent = `${((inv && inv.steps) || []).length} steps`;
  document.getElementById("timeline").innerHTML = empty
    ? `<li>Run an investigation to see observed steps.</li>`
    : (inv.steps || [])
        .map(
          (step) => `<li class="${esc(step.kind || "")}">
        <strong>Step ${step.index}</strong> · ${esc(step.kind)} · ${esc(step.action)}
        <div>${esc(step.reason || "")}</div>
        <div class="meta">
          <span>${esc(step.status || "—")}</span>
          <span>${fmtMs(step.elapsed_ms)}</span>
          <span>${esc(step.result_summary || `${step.evidence_count || 0} evidence`)}</span>
          <span>${esc(step.retrieval_method || "")}</span>
        </div>
        ${step.continued && step.why_next ? `<div class="next">Next step triggered: ${esc(step.why_next)}</div>` : ""}
      </li>`
        )
        .join("") || emptyTrace(inv);
  const cites = empty ? [] : inv.citations || [];
  document.getElementById("citations").innerHTML = cites.length
    ? cites
        .map((item) => `<button type="button" class="cite" data-eid="${esc(item.evidence_id || "")}">${esc(item.evidence_id || "citation")}</button>`)
        .join("")
    : `<span class="hint">${empty ? "" : "No citations."}</span>`;
  document.querySelectorAll(".cite").forEach((btn) => {
    btn.addEventListener("click", () => {
      state.selectedEvidence = btn.dataset.eid;
      renderLineage();
    });
  });
  renderEvidenceStatus(inv);
  const continueBits = empty ? [] : [].concat(inv.why_continued || []).concat(inv.continue_explanation || []);
  document.getElementById("continue-reason").textContent = continueCopy(inv, continueBits);
  document.getElementById("stop-reason").textContent = empty ? "—" : inv.stop_explanation || inv.stop_reason || stopFallback(inv);
  renderMetrics("efficiency", empty ? null : inv.efficiency, [
    ["Tokens", (row) => (row.tokens_unknown ? "unknown" : row.tokens)],
    ["Model calls", (row) => row.model_calls],
    ["Tool calls", (row) => row.tool_calls],
    ["Steps", (row) => row.steps],
    ["Latency", (row) => fmtMs(row.latency_ms)],
    ["Retries", (row) => row.retries],
    ["Methods", (row) => (row.retrieval_methods || []).join(", ") || "—"],
  ]);
  renderMetrics("reliability", empty ? null : inv.reliability, [
    ["Status", (row) => nonempty(row.status_label) ? row.status_label : (inv && inv.status_label) || row.status || "—"],
    ["Attempts", (row) => row.attempts],
    ["Retries", (row) => row.retries],
    ["Failure class", (row) => row.failure_class || "none"],
    ["Tokens unknown", (row) => (row.tokens_unknown ? "yes" : "no")],
    ["Timeout", (row) => (row.timeout ? "yes" : "no")],
    ["Unavailable", (row) => (row.unavailable || (inv && inv.status === "unavailable") ? "yes" : "no")],
    ["Malformed generation", (row) => (row.malformed ? "yes" : "no")],
  ]);
  renderDiff(inv);
  renderLineage();
  renderGraph();
}

function renderEvidenceStatus(inv) {
  const box = document.getElementById("evidence-status");
  if (!inv) {
    box.innerHTML = "";
    return;
  }
  const st = inv.evidence_status || {};
  box.innerHTML = [
    chip("Evidence retrieved", st.retrieved ?? 0),
    chip("Evidence used", st.used ?? 0),
    chip("Citations", st.citation_count ?? 0),
    chip("Graph context", st.graph_present ? "present" : "not present"),
    chip("Methods", (st.retrieval_methods || []).join(", ") || "none"),
  ].join("");
}

function chip(label, value) {
  return `<span class="chip">${esc(label)}: ${esc(value)}</span>`;
}

function renderMetrics(id, row, fields) {
  const node = document.getElementById(id);
  if (!row) {
    node.innerHTML = `<div class="metric-tile"><span class="k">State</span><span class="v">No investigation yet</span></div>`;
    return;
  }
  node.innerHTML = fields
    .map(([label, fn]) => {
      const value = fn(row);
      return `<div class="metric-tile"><span class="k">${esc(label)}</span><span class="v">${esc(displayMetric(value))}</span></div>`;
    })
    .join("");
}

function displayMetric(value) {
  if (value == null || value === "") return "—";
  return value;
}

function nonempty(value) {
  return value && value !== "—";
}

function displayAnswer(inv) {
  if (!inv) return "Enter a question and run an investigation.";
  if (inv.answer_display) return inv.answer_display;
  const answer = String(inv.answer || "").trim();
  if (answer) return answer;
  if (inv.status === "unavailable") return "No answer was produced because this pipeline is unavailable in the current runtime.";
  if (inv.status === "error") return "No answer was produced because the backend failed.";
  if (inv.status === "timeout") return "No answer was produced because the request timed out.";
  if (inv.status === "generation_error") return "No answer was produced because generation failed.";
  if (inv.status) return "The pipeline finished without answer text.";
  return "Enter a question and run an investigation.";
}

function continueCopy(inv, bits) {
  if (!inv) return "—";
  const found = (bits || []).find(Boolean);
  if (found) return found;
  if (inv.status === "unavailable") return "The investigation did not continue because the backend was unavailable.";
  if (inv.status === "error") return "The investigation did not continue because the backend failed.";
  return "No follow-up retrieval was observed.";
}

function stopFallback(inv) {
  if (!inv) return "—";
  if (inv.status === "unavailable") return "Investigation stopped because the backend was unavailable.";
  if (inv.status === "error") return "Investigation stopped because the backend failed.";
  if (inv.status === "timeout") return "Investigation stopped because the request timed out.";
  return "No stop reason was recorded for this result.";
}

function emptyTrace(inv) {
  if (inv && inv.status === "unavailable") {
    return `<li>No investigation steps were observed because the backend was unavailable.</li>`;
  }
  if (inv && (inv.status === "error" || inv.status === "timeout")) {
    return `<li>No investigation steps were observed because the backend did not complete.</li>`;
  }
  return `<li>No observed retrieval or generation steps in this result.</li>`;
}

function renderDiff(inv) {
  const node = document.getElementById("evidence-diff");
  const diff = inv && inv.evidence_diff;
  if (!diff || !(diff.steps || []).length) {
    const unavailable = inv && (inv.status === "unavailable" || inv.status === "error");
    const message = unavailable
      ? "No evidence delta because the backend did not run."
      : inv && inv.pipeline === "agentic_graphrag"
        ? "No per-tool evidence delta in this trace."
        : "Evidence diff is shown for Agentic GraphRAG when tool observations exist.";
    node.innerHTML = `<p class="hint">${message}</p>`;
    return;
  }
  node.innerHTML = diff.steps
    .map(
      (step) => `<article class="delta">
      <strong>After ${esc(step.tool || "tool")} · step ${step.index}</strong>
      <div class="hint">${esc(step.reason || "")}</div>
      <div class="cols">
        <div><h3>Before</h3><p>${(step.before || []).map(esc).join("<br>") || "none"}</p></div>
        <div><h3>New evidence</h3><p>${(step.added || []).map(esc).join("<br>") || "none"}</p></div>
        <div><h3>Resolved</h3><p>${(step.resolved || []).map(esc).join(", ") || "—"}</p></div>
        <div><h3>Remaining</h3><p>${(step.remaining || []).map(esc).join(", ") || "—"}</p></div>
      </div>
    </article>`
    )
    .join("");
}

function renderLineage() {
  const node = document.getElementById("lineage");
  const inv = state.investigation;
  if (!inv) {
    node.innerHTML = `<p class="hint">Citations expand to chunk, document, and graph identifiers.</p>`;
    return;
  }
  const rows = inv.lineage || [];
  if (!rows.length) {
    const unavailable = inv.status === "unavailable" || inv.status === "error";
    node.innerHTML = `<p class="hint">${unavailable ? "No evidence lineage because the backend did not return evidence." : "No retrieved evidence in this result."}</p>`;
    return;
  }
  node.innerHTML = rows
    .map((row) => {
      const open = row.evidence_id === state.selectedEvidence ? " open" : "";
      const refs = (row.graph_refs || []).map((ref) => `${ref.vertex_type || ""} ${ref.vertex_id || ""} ${ref.edge_type || ""}`.trim()).join(" · ");
      return `<details${open}>
        <summary>${esc(row.claim_ref ? "Cited" : "Retrieved")} · ${esc(row.evidence_id || "")}</summary>
        <dl class="metrics stacked">
          <dt>Type</dt><dd>${esc(row.evidence_type)}</dd>
          <dt>Document</dt><dd>${esc(row.document_id || "—")}</dd>
          <dt>Chunk</dt><dd>${esc(row.chunk_id || "—")}</dd>
          <dt>Event</dt><dd>${esc(row.event_id || "—")}</dd>
          <dt>Method</dt><dd>${esc(row.retrieval_method || "—")}</dd>
          <dt>Value</dt><dd>${esc(row.value || "—")}</dd>
        </dl>
        ${row.text ? `<pre>${esc(row.text)}</pre>` : ""}
        <p class="hint">${refs ? `Graph refs: ${esc(refs)}` : "No typed graph refs on this item."}</p>
      </details>`;
    })
    .join("");
}

function renderGraph() {
  const svg = document.getElementById("graph");
  const inv = state.investigation;
  const graph = (inv && inv.graph) || { nodes: [], edges: [] };
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const kinds = nodes.length ? Array.from(new Set(nodes.map((node) => node.kind || "entity"))) : [];
  document.getElementById("graph-legend").innerHTML = kinds.length
    ? kinds
        .map((kind) => `<span><i style="background:${KIND_COLOR[kind] || "#8ea0b5"}"></i>${esc(kind)}</span>`)
        .join("")
    : "";
  const note = document.getElementById("graph-note");
  if (!inv) {
    note.textContent = "";
    svg.innerHTML = `<text x="24" y="40" fill="#8ea0b5">Run an investigation to load graph evidence.</text>`;
    return;
  }
  if (inv.status === "unavailable" || inv.status === "error") {
    note.textContent = "";
    svg.innerHTML = `<text x="24" y="40" fill="#d98c4a">${inv.status === "error" ? "Graph data was not produced because the backend failed." : "Graph backend unavailable."}</text>`;
    return;
  }
  note.textContent = graph.truncated ? `Showing ${nodes.length} of ${graph.node_count} nodes (${graph.edge_count} edges total).` : "";
  if (!nodes.length) {
    svg.innerHTML = `<text x="24" y="40" fill="#8ea0b5">No graph evidence in this result.</text>`;
    return;
  }
  const layers = ["event", "venue", "games", "sport", "athlete", "fact", "document", "chunk", "entity"];
  const grouped = {};
  nodes.forEach((node) => {
    const kind = layers.includes(node.kind) ? node.kind : "entity";
    grouped[kind] = grouped[kind] || [];
    grouped[kind].push(node);
  });
  const positioned = [];
  let col = 0;
  layers.forEach((kind) => {
    const bucket = grouped[kind] || [];
    if (!bucket.length) return;
    bucket.forEach((node, index) => {
      positioned.push({ ...node, x: 90 + col * 150, y: 70 + index * 70 });
    });
    col += 1;
  });
  const byId = Object.fromEntries(positioned.map((node) => [node.id, node]));
  svg.innerHTML = `<g transform="translate(${state.graph.x} ${state.graph.y}) scale(${state.graph.scale})">
    ${edges
      .map((edge) => {
        const a = byId[edge.source];
        const b = byId[edge.target];
        if (!a || !b) return "";
        const cited = edge.evidence_id && edge.evidence_id === state.selectedEvidence;
        return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${cited ? "#3ee0c6" : "#243041"}" stroke-width="${cited ? 2 : 1}" />
          <title>${esc(edge.label)}</title>
          <text x="${(a.x + b.x) / 2}" y="${(a.y + b.y) / 2 - 6}" fill="#8ea0b5" font-size="10">${esc(edge.label)}</text>`;
      })
      .join("")}
    ${positioned
      .map((node) => {
        const selected = node.evidence_id === state.selectedEvidence;
        const color = KIND_COLOR[node.kind] || "#8ea0b5";
        const label = (node.label || node.id).slice(0, 28);
        return `<g class="gnode" data-eid="${esc(node.evidence_id || "")}" data-id="${esc(node.id)}" style="cursor:pointer">
          <title>${esc(node.label || node.id)}</title>
          <circle cx="${node.x}" cy="${node.y}" r="${selected || node.cited ? 16 : 12}" fill="${color}" stroke="${selected ? "#e7eef6" : "transparent"}" stroke-width="2" />
          <text x="${node.x + 20}" y="${node.y + 4}" fill="#e7eef6" font-size="11">${esc(label)}</text>
        </g>`;
      })
      .join("")}
  </g>`;
  svg.querySelectorAll(".gnode").forEach((node) => {
    node.addEventListener("click", (event) => {
      event.stopPropagation();
      state.selectedEvidence = node.dataset.eid || node.dataset.id;
      renderLineage();
      renderGraph();
    });
  });
  enablePanZoom(svg);
}

function enablePanZoom(svg) {
  if (svg.dataset.bound === "1") return;
  svg.dataset.bound = "1";
  let dragging = false;
  let last = { x: 0, y: 0 };
  svg.addEventListener("pointerdown", (event) => {
    dragging = true;
    last = { x: event.clientX, y: event.clientY };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointerup", () => {
    dragging = false;
  });
  svg.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    state.graph.x += event.clientX - last.x;
    state.graph.y += event.clientY - last.y;
    last = { x: event.clientX, y: event.clientY };
    renderGraph();
  });
  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 1.08 : 0.92;
      state.graph.scale = Math.min(2.4, Math.max(0.4, state.graph.scale * delta));
      renderGraph();
    },
    { passive: false }
  );
}

function renderCompare() {
  const table = document.getElementById("compare-table");
  const changed = document.getElementById("what-changed");
  const vs = document.getElementById("fixed-vs-agentic");
  const rows = (state.compare && state.compare.comparison) || [];
  if (!rows.length) {
    table.innerHTML = `<p class="hint">Run Compare All from Investigate. The same live question is executed on all three pipelines.</p>`;
    changed.innerHTML = "";
    vs.innerHTML = "";
    return;
  }
  table.innerHTML = `<table><thead><tr>
    <th>Pipeline</th><th>Status</th><th>Answer</th><th>Latency</th><th>Tokens</th><th>Citations</th><th>Evidence</th><th>Model calls</th><th>Tool calls</th><th>Methods</th>
  </tr></thead><tbody>${rows
    .map(
      (row) => `<tr class="compare-pick" data-pipeline="${esc(row.pipeline)}">
        <td>${esc(row.pipeline_label)}</td>
        <td>${esc(row.status_label || row.status || "—")}</td>
        <td>${esc(row.answer_display || row.answer || displayAnswer(row))}</td>
        <td>${fmtMs(row.latency_ms)}</td>
        <td>${row.tokens_unknown ? "unknown" : row.tokens}</td>
        <td>${row.citation_count}</td>
        <td>${row.evidence_count}</td>
        <td>${row.model_calls}</td>
        <td>${row.tool_calls}</td>
        <td>${esc((row.retrieval_methods || []).join(", ") || "—")}</td>
      </tr>`
    )
    .join("")}</tbody></table>`;
  table.querySelectorAll(".compare-pick").forEach((row) => {
    row.addEventListener("click", () => {
      const found = (state.compare.results || []).find((item) => item.pipeline === row.dataset.pipeline);
      if (!found) return;
      applyInvestigation(found, { skipHistory: true });
      showView("investigate");
    });
  });
  const facts = state.compare.what_changed || [];
  changed.innerHTML = facts.length ? `<h2>What changed?</h2><ul>${facts.map((line) => `<li>${esc(line)}</li>`).join("")}</ul>` : "";
  const pair = state.compare.fixed_vs_agentic;
  if (!pair) {
    vs.innerHTML = "";
    return;
  }
  vs.innerHTML = [pair.graphrag, pair.agentic_graphrag]
    .filter(Boolean)
    .map(
      (card) => `<article class="vs-card">
      <h2>${esc(card.pipeline_label)}</h2>
      <dl class="metrics stacked">
        <dt>Path</dt><dd>${esc(card.retrieval_path)}</dd>
        <dt>Evidence</dt><dd>${card.evidence_count}</dd>
        <dt>Citations</dt><dd>${card.citation_count}</dd>
        <dt>Steps / tools</dt><dd>${card.steps} / ${card.tool_calls}</dd>
        <dt>Tokens</dt><dd>${card.tokens_unknown ? "unknown" : card.tokens}</dd>
        <dt>Model calls</dt><dd>${card.model_calls}</dd>
        <dt>Latency</dt><dd>${fmtMs(card.latency_ms)}</dd>
        <dt>Status</dt><dd>${esc(card.status_label)}</dd>
      </dl>
      <p class="hint">${esc(card.continue_explanation || card.stop_explanation || "")}</p>
    </article>`
    )
    .join("");
}

function renderBenchmark() {
  const bench = state.bootstrap.benchmark;
  if (!bench) {
    document.getElementById("canon-label").textContent = "Published benchmark data is not available in this build.";
    return;
  }
  document.getElementById("canon-label").textContent = `${bench.label} · ${bench.population || ""}`;
  const order = ["rag", "graphrag", "agentic_graphrag"];
  document.getElementById("headline-cards").innerHTML = order
    .map((id) => {
      const row = bench.pipelines[id];
      return `<article class="card ${id}" aria-label="${esc(row.name)} ${fmtPct(row.correctness)}"><div class="name">${esc(row.name)}</div><div class="score">${fmtPct(row.correctness)}</div><div class="hint">${esc(row.subtitle)}</div></article>`;
    })
    .join("");
  const metrics = ["correctness", "exact", "completeness", "grounding", "citation_validity", "errors", "tokens", "latency_mean_ms", "latency_p50_ms", "latency_p95_ms"];
  const labels = {
    correctness: "Accuracy",
    exact: "Exact",
    completeness: "Completeness",
    grounding: "Grounding",
    citation_validity: "Citation validity",
    errors: "Errors",
    tokens: "Tokens",
    latency_mean_ms: "Mean latency",
    latency_p50_ms: "p50 latency",
    latency_p95_ms: "p95 latency",
  };
  const head = `<tr><th>Metric</th>${order.map((id) => `<th>${esc(bench.pipelines[id].name)}</th>`).join("")}</tr>`;
  const body = metrics
    .map((metric) => {
      const cells = order.map((id) => `<td>${fmtMetric(bench.pipelines[id][metric], metric, bench)}</td>`).join("");
      return `<tr><th>${labels[metric]}</th>${cells}</tr>`;
    })
    .join("");
  document.getElementById("metric-table").innerHTML = `<h2>${esc(bench.label)}</h2><table>${head}${body}</table><p class="hint">${esc(Object.values(bench.notes || {}).join(" "))}</p>`;
  document.getElementById("chart-correctness").innerHTML = barChart(
    order.map((id) => ({ label: bench.pipelines[id].name, value: bench.pipelines[id].correctness })),
    { unit: "%", max: 100 }
  );
  document.getElementById("chart-family").innerHTML = groupedBars(
    bench.families.map((row) => ({
      label: `${row.label || row.id} (n=${row.n})`,
      values: [row.rag, row.graphrag, row.agentic_graphrag],
    })),
    ["RAG", "GraphRAG", "Agentic"]
  );
  if ((bench.unpublished_metrics || []).includes("tokens") || order.every((id) => bench.pipelines[id].tokens == null)) {
    document.getElementById("chart-tokens").innerHTML = `<p class="hint">${esc(bench.notes.tokens || "Token totals were not published for this public run.")}</p>`;
  }
  document.getElementById("chart-latency").innerHTML = groupedBars(
    order.map((id) => ({
      label: bench.pipelines[id].name,
      values: [bench.pipelines[id].latency_mean_ms, bench.pipelines[id].latency_p95_ms],
    })),
    ["Mean ms", "p95 ms"]
  );
  const agent = bench.agent_behavior;
  document.getElementById("chart-agent").innerHTML =
    barChart(
      [
        { label: "One call", value: agent.one_call },
        { label: "Follow-ups", value: agent.follow_ups },
        { label: "Neighborhood", value: agent.neighborhood },
      ],
      { unit: "q", max: 100 }
    ) + `<p class="hint">Agentic public set: avg ${agent.avg_tool_calls} tool calls.</p>`;
}

function renderSystem() {
  const sys = state.bootstrap.system;
  const vector = sys.vector_status;
  document.getElementById("vector-status").innerHTML = `
    <dt>TigerGraph</dt><dd>${esc(sys.graph_status.product)}</dd>
    <dt>Graph</dt><dd>${esc(sys.graph_status.graph)}</dd>
    <dt>Version</dt><dd>${esc(sys.graph_status.version)}</dd>
    <dt>Chunk embeddings</dt><dd>${Number(vector.chunk_embeddings_indexed).toLocaleString()} / ${Number(vector.chunk_embeddings_total).toLocaleString()}</dd>
    <dt>Dimension</dt><dd>${vector.dimension}</dd>
    <dt>Metric</dt><dd>${esc(vector.metric)}</dd>
    <dt>Index</dt><dd>${esc(vector.index)}</dd>
    <dt>Status</dt><dd>${esc(vector.status)}</dd>
    <dt>Vector search</dt><dd>${esc(vector.search_query)}</dd>
    <dt>Search status</dt><dd>${esc(vector.search_status)}</dd>`;
  document.getElementById("arch-notes").innerHTML = sys.architecture.notes.map((note) => `<li>${esc(note)}</li>`).join("");
  document.getElementById("why-disclaimer").textContent = sys.why_agentic.disclaimer;
  document.getElementById("why-rules").innerHTML = sys.why_agentic.rules
    .map((rule) => `<li><strong>${esc(rule.when)}</strong> → ${esc(rule.then)}<div class="hint">${esc(rule.detail)}</div></li>`)
    .join("");
  document.getElementById("vector-exp").innerHTML =
    `<p class="hint">${esc(sys.vector_experiment.label)}</p>` +
    sys.vector_experiment.variants.map((row) => `<div>${esc(row.id)} · ${esc(row.mechanism)} · ${row.correctness}%</div>`).join("") +
    `<p class="hint">${esc(sys.vector_experiment.note)}</p>`;
  renderArchitecture(sys.architecture);
}

async function loadHealth() {
  const box = document.getElementById("health-checks");
  const overall = document.getElementById("health-overall");
  try {
    const res = await fetch("/api/health");
    const health = await res.json();
    overall.textContent = health.overall || "Read-only";
    overall.className = `pill ${health.overall === "PASS" ? "ok" : health.overall === "ERROR" ? "danger" : "warn"}`;
    box.innerHTML = (health.checks || [])
      .map(
        (check) => `<div class="check"><div class="st ${esc(check.status)}">${esc(check.status)}</div><div><strong>${esc(check.label)}</strong><div class="hint">${esc(check.detail)}</div></div></div>`
      )
      .join("") || `<p class="hint">Health payload did not include checks.</p>`;
  } catch (err) {
    overall.textContent = "ERROR";
    box.innerHTML = `<div class="check"><div class="st ERROR">ERROR</div><div>${esc(err.message)}</div></div>`;
  }
}

function renderArchitecture(arch) {
  const svg = document.getElementById("arch-svg");
  const pos = {
    question: [320, 36],
    rag: [120, 130],
    parser: [500, 130],
    graphrag: [410, 210],
    agentic: [560, 210],
    tigergraph: [480, 290],
    evidence: [320, 350],
    generator: [320, 400],
    vector: [120, 290],
  };
  const nodes = arch.nodes
    .map((node) => {
      const [x, y] = pos[node.id] || [40, 40];
      const dashed = node.id === "vector";
      return `<rect x="${x - 70}" y="${y - 16}" width="140" height="32" fill="#141b24" stroke="${dashed ? "#d98c4a" : "#3ee0c6"}" stroke-dasharray="${dashed ? "4 3" : "0"}" />
        <text x="${x}" y="${y + 5}" text-anchor="middle" fill="#e7eef6" font-size="11">${esc(node.label)}</text>`;
    })
    .join("");
  const edges = arch.edges
    .map((edge) => {
      const a = pos[edge.from];
      const b = pos[edge.to];
      if (!a || !b) return "";
      return `<line x1="${a[0]}" y1="${a[1] + 16}" x2="${b[0]}" y2="${b[1] - 16}" stroke="${edge.dashed ? "#d98c4a" : "#243041"}" stroke-dasharray="${edge.dashed ? "4 3" : "0"}" />`;
    })
    .join("");
  svg.innerHTML = edges + nodes;
}

function remember(entry) {
  const row = {
    id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    ts: new Date().toISOString(),
    question: entry.question,
    pipeline: entry.pipeline,
    status: entry.status,
    status_label: entry.status_label,
    latency_ms: entry.latency_ms,
    investigation: entry.investigation || null,
    compare: entry.compare || null,
  };
  state.history = [row, ...state.history].slice(0, MAX_HISTORY);
  saveHistory();
  renderHistory();
}

function renderHistory() {
  const node = document.getElementById("history-list");
  if (!state.history.length) {
    node.innerHTML = `<p class="hint">No investigations in this browser session yet.</p>`;
    return;
  }
  node.innerHTML = state.history
    .map(
      (row) => `<button type="button" class="hist" data-id="${esc(row.id)}">
      <strong>${esc(row.question || "")}</strong>
      <div class="hint">${esc(pipelineLabel(row.pipeline))} · ${esc(row.status_label || row.status || "—")} · ${fmtMs(row.latency_ms)} · ${esc(row.ts)}</div>
    </button>`
    )
    .join("");
  node.querySelectorAll(".hist").forEach((btn) => {
    btn.addEventListener("click", () => {
      const found = state.history.find((row) => row.id === btn.dataset.id);
      if (!found) return;
      document.getElementById("question-input").value = found.question || (found.investigation && found.investigation.question) || "";
      if (found.pipeline && found.pipeline !== "compare") {
        const radio = document.querySelector(`input[name=pipeline][value="${found.pipeline}"]`);
        if (radio) radio.checked = true;
      } else if (found.pipeline === "compare") {
        const radio = document.querySelector(`input[name=pipeline][value="compare"]`);
        if (radio) radio.checked = true;
      }
      if (found.compare) {
        state.compare = found.compare;
        renderCompare();
      }
      if (found.investigation) applyInvestigation(found.investigation, { skipHistory: true, replay: true });
      showView("investigate");
    });
  });
}

function renderReports() {
  const preview = document.getElementById("report-preview");
  const payload = exportInvestigation() || exportComparison();
  preview.textContent = payload ? JSON.stringify(payload, null, 2) : "Run an investigation to preview a sanitized report.";
}

function exportInvestigation() {
  if (!state.investigation) return null;
  return state.investigation.export_record || {
    schema_version: state.investigation.schema_version,
    question: state.investigation.question,
    pipeline: state.investigation.pipeline,
    answer: state.investigation.answer,
    status: state.investigation.status,
    citations: state.investigation.citations,
    evidence_ids: state.investigation.evidence_ids,
    stop_reason: state.investigation.stop_reason,
    continue_explanation: state.investigation.continue_explanation,
  };
}

function exportComparison() {
  if (!state.compare) return null;
  return {
    question: state.compare.question,
    comparison: state.compare.comparison,
    what_changed: state.compare.what_changed,
    results: (state.compare.results || []).map((row) => row.export_record || { pipeline: row.pipeline, answer: row.answer, status: row.status }),
  };
}

async function copyPayload(payload) {
  if (!payload) {
    banner("Nothing to export yet.", "warn");
    showView("reports");
    return;
  }
  await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
  banner("Copied sanitized JSON.", "");
}

function downloadPayload(payload, kind) {
  if (!payload) {
    banner("Nothing to export yet.", "warn");
    showView("reports");
    return;
  }
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${kind}-${payload.pipeline || payload.question || "report"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function loadHistory() {
  try {
    return JSON.parse(sessionStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    sessionStorage.setItem(HISTORY_KEY, JSON.stringify(state.history));
  } catch {
    state.history = state.history.slice(0, 5);
  }
}

function pipelineLabel(id) {
  return { rag: "RAG", graphrag: "Fixed GraphRAG", agentic_graphrag: "Agentic GraphRAG", compare: "Compare All" }[id] || id || "—";
}

function barChart(rows, opts) {
  const width = 520;
  const height = 200;
  const max = opts.max || Math.max(...rows.map((row) => Number(row.value) || 0), 1);
  return `<svg class="bar-chart" viewBox="0 0 ${width} ${height}">${rows
    .map((row, index) => {
      const barWidth = 80;
      const gap = 40;
      const x = 40 + index * (barWidth + gap);
      const h = ((Number(row.value) || 0) / max) * 140;
      return `<rect x="${x}" y="${160 - h}" width="${barWidth}" height="${h}" fill="${index === 1 ? "#3ee0c6" : index === 2 ? "#7aa2ff" : "#8ea0b5"}" />
        <text x="${x + barWidth / 2}" y="180" text-anchor="middle" fill="#8ea0b5" font-size="11">${esc(row.label)}</text>
        <text x="${x + barWidth / 2}" y="${150 - h}" text-anchor="middle" fill="#e7eef6" font-size="12">${row.value}${opts.unit || ""}</text>`;
    })
    .join("")}</svg>`;
}

function groupedBars(rows, series) {
  const colors = ["#8ea0b5", "#3ee0c6", "#7aa2ff"];
  const width = 640;
  const height = 220;
  const max = Math.max(...rows.flatMap((row) => row.values.map((value) => Number(value) || 0)), 1);
  const groupWidth = Math.min(160, (width - 40) / rows.length);
  return `<svg class="bar-chart" viewBox="0 0 ${width} ${height}">${rows
    .map((row, index) => {
      const gx = 30 + index * groupWidth;
      const barW = Math.max(10, (groupWidth - 16) / series.length);
      const bars = row.values
        .map((value, si) => {
          const h = ((Number(value) || 0) / max) * 140;
          const x = gx + si * barW;
          return `<rect x="${x}" y="${160 - h}" width="${barW - 2}" height="${h}" fill="${colors[si % colors.length]}" />`;
        })
        .join("");
      return `${bars}<text x="${gx + groupWidth / 2 - 8}" y="188" text-anchor="middle" fill="#8ea0b5" font-size="10">${esc(row.label)}</text>`;
    })
    .join("")}${series
    .map((name, index) => `<text x="${20 + index * 90}" y="214" fill="${colors[index]}" font-size="11">${esc(name)}</text>`)
    .join("")}</svg>`;
}

function fmtPct(value) {
  if (value == null) return "—";
  return `${Number(value)}%`;
}

function fmtMetric(value, metric, bench) {
  if (value == null) {
    if ((bench.unpublished_metrics || []).includes(metric)) return "not published";
    if (metric === "completeness") return "n/a";
    return "—";
  }
  if (metric.includes("latency")) return `${value} ms`;
  if (["correctness", "exact", "completeness", "grounding", "citation_validity"].includes(metric)) return `${value}%`;
  return String(value);
}

function fmtMs(value) {
  if (value == null || value === "") return "—";
  return `${Number(value).toFixed(0)} ms`;
}

function esc(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

boot().catch((err) => {
  document.body.insertAdjacentHTML("beforeend", `<p class="hint">Failed to load console: ${esc(err.message)}</p>`);
});
