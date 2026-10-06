"use strict";
// Exercise actual presentation functions using the existing public catalog; no stdin or live services.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const { test } = require("node:test");
const { execFileSync } = require("node:child_process");
const path = require("node:path");
const bootstrap = JSON.parse(execFileSync("python", ["-B", "-c",
  "import json; from ui.viewmodels import benchmark_view, system_view; print(json.dumps({'benchmark': benchmark_view(), 'system': system_view()}))"
], { cwd: path.resolve(__dirname, ".."), encoding: "utf8", timeout: 15000, windowsHide: true,
  stdio: ["ignore", "pipe", "pipe"] }));
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {
    innerHTML: "", textContent: "", dataset: {}, classList: { toggle() {} },
    setAttribute(name, value) { this[name] = value; }, addEventListener() {}, querySelectorAll() { return []; },
  });
  return elements.get(id);
}
const context = vm.createContext({
  document: { getElementById: element, querySelectorAll: () => [], querySelector: () => ({ value: "", checked: false }) },
  sessionStorage: { getItem: () => null }, AbortController, console,
});
let source = fs.readFileSync("ui/static/console.js", "utf8");
source = source.slice(0, source.lastIndexOf("boot().catch"));
vm.runInContext(source, context);
const run = (code) => vm.runInContext(code, context);
context.fixture = bootstrap;
run("state.bootstrap = fixture");

test("published known token totals actually render from catalog", () => {
  run("renderBenchmark()");
  const html = element("chart-tokens").innerHTML;
  assert.match(html, /<svg/);
  for (const row of Object.values(bootstrap.benchmark.pipelines)) {
    assert.ok(html.includes(String(row.tokens)));
    assert.ok(html.includes(row.name));
  }
});
test("unpublished token totals show an honest empty state", () => {
  run('state.bootstrap = { ...fixture, benchmark: { ...fixture.benchmark, unpublished_metrics: ["tokens"] } }; renderBenchmark()');
  assert.doesNotMatch(element("chart-tokens").innerHTML, /<svg/);
  assert.match(element("chart-tokens").innerHTML, /not published/);
  run("state.bootstrap = fixture");
});
test("all five families and all three series remain visible with exact values", () => {
  run("renderBenchmark()");
  const html = element("chart-family").innerHTML;
  assert.equal((html.match(/class="family-group"/g) || []).length, 5);
  assert.equal((html.match(/class="family-row"/g) || []).length, 15);
  for (const row of bootstrap.benchmark.families) {
    assert.ok(html.includes(row.label));
    for (const key of ["rag", "graphrag", "agentic_graphrag"]) assert.ok(html.includes(row[key] + "%"));
  }
});
test("live vector booleans override stale search metadata", () => {
  const vector = { ...bootstrap.system.vector_status, status: "READY", attempted: true,
    schema_ok: true, search_query_ok: true, index_ok: true };
  context.vector = vector;
  run("renderVectorStatus(vector, fixture.system)");
  const html = element("vector-status").innerHTML;
  assert.match(html, /Search status<\/dt><dd>PASS/);
  assert.equal((html.match(/<dd>PASS<\/dd>/g) || []).length, 3);
  assert.ok(html.includes(vector.counts_source));
});
test("vector missing, failed and unavailable checks are never called ready", () => {
  assert.equal(run('vectorCheckStatus({ status: "READY" }, "index_ok")'), "NOT_CHECKED");
  assert.equal(run('vectorCheckStatus({ attempted: true, index_ok: false }, "index_ok")'), "NOT_READY");
  assert.equal(run('vectorCheckStatus({ status: "UNAVAILABLE", index_ok: false }, "index_ok")'), "UNAVAILABLE");
});
test("trace hierarchy preserves observed action, reason, status and next trigger", () => {
  context.step = { index: 2, kind: "follow_up", action: "supporting_chunks", status: "ok",
    reason: "An evidence slot remains unresolved.", elapsed_ms: 13, evidence_count: 2,
    retrieval_method: "documents", continued: true, why_next: "Additional evidence required." };
  const html = run("renderTraceStep(step)");
  for (const text of ["STEP 2", "supporting_chunks", "ok", "13 ms", "2 evidence", "documents", context.step.reason, context.step.why_next]) assert.ok(html.includes(text));
  assert.match(html, /class="trace-step" data-kind="follow_up"/);
  assert.doesNotMatch(html, /class="primary"/);
});
test("vector trace labeling only appears for an actual vector action", () => {
  assert.match(run('renderTraceStep({ action: "vector_search", kind: "follow_up" })'), /Vector follow-up/);
  assert.doesNotMatch(run('renderTraceStep({ action: "event_neighborhood", kind: "follow_up" })'), /Vector follow-up/);
});
test("trace field escaping does not introduce markup", () => {
  assert.ok(run('renderTraceStep({ action: "<script>", reason: "<b>" })').includes("&lt;script&gt;"));
});
test("short labels use an ellipsis without changing full identifiers", () => {
  assert.equal(run('shortGraphLabel("abcdefghijk", 6)'), "abcde…");
  assert.equal(run('shortGraphLabel("Event")'), "Event");
});
test("dense labels keep focus and central entities without changing topology", () => {
  context.nodes = Array.from({ length: 30 }, (_, i) => ({ id: "node" + i, kind: "chunk", evidence_id: "e" + i }));
  context.nodes[29].kind = "event";
  const original = JSON.stringify(context.nodes);
  const labels = run('graphLabelIds(nodes, [], "e27")');
  assert.equal(labels.size, 12);
  assert.ok(labels.has("node29"));
  assert.ok(labels.has("node27"));
  assert.equal(JSON.stringify(context.nodes), original);
});
test("graph rendering preserves full canonical labels and IDs for accessibility", () => {
  context.inv = { pipeline: "graphrag", status: "answered", graph: { nodes: [
    { id: "event:Q1", kind: "event", label: "A very long canonical event label" },
    { id: "document:doc1", kind: "document", label: "doc1" }],
    edges: [{ source: "event:Q1", target: "document:doc1", label: "DESCRIBES" }] } };
  run("state.investigation = inv; renderGraph()");
  const html = element("graph").innerHTML;
  assert.match(html, /A very long canonical event label · event:Q1/);
  assert.match(html, /tabindex="0" role="button"/);
  assert.match(html, /data-source="event:Q1" data-target="document:doc1"/);
  assert.match(html, /A very long canon/);
});
test("RAG metadata graph is explicitly text evidence, not graph retrieval", () => {
  run('state.investigation = { ...inv, pipeline: "rag" }; renderGraph()');
  assert.match(element("graph-note").textContent, /no graph retrieval/);
});
test("known token subtotal remains visible when accounting is incomplete", () => {
  run('renderMetrics("efficiency", {tokens: 123}, [["Known tokens", row => row.tokens]])');
  assert.match(element("efficiency").innerHTML, /123/);
  assert.match(element("efficiency").innerHTML, /metric-primary/);
});
test("trace styles do not collide with action buttons and motion is optional", () => {
  const css = fs.readFileSync("ui/static/console.css", "utf8");
  assert.match(css, /button\.primary \{/);
  assert.doesNotMatch(css, /(?:^|\n)\.primary \{/);
  assert.match(css, /prefers-reduced-motion/);
  const html = fs.readFileSync("ui/static/index.html", "utf8");
  assert.match(html, /<details class="reliability-details">/);
});

test("trace and status text have readable contrast on the dark palette", () => {
  const css = fs.readFileSync("ui/static/console.css", "utf8");
  const color = (name) => css.match(new RegExp("--" + name + ": (#[0-9a-f]{6})"))[1];
  const luminance = (hex) => {
    const channels = hex.slice(1).match(/../g).map(channel => {
      const value = parseInt(channel, 16) / 255;
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    });
    return channels[0] * .2126 + channels[1] * .7152 + channels[2] * .0722;
  };
  for (const foreground of ["text", "muted", "ok", "warn", "danger", "gold", "accent", "accent-2"]) {
    for (const background of ["bg", "bg-2", "panel", "panel-2"]) {
      const light = luminance(color(foreground)), dark = luminance(color(background));
      const ratio = (Math.max(light, dark) + .05) / (Math.min(light, dark) + .05);
      assert.ok(ratio >= 4.5, foreground + " on " + background + ": " + ratio);
    }
  }
});

test("Evidence Diff favors observed field meaning and preserves provenance", () => {
  context.evidenceItem = { evidence_type: "fact", field_name: "competitors", value: "24", evidence_id: "event_neighborhood:abc:fact:competitors:Q1" };
  const before = JSON.stringify(context.evidenceItem);
  const html = run("renderDiffEvidence([evidenceItem.evidence_id], new Map([[evidenceItem.evidence_id, evidenceItem]]))");
  assert.match(html, /<strong>Competitors<\/strong><span>24<\/span>/);
  assert.match(html, /<details class="diff-identifiers">/);
  assert.ok(html.includes(context.evidenceItem.evidence_id));
  assert.equal(JSON.stringify(context.evidenceItem), before);
});
test("typed ID fallback understands hashed call IDs without inventing values", () => {
  const cases = { "tool:abc:entity:Q1": "Event", "tool:abc:fact:gold:Q1": "Gold medalist",
    "tool:abc:fact:nations:Q1": "Nations", "tool:abc:edge:IN_GAMES:games1": "Games",
    "tool:abc:edge:HELD_AT:venue1": "Venue", "tool:abc:edge:OTHER:Q1": "Relation",
    "tool:abc:chunk:c1": "Supporting chunk", "tool:abc:document:d1": "Document",
    "unstructured-id": "Evidence", "tool:abc:fact:constructor:Q1": "Answer field" };
  for (const [id, label] of Object.entries(cases)) {
    context.id = id;
    const item = run("diffEvidenceSummary(id)");
    assert.equal(item.label, label);
    assert.equal(item.value, "");
  }
});
test("placeholder field names are not presented as observed answer values", () => {
  const item = run('diffEvidenceSummary("tool:fact:competitors:Q1", { evidence_type: "fact", field_name: "competitors", value: "competitors" })');
  assert.equal(item.value, "");
});
test("before/new/resolved/remaining semantics and original data stay intact", () => {
  context.delta = { pipeline: "agentic_graphrag", evidence: [], evidence_diff: { steps: [
    { index: 1, tool: "retrieve_spec", before: [], added: ["initial"], added_items: [
      { evidence_id: "initial", evidence_type: "entity", value: "Sample event" }], resolved: ["event_identity"], remaining: ["graph_relation"] },
    { index: 2, tool: "event_neighborhood", before: ["initial"], added: ["relation"], added_items: [
      { evidence_id: "relation", evidence_type: "edge", field_name: "HELD_AT", value: "Sample venue" }], resolved: ["graph_relation"], remaining: [] }
  ] } };
  const before = JSON.stringify(context.delta);
  run("renderDiff(delta)");
  const html = element("evidence-diff").innerHTML;
  for (const text of ["Before", "New evidence", "Resolved", "Remaining", "Sample event", "Sample venue", "event_identity", "graph_relation"]) assert.ok(html.includes(text));
  assert.equal((html.match(/Sample event/g) || []).length, 2);
  assert.equal(JSON.stringify(context.delta), before);
});
test("Diff metadata and raw identifiers remain HTML escaped", () => {
  const html = run('renderDiffEvidence(["<script>"], new Map([["<script>", { evidence_type: "fact", field_name: "gold", value: "<img>" }]]))');
  assert.ok(html.includes("&lt;script&gt;"));
  assert.ok(html.includes("&lt;img&gt;"));
  assert.doesNotMatch(html, /<script>|<img>/);
});
test("method tags shorten only the UI prefix and preserve canonical titles", () => {
  context.methods = ["gsql:chunks_for_events", "bm25+candidates", "tigergraph_vector"];
  const before = JSON.stringify(context.methods);
  const html = run("renderMethods(methods)");
  assert.match(html, /title="gsql:chunks_for_events">chunks_for_events<\/span>/);
  assert.ok(html.includes("bm25+candidates"));
  assert.ok(html.includes("tigergraph_vector"));
  assert.equal(JSON.stringify(context.methods), before);
  assert.match(run('renderMethods(["gsql:chunks_for_events", "chunks_for_events"])'), /title="gsql:chunks_for_events">gsql:chunks_for_events/);
  assert.match(run('renderMethods(["gsql:<unsafe>"])'), /title="gsql:&lt;unsafe&gt;">&lt;unsafe&gt;/);
});
test("efficiency, evidence status and Compare use the same method tags", () => {
  run('renderMetrics("efficiency", {retrieval_methods: methods}, [["Methods", row => row.retrieval_methods.join(", ")]])');
  run('renderEvidenceStatus({pipeline: "rag", evidence_status: {retrieval_methods: methods}})');
  run('state.compare = { comparison: [{ pipeline: "rag", retrieval_methods: methods }], results: [], what_changed: [] }; renderCompare()');
  for (const id of ["efficiency", "evidence-status", "compare-table"]) {
    const html = element(id).innerHTML;
    assert.match(html, /class="method-chips"/);
    assert.match(html, /title="gsql:chunks_for_events">chunks_for_events<\/span>/);
  }
});
test("saved comparison row replay stays PREVIEW and live comparison row stays LIVE", async () => {
  context.renderInvestigate = () => {};
  context.renderReports = () => {};
  context.showView = () => {};
  context.banner = () => {};
  context.setBusy = () => {};
  context.remember = () => {};
  let historyClick;
  let compareClick;
  const requests = [];
  const recorded = { question: "Which event?", pipeline: "agentic_graphrag", status: "answered",
    runtime_mode: "live", runtime_mode_label: "LIVE", answer: "Observed answer" };
  const comparison = { results: [recorded], comparison: [{ pipeline: recorded.pipeline,
    pipeline_label: "Agentic GraphRAG", status: "answered", answer: recorded.answer,
    retrieval_methods: [] }], what_changed: [] };
  const historyButton = { dataset: { id: "saved-compare" }, addEventListener: (_, fn) => { historyClick = fn; } };
  element("history-list").querySelectorAll = (selector) => selector === ".hist" ? [historyButton] : [];
  element("compare-table").querySelectorAll = (selector) => selector === ".compare-pick"
    ? [{ dataset: { pipeline: recorded.pipeline }, addEventListener: (_, fn) => { compareClick = fn; } }]
    : [];
  context.comparisonFixture = comparison;
  context.recordedFixture = recorded;
  context.fetch = async (url) => {
    requests.push(url);
    return { json: async () => comparison };
  };
  run(`state.history = [{id: "saved-compare", pipeline: "compare", question: recordedFixture.question,
    investigation: recordedFixture, compare: comparisonFixture}]; renderHistory()`);
  historyClick();
  assert.equal(run("state.investigation.runtime_mode"), "preview");
  assert.equal(run("state.investigation.runtime_mode_label"), "PREVIEW");
  assert.equal(run("state.investigation.recorded"), true);
  assert.equal(run("state.investigation.recorded_label"), "SESSION REPLAY");
  compareClick();
  assert.equal(run("state.investigation.runtime_mode"), "preview");
  assert.equal(run("state.investigation.runtime_mode_label"), "PREVIEW");
  assert.equal(run("state.investigation.recorded"), true);
  assert.equal(run("state.investigation.recorded_label"), "SESSION REPLAY");
  assert.equal(requests.length, 0);
  await run("runCompare('Which event?')");
  assert.equal(run("state.compareReplay"), false);
  compareClick();
  assert.equal(run("state.investigation.runtime_mode"), "live");
  assert.equal(run("state.investigation.runtime_mode_label"), "LIVE");
  assert.equal(run("state.investigation.recorded"), undefined);
  assert.deepEqual(requests, ["/api/compare"]);
});

test("method tags wrap between identifiers and bound exceptionally long names", () => {
  const css = fs.readFileSync("ui/static/console.css", "utf8");
  const tags = css.match(/\.method-chip \{([^}]+)\}/)[1];
  assert.match(tags, /white-space: nowrap/);
  assert.match(tags, /word-break: normal/);
  assert.match(tags, /text-overflow: ellipsis/);
  assert.match(css, /\.method-chips \{[^}]+flex-wrap: wrap/);
});

// Isolated DOM double: real handlers and every detail renderer remain active.
function comparisonWorkspace() {
  const nodes = new Map();
  function node(id) {
    const handlers = new Map(), queries = new Map();
    return { id, value: "", innerHTML: "", textContent: "", dataset: {},
      classList: { add() {}, toggle() {} },
      setAttribute(name, value) { this[name] = value; },
      addEventListener(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); },
      dispatchEvent(event) { for (const fn of handlers.get(event.type) || []) fn(event); },
      querySelectorAll(selector) {
        const attribute = selector === ".compare-pick" ? "pipeline" : selector === ".hist" ? "id" : null;
        if (!attribute) return [];
        const cached = queries.get(selector);
        if (cached && cached.html === this.innerHTML) return cached.rows;
        const rows = [...this.innerHTML.matchAll(new RegExp('data-' + attribute + '="([^"]+)"', "g"))]
          .map(match => { const row = node("row"); row.dataset[attribute] = match[1]; return row; });
        queries.set(selector, { html: this.innerHTML, rows });
        return rows;
      },
    };
  }
  const get = id => { if (!nodes.has(id)) nodes.set(id, node(id)); return nodes.get(id); };
  const radios = ["rag", "graphrag", "agentic_graphrag", "compare"].map(value => {
    const radio = node(value); radio.value = value;
    Object.defineProperty(radio, "checked", { get: () => !!radio._checked, set: checked => {
      if (checked) radios.forEach(other => { other._checked = false; });
      radio._checked = !!checked;
    } });
    return radio;
  });
  radios[2].checked = true;
  const storage = new Map(), requests = [];
  const ctx = vm.createContext({
    document: { getElementById: get,
      querySelectorAll: selector => selector === "input[name=pipeline]" ? radios : [],
      querySelector: selector => selector.endsWith(":checked") ? radios.find(radio => radio.checked)
        : radios.find(radio => radio.value === (selector.match(/value="([^"]+)"/) || [])[1]),
    },
    sessionStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
    AbortController, setInterval, clearInterval, console,
  });
  vm.runInContext(source, ctx);
  const execute = code => vm.runInContext(code, ctx);
  ctx.fixture = bootstrap;
  execute("state.bootstrap = fixture; renderShell()");
  const question = "Which Olympic event is described?";
  const results = [["rag", "RAG", 101, 1111, "sparse/bm25"],
    ["graphrag", "GRAPHRAG", 202, 2222, "gsql:fixed-sentinel"],
    ["agentic_graphrag", "AGENTIC", 303, 3333, "gsql:agentic-sentinel"]].map(([pipeline, label, tokens, latency, method]) => {
      const agentic = pipeline === "agentic_graphrag", eid = "evidence-sentinel:" + pipeline + ":end";
      const steps = [{ index: 1, kind: "primary", action: "[" + label + "] RETRIEVAL SENTINEL" }];
      if (agentic) steps.push({ index: 2, kind: "follow_up", action: "AGENTIC FOLLOW-UP SENTINEL" });
      return { question, pipeline, status: "answered", status_label: "ANSWERED",
        runtime_mode: "live", runtime_mode_label: "LIVE", answer: "[" + label + "] ANSWER SENTINEL",
        latency_ms: latency, tokens, retrieval_methods: [method], steps,
        efficiency: { tokens, latency_ms: latency, model_calls: 1, tool_calls: agentic ? 2 : 0,
          steps: agentic ? 2 : 0, retries: 0, retrieval_methods: [method] },
        reliability: { status: "answered", attempts: 1 },
        citations: [{ evidence_id: eid }], evidence: [{ evidence_id: eid }],
        evidence_status: { retrieved: 1, used: 1, citation_count: 1, graph_present: pipeline !== "rag",
          retrieval_methods: [method] },
        lineage: [{ evidence_id: eid, document_id: pipeline + "-document", chunk_id: pipeline + "-chunk",
          value: "[" + label + "] SOURCE SENTINEL", retrieval_method: method }],
        graph: { nodes: [{ id: "node-sentinel:" + pipeline + ":end", evidence_id: eid,
          label: label + " NODE SENTINEL", kind: pipeline === "rag" ? "chunk" : "event" }], edges: [] },
        why_continued: agentic ? ["AGENTIC CONTINUE SENTINEL"] : [],
        stop_explanation: label + " STOP SENTINEL",
        evidence_diff: { steps: agentic ? [{ index: 2, tool: "AGENTIC DELTA SENTINEL", added: [eid] }] : [] },
      };
    });
  const comparison = { question, results, comparison: results.map(result => ({ ...result,
    pipeline_label: result.pipeline, citation_count: 1, evidence_count: 1, model_calls: 1 })), what_changed: [] };
  ctx.fetch = async url => {
    requests.push(url);
    assert.equal(url, "/api/compare", "cached inspection must never request /api/investigate");
    return { json: async () => comparison };
  };
  get("question-input").value = question;
  return { get, execute, results, comparison, requests,
    radio(pipeline) { const radio = radios.find(item => item.value === pipeline); radio.checked = true;
      radio.dispatchEvent({ type: "change" }); },
    row(pipeline) { const row = get("compare-table").querySelectorAll(".compare-pick")
      .find(item => item.dataset.pipeline === pipeline); row.dispatchEvent({ type: "click" }); },
    replay() { const button = get("history-list").querySelectorAll(".hist")[0]; button.dispatchEvent({ type: "click" }); },
  };
}

function assertComparisonDetails(ui, pipeline, replay) {
  const expected = ui.results.find(result => result.pipeline === pipeline);
  const actual = ui.execute("state.investigation");
  for (const key of ["pipeline", "answer", "tokens", "latency_ms"]) assert.equal(actual[key], expected[key]);
  assert.equal(ui.execute("selectedPipeline()"), pipeline);
  assert.ok(ui.get("asked-question").textContent.startsWith(ui.execute(`pipelineLabel('${pipeline}')`) + " · "));
  assert.equal(ui.get("answer-text").textContent, expected.answer);
  assert.ok(ui.get("efficiency").innerHTML.includes(String(expected.tokens)));
  assert.ok(ui.get("efficiency").innerHTML.includes(expected.latency_ms + " ms"));
  assert.ok(ui.get("efficiency").innerHTML.includes(expected.retrieval_methods[0]));
  assert.ok(ui.get("lineage").innerHTML.includes(expected.lineage[0].evidence_id));
  assert.ok(ui.get("citations").innerHTML.includes(expected.citations[0].evidence_id));
  assert.ok(ui.get("graph").innerHTML.includes(expected.graph.nodes[0].id));
  assert.equal(ui.get("step-count").textContent, expected.steps.length + " steps");
  expected.steps.forEach(step => assert.ok(ui.get("timeline").innerHTML.includes(step.action)));
  assert.equal(ui.get("stop-reason").textContent, expected.stop_explanation);
  const detailIds = ["citations", "lineage", "graph", "timeline", "efficiency", "report-preview"];
  for (const other of ui.results.filter(result => result.pipeline !== pipeline)) {
    for (const id of detailIds) {
      const content = ui.get(id).innerHTML + ui.get(id).textContent;
      for (const sentinel of [other.answer, other.graph.nodes[0].id, other.citations[0].evidence_id,
        other.steps[0].action, other.retrieval_methods[0]]) assert.ok(!content.includes(sentinel), id + " retained " + sentinel);
    }
  }
  if (pipeline !== "agentic_graphrag") {
    assert.ok(!ui.get("timeline").innerHTML.includes("AGENTIC FOLLOW-UP SENTINEL"));
    assert.ok(!ui.get("evidence-diff").innerHTML.includes("AGENTIC DELTA SENTINEL"));
    assert.ok(!ui.get("continue-reason").textContent.includes("AGENTIC CONTINUE SENTINEL"));
  }
  assert.equal(ui.get("runtime-mode").textContent, replay ? "PREVIEW" : "LIVE");
  assert.equal(actual.runtime_mode, replay ? "preview" : "live");
  assert.equal(actual.recorded, replay ? true : undefined);
  if (replay) assert.equal(actual.recorded_label, "SESSION REPLAY");
  assert.equal(ui.execute("state.selectedEvidence"), null);
  assert.equal(JSON.stringify(ui.execute("state.graph")), JSON.stringify({ scale: 1, x: 0, y: 0 }));
}

for (const replay of [false, true]) test(`${replay ? "saved PREVIEW" : "fresh LIVE"} comparison switches complete details through radios and rows`, async () => {
  const ui = comparisonWorkspace();
  await ui.execute("runCompare(currentQuestion())");
  if (replay) ui.replay();
  const original = JSON.stringify(ui.comparison);
  for (const pipeline of ["rag", "graphrag", "agentic_graphrag", "rag"]) {
    ui.execute('state.selectedEvidence = "stale-evidence"; state.graph = {scale: 2, x: 150, y: 200}');
    ui.radio(pipeline);
    assertComparisonDetails(ui, pipeline, replay);
    ui.execute('state.selectedEvidence = "stale-evidence"; state.graph = {scale: 2, x: 150, y: 200}');
    ui.row(pipeline);
    assertComparisonDetails(ui, pipeline, replay);
  }
  assert.deepEqual(ui.requests, ["/api/compare"]);
  assert.equal(JSON.stringify(ui.comparison), original, "cached source payloads must not be mutated");
});

test("an incompatible cached question keeps next-run selection and the displayed result's identity", async () => {
  const ui = comparisonWorkspace();
  await ui.execute("runCompare(currentQuestion())");
  const displayed = ui.execute("state.investigation");
  ui.get("question-input").value = "A different question";
  ui.radio("rag");
  assert.equal(ui.execute("selectedPipeline()"), "rag");
  assert.equal(ui.execute("state.investigation"), displayed);
  assert.ok(ui.get("asked-question").textContent.startsWith("Agentic GraphRAG · "));
  assert.deepEqual(ui.requests, ["/api/compare"]);
});
