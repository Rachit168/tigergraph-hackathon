# Judge demo: OlympicGraph Investigation Console

Target: **4 minutes 20 seconds**, with 40 seconds of contingency before the
five-minute limit. Show one public question; the console is the main screen.
Do not run the benchmark to demonstrate the application.

## Before recording

- Start `python -m scripts.serve_ui`; open `http://127.0.0.1:8765/`.
- Configure the external public corpus, read-only TigerGraph access, and semantic
  provider as described in [Reproduction](reproducibility.md). Confirm semantic
  mode for this demo; deterministic mode does not reproduce the published scores.
- Rehearse only the selected question if services are ready. Check its actual
  trace; follow-ups are conditional, not guaranteed. Export its sanitized
  comparison through **Reports** as a backup and keep this browser session open.
- Keep [the published benchmark summary](public_benchmark.md) available. Do not
  display raw scored artifacts, credentials, or provider configuration values.

## Primary question and result order

Public question **pub-017**:

> Who won the gold medal in the event held at Royal Artillery Barracks on
> 28 July 2012?

This asks for a relationship across venue, date, event, and medal evidence.
Existing public example coverage includes `retrieve_spec` and
`event_neighborhood`; that supports the choice, not a promised live trace.

Run **Compare All** once. After completion, click the **RAG** comparison row,
then switch the cached pipeline selector to **GraphRAG**, then **Agentic
GraphRAG** without pressing Run again. Each selection shows that pipeline's
own answer, citations, evidence, methods, timings, and trace.

Do not promise RAG will fail or that Agentic will take another step. If all
answers match, compare the evidence paths and measured costs. If Agentic stops
after initial retrieval, demonstrate that stopping as valid bounded behavior.

## Timed walkthrough

### 00:00–00:30 — Investigate: opening

**Action:** Show the question workspace and three pipeline choices.

**Say:** “An Olympic question can require more than finding a paragraph: it may
connect a venue, a date, an event, and a medal. This console compares three ways
of gathering evidence: BM25 text retrieval, fixed TigerGraph retrieval, and a
bounded agent that investigates further when evidence is missing. They share
the same answer generator. I'll show the answer, its sources, and why the
investigation continued or stopped.”

**Notice:** The comparison is about retrieval; TigerGraph is central to the
structured and adaptive paths.

### 00:30–00:45 — System: live readiness

**Action:** Open **System**. Point to the read-only connection, schema, and query
checks and the provider/corpus indicators. Give checks at most 15 seconds.

**Say:** “These are read-only readiness checks. I'll use the reported state;
unavailable services are shown explicitly.”

**Notice:** Configuration presence is not a successful provider request.
Only call TigerGraph ready if the displayed checks support it.

### 00:45–01:25 — Investigate: one live comparison

**Action:** Enter pub-017 exactly, choose **Compare All**, click **Run** once.
Show the real running indicator and elapsed time.

**Say:** “The same question goes through all three pipelines. Text retrieval
searches chunks; fixed GraphRAG uses typed GSQL; Agentic checks evidence state
before choosing any bounded follow-up.”

**Notice:** No fabricated progress or answers. This segment reserves about
40 seconds for execution; move on sooner if it completes. If still pending at
01:25, use the delay plan below instead of waiting indefinitely.

### 01:25–02:00 — Compare / Investigate: three retrieval paths

**Action:** On **Compare**, point to the actual answers, statuses, methods,
tokens, model calls, and latency. Click the RAG row; switch to GraphRAG, then
Agentic GraphRAG using the cached selector.

**Say:** “Here is text retrieval's result and source set. Fixed GraphRAG queries
venue and date through typed graph retrieval. Agentic has its own evidence
path and stopping condition. These are the observed results of this question.”

**Notice:** RAG has no graph retrieval; Fixed GraphRAG has no adaptive follow-up.
Show actual differences even when the final answer is the same. Do not assign
a winner or confuse these live timings with benchmark averages.

### 02:00–02:40 — Investigate: citations and graph context

**Action:** With Agentic selected, click one returned citation. Expand its
**Evidence lineage** item to inspect source text and available chunk/document
identifiers. In **Graph context**, select the associated event or venue node
if returned; hover a visible labeled relationship.

**Say:** “This citation leads to the retrieved evidence and its source.
The graph shows the relationships returned for this investigation, rather
than a decorative network.”

**Notice:** Answer → citation → evidence → source → graph context. Use only
returned items; say when source detail or graph context is absent.

### 02:40–03:15 — Investigate: trace and evidence diff

**Action:** Show **Investigation trace**, **Evidence diff**, **Why did it
continue?**, and **Why did it stop?**. If present, highlight the follow-up
`event_neighborhood` step after `retrieve_spec`: its recorded reason, new
evidence, and resolved/remaining state where reported.

**Say, if a follow-up occurred:** “This recorded evidence gap triggered the
next tool call. The diff shows what it added; the stop explanation shows why
investigation ended.”

**Otherwise say:** “The initial evidence was sufficient according to the
recorded stop condition, so this investigation did not add a follow-up.”
Use that sentence only if the actual stop condition confirms sufficiency;
otherwise read the recorded failure, ambiguity, or budget reason plainly.

**Notice:** Operational decisions and evidence changes, not private
chain-of-thought. Do not infer resolved slots that the trace does not report.

### 03:15–04:00 — Benchmark: measured quality and efficiency

**Action:** Open **Benchmark**. Show accuracy, token totals, mean latency, and
errors; briefly point to the family breakdown and Agentic tool behavior.

**Say:** “On the published 100-question public run, semantic accuracy was
65 percent for Basic RAG, 98 for Fixed GraphRAG, and 99 for Agentic. Known total
tokens were 295,344, 146,180, and 123,016. Mean latency was about 10.67, 7.59,
and 7.11 seconds, with four, one, and zero generation errors. Agentic made
follow-ups on 28 questions. Its one-point difference from Fixed GraphRAG is a
measured result alongside bounded investigation, not proof that every extra
step helps. These measurements describe this corpus and run.”

**Notice:** Same `SemanticGenerator`, `dynamic/olympic-llm`, temperature 0,
maximum completion tokens 1024. Structured retrieval supports the measured
quality difference; the run does not establish universal superiority or a
causal explanation for every corrected answer. p95 latencies are 30,681 /
17,784 / 17,193 ms, if asked.

### 04:00–04:12 — System: vector scope

**Action:** Briefly show the vector panel and its actual check state. Treat
embedding counts as the labeled verified snapshot, not a fresh live count.

**Say:** “Production Agentic can optionally use TigerGraph Vector; the canonical
benchmark neither wired nor exercised it, so 99 percent is not attributed
to vector retrieval.”

**Notice:** Optional production capability, not the default BM25 retriever.

### 04:12–04:20 — Investigate: close

**Action:** Return to the Agentic result and its evidence/stop explanation.

**Say:** “The result is an answer you can inspect: its evidence, its graph
context, and the bounded decisions behind further investigation.”

**Notice:** The core differentiator is observable investigation, not merely
an accuracy card.

## Failure plan and optional backup question

- **Semantic provider unavailable:** Show the real failure and reliability
  status; use the published benchmark page. Do not silently substitute the
  deterministic generator or claim it reproduces semantic results.
- **TigerGraph health fails:** Show the failed check. Present any genuine RAG
  result if available; explain graph evidence is unavailable and use the
  published summary for the measured comparison. Do not change graph setup.
- **Live comparison exceeds its time slot:** Move to the published dashboard;
  if results complete, resume their inspection. **Stop** cancels browser
  waiting, but the backend may still finish. Do not repeatedly rerun.
- **One pipeline errors:** Keep its error visible, compare the available results,
  and expand reliability only if useful. No answer substitution.
- **History disappears after refresh:** Do not depend on it. Use the previously
  exported sanitized record as an explicitly recorded backup, or the published
  summary. An exported JSON file does not restore History. Any surviving History
  replay must retain **PREVIEW / SESSION REPLAY**, not be described as LIVE.
- **Primary question is unsuitable in rehearsal:** Alternate public **pub-025**:
  “How many nations competed in Judo at the 2016 Summer Olympics – Women's
  57 kg?” This illustrates sufficient initial evidence, not a promised
  follow-up. Use it instead of, not in addition to, the primary question.

Keep 40 seconds of contingency. Cut the vector screen change and detailed
reliability expansion first if needed; retain the vector sentence, evidence
inspection, and truthful stop explanation. Never force a tool call to improve
the demo. Family results remain aggregation 0 / 95.2 / 100, lookup 100 / 100 /
100, multi-hop 78.6 / 96.4 / 96.4, superlative 80 / 100 / 100, and temporal
72.7 / 100 / 100 (RAG / Fixed / Agentic).
