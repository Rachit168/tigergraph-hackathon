"""Recorded public examples built through export_submission_record.

These are static demonstration traces. They are not live TigerGraph results
and they do not include gold or scores.
"""

from __future__ import annotations

from typing import Any

from answering.models import Citation, PipelineTimings
from evaluation.harness import HarnessResult
from evaluation.trace_export import export_submission_record
from retrieval.agentic.trace import AgentTrace, PlanStep, SlotUpdate
from retrieval.agentic.tools import ToolAction, ToolObservation
from retrieval.graphrag.models import GraphEvidence, GraphRef

DEMO_QUESTION_IDS: tuple[str, ...] = ("pub-025", "pub-017", "pub-002", "pub-010")

DEMO_QUESTIONS: dict[str, dict[str, str]] = {
    "pub-025": {
        "qid": "pub-025",
        "qtype": "lookup",
        "question": "How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?",
        "demo_role": "Easy lookup",
    },
    "pub-017": {
        "qid": "pub-017",
        "qtype": "multi_hop",
        "question": "Who won the gold medal in the event held at Royal Artillery Barracks on 28 July 2012?",
        "demo_role": "Multi-hop / follow-up",
    },
    "pub-002": {
        "qid": "pub-002",
        "qtype": "temporal",
        "question": (
            "Who won the gold medal in the men's 20 kilometres walk athletics event at "
            "the Summer Olympics held immediately before 2016?"
        ),
        "demo_role": "Temporal previous event",
    },
    "pub-010": {
        "qid": "pub-010",
        "qtype": "aggregation",
        "question": (
            "According to the provided corpus, how many cycling events at the 2000 Summer "
            "Olympics had more than 30 competitors?"
        ),
        "demo_role": "Complete-set aggregation",
    },
}


def recorded_export_records() -> list[dict[str, Any]]:
    builders = (
        _pub025_rag,
        _pub025_graphrag,
        _pub025_agentic,
        _pub017_rag,
        _pub017_graphrag,
        _pub017_agentic,
        _pub002_rag,
        _pub002_graphrag,
        _pub002_agentic,
        _pub010_rag,
        _pub010_graphrag,
        _pub010_agentic,
    )
    records = [builder() for builder in builders]
    for record in records:
        if record.get("question_id") not in DEMO_QUESTION_IDS:
            raise RuntimeError(f"non-demo qid in recorded traces: {record.get('question_id')}")
    return records


def _export(result: HarnessResult, *, qid: str, pipeline: str) -> dict[str, Any]:
    return export_submission_record(result, question_id=qid, pipeline=pipeline)


def _citation(*, evidence_id: str, document_id: str, chunk_id: str | None, event_id: str) -> Citation:
    return Citation(
        evidence_id=evidence_id,
        document_id=document_id,
        chunk_id=chunk_id,
        event_id=event_id,
        source_url=f"https://en.wikipedia.org/wiki/{document_id}",
    )


def _notes(*, tokens: int, model_calls: int = 1) -> dict[str, Any]:
    return {
        "tokens": tokens,
        "model_calls": model_calls,
        "tokens_unknown": False,
        "attempts": [{"attempt": 1, "ok": True}],
    }


def _entity(
    event_id: str,
    title: str,
    *,
    call: str,
    method: str,
    document_id: str,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call}:entity:{event_id}",
        evidence_type="entity",
        retrieval_method=method,
        tool_call_id=call,
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
        event_id=event_id,
        document_id=document_id,
        field_name="title",
        value=title,
        why_retrieved="typed Event vertex from GSQL retrieval",
    )


def _fact(
    event_id: str,
    field: str,
    value: str,
    *,
    call: str,
    method: str,
    document_id: str,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call}:fact:{field}:{event_id}",
        evidence_type="fact",
        retrieval_method=method,
        tool_call_id=call,
        graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute=field)],
        event_id=event_id,
        document_id=document_id,
        field_name=field,
        value=value,
        why_retrieved=f"Event.{field} from typed retrieval",
    )


def _edge(
    event_id: str,
    vertex_type: str,
    neighbor_id: str,
    edge_type: str,
    *,
    call: str,
    method: str,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call}:edge:{edge_type}:{neighbor_id}",
        evidence_type="edge",
        retrieval_method=method,
        tool_call_id=call,
        graph_refs=[
            GraphRef(vertex_type="Event", vertex_id=event_id, edge_type=edge_type),
            GraphRef(vertex_type=vertex_type, vertex_id=neighbor_id),
        ],
        event_id=event_id,
        field_name=edge_type,
        value=neighbor_id,
        why_retrieved=f"typed {edge_type} hop from Event",
    )


def _chunk(
    event_id: str,
    chunk_id: str,
    text: str,
    *,
    call: str,
    method: str,
    document_id: str,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"{call}:chunk:{chunk_id}",
        evidence_type="chunk",
        retrieval_method=method,
        tool_call_id=call,
        graph_refs=[
            GraphRef(vertex_type="Chunk", vertex_id=chunk_id),
            GraphRef(vertex_type="Document", vertex_id=document_id),
            GraphRef(vertex_type="Event", vertex_id=event_id),
        ],
        event_id=event_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source_chunk_id=chunk_id,
        text=text,
        why_retrieved="typed DESCRIBES + CONTAINS_CHUNK hop from retrieved Event",
    )


def _graph_harness(
    *,
    system_name: str,
    question: str,
    answer: str,
    status: str,
    citations: list[Citation],
    evidence: list[GraphEvidence],
    timings: PipelineTimings,
    metadata: dict[str, Any],
    notes: dict[str, Any],
) -> HarnessResult:
    payload = dict(metadata)
    payload["pipeline_notes"] = dict(notes)
    payload["generator_notes"] = dict(notes)
    return HarnessResult(
        system_name=system_name,
        question=question,
        answer=answer,
        citations=citations,
        latency_ms=timings.total_ms,
        retrieval_metadata=payload,
        evidence_metadata=[item.to_dict() for item in evidence],
        status=status,
        timings=timings,
    )


def _rag_chunk(
    *,
    document_id: str,
    chunk_id: str,
    event_id: str | None,
    text: str,
    rank: int,
    score: float,
) -> dict[str, Any]:
    return {
        "evidence_id": f"rag:chunk:{chunk_id}",
        "evidence_type": "chunk",
        "retrieval_method": "bm25",
        "tool_call_id": "bm25",
        "graph_refs": [],
        "event_id": event_id,
        "document_id": document_id,
        "chunk_id": chunk_id,
        "source_chunk_id": chunk_id,
        "text": text,
        "retrieval_score": score,
        "why_retrieved": f"BM25 rank {rank}",
    }


def _rag_harness(
    *,
    question: str,
    answer: str,
    status: str,
    citations: list[Citation],
    evidence: list[dict[str, Any]],
    timings: PipelineTimings,
    notes: dict[str, Any],
) -> HarnessResult:
    return HarnessResult(
        system_name="rag",
        question=question,
        answer=answer,
        citations=citations,
        latency_ms=timings.total_ms,
        retrieval_metadata={
            "query": question,
            "method": "sparse",
            "elapsed_ms": timings.retrieval_ms,
            "hit_count": len(evidence),
            "generator_notes": dict(notes),
            "retrieval_methods": ["bm25"],
        },
        evidence_metadata=evidence,
        status=status,
        timings=timings,
    )


def _action(tool: str, arguments: dict[str, Any], reason: str, group: str) -> ToolAction:
    return ToolAction(tool=tool, arguments=arguments, reason=reason, parallel_group=group)


def _observation(
    *,
    call: str,
    tool: str,
    arguments: dict[str, Any],
    reason: str,
    elapsed_ms: float,
    evidence_ids: list[str],
    method: str,
    event_ids: list[str],
    group: str,
    started_ms: float = 1.0,
) -> ToolObservation:
    action = _action(tool, arguments, reason, group)
    return ToolObservation(
        tool_call_id=call,
        tool=tool,
        arguments=arguments,
        reason=reason,
        success=True,
        started_ms=started_ms,
        elapsed_ms=elapsed_ms,
        fingerprint=action.fingerprint(),
        evidence_ids=evidence_ids,
        retrieval_method=method,
        parallel_group=group,
    )


def _pub025_rag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-025"]
    chunk_id = "Q26217865::c000"
    evidence = [
        _rag_chunk(
            document_id="Q26217865",
            chunk_id=chunk_id,
            event_id="Q26217865",
            text="Judo at the 2016 Summer Olympics – Women's 57 kg. Nations: 23.",
            rank=1,
            score=18.4,
        )
    ]
    citations = [_citation(evidence_id=evidence[0]["evidence_id"], document_id="Q26217865", chunk_id=chunk_id, event_id="Q26217865")]
    timings = PipelineTimings(retrieval_ms=42.0, packing_ms=4.0, generation_ms=810.0, total_ms=860.0)
    return _export(
        _rag_harness(
            question=q["question"],
            answer="23",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            notes=_notes(tokens=640),
        ),
        qid="pub-025",
        pipeline="rag",
    )


def _pub025_graphrag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-025"]
    method = "gsql:lookup_event"
    event_id = "Q26217865"
    evidence = [
        _entity(event_id, "Judo at the 2016 Summer Olympics – Women's 57 kg", call="g1", method=method, document_id=event_id),
        _fact(event_id, "nations", "23", call="g1", method=method, document_id=event_id),
        _fact(event_id, "year", "2016", call="g1", method=method, document_id=event_id),
        _edge(event_id, "Games", "2016_Summer", "IN_GAMES", call="g1", method=method),
        _edge(event_id, "Sport", "judo", "OF_SPORT", call="g1", method=method),
        _chunk(event_id, "Q26217865::c000", "Nations: 23.", call="g1", method="gsql:chunks_for_events", document_id=event_id),
    ]
    citations = [_citation(evidence_id="g1:fact:nations:Q26217865", document_id=event_id, chunk_id="Q26217865::c000", event_id=event_id)]
    timings = PipelineTimings(parsing_ms=3.0, retrieval_ms=38.0, packing_ms=5.0, generation_ms=720.0, total_ms=770.0)
    return _export(
        _graph_harness(
            system_name="graphrag",
            question=q["question"],
            answer="23",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            metadata={
                "operation": "lookup_nations",
                "query_name": "lookup_event",
                "status": "supported",
                "cardinality": "one",
                "retrieval_method": method,
                "event_ids": [event_id],
                "retrieval_methods": [method, "gsql:chunks_for_events"],
            },
            notes=_notes(tokens=410),
        ),
        qid="pub-025",
        pipeline="graphrag",
    )


def _pub025_agentic() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-025"]
    method = "gsql:lookup_event"
    event_id = "Q26217865"
    action = _action(
        "retrieve_spec",
        {"operation": "lookup_nations", "include_chunks": False},
        "lookup needs an Event title match, not a text snippet ranking",
        "primary",
    )
    observation = _observation(
        call="a1",
        tool="retrieve_spec",
        arguments=action.arguments,
        reason=action.reason,
        elapsed_ms=36.0,
        evidence_ids=["a1:entity:Q26217865", "a1:fact:nations:Q26217865"],
        method=method,
        event_ids=[event_id],
        group="primary",
    )
    evidence = [
        _entity(event_id, "Judo at the 2016 Summer Olympics – Women's 57 kg", call="a1", method=method, document_id=event_id),
        _fact(event_id, "nations", "23", call="a1", method=method, document_id=event_id),
        _edge(event_id, "Games", "2016_Summer", "IN_GAMES", call="a1", method=method),
        _edge(event_id, "Sport", "judo", "OF_SPORT", call="a1", method=method),
    ]
    trace = AgentTrace(
        question=q["question"],
        interpreted={"qtype": "lookup", "operation": "lookup_nations", "status": "parsed"},
        plan_steps=[PlanStep(iteration=0, kind="primary", reason=action.reason, actions=[action])],
        tool_calls=[observation],
        slot_updates=[
            SlotUpdate(iteration=0, slot_id="target_event", status="filled", evidence_ids=["a1:entity:Q26217865"]),
            SlotUpdate(iteration=0, slot_id="answer_field", status="filled", evidence_ids=["a1:fact:nations:Q26217865"]),
        ],
        follow_up_decisions=["stop:answered"],
        strategy_changes=[],
        stop_reason="answered",
        total_steps=1,
        total_tool_calls=1,
        retrieval_methods=[method],
        timings={"retrieval_ms": 36.0, "generation_ms": 690.0, "total_ms": 740.0},
    )
    citations = [_citation(evidence_id="a1:fact:nations:Q26217865", document_id=event_id, chunk_id=None, event_id=event_id)]
    timings = PipelineTimings(parsing_ms=3.0, retrieval_ms=36.0, packing_ms=4.0, generation_ms=690.0, total_ms=740.0)
    harness = _graph_harness(
        system_name="agentic_graphrag",
        question=q["question"],
        answer="23",
        status="answered",
        citations=citations,
        evidence=evidence,
        timings=timings,
        metadata={
            "operation": "lookup_nations",
            "query_name": "lookup_event",
            "retrieval_method": method,
            "event_ids": [event_id],
            "trace": trace.to_dict(),
            "stop_reason": trace.stop_reason,
            "total_tool_calls": 1,
            "retrieval_methods": [method],
        },
        notes=_notes(tokens=390),
    )
    return _export(harness, qid="pub-025", pipeline="agentic_graphrag")


def _pub017_rag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-017"]
    evidence = [
        _rag_chunk(
            document_id="Q1137721",
            chunk_id="Q1137721::c000",
            event_id="Q1137721",
            text="Shooting at the 2012 Summer Olympics – Women's 10 metre air rifle was held at the Royal Artillery Barracks.",
            rank=1,
            score=12.1,
        ),
        _rag_chunk(
            document_id="Q749637",
            chunk_id="Q749637::c002",
            event_id=None,
            text="The Royal Artillery Barracks in Woolwich hosted several shooting events at London 2012.",
            rank=2,
            score=11.4,
        ),
    ]
    citations = [
        _citation(evidence_id=evidence[0]["evidence_id"], document_id="Q1137721", chunk_id="Q1137721::c000", event_id="Q1137721")
    ]
    timings = PipelineTimings(retrieval_ms=51.0, packing_ms=6.0, generation_ms=940.0, total_ms=1010.0)
    return _export(
        _rag_harness(
            question=q["question"],
            answer="Yi Siling",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            notes=_notes(tokens=880),
        ),
        qid="pub-017",
        pipeline="rag",
    )


def _pub017_graphrag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-017"]
    method = "gsql:events_at_venue_date"
    event_id = "Q1137721"
    evidence = [
        _entity(event_id, "Shooting at the 2012 Summer Olympics – Women's 10 metre air rifle", call="g1", method=method, document_id=event_id),
        _fact(event_id, "gold_raw", "Yi Siling", call="g1", method=method, document_id=event_id),
        _fact(event_id, "date_raw", "28 July 2012", call="g1", method=method, document_id=event_id),
        _edge(event_id, "Venue", "royal_artillery_barracks", "HELD_AT", call="g1", method=method),
        _edge(event_id, "Games", "2012_Summer", "IN_GAMES", call="g1", method=method),
        _chunk(
            event_id,
            "Q1137721::c000",
            "Gold: Yi Siling. Venue: Royal Artillery Barracks. Date: 28 July 2012.",
            call="g1",
            method="gsql:chunks_for_events",
            document_id=event_id,
        ),
    ]
    citations = [_citation(evidence_id="g1:fact:gold_raw:Q1137721", document_id=event_id, chunk_id="Q1137721::c000", event_id=event_id)]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=44.0, packing_ms=6.0, generation_ms=760.0, total_ms=820.0)
    return _export(
        _graph_harness(
            system_name="graphrag",
            question=q["question"],
            answer="Yi Siling",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            metadata={
                "operation": "events_at_venue_date",
                "query_name": "events_at_venue_date",
                "status": "supported",
                "cardinality": "one",
                "retrieval_method": method,
                "event_ids": [event_id],
                "retrieval_methods": [method, "gsql:chunks_for_events"],
            },
            notes=_notes(tokens=520),
        ),
        qid="pub-017",
        pipeline="graphrag",
    )


def _pub017_agentic() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-017"]
    event_id = "Q1137721"
    primary = _action(
        "retrieve_spec",
        {"operation": "events_at_venue_date", "include_chunks": False},
        "venue/date question needs typed Event hops rather than lexical lookup",
        "primary",
    )
    neighborhood = _action(
        "event_neighborhood",
        {"event_id": event_id},
        "primary venue/date hit left a relational gap: typed HELD_AT/IN_GAMES hops are still missing",
        "repair",
    )
    chunks = _action(
        "supporting_chunks",
        {"event_ids": [event_id], "max_extra": 0},
        "unique multi-hop Event still needs provenance chunks",
        "repair",
    )
    obs_primary = _observation(
        call="a1",
        tool="retrieve_spec",
        arguments=primary.arguments,
        reason=primary.reason,
        elapsed_ms=41.0,
        evidence_ids=["a1:entity:Q1137721", "a1:fact:gold_raw:Q1137721"],
        method="gsql:events_at_venue_date",
        event_ids=[event_id],
        group="primary",
    )
    obs_n = _observation(
        call="a2",
        tool="event_neighborhood",
        arguments=neighborhood.arguments,
        reason=neighborhood.reason,
        elapsed_ms=28.0,
        evidence_ids=[
            "a2:entity:Q1137721",
            "a2:edge:HELD_AT:royal_artillery_barracks",
            "a2:edge:IN_GAMES:2012_Summer",
        ],
        method="gsql:event_neighborhood",
        event_ids=[event_id],
        group="repair",
        started_ms=42.0,
    )
    obs_c = _observation(
        call="a3",
        tool="supporting_chunks",
        arguments=chunks.arguments,
        reason=chunks.reason,
        elapsed_ms=22.0,
        evidence_ids=["a3:chunk:Q1137721::c000"],
        method="gsql:chunks_for_events",
        event_ids=[event_id],
        group="repair",
        started_ms=42.0,
    )
    evidence = [
        _entity(event_id, "Shooting at the 2012 Summer Olympics – Women's 10 metre air rifle", call="a1", method="gsql:events_at_venue_date", document_id=event_id),
        _fact(event_id, "gold_raw", "Yi Siling", call="a1", method="gsql:events_at_venue_date", document_id=event_id),
        _edge(event_id, "Venue", "royal_artillery_barracks", "HELD_AT", call="a2", method="gsql:event_neighborhood"),
        _edge(event_id, "Games", "2012_Summer", "IN_GAMES", call="a2", method="gsql:event_neighborhood"),
        _chunk(
            event_id,
            "Q1137721::c000",
            "Gold: Yi Siling. Held at the Royal Artillery Barracks on 28 July 2012.",
            call="a3",
            method="gsql:chunks_for_events",
            document_id=event_id,
        ),
    ]
    trace = AgentTrace(
        question=q["question"],
        interpreted={"qtype": "multi_hop", "operation": "events_at_venue_date", "status": "parsed"},
        plan_steps=[
            PlanStep(iteration=0, kind="primary", reason=primary.reason, actions=[primary]),
            PlanStep(iteration=1, kind="follow_up", reason=neighborhood.reason, actions=[neighborhood, chunks]),
        ],
        tool_calls=[obs_primary, obs_n, obs_c],
        slot_updates=[
            SlotUpdate(iteration=0, slot_id="target_event", status="filled", evidence_ids=["a1:entity:Q1137721"]),
            SlotUpdate(iteration=1, slot_id="neighborhood", status="filled", evidence_ids=["a2:edge:HELD_AT:royal_artillery_barracks"]),
        ],
        follow_up_decisions=[
            neighborhood.reason,
            "stop:answered",
        ],
        strategy_changes=["primary_to_neighborhood", "neighborhood_repair"],
        stop_reason="answered",
        total_steps=2,
        total_tool_calls=3,
        retrieval_methods=["gsql:events_at_venue_date", "gsql:event_neighborhood", "gsql:chunks_for_events"],
        timings={"retrieval_ms": 91.0, "generation_ms": 710.0, "total_ms": 830.0},
        parallel_groups=["primary", "repair"],
    )
    citations = [_citation(evidence_id="a1:fact:gold_raw:Q1137721", document_id=event_id, chunk_id="Q1137721::c000", event_id=event_id)]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=91.0, packing_ms=6.0, generation_ms=710.0, total_ms=830.0)
    harness = _graph_harness(
        system_name="agentic_graphrag",
        question=q["question"],
        answer="Yi Siling",
        status="answered",
        citations=citations,
        evidence=evidence,
        timings=timings,
        metadata={
            "operation": "events_at_venue_date",
            "query_name": "events_at_venue_date",
            "trace": trace.to_dict(),
            "stop_reason": trace.stop_reason,
            "total_tool_calls": 3,
            "retrieval_methods": list(trace.retrieval_methods),
            "event_ids": [event_id],
        },
        notes=_notes(tokens=560, model_calls=1),
    )
    return _export(harness, qid="pub-017", pipeline="agentic_graphrag")


def _pub002_rag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-002"]
    evidence = [
        _rag_chunk(
            document_id="Q26233122",
            chunk_id="Q26233122::c000",
            event_id="Q26233122",
            text="Athletics at the 2016 Summer Olympics – Men's 20 kilometres walk.",
            rank=1,
            score=14.2,
        ),
        _rag_chunk(
            document_id="Q1050909",
            chunk_id="Q1050909::c001",
            event_id="Q1050909",
            text="Chen Ding won the gold medal in the men's 20 kilometres walk at the 2012 Summer Olympics.",
            rank=2,
            score=9.8,
        ),
    ]
    citations = [
        _citation(evidence_id=evidence[1]["evidence_id"], document_id="Q1050909", chunk_id="Q1050909::c001", event_id="Q1050909")
    ]
    timings = PipelineTimings(retrieval_ms=48.0, packing_ms=5.0, generation_ms=900.0, total_ms=960.0)
    return _export(
        _rag_harness(
            question=q["question"],
            answer="Chen Ding",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            notes=_notes(tokens=910),
        ),
        qid="pub-002",
        pipeline="rag",
    )


def _pub002_graphrag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-002"]
    method = "gsql:previous_event_gold"
    event_id = "Q1050909"
    evidence = [
        _entity(event_id, "Athletics at the 2012 Summer Olympics – Men's 20 kilometres walk", call="g1", method=method, document_id=event_id),
        _fact(event_id, "gold_raw", "Chen Ding", call="g1", method=method, document_id=event_id),
        _fact(event_id, "year", "2012", call="g1", method=method, document_id=event_id),
        _fact(event_id, "prev_year", "2008", call="g1", method=method, document_id=event_id),
        _edge(event_id, "Games", "2012_Summer", "IN_GAMES", call="g1", method=method),
        _edge(event_id, "Sport", "athletics", "OF_SPORT", call="g1", method=method),
        _chunk(event_id, "Q1050909::c001", "Gold: Chen Ding.", call="g1", method="gsql:chunks_for_events", document_id=event_id),
    ]
    citations = [_citation(evidence_id="g1:fact:gold_raw:Q1050909", document_id=event_id, chunk_id="Q1050909::c001", event_id=event_id)]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=40.0, packing_ms=5.0, generation_ms=700.0, total_ms=755.0)
    return _export(
        _graph_harness(
            system_name="graphrag",
            question=q["question"],
            answer="Chen Ding",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            metadata={
                "operation": "previous_event_gold",
                "query_name": "previous_event_gold",
                "status": "supported",
                "cardinality": "one",
                "retrieval_method": method,
                "event_ids": [event_id],
                "retrieval_methods": [method, "gsql:chunks_for_events"],
            },
            notes=_notes(tokens=480),
        ),
        qid="pub-002",
        pipeline="graphrag",
    )


def _pub002_agentic() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-002"]
    method = "gsql:previous_event_gold"
    event_id = "Q1050909"
    action = _action(
        "retrieve_spec",
        {"operation": "previous_event_gold", "include_chunks": False},
        "temporal question needs previous/next Event gold, not the named year page",
        "primary",
    )
    observation = _observation(
        call="a1",
        tool="retrieve_spec",
        arguments=action.arguments,
        reason=action.reason,
        elapsed_ms=39.0,
        evidence_ids=["a1:entity:Q1050909", "a1:fact:gold_raw:Q1050909"],
        method=method,
        event_ids=[event_id],
        group="primary",
    )
    evidence = [
        _entity(event_id, "Athletics at the 2012 Summer Olympics – Men's 20 kilometres walk", call="a1", method=method, document_id=event_id),
        _fact(event_id, "gold_raw", "Chen Ding", call="a1", method=method, document_id=event_id),
        _edge(event_id, "Games", "2012_Summer", "IN_GAMES", call="a1", method=method),
        _edge(event_id, "Sport", "athletics", "OF_SPORT", call="a1", method=method),
    ]
    trace = AgentTrace(
        question=q["question"],
        interpreted={"qtype": "temporal", "operation": "previous_event_gold", "status": "parsed"},
        plan_steps=[PlanStep(iteration=0, kind="primary", reason=action.reason, actions=[action])],
        tool_calls=[observation],
        slot_updates=[
            SlotUpdate(iteration=0, slot_id="target_event", status="filled", evidence_ids=["a1:entity:Q1050909"]),
            SlotUpdate(iteration=0, slot_id="answer_field", status="filled", evidence_ids=["a1:fact:gold_raw:Q1050909"]),
        ],
        follow_up_decisions=["stop:answered"],
        strategy_changes=[],
        stop_reason="answered",
        total_steps=1,
        total_tool_calls=1,
        retrieval_methods=[method],
        timings={"retrieval_ms": 39.0, "generation_ms": 680.0, "total_ms": 730.0},
    )
    citations = [_citation(evidence_id="a1:fact:gold_raw:Q1050909", document_id=event_id, chunk_id=None, event_id=event_id)]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=39.0, packing_ms=4.0, generation_ms=680.0, total_ms=730.0)
    harness = _graph_harness(
        system_name="agentic_graphrag",
        question=q["question"],
        answer="Chen Ding",
        status="answered",
        citations=citations,
        evidence=evidence,
        timings=timings,
        metadata={
            "operation": "previous_event_gold",
            "query_name": "previous_event_gold",
            "trace": trace.to_dict(),
            "stop_reason": trace.stop_reason,
            "total_tool_calls": 1,
            "retrieval_methods": [method],
            "event_ids": [event_id],
        },
        notes=_notes(tokens=430),
    )
    return _export(harness, qid="pub-002", pipeline="agentic_graphrag")


def _pub010_rag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-010"]
    evidence = [
        _rag_chunk(
            document_id="Q1856784",
            chunk_id="Q1856784::c000",
            event_id="Q1856784",
            text="Cycling at the 2000 Summer Olympics included track and road events in Sydney.",
            rank=1,
            score=8.6,
        ),
        _rag_chunk(
            document_id="Q2133123",
            chunk_id="Q2133123::c000",
            event_id="Q2133123",
            text="The men's sprint cycling event at the 2000 Summer Olympics.",
            rank=2,
            score=7.9,
        ),
    ]
    timings = PipelineTimings(retrieval_ms=55.0, packing_ms=6.0, generation_ms=820.0, total_ms=890.0)
    return _export(
        _rag_harness(
            question=q["question"],
            answer="",
            status="abstained",
            citations=[],
            evidence=evidence,
            timings=timings,
            notes=_notes(tokens=760),
        ),
        qid="pub-010",
        pipeline="rag",
    )


def _pub010_graphrag() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-010"]
    method = "gsql:count_over_threshold"
    event_ids = ["Q1856784", "Q2133123", "Q2426715", "Q2526348"]
    evidence = [
        _fact(event_ids[0], "count", "4", call="g1", method=method, document_id=event_ids[0]),
        _entity(event_ids[0], "Cycling at the 2000 Summer Olympics – Men's sprint", call="g1", method=method, document_id=event_ids[0]),
        _entity(event_ids[1], "Cycling at the 2000 Summer Olympics – Men's keirin", call="g1", method=method, document_id=event_ids[1]),
        _edge(event_ids[0], "Games", "2000_Summer", "IN_GAMES", call="g1", method=method),
        _edge(event_ids[0], "Sport", "cycling", "OF_SPORT", call="g1", method=method),
        _chunk(event_ids[0], "Q1856784::c000", "Competitors exceeded 30.", call="g1", method="gsql:chunks_for_events", document_id=event_ids[0]),
    ]
    citations = [_citation(evidence_id=f"g1:fact:count:{event_ids[0]}", document_id=event_ids[0], chunk_id="Q1856784::c000", event_id=event_ids[0])]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=47.0, packing_ms=6.0, generation_ms=640.0, total_ms=705.0)
    return _export(
        _graph_harness(
            system_name="graphrag",
            question=q["question"],
            answer="4",
            status="answered",
            citations=citations,
            evidence=evidence,
            timings=timings,
            metadata={
                "operation": "count_over_threshold",
                "query_name": "count_over_threshold",
                "status": "supported",
                "cardinality": "complete_set",
                "retrieval_method": method,
                "event_ids": event_ids,
                "retrieval_methods": [method, "gsql:chunks_for_events"],
            },
            notes=_notes(tokens=360),
        ),
        qid="pub-010",
        pipeline="graphrag",
    )


def _pub010_agentic() -> dict[str, Any]:
    q = DEMO_QUESTIONS["pub-010"]
    method = "gsql:count_over_threshold"
    action = _action(
        "retrieve_spec",
        {"operation": "count_over_threshold", "include_chunks": False},
        "aggregation requires the complete Event set for COUNT, not top-k chunks",
        "primary",
    )
    observation = _observation(
        call="a1",
        tool="retrieve_spec",
        arguments=action.arguments,
        reason=action.reason,
        elapsed_ms=46.0,
        evidence_ids=["a1:fact:count:Q1856784"],
        method=method,
        event_ids=["Q1856784", "Q2133123", "Q2426715", "Q2526348"],
        group="primary",
    )
    evidence = [
        _fact("Q1856784", "count", "4", call="a1", method=method, document_id="Q1856784"),
        _entity("Q1856784", "Cycling at the 2000 Summer Olympics – Men's sprint", call="a1", method=method, document_id="Q1856784"),
        _edge("Q1856784", "Games", "2000_Summer", "IN_GAMES", call="a1", method=method),
        _edge("Q1856784", "Sport", "cycling", "OF_SPORT", call="a1", method=method),
    ]
    trace = AgentTrace(
        question=q["question"],
        interpreted={"qtype": "aggregation", "operation": "count_over_threshold", "status": "parsed"},
        plan_steps=[PlanStep(iteration=0, kind="primary", reason=action.reason, actions=[action])],
        tool_calls=[observation],
        slot_updates=[
            SlotUpdate(iteration=0, slot_id="complete_count", status="filled", evidence_ids=["a1:fact:count:Q1856784"])
        ],
        follow_up_decisions=["stop:answered"],
        strategy_changes=[],
        stop_reason="answered",
        total_steps=1,
        total_tool_calls=1,
        retrieval_methods=[method],
        timings={"retrieval_ms": 46.0, "generation_ms": 610.0, "total_ms": 670.0},
    )
    citations = [_citation(evidence_id="a1:fact:count:Q1856784", document_id="Q1856784", chunk_id=None, event_id="Q1856784")]
    timings = PipelineTimings(parsing_ms=4.0, retrieval_ms=46.0, packing_ms=4.0, generation_ms=610.0, total_ms=670.0)
    harness = _graph_harness(
        system_name="agentic_graphrag",
        question=q["question"],
        answer="4",
        status="answered",
        citations=citations,
        evidence=evidence,
        timings=timings,
        metadata={
            "operation": "count_over_threshold",
            "query_name": "count_over_threshold",
            "trace": trace.to_dict(),
            "stop_reason": trace.stop_reason,
            "total_tool_calls": 1,
            "retrieval_methods": [method],
            "event_ids": ["Q1856784", "Q2133123", "Q2426715", "Q2526348"],
            "cardinality": "complete_set",
        },
        notes=_notes(tokens=340),
    )
    return _export(harness, qid="pub-010", pipeline="agentic_graphrag")
