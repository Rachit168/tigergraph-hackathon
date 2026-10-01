"""Normalize installed-query payloads into GraphRAG evidence."""

from __future__ import annotations

from retrieval.graph.client import flatten_query_result
from retrieval.graph.results import GraphQueryResult
from retrieval.graphrag.models import GraphEvidence, GraphRef, GraphRetrievalResult
from retrieval.structured.models import QuerySpec

_COMPLETE_SET_OPS = frozenset({"count_over_threshold"})
_GOLD_OPS = frozenset({"previous_event_gold", "next_event_gold", "events_at_venue_date", "event_neighborhood"})
_NATIONS_OPS = frozenset({"lookup_nations", "lookup_event"})


def cardinality_for(result: GraphQueryResult) -> str:
    if result.status == "unresolved":
        return "unresolved"
    if result.status == "ambiguous":
        return "ambiguous"
    if result.status == "not_found":
        return "empty"
    if result.operation in _COMPLETE_SET_OPS and result.status == "supported":
        return "complete_set"
    if len(result.event_ids) == 0:
        return "empty"
    if len(result.event_ids) == 1:
        return "one"
    return "multiple"


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _str_or_none(value) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def retrieval_method_for(query_name: str) -> str:
    return f"gsql:{query_name}"


def why_for(operation: str, status: str, field: str | None = None) -> str:
    if field:
        return f"{operation} returned {field} with status={status}"
    return f"{operation} matched graph vertices with status={status}"


def normalize_graph_result(
    graph: GraphQueryResult,
    *,
    query_name: str,
    params: dict,
    tool_call_id: str,
    spec: QuerySpec | None = None,
    elapsed_ms: float = 0.0,
) -> GraphRetrievalResult:
    method = retrieval_method_for(query_name)
    cardinality = cardinality_for(graph)
    result = GraphRetrievalResult(
        operation=graph.operation,
        query_name=query_name,
        status=graph.status,
        cardinality=cardinality,
        tool_call_id=tool_call_id,
        retrieval_method=method,
        params=dict(params),
        reason=graph.reason,
        event_ids=list(graph.event_ids),
        notes=dict(graph.notes),
        elapsed_ms=elapsed_ms or graph.elapsed_ms,
        spec=spec,
        graph_result=graph,
    )
    titles = list(graph.titles)
    golds = list(graph.gold)
    nations = list(graph.nations)
    competitors = list(graph.competitors)
    for index, event_id in enumerate(graph.event_ids):
        title = titles[index] if index < len(titles) else None
        result.entities.append(
            GraphEvidence(
                evidence_id=f"{tool_call_id}:entity:{event_id}",
                evidence_type="entity",
                retrieval_method=method,
                tool_call_id=tool_call_id,
                graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
                event_id=event_id,
                field_name="title",
                value=title,
                why_retrieved=why_for(graph.operation, graph.status),
            )
        )
        if graph.status == "supported" and graph.operation in _NATIONS_OPS and index < len(nations):
            result.facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:nations:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="nations")],
                    event_id=event_id,
                    field_name="nations",
                    value=str(nations[index]),
                    why_retrieved=why_for(graph.operation, graph.status, "nations"),
                )
            )
        if graph.status == "supported" and graph.operation in _GOLD_OPS and index < len(golds) and golds[index]:
            result.facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:gold:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="gold_raw")],
                    event_id=event_id,
                    field_name="gold_raw",
                    value=str(golds[index]),
                    why_retrieved=why_for(graph.operation, graph.status, "gold_raw"),
                )
            )
        if graph.status == "supported" and graph.operation == "argmax_competitors" and index < len(competitors):
            result.facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:competitors:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="competitors")],
                    event_id=event_id,
                    field_name="competitors",
                    value=str(competitors[index]),
                    why_retrieved=why_for(graph.operation, graph.status, "competitors"),
                )
            )

    if graph.operation == "count_over_threshold" and graph.status == "supported" and graph.count is not None:
        result.facts.append(
            GraphEvidence(
                evidence_id=f"{tool_call_id}:fact:count",
                evidence_type="fact",
                retrieval_method=method,
                tool_call_id=tool_call_id,
                graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id) for event_id in graph.event_ids],
                field_name="count",
                value=str(graph.count),
                why_retrieved=why_for(graph.operation, graph.status, "complete event set count"),
            )
        )
        result.notes["set_size"] = len(graph.event_ids)
        result.notes["truncated_set"] = False
    if graph.max_competitors is not None and graph.status in {"supported", "ambiguous"}:
        result.notes["max_competitors"] = graph.max_competitors
    if graph.status != "supported":
        result.notes["answer_suppressed"] = True
    return result


def chunk_rows_from_payload(payload: dict) -> list[dict]:
    chunk_ids = _as_list(payload.get("chunk_ids"))
    document_ids = _as_list(payload.get("document_ids") or payload.get("chunk_document_ids"))
    event_ids = _as_list(payload.get("event_ids") or payload.get("chunk_event_ids"))
    texts = _as_list(payload.get("texts"))
    kinds = _as_list(payload.get("kinds"))
    sections = _as_list(payload.get("sections"))
    indexes = _as_list(payload.get("indexes"))
    urls = _as_list(payload.get("urls"))
    rows = []
    for index, chunk_id in enumerate(chunk_ids):
        if not chunk_id:
            continue
        rows.append(
            {
                "chunk_id": str(chunk_id),
                "document_id": _str_or_none(document_ids[index] if index < len(document_ids) else None),
                "event_id": _str_or_none(event_ids[index] if index < len(event_ids) else None),
                "text": _str_or_none(texts[index] if index < len(texts) else None),
                "kind": _str_or_none(kinds[index] if index < len(kinds) else None),
                "section": _str_or_none(sections[index] if index < len(sections) else None),
                "chunk_index": indexes[index] if index < len(indexes) else None,
                "source_url": _str_or_none(urls[index] if index < len(urls) else None),
            }
        )
    return rows


def evidence_from_chunks(
    rows: list[dict],
    *,
    tool_call_id: str,
    retrieval_method: str,
    observation_truncated: bool = False,
) -> list[GraphEvidence]:
    items: list[GraphEvidence] = []
    for row in rows:
        chunk_id = row["chunk_id"]
        event_id = row.get("event_id")
        document_id = row.get("document_id")
        refs = [GraphRef(vertex_type="Chunk", vertex_id=chunk_id)]
        if document_id:
            refs.append(GraphRef(vertex_type="Document", vertex_id=document_id, edge_type="CONTAINS_CHUNK"))
        if event_id:
            refs.append(GraphRef(vertex_type="Event", vertex_id=event_id, edge_type="DESCRIBES"))
        items.append(
            GraphEvidence(
                evidence_id=f"{tool_call_id}:chunk:{chunk_id}",
                evidence_type="chunk",
                retrieval_method=retrieval_method,
                tool_call_id=tool_call_id,
                graph_refs=refs,
                event_id=event_id,
                document_id=document_id,
                chunk_id=chunk_id,
                source_chunk_id=chunk_id,
                field_name=row.get("kind"),
                text=row.get("text"),
                source_url=row.get("source_url"),
                observation_truncated=observation_truncated,
                why_retrieved="typed DESCRIBES + CONTAINS_CHUNK hop from retrieved Event",
            )
        )
    return items


def evidence_from_neighborhood_payload(
    payload: dict,
    *,
    tool_call_id: str,
    query_name: str = "event_neighborhood",
) -> tuple[list[GraphEvidence], list[GraphEvidence], list[GraphEvidence], list[GraphEvidence]]:
    method = retrieval_method_for(query_name)
    event_ids = [str(value) for value in _as_list(payload.get("event_ids")) if value]
    titles = _as_list(payload.get("titles"))
    document_ids = _as_list(payload.get("document_ids"))
    golds = _as_list(payload.get("gold"))
    nations = _as_list(payload.get("nations"))
    competitors = _as_list(payload.get("competitors"))
    entities: list[GraphEvidence] = []
    facts: list[GraphEvidence] = []
    edges: list[GraphEvidence] = []
    for index, event_id in enumerate(event_ids):
        document_id = _str_or_none(document_ids[index] if index < len(document_ids) else None)
        entities.append(
            GraphEvidence(
                evidence_id=f"{tool_call_id}:entity:{event_id}",
                evidence_type="entity",
                retrieval_method=method,
                tool_call_id=tool_call_id,
                graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id)],
                event_id=event_id,
                document_id=document_id,
                field_name="title",
                value=_str_or_none(titles[index] if index < len(titles) else None),
                why_retrieved="event_neighborhood seed vertex",
            )
        )
        if index < len(golds) and golds[index]:
            facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:gold:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="gold_raw")],
                    event_id=event_id,
                    document_id=document_id,
                    field_name="gold_raw",
                    value=str(golds[index]),
                    why_retrieved=why_for(query_name, "supported", "gold_raw"),
                )
            )
        if index < len(nations):
            facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:nations:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="nations")],
                    event_id=event_id,
                    document_id=document_id,
                    field_name="nations",
                    value=str(nations[index]),
                    why_retrieved=why_for(query_name, "supported", "nations"),
                )
            )
        if index < len(competitors):
            facts.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:fact:competitors:{event_id}",
                    evidence_type="fact",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[GraphRef(vertex_type="Event", vertex_id=event_id, attribute="competitors")],
                    event_id=event_id,
                    document_id=document_id,
                    field_name="competitors",
                    value=str(competitors[index]),
                    why_retrieved=why_for(query_name, "supported", "competitors"),
                )
            )
    event_id = event_ids[0] if event_ids else None
    for vertex_type, key, edge_type in (
        ("Games", "games_ids", "IN_GAMES"),
        ("Sport", "sport_ids", "OF_SPORT"),
        ("Venue", "venue_ids", "HELD_AT"),
    ):
        for neighbor_id in _as_list(payload.get(key)):
            if not neighbor_id:
                continue
            edges.append(
                GraphEvidence(
                    evidence_id=f"{tool_call_id}:edge:{edge_type}:{neighbor_id}",
                    evidence_type="edge",
                    retrieval_method=method,
                    tool_call_id=tool_call_id,
                    graph_refs=[
                        GraphRef(vertex_type="Event", vertex_id=event_id, edge_type=edge_type),
                        GraphRef(vertex_type=vertex_type, vertex_id=str(neighbor_id)),
                    ],
                    event_id=event_id,
                    field_name=edge_type,
                    value=str(neighbor_id),
                    why_retrieved=f"typed {edge_type} hop from Event",
                )
            )
    chunk_payload = {
        "chunk_ids": payload.get("chunk_ids"),
        "chunk_document_ids": payload.get("chunk_document_ids"),
        "chunk_event_ids": payload.get("chunk_event_ids"),
        "texts": payload.get("texts"),
        "kinds": payload.get("kinds"),
        "sections": payload.get("sections"),
        "indexes": payload.get("indexes"),
        "urls": payload.get("urls"),
    }
    chunks = evidence_from_chunks(
        chunk_rows_from_payload(chunk_payload),
        tool_call_id=tool_call_id,
        retrieval_method=method,
    )
    return entities, facts, edges, chunks


def empty_result(
    *,
    operation: str,
    query_name: str,
    status: str,
    reason: str,
    tool_call_id: str,
    params: dict | None = None,
    spec: QuerySpec | None = None,
) -> GraphRetrievalResult:
    graph = GraphQueryResult(operation=operation, status=status, reason=reason)
    result = normalize_graph_result(
        graph,
        query_name=query_name,
        params=params or {},
        tool_call_id=tool_call_id,
        spec=spec,
    )
    result.cardinality = "unresolved" if status == "unresolved" else "empty"
    result.notes["answer_suppressed"] = True
    return result


def attach_chunks(result: GraphRetrievalResult, chunks: list[GraphEvidence]) -> GraphRetrievalResult:
    result.chunks = list(chunks)
    by_event = {item.event_id: item for item in chunks if item.event_id}
    by_doc = {item.document_id: item for item in chunks if item.document_id}
    for entity in result.entities:
        chunk = by_event.get(entity.event_id)
        if chunk is None:
            continue
        entity.document_id = entity.document_id or chunk.document_id
        entity.source_chunk_id = chunk.source_chunk_id
    for fact in result.facts:
        chunk = by_event.get(fact.event_id) if fact.event_id else None
        if chunk is None and fact.event_id is None and by_doc:
            continue
        if chunk is None:
            continue
        fact.document_id = fact.document_id or chunk.document_id
        fact.source_chunk_id = fact.source_chunk_id or chunk.source_chunk_id
    result.notes["supporting_chunk_count"] = len(chunks)
    return result


def flatten_payload(raw) -> dict:
    return flatten_query_result(raw)
