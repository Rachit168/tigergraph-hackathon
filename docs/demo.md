# Demo scenarios

Four questions from the official public set (`eval_public.jsonl`). Results
below describe the production architecture on those templates: BM25 RAG, one
typed GSQL call for Fixed GraphRAG, and bounded Agentic GraphRAG. Wording is
the canonical public text. Production routing was not changed for this demo.

## 1. Lookup

**Question** (`pub-025`):

> How many nations competed in Judo at the 2016 Summer Olympics – Women's 57 kg?

| System | Behavior |
|---|---|
| RAG | Answers from the event page when BM25 ranks it. Public lookup RAG is **19/19**. |
| Fixed GraphRAG | `lookup_event` returns nations on the Event vertex. |
| Agentic | Same lookup via primary `retrieve_spec`; no extra hop. |

This is the honest “text is enough” case. Graph still helps on families where
text is not enough.

## 2. Multi-hop / relational investigation

**Question** (`pub-017`):

> Who won the gold medal in the event held at Royal Artillery Barracks on
> 28 July 2012?

| System | Behavior |
|---|---|
| RAG | Relies on lexical overlap with venue/date strings in chunks. Shared-venue pages can overwhelm BM25. |
| Fixed GraphRAG | Resolves via one `events_at_venue_date` call. |
| Agentic | Same structured query first; `event_neighborhood` is available only if relationship evidence is still missing. |

**Why the graph changes the result:** typed venue + date overlap, not lexical
similarity to the venue name alone.

## 3. Temporal previous event

**Question** (`pub-002`):

> Who won the gold medal in the men's 20 kilometres walk athletics event at
> the Summer Olympics held immediately before 2016?

| System | Behavior |
|---|---|
| RAG | Sometimes succeeds when lexical overlap is enough; public temporal RAG is 77.3%. |
| Fixed GraphRAG | Uses `previous_event_gold` (named year 2016 → previous Summer olympiad, same event identity). |
| Agentic | Same primary structured query. |

**Why the graph still matters:** the relation is `prev` on Event, not “find a
page that mentions the previous Games.” On the public set both graph systems
are 100% temporal.

## 4. Aggregation (complete set)

**Question** (`pub-010`):

> According to the provided corpus, how many cycling events at the 2000 Summer
> Olympics had more than 30 competitors?

| System | Behavior |
|---|---|
| RAG | Typically abstains. BM25 can retrieve relevant event pages, but the extractive generator will not emit a count that is not a span in one chunk. Public aggregation for RAG is **0/21**. |
| Fixed GraphRAG | Answers the count from one `count_over_threshold` GSQL call over the complete matching Event set. |
| Agentic | Same structured count via primary `retrieve_spec`; no follow-up required. |

**Why the graph changes the result:** TigerGraph counts the full sport/year set
against a threshold. Text RAG has no complete-set operator.

## How to run a live demo

With `.env` configured and `OlympicGraph` up:

```
python -m scripts.eval_three_way --generator semantic
```

That is the public-facing live entry point. The same four public questions can
also be sent through `RAGAdapter`, `GraphRAGAdapter`, and
`AgenticGraphRAGAdapter`. Do not add a new demo binary.
