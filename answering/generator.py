"""Provider-neutral generation interface and deterministic grounded default."""

from __future__ import annotations

from typing import Protocol

from answering.models import GeneratorRequest, GeneratorResult, PackedContextItem

_EMPTY_STATUSES = frozenset(
    {
        "ambiguous",
        "not_found",
        "unresolved",
        "unsupported",
        "abstained",
        "generation_error",
    }
)


class Generator(Protocol):
    def generate(self, request: GeneratorRequest) -> GeneratorResult: ...


def public_generation_status(generated: GeneratorResult) -> str:
    """Map a generator payload onto pipeline answer status without inventing text.

    Empty/suppressed statuses win even if answer_text is accidentally non-empty.
    """

    if generated.status in _EMPTY_STATUSES:
        return generated.status
    if generated.answer_text.strip():
        return "answered"
    return "abstained"


class DeterministicGroundedGenerator:
    """Render only explicit structured evidence; never infer from chunk prose."""

    def generate(self, request: GeneratorRequest) -> GeneratorResult:
        context = request.context
        if context.answer_suppressed or context.retrieval_status != "supported":
            return GeneratorResult(
                answer_text="",
                status="abstained",
                warnings=("retrieval_answer_suppressed",),
            )

        operation = str(context.notes.get("operation") or "")
        selected = self._dedupe_values(self._select(operation, context.items))
        if not selected:
            return GeneratorResult(
                answer_text="",
                status="abstained",
                warnings=("no_explicit_structured_answer_fact",),
            )

        values = [item.value for item in selected if item.value is not None]
        if not values:
            return GeneratorResult(
                answer_text="",
                status="abstained",
                warnings=("selected_evidence_has_no_value",),
            )
        return GeneratorResult(
            answer_text=" | ".join(values),
            citation_ids=tuple(item.evidence_id for item in selected),
            status="answered",
            notes={"deterministic": True},
        )

    @staticmethod
    def _dedupe_values(items: list[PackedContextItem]) -> list[PackedContextItem]:
        unique: list[PackedContextItem] = []
        seen: set[str] = set()
        for item in items:
            if item.value is None:
                continue
            if item.value in seen:
                continue
            seen.add(item.value)
            unique.append(item)
        return unique

    @staticmethod
    def _select(
        operation: str,
        items: tuple[PackedContextItem, ...],
    ) -> list[PackedContextItem]:
        if operation in {"lookup_event", "lookup_nations"}:
            return [item for item in items if item.evidence_type == "fact" and item.field_name == "nations"]
        if operation == "count_over_threshold":
            return [item for item in items if item.evidence_type == "fact" and item.field_name == "count"]
        if operation == "argmax_competitors":
            return [item for item in items if item.evidence_type == "entity" and item.field_name == "title"]
        if operation in {"previous_event_gold", "next_event_gold", "events_at_venue_date"}:
            return [item for item in items if item.evidence_type == "fact" and item.field_name == "gold_raw"]
        return []
