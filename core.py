"""Unit effects, comparison completion, and R0."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
import math

from .types import Effect, ExecutionRecord, Pair, Proposal

__all__ = [
    "canonical_pair", "normalize_proposals", "compute_effect",
    "merge_candidates", "completion_count", "select_queries",
    "update_scores", "rank_units",
]


def _unit_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Unit IDs must be nonempty strings")
    return value


def _ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a collection of IDs, not one string")
    try:
        return tuple(_unit_id(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of IDs") from exc


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    try:
        result = tuple(_finite(value, name=name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty numeric sequence") from exc
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def canonical_pair(a: str, b: str) -> Pair:
    a, b = _unit_id(a), _unit_id(b)
    if a == b:
        raise ValueError("A comparison requires two distinct Unit IDs")
    return (a, b) if a < b else (b, a)


def _pairs(values: Iterable[Pair]) -> tuple[Pair, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("Pairs must be an iterable of two-item collections")
    unique: set[Pair] = set()
    try:
        for pair in values:
            if isinstance(pair, (str, bytes)):
                raise ValueError("A pair must contain two Unit IDs")
            endpoints = tuple(pair)
            if len(endpoints) != 2:
                raise ValueError("A pair must contain exactly two Unit IDs")
            unique.add(canonical_pair(*endpoints))
    except TypeError as exc:
        raise ValueError("Pairs must contain two Unit IDs each") from exc
    return tuple(sorted(unique))


def normalize_proposals(
    proposals: Iterable[Proposal],
    unit_ids: Iterable[str],
    allowed_evidence_ids: Iterable[str] | None = None,
) -> tuple[Proposal, ...]:
    """Validate the three-card batch before deduplication; preserve proposal meaning."""
    catalog = _ids(unit_ids, name="Unit catalog")
    if not catalog or len(set(catalog)) != len(catalog):
        raise ValueError("Unit catalog must be nonempty and contain unique IDs")
    allowed_units = set(catalog)
    allowed_evidence = (
        None if allowed_evidence_ids is None
        else set(_ids(allowed_evidence_ids, name="Allowed evidence IDs"))
    )
    try:
        cards = tuple(proposals)
    except TypeError as exc:
        raise ValueError("Proposals must be an iterable of Proposal records") from exc
    if len(cards) > 3:
        raise ValueError("At most three proposal cards are allowed")
    normalized: list[Proposal] = []
    seen: set[Pair] = set()
    for card in cards:
        if not isinstance(card, Proposal):
            raise ValueError("Each proposal must be a Proposal record")
        pair = canonical_pair(card.unit_a, card.unit_b)
        if not set(pair) <= allowed_units:
            raise ValueError("Proposal endpoint is not in the Unit catalog")
        if type(card.expected) is not int or card.expected not in (-1, 0, 1):
            raise ValueError("Proposal expected must be an integer in {-1, 0, 1}")
        if not isinstance(card.rationale, str) or not isinstance(card.card_id, str):
            raise ValueError("Proposal rationale and card_id must be strings")
        refs = _ids(card.evidence_ids, name="Proposal evidence IDs")
        if allowed_evidence is not None and not set(refs) <= allowed_evidence:
            raise ValueError("Proposal contains a disallowed evidence reference")
        if pair in seen:
            continue
        seen.add(pair)
        expected = card.expected if card.unit_a == pair[0] else -card.expected
        normalized.append(Proposal(
            unit_a=pair[0], unit_b=pair[1], expected=expected,
            rationale=card.rationale, evidence_ids=refs, card_id=card.card_id,
        ))
    return tuple(normalized)


def compute_effect(
    original: Sequence[float], perturbed: Sequence[float], scales: Sequence[float]
) -> Effect:
    """Compute signed normalized responses and their maximum absolute value."""
    before = _vector(original, name="Original predictions")
    after = _vector(perturbed, name="Perturbed predictions")
    scale_values = _vector(scales, name="Output scales")
    if len(before) != len(after) or len(before) != len(scale_values):
        raise ValueError("Predictions and scales must have matching lengths")
    if any(scale <= 0 for scale in scale_values):
        raise ValueError("Output scales must be strictly positive")
    response = tuple((p - o) / s for o, p, s in zip(before, after, scale_values))
    if any(not math.isfinite(value) for value in response):
        raise ValueError("Normalized responses must remain finite")
    return Effect(response=response, magnitude=max(abs(value) for value in response))


def merge_candidates(
    retained: Iterable[Pair], new_pairs: Iterable[Pair], measured: Iterable[str]
) -> tuple[Pair, ...]:
    """Canonical union, excluding comparisons with both endpoints measured."""
    observed = set(_ids(measured, name="Measured units"))
    pairs = set(_pairs(retained)) | set(_pairs(new_pairs))
    return tuple(sorted(pair for pair in pairs if not set(pair) <= observed))


def completion_count(
    pairs: Iterable[Pair], measured: Iterable[str], queries: Iterable[str] = ()
) -> int:
    """Count covered pairs; supply pending pairs to count newly completed comparisons."""
    observed = set(_ids(measured, name="Measured units"))
    covered = observed | set(_ids(queries, name="Queries"))
    return sum(set(pair) <= covered for pair in _pairs(pairs))


def select_queries(
    pairs: Iterable[Pair], measured: Iterable[str], budget: int = 3
) -> tuple[str, ...]:
    """Maximize pending-comparison completion at min(budget, available endpoints).

    Break ties by lexicographic Unit IDs."""
    if type(budget) is not int or budget < 0:
        raise ValueError("Query budget must be a nonnegative integer")
    observed = set(_ids(measured, name="Measured units"))
    pending = merge_candidates(pairs, (), observed)
    endpoints = sorted({unit for pair in pending for unit in pair} - observed)
    size = min(budget, len(endpoints))
    if size == 0:
        return ()
    best: tuple[str, ...] = ()
    best_count = -1
    for batch in combinations(endpoints, size):
        covered = observed | set(batch)
        count = sum(a in covered and b in covered for a, b in pending)
        if count > best_count:
            best, best_count = batch, count
    return best


def _scores(scores: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(scores, Mapping):
        raise ValueError("Scores must be a mapping from Unit IDs to numbers")
    result: dict[str, float] = {}
    for unit, value in scores.items():
        unit = _unit_id(unit)
        score = _finite(value, name="Unit score")
        if score < 0:
            raise ValueError("SHIR Unit scores must be nonnegative")
        result[unit] = score
    return result


def update_scores(
    reference_scores: Mapping[str, float],
    current_records: Mapping[str, ExecutionRecord],
) -> dict[str, float]:
    """Apply R0: valid current effects replace defaults; other scores remain unchanged."""
    result = _scores(reference_scores)
    if not isinstance(current_records, Mapping):
        raise ValueError("Current records must be a Unit-ID mapping")
    current_scope: tuple[str, str, str, str] | None = None
    for unit, record in current_records.items():
        _unit_id(unit)
        if unit not in result:
            raise ValueError("Current record unit lacks a reference score")
        if not isinstance(record, ExecutionRecord):
            raise ValueError("Current records must contain ExecutionRecord objects")
        if record.key.unit_id != unit:
            raise ValueError("Execution record Unit ID does not match its mapping key")
        if type(record.valid) is not bool:
            raise ValueError("Execution validity must be boolean")
        if not record.valid:
            continue
        scope = (
            record.key.input_id, record.key.input_digest,
            record.key.task_fingerprint, record.group_id,
        )
        if any(not isinstance(value, str) or not value for value in scope):
            raise ValueError("Valid records require nonempty input/configuration identity")
        if current_scope is not None and scope != current_scope:
            raise ValueError("Current records must share one input and configuration")
        current_scope = scope
        original = _vector(record.original_prediction, name="Original predictions")
        perturbed = _vector(record.perturbed_prediction, name="Perturbed predictions")
        response = _vector(record.response, name="Normalized responses")
        if len(original) != len(perturbed) or len(original) != len(response):
            raise ValueError("Valid record numeric vectors must have matching lengths")
        effect = _finite(record.effect, name="Current effect")
        if effect < 0 or effect != max(abs(value) for value in response):
            raise ValueError("Current effect must equal the maximum absolute response")
        result[unit] = effect
    return result


def rank_units(scores: Mapping[str, float]) -> tuple[str, ...]:
    """Rank unrounded scores descending, with Unit-ID tie-breaking."""
    validated = _scores(scores)
    return tuple(sorted(validated, key=lambda unit: (-validated[unit], unit)))
