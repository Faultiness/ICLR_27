"""SHIR input-scoped evidence: shared records contribute once to aggregation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence

from .core import canonical_pair, compute_effect
from .types import (
    ExecutionRecord,
    FrozenReference,
    Pair,
    PairSummary,
    Proposal,
    ProposalRecord,
    TaskSpec,
    UnitSummary,
    canonical_json,
    content_hash,
)


@dataclass(frozen=True)
class ComparisonState:
    pair: Pair
    input_id: str
    input_digest: str
    missing: tuple[str, ...]
    ordering: int | None

    @property
    def completed(self) -> bool:
        return not self.missing

    @property
    def status(self) -> str:
        return "completed" if self.completed else "pending"


def _nonempty_string(value: object, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Nonfinite query parameter: {value}")


def _parameters(record: ExecutionRecord) -> str:
    try:
        parameters = json.loads(
            record.query_parameters_json, parse_constant=_reject_json_constant
        )
        if not isinstance(parameters, dict):
            raise ValueError("Query parameters must be a JSON object")
        normalized = canonical_json(parameters)
        if content_hash(parameters) != record.key.query_signature:
            raise ValueError("Query signature does not match query parameters")
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Query parameters must be a finite JSON object") from exc
    return normalized


def _finite_vector(values: Sequence[float], length: int, label: str) -> tuple[float, ...]:
    try:
        if isinstance(values, (str, bytes)) or any(
            isinstance(value, (str, bytes, bool)) for value in values
        ):
            raise ValueError(f"{label} must contain finite numbers")
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must contain finite numbers") from exc
    if len(result) != length or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{label} must have {length} finite components")
    return result


def _median(values: Sequence[float]) -> float:
    """Avoid midpoint overflow for finite observations."""
    ordered = sorted(values)
    size = len(ordered)
    if not size:
        raise ValueError("A median requires observations")
    high = float(ordered[size // 2])
    if size % 2:
        return high
    low = float(ordered[size // 2 - 1])
    if low <= 0 <= high:
        return (low + high) / 2
    return low + (high - low) / 2


def _record_order(record: ExecutionRecord) -> tuple[str, ...]:
    key = record.key
    return (key.input_id, key.input_digest, key.unit_id, key.query_signature)


class EvidenceStore:
    """Store unique valid executions separately from all query attempts."""

    def __init__(self, task: TaskSpec) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("EvidenceStore needs a TaskSpec")
        self.task = task
        self._units = frozenset(task.unit_ids)
        self._pairs: set[Pair] = set()
        self._proposals: list[ProposalRecord] = []
        self._records: dict[tuple[str, str, str], ExecutionRecord] = {}
        self._attempts: list[ExecutionRecord] = []
        self._input_digests: dict[str, str] = {}
        self._input_groups: dict[str, str] = {}
        self._original_predictions: dict[tuple[str, str], tuple[float, ...]] = {}
        self._links: dict[tuple[Pair, str, str], set[str]] = {}

    def _pair(self, pair: Pair) -> Pair:
        if isinstance(pair, (str, bytes)):
            raise ValueError("A comparison needs two distinct catalog units")
        try:
            if len(pair) != 2:
                raise ValueError("A comparison needs two distinct catalog units")
            result = canonical_pair(*pair)
        except TypeError as exc:
            raise ValueError("A comparison needs two distinct catalog units") from exc
        if not set(result) <= self._units:
            raise ValueError("A comparison needs two distinct catalog units")
        return result

    def _scope(self, input_id: str, input_digest: str) -> None:
        _nonempty_string(input_id, "Input ID")
        _nonempty_string(input_digest, "Input digest")
        previous = self._input_digests.get(input_id)
        if previous is not None and previous != input_digest:
            raise ValueError("An input ID cannot identify different input digests")

    def register_comparison(
        self, pair: Pair, proposal_record: ProposalRecord | None = None
    ) -> Pair:
        pair = self._pair(pair)
        if proposal_record is not None:
            if not isinstance(proposal_record, ProposalRecord):
                raise TypeError("Expected a ProposalRecord")
            self._scope(proposal_record.input_id, proposal_record.input_digest)
            proposal = proposal_record.proposal
            if not isinstance(proposal, Proposal):
                raise ValueError("Proposal records need a Proposal")
            if self._pair((proposal.unit_a, proposal.unit_b)) != pair:
                raise ValueError("Proposal endpoints do not match the comparison")
            if type(proposal.expected) is not int or proposal.expected not in (-1, 0, 1):
                raise ValueError("Proposed ordering must be -1, 0, or 1")
            if not isinstance(proposal.rationale, str) or not isinstance(proposal.card_id, str):
                raise ValueError("Proposal rationale and card ID must be strings")
            for evidence_id in proposal.evidence_ids:
                _nonempty_string(evidence_id, "Proposal evidence ID")
            if type(proposal_record.round_index) is not int or proposal_record.round_index < 0:
                raise ValueError("Proposal round index must be a nonnegative integer")
            self._input_digests[proposal_record.input_id] = proposal_record.input_digest
            if proposal_record not in self._proposals:
                self._proposals.append(proposal_record)
        self._pairs.add(pair)
        return pair

    def add_record(self, record: ExecutionRecord) -> None:
        if not isinstance(record, ExecutionRecord):
            raise TypeError("Expected an ExecutionRecord")
        key = record.key
        if key.task_fingerprint != self.task.fingerprint:
            raise ValueError("Execution belongs to a different task configuration")
        if key.unit_id not in self._units:
            raise ValueError("Execution unit is outside the task catalog")
        self._scope(key.input_id, key.input_digest)
        _nonempty_string(key.query_signature, "Query signature")
        _nonempty_string(record.group_id, "Group ID")
        if type(record.valid) is not bool:
            raise ValueError("Execution validity must be a boolean")
        previous_group = self._input_groups.get(key.input_id)
        if previous_group is not None and previous_group != record.group_id:
            raise ValueError("An input cannot belong to different reference groups")
        parameters = _parameters(record)
        scope = (key.input_id, key.input_digest)
        canonical_query = (*scope, key.unit_id)
        original: tuple[float, ...] | None = None

        if record.valid:
            length = len(self.task.scales)
            original = _finite_vector(record.original_prediction, length, "Original prediction")
            if record.perturbed_prediction is None or record.response is None or record.effect is None:
                raise ValueError("A valid execution needs predictions, response, and effect")
            recomputed = compute_effect(
                record.original_prediction, record.perturbed_prediction, self.task.scales
            )
            response = _finite_vector(record.response, length, "Normalized response")
            if response != recomputed.response:
                raise ValueError("Normalized response is inconsistent with predictor outputs and scales")
            try:
                if isinstance(record.effect, (str, bytes, bool)):
                    raise ValueError("A valid effect must be finite")
                effect = float(record.effect)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("A valid effect must be finite") from exc
            if not math.isfinite(effect) or effect != recomputed.magnitude:
                raise ValueError("Effect is inconsistent with the normalized response")
            if record.error is not None:
                raise ValueError("A valid execution cannot contain an error")
            if scope in self._original_predictions and self._original_predictions[scope] != original:
                raise ValueError("Original predictions conflict for the same frozen input")

            previous = self._records.get(canonical_query)
            if previous is not None:
                if previous.key != key:
                    raise ValueError("A canonical input/unit query cannot have multiple signatures")
                same_fact = (
                    previous.group_id == record.group_id
                    and previous.original_prediction == record.original_prediction
                    and previous.perturbed_prediction == record.perturbed_prediction
                    and previous.response == record.response
                    and previous.effect == record.effect
                    and _parameters(previous) == parameters
                )
                if not same_fact:
                    raise ValueError("Conflicting results for the same execution identity")

        self._input_digests[key.input_id] = key.input_digest
        self._input_groups[key.input_id] = record.group_id
        self._attempts.append(record)
        if record.valid:
            assert original is not None
            self._original_predictions[scope] = original
            self._records.setdefault(canonical_query, record)

    def current_records(self, input_id: str, input_digest: str) -> dict[str, ExecutionRecord]:
        self._scope(input_id, input_digest)
        return {
            unit: record
            for (record_input, digest, unit), record in sorted(self._records.items())
            if record_input == input_id and digest == input_digest
        }

    @property
    def registered_pairs(self) -> tuple[Pair, ...]:
        return tuple(sorted(self._pairs))

    @property
    def records(self) -> tuple[ExecutionRecord, ...]:
        return tuple(sorted(self._records.values(), key=_record_order))

    @property
    def attempts(self) -> tuple[ExecutionRecord, ...]:
        return tuple(self._attempts)

    @property
    def proposals(self) -> tuple[ProposalRecord, ...]:
        return tuple(sorted(
            self._proposals,
            key=lambda record: (
                record.input_id, record.input_digest, record.round_index,
                canonical_json(record.proposal.to_dict()),
            ),
        ))

    def link_all(self, input_id: str, input_digest: str) -> None:
        current = self.current_records(input_id, input_digest)
        for pair in self.registered_pairs:
            self._links[(pair, input_id, input_digest)] = {
                current[unit].execution_id for unit in pair if unit in current
            }

    def linked_records(
        self, pair: Pair, input_id: str, input_digest: str
    ) -> tuple[ExecutionRecord, ...]:
        pair = self._pair(pair)
        if pair not in self._pairs:
            raise ValueError("Comparison is not registered")
        current = self.current_records(input_id, input_digest)
        linked = self._links.get((pair, input_id, input_digest), set())
        return tuple(
            current[unit] for unit in pair
            if unit in current and current[unit].execution_id in linked
        )

    def comparison_state(
        self, pair: Pair, input_id: str, input_digest: str
    ) -> ComparisonState:
        pair = self._pair(pair)
        if pair not in self._pairs:
            raise ValueError("Comparison is not registered")
        current = self.current_records(input_id, input_digest)
        missing = tuple(unit for unit in pair if unit not in current)
        ordering = None
        if not missing:
            first, second = (current[unit].effect for unit in pair)
            assert first is not None and second is not None
            ordering = int(first > second) - int(first < second)
        return ComparisonState(pair, input_id, input_digest, missing, ordering)


def freeze_reference(
    stores: Sequence[EvidenceStore], task: TaskSpec, kappa: float = 5.0
) -> FrozenReference:
    """Take component medians within groups, then across groups.

    Center across supported units, shrink, then take the maximum component.
    Pair differences are matched within inputs; signs are counted per group."""
    try:
        kappa = float(kappa)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("kappa must be finite and nonnegative") from exc
    if not math.isfinite(kappa) or kappa < 0:
        raise ValueError("kappa must be finite and nonnegative")
    merged = EvidenceStore(task)
    for store in stores:
        if not isinstance(store, EvidenceStore):
            raise TypeError("Reference episodes must be EvidenceStore instances")
        if store.task.fingerprint != task.fingerprint:
            raise ValueError("Reference episodes must share the same task configuration")
        for pair in store.registered_pairs:
            merged.register_comparison(pair)
        for proposal in store.proposals:
            merged.register_comparison(
                (proposal.proposal.unit_a, proposal.proposal.unit_b), proposal
            )
        for record in store.attempts:
            merged.add_record(record)
    records = merged.records
    if not records:
        raise ValueError("Cannot build reference centers without valid execution evidence")

    by_unit: dict[str, dict[str, list[tuple[float, ...]]]] = {}
    by_input: dict[tuple[str, str], dict[str, ExecutionRecord]] = {}
    for record in records:
        assert record.response is not None
        absolute = tuple(abs(float(value)) for value in record.response)
        by_unit.setdefault(record.key.unit_id, {}).setdefault(record.group_id, []).append(absolute)
        scope = (record.key.input_id, record.key.input_digest)
        by_input.setdefault(scope, {})[record.key.unit_id] = record

    components = len(task.scales)
    medians: dict[str, tuple[float, ...]] = {}
    for unit, groups in by_unit.items():
        group_values = [
            tuple(_median([window[c] for window in windows]) for c in range(components))
            for windows in groups.values()
        ]
        medians[unit] = tuple(_median([group[c] for group in group_values]) for c in range(components))
    centers = tuple(_median([values[c] for values in medians.values()]) for c in range(components))

    units = []
    for unit in task.unit_ids:
        if unit in medians:
            count = len(by_unit[unit])
            weight = count / (count + kappa)
            score = max(
                weight * observed + (1.0 - weight) * center
                for observed, center in zip(medians[unit], centers)
            )
            units.append(UnitSummary(unit, count, medians[unit], score))
        else:
            # Unmeasured units use the neutral-center default.
            units.append(UnitSummary(unit, 0, None, max(centers)))

    pairs = []
    for pair in merged.registered_pairs:
        differences: dict[str, list[float]] = {}
        for current in by_input.values():
            if all(unit in current for unit in pair):
                first, second = (current[unit] for unit in pair)
                assert first.effect is not None and second.effect is not None
                differences.setdefault(first.group_id, []).append(float(first.effect) - float(second.effect))
        group_differences = [_median(values) for values in differences.values()]
        pairs.append(PairSummary(
            pair=pair,
            group_count=len(group_differences),
            median_difference=_median(group_differences) if group_differences else None,
            positive=sum(value > 0 for value in group_differences),
            negative=sum(value < 0 for value in group_differences),
            ties=sum(value == 0 for value in group_differences),
        ))

    return FrozenReference(
        task.fingerprint, tuple(units), tuple(pairs), merged.registered_pairs,
        records, merged.proposals, kappa, centers,
    )
