"""SHIR task, proposal, execution, and reference records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

Pair = tuple[str, str]


def json_safe(value: Any) -> Any:
    """Convert nonfinite values and unsupported objects to JSON-safe diagnostics."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"unsupported_type": type(value).__name__, "repr": repr(value)[:500]}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    predictor_id: str
    adapter_id: str
    operator_id: str
    target_names: tuple[str, ...]
    scales: tuple[float, ...]
    unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        # Freeze caller-owned sequences.
        object.__setattr__(self, "target_names", tuple(self.target_names))
        object.__setattr__(self, "scales", tuple(float(v) for v in self.scales))
        object.__setattr__(self, "unit_ids", tuple(self.unit_ids))
        for name in (self.predictor_id, self.adapter_id, self.operator_id):
            if not isinstance(name, str) or not name:
                raise ValueError("Task identity fields must be nonempty strings")
        if not self.target_names or len(self.target_names) != len(self.scales):
            raise ValueError("Targets and scales must have the same nonzero length")
        if any(not isinstance(v, str) or not v for v in self.target_names):
            raise ValueError("Target names must be nonempty strings")
        if len(set(self.target_names)) != len(self.target_names):
            raise ValueError("Target names must be unique")
        if any(not math.isfinite(v) or v <= 0 for v in self.scales):
            raise ValueError("Scales must be finite and strictly positive")
        if not self.unit_ids or any(not isinstance(v, str) or not v for v in self.unit_ids):
            raise ValueError("Unit IDs must be nonempty strings")
        if len(set(self.unit_ids)) != len(self.unit_ids):
            raise ValueError("Unit IDs must be unique")

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Instance:
    input_id: str
    group_id: str
    payload: Any

    def __post_init__(self) -> None:
        if not self.input_id or not self.group_id:
            raise ValueError("An instance needs input_id and group_id")


@dataclass(frozen=True)
class CachedPrediction:
    input_id: str
    input_digest: str
    task_fingerprint: str
    prediction: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prediction", tuple(self.prediction))


@dataclass(frozen=True)
class Perturbation:
    payload: Any
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class Proposal:
    """Expected sign of a_x(unit_a) - a_x(unit_b)."""

    unit_a: str
    unit_b: str
    expected: int
    rationale: str = ""
    evidence_ids: tuple[str, ...] = ()
    card_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_ids", tuple(self.evidence_ids))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProposalRecord:
    input_id: str
    input_digest: str
    round_index: int
    proposal: Proposal


@dataclass(frozen=True)
class Effect:
    response: tuple[float, ...]
    magnitude: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "response", tuple(self.response))


@dataclass(frozen=True)
class ExecutionKey:
    input_id: str
    input_digest: str
    task_fingerprint: str
    unit_id: str
    query_signature: str


@dataclass(frozen=True)
class ExecutionRecord:
    key: ExecutionKey
    group_id: str
    original_prediction: tuple[float, ...]
    perturbed_prediction: tuple[float, ...] | None
    response: tuple[float, ...] | None
    effect: float | None
    valid: bool
    error: str | None = None
    query_parameters_json: str = "{}"

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_prediction", tuple(self.original_prediction))
        if self.perturbed_prediction is not None:
            object.__setattr__(self, "perturbed_prediction", tuple(self.perturbed_prediction))
        if self.response is not None:
            object.__setattr__(self, "response", tuple(self.response))

    @property
    def execution_id(self) -> str:
        return content_hash(asdict(self.key))

    def to_dict(self) -> dict[str, Any]:
        return json_safe({"execution_id": self.execution_id, **asdict(self)})


@dataclass(frozen=True)
class UnitSummary:
    unit_id: str
    group_count: int
    component_medians: tuple[float, ...] | None
    reference_score: float

    def __post_init__(self) -> None:
        if self.component_medians is not None:
            object.__setattr__(self, "component_medians", tuple(self.component_medians))


@dataclass(frozen=True)
class PairSummary:
    pair: Pair
    group_count: int
    median_difference: float | None
    positive: int
    negative: int
    ties: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", tuple(self.pair))


@dataclass(frozen=True)
class FrozenReference:
    task_fingerprint: str
    unit_summaries: tuple[UnitSummary, ...]
    pair_summaries: tuple[PairSummary, ...]
    registered_pairs: tuple[Pair, ...]
    records: tuple[ExecutionRecord, ...]
    proposals: tuple[ProposalRecord, ...]
    kappa: float
    centers: tuple[float, ...]

    def __post_init__(self) -> None:
        for field in ("unit_summaries", "pair_summaries", "records", "proposals", "centers"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        object.__setattr__(self, "registered_pairs", tuple(tuple(p) for p in self.registered_pairs))

    @property
    def scores(self) -> dict[str, float]:
        # Return independent scores for local updates.
        return {s.unit_id: s.reference_score for s in self.unit_summaries}

    @property
    def digest(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))
