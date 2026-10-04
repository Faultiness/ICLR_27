"""SHIR reference construction and per-input explanation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from threading import Lock
from typing import Any, Mapping, Sequence

from .core import (
    canonical_pair, compute_effect, merge_candidates, normalize_proposals,
    rank_units, select_queries, update_scores,
)
from .evidence import EvidenceStore, freeze_reference
from .interfaces import DatasetAdapter, Predictor, ProposalBackend
from .types import (
    CachedPrediction, ExecutionKey, ExecutionRecord, FrozenReference, Instance, Pair, Proposal,
    ProposalRecord, TaskSpec, canonical_json, content_hash, json_safe,
)


@dataclass(frozen=True)
class EngineConfig:
    query_budget: int = 3
    max_reference_rounds: int = 4
    context_comparisons: int = 6
    explanation_budgets: tuple[int, ...] | None = None
    parallel_workers: int = 3
    kappa: float = 5.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("query_budget", 1, 3),
            ("max_reference_rounds", 1, 4),
            ("context_comparisons", 1, 6),
            ("parallel_workers", 1, 3),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if not math.isfinite(self.kappa) or self.kappa <= 0:
            raise ValueError("kappa must be finite and positive")
        if self.explanation_budgets is not None:
            object.__setattr__(self, "explanation_budgets", tuple(self.explanation_budgets))


def _empty_counts() -> dict[str, int]:
    return dict(original_predictor_calls=0, perturbed_predictor_calls=0,
                query_attempts=0, valid_measurements=0, failed_queries=0,
                proposal_backend_calls=0)


@dataclass
class EpisodeResult:
    stage: str
    input_id: str
    input_digest: str
    group_id: str
    serialized_input: Any
    original_prediction: tuple[float, ...] | None
    store: EvidenceStore
    rounds: tuple[dict[str, Any], ...]
    counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return json_safe({
            "stage": self.stage, "input_id": self.input_id,
            "input_digest": self.input_digest, "group_id": self.group_id,
            "input": deepcopy(self.serialized_input),
            "original_prediction": self.original_prediction,
            "rounds": deepcopy(self.rounds), "counts": dict(self.counts),
            "records": [r.to_dict() for r in self.store.records],
            "registered_pairs": self.store.registered_pairs,
            "proposals": [asdict(p) for p in self.store.proposals],
        })


@dataclass
class ReferenceRun:
    reference: FrozenReference
    episodes: tuple[EpisodeResult, ...]

    @property
    def counts(self) -> dict[str, int]:
        return {key: sum(e.counts[key] for e in self.episodes) for key in _empty_counts()}

    def to_dict(self) -> dict[str, Any]:
        return {"reference": self.reference.to_dict(),
                "reference_digest": self.reference.digest,
                "episodes": [e.to_dict() for e in self.episodes],
                "counts": self.counts}


class ReferenceConstructionError(ValueError):
    """Reference construction failure with completed episode records."""

    def __init__(self, message: str, episodes: Sequence[EpisodeResult]):
        super().__init__(message)
        self.episodes = tuple(episodes)


@dataclass
class ExplanationResult:
    episode: EpisodeResult
    scores: dict[str, float]
    ranking: tuple[str, ...]
    reference_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {"episode": self.episode.to_dict(), "scores": dict(self.scores),
                "ranking": self.ranking, "reference_digest": self.reference_digest}


class SHIREngine:
    def __init__(
        self, task: TaskSpec, predictor: Predictor, adapter: DatasetAdapter,
        proposer: ProposalBackend, config: EngineConfig | None = None,
    ) -> None:
        self.task = task
        self.predictor = predictor
        self.adapter = adapter
        self.proposer = proposer
        self.config = config or EngineConfig()
        # Snapshot adapter outputs before parallel predictor calls.
        self._adapter_lock = Lock()
        self.budgets = self.config.explanation_budgets
        if self.budgets is None:
            self.budgets = tuple(math.ceil(len(task.unit_ids) * f)
                                 for f in (0.05, 0.075, 0.10, 0.15))
        if not self.budgets or any(type(k) is not int or not 1 <= k <= len(task.unit_ids)
                                   for k in self.budgets):
            raise ValueError("Explanation budgets must be valid nonempty catalog sizes")

    def build_reference(self, instances: Sequence[Instance]) -> ReferenceRun:
        instances = tuple(instances)
        if not instances or len({x.input_id for x in instances}) != len(instances):
            raise ValueError("Reference inputs must be nonempty and have distinct IDs")
        episodes = tuple(self._run_episode(x, reference=None) for x in instances)
        try:
            reference = freeze_reference([e.store for e in episodes], self.task,
                                         kappa=self.config.kappa)
        except ValueError as exc:
            raise ReferenceConstructionError(str(exc), episodes) from exc
        return ReferenceRun(reference, episodes)

    def explain(
        self, instance: Instance, reference: FrozenReference,
        *, cached_prediction: CachedPrediction | None = None,
    ) -> ExplanationResult:
        if reference.task_fingerprint != self.task.fingerprint:
            raise ValueError("Reference belongs to a different task configuration")
        reference_digest = reference.digest
        episode = self._run_episode(instance, reference, cached_prediction)
        records = episode.store.current_records(episode.input_id, episode.input_digest)
        scores = update_scores(reference.scores, records)
        if reference.digest != reference_digest:
            raise RuntimeError("Frozen reference changed during explanation")
        return ExplanationResult(episode, scores, rank_units(scores), reference_digest)

    def cache_prediction(self, instance: Instance) -> CachedPrediction:
        """Cache f(x) with input and task identity; one predictor call."""
        snapshot = deepcopy(instance)
        digest = content_hash(self.adapter.serialize_input(deepcopy(snapshot)))
        original = self._validate_prediction(self.predictor(deepcopy(snapshot.payload)))
        return CachedPrediction(snapshot.input_id, digest, self.task.fingerprint, original)

    def _validate_prediction(self, values: Sequence[float]) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)):
            raise ValueError("Predictions must be a numeric vector, not text")
        raw = tuple(values)
        compute_effect(raw, raw, self.task.scales)
        return tuple(float(v) for v in raw)

    def _run_episode(
        self, instance: Instance, reference: FrozenReference | None,
        cached_prediction: CachedPrediction | None = None,
    ) -> EpisodeResult:
        instance = deepcopy(instance)
        serialized = deepcopy(self.adapter.serialize_input(deepcopy(instance)))
        digest = content_hash(serialized)
        counts = _empty_counts()
        if cached_prediction is None:
            counts["original_predictor_calls"] = 1
            original = self._validate_prediction(self.predictor(deepcopy(instance.payload)))
        else:
            if not isinstance(cached_prediction, CachedPrediction):
                raise ValueError("Cached predictions require input and task identity")
            expected_scope = (instance.input_id, digest, self.task.fingerprint)
            actual_scope = (cached_prediction.input_id, cached_prediction.input_digest,
                            cached_prediction.task_fingerprint)
            if actual_scope != expected_scope:
                raise ValueError("Cached prediction belongs to another input or task")
            original = self._validate_prediction(cached_prediction.prediction)
        compute_effect(original, original, self.task.scales)
        store = EvidenceStore(self.task)
        if reference is not None:
            for pair in reference.registered_pairs:
                store.register_comparison(pair)
        stage = "reference" if reference is None else "explanation"
        max_rounds = self.config.max_reference_rounds if reference is None else 1
        rounds = []
        for round_index in range(1, max_rounds + 1):
            records = store.current_records(instance.input_id, digest)
            if len(records) == len(self.task.unit_ids):
                break
            context, retained = self._context(instance, digest, original, store,
                                              reference, round_index, max_rounds)
            calls = []
            proposals, call = self._propose(context)
            calls.append(call)
            measured = set(records)
            candidate_pairs = merge_candidates(
                retained, [canonical_pair(p.unit_a, p.unit_b) for p in proposals], measured)
            endpoints = {u for pair in candidate_pairs for u in pair} - measured
            if len(endpoints) < self.config.query_budget:
                proposals, call = self._propose(context, supplement=True, previous=proposals)
                calls.append(call)
                candidate_pairs = merge_candidates(
                    retained, [canonical_pair(p.unit_a, p.unit_b) for p in proposals], measured)
            counts["proposal_backend_calls"] += len(calls)
            for pair in candidate_pairs:
                store.register_comparison(pair)
            for p in proposals:
                pair = canonical_pair(p.unit_a, p.unit_b)
                # Completed proposals remain registered but do not enter query selection.
                store.register_comparison(pair, ProposalRecord(
                    instance.input_id, digest, round_index, p))
            store.link_all(instance.input_id, digest)
            queries = select_queries(candidate_pairs, measured, self.config.query_budget)
            trace = {
                "round_index": round_index, "context": context,
                "proposal_calls": calls, "retained_pairs": retained,
                "candidate_pairs": candidate_pairs, "measured_before": sorted(measured),
                "selected_queries": queries,
                "planned_completions": sum(set(pair) <= measured | set(queries)
                                           for pair in candidate_pairs),
                "executions": [],
            }
            if not queries:
                trace["stop_reason"] = "no_admissible_unmeasured_endpoint"
                self._snapshot_graph(trace, store, instance.input_id, digest)
                rounds.append(trace)
                break
            # Select the entire batch before observing any response.
            if self.config.parallel_workers == 1 or len(queries) == 1:
                outcomes = [self._execute(instance, digest, u, original) for u in queries]
            else:
                with ThreadPoolExecutor(max_workers=min(self.config.parallel_workers, len(queries))) as pool:
                    futures = [pool.submit(self._execute, instance, digest, u, original)
                               for u in queries]
                    # Preserve query order when collecting parallel responses.
                    outcomes = [future.result() for future in futures]
            for record, predictor_called in outcomes:
                counts["query_attempts"] += 1
                counts["perturbed_predictor_calls"] += int(predictor_called)
                counts["valid_measurements"] += int(record.valid)
                counts["failed_queries"] += int(not record.valid)
                store.add_record(record)
                trace["executions"].append(record.to_dict())
            store.link_all(instance.input_id, digest)
            self._snapshot_graph(trace, store, instance.input_id, digest)
            rounds.append(trace)
        return EpisodeResult(stage, instance.input_id, digest, instance.group_id,
                             serialized, original, store, tuple(rounds), counts)

    @staticmethod
    def _snapshot_graph(trace: dict[str, Any], store: EvidenceStore,
                        input_id: str, digest: str) -> None:
        trace["measured_after"] = sorted(store.current_records(input_id, digest))
        trace["comparison_states"] = [asdict(store.comparison_state(pair, input_id, digest))
                                      for pair in store.registered_pairs]
        trace["comparison_links"] = [
            {"pair": pair, "execution_ids": [r.execution_id for r in
             store.linked_records(pair, input_id, digest)]}
            for pair in store.registered_pairs]

    def _execute(
        self, instance: Instance, digest: str, unit_id: str,
        original: tuple[float, ...],
    ) -> tuple[ExecutionRecord, bool]:
        params_json = canonical_json({"failure_stage": "adapter", "unit_id": unit_id})
        query_signature = content_hash({"failure_stage": "adapter", "unit_id": unit_id})
        prediction = None
        predictor_called = False
        try:
            with self._adapter_lock:
                perturbation = self.adapter.perturb(deepcopy(instance), unit_id)
                if not isinstance(perturbation.parameters, Mapping):
                    raise ValueError("Query parameters must be a JSON object")
                parameters = deepcopy(dict(perturbation.parameters))
                candidate_json = canonical_json(parameters)
                candidate_signature = content_hash(parameters)
                payload = deepcopy(perturbation.payload)
                params_json, query_signature = candidate_json, candidate_signature
            predictor_called = True
            raw = self.predictor(payload)
            if isinstance(raw, (str, bytes)):
                raise ValueError("Predictions must be a numeric vector, not text")
            prediction = tuple(raw)
            effect = compute_effect(original, prediction, self.task.scales)
            prediction = tuple(float(v) for v in prediction)
            valid, response, magnitude, error = True, effect.response, effect.magnitude, None
        except Exception as exc:
            valid, response, magnitude = False, None, None
            error = f"{type(exc).__name__}: {exc}"
        record = ExecutionRecord(
            ExecutionKey(instance.input_id, digest, self.task.fingerprint, unit_id, query_signature),
            instance.group_id, original, prediction, response, magnitude, valid, error, params_json)
        return record, predictor_called

    def _propose(
        self, context: Mapping[str, Any], *, supplement: bool = False,
        previous: Sequence[Proposal] = (),
    ) -> tuple[tuple[Proposal, ...], dict[str, Any]]:
        trace: dict[str, Any] = {"supplement": supplement,
                                 "previous": [p.to_dict() for p in previous]}
        try:
            raw = tuple(self.proposer.propose(deepcopy(context), supplement=supplement,
                                             previous=tuple(previous)))
            trace["raw"] = json_safe([p.to_dict() if isinstance(p, Proposal) else repr(p) for p in raw])
            wrapped = tuple(Proposal(p.unit_a, p.unit_b, p.expected, p.rationale,
                                     p.evidence_ids, p.card_id or f"P{i}")
                            for i, p in enumerate(raw, 1))
            if len({p.card_id for p in wrapped}) != len(wrapped):
                raise ValueError("Grounded Card IDs must remain unique")
            admitted = normalize_proposals(wrapped, self.task.unit_ids,
                                            context["allowed_evidence_ids"])
            trace["admitted"] = [p.to_dict() for p in admitted]
            trace["error"] = None
            return admitted, trace
        except Exception as exc:
            trace["admitted"] = []
            trace["error"] = f"{type(exc).__name__}: {exc}"
            return (), trace
        finally:
            audit = getattr(self.proposer, "consume_audit", None)
            if callable(audit):
                trace["backend_audit"] = json_safe(audit())

    def _context(
        self, instance: Instance, digest: str, original: tuple[float, ...],
        store: EvidenceStore, reference: FrozenReference | None,
        round_index: int, max_rounds: int,
    ) -> tuple[dict[str, Any], tuple[Pair, ...]]:
        current = store.current_records(instance.input_id, digest)
        measured = set(current)
        scores = None if reference is None else update_scores(reference.scores, current)
        ranking = None if scores is None else rank_units(scores)
        positions = {} if ranking is None else {u: i for i, u in enumerate(ranking)}

        def crosses(pair: Pair) -> bool:
            return bool(positions) and any((positions[pair[0]] < k) != (positions[pair[1]] < k)
                                           for k in set(self.budgets))

        pairs = sorted(store.registered_pairs,
                       key=lambda pair: (not bool(set(pair) - measured), not crosses(pair), pair))
        displayed = tuple(pairs[:self.config.context_comparisons])
        retained = tuple(pair for pair in displayed if set(pair) - measured)
        allowed_ids = {"INSTANCE_CARD", *(r.execution_id for r in current.values())}
        historical = None
        pair_summaries = {} if reference is None else {s.pair: s for s in reference.pair_summaries}
        if reference is not None:
            summaries = []
            for summary in reference.unit_summaries:
                item = asdict(summary)
                item["evidence_id"] = f"reference-unit:{summary.unit_id}"
                # Mark whether the reference summary has execution support.
                item["observed"] = summary.group_count > 0
                summaries.append(item)
                if summary.group_count > 0:
                    allowed_ids.add(item["evidence_id"])
            historical = {"reference_digest": reference.digest,
                          "unit_summaries": summaries,
                          "reference_ranking": rank_units(reference.scores)}
        comparisons = []
        for pair in displayed:
            state = store.comparison_state(pair, instance.input_id, digest)
            linked = store.linked_records(pair, instance.input_id, digest)
            organized = {record.key.unit_id: record for record in linked}
            summary = pair_summaries.get(pair)
            paired = None
            if summary is not None and summary.group_count:
                paired = asdict(summary)
                paired["evidence_id"] = f"reference-pair:{content_hash(pair)}"
                allowed_ids.add(paired["evidence_id"])
            def expectations(records: Sequence[ProposalRecord]) -> list[dict[str, Any]]:
                matching = [p for p in records if canonical_pair(p.proposal.unit_a, p.proposal.unit_b) == pair]
                groups = []
                for direction in (-1, 0, 1):
                    choices = [p for p in matching if p.proposal.expected == direction]
                    if choices:
                        representative = min(choices, key=lambda p: (p.input_id, p.round_index,
                                                                    p.proposal.card_id, p.proposal.rationale))
                        groups.append({"expected": direction, "proposal_count": len(choices),
                                       "representative": asdict(representative)})
                return groups
            comparisons.append({
                "pair": pair, "missing_current_endpoints": state.missing,
                "current_ordering": state.ordering if set(pair) <= set(organized) else None,
                "current_records": [organized[u].to_dict() for u in pair if u in organized],
                "crosses_budget_boundary": crosses(pair),
                "reference_paired_summary": paired,
                "current_expectations": expectations(store.proposals),
                "reference_expectations": expectations(reference.proposals) if reference else [],
            })
        card = deepcopy(dict(self.adapter.describe_input(deepcopy(instance))))
        canonical_json(card)
        context = {
            "schema_version": "shir-context-v1", "stage": "reference" if reference is None else "explanation",
            "input_id": instance.input_id, "input_digest": digest, "group_id": instance.group_id,
            "round_index": round_index, "maximum_rounds": max_rounds,
            "query_budget": self.config.query_budget, "task": self.task.to_dict(),
            "instance_card": card, "original_prediction": original,
            "explanation_budgets": self.budgets, "working_ranking": ranking,
            "historical_reference": historical,
            "current_execution_table": [current[u].to_dict() for u in sorted(current)],
            "registered_comparisons": comparisons,
            "allowed_evidence_ids": sorted(allowed_ids),
        }
        metadata = getattr(self.adapter, "prompt_metadata", None)
        if callable(metadata):
            context["task_metadata"] = deepcopy(dict(metadata()))
        return context, retained
