"""Reconstruct reference summaries from stored execution records."""

from __future__ import annotations

from typing import Any, Mapping

from .evidence import EvidenceStore, freeze_reference
from .types import ExecutionKey, ExecutionRecord, Proposal, ProposalRecord, TaskSpec, content_hash


def load_reference(data: Mapping[str, Any], task: TaskSpec):
    if data.get("task_fingerprint") != task.fingerprint or data.get("kappa") != 5.0:
        raise ValueError("Saved reference must match the paper task and kappa=5")
    store = EvidenceStore(task)
    for pair in data["registered_pairs"]:
        store.register_comparison(tuple(pair))
    for item in data["records"]:
        if item.get("valid") is not True:
            raise ValueError("Frozen reference may only contain valid unique executions")
        record = ExecutionRecord(
            key=ExecutionKey(**item["key"]), group_id=item["group_id"],
            original_prediction=tuple(item["original_prediction"]),
            perturbed_prediction=tuple(item["perturbed_prediction"]),
            response=tuple(item["response"]), effect=item["effect"], valid=True,
            error=item.get("error"), query_parameters_json=item["query_parameters_json"],
        )
        store.add_record(record)
    for item in data["proposals"]:
        proposal = Proposal(**item["proposal"])
        store.register_comparison((proposal.unit_a, proposal.unit_b), ProposalRecord(
            item["input_id"], item["input_digest"], item["round_index"], proposal))
    reference = freeze_reference((store,), task, kappa=5)
    if content_hash(reference.to_dict()) != content_hash(data):
        raise ValueError("Saved reference summaries, order or provenance do not match its execution facts")
    return reference
