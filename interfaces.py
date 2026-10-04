"""Predictor, adapter, and proposal interfaces."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .types import Instance, Perturbation, Proposal


class Predictor(Protocol):
    """Frozen predictor returning independent outputs.
    Concurrent calls must be safe when parallel_workers > 1.
    """

    def __call__(self, payload: Any) -> Sequence[float]: ...


class DatasetAdapter(Protocol):
    def serialize_input(self, instance: Instance) -> Any:
        """Return stable, finite JSON data identifying the complete input."""
        ...

    def describe_input(self, instance: Instance) -> Mapping[str, Any]:
        """Describe observed inputs without inferred perturbation effects."""
        ...

    def perturb(self, instance: Instance, unit_id: str) -> Perturbation:
        """Apply the fixed perturbation to one unit."""
        ...

    def joint_transform(
        self, instance: Instance, unit_ids: Sequence[str], donor: Instance, *, keep: bool
    ) -> Any:
        """Replace selected units (or catalog complement when keep=True) together."""
        ...


class ProposalBackend(Protocol):
    def propose(
        self,
        context: Mapping[str, Any],
        *,
        supplement: bool = False,
        previous: Sequence[Proposal] = (),
    ) -> Sequence[Proposal]:
        """Return up to three grounded proposals from pre-query information.

        A supplement replaces the previous proposal set.
        """
        ...
