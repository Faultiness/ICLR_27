"""SHIR input cards from training-only float32 attribute distributions."""

from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping, Sequence

from .types import Instance, content_hash


ASSET_SCHEMA = "shir-input-card-reference-v2"
CARD_SCHEMA = "shir-instance-card-v2"
PERCENTILE_POLICY = "(1 + count(training_values <= value)) / (n + 1)"
TRAINING_POLICY = "one value per training window; only quality == valid and finite"
STORAGE_DTYPE = "float32"
_QUALITIES = {"valid", "partial", "unavailable"}


def _number(value: Any) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError("Attribute values must be finite numbers or None")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Attribute values must be finite numbers or None") from exc
    if not math.isfinite(value):
        raise ValueError("Nonfinite input attributes must be explicitly marked None")
    return value


def _float32(value: Any) -> float:
    number = _number(value)
    try:
        stored = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError("Training attribute is not representable as finite float32") from exc
    if not math.isfinite(stored):
        raise ValueError("Training attribute is not representable as finite float32")
    return stored


def _catalog(adapter) -> tuple[dict[str, Any], ...]:
    catalog = tuple(deepcopy(dict(item)) for item in adapter.unit_catalog)
    ids = [item.get("unit_id") for item in catalog]
    if not ids or any(not isinstance(u, str) or not u for u in ids) or len(ids) != len(set(ids)):
        raise ValueError("Adapter must provide unique nonempty Unit IDs")
    return catalog


def _observations(adapter, instance: Instance) -> dict[str, dict[str, dict[str, Any]]]:
    ids = {item["unit_id"] for item in _catalog(adapter)}
    raw = adapter.unit_observations(deepcopy(instance))
    if not isinstance(raw, Mapping) or set(raw) != ids:
        raise ValueError("Attribute observations must cover the complete Unit catalog")
    result = {}
    for unit in sorted(ids):
        if not isinstance(raw[unit], Mapping) or not raw[unit]:
            raise ValueError("Each Unit needs named attributes")
        attrs = {}
        for key, observation in raw[unit].items():
            if not isinstance(key, str) or not key:
                raise ValueError("Attribute names must be nonempty strings")
            if not isinstance(observation, Mapping) or set(observation) != {"value", "unit", "quality"}:
                raise ValueError("Each attribute needs exactly value, unit, and quality")
            quality = observation["quality"]
            if not isinstance(quality, str) or quality not in _QUALITIES:
                raise ValueError("Attribute quality must be valid, partial, or unavailable")
            symbol = observation["unit"]
            if symbol is not None and (not isinstance(symbol, str) or not symbol):
                raise ValueError("An attribute unit must be a nonempty string or None")
            value = None if observation["value"] is None else _number(observation["value"])
            if quality == "valid" and value is None:
                raise ValueError("A valid attribute must have a finite value")
            if quality == "unavailable" and value is not None:
                raise ValueError("An unavailable attribute must have value None")
            attrs[key] = {"value": value, "unit": symbol, "quality": quality}
        result[unit] = attrs
    return result


def _schema(observations: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[tuple[str, str], str | None]:
    return {(unit, attr): observation["unit"]
            for unit, attrs in observations.items() for attr, observation in attrs.items()}


@dataclass(frozen=True)
class InputCardReference:
    adapter_id: str
    catalog_fingerprint: str
    # (unit ID, attribute name, physical unit or None, sorted float32 values)
    distributions: tuple[tuple[str, str, str | None, tuple[float, ...]], ...]
    # (input ID, input digest, group ID)
    source_inputs: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "distributions", tuple(
            (u, a, symbol, tuple(_float32(value) for value in values))
            for u, a, symbol, values in self.distributions))
        object.__setattr__(self, "source_inputs", tuple(tuple(v) for v in self.source_inputs))
        if (not isinstance(self.adapter_id, str) or not self.adapter_id
                or not isinstance(self.catalog_fingerprint, str) or not self.catalog_fingerprint
                or not self.source_inputs):
            raise ValueError("Input-card reference needs adapter identity and training provenance")
        seen = set()
        for unit, attr, symbol, values in self.distributions:
            if not isinstance(unit, str) or not unit or not isinstance(attr, str) or not attr or (unit, attr) in seen:
                raise ValueError("Duplicate or empty attribute identity")
            if symbol is not None and (not isinstance(symbol, str) or not symbol):
                raise ValueError("An attribute unit must be a nonempty string or None")
            seen.add((unit, attr))
            if values != tuple(sorted(values)):
                raise ValueError("Training distributions must be sorted")
            if len(values) > len(self.source_inputs):
                raise ValueError("Each input contributes at most one value per attribute")
        if not self.distributions:
            raise ValueError("Input-card reference needs attribute definitions")
        if any(len(row) != 3 or not all(isinstance(v, str) and v for v in row) for row in self.source_inputs):
            raise ValueError("Training input identities must be unique and complete")
        ids = [row[0] for row in self.source_inputs]
        if len(ids) != len(set(ids)):
            raise ValueError("Training input identities must be unique and complete")

    @classmethod
    def fit(cls, adapter, training_inputs: Sequence[Instance], *, adapter_id: str) -> InputCardReference:
        catalog = _catalog(adapter)
        observations: dict[tuple[str, str], list[float]] = {}
        sources = {}
        attribute_schema = None
        for instance in training_inputs:
            if not isinstance(instance.payload, Mapping) or instance.payload.get("split") != "train":
                raise ValueError("Instance-card references may only use training inputs")
            digest = content_hash(adapter.serialize_input(deepcopy(instance)))
            identity = (digest, instance.group_id)
            if instance.input_id in sources:
                if sources[instance.input_id] != identity:
                    raise ValueError("Conflicting training input identity")
                continue
            attrs = _observations(adapter, instance)
            schema = _schema(attrs)
            if attribute_schema is not None and schema != attribute_schema:
                raise ValueError("Adapter attribute names and units must remain fixed across inputs")
            attribute_schema = schema
            sources[instance.input_id] = identity
            for unit, values in attrs.items():
                for attr, observation in values.items():
                    observations.setdefault((unit, attr), [])
                    if observation["quality"] == "valid":
                        observations[unit, attr].append(_float32(observation["value"]))
        if attribute_schema is None:
            raise ValueError("Input-card reference needs training inputs and attribute definitions")
        return cls(adapter_id, content_hash(catalog),
                   tuple((u, a, attribute_schema[u, a], tuple(sorted(values)))
                         for (u, a), values in sorted(observations.items())),
                   tuple((key, *value) for key, value in sorted(sources.items())))

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSET_SCHEMA, "source_split": "train",
            "adapter_id": self.adapter_id, "catalog_fingerprint": self.catalog_fingerprint,
            "training_policy": TRAINING_POLICY, "storage_dtype": STORAGE_DTYPE,
            "percentile_policy": PERCENTILE_POLICY, "percentile_range": [0, 1],
            "attribute_schema": [{"unit_id": u, "attribute": a, "unit": symbol}
                                 for u, a, symbol, _ in self.distributions],
            "distributions": [{"unit_id": u, "attribute": a, "unit": symbol, "values": values}
                              for u, a, symbol, values in self.distributions],
            "source_inputs": [{"input_id": i, "input_digest": d, "group_id": g}
                              for i, d, g in self.source_inputs],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> InputCardReference:
        if (not isinstance(data, Mapping) or data.get("schema_version") != ASSET_SCHEMA
                or data.get("source_split") != "train"):
            raise ValueError(
                f"Input-card reference requires schema_version={ASSET_SCHEMA!r} and source_split='train'"
            )
        if (data.get("training_policy") != TRAINING_POLICY or data.get("storage_dtype") != STORAGE_DTYPE
                or data.get("percentile_policy") != PERCENTILE_POLICY or data.get("percentile_range") != [0, 1]):
            raise ValueError("Input-card reference policies do not match the v2 observation contract")
        result = cls(data["adapter_id"], data["catalog_fingerprint"],
                     tuple((x["unit_id"], x["attribute"], x["unit"], tuple(x["values"])) for x in data["distributions"]),
                     tuple((x["input_id"], x["input_digest"], x["group_id"]) for x in data["source_inputs"]))
        if data.get("attribute_schema") != result.to_dict()["attribute_schema"]:
            raise ValueError("Input-card attribute schema and distributions disagree")
        return result


class ReferenceCardAdapter:

    def __init__(self, adapter, reference: InputCardReference, *, adapter_id: str) -> None:
        self.adapter = adapter
        self.reference = reference
        if adapter_id != reference.adapter_id or content_hash(_catalog(adapter)) != reference.catalog_fingerprint:
            raise ValueError("Input-card reference belongs to another adapter or catalog")
        self._distributions = {(u, a): values for u, a, _, values in reference.distributions}
        self._attribute_units = {(u, a): symbol for u, a, symbol, _ in reference.distributions}
        self._source_identity = {i: (d, g) for i, d, g in reference.source_inputs}
        self._source_groups = {g for _, _, g in reference.source_inputs}
        self._patient_grouped = all(item["unit_id"].startswith("vitaldb::") for item in _catalog(adapter))

    @property
    def unit_catalog(self):
        return self.adapter.unit_catalog

    def serialize_input(self, instance):
        self._validate_identity(instance)
        return self.adapter.serialize_input(instance)

    def unit_attributes(self, instance):
        return {unit: {attr: observation["value"] for attr, observation in attrs.items()}
                for unit, attrs in self.unit_observations(instance).items()}

    def unit_observations(self, instance):
        self._validate_identity(instance)
        return _observations(self.adapter, instance)

    def model_input(self, payload):
        return self.adapter.model_input(payload)

    def perturb(self, instance, unit_id):
        self._validate_identity(instance)
        return self.adapter.perturb(instance, unit_id)

    def joint_transform(self, instance, unit_ids, donor, *, keep):
        self._validate_identity(instance)
        self._validate_identity(donor)
        return self.adapter.joint_transform(instance, unit_ids, donor, keep=keep)

    def prompt_metadata(self):
        metadata = deepcopy(dict(self.adapter.prompt_metadata()))
        metadata["instance_card_policy"] = {
            "schema_version": CARD_SCHEMA, "training_policy": TRAINING_POLICY,
            "storage_dtype": STORAGE_DTYPE, "percentile_policy": PERCENTILE_POLICY,
            "percentile_range": [0, 1],
        }
        metadata["instance_card_attribute_schema"] = self.reference.to_dict()["attribute_schema"]
        metadata["input_card_reference_id"] = self.reference.fingerprint
        return metadata

    def _validate_identity(self, instance: Instance) -> None:
        split = instance.payload.get("split") if isinstance(instance.payload, Mapping) else None
        if instance.input_id in self._source_identity:
            identity = (content_hash(self.adapter.serialize_input(instance)), instance.group_id)
            if identity != self._source_identity[instance.input_id] or split != "train":
                raise ValueError("Training card-reference identity cannot be reused for different content or split")
        if self._patient_grouped and split != "train" and instance.group_id in self._source_groups:
            raise ValueError("Input-card training patients cannot appear in validation/test inputs")

    def describe_input(self, instance: Instance) -> dict[str, Any]:
        self._validate_identity(instance)
        attributes = _observations(self.adapter, instance)
        if _schema(attributes) != self._attribute_units:
            raise ValueError("Current attribute names or units differ from the frozen training reference")
        units = []
        for item in _catalog(self.adapter):
            unit = item["unit_id"]
            values = attributes[unit]
            qualities = {observation["quality"] for observation in values.values()}
            if qualities == {"unavailable"}:
                state = "unavailable"
            elif qualities == {"valid"}:
                state = "valid"
            else:
                state = "partially_available"
            rows = {}
            for attr, observation in sorted(values.items()):
                value = observation["value"]
                reference_values = self._distributions[unit, attr]
                percentile = None
                if value is not None and reference_values:
                    percentile = (1 + bisect_right(reference_values, value)) / (len(reference_values) + 1)
                rows[attr] = {"value": value, "training_percentile": percentile,
                              "quality": observation["quality"], "unit": observation["unit"],
                              "training_support": len(reference_values)}
            units.append({"unit_id": unit, "description": item["description"],
                          "availability": state, "attributes": rows})
        return {"schema_version": CARD_SCHEMA, "input_id": instance.input_id,
                "reference_id": self.reference.fingerprint, "reference_split": "train",
                "percentile_policy": PERCENTILE_POLICY, "percentile_range": [0, 1],
                "units": units}
