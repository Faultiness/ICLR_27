"""VitalDB interventions on 42x20 physical histories with observed and quality masks.

Twelve prehistory bins support derived features; models use the final 30 bins.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import struct
from typing import Any, Mapping, Sequence

from ..types import Instance, Perturbation, TaskSpec, content_hash


DIRECT_FACTORS = (
    "ART_MBP", "HR", "SpO2", "ETCO2", "BIS", "MAC", "Propofol_CE",
    "Remifentanil_CE", "PEEP", "MAWP", "RR_CO2", "TV",
)
DERIVED_FACTORS = (
    "MAP_slope", "MAP_volatility", "pulse_pressure", "HR_slope", "ETCO2_slope",
    "ECG_log_RMSSD", "PPG_amplitude_log_ratio", "PPG_amplitude_slope",
)
FACTOR_NAMES = DIRECT_FACTORS + DERIVED_FACTORS
FACTOR_INDEX = {name: index for index, name in enumerate(FACTOR_NAMES)}
FACTOR_UNITS = {
    "ART_MBP": "mmHg", "HR": "bpm", "SpO2": "%", "ETCO2": "mmHg",
    "BIS": None, "MAC": None, "Propofol_CE": None, "Remifentanil_CE": None,
    "PEEP": "mbar", "MAWP": "mbar", "RR_CO2": "/min", "TV": "mL",
    "MAP_slope": "mmHg/min", "MAP_volatility": "mmHg", "pulse_pressure": "mmHg",
    "HR_slope": "bpm/min", "ETCO2_slope": "mmHg/min", "ECG_log_RMSSD": "log(1+ms)",
    "PPG_amplitude_log_ratio": None, "PPG_amplitude_slope": "amplitude/min",
}
SOURCE_NUMERIC_TRACKS = {
    "ART_MBP": "Solar8000/ART_MBP", "ART_SBP": "Solar8000/ART_SBP",
    "ART_DBP": "Solar8000/ART_DBP", "HR": "Solar8000/HR",
    "SpO2": "Solar8000/PLETH_SPO2", "ETCO2": "Primus/ETCO2", "BIS": "BIS/BIS",
    "MAC": "Primus/MAC", "Propofol_CE": "Orchestra/PPF20_CE",
    "Remifentanil_CE": "Orchestra/RFTN20_CE", "PEEP": "Primus/PEEP_MBAR",
    "MAWP": "Primus/MAWP_MBAR", "RR_CO2": "Primus/RR_CO2", "TV": "Primus/TV",
}
SOURCE_WAVEFORM_TRACKS = {"ECG": "SNUADC/ECG_II", "PPG": "SNUADC/PLETH"}
HISTORY_BINS, INPUT_BINS, PREHISTORY_BINS = 42, 30, 12
EPSILON = 1e-6
GROUPS = ("MAP_family", "HR_family", "ETCO2_family", "SpO2")
BASE_FACTORS = ("ART_MBP", "HR", "ETCO2", "SpO2")
GROUP_BASES = dict(zip(GROUPS, BASE_FACTORS))
REGISTERED_DERIVED = {
    "ART_MBP": ("MAP_slope", "MAP_volatility"), "HR": ("HR_slope",),
    "ETCO2": ("ETCO2_slope",), "SpO2": (),
}
UNIT_IDS = tuple(f"vitaldb::block_{block}::{group}" for block in range(1, 6) for group in GROUPS)
OUTPUT_SCALES = (13.435533437311783, 12.55355789716196, 11.77440668760714)
_UNIT_SPEC = {
    f"vitaldb::block_{block}::{group}": (6 * (block - 1), 6 * block, GROUP_BASES[group])
    for block in range(1, 6) for group in GROUPS
}
_ESTIMATOR = "last30_quality_valid_prepared_bins_linear_type7_float32"
_SCALE_POLICY = "where(iqr>0,iqr,max(q999-q001,1e-6))+1e-6; float32"
_DERIVED_POLICY = {
    "window_bins": 6, "includes_current_bin": True, "prehistory_bins": 12,
    "sampling_seconds": 10, "minimum_quality_valid_bins": 4,
    "slope": "OLS_per_minute_on_quality_valid_positions",
    "volatility": "1.4826_times_median_absolute_deviation",
    "updates": "only_original_derived_quality_valid_bins; validity_mismatch_is_error",
    "masks": "preserve_observed_and_quality_masks", "unavailable_values": "retain_prepared_imputation",
}


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _f32(value: Any, name: str = "Prepared numerical factor") -> float:
    value = _number(value, name)
    try:
        result = struct.unpack("!f", struct.pack("!f", value))[0]
    except OverflowError as exc:
        raise ValueError(f"{name} must be representable as finite float32") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be representable as finite float32")
    return result


def _matrix_rows(value: Any, name: str) -> Sequence[Sequence[Any]]:
    if not isinstance(value, (list, tuple)) or len(value) != HISTORY_BINS:
        raise ValueError(f"VitalDB {name} must have exactly 42 rows")
    if any(not isinstance(row, (list, tuple)) or len(row) != len(FACTOR_NAMES) for row in value):
        raise ValueError(f"VitalDB {name} must have exactly 20 columns")
    return value


def _mask(value: Any, name: str) -> list[list[bool]]:
    result = []
    for row in _matrix_rows(value, name):
        if any(not isinstance(item, (bool, int, float)) or item not in (0, 1) for item in row):
            raise ValueError("VitalDB masks must contain booleans or numeric 0/1")
        result.append([bool(item) for item in row])
    return result


def _payload(payload: Any) -> dict[str, Any]:
    required = {"history_values", "history_observed_mask", "history_quality_mask", "split"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("VitalDB v2 needs history_values, history_observed_mask, history_quality_mask, and split")
    if payload["split"] not in ("train", "validation", "test"):
        raise ValueError("Prepared VitalDB split must be train, validation, or test")
    return {
        "history_values": [[_f32(value) for value in row]
                           for row in _matrix_rows(payload["history_values"], "history_values")],
        "history_observed_mask": _mask(payload["history_observed_mask"], "history_observed_mask"),
        "history_quality_mask": _mask(payload["history_quality_mask"], "history_quality_mask"),
        "split": payload["split"],
    }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot estimate a median without valid observations")
    high = ordered[len(ordered) // 2]
    if len(ordered) % 2:
        return high
    low = ordered[len(ordered) // 2 - 1]
    return (low + high) / 2 if low <= 0 <= high else low + (high - low) / 2


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return (1 - weight) * ordered[low] + weight * ordered[high]


def _ols_slope(values: Sequence[float], valid: Sequence[bool]) -> float | None:
    positions = [index / 6.0 for index, is_valid in enumerate(valid) if is_valid]
    selected = [value for value, is_valid in zip(values, valid) if is_valid]
    if len(selected) < 4:
        return None
    mean_t = math.fsum(positions) / len(positions)
    mean_y = math.fsum(selected) / len(selected)
    denominator = math.fsum((position - mean_t) ** 2 for position in positions)
    if denominator <= 0:
        return None
    return _number(math.fsum((position - mean_t) * (value - mean_y)
                            for position, value in zip(positions, selected)) / denominator, "OLS slope")


def _mad_volatility(values: Sequence[float], valid: Sequence[bool]) -> float | None:
    selected = [value for value, is_valid in zip(values, valid) if is_valid]
    if len(selected) < 4:
        return None
    center = _f32(_median(selected))
    deviations = [_f32(abs(value - center)) for value in selected]
    return _f32(1.4826 * _f32(_median(deviations)), "MAP volatility")


@dataclass(frozen=True)
class VitalTrainingReference:
    """Float32 training statistics over quality-valid bins in each window's final 30 steps."""

    factor_median: tuple[float, ...]
    factor_iqr: tuple[float, ...]
    robust_lower: tuple[float, ...]
    robust_upper: tuple[float, ...]
    valid_bin_counts: tuple[int, ...]
    training_inputs: tuple[tuple[str, str, str], ...]

    def __post_init__(self) -> None:
        for name in ("factor_median", "factor_iqr", "robust_lower", "robust_upper"):
            values = tuple(_f32(value, name) for value in getattr(self, name))
            if len(values) != len(FACTOR_NAMES):
                raise ValueError("VitalDB v2 training statistics must cover all 20 factors")
            object.__setattr__(self, name, values)
        if any(value < 0 for value in self.factor_iqr):
            raise ValueError("Factor IQR must be nonnegative")
        if any(low > median or median > high for low, median, high in
               zip(self.robust_lower, self.factor_median, self.robust_upper)):
            raise ValueError("Factor median must lie within its robust quantiles")
        counts = tuple(self.valid_bin_counts)
        records = tuple(tuple(record) for record in self.training_inputs)
        if len(counts) != len(FACTOR_NAMES) or any(type(count) is not int or count < 1 for count in counts):
            raise ValueError("All 20 factors need positive valid-bin support counts")
        if not records:
            raise ValueError("Training-reference provenance must not be empty")
        seen = set()
        for record in records:
            if len(record) != 3:
                raise ValueError("Training identity needs input_id, group_id, and digest")
            input_id, group_id, digest = record
            _string(input_id, "Training input ID")
            _string(group_id, "Training group ID")
            _string(digest, "Training input digest")
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError("Training digest must be a lowercase SHA-256")
            if input_id in seen:
                raise ValueError("Training input IDs must be unique")
            seen.add(input_id)
        if any(count > INPUT_BINS * len(records) for count in counts):
            raise ValueError("Valid-bin support exceeds the declared last-30-bin training windows")
        object.__setattr__(self, "valid_bin_counts", counts)
        object.__setattr__(self, "training_inputs", tuple(sorted(records)))
        self.factor_scale

    @property
    def medians(self) -> dict[str, float]:
        return dict(zip(FACTOR_NAMES, self.factor_median))

    @property
    def factor_scale(self) -> tuple[float, ...]:
        epsilon = _f32(EPSILON)
        result = []
        for iqr, low, high in zip(self.factor_iqr, self.robust_lower, self.robust_upper):
            base = iqr if iqr > 0 else max(_f32(high - low, "Robust factor range"), epsilon)
            result.append(_f32(base + epsilon, "Factor scale"))
        return tuple(result)

    @property
    def fingerprint(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def fit(cls, training_instances: Sequence[Instance]) -> VitalTrainingReference:
        bins: list[list[float]] = [[] for _ in FACTOR_NAMES]
        records, seen = [], set()
        for instance in training_instances:
            _string(instance.input_id, "Training input ID")
            _string(instance.group_id, "Training group ID")
            if instance.input_id in seen:
                raise ValueError("Training input IDs must be unique")
            seen.add(instance.input_id)
            prepared = _payload(instance.payload)
            if prepared["split"] != "train":
                raise ValueError("VitalTrainingReference.fit accepts train split only")
            records.append((instance.input_id, instance.group_id, content_hash(prepared)))
            for row in range(PREHISTORY_BINS, HISTORY_BINS):
                for column, valid in enumerate(prepared["history_quality_mask"][row]):
                    if valid:
                        bins[column].append(prepared["history_values"][row][column])
        if not records or any(not values for values in bins):
            raise ValueError("Train windows need at least one quality-valid input bin for every factor")
        medians, iqrs, lowers, uppers = [], [], [], []
        for values in bins:
            values.sort()
            q001, q25, median, q75, q999 = [_quantile(values, q) for q in (.001, .25, .5, .75, .999)]
            medians.append(_f32(median))
            iqrs.append(_f32(q75 - q25))
            lowers.append(_f32(q001))
            uppers.append(_f32(q999))
        return cls(tuple(medians), tuple(iqrs), tuple(lowers), tuple(uppers),
                   tuple(len(values) for values in bins), tuple(records))

    def impute_history(self, values: Sequence[Sequence[float]], quality_mask: Sequence[Sequence[bool]]) -> list[list[float]]:
        """Apply training-median imputation before intervention."""
        rows = _matrix_rows(values, "history_values")
        quality = _mask(quality_mask, "history_quality_mask")
        result = []
        for row, valid in zip(rows, quality):
            output = []
            for column, (value, is_valid) in enumerate(zip(row, valid)):
                if isinstance(value, (str, bytes, bool)):
                    raise ValueError("Imputation requires numerical entries, possibly NaN/Inf")
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("Imputation requires numerical entries, possibly NaN/Inf") from exc
                output.append(_f32(number) if is_valid and math.isfinite(number) else self.factor_median[column])
            result.append(output)
        return result

    def model_input(self, payload: Any) -> list[list[float]]:
        """Normalize physical factors and append observed masks, preserving intervened values."""
        prepared = _payload(payload)
        scales = self.factor_scale
        result = []
        for row in range(PREHISTORY_BINS, HISTORY_BINS):
            normalized = [
                _f32(_f32(value - median, "Centered factor") / scale, "Normalized factor")
                for value, median, scale in zip(prepared["history_values"][row], self.factor_median, scales)
            ]
            result.append(normalized + [float(value) for value in prepared["history_observed_mask"][row]])
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "vitaldb-training-reference-v2", "factor_names": list(FACTOR_NAMES),
            "history_bins": HISTORY_BINS, "input_bins": INPUT_BINS,
            "estimator": _ESTIMATOR, "factor_scale_policy": _SCALE_POLICY,
            "factor_median": list(self.factor_median), "factor_iqr": list(self.factor_iqr),
            "robust_lower": list(self.robust_lower), "robust_upper": list(self.robust_upper),
            "valid_bin_counts": list(self.valid_bin_counts),
            "training_inputs": [
                {"input_id": identifier, "group_id": group, "input_digest": digest, "split": "train"}
                for identifier, group, digest in self.training_inputs
            ],
        }

    @classmethod
    def from_dict(cls, source: Mapping[str, Any]) -> VitalTrainingReference:
        if not isinstance(source, Mapping) or source.get("schema_version") != "vitaldb-training-reference-v2":
            raise ValueError("VitalDB training reference must use v2; single-mask/v1 assets are unsupported")
        names = source.get("factor_names")
        if not isinstance(names, (list, tuple)) or tuple(names) != FACTOR_NAMES:
            raise ValueError("Training factor order does not match the adapter")
        expected = {"history_bins": HISTORY_BINS, "input_bins": INPUT_BINS,
                    "estimator": _ESTIMATOR, "factor_scale_policy": _SCALE_POLICY}
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError("Training-reference data or statistics policy differs from VitalDB v2")
        vectors = []
        for name in ("factor_median", "factor_iqr", "robust_lower", "robust_upper", "valid_bin_counts"):
            vector = source.get(name)
            if not isinstance(vector, (list, tuple)):
                raise ValueError(f"Training reference needs vector {name}")
            vectors.append(tuple(vector))
        inputs = source.get("training_inputs")
        if not isinstance(inputs, (list, tuple)):
            raise ValueError("Training input provenance must be a sequence")
        records = []
        for entry in inputs:
            if not isinstance(entry, Mapping) or entry.get("split") != "train":
                raise ValueError("Training input provenance must use train split only")
            records.append((entry.get("input_id"), entry.get("group_id"), entry.get("input_digest")))
        return cls(*vectors, tuple(records))


class VitalDBAdapter:

    method_label = "prepared VitalDB v2 adapter; new execution, paper results not reproduced"
    unit_ids = UNIT_IDS

    def __init__(self, training_reference: VitalTrainingReference) -> None:
        if not isinstance(training_reference, VitalTrainingReference):
            raise TypeError("VitalDBAdapter requires a VitalTrainingReference")
        self._training_reference = training_reference
        self._training_groups = frozenset(group for _, group, _ in training_reference.training_inputs)
        self._training_identity = {identifier: (group, digest)
                                   for identifier, group, digest in training_reference.training_inputs}

    @property
    def training_reference(self) -> VitalTrainingReference:
        return self._training_reference

    @property
    def unit_catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple({
            "unit_id": unit_id,
            "description": (f"{unit_id.rsplit('::', 1)[-1]}: {base} ({FACTOR_UNITS[base]}), input bins [{start},{stop}), "
                            f"relative seconds [{10 * start - 300},{10 * stop - 300}); "
                            + ("registered derived updates extend at most fifty seconds after the block."
                               if REGISTERED_DERIVED[base] else "no registered derived updates.")),
            "scope": {"start_bin": start, "stop_bin": stop,
                      "history_start_bin": start + PREHISTORY_BINS, "history_stop_bin": stop + PREHISTORY_BINS,
                      "base_variable": base, "base_column": FACTOR_INDEX[base],
                      "derived_variables": list(REGISTERED_DERIVED[base])},
        } for unit_id, (start, stop, base) in _UNIT_SPEC.items())

    @property
    def fingerprint(self) -> str:
        return content_hash(self.prompt_metadata())

    def prompt_metadata(self) -> dict[str, Any]:
        return {
            "adapter_version": "vitaldb-prepared-v2",
            "dataset_description": (
                "VitalDB: 42 ten-second physical-value history bins, including twelve auxiliary prehistory bins; "
                "the model uses the final thirty bins of twenty normalized factors plus twenty observed masks. "
                "Quality masks separately identify values available for numerical statistics."
            ),
            "target_description": (
                "Three frozen-predictor physical MAP outputs: medians over sixty-second windows ending "
                "at one, three, and five minutes after the input cutoff."
            ),
            "primary_operator_description": (
                "Replace all six selected base-factor values by the median of quality-valid values in "
                "the other twenty-four model-input bins; at least six valid bins are required, otherwise "
                "use the frozen training median. Prehistory never enters this replacement median. "
                "Recompute registered derived values using the unchanged six-bin history definition."
            ),
            "unit_scope_and_coupling_description": (
                "Twenty units: five one-minute blocks times MAP_family, HR_family, ETCO2_family, and SpO2. "
                "MAP updates MAP_slope and MAP_volatility; HR updates HR_slope; ETCO2 updates ETCO2_slope; "
                "SpO2 has no derived updates. Both masks, prehistory, pulse_pressure, waveform factors, "
                "medication, ventilation, and all other context are preserved."
            ),
            "unit_catalog": self.unit_catalog, "factor_names": list(FACTOR_NAMES),
            "factor_units": dict(FACTOR_UNITS),
            "source_numeric_tracks": dict(SOURCE_NUMERIC_TRACKS),
            "source_waveform_tracks": dict(SOURCE_WAVEFORM_TRACKS),
            "training_reference_fingerprint": self.training_reference.fingerprint,
            "derived_policy": copy.deepcopy(_DERIVED_POLICY),
            "input_normalization": _SCALE_POLICY,
            "attributes": {"level_or_state": "quality-valid block median", "trend": "quality-valid OLS per minute; at least four bins"},
            "quality": "four or more valid bins: valid; one to three: partial; zero: unavailable; no inferred clinical thresholds",
            "scope": self.method_label,
        }

    def _instance(self, instance: Instance) -> dict[str, Any]:
        _string(instance.input_id, "Input ID")
        _string(instance.group_id, "Patient/group ID")
        prepared = _payload(instance.payload)
        if prepared["split"] != "train" and instance.group_id in self._training_groups:
            raise ValueError("A training patient cannot also belong to validation or test")
        known = self._training_identity.get(instance.input_id)
        if known is not None and known != (instance.group_id, content_hash(prepared)):
            raise ValueError("A training input ID cannot identify a different payload or patient")
        return prepared

    def serialize_input(self, instance: Instance) -> Any:
        return self._instance(instance)

    def model_input(self, payload: Any) -> list[list[float]]:
        return self.training_reference.model_input(payload)

    def unit_observations(self, instance: Instance) -> dict[str, dict[str, dict[str, Any]]]:
        prepared = self._instance(instance)
        observations = {}
        for unit_id, (start, stop, base) in _UNIT_SPEC.items():
            column = FACTOR_INDEX[base]
            positions = range(PREHISTORY_BINS + start, PREHISTORY_BINS + stop)
            values = [prepared["history_values"][row][column] for row in positions]
            valid = [prepared["history_quality_mask"][row][column] for row in positions]
            selected = [value for value, is_valid in zip(values, valid) if is_valid]
            quality = "valid" if len(selected) >= 4 else ("partial" if selected else "unavailable")
            observations[unit_id] = {
                "level_or_state": {"value": _f32(_median(selected)) if selected else None,
                                   "unit": FACTOR_UNITS[base], "quality": quality},
                "trend": {"value": _ols_slope(values, valid), "unit": f"{FACTOR_UNITS[base]}/min", "quality": quality},
            }
        return observations

    def unit_attributes(self, instance: Instance) -> dict[str, dict[str, float | None]]:
        return {unit: {attribute: observation["value"] for attribute, observation in attributes.items()}
                for unit, attributes in self.unit_observations(instance).items()}

    def describe_input(self, instance: Instance) -> Mapping[str, Any]:
        return {"input_id": instance.input_id, "group_id": instance.group_id,
                "description": "Raw input-only VitalDB level/trend observations; training percentiles are supplied by the card wrapper",
                "unit_observations": self.unit_observations(instance)}

    @staticmethod
    def _units(unit_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(unit_ids, (str, bytes)):
            raise ValueError("Unit IDs must be a sequence")
        ids = tuple(unit_ids)
        if any(not isinstance(unit, str) or unit not in _UNIT_SPEC for unit in ids):
            raise ValueError("Unit is outside the VitalDB intervention catalog")
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate intervention units are not allowed")
        return ids

    @staticmethod
    def _recompute(prepared: dict[str, Any], changed: Mapping[str, set[int]]) -> dict[str, Any]:
        report = {}
        for base, model_positions in changed.items():
            base_column = FACTOR_INDEX[base]
            affected = sorted({row for position in model_positions
                               for row in range(PREHISTORY_BINS + position, min(PREHISTORY_BINS + position + 6, HISTORY_BINS))})
            for derived in REGISTERED_DERIVED[base]:
                column = FACTOR_INDEX[derived]
                updated, retained = [], []
                for row in affected:
                    window = range(max(0, row - 5), row + 1)
                    values = [prepared["history_values"][index][base_column] for index in window]
                    quality = [prepared["history_quality_mask"][index][base_column] for index in window]
                    value = _ols_slope(values, quality) if derived.endswith("_slope") else _mad_volatility(values, quality)
                    expected_valid = prepared["history_quality_mask"][row][column]
                    if (value is not None) != expected_valid:
                        raise ValueError(f"derived_mask_changed:{derived}")
                    if expected_valid:
                        prepared["history_values"][row][column] = _f32(value, derived)
                        updated.append(row - PREHISTORY_BINS)
                    else:
                        retained.append(row - PREHISTORY_BINS)
                report[derived] = {"affected_bins": [row - PREHISTORY_BINS for row in affected],
                                   "updated_bins": updated, "retained_unavailable_bins": retained}
        return report

    def perturb(self, instance: Instance, unit_id: str) -> Perturbation:
        self._units([unit_id])
        prepared = self._instance(instance)
        start, stop, base = _UNIT_SPEC[unit_id]
        column = FACTOR_INDEX[base]
        valid = [prepared["history_values"][PREHISTORY_BINS + row][column] for row in range(INPUT_BINS)
                 if not start <= row < stop and prepared["history_quality_mask"][PREHISTORY_BINS + row][column]]
        fallback = len(valid) < 6
        replacement = self.training_reference.medians[base] if fallback else _f32(_median(valid))
        for row in range(PREHISTORY_BINS + start, PREHISTORY_BINS + stop):
            prepared["history_values"][row][column] = replacement
        updates = self._recompute(prepared, {base: set(range(start, stop))})
        return Perturbation(prepared, {
            "operator": "vitaldb-out-of-block-median-v2", "unit_id": unit_id, "base_variable": base,
            "selected_bins": [start, stop], "out_of_block_valid_bins": len(valid),
            "replacement": replacement, "intensity": 1.0, "used_training_fallback": fallback,
            "training_reference_fingerprint": self.training_reference.fingerprint,
            "derived_policy": copy.deepcopy(_DERIVED_POLICY), "derived_updates": updates,
        })

    def joint_transform(self, instance: Instance, unit_ids: Sequence[str], donor: Instance, *, keep: bool) -> Any:
        selected = set(self._units(unit_ids))
        if not isinstance(keep, bool):
            raise ValueError("keep must be a boolean")
        prepared, background = self._instance(instance), self._instance(donor)
        if background["split"] != "validation":
            raise ValueError("VitalDB evaluation donors must come from validation")
        if prepared["split"] != "validation" and instance.group_id == donor.group_id:
            raise ValueError("A patient cannot span the donor validation split and another split")
        if instance.input_id == donor.input_id and content_hash(prepared) != content_hash(background):
            raise ValueError("An input ID cannot identify conflicting current and donor windows")
        replaced = set(UNIT_IDS) - selected if keep else selected
        changed: dict[str, set[int]] = {}
        for unit_id in sorted(replaced):
            start, stop, base = _UNIT_SPEC[unit_id]
            column = FACTOR_INDEX[base]
            for row in range(PREHISTORY_BINS + start, PREHISTORY_BINS + stop):
                prepared["history_values"][row][column] = background["history_values"][row][column]
            changed.setdefault(base, set()).update(range(start, stop))
        self._recompute(prepared, changed)
        return prepared


def predictor_tensor(payload: Any, reference: VitalTrainingReference) -> list[list[float]]:
    if not isinstance(reference, VitalTrainingReference):
        raise TypeError("predictor_tensor requires an explicit VitalTrainingReference")
    return reference.model_input(payload)


def make_vitaldb_task(predictor_id: str, adapter: VitalDBAdapter) -> TaskSpec:
    if not isinstance(adapter, VitalDBAdapter):
        raise TypeError("make_vitaldb_task requires a VitalDBAdapter")
    return TaskSpec(predictor_id=predictor_id, adapter_id=f"vitaldb-prepared-v2:{adapter.fingerprint}",
                    operator_id="vitaldb-out-of-block-median-v2", target_names=("map_1m", "map_3m", "map_5m"),
                    scales=OUTPUT_SCALES, unit_ids=UNIT_IDS)
