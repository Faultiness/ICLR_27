"""SHIR WeatherBench2 interventions in physical units and frozen per-channel standardization.

Seasonal references use a cyclic 366-day month/day calendar.
"""

from __future__ import annotations

from array import array
from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import math
import re
import struct
import sys
from threading import RLock
from typing import Any, Callable

from ..types import Instance, Perturbation, TaskSpec, canonical_json, content_hash


@dataclass(frozen=True)
class WeatherChannel:
    channel_id: str
    source_variable: str
    level_hpa: int | None
    unit: str


WEATHER_CHANNELS = (
    WeatherChannel("t2m", "2m_temperature", None, "K"),
    WeatherChannel("msl", "mean_sea_level_pressure", None, "Pa"),
    WeatherChannel("u10", "10m_u_component_of_wind", None, "m s**-1"),
    WeatherChannel("v10", "10m_v_component_of_wind", None, "m s**-1"),
    WeatherChannel("z500", "geopotential", 500, "m**2 s**-2"),
    WeatherChannel("t850", "temperature", 850, "K"),
    WeatherChannel("q850", "specific_humidity", 850, "kg kg**-1"),
    WeatherChannel("u850", "u_component_of_wind", 850, "m s**-1"),
    WeatherChannel("v850", "v_component_of_wind", 850, "m s**-1"),
)
CHANNELS = tuple(channel.channel_id for channel in WEATHER_CHANNELS)
CHANNEL_UNITS = tuple(channel.unit for channel in WEATHER_CHANNELS)
LATITUDES = tuple(1.5 * index for index in range(41))
LONGITUDES = tuple(75.0 + 1.5 * index for index in range(61))
TARGET_LATITUDES = tuple(15.0 + 1.5 * index for index in range(21))
TARGET_LONGITUDES = tuple(100.5 + 1.5 * index for index in range(31))
TARGET_NAMES = ("lead_6h", "lead_12h", "lead_24h")
OUTPUT_SCALES = (7.9266566744252325, 7.919746033395541, 7.879262215796636)
CALENDAR_RULE = "month-day-aligned-366-day-cycle-reference-year-2000"
OPERATOR_ID = "weather-training-seasonal-climatology-v1"
ASSET_SCHEMA = "shir-weather-climatology-v2"
FRAME_SHAPE = (9, 41, 61)
INPUT_SHAPE = (12, *FRAME_SHAPE)
FIELD_SHAPE = (3, 21, 31)
_FRAME_SIZE = math.prod(FRAME_SHAPE)
_GRID_SIZE = 41 * 61
_UTC = timezone.utc
_SEASONAL_ORIGIN = datetime(2000, 1, 1, tzinfo=_UTC)
_SPLIT_YEARS = {"train": (1979, 2017), "validation": (2018, 2019), "test": (2020, 2021)}
_TEMPORAL_SEGMENTS = (("old", 0, 4), ("middle", 4, 8), ("recent", 8, 12))
_VARIABLE_GROUPS = (
    ("surface_temperature", (0,)),
    ("surface_pressure", (1,)),
    ("surface_wind", (2, 3)),
    ("midlevel_circulation", (4,)),
    ("lower_thermodynamics", (5, 6)),
    ("lower_wind", (7, 8)),
)
# Zero-based, half-open bounds on increasing latitude/longitude axes.
_REGIONS = (
    ("south_belt", 0, 10, 0, 61),
    ("north_belt", 30, 41, 0, 61),
    ("upstream_west", 10, 30, 0, 17),
    ("target_core", 10, 30, 17, 47),
    ("downstream_east", 10, 30, 47, 61),
)


def _number(value: Any, label: str) -> float:
    if isinstance(value, (str, bytes, bool)):
        raise ValueError(f"{label} must be a finite physical-unit number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite physical-unit number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _float32(value: Any, label: str = "Weather value") -> float:
    value = _number(value, label)
    try:
        rounded = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError(f"{label} is not representable as finite float32") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{label} is not representable as finite float32")
    return rounded


def _sequence(value: Any, length: int, label: str) -> Any:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{label} must have length {length}")
    try:
        if len(value) != length:
            raise ValueError(f"{label} must have length {length}")
    except TypeError as exc:
        raise ValueError(f"{label} must have length {length}") from exc
    return value


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO strings in UTC")
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Timestamps must be ISO strings in UTC") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
        raise ValueError("Timestamps must explicitly use UTC, not local or naive time")
    if timestamp.hour not in (0, 6, 12, 18) or timestamp.minute or timestamp.second or timestamp.microsecond:
        raise ValueError("Weather timestamps must lie on the 00/06/12/18 UTC grid")
    return timestamp.astimezone(_UTC)


def _iso(timestamp: datetime) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _seasonal_day(timestamp: datetime) -> int:
    aligned = timestamp.replace(year=2000, hour=0)
    return (aligned - _SEASONAL_ORIGIN).days


def _seasonal_distance(first: datetime, second: datetime) -> int:
    difference = abs(_seasonal_day(first) - _seasonal_day(second))
    return min(difference, 366 - difference)


def _flat_frame(values: Any, label: str = "Weather frame", *, float32_input: bool = True) -> array:
    channels = _sequence(values, 9, f"{label} channels")
    flattened = array("d")
    for channel_index, channel in enumerate(channels):
        for row in _sequence(channel, 41, f"{label} latitude axis"):
            for value in _sequence(row, 61, f"{label} longitude axis"):
                number = _float32(value, label) if float32_input else _number(value, label)
                if channel_index == 6 and number < 0:
                    raise ValueError(f"{label} contains negative q850 specific humidity")
                flattened.append(number)
    return flattened


def _pack(values: Sequence[float]) -> bytes:
    packed = array("d", values)
    if packed.itemsize != 8:
        raise RuntimeError("Weather assets require an eight-byte IEEE binary64 double")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _view(packed: bytes) -> Sequence[float]:
    if sys.byteorder == "little":
        return memoryview(packed).cast("d")
    values = array("d")
    values.frombytes(packed)
    values.byteswap()
    return values


def _pack_float32(values: Sequence[float]) -> bytes:
    packed = array("f", (_float32(value) for value in values))
    if packed.itemsize != 4:
        raise RuntimeError("Weather assets require four-byte IEEE binary32 floats")
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _view_float32(packed: bytes) -> Sequence[float]:
    if sys.byteorder == "little":
        return memoryview(packed).cast("f")
    values = array("f")
    values.frombytes(packed)
    values.byteswap()
    return values


def _nested_frame(flattened: Sequence[float]) -> list[list[list[float]]]:
    return [
        [list(flattened[(channel * 41 + row) * 61:(channel * 41 + row + 1) * 61]) for row in range(41)]
        for channel in range(9)
    ]


def _mean(values: Sequence[float]) -> float:
    try:
        return math.fsum(values) / len(values)
    except OverflowError:
        # Scale before summation to avoid overflow.
        scale = max(abs(value) for value in values)
        return scale * (math.fsum(value / scale for value in values) / len(values))


@dataclass(frozen=True)
class _Bin:
    seasonal_day: int
    utc_hour: int
    count: int
    sums: bytes


class MissingClimatologyError(ValueError):
    """No training frame matches the seasonal window and UTC slot."""


class SeasonalClimatology:
    """Training climatology and per-channel normalization.

    Unique raw frames are pooled within 15 seasonal days at matching UTC slots.
    Precomputed climatology is already smoothed and is indexed directly.
    """

    def __init__(
        self, bins: tuple[_Bin, ...], sources: tuple[tuple[str, str], ...], *,
        channel_mean: tuple[float, ...], channel_std: tuple[float, ...],
        reference_years: tuple[int, ...], normalization_count: int | None,
        precomputed: tuple[tuple[int, bytes], ...] = (),
        provenance: Mapping[str, str] | None = None,
    ) -> None:


        self._bins = tuple(bins)
        self._sources = tuple(sources)
        self._channel_mean = tuple(channel_mean)
        self._channel_std = tuple(channel_std)
        self._reference_years = tuple(reference_years)
        self._normalization_count = normalization_count
        self._precomputed = tuple(precomputed)
        self._calendar_lookup = dict(precomputed)
        self._provenance = tuple(sorted((provenance or {}).items()))
        self._mode = "precomputed_lookup" if precomputed else "fitted_daily_sums"
        self._cache: OrderedDict[tuple[int, int], bytes] = OrderedDict()
        self._cache_lock = RLock()
        digest = hashlib.sha256()
        digest.update(canonical_json({
            "schema_version": ASSET_SCHEMA, "calendar_rule": CALENDAR_RULE,
            "window_days": 15, "training_years": [1979, 2017],
            "channels": CHANNELS, "latitudes": LATITUDES, "longitudes": LONGITUDES,
            "input_dtype": "float32", "sum_dtype": "float64", "lookup_dtype": "float32",
            "sum_encoding": "IEEE-754-binary64-little-endian", "sources": self._sources,
            "mode": self._mode, "reference_years": self._reference_years,
            "channel_mean": self._channel_mean, "channel_std": self._channel_std,
            "normalization_count": self._normalization_count,
            "provenance": self._provenance,
        }).encode("utf-8"))
        for item in self._bins:
            digest.update(canonical_json((item.seasonal_day, item.utc_hour, item.count)).encode("utf-8"))
            digest.update(item.sums)
        for key, values in self._precomputed:
            digest.update(canonical_json(key).encode("utf-8"))
            digest.update(values)
        self._fingerprint = digest.hexdigest()

    @classmethod
    def fit_training_frames(cls, frames: Iterable[Mapping[str, Any]]) -> SeasonalClimatology:
        """Fit 1979--2017 training frames shaped [9,41,61].

        Per-channel population statistics use unique frames and all grid cells.
        Accumulate float32 values in float64; freeze the statistics as float32.
        """
        sums: dict[tuple[int, int], array] = {}
        counts: Counter[tuple[int, int]] = Counter()
        sources: dict[str, str] = {}
        channel_totals = [0.0] * 9
        channel_squared_totals = [0.0] * 9
        for frame in frames:
            if not isinstance(frame, Mapping) or set(frame) != {"timestamp", "values", "split"}:
                raise ValueError("Training frames need exactly timestamp, values, and split")
            if frame["split"] != "train":
                raise ValueError("Climatology accepts training frames only")
            timestamp = _timestamp(frame["timestamp"])
            if not 1979 <= timestamp.year <= 2017:
                raise ValueError("Climatology training years are restricted to 1979--2017")
            values = _flat_frame(frame["values"])
            timestamp_id = _iso(timestamp)
            frame_digest = hashlib.sha256(_pack(values)).hexdigest()
            if timestamp_id in sources:
                if sources[timestamp_id] != frame_digest:
                    raise ValueError("Conflicting training frames at the same UTC timestamp")
                continue
            for channel in range(9):
                channel_values = values[channel * _GRID_SIZE:(channel + 1) * _GRID_SIZE]
                channel_totals[channel] += math.fsum(channel_values)
                channel_squared_totals[channel] += math.fsum(value * value for value in channel_values)
            key = (_seasonal_day(timestamp), timestamp.hour)
            if key not in sums:
                sums[key] = array("d", [0.0]) * _FRAME_SIZE
            accumulated = sums[key]
            for index, value in enumerate(values):
                total = accumulated[index] + value
                if not math.isfinite(total):
                    raise ValueError("Climatology sum overflowed; no nonfinite statistics are stored")
                accumulated[index] = total
            counts[key] += 1
            sources[timestamp_id] = frame_digest
        if not sources:
            raise ValueError("Climatology fitting requires at least one training frame")
        bins = tuple(_Bin(day, hour, counts[(day, hour)], _pack(sums[(day, hour)]))
                     for day, hour in sorted(sums))
        normalization_count = len(sources) * _GRID_SIZE
        means = tuple(total / normalization_count for total in channel_totals)
        deviations = tuple(math.sqrt(max(square / normalization_count - mean * mean, 0.0))
                           for square, mean in zip(channel_squared_totals, means))
        return cls(
            bins, tuple(sorted(sources.items())),
            channel_mean=tuple(_float32(value, "Training channel mean") for value in means),
            channel_std=tuple(_float32(value, "Training channel standard deviation") for value in deviations),
            reference_years=tuple(sorted({_timestamp(value).year for value in sources})),
            normalization_count=normalization_count,
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def channel_mean(self) -> tuple[float, ...]:
        return self._channel_mean

    @property
    def channel_std(self) -> tuple[float, ...]:
        return self._channel_std

    @property
    def reference_years(self) -> tuple[int, ...]:
        return self._reference_years

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def source_frame_count(self) -> int | None:
        # Precomputed references may omit source-frame records.
        return None if self._precomputed else len(self._sources)

    def _eligible_bins(self, timestamp: datetime) -> tuple[_Bin, ...]:
        day = _seasonal_day(timestamp)
        return tuple(item for item in self._bins if item.utc_hour == timestamp.hour
                     and min(abs(item.seasonal_day - day), 366 - abs(item.seasonal_day - day)) <= 15)

    def support_count(self, timestamp: str) -> int | None:
        value = _timestamp(timestamp)
        if self._precomputed:
            key = _seasonal_day(value) * 4 + value.hour // 6
            return None if key in self._calendar_lookup else 0
        return sum(item.count for item in self._eligible_bins(value))

    def _lookup_flat(self, timestamp: datetime) -> Sequence[float]:
        if self._precomputed:
            calendar_key = _seasonal_day(timestamp) * 4 + timestamp.hour // 6
            if calendar_key not in self._calendar_lookup:
                raise MissingClimatologyError("Precomputed climatology lacks this exact calendar key")
            return _view_float32(self._calendar_lookup[calendar_key])
        key = (_seasonal_day(timestamp), timestamp.hour)
        with self._cache_lock:
            if key not in self._cache:
                bins = self._eligible_bins(timestamp)
                count = sum(item.count for item in bins)
                if not count:
                    raise MissingClimatologyError("No training climatology for this seasonal window and UTC slot")
                arrays = [_view(item.sums) for item in bins]
                means = array("d", (math.fsum(values[index] / count for values in arrays)
                                    for index in range(_FRAME_SIZE)))
                self._cache[key] = _pack_float32(means)
                if len(self._cache) > 32:
                    self._cache.popitem(last=False)
            else:
                self._cache.move_to_end(key)
            result = self._cache[key]
        return _view_float32(result)

    def lookup(self, timestamp: str) -> list[list[list[float]]]:
        """Return a [9,41,61] seasonal-mean frame."""
        return _nested_frame(self._lookup_flat(_timestamp(timestamp)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ASSET_SCHEMA,
            "mode": self._mode,
            "calendar_rule": CALENDAR_RULE,
            "window_days": 15,
            "training_years": [1979, 2017],
            "reference_years": list(self._reference_years),
            "channels": list(CHANNELS),
            "latitudes": list(LATITUDES),
            "longitudes": list(LONGITUDES),
            "input_dtype": "float32", "sum_dtype": "float64", "lookup_dtype": "float32",
            "channel_mean": list(self._channel_mean), "channel_std": list(self._channel_std),
            "normalization_count": self._normalization_count,
            "source_hash_encoding": "sha256-IEEE-754-binary64-little-endian-C-order",
            "source_frames": [{"timestamp": timestamp, "split": "train", "sha256": digest}
                              for timestamp, digest in self._sources],
            "bins": [{"seasonal_day": item.seasonal_day, "utc_hour": item.utc_hour,
                      "count": item.count, "sums": _nested_frame(_view(item.sums))}
                     for item in self._bins],
            "precomputed": ({
                **dict(self._provenance),
                "calendar_keys": [key for key, _ in self._precomputed],
                "climatology_values": [_nested_frame(_view_float32(values)) for _, values in self._precomputed],
            } if self._precomputed else None),
            "fingerprint": self.fingerprint,
        }

    @staticmethod
    def _normalization(data: Mapping[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        means = tuple(_float32(value, "Training channel mean") for value in
                      _sequence(data["channel_mean"], 9, "Channel means"))
        deviations = tuple(_float32(value, "Training channel standard deviation") for value in
                           _sequence(data["channel_std"], 9, "Channel standard deviations"))
        if any(value < 0 for value in deviations):
            raise ValueError("Channel standard deviations must be nonnegative")
        return means, deviations

    @staticmethod
    def _years(values: Any) -> tuple[int, ...]:
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Climatology needs explicit training reference years")
        if any(type(year) is not int or not 1979 <= year <= 2017 for year in values):
            raise ValueError("Climatology reference years must lie within 1979--2017")
        if len(set(values)) != len(values):
            raise ValueError("Climatology reference years must be unique")
        return tuple(sorted(values))

    @classmethod
    def from_precomputed(cls, data: Mapping[str, Any]) -> SeasonalClimatology:
        """Import already-smoothed climatology without another smoothing pass.

        calendar_keys = day*4 + UTC slot on the 2000 calendar; values: [N,9,41,61].
        Supply channel_mean/std [9], training reference_years, reference_id,
        source_split='train', day_window=15, and calendar_rule=CALENDAR_RULE.
        """
        required = {"calendar_keys", "climatology_values", "channel_mean", "channel_std",
                    "reference_years", "source_split", "reference_id", "day_window", "calendar_rule"}
        if (not isinstance(data, Mapping) or not required <= set(data)
                or set(data) - required - {"source_sha256"}):
            raise ValueError("Precomputed climatology fields do not match the import contract")
        if data["source_split"] != "train":
            raise ValueError("Precomputed climatology must have train-only provenance")
        if data["day_window"] != 15 or data["calendar_rule"] != CALENDAR_RULE:
            raise ValueError("Precomputed climatology must use the fixed 366-day calendar and +/-15-day window")
        reference_id = data["reference_id"]
        if not isinstance(reference_id, str) or not reference_id:
            raise ValueError("Precomputed climatology needs a source reference_id")
        years = cls._years(data["reference_years"])
        means, deviations = cls._normalization(data)
        keys = data["calendar_keys"]
        if not isinstance(keys, (list, tuple)) or not keys:
            raise ValueError("Precomputed climatology needs a nonempty calendar-key list")
        if any(type(key) is not int or not 0 <= key < 1464 for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Precomputed calendar keys must be unique integers in [0,1464)")
        values = _sequence(data["climatology_values"], len(keys), "Precomputed climatology frames")
        frames = tuple(sorted((key, _pack_float32(_flat_frame(frame, "Precomputed climatology")))
                              for key, frame in zip(keys, values)))
        provenance = {"reference_id": reference_id, "source_split": "train"}
        if "source_sha256" in data:
            source_digest = data["source_sha256"]
            if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
                raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
            provenance["source_sha256"] = source_digest
        return cls((), (), channel_mean=means, channel_std=deviations, reference_years=years,
                   normalization_count=None, precomputed=frames, provenance=provenance)

    @classmethod
    def from_dict(cls, artifact: Mapping[str, Any]) -> SeasonalClimatology:
        if not isinstance(artifact, Mapping):
            raise ValueError("Climatology artifact must be a mapping")
        expected = {
            "schema_version": ASSET_SCHEMA, "calendar_rule": CALENDAR_RULE, "window_days": 15,
            "training_years": [1979, 2017], "channels": list(CHANNELS),
            "latitudes": list(LATITUDES), "longitudes": list(LONGITUDES),
            "input_dtype": "float32", "sum_dtype": "float64", "lookup_dtype": "float32",
            "source_hash_encoding": "sha256-IEEE-754-binary64-little-endian-C-order",
        }
        additional = {"mode", "reference_years", "channel_mean", "channel_std", "normalization_count",
                      "source_frames", "bins", "precomputed", "fingerprint"}
        if set(artifact) != set(expected) | additional:
            raise ValueError("Climatology artifact fields do not match its schema")
        if any(artifact[key] != value for key, value in expected.items()):
            raise ValueError("Climatology metadata or coordinate/channel convention is incompatible")
        years = cls._years(artifact["reference_years"])
        means, deviations = cls._normalization(artifact)
        if artifact["mode"] == "precomputed_lookup":
            if artifact["source_frames"] != [] or artifact["bins"] != [] or artifact["normalization_count"] is not None:
                raise ValueError("Precomputed lookup cannot masquerade as source-frame daily sums")
            precomputed = artifact["precomputed"]
            if not isinstance(precomputed, Mapping):
                raise ValueError("Precomputed lookup data is missing")
            required_lookup = {"reference_id", "source_split", "calendar_keys", "climatology_values"}
            if not required_lookup <= set(precomputed) or set(precomputed) - required_lookup - {"source_sha256"}:
                raise ValueError("Precomputed lookup fields do not match the asset schema")
            result = cls.from_precomputed({
                **precomputed, "channel_mean": list(means), "channel_std": list(deviations),
                "reference_years": list(years), "day_window": 15, "calendar_rule": CALENDAR_RULE,
            })
            if artifact["fingerprint"] != result.fingerprint:
                raise ValueError("Climatology fingerprint mismatch")
            return result
        if artifact["mode"] != "fitted_daily_sums" or artifact["precomputed"] is not None:
            raise ValueError("Invalid climatology storage mode")
        if not isinstance(artifact["source_frames"], (list, tuple)) or not artifact["source_frames"]:
            raise ValueError("Climatology requires a nonempty source-frame manifest")
        sources: dict[str, str] = {}
        counts: Counter[tuple[int, int]] = Counter()
        for source in artifact["source_frames"]:
            if not isinstance(source, Mapping) or set(source) != {"timestamp", "split", "sha256"}:
                raise ValueError("Malformed climatology source-frame entry")
            timestamp = _timestamp(source["timestamp"])
            if source["split"] != "train" or not 1979 <= timestamp.year <= 2017:
                raise ValueError("Climatology sources must be training frames from 1979--2017")
            timestamp_id = _iso(timestamp)
            if timestamp_id in sources:
                raise ValueError("Climatology source timestamps must be unique")
            source_digest = source["sha256"]
            if not isinstance(source_digest, str) or re.fullmatch(r"[0-9a-f]{64}", source_digest) is None:
                raise ValueError("A source frame needs its binary64 SHA-256 digest")
            sources[timestamp_id] = source_digest
            counts[(_seasonal_day(timestamp), timestamp.hour)] += 1
        if not isinstance(artifact["bins"], (list, tuple)):
            raise ValueError("Climatology bins must be a sequence")
        bins: dict[tuple[int, int], _Bin] = {}
        for item in artifact["bins"]:
            if not isinstance(item, Mapping) or set(item) != {"seasonal_day", "utc_hour", "count", "sums"}:
                raise ValueError("Malformed climatology statistics bin")
            day, hour, count = item["seasonal_day"], item["utc_hour"], item["count"]
            if (type(day) is not int or not 0 <= day < 366 or type(hour) is not int
                    or hour not in (0, 6, 12, 18) or type(count) is not int or count <= 0):
                raise ValueError("Invalid climatology seasonal day, UTC slot, or source count")
            key = (day, hour)
            if key in bins or counts[key] != count:
                raise ValueError("Climatology bin counts must match unique source timestamps")
            bins[key] = _Bin(day, hour, count, _pack(_flat_frame(item["sums"], "Climatology sum", float32_input=False)))
        if set(bins) != set(counts):
            raise ValueError("Climatology bins and source-frame manifest do not match")
        if years != tuple(sorted({_timestamp(value).year for value in sources})):
            raise ValueError("Climatology reference years disagree with its source-frame manifest")
        normalization_count = artifact["normalization_count"]
        if type(normalization_count) is not int or normalization_count != len(sources) * _GRID_SIZE:
            raise ValueError("Normalization count must reflect all unique training-frame grid cells")
        result = cls(tuple(bins[key] for key in sorted(bins)), tuple(sorted(sources.items())),
                     channel_mean=means, channel_std=deviations, reference_years=years,
                     normalization_count=normalization_count)
        if artifact["fingerprint"] != result.fingerprint:
            raise ValueError("Climatology fingerprint mismatch")
        return result


@dataclass(frozen=True)
class _Unit:
    unit_id: str
    temporal: str
    start: int
    stop: int
    variable_group: str
    channels: tuple[int, ...]
    region: str
    row_start: int
    row_stop: int
    column_start: int
    column_stop: int


_UNITS = tuple(
    _Unit(f"wb2::{temporal}::{group}::{region}", temporal, start, stop, group, channels,
          region, row_start, row_stop, column_start, column_stop)
    for temporal, start, stop in _TEMPORAL_SEGMENTS
    for group, channels in _VARIABLE_GROUPS
    for region, row_start, row_stop, column_start, column_stop in _REGIONS
)
UNIT_IDS = tuple(unit.unit_id for unit in _UNITS)
_UNIT_BY_ID = {unit.unit_id: unit for unit in _UNITS}


def _selected_units(unit_ids: Sequence[str]) -> tuple[_Unit, ...]:
    if isinstance(unit_ids, (str, bytes)):
        raise ValueError("Unit IDs must be a sequence")
    try:
        ids = tuple(unit_ids)
    except TypeError as exc:
        raise ValueError("Unit IDs must be a sequence") from exc
    if any(not isinstance(unit_id, str) or unit_id not in _UNIT_BY_ID for unit_id in ids):
        raise ValueError("Unknown Weather unit")
    if len(ids) != len(set(ids)):
        raise ValueError("Weather unit IDs must be unique")
    return tuple(_UNIT_BY_ID[unit_id] for unit_id in ids)


@dataclass(frozen=True)
class _Prepared:
    values: tuple[bytes, ...]
    timestamps: tuple[datetime, ...]
    split: str

    def to_payload(self) -> dict[str, Any]:
        return {"values": [_nested_frame(_view(frame)) for frame in self.values],
                "timestamps": [_iso(timestamp) for timestamp in self.timestamps], "split": self.split}


def _prepared(payload: Any) -> _Prepared:
    if not isinstance(payload, Mapping) or set(payload) != {"values", "timestamps", "split"}:
        raise ValueError("Prepared weather payload needs exactly values, timestamps, and split")
    split = payload["split"]
    if not isinstance(split, str) or split not in _SPLIT_YEARS:
        raise ValueError("Weather split must be train, validation, or test")
    timestamps = tuple(_timestamp(value) for value in _sequence(payload["timestamps"], 12, "Input timestamps"))
    if any(second - first != timedelta(hours=6) for first, second in zip(timestamps, timestamps[1:])):
        raise ValueError("Input timestamps must be consecutive six-hourly UTC frames")
    first_year, last_year = _SPLIT_YEARS[split]
    start, end = datetime(first_year, 1, 1, tzinfo=_UTC), datetime(last_year + 1, 1, 1, tzinfo=_UTC)
    if timestamps[0] < start or timestamps[-1] + timedelta(hours=24) >= end:
        raise ValueError("The complete input-to-last-target interval must stay within its data split")
    frames = tuple(_pack(_flat_frame(frame)) for frame in _sequence(payload["values"], 12, "Input frame axis"))
    return _Prepared(frames, timestamps, split)


class WeatherAdapter:

    def __init__(self, climatology: SeasonalClimatology) -> None:
        if not isinstance(climatology, SeasonalClimatology):
            raise TypeError("WeatherAdapter requires a fitted SeasonalClimatology")
        self._climatology = climatology
        self._fingerprint = content_hash({
            "adapter_contract": "weather-prepared-physical-v2", "climatology": climatology.fingerprint,
            "channels": CHANNELS, "latitudes": LATITUDES, "longitudes": LONGITUDES,
            "channel_metadata": [asdict(channel) for channel in WEATHER_CHANNELS],
            "attributes": "cos-lat-climatology-anomaly-and-adjacent-rms-v1",
            "normalization": "training-per-channel-population-binary32-v1",
        })

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def climatology(self) -> SeasonalClimatology:
        return self._climatology

    @property
    def unit_catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple({
            "unit_id": unit.unit_id,
            "description": (
                f"{unit.temporal} six-hourly input frames {unit.start}:{unit.stop}; "
                f"{unit.variable_group} ({', '.join(CHANNELS[c] for c in unit.channels)}); "
                f"{unit.region}, {LATITUDES[unit.row_start]}--{LATITUDES[unit.row_stop - 1]} N, "
                f"{LONGITUDES[unit.column_start]}--{LONGITUDES[unit.column_stop - 1]} E"
            ),
            "scope": {
                "time_indices": [unit.start, unit.stop],
                "channels": [CHANNELS[c] for c in unit.channels],
                "channel_units": {CHANNELS[c]: CHANNEL_UNITS[c] for c in unit.channels},
                "latitude_indices": [unit.row_start, unit.row_stop],
                "longitude_indices": [unit.column_start, unit.column_stop],
            },
        } for unit in _UNITS)

    def serialize_input(self, instance: Instance) -> dict[str, Any]:
        return _prepared(instance.payload).to_payload()

    def model_input(self, payload: Any) -> list[list[list[array]]]:
        """Standardize [12,9,41,61] physical inputs with float32 subtraction and division."""
        if any(scale <= 0 for scale in self.climatology.channel_std):
            raise ValueError("Cannot standardize Weather input: a training channel has zero standard deviation")
        prepared = _prepared(payload)
        output = []
        for packed in prepared.values:
            frame = _view(packed)
            channels = []
            for channel, (mean, scale) in enumerate(zip(self.climatology.channel_mean, self.climatology.channel_std)):
                rows = []
                for row in range(41):
                    offset = (channel * 41 + row) * 61
                    rows.append(array("f", (
                        _float32(_float32(frame[offset + column] - mean, "Centered Weather input") / scale,
                                 "Standardized Weather input")
                        for column in range(61)
                    )))
                channels.append(rows)
            output.append(channels)
        return output

    def unit_observations(self, instance: Instance) -> dict[str, dict[str, dict[str, Any]]]:
        """Compute 210 physical input observations.

        Levels are latitude-weighted climatology anomalies; variability is the
        weighted RMS of adjacent six-hour differences. Wind uses speed anomalies
        and vector changes. t850/q850 retain separate properties.
        """
        prepared = _prepared(instance.payload)
        frames = [_view(frame) for frame in prepared.values]
        climates = [self.climatology._lookup_flat(timestamp) for timestamp in prepared.timestamps]
        observations: dict[str, dict[str, dict[str, Any]]] = {}
        for unit in _UNITS:
            positions = [(row * 61 + column, math.cos(math.radians(LATITUDES[row])))
                         for row in range(unit.row_start, unit.row_stop)
                         for column in range(unit.column_start, unit.column_stop)]
            spatial_weight = (unit.column_stop - unit.column_start) * math.fsum(
                math.cos(math.radians(LATITUDES[row])) for row in range(unit.row_start, unit.row_stop)
            )
            level_denominator = (unit.stop - unit.start) * spatial_weight
            variability_denominator = (unit.stop - unit.start - 1) * spatial_weight
            properties = {}
            if unit.variable_group in {"surface_wind", "lower_wind"}:
                u_offset, v_offset = (channel * _GRID_SIZE for channel in unit.channels)
                level_terms = []
                variability_terms = []
                for time in range(unit.start, unit.stop):
                    for position, weight in positions:
                        u, v = frames[time][u_offset + position], frames[time][v_offset + position]
                        cu, cv = climates[time][u_offset + position], climates[time][v_offset + position]
                        level_terms.append((math.sqrt(u * u + v * v) - math.sqrt(cu * cu + cv * cv)) * weight)
                        if time + 1 < unit.stop:
                            du = _float32(frames[time + 1][u_offset + position] - u, "Wind frame difference")
                            dv = _float32(frames[time + 1][v_offset + position] - v, "Wind frame difference")
                            variability_terms.append((du * du + dv * dv) * weight)
                properties["level_or_state"] = {
                    "value": math.fsum(level_terms) / level_denominator,
                    "unit": "m s**-1", "quality": "valid",
                }
                properties["local_variability"] = {
                    "value": math.sqrt(max(math.fsum(variability_terms) / variability_denominator, 0.0)),
                    "unit": "m s**-1", "quality": "valid",
                }
            else:
                for channel in unit.channels:
                    offset = channel * _GRID_SIZE
                    suffix = f"::{CHANNELS[channel]}" if unit.variable_group == "lower_thermodynamics" else ""
                    level_terms = []
                    variability_terms = []
                    for time in range(unit.start, unit.stop):
                        for position, weight in positions:
                            current = frames[time][offset + position]
                            anomaly = _float32(current - climates[time][offset + position], "Weather anomaly")
                            level_terms.append(anomaly * weight)
                            if time + 1 < unit.stop:
                                delta = _float32(frames[time + 1][offset + position] - current, "Weather frame difference")
                                variability_terms.append(delta * delta * weight)
                    properties[f"level_or_state{suffix}"] = {
                        "value": math.fsum(level_terms) / level_denominator,
                        "unit": CHANNEL_UNITS[channel], "quality": "valid",
                    }
                    properties[f"local_variability{suffix}"] = {
                        "value": math.sqrt(max(math.fsum(variability_terms) / variability_denominator, 0.0)),
                        "unit": CHANNEL_UNITS[channel], "quality": "valid",
                    }
            if any(not math.isfinite(item["value"]) for item in properties.values()):
                raise ValueError("Weather observation statistics overflowed")
            observations[unit.unit_id] = properties
        return observations

    def unit_attributes(self, instance: Instance) -> dict[str, dict[str, float]]:
        return {unit: {name: item["value"] for name, item in values.items()}
                for unit, values in self.unit_observations(instance).items()}

    def describe_input(self, instance: Instance) -> dict[str, Any]:
        return {"dataset": "WeatherBench2", "observation_contract":
                "Cosine-latitude-weighted climatology anomaly levels and adjacent-frame-change RMS; coupled wind vectors",
                "unit_observations": self.unit_observations(instance)}

    def prompt_metadata(self) -> dict[str, Any]:
        return {
            "dataset_description": {"name": "WeatherBench2", "input_shape": list(INPUT_SHAPE),
                        "channels": list(CHANNELS), "channel_metadata": [asdict(channel) for channel in WEATHER_CHANNELS],
                        "sampling_hours": 6, "physical_units": True, "physical_input_dtype": "float32",
                        "model_input_space": "per-channel training-standardized float32",
                        "normalization": {"mean": self.climatology.channel_mean, "std": self.climatology.channel_std,
                                          "estimator": "population over unique training frames and all grid cells",
                                          "zero_standard_deviation": "fail at model_input; never replace by one"},
                        "observation_properties": "210 latitude-weighted level anomalies and adjacent-frame RMS properties"},
            "target_description": {"names": list(TARGET_NAMES), "horizons_hours": [6, 12, 24],
                       "field_shape": list(FIELD_SHAPE), "latitude_centers": list(TARGET_LATITUDES),
                       "longitude_centers": list(TARGET_LONGITUDES),
                       "aggregation": "cosine-latitude-weighted regional mean", "unit": "K"},
            "primary_operator_description": {"id": OPERATOR_ID, "intensity": 1.0, "window_days": 15,
                         "training_years": [1979, 2017], "calendar_rule": CALENDAR_RULE,
                         "reference_years": self.climatology.reference_years,
                         "climatology_mode": self.climatology.mode,
                         "climatology_fingerprint": self.climatology.fingerprint,
                         "missing_climatology": "fail; no fallback", "utc_slots": [0, 6, 12, 18]},
            "unit_scope_and_coupling_description": {"count": 90, "time_segments": 3, "variable_groups": 6, "spatial_regions": 5,
                           "coupled_channels": [["u10", "v10"], ["t850", "q850"], ["u850", "v850"]],
                           "derived_updates": [], "coverage": "entire 12x9x41x61 input"},
            "unit_catalog": self.unit_catalog,
        }

    def perturb(self, instance: Instance, unit_id: str) -> Perturbation:
        unit = _selected_units((unit_id,))[0]
        prepared = _prepared(instance.payload)
        payload = prepared.to_payload()
        for time in range(unit.start, unit.stop):
            baseline = self.climatology._lookup_flat(prepared.timestamps[time])
            for channel in unit.channels:
                for row in range(unit.row_start, unit.row_stop):
                    offset = (channel * 41 + row) * 61
                    for column in range(unit.column_start, unit.column_stop):
                        payload["values"][time][channel][row][column] = baseline[offset + column]
        return Perturbation(payload, {
            "operator": OPERATOR_ID, "unit_id": unit_id, "intensity": 1.0,
            "calendar_rule": CALENDAR_RULE, "window_days": 15,
            "climatology_fingerprint": self.climatology.fingerprint,
            "timestamps": [_iso(prepared.timestamps[time]) for time in range(unit.start, unit.stop)],
        })

    def joint_transform(
        self, instance: Instance, unit_ids: Sequence[str], donor: Instance, *, keep: bool
    ) -> dict[str, Any]:
        selected = {unit.unit_id for unit in _selected_units(unit_ids)}
        if not isinstance(keep, bool):
            raise ValueError("keep must be a boolean")
        original, replacement = _prepared(instance.payload), _prepared(donor.payload)
        if replacement.split != "validation":
            raise ValueError("Weather evaluation donors must come from validation")
        if (original.timestamps[-1].hour != replacement.timestamps[-1].hour
                or _seasonal_distance(original.timestamps[-1], replacement.timestamps[-1]) > 15):
            raise ValueError("Weather donors must match cutoff UTC slot and be within 15 seasonal days")
        replaced = set(UNIT_IDS) - selected if keep else selected
        payload = original.to_payload()
        donor_frames = [_view(frame) for frame in replacement.values]
        for unit in _UNITS:
            if unit.unit_id not in replaced:
                continue
            for time in range(unit.start, unit.stop):
                for channel in unit.channels:
                    for row in range(unit.row_start, unit.row_stop):
                        offset = (channel * 41 + row) * 61
                        for column in range(unit.column_start, unit.column_stop):
                            payload["values"][time][channel][row][column] = donor_frames[time][offset + column]
        return payload


def make_weather_task(predictor_id: str, adapter: WeatherAdapter) -> TaskSpec:
    if not isinstance(adapter, WeatherAdapter):
        raise TypeError("make_weather_task needs a WeatherAdapter")
    return TaskSpec(predictor_id, f"weather-prepared-v2:{adapter.fingerprint}", OPERATOR_ID,
                    TARGET_NAMES, OUTPUT_SCALES, UNIT_IDS)


class RegionalMeanPredictor:
    """Reduce physical-K [3,21,31] forecasts to cosine-latitude-weighted means.

    Axes: horizon, increasing latitude 15--45 N, increasing longitude 100.5--145.5 E.
    The wrapped callable receives the prepared payload.
    """

    def __init__(self, field_predictor: Callable[[Any], Any]) -> None:
        if not callable(field_predictor):
            raise TypeError("field_predictor must be callable")
        self.field_predictor = field_predictor
        self._weights = tuple(math.cos(math.radians(latitude)) for latitude in TARGET_LATITUDES)
        self._weight_sum = math.fsum(self._weights)

    def __call__(self, payload: Any) -> tuple[float, float, float]:
        fields = _sequence(self.field_predictor(payload), 3, "Forecast horizon axis")
        outputs = []
        for field in fields:
            row_means = []
            for row in _sequence(field, 21, "Forecast latitude axis"):
                values = [_number(value, "Forecast temperature") for value in
                          _sequence(row, 31, "Forecast longitude axis")]
                row_means.append(_mean(values))
            outputs.append(math.fsum(value * (weight / self._weight_sum)
                                     for value, weight in zip(row_means, self._weights)))
        if not all(math.isfinite(value) for value in outputs):
            raise ValueError("Regional forecast mean is nonfinite")
        return tuple(outputs)
