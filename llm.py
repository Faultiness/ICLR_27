"""SHIR proposal generation, grounding, and request auditing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
from importlib.resources import files
import json
import math
import os
import re
import ssl
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlsplit
from urllib.request import Request, urlopen

from .core import canonical_pair, compute_effect
from .types import Proposal, TaskSpec, canonical_json, content_hash, json_safe

__all__ = [
    "BackendError", "TransportError", "LLMConfig", "HTTPChatTransport",
    "LLMProposalBackend", "ComparisonCard", "parse_comparison_cards",
    "validate_grounding", "render_template", "load_prompt_manifest",
]


class BackendError(RuntimeError):
    """Proposal or grounding failure."""


class TransportError(BackendError):
    """Retry only transient failures."""

    def __init__(self, message: str, *, transient: bool = False, status: int | None = None):
        super().__init__(message)
        self.transient = transient
        self.status = status


class _FormatError(ValueError):
    pass


_PROTECTED_PARAMETERS = {
    "model", "messages", "temperature", "max_tokens", "max_completion_tokens",
    "response_format", "stream", "n", "api_key", "authorization", "access_token",
}


@dataclass(frozen=True)
class LLMConfig:

    endpoint: str = ""
    key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-flash-v4.1"
    timeout: float = 120.0
    proposal_vendor_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"reasoning_effort": "max"})
    grounding_vendor_parameters: Mapping[str, Any] = field(
        default_factory=lambda: {"thinking": {"type": "disabled"}})
    token_limit_field: str = "max_tokens"
    retry_backoff_seconds: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty service model identifier")
        if not isinstance(self.key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.key_env):
            raise ValueError("key_env must name an environment variable")
        if not isinstance(self.endpoint, str):
            raise ValueError("endpoint must be a string")
        if self.endpoint:
            url = urlsplit(self.endpoint)
            if url.scheme not in ("http", "https") or not url.netloc:
                raise ValueError("endpoint must be an HTTP(S) URL")
            if url.username or url.password or url.fragment:
                raise ValueError("endpoint must not contain credentials or a fragment")
            if any(key.lower() in {"key", "api_key", "token", "access_token"}
                   for key, _ in parse_qsl(url.query)):
                raise ValueError("Credentials belong in key_env, not the endpoint URL")
        for name in ("timeout", "retry_backoff_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if (name == "timeout" and value <= 0) or (name == "retry_backoff_seconds" and value < 0):
                raise ValueError(f"{name} is outside its valid range")
        if self.token_limit_field not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token_limit_field must be max_tokens or max_completion_tokens")
        for name in ("proposal_vendor_parameters", "grounding_vendor_parameters"):
            value = getattr(self, name)
            if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
                raise ValueError(f"{name} must be a JSON object with string keys")
            if set(value) & _PROTECTED_PARAMETERS:
                raise ValueError(f"{name} cannot override protocol or credential fields")
            object.__setattr__(self, name, json.loads(canonical_json(dict(value))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint, "key_env": self.key_env, "model": self.model,
            "timeout": self.timeout, "token_limit_field": self.token_limit_field,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "proposal_vendor_parameters": deepcopy(dict(self.proposal_vendor_parameters)),
            "grounding_vendor_parameters": deepcopy(dict(self.grounding_vendor_parameters)),
            "settings_scope": "submitted chat-completions request configuration",
        }


def _reject_constant(value: str) -> None:
    raise _FormatError(f"Nonfinite JSON value is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _FormatError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _FormatError("Response must be strict JSON without surrounding prose") from exc


class HTTPChatTransport:

    def __init__(self, config: LLMConfig):
        if not config.endpoint:
            raise ValueError("A complete chat-completions endpoint is required for HTTP transport")
        self.config = config

    def __call__(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        key = os.environ.get(self.config.key_env)
        if not key:
            raise TransportError(f"Credential environment variable {self.config.key_env} is unset")
        request = Request(
            self.config.endpoint, data=canonical_json(dict(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.config.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as exc:
            raise TransportError(
                f"Chat-completions service returned HTTP {exc.code}",
                transient=exc.code in {408, 409, 425, 429, 500, 502, 503, 504},
                status=exc.code,
            ) from exc
        except URLError as exc:
            permanent = isinstance(exc.reason, (ssl.SSLError, ValueError))
            raise TransportError(
                f"Chat-completions connection failed ({type(exc.reason).__name__})",
                transient=not permanent,
            ) from exc
        except (TimeoutError, ConnectionError) as exc:
            raise TransportError("Chat-completions connection timed out or was interrupted", transient=True) from exc
        except UnicodeDecodeError as exc:
            raise TransportError("Chat-completions response is not UTF-8") from exc
        try:
            parsed = _strict_json(body)
        except _FormatError as exc:
            raise TransportError("Chat-completions HTTP response is not a strict JSON object") from exc
        if not isinstance(parsed, dict):
            raise TransportError("Chat-completions HTTP response must be a JSON object")
        return parsed


def load_prompt_manifest() -> dict[str, Any]:
    return json.loads(files(__package__).joinpath("prompts", "manifest.json").read_text(encoding="utf-8"))


def _load_templates() -> tuple[dict[str, str], dict[str, Any]]:
    manifest = load_prompt_manifest()
    templates = {}
    for entry in manifest["templates"]:
        raw = files(__package__).joinpath("prompts", entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise BackendError(f"Paper prompt integrity check failed: {entry['file']}")
        templates[entry["file"]] = raw.decode("utf-8")
    if len(templates) != 10:
        raise BackendError("The ten paper prompt templates are required")
    return templates, manifest


_PLACEHOLDER = re.compile(r"\{([a-z][a-z0-9_]*)\}")


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Substitute named fields once, preserving literal braces."""
    missing = {match.group(1) for match in _PLACEHOLDER.finditer(template)} - values.keys()
    if missing:
        raise BackendError(f"Missing prompt fields: {', '.join(sorted(missing))}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def _display(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)


@dataclass(frozen=True)
class ComparisonCard:
    card_id: str
    unit_a: str
    unit_b: str
    expected_order: str
    evidence_ids: tuple[str, ...]
    rationale: str


_HEADERS = re.compile(r"(?m)^## COMPARISON ([1-9][0-9]*)[ \t]*$")
_FIELDS = re.compile(r"(?m)^(Unit A|Unit B|Expected order|Evidence references|Brief rationale):[ \t]*")
_FIELD_NAMES = ("Unit A", "Unit B", "Expected order", "Evidence references", "Brief rationale")
_EXPECTED = {"A_GT_B": 1, "B_GT_A": -1, "TIE": 0}
_MAPPED = {"A_GT_B": "first_gt_second", "B_GT_A": "second_gt_first", "TIE": "tie"}


def _fields(block: str) -> dict[str, str]:
    matches = list(_FIELDS.finditer(block))
    names = [match.group(1) for match in matches]
    if names != list(_FIELD_NAMES) or (matches and block[:matches[0].start()].strip()):
        raise _FormatError("Each card must contain exactly the five named fields in contract order")
    return {
        match.group(1): block[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(block)].strip()
        for index, match in enumerate(matches)
    }


def _references(value: str) -> tuple[str, ...]:
    parts = tuple(part.strip() for part in value.split(","))
    if not value or any(not part or "\n" in part for part in parts):
        raise _FormatError("Evidence references must be nonempty comma-separated IDs")
    return parts


def parse_comparison_cards(text: str) -> tuple[ComparisonCard, ...]:
    """Parse comparison cards before grounding unit identities."""
    if not isinstance(text, str):
        raise _FormatError("Proposal content must be text")
    text = text.strip()
    if text == "NO_VALID_COMPARISON":
        return ()
    headers = list(_HEADERS.finditer(text))
    if not headers or text[:headers[0].start()].strip():
        raise _FormatError("Return comparison cards without a preamble, or NO_VALID_COMPARISON")
    if len(headers) > 3 or [int(match.group(1)) for match in headers] != list(range(1, len(headers) + 1)):
        raise _FormatError("Return at most three cards, numbered consecutively from one")
    result = []
    for index, header in enumerate(headers):
        body = text[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)]
        values = _fields(body)
        if any(not values[name] for name in _FIELD_NAMES):
            raise _FormatError("Every comparison-card field must have a value")
        if any("\n" in values[name] for name in ("Unit A", "Unit B", "Expected order")):
            raise _FormatError("Unit IDs and expected order must each occupy one line")
        if values["Expected order"] not in _EXPECTED:
            raise _FormatError("Expected order must be A_GT_B, B_GT_A, or TIE")
        result.append(ComparisonCard(
            f"P{index + 1}", values["Unit A"], values["Unit B"], values["Expected order"],
            _references(values["Evidence references"]), values["Brief rationale"],
        ))
    return tuple(result)


def _semantic_locks(text: str) -> dict[int, dict[str, Any]]:
    """Preserve parsed fields during format repair."""
    text = _without_outer_fence(text)
    headers = list(_HEADERS.finditer(text))
    # Source order identifies malformed cards.
    blocks = [(index + 1, text[header.end():headers[index + 1].start() if index + 1 < len(headers) else len(text)])
              for index, header in enumerate(headers)]
    if not blocks and _FIELDS.search(text):
        blocks = [(1, text)]
    locks: dict[int, dict[str, Any]] = {}
    for index, block in blocks:
        matches = list(_FIELDS.finditer(block))
        fields: dict[str, Any] = {}
        for position, match in enumerate(matches):
            name = match.group(1)
            if sum(other.group(1) == name for other in matches) != 1:
                continue
            value = block[match.end():matches[position + 1].start() if position + 1 < len(matches) else len(block)].strip()
            if not value:
                continue
            if name in ("Unit A", "Unit B") and "\n" not in value:
                fields[name] = value
            elif name == "Expected order" and value in _EXPECTED:
                fields[name] = value
            elif name == "Evidence references":
                try:
                    fields[name] = _references(value)
                except _FormatError:
                    pass
            elif name == "Brief rationale":
                fields[name] = " ".join(value.split())
        locks[index] = fields
    return locks


def _without_outer_fence(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= 2 and re.fullmatch(r"```[A-Za-z0-9_-]*", lines[0]) and lines[-1] == "```":
        return "\n".join(lines[1:-1])
    return text


def _check_repair_meaning(original: str, cards: Sequence[ComparisonCard]) -> None:
    if _without_outer_fence(original).strip() == "NO_VALID_COMPARISON" and cards:
        raise _FormatError("Format repair replaced NO_VALID_COMPARISON with new hypotheses")
    locks = _semantic_locks(original)
    if locks and len(cards) != len(locks):
        raise _FormatError("Format repair added or removed an existing comparison card")
    for index, fields in locks.items():
        if index > len(cards):
            raise _FormatError("Format repair removed an existing comparison card")
        card = cards[index - 1]
        values = {"Unit A": card.unit_a, "Unit B": card.unit_b,
                  "Expected order": card.expected_order, "Evidence references": card.evidence_ids,
                  "Brief rationale": " ".join(card.rationale.split())}
        if any(values[name] != value for name, value in fields.items()):
            raise _FormatError("Format repair changed already parsed proposal meaning")


def validate_grounding(
    text: str, cards: Sequence[ComparisonCard], unit_ids: Sequence[str], allowed_evidence_ids: Sequence[str]
) -> tuple[Proposal, ...]:
    """Validate grounded cards against the original comparisons and evidence."""
    parsed = _strict_json(text)
    if not isinstance(parsed, dict) or set(parsed) != {"comparisons"} or not isinstance(parsed["comparisons"], list):
        raise _FormatError("Grounding must return exactly an object with a comparisons array")
    items = parsed["comparisons"]
    if len(items) != len(cards):
        raise _FormatError("Grounding must return one item per card, in supplied order")
    catalog, evidence = set(unit_ids), set(allowed_evidence_ids)
    fields = {"card_id", "status", "units", "expected", "rationale", "evidence_refs"}
    admitted: list[Proposal] = []
    seen = set()
    for card, item in zip(cards, items):
        if not isinstance(item, dict) or set(item) != fields or item["card_id"] != card.card_id:
            raise _FormatError("Grounding item fields, order, or Card ID do not match the supplied card")
        if not isinstance(item["rationale"], str) or " ".join(item["rationale"].split()) != " ".join(card.rationale.split()):
            raise _FormatError("Grounding must preserve the original rationale")
        valid_raw = card.unit_a in catalog and card.unit_b in catalog and card.unit_a != card.unit_b and set(card.evidence_ids) <= evidence
        pair = canonical_pair(card.unit_a, card.unit_b) if valid_raw else None
        duplicate = valid_raw and pair in seen
        if item["status"] == "untranslatable":
            if item["units"] != [] or item["expected"] is not None or item["evidence_refs"] != []:
                raise _FormatError("Untranslatable cards require empty units/references and a null expectation")
            if valid_raw and not duplicate:
                raise _FormatError("An exact valid card cannot be silently discarded by Grounding")
            continue
        if item["status"] != "grounded" or not valid_raw or duplicate:
            raise _FormatError("Unsupported or later duplicate cards must be untranslatable")
        if item["units"] != [card.unit_a, card.unit_b] or item["expected"] != _MAPPED[card.expected_order]:
            raise _FormatError("Grounding changed exact endpoints or expected order")
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise _FormatError("Grounding evidence_refs must be a list of IDs")
        if set(refs) != set(card.evidence_ids) or not set(refs) <= evidence:
            raise _FormatError("Grounding changed or invented supplied evidence references")
        seen.add(pair)
        admitted.append(Proposal(
            card.unit_a, card.unit_b, _EXPECTED[card.expected_order],
            card.rationale, card.evidence_ids, card.card_id,
        ))
    return tuple(admitted)


class LLMProposalBackend:
    """Generate and ground comparisons; transport maps request payloads to chat responses."""

    def __init__(
        self, config: LLMConfig | None = None,
        transport: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        *, sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or LLMConfig()
        self.transport = transport if transport is not None else HTTPChatTransport(self.config)
        self._sleep = sleep
        self._templates, self._manifest = _load_templates()
        self._audit: dict[str, Any] | None = None

    def _redacted(self, value: Any) -> Any:
        secret = os.environ.get(self.config.key_env)
        def visit(item: Any) -> Any:
            if isinstance(item, str):
                return item.replace(secret, "[REDACTED]") if secret else item
            if isinstance(item, Mapping):
                return {str(key): visit(value) for key, value in item.items()}
            if isinstance(item, (list, tuple)):
                return [visit(value) for value in item]
            return item
        return visit(json_safe(value))

    def consume_audit(self) -> dict[str, Any]:
        result = self._redacted(deepcopy(self._audit)) if self._audit is not None else {}
        self._audit = None
        return result

    def _request(self, stage: str, messages: list[dict[str, str]]) -> tuple[str, bool]:
        assert self._audit is not None
        proposal = stage.startswith("proposal")
        payload = {
            "model": self.config.model, "messages": deepcopy(messages),
            "temperature": 0.3 if proposal else 0,
            self.config.token_limit_field: 65536 if proposal else 4096,
            "stream": False,
        }
        payload.update(deepcopy(dict(self.config.proposal_vendor_parameters if proposal else self.config.grounding_vendor_parameters)))
        if not proposal:
            payload["response_format"] = {"type": "json_object"}
        canonical_json(payload)
        request_trace = {"stage": stage, "request": self._redacted(payload), "attempts": []}
        self._audit["requests"].append(request_trace)
        self._audit["counts"]["logical_requests"] += 1
        for attempt in range(1, 4):
            self._audit["counts"]["transport_attempts"] += 1
            event: dict[str, Any] = {"attempt": attempt}
            request_trace["attempts"].append(event)
            try:
                response = self.transport(deepcopy(payload))
                if not isinstance(response, Mapping):
                    raise TransportError("Transport must return a chat-completions response object")
                event["response"] = self._redacted(response)
                response = json.loads(canonical_json(dict(response)))
                # Count usage even for refused responses.
                usage = response.get("usage")
                if isinstance(usage, dict):
                    self._audit["counts"]["responses_with_usage"] += 1
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(key)
                        if type(value) is int and value >= 0:
                            self._audit["usage"][key] = self._audit["usage"].get(key, 0) + value
                choices = response.get("choices")
                if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                    raise TransportError("Chat-completions response must contain one choice")
                message = choices[0].get("message")
                if not isinstance(message, dict) or message.get("refusal"):
                    raise TransportError("Chat-completions response is missing an assistant message or was refused")
                content = message.get("content")
                if not isinstance(content, str):
                    raise TransportError("Chat-completions assistant content must be text")
                if choices[0].get("finish_reason") not in (None, "stop", "length"):
                    raise TransportError("Chat-completions response did not finish with a usable answer")
                event["status"] = "success"
                self._audit["counts"]["successful_responses"] += 1
                return content, choices[0].get("finish_reason") == "length"
            except Exception as exc:
                transient = isinstance(exc, TransportError) and exc.transient
                event.update(status="transient_error" if transient else "permanent_error",
                             error_type=type(exc).__name__, error=self._redacted(str(exc)))
                if isinstance(exc, TransportError) and exc.status is not None:
                    event["http_status"] = exc.status
                if not transient or attempt == 3:
                    raise BackendError(f"{stage} transport failed: {self._redacted(str(exc))}") from exc
                self._sleep(self.config.retry_backoff_seconds * (2 ** (attempt - 1)))
        raise AssertionError("Bounded transport loop did not return or raise")

    def _values(self, context: Mapping[str, Any]) -> dict[str, str]:
        canonical_json(dict(context))
        task = TaskSpec(**context["task"])
        compute_effect(context["original_prediction"], context["original_prediction"], task.scales)
        if context["stage"] not in ("reference", "explanation"):
            raise BackendError("Unknown exploration stage")
        if context["query_budget"] != 3:
            raise BackendError("The preserved paper templates require a three-query budget")
        if context["stage"] == "reference" and context.get("historical_reference") is not None:
            raise BackendError("A reference episode cannot receive cross-input model-behavior history")
        metadata = context["task_metadata"]
        required = ("dataset_description", "target_description", "primary_operator_description",
                    "unit_scope_and_coupling_description", "unit_catalog")
        if not isinstance(metadata, Mapping) or any(not metadata.get(name) for name in required):
            raise BackendError("task_metadata must provide all five task/catalog descriptions")
        stage_file = "reference_stage.txt" if context["stage"] == "reference" else "explanation_stage.txt"
        stage_instruction = self._templates[stage_file] + (
            "\nRequested explanation sizes k (number of units): "
            f"{list(context['explanation_budgets'])}.\n"
            "These are explanation sizes, not predictor-query budgets. "
            "All k values use prefixes of the same final ranking. "
            "During per-input explanation, they share one preselected batch of at most 3 "
            "perturbation queries; do not add queries per k.\n"
        )
        comparison_view = context["registered_comparisons"]
        values = {name: _display(metadata[name]) for name in required}
        values.update({
            "instance_card": _display(context["instance_card"]),
            "target_components": _display(task.target_names),
            "prediction_by_component": _display(dict(zip(task.target_names, context["original_prediction"]))),
            "scale_by_component": _display(dict(zip(task.target_names, task.scales))),
            "stage_instruction": stage_instruction,
            "round_number": str(context["round_index"]), "maximum_rounds": str(context["maximum_rounds"]),
            "historical_reference_context": _display(context.get("historical_reference")),
            "current_execution_table": _display(context["current_execution_table"]),
            "registered_comparison_view": _display(comparison_view),
            "allowed_evidence_ids": _display(context["allowed_evidence_ids"]),
        })
        return values

    def _wrapped_cards(self, cards: Sequence[ComparisonCard]) -> str:
        if not cards:
            return "NO CARDS"
        wrappers = []
        for index, card in enumerate(cards, 1):
            wrapper = self._templates["card_wrapper.txt"]
            replacements = {
                "## CARD 1": f"## CARD {index}", "Card ID: P1": f"Card ID: {card.card_id}",
                "Unit A: ...": f"Unit A: {card.unit_a}", "Unit B: ...": f"Unit B: {card.unit_b}",
                "Expected order: A_GT_B": f"Expected order: {card.expected_order}",
                "Evidence references: ...": f"Evidence references: {', '.join(card.evidence_ids)}",
                "Brief rationale: ...": f"Brief rationale: {card.rationale}",
            }
            wrapper = "\n".join(replacements.get(line, line) for line in wrapper.splitlines()) + "\n"
            wrappers.append(wrapper)
        return "\n".join(wrappers)

    def propose(
        self, context: Mapping[str, Any], *, supplement: bool = False,
        previous: Sequence[Proposal] = (),
    ) -> tuple[Proposal, ...]:
        self._audit = {
            "schema_version": "shir-llm-audit-v1", "supplement": supplement,
            "configuration": self._redacted(self.config.to_dict()),
            "prompt_source": deepcopy(self._manifest), "requests": [], "usage": {},
            "counts": {"logical_requests": 0, "transport_attempts": 0, "proposal_repairs": 0,
                       "grounding_repairs": 0, "successful_responses": 0, "responses_with_usage": 0},
            "status": "running",
        }
        try:
            values = self._values(context)
            self._audit["context_digest"] = content_hash(context)
            messages = [
                {"role": "system", "content": self._templates["proposal_system.txt"]},
                {"role": "user", "content": render_template(self._templates["proposal_user.txt"], values)},
            ]
            if supplement:
                supplement_values = {"previous_grounded_proposal": _display([p.to_dict() for p in previous])}
                messages.append({"role": "user", "content": render_template(self._templates["supplement.txt"], supplement_values)})
            content, truncated = self._request("proposal", messages)
            try:
                cards = parse_comparison_cards(content)
                if truncated:
                    raise _FormatError("Proposal response was truncated at its token limit")
            except _FormatError as error:
                self._audit["counts"]["proposal_repairs"] += 1
                repair = render_template(self._templates["proposal_repair.txt"], {
                    "validation_errors": str(error), "previous_response": content})
                repaired, truncated = self._request("proposal_repair", messages + [
                    {"role": "assistant", "content": content}, {"role": "user", "content": repair}])
                cards = parse_comparison_cards(repaired)
                if truncated:
                    raise _FormatError("Repaired Proposal response is still truncated")
                _check_repair_meaning(content, cards)
            self._audit["parsed_card_ids"] = [card.card_id for card in cards]
            grounding_values = dict(values, raw_comparison_cards_with_card_ids=self._wrapped_cards(cards))
            grounding_messages = [
                {"role": "system", "content": self._templates["grounding_system.txt"]},
                {"role": "user", "content": render_template(self._templates["grounding_user.txt"], grounding_values)},
            ]
            mapped, truncated = self._request("grounding", grounding_messages)
            try:
                proposals = validate_grounding(mapped, cards, context["task"]["unit_ids"], context["allowed_evidence_ids"])
                if truncated:
                    raise _FormatError("Grounding response was truncated at its token limit")
            except _FormatError as error:
                self._audit["counts"]["grounding_repairs"] += 1
                repair = render_template(self._templates["grounding_repair.txt"], {
                    "validation_errors": str(error), "previous_response": mapped})
                repaired, truncated = self._request("grounding_repair", grounding_messages + [
                    {"role": "assistant", "content": mapped}, {"role": "user", "content": repair}])
                proposals = validate_grounding(repaired, cards, context["task"]["unit_ids"], context["allowed_evidence_ids"])
                if truncated:
                    raise _FormatError("Repaired Grounding response is still truncated")
            self._audit["status"] = "success"
            self._audit["admitted_card_ids"] = [proposal.card_id for proposal in proposals]
            return proposals
        except Exception as exc:
            self._audit["status"] = "failed"
            self._audit["error_type"] = type(exc).__name__
            self._audit["error"] = self._redacted(str(exc))
            if isinstance(exc, BackendError):
                raise
            raise BackendError(f"Proposal/Grounding contract failed: {self._redacted(str(exc))}") from exc
