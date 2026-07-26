"""Lean-backed validation for PITA reasoning traces.

This script is phase two of the evaluation: it evaluates either gold PITA
completions or predictions generated on the cluster and stored in JSONL. It
reports:

* final-label accuracy;
* strict (outer-whitespace-trimmed) exact match against the gold completion;
* longest Lean-valid prefix fraction;
* generated-state agreement with Lean;
* executable-trace rate;
* proof validity among predicted-success and gold-true examples; and
* final-label accuracy conditional on an executable trace or valid proof.

Why several notions of validity?

PITA contains both successful proof traces and unsuccessful proof-search
traces.  Lean can certify that a successful tactic path closes the theorem,
but an unfinished search trace is not a certificate that a proposition is
false.  Accordingly, ``proof_valid`` is reserved for generated traces that
emit ``<success />`` and whose active tactic path closes the Lean goal.
``trace_executable`` is the stricter, label-agnostic property that every
emitted tactic/backtrack transition—including abandoned search branches—is
accepted by Lean and every emitted state agrees with the resulting Lean
goal. A proof may still be valid when an invalid speculative branch is later
discarded by a valid backtrack.

The completion format can contain explicit ``<backtrack to="..."/>`` events
and terminal ``<success />`` or ``<failure />`` events. All three are valid
trace tokens: terminal events are checked structurally rather than sent to
Lean, while backtracks restore the tactic-script snapshot for the referenced
generated state. For branching tactics, the evaluator also reproduces the
indentation convention used by ``task/prop_gen/util/out.py``.

Examples
--------

Pull a few gold examples from every PITA split:

    python experiment/20_lean_trace_eval.py pita \
        --tasks full imply or php --num-samples 3

Run a fast synthetic regression test of the Lean integration:

    python experiment/20_lean_trace_eval.py self-test

Evaluate the four JSONL files downloaded from the cluster. Each row must have
``prompt``, ``completion`` (gold), and ``prediction``:

    python experiment/20_lean_trace_eval.py jsonl \
        --input /path/to/pita_traces \
        --report full_trace_report.md

The REPL location is resolved in this order: ``--repl-dir``,
``LEAN_REPL_DIR``, the old repository-local location, and the cache location
used during development.  The command is ``lake exe repl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import re
import select
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_REPO = "williamtong105/pita"
DEFAULT_REPL_CACHE = (
    Path.home() / ".cache" / "lean-repl-v4.21.0-rc3"
)
OLD_LOCAL_REPL = REPO_ROOT / "task" / "prop_gen" / "util" / "repl"

TASK_CUTOFFS = {
    "full": 4,
    "imply": 6,
    "or": 18,
    "php": 60,
}

TRACE_ELEMENT_RE = re.compile(
    r"""
    <tactic>.*?</tactic>
    |<state\b[^>]*>.*?</state>
    |<backtrack\b[^>]*/>
    |<success\s*/>
    |<failure\s*/>
    """,
    re.DOTALL | re.VERBOSE,
)
TAG_START_RE = re.compile(
    r"<(?:tactic|backtrack)\b",
    re.IGNORECASE,
)
UNIVERSE_RE = re.compile(r"\bSort\s+(u(?:_[A-Za-z0-9]+|\d+)?)\b")
WHITESPACE_RE = re.compile(r"\s+")


@dataclasses.dataclass(frozen=True)
class TraceEvent:
    kind: str
    value: Any = None


@dataclasses.dataclass(frozen=True)
class ParsedTrace:
    events: tuple[TraceEvent, ...]
    label: str | None
    structurally_complete: bool
    parse_error: str | None
    estimated_steps: int


@dataclasses.dataclass(frozen=True)
class ScriptSnapshot:
    """Lean tactic lines plus the indentation for the next tactic."""

    lines: tuple[tuple[int, str], ...] = ()
    next_indent: int = 1

    def append_tactic(self, tactic: str) -> "ScriptSnapshot":
        tactic = tactic.strip()
        opens_block = tactic.endswith("=>") or bool(
            re.search(r"(?:^|\s)by$", tactic)
        )
        next_indent = self.next_indent + int(opens_block)
        return ScriptSnapshot(
            lines=(*self.lines, (self.next_indent, tactic)),
            next_indent=next_indent,
        )

    def dedent(self) -> "ScriptSnapshot":
        return ScriptSnapshot(
            lines=self.lines,
            next_indent=max(1, self.next_indent - 1),
        )


@dataclasses.dataclass(frozen=True)
class PrefixCheck:
    accepted: bool
    complete_at_current_scope: bool
    goal: str | None
    response: dict[str, Any]
    error_messages: tuple[str, ...]


@dataclasses.dataclass
class TraceResult:
    task: str | None
    index: int
    gold_label: str | None
    predicted_label: str | None
    label_correct: bool
    exact_match: bool
    parse_valid: bool
    total_steps: int
    valid_prefix_steps: int
    prefix_validity: float
    total_states: int
    matching_states: int
    state_match_rate: float
    all_tactics_accepted: bool
    all_states_match: bool
    all_backtracks_valid: bool
    trace_executable: bool
    proof_complete: bool
    proof_valid: bool
    error_kind: str | None
    error_detail: str | None


class LeanReplError(RuntimeError):
    pass


class LeanRepl:
    """Persistent JSON client for leanprover-community/repl."""

    def __init__(
        self,
        repl_dir: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.repl_dir = repl_dir
        self.timeout_seconds = timeout_seconds
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout_buffer = b""

    def __enter__(self) -> "LeanRepl":
        if not self.repl_dir.is_dir():
            raise LeanReplError(f"Lean REPL directory not found: {self.repl_dir}")
        self.process = subprocess.Popen(
            ["lake", "exe", "repl"],
            cwd=self.repl_dir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise LeanReplError("Lean REPL is not running")
        if self.process.poll() is not None:
            stderr = ""
            if self.process.stderr is not None:
                stderr = self.process.stderr.read().decode(errors="replace")
            raise LeanReplError(
                f"Lean REPL exited with {self.process.returncode}: {stderr}"
            )

        encoded = (json.dumps(payload) + "\n\n").encode()
        self.process.stdin.write(encoded)
        self.process.stdin.flush()

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            if b"\n\n" in self.stdout_buffer:
                raw_bytes, self.stdout_buffer = self.stdout_buffer.split(
                    b"\n\n",
                    1,
                )
                raw = raw_bytes.decode()
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LeanReplError(
                    f"Lean REPL timed out after {self.timeout_seconds}s"
                )
            readable, _, _ = select.select(
                [self.process.stdout],
                [],
                [],
                remaining,
            )
            if not readable:
                raise LeanReplError(
                    f"Lean REPL timed out after {self.timeout_seconds}s"
                )
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                raise LeanReplError("Lean REPL closed stdout unexpectedly")
            self.stdout_buffer += chunk

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LeanReplError(
                f"Lean REPL returned invalid JSON: {raw!r}"
            ) from exc


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text.strip())


def normalize_goal_shape(text: str) -> str:
    """Ignore whitespace and redundant pretty-printer parentheses."""

    # Lean prints a non-dependent arrow from an arbitrary Sort as a forall
    # binder.  PITA's source trace prints the same proposition with ``→``.
    text = re.sub(
        r"∀\s*\([^:()]+:\s*([^()]+)\)\s*,",
        r"\1 →",
        text,
    )
    return re.sub(r"[\s()]", "", text)


def parse_xml_element(fragment: str) -> ET.Element:
    return ET.fromstring(fragment)


def parse_trace(text: str) -> ParsedTrace:
    """Parse the longest contiguous sequence of known PITA XML elements."""

    events: list[TraceEvent] = []
    label: str | None = None
    position = 0
    parse_error: str | None = None

    for match in TRACE_ELEMENT_RE.finditer(text):
        gap = text[position : match.start()]
        if gap.strip(" \t\r\n|"):
            parse_error = (
                f"unexpected text at character {position}: "
                f"{gap[:80]!r}"
            )
            break

        fragment = match.group(0)
        try:
            element = parse_xml_element(fragment)
        except ET.ParseError as exc:
            parse_error = f"malformed XML at character {match.start()}: {exc}"
            break

        if element.tag == "tactic":
            events.append(TraceEvent("tactic", element.text or ""))
        elif element.tag == "state":
            events.append(TraceEvent("state", element))
        elif element.tag == "backtrack":
            target = element.get("to")
            try:
                events.append(TraceEvent("backtrack", int(str(target))))
            except (TypeError, ValueError):
                parse_error = f"invalid backtrack target: {target!r}"
                break
        elif element.tag in {"success", "failure"}:
            current_label = element.tag
            events.append(TraceEvent("label", current_label))
            if label is None:
                label = current_label
            else:
                parse_error = "multiple final-label elements"
                break
        position = match.end()

    if parse_error is None:
        remainder = text[position:]
        if remainder.strip(" \t\r\n|"):
            parse_error = (
                f"unparsed suffix at character {position}: "
                f"{remainder[:80]!r}"
            )

    estimated_steps = len(TAG_START_RE.findall(text))
    structurally_complete = (
        parse_error is None
        and label is not None
        and bool(events)
        and events[-1].kind == "label"
    )
    return ParsedTrace(
        events=tuple(events),
        label=label,
        structurally_complete=structurally_complete,
        parse_error=parse_error,
        estimated_steps=estimated_steps,
    )


def parse_initial_state(prompt: str) -> ET.Element:
    match = re.search(r"<state\b[^>]*>.*?</state>", prompt, re.DOTALL)
    if match is None:
        raise ValueError("prompt does not contain an initial <state>")
    try:
        return parse_xml_element(match.group(0))
    except ET.ParseError as exc:
        raise ValueError(f"malformed initial state: {exc}") from exc


def state_goal_text(state: ET.Element) -> str | None:
    if state.find("complete") is not None:
        return None
    then = state.find("then")
    if then is None or then.text is None:
        raise ValueError("state lacks <then> goal")
    goal = then.text.strip()
    if goal.startswith("⊢"):
        goal = goal[1:].strip()
    return normalize_space(goal)


def state_context_lines(state: ET.Element) -> list[str]:
    raw_lines = []
    for child in state.findall("if"):
        line = normalize_space(child.text or "")
        if not line:
            continue
        raw_lines.append(line)

    lines: list[str] = []
    for line in raw_lines:
        # Lean's pretty-printer wraps very large declarations after the
        # colon; PITA serializes each wrapped line as a separate <if>.
        if lines and lines[-1].endswith(":"):
            lines[-1] = normalize_space(lines[-1] + " " + line)
        else:
            lines.append(line)

    # Some cleaned Or/PHP traces removed "p1 p2 p3" but retained the
    # dangling ": Prop".  It carries no checkable information.
    return [line for line in lines if not line.startswith(":")]


def prompt_to_lean_header(prompt: str) -> tuple[str, ET.Element]:
    state = parse_initial_state(prompt)
    goal = state_goal_text(state)
    if goal is None:
        raise ValueError("initial PITA state is already complete")

    context = state_context_lines(state)
    universes = sorted(
        {
            universe
            for line in context
            for universe in UNIVERSE_RE.findall(line)
        }
    )
    lines = [*(f"universe {name}" for name in universes)]
    lines.extend(f"variable ({line})" for line in context)
    lines.append(f"example : {goal} := by")
    return "\n".join(lines), state


def lean_goal_parts(goal: str) -> tuple[list[str], str]:
    lines = [normalize_space(line) for line in goal.splitlines()]
    lines = [line for line in lines if line]
    turnstile_index = next(
        (index for index, line in enumerate(lines) if "⊢" in line),
        None,
    )
    if turnstile_index is None:
        raise ValueError(f"Lean goal has no turnstile: {goal!r}")
    before, after = lines[turnstile_index].split("⊢", 1)
    context = lines[:turnstile_index]
    if before.strip():
        context.append(normalize_space(before))
    coalesced_context: list[str] = []
    for line in context:
        if coalesced_context and coalesced_context[-1].endswith(":"):
            coalesced_context[-1] = normalize_space(
                coalesced_context[-1] + " " + line
            )
        else:
            coalesced_context.append(line)
    conclusion = normalize_space(
        " ".join([after, *lines[turnstile_index + 1 :]])
    )
    return coalesced_context, conclusion


def state_matches_goal(
    state: ET.Element,
    lean_goal: str | None,
    strict_context: bool,
) -> bool:
    state_complete = state.find("complete") is not None
    if state_complete:
        return lean_goal is None
    if lean_goal is None:
        return False

    expected_goal = state_goal_text(state)
    lean_context, lean_conclusion = lean_goal_parts(lean_goal)
    if strict_context:
        goals_match = expected_goal == lean_conclusion
    else:
        goals_match = normalize_goal_shape(
            expected_goal
        ) == normalize_goal_shape(lean_conclusion)
    if not goals_match:
        return False

    expected_context = state_context_lines(state)
    if not strict_context:
        # Context pretty-printing changed between the Lean versions used to
        # generate PITA and current Lean.  In addition, the cleaned Or/PHP
        # traces intentionally omit p1/p2/p3 from completion states.  The
        # stable default therefore validates the goal conclusion.  Use
        # --strict-context with the dataset's original Lean version to audit
        # every emitted hypothesis as well.
        return True
    return expected_context == lean_context


class LeanTraceVerifier:
    def __init__(
        self,
        repl: LeanRepl,
        strict_context: bool = False,
    ) -> None:
        self.repl = repl
        self.strict_context = strict_context
        self.prefix_cache: dict[tuple[str, ScriptSnapshot], PrefixCheck] = {}
        self.completion_cache: dict[tuple[str, ScriptSnapshot], bool] = {}

    @staticmethod
    def render_script(
        header: str,
        snapshot: ScriptSnapshot,
        add_placeholder: bool,
    ) -> str:
        lines = [header]
        lines.extend(
            f"{'  ' * indent}{tactic}"
            for indent, tactic in snapshot.lines
        )
        if add_placeholder:
            lines.append(f"{'  ' * snapshot.next_indent}sorry")
        return "\n".join(lines)

    @staticmethod
    def response_errors(response: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            str(message.get("data", ""))
            for message in response.get("messages", [])
            if message.get("severity") == "error"
        )

    @staticmethod
    def is_placeholder_only_error(message: str) -> bool:
        normalized = message.strip().lower()
        return normalized.startswith("unsolved goals") or normalized.startswith(
            "no goals to be solved"
        )

    def check_prefix(
        self,
        header: str,
        snapshot: ScriptSnapshot,
    ) -> PrefixCheck:
        key = (header, snapshot)
        if key in self.prefix_cache:
            return self.prefix_cache[key]

        command = self.render_script(header, snapshot, add_placeholder=True)
        response = self.repl.request({"cmd": command})
        errors = self.response_errors(response)
        substantive_errors = tuple(
            error
            for error in errors
            if not self.is_placeholder_only_error(error)
        )
        sorries = response.get("sorries", [])
        goal = str(sorries[-1]["goal"]) if sorries else None
        complete_scope = not sorries and any(
            error.lower().startswith("no goals to be solved")
            for error in errors
        )
        accepted = not substantive_errors and (
            bool(sorries) or complete_scope
        )
        result = PrefixCheck(
            accepted=accepted,
            complete_at_current_scope=complete_scope,
            goal=goal,
            response=response,
            error_messages=errors,
        )
        self.prefix_cache[key] = result
        return result

    def proof_complete(
        self,
        header: str,
        snapshot: ScriptSnapshot,
    ) -> bool:
        key = (header, snapshot)
        if key in self.completion_cache:
            return self.completion_cache[key]
        command = self.render_script(header, snapshot, add_placeholder=False)
        response = self.repl.request({"cmd": command})
        complete = (
            not self.response_errors(response)
            and not response.get("sorries")
            and "message" not in response
        )
        self.completion_cache[key] = complete
        return complete

    def evaluate(
        self,
        prompt: str,
        prediction: str,
        gold_completion: str,
        task: str | None,
        index: int,
    ) -> TraceResult:
        parsed = parse_trace(prediction)
        gold_parsed = parse_trace(gold_completion)
        gold_label = gold_parsed.label
        predicted_label = parsed.label
        exact_match = prediction.strip() == gold_completion.strip()
        label_correct = (
            predicted_label is not None and predicted_label == gold_label
        )

        try:
            header, initial_state = prompt_to_lean_header(prompt)
        except ValueError as exc:
            return TraceResult(
                task=task,
                index=index,
                gold_label=gold_label,
                predicted_label=predicted_label,
                label_correct=label_correct,
                exact_match=exact_match,
                parse_valid=False,
                total_steps=max(1, parsed.estimated_steps),
                valid_prefix_steps=0,
                prefix_validity=0.0,
                total_states=0,
                matching_states=0,
                state_match_rate=math.nan,
                all_tactics_accepted=False,
                all_states_match=False,
                all_backtracks_valid=False,
                trace_executable=False,
                proof_complete=False,
                proof_valid=False,
                error_kind="prompt",
                error_detail=str(exc),
            )

        snapshots: dict[int, ScriptSnapshot] = {0: ScriptSnapshot()}
        current = ScriptSnapshot()
        initial_check = self.check_prefix(header, current)
        initial_matches = state_matches_goal(
            initial_state,
            initial_check.goal,
            self.strict_context,
        )

        total_states = 1
        matching_states = int(initial_matches)
        all_tactics_accepted = True
        all_states_match = initial_matches
        all_backtracks_valid = True
        valid_prefix_steps = 0
        prefix_open = initial_check.accepted and initial_matches
        error_kind: str | None = None
        error_detail: str | None = parsed.parse_error
        last_check = initial_check

        events = list(parsed.events)
        event_index = 0
        while event_index < len(events):
            event = events[event_index]
            if event.kind == "label":
                event_index += 1
                continue

            if event.kind == "tactic":
                tactic = str(event.value).strip()
                # A model should not be able to obtain validity by admitting
                # the theorem.
                forbidden = re.search(
                    r"\b(?:sorry|admit)\b",
                    tactic,
                )
                candidate = current.append_tactic(tactic)
                if forbidden:
                    check = PrefixCheck(
                        accepted=False,
                        complete_at_current_scope=False,
                        goal=None,
                        response={},
                        error_messages=("admission tactic is forbidden",),
                    )
                else:
                    check = self.check_prefix(header, candidate)

                state: ET.Element | None = None
                if (
                    event_index + 1 < len(events)
                    and events[event_index + 1].kind == "state"
                ):
                    state = events[event_index + 1].value
                    event_index += 1

                state_matches = True
                if state is not None:
                    total_states += 1
                    state_matches = state_matches_goal(
                        state,
                        check.goal,
                        self.strict_context,
                    )
                    matching_states += int(state_matches)
                    state_id_text = state.get("id")
                    try:
                        state_id = int(str(state_id_text))
                    except (TypeError, ValueError):
                        state_matches = False
                        state_id = None
                    if state_id is not None:
                        snapshots[state_id] = candidate

                step_valid = check.accepted and state_matches
                all_tactics_accepted &= check.accepted
                all_states_match &= state_matches
                if prefix_open and step_valid:
                    valid_prefix_steps += 1
                elif prefix_open:
                    prefix_open = False
                    error_kind = (
                        "state_mismatch" if check.accepted else "lean_tactic"
                    )
                    error_detail = (
                        "generated state does not match Lean"
                        if check.accepted
                        else "; ".join(check.error_messages)
                    )
                current = candidate
                last_check = check

            elif event.kind == "backtrack":
                target = int(event.value)
                backtrack_valid = target in snapshots
                all_backtracks_valid &= backtrack_valid
                if backtrack_valid:
                    current = snapshots[target]
                    check = self.check_prefix(header, current)
                    last_check = check
                else:
                    check = PrefixCheck(
                        accepted=False,
                        complete_at_current_scope=False,
                        goal=None,
                        response={},
                        error_messages=(
                            f"unknown backtrack state {target}",
                        ),
                    )

                state_matches = True
                if (
                    event_index + 1 < len(events)
                    and events[event_index + 1].kind == "state"
                ):
                    state = events[event_index + 1].value
                    event_index += 1
                    total_states += 1
                    state_matches = (
                        backtrack_valid
                        and state_matches_goal(
                            state,
                            check.goal,
                            self.strict_context,
                        )
                    )
                    matching_states += int(state_matches)

                step_valid = (
                    backtrack_valid and check.accepted and state_matches
                )
                all_states_match &= state_matches
                if prefix_open and step_valid:
                    valid_prefix_steps += 1
                elif prefix_open:
                    prefix_open = False
                    error_kind = (
                        "backtrack"
                        if not backtrack_valid
                        else "state_mismatch"
                    )
                    error_detail = (
                        f"unknown backtrack state {target}"
                        if not backtrack_valid
                        else "backtracked state does not match Lean"
                    )

            elif event.kind == "state":
                # A state-only transition is how the generator records
                # dedenting from a completed branch before the sibling case.
                state = event.value
                candidate = current.dedent()
                check = self.check_prefix(header, candidate)
                total_states += 1
                state_matches = state_matches_goal(
                    state,
                    check.goal,
                    self.strict_context,
                )
                matching_states += int(state_matches)
                all_states_match &= state_matches
                state_id_text = state.get("id")
                try:
                    snapshots[int(str(state_id_text))] = candidate
                except (TypeError, ValueError):
                    all_states_match = False
                current = candidate
                last_check = check

            event_index += 1

        total_steps = max(parsed.estimated_steps, 0)
        if total_steps == 0:
            prefix_validity = 0.0
        else:
            prefix_validity = min(1.0, valid_prefix_steps / total_steps)
        state_match_rate = (
            matching_states / total_states if total_states else math.nan
        )

        parse_valid = parsed.structurally_complete
        trace_executable = (
            parse_valid
            and all_tactics_accepted
            and all_states_match
            and all_backtracks_valid
            and valid_prefix_steps == total_steps
        )
        try:
            proof_complete = self.proof_complete(header, current)
        except LeanReplError as exc:
            proof_complete = False
            if error_kind is None:
                error_kind = "repl"
                error_detail = str(exc)
        proof_valid = (
            predicted_label == "success"
            and parse_valid
            and all_backtracks_valid
            and proof_complete
        )

        if error_kind is None and parsed.parse_error is not None:
            error_kind = "parse"
        if error_kind is None and not parsed.structurally_complete:
            error_kind = "structure"
            error_detail = error_detail or "missing or misplaced final label"
        if error_kind is None and not proof_complete and predicted_label == "success":
            error_kind = "incomplete_proof"
            error_detail = "success label emitted before Lean goal was closed"

        return TraceResult(
            task=task,
            index=index,
            gold_label=gold_label,
            predicted_label=predicted_label,
            label_correct=label_correct,
            exact_match=exact_match,
            parse_valid=parse_valid,
            total_steps=total_steps,
            valid_prefix_steps=valid_prefix_steps,
            prefix_validity=prefix_validity,
            total_states=total_states,
            matching_states=matching_states,
            state_match_rate=state_match_rate,
            all_tactics_accepted=all_tactics_accepted,
            all_states_match=all_states_match,
            all_backtracks_valid=all_backtracks_valid,
            trace_executable=trace_executable,
            proof_complete=proof_complete,
            proof_valid=proof_valid,
            error_kind=error_kind,
            error_detail=error_detail,
        )


def resolve_repl_dir(argument: Path | None) -> Path:
    candidates = [
        argument,
        Path(os.environ["LEAN_REPL_DIR"])
        if os.environ.get("LEAN_REPL_DIR")
        else None,
        OLD_LOCAL_REPL,
        DEFAULT_REPL_CACHE,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_dir():
            return candidate.expanduser().resolve()
    rendered = ", ".join(str(path) for path in candidates if path is not None)
    raise SystemExit(
        "Lean REPL checkout not found. Pass --repl-dir or set "
        f"LEAN_REPL_DIR. Checked: {rendered}"
    )


def fetch_hub_rows(
    dataset_repo: str,
    tasks: Sequence[str],
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Fetch deterministic random, untruncated rows through Dataset Viewer."""

    rng = random.Random(seed)
    output: list[dict[str, Any]] = []
    base_url = "https://datasets-server.huggingface.co/rows"

    for task in tasks:
        first_url = base_url + "?" + urllib.parse.urlencode(
            {
                "dataset": dataset_repo,
                "config": "default",
                "split": task,
                "offset": 0,
                "length": 1,
            }
        )
        with urllib.request.urlopen(first_url, timeout=60) as response:
            first_payload = json.load(response)
        total = int(first_payload["num_rows_total"])

        chosen: set[int] = set()
        attempts = 0
        while len(chosen) < num_samples:
            if attempts >= max(100, 20 * num_samples):
                raise RuntimeError(
                    f"Could not fetch {num_samples} untruncated {task} rows"
                )
            attempts += 1
            offset = rng.randrange(total)
            if offset in chosen:
                continue
            row_url = base_url + "?" + urllib.parse.urlencode(
                {
                    "dataset": dataset_repo,
                    "config": "default",
                    "split": task,
                    "offset": offset,
                    "length": 1,
                }
            )
            with urllib.request.urlopen(row_url, timeout=60) as response:
                payload = json.load(response)
            item = payload["rows"][0]
            if item.get("truncated_cells"):
                continue
            row = dict(item["row"])
            row["task"] = task
            row["dataset_index"] = offset
            row["prediction"] = row["completion"]
            output.append(row)
            chosen.add(offset)
    return output


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on {path}:{line_number}: {exc}"
                ) from exc
            rows.append(row)
    return rows


def read_jsonl_inputs(inputs: Sequence[Path]) -> list[dict[str, Any]]:
    """Read explicit JSONL files or phase-one trace directories."""

    paths: list[Path] = []
    for input_path in inputs:
        input_path = input_path.expanduser()
        if input_path.is_dir():
            matches = sorted(input_path.glob("traces.*.jsonl"))
            if not matches:
                matches = sorted(input_path.glob("*.jsonl"))
            if not matches:
                raise RuntimeError(f"No JSONL files found in {input_path}")
            paths.extend(matches)
        elif input_path.is_file():
            paths.append(input_path)
        else:
            raise RuntimeError(f"JSONL input not found: {input_path}")

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(resolved)
            seen.add(resolved)

    rows: list[dict[str, Any]] = []
    for path in unique_paths:
        rows.extend(read_jsonl(path))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def require_row_columns(
    rows: Sequence[dict[str, Any]],
    prediction_column: str,
) -> None:
    required = {"prompt", "completion", prediction_column}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise RuntimeError(
                f"Input row {index} lacks columns: "
                + ", ".join(sorted(missing))
            )


def build_self_test_rows() -> list[dict[str, Any]]:
    prompt = (
        '<state id="0"><if>p : Prop</if>'
        "<then>⊢ p → p</then></state>||"
    )
    completion = (
        "<tactic>intro h</tactic>"
        '<state id="1"><if>p : Prop</if><if>h : p</if>'
        "<then>⊢ p</then></state>"
        "<tactic>exact h</tactic>"
        '<state id="2"><complete /></state>'
        "<success />"
    )
    invalid_tactic = completion.replace(
        "<tactic>intro h</tactic>",
        "<tactic>exact missing_hypothesis</tactic>",
        1,
    )
    fabricated_state = completion.replace(
        "<then>⊢ p</then>",
        "<then>⊢ False</then>",
        1,
    )
    truncated = completion.removesuffix("<success />")
    backtracked = (
        "<tactic>intro h</tactic>"
        '<state id="1"><if>p : Prop</if><if>h : p</if>'
        "<then>⊢ p</then></state>"
        "<tactic>exact h</tactic>"
        '<state id="2"><complete /></state>'
        '<backtrack to="1"/>'
        '<state id="3"><if>p : Prop</if><if>h : p</if>'
        "<then>⊢ p</then></state>"
        "<tactic>exact h</tactic>"
        '<state id="4"><complete /></state>'
        "<success />"
    )
    executable_failure = (
        "<tactic>intro h</tactic>"
        '<state id="1"><if>p : Prop</if><if>h : p</if>'
        "<then>⊢ p</then></state>"
        "<failure />"
    )

    variants = [
        ("gold", completion),
        ("backtracked", backtracked),
        ("executable_failure", executable_failure),
        ("invalid_tactic", invalid_tactic),
        ("fabricated_state", fabricated_state),
        ("truncated", truncated),
    ]
    return [
        {
            "task": "self-test",
            "dataset_index": index,
            "variant": name,
            "prompt": prompt,
            "completion": completion,
            "prediction": prediction,
        }
        for index, (name, prediction) in enumerate(variants)
    ]


def assert_self_test(results: Sequence[TraceResult]) -> None:
    if len(results) != 6:
        raise RuntimeError("self-test produced the wrong number of results")
    (
        gold,
        backtracked,
        executable_failure,
        invalid_tactic,
        fabricated_state,
        truncated,
    ) = results
    checks = [
        (
            gold.exact_match
            and gold.trace_executable
            and gold.proof_valid
            and math.isclose(gold.prefix_validity, 1.0),
            "gold proof was not fully validated",
        ),
        (
            backtracked.parse_valid
            and backtracked.trace_executable
            and backtracked.proof_valid
            and backtracked.all_backtracks_valid,
            "valid backtrack token was not honored",
        ),
        (
            executable_failure.parse_valid
            and executable_failure.trace_executable
            and executable_failure.predicted_label == "failure"
            and not executable_failure.proof_valid,
            "failure token was not accepted as an executable non-proof trace",
        ),
        (
            not invalid_tactic.proof_valid
            and invalid_tactic.prefix_validity < 1.0
            and not invalid_tactic.all_tactics_accepted,
            "invalid tactic was not rejected",
        ),
        (
            fabricated_state.proof_valid
            and not fabricated_state.trace_executable
            and fabricated_state.prefix_validity < 1.0
            and not fabricated_state.all_states_match,
            "fabricated state was not separated from active-proof validity",
        ),
        (
            not truncated.parse_valid and not truncated.trace_executable,
            "truncated XML trace was not rejected",
        ),
    ]
    failures = [message for passed, message in checks if not passed]
    if failures:
        raise RuntimeError("Lean trace self-test failed: " + "; ".join(failures))
    print("Lean trace self-test passed.", file=sys.stderr)


def mean_bool(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def mean_float(values: Iterable[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    return sum(clean) / len(clean) if clean else math.nan


def summarize_group(results: Sequence[TraceResult]) -> dict[str, Any]:
    predicted_success = [
        result for result in results if result.predicted_label == "success"
    ]
    gold_true = [
        result for result in results if result.gold_label == "success"
    ]
    gold_false = [
        result for result in results if result.gold_label == "failure"
    ]
    no_final_label = [
        result for result in results if result.predicted_label is None
    ]
    executable = [result for result in results if result.trace_executable]
    valid_proofs = [result for result in results if result.proof_valid]
    state_total = sum(result.total_states for result in results)
    state_matches = sum(result.matching_states for result in results)
    success_recall = mean_bool(
        result.predicted_label == "success" for result in gold_true
    )
    failure_recall = mean_bool(
        result.predicted_label == "failure" for result in gold_false
    )

    return {
        "n": len(results),
        "gold_success_rate": mean_bool(
            result.gold_label == "success" for result in results
        ),
        "predicted_success_rate": mean_bool(
            result.predicted_label == "success" for result in results
        ),
        "success_recall": success_recall,
        "failure_recall": failure_recall,
        "balanced_label_accuracy": mean_float(
            [success_recall, failure_recall]
        ),
        "missing_label_rate": mean_bool(
            result.predicted_label is None for result in results
        ),
        "label_accuracy": mean_bool(
            result.label_correct for result in results
        ),
        "exact_match_accuracy": mean_bool(
            result.exact_match for result in results
        ),
        "mean_prefix_validity": mean_float(
            result.prefix_validity for result in results
        ),
        "fully_valid_prefix_rate": mean_bool(
            math.isclose(result.prefix_validity, 1.0)
            for result in results
        ),
        "parse_valid_rate": mean_bool(
            result.parse_valid for result in results
        ),
        "state_match_rate": (
            state_matches / state_total if state_total else math.nan
        ),
        "trace_executable_rate": mean_bool(
            result.trace_executable for result in results
        ),
        "proof_valid_rate": mean_bool(
            result.proof_valid for result in results
        ),
        "proof_validity_given_predicted_success": mean_bool(
            result.proof_valid for result in predicted_success
        ),
        "proof_completion_given_gold_true": mean_bool(
            (
                result.parse_valid
                and result.all_backtracks_valid
                and result.proof_complete
            )
            for result in gold_true
        ),
        "label_accuracy_given_executable_trace": mean_bool(
            result.label_correct for result in executable
        ),
        "label_accuracy_given_valid_proof": mean_bool(
            result.label_correct for result in valid_proofs
        ),
        "n_predicted_success": len(predicted_success),
        "n_gold_true": len(gold_true),
        "n_gold_false": len(gold_false),
        "n_no_final_label": len(no_final_label),
        "n_executable": len(executable),
        "n_valid_proofs": len(valid_proofs),
    }


def summarize(results: Sequence[TraceResult]) -> dict[str, Any]:
    grouped: dict[str, list[TraceResult]] = defaultdict(list)
    for result in results:
        grouped[result.task or "unknown"].append(result)
    by_task = {
        task: summarize_group(task_results)
        for task, task_results in sorted(grouped.items())
    }
    return {
        "overall": summarize_group(results),
        "by_task": by_task,
    }


def percentage(value: float) -> str:
    if value is None or math.isnan(value):
        return "—"
    return f"{100 * value:.1f}%"


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(escape(item) for item in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(escape(item) for item in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def render_markdown(summary: dict[str, Any]) -> str:
    trace_rows = []
    label_rows = []
    groups = [("Overall", summary["overall"])]
    groups.extend(
        (task.capitalize(), values)
        for task, values in summary["by_task"].items()
    )
    for label, values in groups:
        trace_rows.append(
            [
                label,
                values["n"],
                percentage(values["label_accuracy"]),
                percentage(values["exact_match_accuracy"]),
                percentage(values["mean_prefix_validity"]),
                percentage(values["state_match_rate"]),
                percentage(values["parse_valid_rate"]),
                percentage(values["trace_executable_rate"]),
                percentage(values["proof_valid_rate"]),
                percentage(
                    values["proof_validity_given_predicted_success"]
                ),
                percentage(values["proof_completion_given_gold_true"]),
                percentage(
                    values["label_accuracy_given_executable_trace"]
                ),
                percentage(values["label_accuracy_given_valid_proof"]),
            ]
        )
        label_rows.append(
            [
                label,
                percentage(values["gold_success_rate"]),
                percentage(values["predicted_success_rate"]),
                percentage(values["success_recall"]),
                percentage(values["failure_recall"]),
                percentage(values["balanced_label_accuracy"]),
                percentage(values["missing_label_rate"]),
            ]
        )

    return "\n".join(
        [
            "# PITA Lean trace-validation report",
            "",
            (
                "Proof validity is reported among predicted-success traces. "
                "A failure trace can be Lean-executable, but incomplete proof "
                "search is not a Lean certificate of falsehood."
            ),
            "",
            markdown_table(
                [
                    "Split",
                    "n",
                    "Label acc.",
                    "Exact match",
                    "Mean valid prefix",
                    "State match",
                    "Complete syntax",
                    "Executable trace",
                    "Valid proof",
                    "Valid proof / predicted success",
                    "Complete proof / gold true",
                    "Label acc. / executable",
                    "Label acc. / valid proof",
                ],
                trace_rows,
            ),
            "",
            "Label-prior diagnostics:",
            "",
            markdown_table(
                [
                    "Split",
                    "Gold success",
                    "Predicted success",
                    "Success recall",
                    "Failure recall",
                    "Balanced acc.",
                    "Missing label",
                ],
                label_rows,
            ),
            "",
            "Definitions:",
            "",
            (
                "- **Mean valid prefix**: mean fraction of tactic/backtrack "
                "steps accepted from the start, stopping at the first invalid "
                "tactic, backtrack, or mismatched emitted state."
            ),
            (
                "- **Executable trace**: well-formed XML with a final label, "
                "all tactics accepted, all backtracks resolvable, and emitted "
                "states consistent with Lean."
            ),
            (
                "- **Valid proof**: a structurally valid predicted-success "
                "trace with resolvable backtracks whose final active tactic "
                "script closes the theorem without `sorry`/`admit`. Invalid "
                "abandoned branches still reduce prefix/executable-trace "
                "metrics, but do not invalidate the final proof."
            ),
            "",
        ]
    )


def evaluate_rows(
    rows: Sequence[dict[str, Any]],
    prediction_column: str,
    repl_dir: Path,
    repl_timeout: float,
    strict_context: bool,
    repl_restart_every: int = 10,
) -> list[TraceResult]:
    require_row_columns(rows, prediction_column)
    results: list[TraceResult] = []
    repl: LeanRepl | None = None
    verifier: LeanTraceVerifier | None = None
    examples_since_restart = repl_restart_every

    def restart_repl() -> None:
        nonlocal repl, verifier, examples_since_restart
        if repl is not None:
            repl.__exit__(None, None, None)
        repl = LeanRepl(repl_dir, repl_timeout)
        repl.__enter__()
        verifier = LeanTraceVerifier(repl, strict_context=strict_context)
        examples_since_restart = 0

    try:
        for index, row in enumerate(rows):
            if (
                verifier is None
                or (
                    repl_restart_every > 0
                    and examples_since_restart >= repl_restart_every
                )
            ):
                restart_repl()

            for attempt in range(2):
                assert verifier is not None
                try:
                    result = verifier.evaluate(
                        prompt=str(row["prompt"]),
                        prediction=str(row[prediction_column]),
                        gold_completion=str(row["completion"]),
                        task=(
                            str(row["task"])
                            if row.get("task")
                            else None
                        ),
                        index=int(row.get("dataset_index", index)),
                    )
                    break
                except LeanReplError:
                    if attempt:
                        raise
                    print(
                        f"Lean REPL exited while evaluating row {index}; "
                        "restarting and retrying once.",
                        file=sys.stderr,
                        flush=True,
                    )
                    restart_repl()
            examples_since_restart += 1
            results.append(result)
            print(
                f"[{index + 1}/{len(rows)}] "
                f"task={result.task or 'unknown'} "
                f"prefix={result.prefix_validity:.3f} "
                f"executable={result.trace_executable} "
                f"proof_valid={result.proof_valid}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        if repl is not None:
            repl.__exit__(None, None, None)
    return results


def add_common_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repl-dir",
        type=Path,
        help="leanprover-community/repl checkout (or set LEAN_REPL_DIR).",
    )
    parser.add_argument(
        "--repl-timeout",
        type=float,
        default=30.0,
        help="Seconds allowed for each REPL request.",
    )
    parser.add_argument(
        "--repl-restart-every",
        type=int,
        default=10,
        help=(
            "Restart the Lean REPL after this many examples to bound its "
            "memory use; use 0 to keep one process."
        ),
    )
    parser.add_argument(
        "--strict-context",
        action="store_true",
        help="Require exact context lines in addition to the goal conclusion.",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
        help="Input field containing the generated completion.",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        help="Optional per-example metric JSONL.",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        help="Optional aggregate metric JSON.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional aggregate Markdown report.",
    )


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Lean-backed validation of PITA reasoning traces."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    pita = subparsers.add_parser(
        "pita",
        help="Pull gold PITA examples from the Hub and validate the harness.",
    )
    pita.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
    )
    pita.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_CUTOFFS),
        default=sorted(TASK_CUTOFFS),
    )
    pita.add_argument("--num-samples", type=int, default=2)
    pita.add_argument("--seed", type=int, default=0)
    add_common_evaluation_arguments(pita)

    jsonl = subparsers.add_parser(
        "jsonl",
        help="Evaluate prompts, gold completions, and predictions in JSONL.",
    )
    jsonl.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "One or more trace JSONL files, or a directory containing "
            "traces.<task>.jsonl files."
        ),
    )
    add_common_evaluation_arguments(jsonl)

    self_test = subparsers.add_parser(
        "self-test",
        help="Run synthetic gold/corruption checks against Lean.",
    )
    add_common_evaluation_arguments(self_test)

    args = parser.parse_args(argv)
    if hasattr(args, "num_samples") and args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.repl_timeout <= 0:
        parser.error("--repl-timeout must be positive")
    if args.repl_restart_every < 0:
        parser.error("--repl-restart-every cannot be negative")
    return args


def save_outputs(
    args,
    rows: list[dict[str, Any]],
    results: list[TraceResult],
) -> dict[str, Any]:
    summary = summarize(results)
    result_dicts = [dataclasses.asdict(result) for result in results]

    if args.results_out is not None:
        write_jsonl(args.results_out, result_dicts)
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(
            json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
        )
    markdown = render_markdown(summary)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown)

    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    if args.report is not None:
        print(f"Wrote {args.report}", file=sys.stderr)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repl_dir = resolve_repl_dir(args.repl_dir)

    try:
        if args.mode == "pita":
            rows = fetch_hub_rows(
                dataset_repo=args.dataset_repo,
                tasks=args.tasks,
                num_samples=args.num_samples,
                seed=args.seed,
            )
        elif args.mode == "jsonl":
            rows = read_jsonl_inputs(args.input)
        elif args.mode == "self-test":
            rows = build_self_test_rows()
        else:
            raise AssertionError(f"unknown mode: {args.mode}")

        # Gold-only PITA mode evaluates each completion against itself.
        if args.mode == "pita":
            for row in rows:
                row[args.prediction_column] = row["completion"]

        results = evaluate_rows(
            rows=rows,
            prediction_column=args.prediction_column,
            repl_dir=repl_dir,
            repl_timeout=args.repl_timeout,
            strict_context=args.strict_context,
            repl_restart_every=args.repl_restart_every,
        )
        if args.mode == "self-test":
            assert_self_test(results)
        save_outputs(args, rows, results)
    except (LeanReplError, OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
