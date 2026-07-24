"""Generate Markdown tables for the PITA class-balance accuracy audit.

The saved ``res_eval.*.pkl`` files contain aggregate generation outcomes for
100-example evaluation samples.  Newer ("granular") files store TP/TN/FP/FN
and no-label rates in addition to raw accuracy.  This script discovers those
files, selects the paper's default checkpoint and out-of-distribution test
range, and reports metrics across training runs.

Dataset class counts can be read from a local Hugging Face DatasetDict or
computed directly from the Parquet files on the Hugging Face Hub with DuckDB.
The latter reads only ``is_true`` and ``length`` rather than downloading the
roughly 150 GB uncompressed dataset.

Examples:

    python experiment/19_accuracy_report.py \
        --dataset-path /path/to/pita \
        --output experiment/accuracy_report.md

    # Remote Parquet scan (set HF_TOKEN to avoid anonymous rate limits):
    python experiment/19_accuracy_report.py \
        --output experiment/accuracy_report.md

    # Reuse counts from a previous run:
    python experiment/19_accuracy_report.py \
        --counts-json pita_class_counts.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "remote" / "17_llm_clean"
DEFAULT_DATASET_REPO = "williamtong105/pita"
DEFAULT_CHECKPOINT = 2000
EVALUATION_SAMPLE_SIZE = 100


@dataclass(frozen=True)
class TaskSpec:
    key: str
    display_name: str
    train_max_length: int
    dataset_split: str


TASKS = {
    "full": TaskSpec("full", "Full", 4, "full"),
    "imply": TaskSpec("imply", "Imply", 6, "imply"),
    "or": TaskSpec("or", "Or", 18, "or"),
    "php": TaskSpec("php", "PHP", 60, "php"),
}

CONFUSION_KEYS = {
    "gen_acc",
    "true_pos",
    "true_neg",
    "false_pos",
    "false_neg",
    "prop_none",
}

RANGE_RE = re.compile(
    r"^range_\(\s*(?P<start>-?\d+(?:\.\d+)?)\s*,\s*"
    r"(?P<stop>inf|-?\d+(?:\.\d+)?)\s*\)$"
)
ITERATION_RE = re.compile(r"\bitr=(?P<iteration>\d+)\b")


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "This report reads pandas pickle files. Install pandas first "
            "(for example, `python -m pip install pandas`)."
        ) from exc
    return pd


def clamp_rate(value: Any) -> float:
    """Remove harmless float32 excursions just outside [0, 1]."""

    number = float(value)
    if -1e-6 <= number <= 0:
        return 0.0
    if 1 <= number <= 1 + 1e-6:
        return 1.0
    return number


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return math.nan
    return numerator / denominator


def normalize_task(row: Any) -> str | None:
    candidates = [
        str(row.get("task", "")).lower(),
        str(row.get("project_name", "")).lower(),
    ]
    joined = " ".join(candidates)
    if "php" in joined:
        return "php"
    if "imply" in joined:
        return "imply"
    if any(value == "or" for value in candidates) or "_or" in joined:
        return "or"
    if "full" in joined:
        return "full"
    return None


def normalize_prompt(run_name: str) -> str | None:
    prefix = run_name.partition(" (")[0].lower()
    if prefix.endswith("-dp"):
        return "DP"
    if prefix.endswith("-ar_cot") or prefix.endswith("-cot"):
        return "RT"
    return None


def display_model(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1]


def model_family(model_name: str) -> str:
    lowered = model_name.lower()
    if "qwen2.5" in lowered:
        return "Qwen2.5-Coder"
    if "qwen3" in lowered:
        return "Qwen3"
    if "gemma" in lowered:
        return "Gemma 3"
    if "llama" in lowered:
        return "Llama 3"
    return model_name.split("/", 1)[0]


def is_test_range(range_name: str, task: TaskSpec) -> bool:
    match = RANGE_RE.match(range_name)
    if match is None:
        return False
    start = float(match.group("start"))
    stop_text = match.group("stop")
    return (
        math.isclose(start, task.train_max_length + 1)
        and stop_text == "inf"
    )


def discover_result_records(
    results_root: Path,
    checkpoint: int | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Read all usable test-result rows with confusion statistics."""

    pd = import_pandas()
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    unreadable = 0
    empty = 0
    raw_only_rows = Counter()

    for pickle_path in sorted(results_root.rglob("*.pkl")):
        try:
            frame = pd.read_pickle(pickle_path)
        except Exception:
            unreadable += 1
            continue

        if frame.empty or "res" not in frame.columns:
            empty += 1
            continue

        if checkpoint is None:
            selected = frame.loc[
                frame.groupby("run_name")["ckpt_num"].transform("max")
                == frame["ckpt_num"]
            ]
        else:
            selected = frame[frame["ckpt_num"] == checkpoint]

        for _, row in selected.iterrows():
            task_key = normalize_task(row)
            run_name = str(row.get("run_name", ""))
            prompt = normalize_prompt(run_name)
            model_name = str(row.get("model_name", ""))
            result_map = row.get("res")

            if (
                task_key is None
                or prompt is None
                or not model_name
                or not isinstance(result_map, dict)
            ):
                continue

            task = TASKS[task_key]
            matching = [
                metrics
                for range_name, metrics in result_map.items()
                if is_test_range(str(range_name), task)
            ]
            if len(matching) != 1 or not isinstance(matching[0], dict):
                continue

            metrics = matching[0]
            if not CONFUSION_KEYS.issubset(metrics):
                raw_only_rows[(model_family(model_name), task_key)] += 1
                continue

            iteration_match = ITERATION_RE.search(run_name)
            iteration = (
                int(iteration_match.group("iteration"))
                if iteration_match
                else None
            )
            record = {
                "task": task_key,
                "model_name": model_name,
                "model": display_model(model_name),
                "family": model_family(model_name),
                "prompt": prompt,
                "run_name": run_name,
                "iteration": iteration,
                "checkpoint": int(row["ckpt_num"]),
                "source_file": str(pickle_path.relative_to(results_root)),
            }
            record.update(
                {key: clamp_rate(metrics[key]) for key in CONFUSION_KEYS}
            )
            records.append(record)

    if unreadable:
        warnings.append(f"Skipped {unreadable} unreadable pickle file(s).")
    if empty:
        warnings.append(f"Skipped {empty} empty result file(s).")

    if raw_only_rows:
        coverage = ", ".join(
            f"{family}/{TASKS[task].display_name} ({count} rows)"
            for (family, task), count in sorted(raw_only_rows.items())
        )
        warnings.append(
            "Raw-accuracy-only artifacts cannot provide class metrics: "
            + coverage
            + "."
        )

    return records, warnings


def balanced_accuracy_bounds(
    true_pos: float,
    true_neg: float,
    false_pos: float,
    false_neg: float,
    no_label: float,
) -> tuple[float, float]:
    """Bounds when the saved file does not identify no-label examples' class.

    Let x be the no-label mass whose gold class is True.  Standard balanced
    accuracy, counting no-label outcomes as errors, is

        1/2 [TP/(TP+FN+x) + TN/(TN+FP+no_label-x)].

    The expression is convex in x, so its maximum is at an endpoint and its
    minimum is at an endpoint or the single stationary point.
    """

    positive_labeled = true_pos + false_neg
    negative_labeled = true_neg + false_pos

    def value(true_no_label: float) -> float:
        true_total = positive_labeled + true_no_label
        false_total = negative_labeled + no_label - true_no_label
        true_accuracy = safe_divide(true_pos, true_total)
        false_accuracy = safe_divide(true_neg, false_total)

        # A zero numerator and zero denominator can only occur at an endpoint.
        # Approach that endpoint from the interior, where the recall is zero.
        if math.isnan(true_accuracy) and true_pos == 0:
            true_accuracy = 0.0
        if math.isnan(false_accuracy) and true_neg == 0:
            false_accuracy = 0.0
        if math.isnan(true_accuracy) or math.isnan(false_accuracy):
            return math.nan
        return 0.5 * (true_accuracy + false_accuracy)

    candidates = [0.0, no_label]
    if true_pos > 0 and true_neg > 0:
        sqrt_tp = math.sqrt(true_pos)
        sqrt_tn = math.sqrt(true_neg)
        stationary = (
            sqrt_tp * (negative_labeled + no_label)
            - sqrt_tn * positive_labeled
        ) / (sqrt_tp + sqrt_tn)
        candidates.append(min(no_label, max(0.0, stationary)))

    values = [value(candidate) for candidate in candidates]
    values = [item for item in values if not math.isnan(item)]
    if not values:
        return math.nan, math.nan
    return min(values), max(values)


def add_derived_metrics(records: list[dict[str, Any]]) -> None:
    for record in records:
        true_denominator = record["true_pos"] + record["false_neg"]
        false_denominator = record["true_neg"] + record["false_pos"]
        true_emitted_accuracy = safe_divide(
            record["true_pos"], true_denominator
        )
        false_emitted_accuracy = safe_divide(
            record["true_neg"], false_denominator
        )
        if math.isnan(true_emitted_accuracy) or math.isnan(
            false_emitted_accuracy
        ):
            emitted_balanced_accuracy = math.nan
        else:
            emitted_balanced_accuracy = 0.5 * (
                true_emitted_accuracy + false_emitted_accuracy
            )

        lower, upper = balanced_accuracy_bounds(
            true_pos=record["true_pos"],
            true_neg=record["true_neg"],
            false_pos=record["false_pos"],
            false_neg=record["false_neg"],
            no_label=record["prop_none"],
        )
        true_inclusive_lower = safe_divide(
            record["true_pos"],
            true_denominator + record["prop_none"],
        )
        true_inclusive_upper = true_emitted_accuracy
        false_inclusive_lower = safe_divide(
            record["true_neg"],
            false_denominator + record["prop_none"],
        )
        false_inclusive_upper = false_emitted_accuracy
        record.update(
            {
                "true_emitted_accuracy": true_emitted_accuracy,
                "false_emitted_accuracy": false_emitted_accuracy,
                "emitted_balanced_accuracy": emitted_balanced_accuracy,
                "true_inclusive_lower": true_inclusive_lower,
                "true_inclusive_upper": true_inclusive_upper,
                "false_inclusive_lower": false_inclusive_lower,
                "false_inclusive_upper": false_inclusive_upper,
                "balanced_accuracy_lower": lower,
                "balanced_accuracy_upper": upper,
            }
        )


def average_duplicate_evaluations(records: list[dict[str, Any]]):
    """Give each training run equal weight if it was evaluated more than once."""

    pd = import_pandas()
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    metric_columns = [
        "gen_acc",
        "true_pos",
        "true_neg",
        "false_pos",
        "false_neg",
        "prop_none",
        "true_emitted_accuracy",
        "false_emitted_accuracy",
        "emitted_balanced_accuracy",
        "true_inclusive_lower",
        "true_inclusive_upper",
        "false_inclusive_lower",
        "false_inclusive_upper",
        "balanced_accuracy_lower",
        "balanced_accuracy_upper",
    ]
    group_columns = [
        "task",
        "model_name",
        "model",
        "family",
        "prompt",
        "run_name",
        "iteration",
        "checkpoint",
    ]
    return (
        frame.groupby(group_columns, dropna=False, as_index=False)[
            metric_columns
        ]
        .mean()
        .sort_values(["task", "family", "model", "prompt", "run_name"])
    )


def validate_counts(counts: dict[str, Any]) -> dict[str, dict[str, dict[str, int]]]:
    normalized: dict[str, dict[str, dict[str, int]]] = {}
    for task_key in TASKS:
        try:
            task_counts = counts[task_key]
            normalized[task_key] = {}
            for partition in ("train", "test"):
                partition_counts = task_counts[partition]
                normalized[task_key][partition] = {
                    "true": int(partition_counts["true"]),
                    "false": int(partition_counts["false"]),
                }
                if min(normalized[task_key][partition].values()) < 0:
                    raise ValueError("counts must be non-negative")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Counts JSON must contain non-negative integer counts at "
                f"{task_key}.{{train,test}}.{{true,false}}"
            ) from exc
    return normalized


def counts_from_local_dataset(
    dataset_path: Path,
) -> dict[str, dict[str, dict[str, int]]]:
    try:
        from datasets import DatasetDict
    except ImportError as exc:
        raise RuntimeError(
            "Reading --dataset-path requires the `datasets` package."
        ) from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Reading --dataset-path requires NumPy."
        ) from exc

    dataset = DatasetDict.load_from_disk(str(dataset_path.expanduser()))
    counts: dict[str, dict[str, dict[str, int]]] = {}

    for task_key, task in TASKS.items():
        if task.dataset_split not in dataset:
            raise RuntimeError(
                f"Local dataset lacks split {task.dataset_split!r}; "
                f"available splits: {', '.join(dataset.keys())}"
            )
        split = dataset[task.dataset_split]
        missing = {"is_true", "length"}.difference(split.column_names)
        if missing:
            raise RuntimeError(
                f"Dataset split {task.dataset_split!r} lacks columns: "
                + ", ".join(sorted(missing))
            )

        accumulator = {
            "train": {"true": 0, "false": 0},
            "test": {"true": 0, "false": 0},
        }
        label_and_length = split.select_columns(["is_true", "length"])
        for batch in label_and_length.iter(batch_size=1_000_000):
            lengths = np.asarray(batch["length"])
            labels = np.asarray(batch["is_true"], dtype=bool)
            included = lengths >= 1
            train = included & (lengths <= task.train_max_length)
            test = included & (lengths > task.train_max_length)
            accumulator["train"]["true"] += int((train & labels).sum())
            accumulator["train"]["false"] += int((train & ~labels).sum())
            accumulator["test"]["true"] += int((test & labels).sum())
            accumulator["test"]["false"] += int((test & ~labels).sum())
        counts[task_key] = accumulator

    return validate_counts(counts)


def counts_from_huggingface_parquet(
    dataset_repo: str,
) -> dict[str, dict[str, dict[str, int]]]:
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "Remote class counts require DuckDB. Install it with "
            "`python -m pip install duckdb`, provide --dataset-path, or "
            "reuse a file with --counts-json."
        ) from exc

    connection = duckdb.connect()
    connection.execute("SET threads = 1")
    connection.execute("SET http_retries = 10")

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    if token:
        # Parameter binding keeps the token out of SQL text and error output.
        connection.execute(
            "CREATE OR REPLACE SECRET pita_hf_secret "
            "(TYPE huggingface, TOKEN ?)",
            [token],
        )

    counts: dict[str, dict[str, dict[str, int]]] = {}
    for task_key, task in TASKS.items():
        print(
            f"Counting {task.display_name} labels from {dataset_repo} ...",
            file=sys.stderr,
            flush=True,
        )
        source = (
            f"hf://datasets/{dataset_repo}/data/"
            f"{task.dataset_split}-*.parquet"
        )
        query = """
            SELECT
                CASE
                    WHEN length BETWEEN 1 AND ? THEN 'train'
                    WHEN length > ? THEN 'test'
                END AS partition,
                is_true,
                COUNT(*) AS n
            FROM read_parquet(?)
            WHERE length >= 1
            GROUP BY partition, is_true
        """
        try:
            rows = connection.execute(
                query,
                [task.train_max_length, task.train_max_length, source],
            ).fetchall()
        except Exception as exc:
            raise RuntimeError(
                f"Unable to scan the {task.display_name} Parquet files. "
                "If Hugging Face reports a rate limit, set HF_TOKEN; "
                "alternatively use --dataset-path or --counts-json."
            ) from exc
        task_counts = {
            "train": {"true": 0, "false": 0},
            "test": {"true": 0, "false": 0},
        }
        for partition, is_true, count in rows:
            label = "true" if is_true else "false"
            task_counts[str(partition)][label] = int(count)
        counts[task_key] = task_counts

    return validate_counts(counts)


def load_dataset_counts(args) -> dict[str, dict[str, dict[str, int]]] | None:
    if args.skip_dataset_proportions:
        return None
    if args.counts_json is not None:
        with args.counts_json.open() as handle:
            return validate_counts(json.load(handle))
    if args.dataset_path is not None:
        counts = counts_from_local_dataset(args.dataset_path)
    else:
        counts = counts_from_huggingface_parquet(args.dataset_repo)

    if args.write_counts_json is not None:
        args.write_counts_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_counts_json.write_text(
            json.dumps(counts, indent=2, sort_keys=True) + "\n"
        )
    return counts


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    def escape(value: Any) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(map(escape, headers)) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(escape(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def format_percentage(value: float, digits: int = 1) -> str:
    if value is None or math.isnan(float(value)):
        return "—"
    return f"{100 * float(value):.{digits}f}%"


def format_mean_sd(series, digits: int = 1) -> str:
    clean = series.dropna()
    if clean.empty:
        return "—"
    mean = 100 * clean.mean()
    if len(clean) == 1:
        return f"{mean:.{digits}f}%"
    standard_deviation = 100 * clean.std(ddof=1)
    return f"{mean:.{digits}f} ± {standard_deviation:.{digits}f}%"


def render_dataset_table(
    counts: dict[str, dict[str, dict[str, int]]],
) -> str:
    rows = []
    for task_key, task in TASKS.items():
        train = counts[task_key]["train"]
        test = counts[task_key]["test"]
        train_total = train["true"] + train["false"]
        test_total = test["true"] + test["false"]
        train_true = safe_divide(train["true"], train_total)
        test_true = safe_divide(test["true"], test_total)
        rows.append(
            [
                task.display_name,
                f"1–{task.train_max_length}",
                f"{task.train_max_length + 1}–∞",
                f"{train_total:,}",
                format_percentage(train_true),
                format_percentage(1 - train_true),
                f"{test_total:,}",
                format_percentage(test_true),
                format_percentage(1 - test_true),
                f"{100 * (test_true - train_true):+.1f} pp",
            ]
        )
    return markdown_table(
        [
            "PITA split",
            "Train lengths",
            "Test lengths",
            "Train n",
            "Train true",
            "Train false",
            "Test n",
            "Test true",
            "Test false",
            "Δ true",
        ],
        rows,
    )


def grouped_result_rows(run_frame):
    group_columns = ["task", "family", "model", "prompt"]
    for keys, group in run_frame.groupby(group_columns, sort=True):
        yield keys, group


def render_model_metrics(run_frame) -> str:
    rows = []
    for (task_key, family, model, prompt), group in grouped_result_rows(
        run_frame
    ):
        class_metric_runs = int(
            group["emitted_balanced_accuracy"].notna().sum()
        )
        rows.append(
            [
                TASKS[task_key].display_name,
                family,
                model,
                prompt,
                len(group),
                class_metric_runs,
                format_mean_sd(group["gen_acc"]),
                format_mean_sd(group["true_emitted_accuracy"]),
                format_mean_sd(group["false_emitted_accuracy"]),
                format_mean_sd(group["emitted_balanced_accuracy"]),
                format_mean_sd(group["prop_none"]),
            ]
        )
    return markdown_table(
        [
            "PITA split",
            "Family",
            "Model",
            "Target",
            "Runs",
            "Class-metric runs",
            "Raw accuracy",
            "True accuracy†",
            "False accuracy†",
            "Balanced accuracy†",
            "No label",
        ],
        rows,
    )


def render_bound_table(run_frame) -> str:
    rows = []

    def interval(lower: float, upper: float) -> str:
        if math.isnan(lower) or math.isnan(upper):
            return "—"
        if upper - lower < 0.0005:
            return format_percentage(0.5 * (lower + upper))
        return f"{format_percentage(lower)}–{format_percentage(upper)}"

    for (task_key, family, model, prompt), group in grouped_result_rows(
        run_frame
    ):
        true_lower = group["true_inclusive_lower"].mean()
        true_upper = group["true_inclusive_upper"].mean()
        false_lower = group["false_inclusive_lower"].mean()
        false_upper = group["false_inclusive_upper"].mean()
        lower = group["balanced_accuracy_lower"].mean()
        upper = group["balanced_accuracy_upper"].mean()
        no_label = group["prop_none"].mean()
        rows.append(
            [
                TASKS[task_key].display_name,
                family,
                model,
                prompt,
                len(group),
                format_percentage(no_label),
                interval(true_lower, true_upper),
                interval(false_lower, false_upper),
                interval(lower, upper),
            ]
        )
    return markdown_table(
        [
            "PITA split",
            "Family",
            "Model",
            "Target",
            "Runs",
            "Mean no-label rate",
            "True accuracy incl. no-label‡",
            "False accuracy incl. no-label‡",
            "Balanced accuracy incl. no-label‡",
        ],
        rows,
    )


def render_rt_dp_comparison(run_frame) -> str:
    rows = []
    group_columns = ["task", "family", "model"]
    for (task_key, family, model), group in run_frame.groupby(
        group_columns, sort=True
    ):
        by_prompt = {
            prompt: prompt_group
            for prompt, prompt_group in group.groupby("prompt")
        }
        if not {"DP", "RT"}.issubset(by_prompt):
            continue

        dp = by_prompt["DP"]
        rt = by_prompt["RT"]
        dp_raw = dp["gen_acc"].mean()
        rt_raw = rt["gen_acc"].mean()
        dp_lower = dp["balanced_accuracy_lower"].mean()
        dp_upper = dp["balanced_accuracy_upper"].mean()
        rt_lower = rt["balanced_accuracy_lower"].mean()
        rt_upper = rt["balanced_accuracy_upper"].mean()

        def interval(lower: float, upper: float, signed: bool = False) -> str:
            if signed:
                if upper - lower < 0.0005:
                    return f"{100 * (lower + upper) / 2:+.1f} pp"
                return f"{100 * lower:+.1f} to {100 * upper:+.1f} pp"
            if upper - lower < 0.0005:
                return format_percentage((lower + upper) / 2)
            return (
                f"{format_percentage(lower)}–{format_percentage(upper)}"
            )

        rows.append(
            [
                TASKS[task_key].display_name,
                family,
                model,
                format_percentage(dp_raw),
                format_percentage(rt_raw),
                f"{100 * (rt_raw - dp_raw):+.1f} pp",
                interval(dp_lower, dp_upper),
                interval(rt_lower, rt_upper),
                interval(
                    rt_lower - dp_upper,
                    rt_upper - dp_lower,
                    signed=True,
                ),
            ]
        )

    return markdown_table(
        [
            "PITA split",
            "Family",
            "Model",
            "DP raw",
            "RT raw",
            "Δ raw (RT−DP)",
            "DP balanced‡",
            "RT balanced‡",
            "Δ balanced‡ (RT−DP)",
        ],
        rows,
    )


def render_coverage(run_frame) -> str:
    rows = []
    for (family, task_key), group in run_frame.groupby(
        ["family", "task"], sort=True
    ):
        models = group["model"].nunique()
        prompts = ", ".join(sorted(group["prompt"].unique()))
        rows.append(
            [
                family,
                TASKS[task_key].display_name,
                models,
                prompts,
                len(group),
            ]
        )
    return markdown_table(
        ["Family", "PITA split", "Models", "Targets", "Training runs"],
        rows,
    )


def render_report(
    run_frame,
    counts: dict[str, dict[str, dict[str, int]]] | None,
    warnings: list[str],
    checkpoint_label: str,
    results_root: Path,
) -> str:
    lines = [
        "# PITA class-balance and accuracy report",
        "",
        (
            f"Results source: `{results_root}`. Checkpoint selection: "
            f"{checkpoint_label}. Evaluation slice: proof lengths strictly "
            "above each split's training cutoff."
        ),
        "",
        "## Definitions",
        "",
        (
            "Raw accuracy is `(TP + TN) / n`; a generation with neither a "
            "`success` nor `failure` label is incorrect. True and false "
            "per-class accuracies are recalls for their respective classes. "
            "Balanced accuracy is their unweighted mean."
        ),
        "",
        (
            "Accuracy after exactly reweighting the two test classes to 50/50 "
            "is therefore the same estimand as balanced accuracy. A newly "
            "sampled finite 50/50 test slice is a useful robustness rerun, "
            "but is not a different population metric."
        ),
        "",
        (
            "The saved result schema records no-label outcomes only in "
            "aggregate, without their gold class. Consequently, † metrics "
            "condition on generation of an explicit label. They equal "
            "ordinary per-class/balanced accuracy when the no-label rate is "
            "zero. Table 3 gives the identifiable abstention-inclusive "
            "balanced-accuracy bounds; exact values require per-example "
            "predictions or class-specific no-label counts."
        ),
        "",
        "## 1. Train/test class proportions",
        "",
    ]

    if counts is None:
        lines.append(
            "_Skipped. Re-run without `--skip-dataset-proportions` to include "
            "this table._"
        )
    else:
        lines.append(render_dataset_table(counts))

    lines.extend(
        [
            "",
            "## 2. Test accuracy by model, target, and class",
            "",
            (
                "Entries are mean ± sample standard deviation across training "
                "runs. Duplicate evaluation files for the same training run "
                "are averaged first, so reruns do not receive extra weight."
            ),
            "",
            render_model_metrics(run_frame),
            "",
            "## 3. Abstention-inclusive balanced-accuracy bounds",
            "",
            (
                "‡ Bounds assign the aggregate no-label mass to gold classes "
                "in every possible way. A single percentage is exact under "
                "the saved aggregate schema."
            ),
            "",
            render_bound_table(run_frame),
            "",
            "## 4. Does the RT/DP reversal survive class balancing?",
            "",
            (
                "Positive differences favor RT; negative differences favor "
                "DP. Intervals propagate the unknown gold-class allocation "
                "of no-label generations."
            ),
            "",
            render_rt_dp_comparison(run_frame),
            "",
            "## 5. Detailed-artifact coverage",
            "",
            render_coverage(run_frame),
        ]
    )

    if warnings:
        lines.extend(["", "## Data-quality notes", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    lines.extend(
        [
            "",
            "## Reproducibility notes",
            "",
            (
                f"- Each saved evaluation uses {EVALUATION_SAMPLE_SIZE} "
                "examples per length range."
            ),
            (
                "- The report excludes length 0, matching the training and "
                "evaluation loaders (`1 <= length`)."
            ),
            (
                "- For a literal class-balanced-slice experiment, rerun "
                "generation on equal numbers of true and false examples and "
                "save per-example gold labels/predictions (or at minimum "
                "`true_none` and `false_none`)."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Generate the PITA class-balance accuracy report."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root searched recursively for res_eval.*.pkl files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Checkpoint number to report, or 'latest' per run (default: 2000).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Local flat PITA DatasetDict saved with save_to_disk().",
    )
    parser.add_argument(
        "--dataset-repo",
        default=DEFAULT_DATASET_REPO,
        help="Hugging Face dataset repo used for a remote Parquet scan.",
    )
    parser.add_argument(
        "--counts-json",
        type=Path,
        help="Read precomputed class counts instead of scanning PITA.",
    )
    parser.add_argument(
        "--write-counts-json",
        type=Path,
        help="Write computed class counts for faster later runs.",
    )
    parser.add_argument(
        "--skip-dataset-proportions",
        action="store_true",
        help="Generate only result-artifact tables.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write Markdown here instead of standard output.",
    )
    args = parser.parse_args(argv)

    if args.checkpoint == "latest":
        args.checkpoint_number = None
        args.checkpoint_label = "latest checkpoint available per run"
    else:
        try:
            args.checkpoint_number = int(args.checkpoint)
        except ValueError:
            parser.error("--checkpoint must be an integer or 'latest'")
        args.checkpoint_label = f"{args.checkpoint_number}"

    if args.skip_dataset_proportions and (
        args.dataset_path is not None
        or args.counts_json is not None
        or args.write_counts_json is not None
    ):
        parser.error(
            "--skip-dataset-proportions cannot be combined with dataset/count "
            "arguments"
        )
    if args.dataset_path is not None and args.counts_json is not None:
        parser.error("--dataset-path and --counts-json are mutually exclusive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.results_root.is_dir():
        raise SystemExit(f"Results root not found: {args.results_root}")

    records, warnings = discover_result_records(
        args.results_root,
        args.checkpoint_number,
    )
    if not records:
        raise SystemExit(
            "No detailed out-of-distribution result rows matched the requested "
            "checkpoint."
        )

    add_derived_metrics(records)
    run_frame = average_duplicate_evaluations(records)
    try:
        counts = load_dataset_counts(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    report = render_report(
        run_frame=run_frame,
        counts=counts,
        warnings=warnings,
        checkpoint_label=args.checkpoint_label,
        results_root=args.results_root,
    )

    if args.output is None:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report)
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
