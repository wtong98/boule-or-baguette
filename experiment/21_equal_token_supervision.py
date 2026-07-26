"""Plot and tabulate experiment 21's equal-token continuation results.

The evaluator writes one or more pickled DataFrames to
``remote/21_equal_token_supervision/qwen_7b_pita/set``. This script keeps the
newest evaluation of each training checkpoint, selects the final checkpoint
for every run, plots raw accuracy, and prints rebuttal-ready Markdown tables.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = (
    BASE_DIR / "remote/21_equal_token_supervision/qwen_7b_pita/set"
)
DEFAULT_PLOT_PATH = (
    BASE_DIR
    / "fig/final/fig_equal_token_supervision/equal_token_accuracy.svg"
)

TASK_ORDER = ["full", "imply", "or", "php"]
TASK_LABELS = {
    "full": "Full",
    "imply": "Imply",
    "or": "Or",
    "php": "PHP",
}
SOURCE_CHECKPOINT_STEPS = {
    "full": 2_000,
    "imply": 2_000,
    "or": 2_000,
    "php": 500,
}
REGIME_ORDER = ["In-distribution", "Out-of-distribution"]
PLOT_REGIME_LABELS = {
    "In-distribution": "In-dist.",
    "Out-of-distribution": "Out-of-dist.",
}
OBJECTIVE_LABELS = {
    "dp": "DP",
    "ar_cot": "RT",
}

OBJECTIVE_PATTERN = re.compile(
    r"-exp21_equal_token-(?P<objective>dp|ar_cot)\s"
)
ITERATION_PATTERN = re.compile(r"\bitr=(?P<iteration>\d+)\)")


def _read_result_frames(results_dir: Path) -> pd.DataFrame:
    if not results_dir.is_dir():
        raise FileNotFoundError(
            f"evaluation result directory does not exist: {results_dir}"
        )

    frames = []
    for result_path in sorted(results_dir.glob("*.pkl")):
        frame = pd.read_pickle(result_path).copy()
        frame["_result_file"] = str(result_path)
        frame["_result_mtime_ns"] = result_path.stat().st_mtime_ns
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(
            f"no evaluation .pkl files found in {results_dir}"
        )

    results = pd.concat(frames, ignore_index=True)
    required = {
        "run_name",
        "task",
        "model_name",
        "train_split",
        "ckpt_num",
        "res",
    }
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(
            "evaluation result frames are missing columns: "
            + ", ".join(missing)
        )

    # Re-running an evaluation produces another pickle for the same trained
    # adapter. Retain the newest result rather than treating it as another
    # independent training repetition.
    dedup_columns = ["run_name", "ckpt_num"]
    duplicate_count = int(results.duplicated(dedup_columns, keep=False).sum())
    if duplicate_count:
        print(
            "Warning: found "
            f"{duplicate_count} duplicate run/checkpoint evaluations; "
            "using the newest copy of each."
        )
    return (
        results.sort_values("_result_mtime_ns")
        .drop_duplicates(dedup_columns, keep="last")
        .reset_index(drop=True)
    )


def _parse_objective(run_name: str) -> str:
    match = OBJECTIVE_PATTERN.search(run_name)
    if match is None:
        raise ValueError(
            f"could not parse continuation objective from run name: {run_name}"
        )
    return match.group("objective")


def _parse_iteration(run_name: str) -> int:
    match = ITERATION_PATTERN.search(run_name)
    if match is None:
        raise ValueError(
            f"could not parse iteration id from run name: {run_name}"
        )
    return int(match.group("iteration"))


def _regime_label(range_name: str, train_split: int) -> str:
    in_distribution = f"range_(1, {train_split + 1})"
    out_of_distribution = f"range_({train_split + 1}, inf)"
    if range_name == in_distribution:
        return "In-distribution"
    if range_name == out_of_distribution:
        return "Out-of-distribution"
    return range_name.removeprefix("range_")


def _short_model_name(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].replace("Qwen2.5-Coder-", "Qwen-")


def _select_final_checkpoints(results: pd.DataFrame) -> pd.DataFrame:
    checkpoint_numbers = pd.to_numeric(results["ckpt_num"], errors="raise")
    results = results.assign(ckpt_num=checkpoint_numbers)
    final_indices = results.groupby("run_name")["ckpt_num"].idxmax()
    return results.loc[final_indices].reset_index(drop=True)


def build_plot_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Flatten final-checkpoint evaluator output into one row per test range."""

    final_results = _select_final_checkpoints(results)
    rows = []
    for result in final_results.itertuples(index=False):
        objective = _parse_objective(result.run_name)
        iteration = _parse_iteration(result.run_name)
        for range_name, metrics in result.res.items():
            rows.append(
                {
                    "task": result.task,
                    "task_label": TASK_LABELS.get(
                        result.task, str(result.task).capitalize()
                    ),
                    "model": _short_model_name(result.model_name),
                    "objective": objective,
                    "objective_label": OBJECTIVE_LABELS[objective],
                    "iteration": iteration,
                    "train_split": int(result.train_split),
                    "regime": _regime_label(
                        range_name, int(result.train_split)
                    ),
                    "range": range_name,
                    "checkpoint": int(result.ckpt_num),
                    "accuracy": float(metrics.get("gen_acc", math.nan)),
                }
            )

    plot_frame = pd.DataFrame(rows)
    if plot_frame.empty:
        raise ValueError("the evaluation result frames contain no metrics")
    return plot_frame


def _mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "—"
    mean = 100 * values.mean()
    if len(values) == 1:
        return f"{mean:.1f}%"
    return f"{mean:.1f} ± {100 * values.std(ddof=1):.1f}%"


def _signed_mean_sd(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "—"
    mean = 100 * values.mean()
    if len(values) == 1:
        return f"{mean:+.1f} pp"
    return f"{mean:+.1f} ± {100 * values.std(ddof=1):.1f} pp"


def _markdown_table(
    headers: Sequence[str], rows: Iterable[Sequence[object]]
) -> str:
    def escape(value: object) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    header = "| " + " | ".join(escape(item) for item in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(escape(item) for item in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _comparison_rows(plot_frame: pd.DataFrame, regime: str) -> list[list[str]]:
    rows = []
    regime_frame = plot_frame[plot_frame["regime"] == regime]
    for task in TASK_ORDER:
        task_frame = regime_frame[regime_frame["task"] == task]
        if task_frame.empty:
            continue
        pivot = task_frame.pivot_table(
            index="iteration",
            columns="objective",
            values="accuracy",
            aggfunc="first",
        ).reindex(columns=["dp", "ar_cot"])
        dp = pivot["dp"]
        rt = pivot["ar_cot"]
        paired = pivot.dropna(subset=["dp", "ar_cot"])
        rows.append(
            [
                TASK_LABELS.get(task, task.capitalize()),
                ", ".join(sorted(task_frame["model"].unique())),
                _mean_sd(dp),
                _mean_sd(rt),
                _signed_mean_sd(paired["dp"] - paired["ar_cot"]),
                f"{dp.notna().sum()}/{rt.notna().sum()}; "
                f"{len(paired)} paired",
            ]
        )
    return rows


def _checkpoint_rows(plot_frame: pd.DataFrame) -> list[list[object]]:
    rows = []
    for task in TASK_ORDER:
        task_frame = plot_frame[plot_frame["task"] == task]
        if task_frame.empty:
            continue
        checkpoints = (
            task_frame[
                ["iteration", "objective", "checkpoint"]
            ]
            .drop_duplicates()
            .groupby("objective")["checkpoint"]
            .apply(lambda values: ", ".join(map(str, sorted(values.unique()))))
        )
        rows.append(
            [
                TASK_LABELS.get(task, task.capitalize()),
                ", ".join(sorted(task_frame["model"].unique())),
                SOURCE_CHECKPOINT_STEPS.get(task, "—"),
                checkpoints.get("dp", "—"),
                checkpoints.get("ar_cot", "—"),
            ]
        )
    return rows


def render_markdown(plot_frame: pd.DataFrame) -> str:
    """Return compact tables suitable for pasting into a rebuttal."""

    comparison_headers = [
        "PITA split",
        "Source model",
        "DP accuracy",
        "RT accuracy",
        "Δ (DP − RT)",
        "Runs (DP/RT; paired)",
    ]
    sections = [
        "# Equal-token supervision results",
        "",
        (
            "Values are mean ± sample standard deviation across independently "
            "trained adapters. Deltas are paired by repetition. For every "
            "continuation, only its final evaluated checkpoint is included."
        ),
    ]
    for regime in REGIME_ORDER:
        rows = _comparison_rows(plot_frame, regime)
        if rows:
            sections.extend(
                [
                    "",
                    f"## {regime}",
                    "",
                    _markdown_table(comparison_headers, rows),
                ]
            )

    sections.extend(
        [
            "",
            "## Evaluated continuation checkpoints",
            "",
            _markdown_table(
                [
                    "PITA split",
                    "Source model",
                    "Source RT checkpoint",
                    "Final DP continuation step",
                    "Final RT continuation step(s)",
                ],
                _checkpoint_rows(plot_frame),
            ),
        ]
    )
    return "\n".join(sections) + "\n"


def save_plot(plot_frame: pd.DataFrame, output_path: Path) -> None:
    sns.set_theme(
        style="ticks",
        font_scale=1.05,
        rc={"axes.spines.right": False, "axes.spines.top": False},
    )
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.5), sharey=True)

    legend_handles = legend_labels = None
    for axis_index, (axis, task) in enumerate(zip(axes.ravel(), TASK_ORDER)):
        task_frame = plot_frame[plot_frame["task"] == task].copy()
        if task_frame.empty:
            axis.set_visible(False)
            continue
        task_frame["plot_regime"] = task_frame["regime"].replace(
            PLOT_REGIME_LABELS
        )
        plot_regime_order = [
            PLOT_REGIME_LABELS[regime] for regime in REGIME_ORDER
        ]

        sns.boxplot(
            data=task_frame,
            x="plot_regime",
            y="accuracy",
            hue="objective_label",
            order=plot_regime_order,
            hue_order=["DP", "RT"],
            fill=False,
            fliersize=0,
            ax=axis,
        )
        sns.stripplot(
            data=task_frame,
            x="plot_regime",
            y="accuracy",
            hue="objective_label",
            order=plot_regime_order,
            hue_order=["DP", "RT"],
            dodge=True,
            size=3.5,
            alpha=0.8,
            ax=axis,
        )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            legend_handles, legend_labels = handles[:2], labels[:2]
        if axis.get_legend() is not None:
            axis.get_legend().remove()

        model_label = ", ".join(sorted(task_frame["model"].unique()))
        axis.set_title(f"{TASK_LABELS.get(task, task.capitalize())} ({model_label})")
        axis.set_ylim(0, 1)
        axis.set_xlabel("")
        axis.set_ylabel("Accuracy" if axis_index % 2 == 0 else "")

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.01),
        )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot experiment 21 evaluation pickles and print Markdown tables."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=f"directory containing evaluator pickles (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help=f"plot output path (default: {DEFAULT_PLOT_PATH})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="also write the printed Markdown report to this path",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="print tables without creating a plot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = _read_result_frames(args.results_dir)
    plot_frame = build_plot_frame(results)
    markdown = render_markdown(plot_frame)
    print(markdown, end="")

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(markdown)
        print(f"\nSaved Markdown report to {args.report}")
    if not args.no_plot:
        save_plot(plot_frame, args.output)
        print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
