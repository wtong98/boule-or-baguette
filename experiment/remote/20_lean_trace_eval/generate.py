"""Generate PITA traces on the cluster for later Lean validation.

This is phase one of experiment 20.  Each invocation loads the ``itr=0``
Qwen2.5-Coder-7B ``ar_cot`` LoRA for one PITA task, samples 100 examples
from that task's held-out length split, and writes a self-contained JSONL
file.  Lean is deliberately not needed on the generation node.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
LLM_CLEAN_DIR = SCRIPT_DIR.parent / "17_llm_clean"


DEFAULT_DATASET_PATH = Path(
    "/n/netscratch/pehlevan_lab/Lab/wlt/data/pita"
)
DEFAULT_OUTPUT_DIR = Path(
    "/n/netscratch/pehlevan_lab/Lab/wlt/pita_traces"
)
DEFAULT_CHECKPOINT_ROOT = Path("~/scratch/ckpt/final")
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B"
ITERATION_ID = 0
PROMPT_STYLE = "ar_cot"


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    name: str
    cutoff: int
    project_name: str
    checkpoint_step: int
    max_sequence_length: int

    def checkpoint_path(self, checkpoint_root: Path) -> Path:
        run_name = (
            f"Qwen2.5-Coder-7B-{PROMPT_STYLE}_"
            f"(split={self.cutoff},_itr={ITERATION_ID})"
        )
        return (
            checkpoint_root.expanduser()
            / self.project_name
            / run_name
            / f"checkpoint-{self.checkpoint_step}"
        )


TASKS = (
    TaskSpec("full", 4, "qwen_full", 2_000, 2_048),
    TaskSpec("imply", 6, "qwen_imply", 2_000, 16_384),
    TaskSpec("or", 18, "qwen_or", 2_000, 32_768),
    TaskSpec("php", 60, "qwen_php_enum", 500, 32_768),
)
TASK_BY_NAME = {task.name: task for task in TASKS}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate held-out PITA traces with an itr=0 Qwen LoRA."
    )
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", choices=tuple(TASK_BY_NAME))
    task_group.add_argument(
        "--task-index",
        type=int,
        choices=range(len(TASKS)),
        help="Zero-based task index used by the Slurm array.",
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument(
        "--dataset-cache-dir",
        type=Path,
        default=DEFAULT_DATASET_PATH / ".datasets_cache",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--quantization",
        default="bitsandbytes",
        help="vLLM quantization mode; use 'none' to disable.",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing task JSONL.",
    )
    args = parser.parse_args(argv)
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    return args


def select_rows(
    dataset_path: Path,
    cache_dir: Path | None,
    spec: TaskSpec,
    num_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select deterministic random rows from lengths above the train cutoff."""

    sys.path.insert(0, str(LLM_CLEAN_DIR))
    try:
        from pita_dataset import load_pita_split
    except ImportError as exc:
        raise RuntimeError(
            "PITA sampling requires the Hugging Face `datasets` package."
        ) from exc

    split = load_pita_split(
        str(dataset_path),
        spec.name,
        str(cache_dir) if cache_dir is not None else None,
    )
    evaluation_indices = [
        index
        for index, length in enumerate(split["length"])
        if int(length) > spec.cutoff
    ]
    if len(evaluation_indices) < num_samples:
        raise RuntimeError(
            f"{spec.name} has only {len(evaluation_indices)} held-out examples; "
            f"cannot sample {num_samples}"
        )

    selected_indices = random.Random(seed).sample(
        evaluation_indices,
        num_samples,
    )
    selected = split.select(selected_indices)
    rows: list[dict[str, Any]] = []
    for dataset_index, row in zip(selected_indices, selected):
        normalized = dict(row)
        normalized.update(
            {
                "task": spec.name,
                "dataset_index": dataset_index,
                "sample_seed": seed,
                "model_name": MODEL_NAME,
                "prompt_style": PROMPT_STYLE,
                "iteration_id": ITERATION_ID,
                "checkpoint_step": spec.checkpoint_step,
            }
        )
        rows.append(normalized)
    return rows


def generate(
    rows: list[dict[str, Any]],
    spec: TaskSpec,
    checkpoint: Path,
    quantization: str | None,
    tensor_parallel_size: int,
) -> None:
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "Trace generation requires a vLLM environment with LoRA support."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "model": MODEL_NAME,
        "enable_lora": True,
        "tensor_parallel_size": tensor_parallel_size,
        "max_model_len": spec.max_sequence_length,
    }
    if quantization and quantization != "none":
        model_kwargs["quantization"] = quantization

    model = LLM(**model_kwargs)
    prompts = [str(row["prompt"]) for row in rows]
    tokenizer = model.get_tokenizer()
    longest_prompt = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    max_tokens = spec.max_sequence_length - longest_prompt
    if max_tokens <= 0:
        raise RuntimeError(
            f"Longest {spec.name} prompt has {longest_prompt} tokens, which "
            f"does not fit max_model_len={spec.max_sequence_length}"
        )

    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        stop=["<success />", "<failure />"],
        include_stop_str_in_output=True,
    )
    lora_request = LoRARequest(
        f"pita_{spec.name}_itr{ITERATION_ID}",
        1,
        str(checkpoint),
    )
    outputs = model.generate(
        prompts,
        sampling,
        lora_request=lora_request,
    )
    if len(outputs) != len(rows):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(rows)} prompts"
        )

    for row, output in zip(rows, outputs):
        row["prediction"] = output.outputs[0].text
        row["checkpoint"] = str(checkpoint)
        row["max_generation_tokens"] = max_tokens


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass --overwrite to replace it"
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = (
        TASKS[args.task_index]
        if args.task_index is not None
        else TASK_BY_NAME[args.task]
    )
    checkpoint = spec.checkpoint_path(args.checkpoint_root).resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {checkpoint}")

    rows = select_rows(
        dataset_path=args.dataset_path.expanduser(),
        cache_dir=(
            args.dataset_cache_dir.expanduser()
            if args.dataset_cache_dir is not None
            else None
        ),
        spec=spec,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    generate(
        rows=rows,
        spec=spec,
        checkpoint=checkpoint,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    output = args.output_dir.expanduser() / f"traces.{spec.name}.jsonl"
    write_jsonl(output, rows, overwrite=args.overwrite)
    print(f"Wrote {len(rows)} {spec.name} traces to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
