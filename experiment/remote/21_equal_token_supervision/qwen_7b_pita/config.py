"""Experiment 21: equal-token continuation from Qwen RT checkpoints."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.append("../")
from run import perform_run


DATA_PATH = "/n/netscratch/pehlevan_lab/Lab/wlt/data/pita"
SOURCE_ROOT = Path("~/scratch/ckpt/final").expanduser()
OUTPUT_ROOT = Path("~/scratch/ckpt/final/experiment_21_equal_token").expanduser()

ITR_IDS = list(range(5))
DP_STEPS = 500
BASE_LENGTH = 1_024
TOTAL_BATCH_SIZE = 32

QWEN_7B = {
    "name": "Qwen/Qwen2.5-Coder-7B",
    "2k_bs": 32,
}
QWEN_0P5B = {
    "name": "Qwen/Qwen2.5-Coder-0.5B",
    "2k_bs": 32,
}

OBJECTIVES = ("dp", "ar_cot")

TASKS = [
    {
        "name": "full",
        "dataset_split": "full",
        "train_split": 4,
        "train_len": 512,
        "test_len": 2_048,
        "model": QWEN_7B,
        "source_experiment": "17_llm_clean/qwen_7b_pita",
        "source_project": "qwen_full",
        "source_checkpoint_step": 2_000,
    },
    {
        "name": "imply",
        "dataset_split": "imply",
        "train_split": 6,
        "train_len": 1_024,
        "test_len": 16_384,
        "model": QWEN_7B,
        "source_experiment": "17_llm_clean/qwen_7b_pita",
        "source_project": "qwen_imply",
        "source_checkpoint_step": 2_000,
    },
    {
        "name": "or",
        "dataset_split": "or",
        "train_split": 18,
        "train_len": 8_192,
        "test_len": 32_768,
        "model": QWEN_7B,
        "source_experiment": "17_llm_clean/qwen_7b_pita",
        "source_project": "qwen_or",
        "source_checkpoint_step": 2_000,
    },
    {
        "name": "php",
        "dataset_split": "php",
        "train_split": 60,
        "train_len": 32_768,
        "test_len": 32_768,
        "model": QWEN_0P5B,
        "source_experiment": "17_llm_clean/qwen_0p5b_php",
        "source_project": "qwen_php_enum",
        "source_checkpoint_step": 500,
    },
]


def source_checkpoint(task, itr_id):
    model = task["model"]
    source_run_name = (
        f"{model['name'].split('/')[-1]}-ar_cot "
        f"(split={task['train_split']}, itr={itr_id})"
    )
    source_run_dir = source_run_name.replace(" ", "_")
    return (
        SOURCE_ROOT
        / task["source_project"]
        / source_run_dir
        / f"checkpoint-{task['source_checkpoint_step']}"
    )


configs = []
for itr_id, task, objective in itertools.product(
    ITR_IDS, TASKS, OBJECTIVES
):
    model = task["model"]
    length_scale = BASE_LENGTH / task["train_len"]
    batch_size = max(
        1,
        min(int(model["2k_bs"] * length_scale), TOTAL_BATCH_SIZE),
    )
    accum_steps = max(TOTAL_BATCH_SIZE // batch_size, 1)
    objective_label = "dp" if objective == "dp" else "ar_cot"
    run_name = (
        f"{model['name'].split('/')[-1]}-exp21_equal_token-{objective_label} "
        f"(split={task['train_split']}, itr={itr_id})"
    )

    configs.append(
        {
            "model_name": model["name"],
            "task_name": task["name"],
            "dataset_split": task["dataset_split"],
            "dataset_num_proc": 16,
            "dataset_cache_dir": DATA_PATH + "/.datasets_cache",
            "batch_size": batch_size,
            "accum_steps": accum_steps,
            "dp_steps": DP_STEPS,
            "num_samples": 16,
            "max_length": task["train_len"],
            "max_test_length": task["test_len"],
            "log_every": 50,
            "save_every": 250,
            "learning_rate": 2e-4,
            "token_calibration_samples": 2_048,
            "tokenization_batch_size": 64,
            "ds_path": DATA_PATH,
            "splits": [task["train_split"]],
            "test_splits": [task["train_split"]],
            "train_split": task["train_split"],
            "objective": objective,
            # The legacy evaluator and accuracy-report parser use "prompt".
            "prompt": objective,
            "source_checkpoint": str(source_checkpoint(task, itr_id)),
            "source_checkpoint_step": task["source_checkpoint_step"],
            "source_experiment": task["source_experiment"],
            "project_name": f"exp21_qwen_{task['name']}",
            "run_name": run_name,
            "output_dir": str(
                OUTPUT_ROOT / task["name"] / run_name.replace(" ", "_")
            ),
            "packing": False,
        }
    )


mode = sys.argv[-1] if len(sys.argv) > 1 else "run"
if mode == "eval":
    from eval import perform_eval

    perform_eval(configs)
elif mode == "run":
    perform_run(configs)
else:
    raise SystemExit(f"unrecognized mode: {mode!r}; expected 'run' or 'eval'")
