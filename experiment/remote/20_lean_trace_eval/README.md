# PITA trace generation (experiment 20, phase one)

This Slurm array generates 100 traces for each of the `full`, `imply`, `or`,
and `php` PITA evaluation splits. It uses the `itr=0`, `ar_cot`
Qwen2.5-Coder-7B models from `17_llm_clean/qwen_7b_pita`.

The selected checkpoints are:

| task | checkpoint |
| --- | --- |
| `full` | `qwen_full/...itr=0/checkpoint-2000` |
| `imply` | `qwen_imply/...itr=0/checkpoint-2000` |
| `or` | `qwen_or/...itr=0/checkpoint-2000` |
| `php` | `qwen_php_enum/...itr=0/checkpoint-500` |

From this directory on the cluster, run:

```bash
VLLM_ENV=/path/to/vllm/environment sbatch sb_generate.sh
```

The four array tasks write `traces.<task>.jsonl` under
`/n/netscratch/pehlevan_lab/Lab/wlt/pita_traces`. Each row contains the
prompt, gold completion, generated prediction, original dataset index, and
checkpoint metadata needed by the local Lean evaluation.

After downloading that directory, run phase two from the repository root:

```bash
python experiment/20_lean_trace_eval.py jsonl \
    --input /local/path/to/pita_traces \
    --results-out experiment/20_lean_trace_results.jsonl \
    --summary-out experiment/20_lean_trace_summary.json \
    --report experiment/20_lean_trace_report.md
```
