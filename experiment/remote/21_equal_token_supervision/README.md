# Experiment 21: equal-token supervision

This experiment branches from each Qwen2.5-Coder-7B RT (`ar_cot`) adapter
produced by `17_llm_clean/qwen_7b_pita` at checkpoint 2000.

Each source adapter gets two continuations with a fresh optimizer:

1. DP for 500 optimizer steps.
2. RT for an integer number of optimizer steps selected to match the DP
   branch's number of supervised completion tokens.

The token counter mirrors TRL's completion-only mask: it includes the EOS
token, accounts for prompt/completion tokenizer-boundary effects, and applies
the configured context-length truncation. It calibrates on 2,048 examples from
the same deterministic shuffled training split used for fine-tuning.

For each task, the target budget is

```text
500 * per_device_batch * DP_gradient_accumulation * mean_DP_target_tokens
```

The RT step count and, only when needed, RT gradient accumulation are chosen
to minimize absolute error from this target. Allowing accumulation to change
is necessary for PHP: a single original 32-example RT update already greatly
exceeds the entire DP target-token budget.

A preliminary calculation against the public Hugging Face dataset produced:

| Task | Mean DP tokens | Mean RT tokens | RT steps | RT accumulation |
| --- | ---: | ---: | ---: | ---: |
| Full | 4.0 | 220.4 | 9 | 1 |
| Imply | 4.0 | 309.4 | 6 | 1 |
| Or | 4.0 | 1004.2 | 2 | 8 |
| PHP | 4.0 | 18575.7 | 1 | 3 |

The actual schedule is recomputed inside every job from the cluster-local
dataset. Candidate schedules are compared using the exact RT counts for the
examples that the selected continuation will consume. The result is logged to
stdout and Weights & Biases. Both branches use a fresh paged AdamW optimizer,
the original QLoRA learning rate, a constant scheduler, and no new warmup. Only
adapter weights are loaded from experiment 17.

There are 40 configurations: five repetitions, four PITA tasks, and two
continuation objectives. From `qwen_7b_pita`, launch training with:

```bash
sbatch sb_qwen_7b_pita_run.sh
```

All 20 experiment-17 RT source checkpoints (five repetitions by four tasks)
must exist first. The current experiment-17 Slurm script has `--array=0-7`,
which covers only repetition 0 after expanding its config to five repetitions;
submit the missing experiment-17 indices or reduce `ITR_IDS` here if those
checkpoints have not already been produced.

After training completes, run the inherited experiment-17 vLLM evaluator:

```bash
sbatch sb_qwen_7b_pita_eval.sh
```

The source checkpoint and output roots are controlled by
`SOURCE_ROOT`, `SOURCE_CHECKPOINT_STEP`, and `OUTPUT_ROOT` in `config.py`.
