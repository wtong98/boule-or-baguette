"""Utilities for matching DP and RT completion-token supervision budgets."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TokenSchedule:
    dp_mean_tokens: float
    rt_mean_tokens: float
    target_tokens: float
    rt_steps: int
    rt_accum_steps: int
    estimated_rt_tokens: float

    @property
    def achieved_ratio(self) -> float:
        return self.estimated_rt_tokens / self.target_tokens


def completion_token_counts(
    tokenizer,
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    max_length: int,
    batch_size: int = 64,
) -> list[int]:
    """Count tokens selected by TRL's completion-only loss mask.

    TRL tokenizes the prompt and the concatenated prompt/completion separately,
    appends EOS to non-conversational completions, and truncates the resulting
    mask to ``max_length``. Counting the difference of the two truncated
    lengths mirrors that behavior, including tokenizer boundary effects.
    """

    if len(prompts) != len(completions):
        raise ValueError("prompts and completions must have the same length")
    if not tokenizer.eos_token:
        raise ValueError("the tokenizer must define eos_token")

    counts: list[int] = []
    for start in range(0, len(prompts), batch_size):
        prompt_batch = list(prompts[start : start + batch_size])
        completion_batch = list(completions[start : start + batch_size])
        completion_batch = [
            completion
            if completion.endswith(tokenizer.eos_token)
            else completion + tokenizer.eos_token
            for completion in completion_batch
        ]
        full_batch = [
            prompt + completion
            for prompt, completion in zip(prompt_batch, completion_batch)
        ]

        prompt_ids = tokenizer(
            prompt_batch,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        full_ids = tokenizer(
            full_batch,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        counts.extend(
            max(0, len(full) - len(prompt))
            for prompt, full in zip(prompt_ids, full_ids)
        )

    return counts


def choose_rt_schedule(
    *,
    dp_counts: Iterable[int],
    rt_counts: Iterable[int],
    dp_steps: int,
    batch_size: int,
    dp_accum_steps: int,
) -> TokenSchedule:
    """Choose integer RT steps/accumulation closest to the DP token budget.

    The original accumulation is retained whenever possible. If even one
    original RT optimizer step would substantially overshoot the target (the
    PHP case), smaller accumulation factors are considered.
    """

    dp_counts = list(dp_counts)
    rt_counts = list(rt_counts)
    if not dp_counts or not rt_counts:
        raise ValueError("token calibration requires non-empty counts")
    if min(dp_steps, batch_size, dp_accum_steps) < 1:
        raise ValueError("steps, batch size, and accumulation must be positive")

    dp_mean = sum(dp_counts) / len(dp_counts)
    rt_mean = sum(rt_counts) / len(rt_counts)
    if dp_mean <= 0 or rt_mean <= 0:
        raise ValueError("mean supervised-token counts must be positive")

    target = dp_steps * batch_size * dp_accum_steps * dp_mean
    rt_prefix_tokens = [0.0]
    for count in rt_counts:
        rt_prefix_tokens.append(rt_prefix_tokens[-1] + count)

    original_step_examples = batch_size * dp_accum_steps
    if original_step_examples < len(rt_prefix_tokens):
        original_step_tokens = rt_prefix_tokens[original_step_examples]
    else:
        original_step_tokens = original_step_examples * rt_mean

    # Keep the experiment-17 effective batch unless its smallest possible
    # continuation (one update) already exceeds the complete DP budget.
    accumulation_options = (
        range(1, dp_accum_steps + 1)
        if original_step_tokens > target
        else [dp_accum_steps]
    )
    candidates: list[tuple[float, int, int, float]] = []

    for accum_steps in accumulation_options:
        tokens_per_step = batch_size * accum_steps * rt_mean
        approximate_steps = target / tokens_per_step
        first_step = max(1, int(approximate_steps) - 2)
        last_step = max(first_step, ceil(approximate_steps) + 2)
        step_options = range(first_step, last_step + 1)
        for rt_steps in step_options:
            num_examples = rt_steps * batch_size * accum_steps
            if num_examples < len(rt_prefix_tokens):
                # These are the exact examples selected by run.py.
                estimated = rt_prefix_tokens[num_examples]
            else:
                estimated = num_examples * rt_mean
            candidates.append(
                (
                    abs(estimated - target),
                    abs(accum_steps - dp_accum_steps),
                    rt_steps,
                    estimated,
                )
            )

    _, accumulation_delta, rt_steps, estimated = min(candidates)
    rt_accum_steps = dp_accum_steps - accumulation_delta
    return TokenSchedule(
        dp_mean_tokens=dp_mean,
        rt_mean_tokens=rt_mean,
        target_tokens=target,
        rt_steps=rt_steps,
        rt_accum_steps=rt_accum_steps,
        estimated_rt_tokens=estimated,
    )
