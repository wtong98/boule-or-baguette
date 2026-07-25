"""Continue experiment-17 RT adapters with equal-token DP and RT objectives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import datasets
import torch
import wandb
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

LEGACY_EXPERIMENT_DIR = Path(__file__).resolve().parents[1] / "17_llm_clean"
sys.path.insert(0, str(LEGACY_EXPERIMENT_DIR))
from pita_dataset import make_pita_dataset  # noqa: E402

from token_budget import completion_token_counts, choose_rt_schedule  # noqa: E402


datasets.disable_caching()
torch._dynamo.config.suppress_errors = True


def _run_index() -> int:
    try:
        run_idx = int(sys.argv[1])
    except (IndexError, ValueError) as exc:
        print("warn: could not parse run index:", exc)
        print("warn: defaulting to run_idx=0; received:", sys.argv)
        return 0
    print(f"info: using run_idx={run_idx}")
    return run_idx


def _format_dataset(dataset, objective: str, num_proc: int):
    if objective == "dp":
        return dataset.map(
            lambda example: {
                "completion": (
                    "<success />" if example["is_true"] else "<failure />"
                )
            },
            num_proc=num_proc,
        )
    if objective != "ar_cot":
        raise ValueError(f"unrecognized objective: {objective!r}")
    return dataset


def _calibrate_schedule(run_config, tokenizer, train_dataset):
    sample_size = min(run_config["token_calibration_samples"], len(train_dataset))
    if sample_size == 0:
        raise ValueError("the selected PITA training split is empty")
    calibration = train_dataset.select(range(sample_size))
    prompts = calibration["prompt"]
    rt_completions = calibration["completion"]
    dp_completions = [
        "<success />" if is_true else "<failure />"
        for is_true in calibration["is_true"]
    ]
    count_kwargs = {
        "max_length": run_config["max_length"],
        "batch_size": run_config["tokenization_batch_size"],
    }
    dp_counts = completion_token_counts(
        tokenizer, prompts, dp_completions, **count_kwargs
    )
    rt_counts = completion_token_counts(
        tokenizer, prompts, rt_completions, **count_kwargs
    )
    return choose_rt_schedule(
        dp_counts=dp_counts,
        rt_counts=rt_counts,
        dp_steps=run_config["dp_steps"],
        batch_size=run_config["batch_size"],
        dp_accum_steps=run_config["accum_steps"],
    )


def _load_source_adapter(run_config, quantization_config):
    source_checkpoint = Path(run_config["source_checkpoint"]).expanduser()
    if not source_checkpoint.is_dir():
        raise FileNotFoundError(
            f"RT source checkpoint does not exist: {source_checkpoint}"
        )
    if not (source_checkpoint / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"RT source lacks adapter_config.json: {source_checkpoint}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        run_config["model_name"],
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(
        base_model,
        str(source_checkpoint),
        is_trainable=True,
    )
    model.config.use_cache = False
    return model


def perform_run(configs):
    run_config = configs[_run_index() % len(configs)].copy()
    print("info: using config:", run_config)

    tokenizer = AutoTokenizer.from_pretrained(run_config["model_name"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    if pad_token_id is None:
        raise ValueError("the tokenizer has neither a pad token nor an EOS token")

    train_dataset = make_pita_dataset(
        run_config,
        run_config["train_split"],
        "train",
    )
    schedule = _calibrate_schedule(run_config, tokenizer, train_dataset)

    if run_config["objective"] == "dp":
        additional_steps = run_config["dp_steps"]
        accum_steps = run_config["accum_steps"]
    else:
        additional_steps = schedule.rt_steps
        accum_steps = schedule.rt_accum_steps

    if run_config["objective"] == "dp":
        estimated_branch_tokens = (
            additional_steps
            * run_config["batch_size"]
            * accum_steps
            * schedule.dp_mean_tokens
        )
    else:
        estimated_branch_tokens = schedule.estimated_rt_tokens
    token_ratio = estimated_branch_tokens / schedule.target_tokens
    run_config.update(
        {
            "max_steps": additional_steps,
            "active_accum_steps": accum_steps,
            "dp_mean_target_tokens": schedule.dp_mean_tokens,
            "rt_mean_target_tokens": schedule.rt_mean_tokens,
            "target_supervised_tokens": schedule.target_tokens,
            "estimated_supervised_tokens": estimated_branch_tokens,
            "estimated_token_ratio": token_ratio,
            "selected_rt_steps": schedule.rt_steps,
            "selected_rt_accum_steps": schedule.rt_accum_steps,
        }
    )
    print(
        "info: equal-token schedule:",
        {
            key: run_config[key]
            for key in (
                "objective",
                "max_steps",
                "active_accum_steps",
                "dp_mean_target_tokens",
                "rt_mean_target_tokens",
                "target_supervised_tokens",
                "estimated_supervised_tokens",
                "estimated_token_ratio",
                "selected_rt_steps",
                "selected_rt_accum_steps",
            )
        },
    )

    train_examples = (
        run_config["batch_size"] * accum_steps * additional_steps
    )
    if len(train_dataset) < train_examples:
        print(
            "warn: training dataset is smaller than the requested example "
            "count; Trainer will cycle over it"
        )
    else:
        train_dataset = train_dataset.select(range(train_examples))
    train_dataset = _format_dataset(
        train_dataset,
        run_config["objective"],
        run_config["dataset_num_proc"],
    )

    wandb.init(
        project=run_config["project_name"],
        name=run_config["run_name"],
        config=run_config,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    model = _load_source_adapter(run_config, quantization_config)
    data_collator = DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=True,
    )

    save_steps = min(run_config["save_every"], additional_steps)
    args = SFTConfig(
        output_dir=run_config["output_dir"],
        overwrite_output_dir=False,
        num_train_epochs=1,
        max_steps=additional_steps,
        per_device_train_batch_size=run_config["batch_size"],
        per_device_eval_batch_size=run_config["batch_size"],
        gradient_accumulation_steps=accum_steps,
        learning_rate=run_config["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=min(run_config["log_every"], additional_steps),
        save_steps=save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        weight_decay=0.0,
        completion_only_loss=True,
        packing=False,
        max_length=run_config["max_length"],
        eval_strategy="no",
        torch_compile=True,
        dataset_num_proc=run_config["dataset_num_proc"],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    # TRL prepares an already-created QLoRA model for k-bit training, which
    # freezes every parameter. Reactivating the loaded adapter here restores
    # requires_grad only for the experiment-17 LoRA weights, before Trainer
    # creates its optimizer.
    trainer.model.set_adapter(trainer.model.active_adapter)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    if trainable_parameters == 0:
        raise RuntimeError("the loaded RT adapter has no trainable parameters")
    print(f"info: trainable adapter parameters: {trainable_parameters:,}")

    trainer.train()
    trainer.save_model()
    wandb.finish()
