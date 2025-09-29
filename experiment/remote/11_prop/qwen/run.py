"""
Big qwen run
"""

# <codecell>
import datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
import torch

import sys
sys.path.append('../../../../')
from common import *
from task.prop import *

model_name = "Qwen/Qwen2.5-Coder-7B"

def make_ds(depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=full_text_ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle(seed=new_seed())
    return ds

train_ds = make_ds(6, 'train')
test_ds = make_ds(6, 'test')

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='flash_attention_2',
    quantization_config=quant_config
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",  # NOTE: potentially shady parameter
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
)

args = SFTConfig(
    output_dir="~/scratch/qwen25_coder7b_prop_qlora", # TODO: pick destination
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=8,      
    learning_rate=2e-4,                  # QLoRA LR baseline
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_steps=500,
    bf16=True,                           
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",              # QLoRA-friendly optimizer. Consider paging?
    max_grad_norm=0.3,
    weight_decay=0.0,
    report_to="none",
    completion_only_loss=True,
    packing=True,
    max_seq_length=4096,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
)

trainer.train()

# TODO: perform evaluation