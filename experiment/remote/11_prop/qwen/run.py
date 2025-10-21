"""
Big qwen run
"""

# <codecell>
from pathlib import Path
import os
import sys

import datasets
import numpy as np
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig, DataCollatorWithPadding
from transformers.integrations import WandbCallback
from trl import SFTTrainer, SFTConfig
import torch
from tqdm import tqdm
import wandb

sys.path.append('../../../../')
from common import new_seed
from task.prop import PropTask, full_text_ds_path, or_text_ds_path, imply_text_ds_path

model_name = "Qwen/Qwen2.5-Coder-7B"

with (Path(os.environ['HOME']) / 'wandb.txt').open() as fp:
    key = fp.read().strip()

os.environ['WANDB_API_KEY'] = key
os.environ['WANDB_PROJECT'] = 'qwen_imply'
os.environ['WANDB_LOG_MODEL'] = 'false'

split = 6
try:
    split = int(sys.argv[1])
except:
    print("warn: unrecognized train split, defaulting to split=6")

print(f'info: using split={split}')
os.environ['WANDB_NAME'] = f'Qwen-7b-coder AR split={split}'

wandb.login()


def score(preds, labels, succ_id, fail_id):
    is_true = torch.argmax((labels == succ_id).int(), axis=-1) > 0

    t = torch.argmax((preds == succ_id).int(), axis=-1)
    f = torch.argmax((preds == fail_id).int(), axis=-1)

    pred_is_true = (t != 0) * ((f == 0) + (t < f))
    pred_is_false = (f != 0) * ((t == 0) + (f < t))

    true_pos = is_true * pred_is_true
    true_neg = (~is_true) * pred_is_false
    false_pos = (~is_true) * pred_is_true
    false_neg = is_true * pred_is_false

    true_pos = torch.mean(true_pos.float()).to('cpu').item()
    true_neg = torch.mean(true_neg.float()).to('cpu').item()
    false_pos = torch.mean(false_pos.float()).to('cpu').item()
    false_neg = torch.mean(false_neg.float()).to('cpu').item()
    
    return {
        'gen_acc': true_pos + true_neg,
        # 'true_pos': true_pos,
        # 'true_neg': true_neg,
        # 'false_pos': false_pos,
        # 'false_neg': false_neg
    }


ds_path = full_text_ds_path

try:
    split = sys.argv[2]

    if split == 'full':
        ds_path = full_text_ds_path
    elif split == 'or':
        ds_path = or_text_ds_path
    elif split == 'imply':
        ds_path = imply_text_ds_path
    else:
        raise ValueError(f"unrecognized ds_path {split}")

except:
    print("warn: unrecognized ds_path, defaulting to full_text_ds_path")

print('info: using ds_path =', ds_path)

def make_ds(depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle()

    # temporarily reduce size for debugging
    # ds = ds.select(range(1000))

    # TODO: reformat permanently in dataset
    # ds = ds.map(lambda x: {'text': x['prompt'] + x['completion']}, num_proc=16)
    # ds = ds.rename_column('prompt', 'proposition')
    # ds = ds.remove_columns(['completion'])
    return ds


try:
    split = int(sys.argv[1])
except:
    print("warn: unrecognized train split, defaulting to split=6")

print(f'info: using split={split}')
train_split = split
# test_splits = [2, 4, 6, 10]
test_splits = [4, 6, 10]
range_hops = [1] + [h + 1 for h in test_splits] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))

train_ds = make_ds(train_split, 'train')
test_ds = make_ds(train_split, 'test')
test_ds = test_ds.select(range(100))  # TODO: should preferably incorporate logic by subsampling during training

val_ds_set = [make_ds(r, split='range') for r in ranges]


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
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      
    learning_rate=2e-4,                  # QLoRA LR baseline
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=100,
    save_steps=1000,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_bnb_8bit",
    # optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    weight_decay=0.0,
    completion_only_loss=True,
    packing=True,
    max_length=2048,
    eval_strategy='steps',
    torch_compile=False   
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)


def evaluate(succ_id, fail_id, num_samples=100):
    tokenizer = trainer.processing_class
    model.eval()

    with torch.no_grad():
        all_res = {}
        for r, val_ds in tqdm(zip(ranges, val_ds_set), total=len(val_ds_set)):
            ds = val_ds.shuffle().select(range(num_samples))
            tokenizer.padding_side = 'left'
            coll = DataCollatorWithPadding(tokenizer=tokenizer)
            inp_ids = [tokenizer(text) for text in ds['prompt']]
            lab_ids = [tokenizer(text) for text in ds['completion']]
            inp = coll(inp_ids)
            lab = coll(lab_ids)
            tokenizer.padding_side = 'right'

            inp['input_ids'] = inp['input_ids'].to(device='cuda')
            inp['attention_mask'] = inp['attention_mask'].to(device='cuda')
            lab['input_ids'] = lab['input_ids'].to(device='cuda')

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                preds = trainer.model.generate(**inp, max_new_tokens=trainer.args.max_length)

            res = score(preds, lab['input_ids'], succ_id, fail_id)
            all_res[f'range_{r}'] = res
        
    model.train()
    return all_res


class WandbEvalCallback(WandbCallback):
    def __init__(self, trainer, val_ds_set, num_samples=100):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = self.trainer.processing_class
        self.val_ds_set = val_ds_set
        self.num_samples = num_samples
        
        self.succ_id = self.tokenizer.encode('success')[0]
        self.fail_id = self.tokenizer.encode('failure')[0]
        

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        all_res = evaluate(self.succ_id, self.fail_id, self.num_samples)
        self._wandb.log(all_res)

        
eval_callback = WandbEvalCallback(trainer, val_ds_set)
trainer.add_callback(eval_callback)

trainer.train()
wandb.finish()

final_res = evaluate(eval_callback.succ_id, eval_callback.fail_id, num_samples=100)
df = pd.DataFrame([{
    'name': 'AR Qwen (imply)',
    'train_hop': train_split,
    'info': final_res
}])

df.to_pickle(f'res.{new_seed()}.pkl')
