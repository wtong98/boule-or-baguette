"""
Big DP qwen run
"""

# <codecell>
from pathlib import Path
import os

with (Path(os.environ['HOME']) / 'wandb.txt').open() as fp:
    key = fp.read().strip()

os.environ['WANDB_API_KEY'] = key
os.environ['WANDB_PROJECT'] = 'qwen_dp'
os.environ['WANDB_LOG_MODEL'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import datasets
import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from transformers.integrations import WandbCallback
from trl import SFTTrainer, SFTConfig
import torch
from tqdm import tqdm
import wandb

import sys
sys.path.append('../../../../')
from common import new_seed
from task.prop import PropTask, full_text_ds_path

model_name = "Qwen/Qwen2.5-Coder-7B"

wandb.login()


def make_ds(depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=full_text_ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle()

    # temporarily reduce size for debugging
    # ds = ds.select(range(500))

    ds = ds.rename_column('prompt', 'text')
    ds = ds.map(lambda x: {'label': 1 if x['is_true'] else 0}, num_proc=16) 
    ds = ds.remove_columns(['completion'])
    return ds


try:
    split = int(sys.argv[1])
except:
    print("warn: unrecognized train split, defaulting to split=6")

print(f'info: using split={split}')
train_split = split
test_splits = [2, 4, 6, 10]
range_hops = [1] + [h + 1 for h in test_splits] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))

train_ds = make_ds(train_split, 'train')
test_ds = make_ds(train_split, 'test')
test_ds = test_ds.select(range(100))  # TODO: should preferably incorporate logic by subsampling during training

val_ds_set = [make_ds(r, split='range') for r in ranges]


tokenizer = AutoTokenizer.from_pretrained(model_name)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

def to_toks(examples):
    return tokenizer(examples['text'])

train_ds = train_ds.map(to_toks, batched=True, num_proc=16)
test_ds = test_ds.map(to_toks, batched=True, num_proc=16)

accuracy = evaluate.load('accuracy')

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='flash_attention_2',
    quantization_config=quant_config
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
)

model = get_peft_model(model, peft_config)

args = TrainingArguments(
    output_dir="~/scratch/qwen25_coder7b_prop_qlora_dp",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,      
    learning_rate=2e-4,
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
    # completion_only_loss=True,
    # packing=True,
    # max_length=2048,
    eval_strategy='steps',
    torch_compile=False,
    # report_to=None   
)

trainer = Trainer(
    model=model,
    args=args,
    processing_class=tokenizer,
    data_collator=collator,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    # eval_dataset=train_ds.select(range(100)),
    compute_metrics=compute_metrics
)


def evaluate(full=False):
    model.eval()
    with torch.no_grad():
        all_res = {}
        for r, val_ds in tqdm(zip(ranges, val_ds_set), total=len(val_ds_set)):
            ds = val_ds
            if not full:
                ds = ds.shuffle().select(range(100))

            inp_ids = [tokenizer(text) for text in ds['text']]
            labels = torch.tensor(ds['label'], device='cuda')
            inp = collator(inp_ids)

            inp = {k: v.to('cuda') for k, v in inp.items()}
            out = model(**inp)

            preds = out.logits.argmax(-1)
            acc = torch.mean((preds == labels).float()).to('cpu').item()
            all_res[f'range_{r}'] = {'gen_acc': acc}
            
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
        all_res = evaluate()
        self._wandb.log(all_res)

        
eval_callback = WandbEvalCallback(trainer, val_ds_set)
trainer.add_callback(eval_callback)

trainer.train()
wandb.finish()

final_res = evaluate(full=False) # TODO: may require batching for full evaluation
df = pd.DataFrame([{
    'name': 'DP Qwen',
    'train_hop': train_split,
    'info': final_res
}])

df.to_pickle(f'res.{new_seed()}.pkl')