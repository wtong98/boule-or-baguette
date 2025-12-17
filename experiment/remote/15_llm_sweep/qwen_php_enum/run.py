"""
Big qwen run
"""

# <codecell>
import os
import sys

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import datasets
import numpy as np
import pandas as pd
from peft import LoraConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding
from transformers.integrations import WandbCallback
from trl import SFTTrainer, SFTConfig
import torch
from tqdm import tqdm
import wandb

sys.path.append('../../../../')
from common import new_seed
from task.prop import PropTask

from config import configs


import torch._dynamo
torch._dynamo.config.suppress_errors = True

datasets.disable_caching()

run_idx = 0
try:
    run_idx = int(sys.argv[1])
    print(f'info: using run_idx={run_idx}')
except Exception as e:
    print('warn: could not parse run_idx from argv:', e)
    print("warn: unrecognized run_idx, defaulting to run_idx=0")
    print('info: received kwargs:', sys.argv)

run_config = configs[run_idx % len(configs)]
print('info: using config:', run_config)

wandb.init(
    project=run_config['project_name'],
    name=run_config['run_name']
)

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
        'true_pos': true_pos,
        'true_neg': true_neg,
        'false_pos': false_pos,
        'false_neg': false_neg
    }


ds_path = run_config['ds_path']
print('info: using ds_path =', ds_path)

def make_ds(depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle()
    return ds


def format_ds(ds):
    if run_config['prompt'] == 'dp':
        ds = ds.map(lambda x: 
                    {
                        'completion': '<success />' if x['is_true'] else '<failure />'
                    }, 
                    num_proc=16)
    return ds


test_splits = run_config['test_splits']
range_hops = [1] + [h + 1 for h in test_splits] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))

n_train_examples = run_config['batch_size'] * run_config['accum_steps'] * run_config['max_steps']
n_val_examples = run_config['num_samples'] * 10

train_split = run_config['train_split']
train_ds = format_ds(make_ds(train_split, 'train').select(range(n_train_examples)))
test_ds = format_ds(make_ds(train_split, 'test').select(range(100)))

val_ds_set = [format_ds(make_ds(r, split='range').select(range(n_val_examples))) for r in ranges]


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type='nf4'
)

model = AutoModelForCausalLM.from_pretrained(
    run_config['model_name'],
    dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='flash_attention_2',
    quantization_config=quant_config
)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
)

args = SFTConfig(
    output_dir=run_config['output_dir'],
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=run_config['batch_size'],
    per_device_eval_batch_size=run_config['batch_size'],
    gradient_accumulation_steps=run_config['accum_steps'],      
    learning_rate=2e-4,                  # QLoRA LR baseline
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=run_config['log_every'],
    save_steps=run_config['save_every'],
    bf16=True,
    gradient_checkpointing=True,
    # optim="adamw_bnb_8bit",
    optim="paged_adamw_8bit",
    max_grad_norm=0.3,
    weight_decay=0.0,
    completion_only_loss=True,
    packing=run_config['packing'],
    max_length=run_config['max_length'],
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


# def evaluate(succ_id, fail_id, num_samples=100):
#     tokenizer = trainer.processing_class
#     model.eval()

#     with torch.no_grad():
#         all_res = {}
#         for r, val_ds in tqdm(zip(ranges, val_ds_set), total=len(val_ds_set)):
#             ds = val_ds.shuffle().select(range(num_samples))
#             tokenizer.padding_side = 'left'
#             coll = DataCollatorWithPadding(tokenizer=tokenizer)
#             inp_ids = [tokenizer(text) for text in ds['prompt']]
#             lab_ids = [tokenizer(text) for text in ds['completion']]

#             filtered = [
#                 (inp, lab)
#                 for inp, lab in zip(inp_ids, lab_ids)
#                 if len(inp['input_ids']) + len(lab['input_ids']) <= trainer.args.max_length
#             ]

#             if not filtered:
#                 print('warn: filtered everything out for range', r, 'with num_samples', num_samples)
#                 all_res[f'range_{r}'] = {}
#                 continue

#             inp_ids, lab_ids = zip(*filtered)
#             inp_ids = list(inp_ids)
#             lab_ids = list(lab_ids)

#             inp = coll(inp_ids)
#             lab = coll(lab_ids)

#             tokenizer.padding_side = 'right'

#             inp['input_ids'] = inp['input_ids'].to(device='cuda')
#             inp['attention_mask'] = inp['attention_mask'].to(device='cuda')
#             lab['input_ids'] = lab['input_ids'].to(device='cuda')

#             with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
#                 preds = trainer.model.generate(**inp, max_length=trainer.args.max_length, max_new_tokens=None)

#             res = score(preds, lab['input_ids'], succ_id, fail_id)
#             all_res[f'range_{r}'] = res
        
#     model.train()
#     return all_res


def evaluate(succ_id, fail_id, num_samples=100, batch_size=16):
    tokenizer = trainer.processing_class
    model.eval()

    with torch.autograd.grad_mode.inference_mode():
        all_res = {}
        for r, val_ds in tqdm(zip(ranges, val_ds_set), total=len(val_ds_set)):
            shuffled_ds = val_ds.shuffle()
            num_eval_samples = min(num_samples, len(shuffled_ds))
            if num_eval_samples == 0:
                all_res[f'range_{r}'] = {}
                continue

            ds = shuffled_ds.select(list(range(num_eval_samples)))
            metric_sums = {}
            total_examples = 0
            effective_batch_size = max(1, batch_size)

            tokenizer.padding_side = 'left'
            coll = DataCollatorWithPadding(tokenizer=tokenizer)

            for start in range(0, num_eval_samples, effective_batch_size):
                end = min(start + effective_batch_size, num_eval_samples)
                batch_ds = ds.select(list(range(start, end)))

                inp_ids = [tokenizer(text) for text in batch_ds['prompt']]
                lab_ids = [tokenizer(text) for text in batch_ds['completion']]

                filtered = [
                    (inp, lab)
                    for inp, lab in zip(inp_ids, lab_ids)
                    if len(inp['input_ids']) + len(lab['input_ids']) <= trainer.args.max_length
                ]

                if not filtered:
                    print('warn: filtered everything out for range', r, 'with num_samples', num_samples)
                    continue

                inp_ids, lab_ids = zip(*filtered)
                inp_ids = list(inp_ids)
                lab_ids = list(lab_ids)

                inp = coll(inp_ids)
                lab = coll(lab_ids)

                inp['input_ids'] = inp['input_ids'].to(device='cuda')
                inp['attention_mask'] = inp['attention_mask'].to(device='cuda')
                lab['input_ids'] = lab['input_ids'].to(device='cuda')

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = trainer.model.generate(**inp, max_length=trainer.args.max_length, max_new_tokens=None)

                res = score(preds, lab['input_ids'], succ_id, fail_id)
                batch_count = preds.shape[0]
                total_examples += batch_count
                for key, value in res.items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_count

            tokenizer.padding_side = 'right'
            all_res[f'range_{r}'] = (
                {key: value / total_examples for key, value in metric_sums.items()}
                if total_examples
                else {}
            )

    model.train()
    return all_res


class WandbEvalCallback(WandbCallback):
    def __init__(self, trainer, val_ds_set, num_samples=100, batch_size=16):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = self.trainer.processing_class
        self.val_ds_set = val_ds_set
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        self.succ_id = self.tokenizer.encode('success')[0]
        self.fail_id = self.tokenizer.encode('failure')[0]

        self.hist = []
        

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        all_res = evaluate(self.succ_id, self.fail_id, num_samples=self.num_samples, batch_size=self.batch_size)
        self._wandb.log(all_res)
        self.hist.append(all_res)

        
eval_callback = WandbEvalCallback(trainer, val_ds_set, num_samples=run_config['num_samples'], batch_size=run_config['batch_size'])
trainer.add_callback(eval_callback)

trainer.train()
wandb.finish()

# final_res = evaluate(eval_callback.succ_id, eval_callback.fail_id, num_samples=run_config['num_samples'])
# df = pd.DataFrame([{
#     'name': run_config['run_name'],
#     'train_hop': train_split,
#     'info': final_res,
#     'hist': eval_callback.hist
# }])

# df.to_pickle(f'res.{new_seed()}.pkl')
