# TODO: test against logged metrics from wandb
import itertools
from pathlib import Path

import datasets
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer 
from tqdm import tqdm

from vllm import SamplingParams, LLM
from vllm.lora.request import LoRARequest

import sys
sys.path.append('../../../../')
from common import new_seed
from task.prop import PropTask

from config import configs

num_eval_samples = 5


datasets.disable_caching()
torch.set_grad_enabled(False)


def make_ds(ds_path, depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle()
    return ds


def score(preds, labs, succ_id, fail_id):
    is_true = torch.tensor(labs).to('cuda')

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


def evaluate(model, params, lora_req, succ_id, fail_id, num_samples=100):
    all_res = {}
    for r, val_ds in tqdm(zip(ranges, val_ds_set), total=len(val_ds_set)):
        ds = val_ds.shuffle().select(range(num_samples))
        
        prompts = [ex['prompt'] for ex in ds]
        labs = [ex['is_true'] for ex in ds]
        
        outputs = model.generate(prompts, params, lora_request=lora_req)
        preds = torch.nested.nested_tensor([o.outputs[0].token_ids for o in outputs], layout=torch.jagged).to('cuda')
        preds = preds.to_padded_tensor(0)
        
        res = score(preds, labs, succ_id, fail_id)
        all_res[f'range_{r}'] = res
    
    return all_res


run_idx = 0
try:
    run_idx = int(sys.argv[1])
    print(f'info: using run_idx={run_idx}')
except Exception as e:
    print('warn: unable to parse run_idx from args:', e)
    print("warn: unrecognized run_idx, defaulting to run_idx=0")

print('info: received kwargs:', sys.argv)

run_config = configs[run_idx % len(configs)]
print('info: using config:', run_config)

ckpt_dir = Path(run_config['output_dir'])
ckpts = sorted(
    [f for f in ckpt_dir.iterdir() 
     if f.is_dir() and f.name.startswith('checkpoint-')])

test_splits = run_config['splits']
ds_path = run_config['ds_path']

range_hops = [1] + [h + 1 for h in test_splits] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))
val_ds_set = [make_ds(ds_path, r, split='range') for r in ranges]


model_name = run_config['model_name']
max_length = run_config['max_length']

tokenizer = AutoTokenizer.from_pretrained(model_name)
succ_id = tokenizer.encode('success')[0]
fail_id = tokenizer.encode('failure')[0]

model = LLM(model_name, quantization='bitsandbytes', enable_lora=True)
params = SamplingParams(max_tokens=max_length, temperature=0)


all_res = []

for ckpt in itertools.product(ckpts, test_splits):
    ckpt_name = ckpt.name
    print(f'info: processing ckpt={ckpt_name}')
    ckpt_num = int(ckpt_name.split('-')[-1])

    lora_path = str(ckpt.absolute())
    lora_req = LoRARequest('prop_lora', ckpt_num, lora_path)

    res = evaluate(model, params, lora_req, succ_id, fail_id, num_samples=num_eval_samples)
    all_res.append({
        'run_name': run_config['run_name'],
        'project_name': run_config['project_name'],  # TODO: convert to task name
        'model_name': model_name,
        'train_split': run_config['train_split'],
        'ckpt': ckpt.name,
        'ckpt_num': ckpt_num,
        'res': res
    })
    
    print('########---   INFO: split complete   ---#######') 


print('RES', all_res)
df = pd.DataFrame(all_res)
df.to_pickle(f'eval.{new_seed()}.pkl')
    

