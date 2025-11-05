# TODO: test against logged metrics from wandb
import gc
import itertools

import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from tqdm import tqdm

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, LLM
from vllm.lora.request import LoRARequest

import sys
sys.path.append('../../../../')
from task.prop import PropTask, full_text_ds_path, or_text_ds_path, imply_text_ds_path, php_text_ds_path

def score(preds, labs, succ_id, fail_id):
    # is_true = torch.argmax((labels == succ_id).int(), axis=-1) > 0
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

# TODO: make DS true/false agnostic
ds_path = '/n/home09/wlt/scratch/data/prop_gen/data/hf_php_text_32768'
def make_ds(depth, split):
    task = PropTask(depth=depth, split=split, cot='text', ds_path=ds_path)
    task.load_ds()

    ds = datasets.concatenate_datasets([task.true_ds, task.false_ds]).shuffle()
    return ds


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


torch.set_grad_enabled(False)

ckpts = [
    'checkpoint-200',
    'checkpoint-400'
]

test_splits = [80, 120]
range_hops = [1] + [h + 1 for h in test_splits] + [np.inf]
ranges = list(zip(range_hops[:-1], range_hops[1:]))
val_ds_set = [make_ds(r, split='range') for r in ranges]


# TODO: unique checkpoints need to be split to different machines <-- STOPPED HERE
# NOTE: need to load later GCC version in run script
# NOTE: also need to load: module load cuda/12.9

ckpt = ckpts[0]
lora_path = f'/n/home09/wlt/scratch/ckpt/qwen_php_120/{ckpt}'
# tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
# model = AutoModelForCausalLM.from_pretrained(model_path, dtype='auto', device_map='auto')
# model.eval()

tokenizer = AutoTokenizer.from_pretrained(lora_path)
succ_id = tokenizer.encode('success')[0]
fail_id = tokenizer.encode('failure')[0]


model = LLM('Qwen/Qwen2.5-Coder-7B', quantization='bitsandbytes', enable_lora=True)
lora_req = LoRARequest('woot', 1, lora_path)
params = SamplingParams(max_tokens=32768, temperature=0)

all_res = []

for split in itertools.product(test_splits):
    print(f'info: processing split={split}, ckpt={ckpt}')

    res = evaluate(model, params, lora_req, succ_id, fail_id, num_samples=5)
    all_res.append({
        'split': split,
        'ckpt': ckpt,
        'res': res
    })
    
    print('########---   INFO: split complete   ---#######') 


print('RES', all_res)
df = pd.DataFrame(all_res)
df.to_pickle('eval.pkl')
    

