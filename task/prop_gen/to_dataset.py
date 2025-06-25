"""Format generated examples into HuggingFace datasets"""

# <codecell>
from collections import defaultdict

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = '!'
    return tokenizer


def count_lens(dataset):
    ds_lens = dataset.remove_columns(['input', 'proof', 'ops'])

    def make_entry(): return {True: 0, False: 0}

    counts = defaultdict(make_entry)
    for ex in ds_lens:
        counts[ex['length']][ex['is_true']] += 1

    return counts
    

def count_ops(dataset):
    ds_ops = dataset.remove_columns(['input', 'proof', 'length'])

    def make_entry(): return {True: 0, False: 0}

    counts = defaultdict(make_entry)

    for ex in ds_ops:
        ops = frozenset(ex['ops'])
        counts[ops][ex['is_true']] += 1

    return counts


if __name__ == '__main__':
    dataset = load_dataset('json', data_files='data.json', split='train', keep_in_memory=True)
    
    lens = count_lens(dataset)
    max_len = max(lens.keys())
    
    all_lds = {}
    for i in range(1, max_len + 1):
        for t in True, False:
            lds = dataset.filter(lambda ex: ex['length'] == i and ex['is_true'] == t)
            all_lds[f'{t}_{i}'] = lds

    ds = DatasetDict(all_lds)

    tokenizer = get_tokenizer()

    def to_toks(ex):
        inp_toks = tokenizer(ex['input'], return_attention_mask=False)
        inp_and_proof_toks = tokenizer(ex['input'] + ex['proof'], return_attention_mask=False)
        return {
            'input_ids': inp_toks['input_ids'],
            'full_ids': inp_and_proof_toks['input_ids'],
        }

    ds = ds.map(to_toks, batched=False)
    ds = ds.remove_columns(['input', 'proof'])

    ds.save_to_disk('data/hf')
