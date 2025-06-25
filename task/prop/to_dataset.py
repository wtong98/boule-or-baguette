"""Format generated examples into HuggingFace datasets"""

from collections import defaultdict

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = tokenizer.eos_token
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
    tokenizer = get_tokenizer()

    def to_toks(ex):
        inp_toks = tokenizer(ex['input'])
        inp_and_proof_toks = tokenizer(ex['input'] + ex['proof'])
        return {
            'input_ids': inp_toks['input_ids'],
            'input_att_mask': inp_toks['attention_mask'],
            'full_ids': inp_and_proof_toks['input_ids'],
            'full_mask': inp_and_proof_toks['attention_mask']
        }

    ds = dataset.map(to_toks, batched=False)
    ds = ds.remove_columns(['input', 'proof'])

    ds_true = ds.filter(lambda ex: ex['is_true'])
    ds_false = ds.filter(lambda ex: not ex['is_true'])

    ds = DatasetDict(true=ds_true, false=ds_false)
    ds.save_to_disk('data/hf/prop')