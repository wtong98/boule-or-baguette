"""Format generated examples into HuggingFace datasets"""

# <codecell>
from collections import defaultdict

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')
    tokenizer.pad_token = '!'
    return tokenizer


def split_by_len(dataset):
    def make_entry(): return {True: [], False: []}

    splits = defaultdict(make_entry)
    print('info: splitting by length')
    for ex in tqdm(dataset):
        splits[ex['length']][ex['is_true']].append(ex)

    ds = {}
    print('info: assembling datasets')
    for key in tqdm(splits.keys()):
        for switch in [True, False]:
            ds[f'{switch}_{key}'] = Dataset.from_list(splits[key][switch])

    return DatasetDict(ds)
    

def count_ops(dataset):
    def make_entry(): return {True: 0, False: 0}

    counts = defaultdict(make_entry)

    for ex in dataset:
        ops = frozenset(ex['ops'])
        counts[ops][ex['is_true']] += 1

    return counts


if __name__ == '__main__':
    dataset = load_dataset('json', data_dir='data/raw', split='train', keep_in_memory=True)

    ds = split_by_len(dataset)
    tokenizer = get_tokenizer()

    def to_toks(ex):
        inp_toks = tokenizer(ex['input'], return_attention_mask=False)
        inp_and_proof_toks = tokenizer(ex['input'] + ex['proof'], return_attention_mask=False)
        return {
            'input_ids': inp_toks['input_ids'],
            'full_ids': inp_and_proof_toks['input_ids'],
        }

    ds = ds.map(to_toks, batched=False, num_proc=16)
    ds = DatasetDict({k: dataset.remove_columns(column_names=['proof', 'input']) for k, dataset in ds.items() if len(dataset) > 0})

    ds.save_to_disk('data/hf_trunc')

