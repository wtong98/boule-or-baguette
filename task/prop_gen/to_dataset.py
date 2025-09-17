"""Format generated examples into HuggingFace datasets"""

# <codecell>
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm


def get_tokenizer():
    tok_path = Path(__file__).parent / 'data' / 'tok'
    tokenizer = AutoTokenizer.from_pretrained(str(tok_path))
    tokenizer.pad_token = tokenizer.eos_token
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
    # maxlen = 100
    # dataset = load_dataset('json', data_dir='test_data', split='train', keep_in_memory=True, num_proc=16)

    maxlens = [1024, 2048]
    dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop', split='train', keep_in_memory=True, num_proc=16)

    if Path('data/tok').exists():
        tokenizer = AutoTokenizer.from_pretrained('data/tok')
    else:
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

        def make_corps(batch_size):
            for i in range(0, len(dataset), batch_size):
                inp = dataset[i:i+batch_size]['input']
                proof = dataset[i:i+batch_size]['proof']
                full = [a + b for a, b in zip(inp, proof)]
                yield full

        corps = make_corps(1000)

        print('info: training tokenizer...')
        tokenizer = tokenizer.train_new_from_iterator(corps, vocab_size=512)

        tokenizer.save_pretrained('data/tok')

    ds = split_by_len(dataset)

    def to_toks(ex):
        inp_toks = tokenizer(ex['input'], return_attention_mask=False)
        inp_and_proof_toks = tokenizer(ex['input'] + ex['proof'], return_attention_mask=False)

        return {
            'input_ids': inp_toks['input_ids'],
            'full_ids': inp_and_proof_toks['input_ids'],
        }

    ds = ds.map(to_toks, batched=False, num_proc=16)

    for maxlen in maxlens:
        print(f'info: filtering by length {maxlen}')

        def filter_len(example):
            return len(example['full_ids']) <= maxlen
        
        ds_small = ds.filter(filter_len, num_proc=16)

        ds_small = DatasetDict({k: dataset.remove_columns(column_names=['proof', 'input']) for k, dataset in ds_small.items() if len(dataset) > 0})
        # ds_small.save_to_disk(f'data/hf_implies_{maxlen}')
        ds.save_to_disk(f'/n/home09/wlt/scratch/data/prop_gen/data/hf_implies_{maxlen}')

