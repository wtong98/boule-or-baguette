"""Format generated examples into HuggingFace datasets"""

# <codecell>
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm

disable_caching()


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
            bundle = splits[key][switch]
            if len(bundle) > 0:
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
    maxlens= [32768]
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

    dataset = load_dataset('json', data_dir='/n/netscratch/pehlevan_lab/Lab/wlt/prop/php_enum', split='train', num_proc=16)

    # add separator to distinguish prompt and completion
    dataset = dataset.map(lambda x: {'prompt': x['input'] + '|'}, remove_columns=['input'], num_proc=16)
    dataset = dataset.rename_column('proof', 'completion')
    


    #             yield full




    ds = split_by_len(dataset)

    for maxlen in maxlens:
        print(f'info: filtering by length {maxlen}')

        def filter_len(example):
            toks = tokenizer(example['prompt'] + example['completion'], return_attention_mask=False)
            return len(toks['input_ids']) <= maxlen
        
        filtered = ds.filter(filter_len, num_proc=16)
        ds_small = DatasetDict({name: subset for name, subset in filtered.items() if len(subset) > 0})
        ds_small.save_to_disk(f'/n/home09/wlt/scratch/data/prop_gen/data/hf_php_enum_text_{maxlen}')






