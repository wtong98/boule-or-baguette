"""Fix ds by adding extra pipe"""

# <codecell>
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset, DatasetDict, Dataset, disable_caching
from transformers import AutoTokenizer
from tqdm import tqdm

disable_caching()


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B')

    ds_configs = [
        {'name': 'full', 'path': '/n/netscratch/pehlevan_lab/Lab/wlt/data/prop_gen/data/hf_full_text_pipe'},
    ]

    for conf in ds_configs:
        print(f"info: processing dataset {conf['name']}")
        dataset = DatasetDict.load_from_disk(conf['path'])
        dataset.clear_cache()

        dataset = dataset.map(lambda x: {'completion': x['completion'][:-13]}, num_proc=16)
        dataset.save_to_disk(conf['path'] + '_clean')
        del dataset


