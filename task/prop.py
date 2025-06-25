"""Proposition task"""

# <codecell>
from datasets import DatasetDict, concatenate_datasets
import numpy as np
from transformers import DataCollatorWithPadding

from prop_gen.to_dataset import get_tokenizer


ds_path = 'prop_gen/data/hf'

class PropTask:
    def __init__(self, depth, split='train', cot=False, batch_size=128) -> None:
        self.depth = depth
        self.split = split
        self.cot = cot
        self.batch_size = batch_size

        self.ds = None

        self.true_ds = None
        self.max_true = None
        self.false_ds = None
        self.max_false = None

        self.tokenizer = get_tokenizer()
        self.collate = DataCollatorWithPadding(self.tokenizer, return_tensors='np')
    

    def load_ds(self):
        self.ds = DatasetDict.load_from_disk(ds_path)

        max_len = len(self.ds.keys()) // 2
        if self.split == 'train':
            self.true_ds = concatenate_datasets([self.ds[f'True_{i}'] for i in range(1, self.depth+1)])
            self.false_ds = concatenate_datasets([self.ds[f'False_{i}'] for i in range(1, self.depth+1)])
        elif self.split == 'test':
            self.true_ds = concatenate_datasets([self.ds[f'True_{i}'] for i in range(self.depth+1, max_len + 1)])
            self.false_ds = concatenate_datasets([self.ds[f'False_{i}'] for i in range(self.depth+1, max_len + 1)])
        else:
            raise ValueError(f'unrecognized split: {self.split}')
        
        self.max_true = len(self.true_ds)
        self.max_false = len(self.false_ds)

        if self.cot:
            self.true_ds = self.true_ds.remove_columns(['input_ids']) \
                               .rename_columns({'full_ids': 'input_ids'})
            self.false_ds = self.false_ds.remove_columns(['input_ids']) \
                                .rename_columns({'full_ids': 'input_ids'})
        else:
            self.true_ds = self.true_ds.remove_columns(['full_ids'])
            self.false_ds = self.false_ds.remove_columns(['full_ids'])
    

    def del_ds(self):
        self.ds = None
        self.true_ds = None
        self.max_true = None
        self.false_ds = None
        self.max_false = None

    
    def __next__(self):
        if self.true_ds is None or self.false_ds is None:
            self.load_ds()
        
        true_idxs = np.random.randint(1, self.max_true, size=self.batch_size // 2)
        false_idxs = np.random.randint(1, self.max_false, size=self.batch_size // 2)

        true_batch = self.true_ds[true_idxs]
        false_batch = self.false_ds[false_idxs]

        batch = {k: true_batch[k] + false_batch[k] for k in true_batch.keys()}
        batch = self.collate(batch)
        
        xs = batch['input_ids']
        if self.cot:
            ys = xs[:,1:]
            xs = xs[:,:-1]
        else:
            ys = batch['is_true'].astype(int)

        return xs, ys


    def __iter__(self):
        return self


task = PropTask(depth=3, batch_size=4, cot=True)
xs, ys = next(task)
print(xs[:1])
print(ys[:1])

# TODO: prepare model for training <-- STOPPED HERE

# <codecell>
len(task.tokenizer)
