"""Helpers for loading the flat Hugging Face PITA dataset."""

from functools import lru_cache
from pathlib import Path

from datasets import DatasetDict


@lru_cache(maxsize=None)
def _load_split(ds_path, dataset_split):
    dataset = DatasetDict.load_from_disk(str(Path(ds_path).expanduser()))

    if dataset_split not in dataset:
        available = ', '.join(dataset.keys())
        raise KeyError(
            f"PITA split {dataset_split!r} not found in {ds_path!r}; "
            f"available splits: {available}"
        )

    split_ds = dataset[dataset_split]
    required_columns = {'is_true', 'length', 'prompt', 'completion'}
    missing_columns = required_columns.difference(split_ds.column_names)
    if missing_columns:
        missing = ', '.join(sorted(missing_columns))
        raise ValueError(
            f"PITA split {dataset_split!r} is missing required columns: {missing}"
        )

    return split_ds


def make_pita_dataset(run_config, depth, split):
    """Select a train, test, or length-range slice from one PITA split."""
    dataset = _load_split(run_config['ds_path'], run_config['dataset_split'])

    if split == 'train':
        start, stop = 1, depth + 1
    elif split == 'test':
        start, stop = depth + 1, float('inf')
    elif split == 'range':
        start, stop = depth
    else:
        raise ValueError(f'unrecognized split: {split}')

    filter_kwargs = {
        'input_columns': ['length'],
    }
    num_proc = run_config.get('dataset_num_proc')
    if num_proc is not None:
        filter_kwargs['num_proc'] = num_proc

    dataset = dataset.filter(
        lambda length: start <= length < stop,
        **filter_kwargs,
    )
    return dataset.shuffle()
